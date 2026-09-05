#!/usr/bin/env python3
"""
Pump.fun early-launch scanner - the "moonshot" half of the bot, fully
independent of both engine.py's Crypto.com trading sleeves AND of
scan_memecoins.py.

WHY THIS EXISTS SEPARATELY FROM scan_memecoins.py:
scan_memecoins.py watches tokens that are ALREADY trending on GeckoTerminal
and auto-buys the safer-looking ones with its own play money. Its collected
outcome data says the median return at every checkpoint offset is at or
below 0% - buying something already trending means buying near a local
momentum peak. This script attacks the opposite end: brand-new pump.fun
launches, minutes old, where the upside is enormous and so is the failure
rate. The two sleeves share a repo and nothing else - separate data files,
separate play-money balance, separate cadence. scan_memecoins.py keeps
running untouched.

WHY PRE-GRADUATION TOKENS NEED A DIFFERENT DATA SOURCE ENTIRELY:
A pump.fun token trades against an internal bonding-curve program account,
not a liquidity pool, until it "graduates" to PumpSwap. GeckoTerminal's API
is pool-indexed - there is literally no pool to look up until graduation, so
none of scan_memecoins.py's helpers can see these tokens at all. Discovery
therefore comes from PumpPortal's free WebSocket (`subscribeNewToken` /
`subscribeMigration`), and pre-graduation state is read straight off the
bonding-curve account via plain Solana JSON-RPC.

GRADUATION MATH (derived from a real account, not from memory - see
GRADUATION_SOL_TARGET): the curve holds a constant product
k = 30 SOL * 1.073e9 tokens. Graduation is when all 793.1e6 "real" tokens
have been sold out of the curve, leaving 279.9e6 virtual tokens, which puts
virtual SOL at k/279.9e6 = 115.01 - i.e. 85.01 SOL must flow in. That makes
`real_sol_reserves / 85` a direct, free progress-to-graduation percentage.

THIS IS PHASE 1: DISCOVERY + OBSERVATION ONLY.
No scoring, no security checks, no Telegram alerts, no positions. It listens,
records what launched, accumulates per-creator launch history, and writes a
raw snapshot. Everything else is deliberately absent so the discovery layer
can be verified against real data before anything is built on top of it.

Empirically measured during validation (2026-09): ~34 new tokens/minute
(~49,000/day) and roughly 1 migration per 34 creations, i.e. only ~3% of
pump.fun launches ever graduate. Any filter built on top of this has to be
brutally selective - see the staged funnel in the plan doc.

Runs as a short-lived job like the other sleeves, but with one structural
difference: a WebSocket has to be held open to hear anything, so the bulk of
each invocation is spent listening (LISTEN_WINDOW_SECONDS) rather than making
a few quick HTTP calls. Back-to-back invocations from the external 5-minute
trigger give near-continuous coverage.
"""
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import websocket  # the one third-party dep in this repo; see workflow's pip step

import engine  # reused only for load_json/save_json (atomic writes)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEV_HISTORY_PATH = os.path.join(DATA_DIR, "pumpfun_dev_history.json")
BONDING_STATE_PATH = os.path.join(DATA_DIR, "pumpfun_bonding_state.json")
RUN_SNAPSHOT_PATH = os.path.join(DATA_DIR, "pumpfun_run_snapshot.jsonl")  # per-run only, not committed

PUMPPORTAL_WS_URL = "wss://pumpportal.fun/api/data?api-key={api_key}"
PUMPPORTAL_API_KEY = os.environ.get("PUMPPORTAL_API_KEY")

# Public endpoint by default. Confirmed working for getBalance/getAccountInfo
# from a local machine during validation, but NOT yet proven from a GitHub
# Actions runner IP (the public endpoint rate-limits datacenter IPs harder
# than it documents) - SOLANA_RPC_URL exists so a provider URL can be dropped
# in via secret without a code change if that turns out to be necessary.
SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

# --- Timing ---------------------------------------------------------------
# The external cron trigger fires every ~5 minutes; listen for most of that
# window and leave headroom for checkout/setup/teardown so one run doesn't
# still be listening when the next is triggered. Starting guess, calibrated
# against the measured ~34 tokens/min (a 240s window sees ~136 launches -
# plenty of data per tick, no need to stretch this longer).
LISTEN_WINDOW_SECONDS = 240
WS_RECV_TIMEOUT_SECONDS = 20   # per-recv timeout, so a quiet stream still notices the window expiring
WS_MAX_RECONNECTS = 1          # one retry, then proceed with whatever was collected

# --- Bonding curve ---------------------------------------------------------
PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
# See module docstring for the derivation. Verified against a real account.
GRADUATION_SOL_TARGET = 85.01
LAMPORTS_PER_SOL = 1e9
TOKEN_DECIMALS = 1e6
RPC_ACCOUNTS_PER_CALL = 100    # getMultipleAccounts hard limit

# --- Dev history -----------------------------------------------------------
# There is no API for "what has this creator launched before" - it gets built
# from our own observations over time, same cold-start caveat as
# scan_memecoins.py's holder-growth tracking (worthless on day one, firms up
# over the system's own runtime). Bounded so one prolific bot-deployer can't
# grow the file without limit.
DEV_HISTORY_MAX_LAUNCHES = 50
DEV_HISTORY_MAX_AGE_SECONDS = 30 * 24 * 3600  # forget launches older than 30 days


def _as_float(v, default=0.0):
    """Duplicated from scan_memecoins.py rather than imported - each sleeve
    stays standalone (only engine.load_json/save_json is ever shared)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --- PumpPortal discovery --------------------------------------------------

def listen_pumpfun_events(api_key, window_seconds=None):
    """Hold the PumpPortal WebSocket open for `window_seconds`, collecting
    every token-creation and migration event seen in that window.

    `window_seconds` resolves to LISTEN_WINDOW_SECONDS at call time rather
    than being bound as a default at import time, so a shorter window can
    actually be injected when testing (a def-time default silently ignores
    reassignment of the module constant).

    Returns (new_tokens, migrations, notices). A dropped connection is
    retried once and then given up on - a WS hiccup must never lose the whole
    tick, same principle as the market-condition lookup in scan_memecoins.py
    being allowed to fail without blocking a scan.

    Only the free subscriptions are used. PumpSwap/trade streams are metered
    and need a funded wallet; the connection deliberately never asks for them
    (validation confirmed the server answers those with a
    "Minimum balance not met" notice rather than silently billing).
    """
    if window_seconds is None:
        window_seconds = LISTEN_WINDOW_SECONDS
    new_tokens, migrations, notices = [], [], []
    deadline = time.monotonic() + window_seconds
    reconnects = 0

    while time.monotonic() < deadline:
        ws = None
        try:
            ws = websocket.create_connection(
                PUMPPORTAL_WS_URL.format(api_key=api_key),
                timeout=WS_RECV_TIMEOUT_SECONDS,
                sslopt={"cert_reqs": ssl.CERT_REQUIRED},
            )
            ws.send(json.dumps({"method": "subscribeNewToken"}))
            ws.send(json.dumps({"method": "subscribeMigration"}))

            while time.monotonic() < deadline:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue  # quiet stretch, just re-check the deadline
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                tx_type = msg.get("txType")
                if tx_type == "create":
                    msg["observed_at"] = datetime.now(timezone.utc).isoformat()
                    new_tokens.append(msg)
                elif tx_type == "migrate":
                    msg["observed_at"] = datetime.now(timezone.utc).isoformat()
                    migrations.append(msg)
                else:
                    # subscription confirmations and the metered-stream notice
                    notices.append(msg)
        except (websocket.WebSocketException, OSError) as e:
            if reconnects >= WS_MAX_RECONNECTS or time.monotonic() >= deadline:
                print(f"WARN: PumpPortal WS failed ({e}); proceeding with "
                      f"{len(new_tokens)} creation(s) collected so far.", file=sys.stderr)
                break
            reconnects += 1
            print(f"WARN: PumpPortal WS dropped ({e}); reconnecting "
                  f"({reconnects}/{WS_MAX_RECONNECTS})...", file=sys.stderr)
            time.sleep(2)
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

    return new_tokens, migrations, notices


# --- Bonding curve state via Solana RPC ------------------------------------

def _rpc_post(payload, timeout=20):
    req = urllib.request.Request(
        SOLANA_RPC_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "cryptobot-pumpfun-scan/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def parse_bonding_curve(raw_bytes):
    """Decode a pump.fun BondingCurve account.

    Layout confirmed by parsing a real account during validation and
    cross-checking the decoded creator against the same token's
    `traderPublicKey` from the WebSocket: 8-byte Anchor discriminator, then
    five little-endian u64s, then a 1-byte `complete` flag, then a 32-byte
    creator pubkey.
    """
    import struct
    if raw_bytes is None or len(raw_bytes) < 49:
        return None
    v_tok, v_sol, r_tok, r_sol, supply = struct.unpack_from("<QQQQQ", raw_bytes, 8)
    complete = bool(raw_bytes[48])
    real_sol = r_sol / LAMPORTS_PER_SOL
    return {
        "virtual_token_reserves": v_tok / TOKEN_DECIMALS,
        "virtual_sol_reserves": v_sol / LAMPORTS_PER_SOL,
        "real_token_reserves": r_tok / TOKEN_DECIMALS,
        "real_sol_reserves": real_sol,
        "token_total_supply": supply / TOKEN_DECIMALS,
        "complete": complete,
        # The headline number: how far along the 85-SOL path to graduation.
        "graduation_pct": round(real_sol / GRADUATION_SOL_TARGET * 100.0, 2),
    }


def fetch_bonding_curves(bonding_curve_keys):
    """Batch-read bonding-curve accounts, RPC_ACCOUNTS_PER_CALL at a time.
    Returns {bonding_curve_key: parsed_state}. Missing/failed lookups are
    simply absent from the result rather than raising - one bad batch must
    not take down the tick."""
    import base64
    out = {}
    keys = [k for k in bonding_curve_keys if k]
    for i in range(0, len(keys), RPC_ACCOUNTS_PER_CALL):
        batch = keys[i:i + RPC_ACCOUNTS_PER_CALL]
        try:
            resp = _rpc_post({
                "jsonrpc": "2.0", "id": 1, "method": "getMultipleAccounts",
                "params": [batch, {"encoding": "base64"}],
            })
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"WARN: getMultipleAccounts failed for batch of {len(batch)}: {e}", file=sys.stderr)
            continue
        values = (resp.get("result") or {}).get("value") or []
        for key, value in zip(batch, values):
            if not value:
                continue  # account gone/closed
            try:
                raw = base64.b64decode(value["data"][0])
            except (KeyError, IndexError, ValueError):
                continue
            parsed = parse_bonding_curve(raw)
            if parsed is not None:
                out[key] = parsed
    return out


# --- Dev/creator launch history -------------------------------------------

def record_launches(dev_history, new_tokens, timestamp):
    """Accumulate per-creator launch history from what we observe.

    Deliberately records EVERY observed launch, not just interesting ones -
    the serial-deployer pattern (one wallet spraying dozens of tokens a day)
    is only visible if the boring ones are counted too.
    """
    for token in new_tokens:
        creator = token.get("traderPublicKey")
        mint = token.get("mint")
        if not creator or not mint:
            continue
        entry = dev_history.setdefault(creator, {"launches": []})
        if any(l.get("mint") == mint for l in entry["launches"]):
            continue  # already recorded (e.g. a duplicate event)
        entry["launches"].append({
            "mint": mint,
            "ts": timestamp,
            "name": token.get("name"),
            "symbol": token.get("symbol"),
            "initial_buy_sol": _as_float(token.get("solAmount")),
            "market_cap_sol_at_launch": _as_float(token.get("marketCapSol")),
            "bonding_curve_key": token.get("bondingCurveKey"),
            "graduated": False,   # flipped by record_migrations when/if it happens
        })
        entry["launches"] = entry["launches"][-DEV_HISTORY_MAX_LAUNCHES:]
        entry["last_seen"] = timestamp


def record_migrations(dev_history, migrations):
    """Mark previously-observed launches as graduated.

    Only tokens whose creation we already saw can be matched - a migration of
    something launched before this scanner existed is simply unmatched, which
    is expected and not an error.
    """
    migrated_mints = {m.get("mint") for m in migrations if m.get("mint")}
    if not migrated_mints:
        return 0
    matched = 0
    for entry in dev_history.values():
        for launch in entry["launches"]:
            if launch.get("mint") in migrated_mints and not launch.get("graduated"):
                launch["graduated"] = True
                matched += 1
    return matched


def prune_dev_history(dev_history, now_wall):
    """Drop creators whose every recorded launch is older than the retention
    window, so the file tracks recent behaviour rather than growing forever
    (~49k launches/day would otherwise make this unbounded very fast)."""
    cutoff = now_wall.timestamp() - DEV_HISTORY_MAX_AGE_SECONDS
    stale = []
    for creator, entry in dev_history.items():
        newest = 0.0
        for launch in entry.get("launches", []):
            try:
                newest = max(newest, datetime.fromisoformat(launch["ts"]).timestamp())
            except (KeyError, ValueError):
                continue
        if newest < cutoff:
            stale.append(creator)
    for creator in stale:
        del dev_history[creator]
    return len(stale)


def dev_stats(dev_history, creator):
    """Summary of what we've observed from this creator - the cheap, free
    input Stage 0 of the filter uses. Returns None for an unseen creator
    (which is NOT the same as a bad one: on day one every creator is unseen)."""
    entry = dev_history.get(creator)
    if not entry:
        return None
    launches = entry.get("launches", [])
    if not launches:
        return None
    graduated = sum(1 for l in launches if l.get("graduated"))
    return {
        "launches_seen": len(launches),
        "graduated": graduated,
        "graduation_rate_pct": round(graduated / len(launches) * 100.0, 1),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not PUMPPORTAL_API_KEY:
        print("ERROR: PUMPPORTAL_API_KEY not set - cannot connect to the discovery stream.", file=sys.stderr)
        return 1

    dev_history = engine.load_json(DEV_HISTORY_PATH, {})
    bonding_state = engine.load_json(BONDING_STATE_PATH, {})

    timestamp = datetime.now(timezone.utc).isoformat()
    new_tokens, migrations, notices = listen_pumpfun_events(PUMPPORTAL_API_KEY)

    record_launches(dev_history, new_tokens, timestamp)
    matched_migrations = record_migrations(dev_history, migrations)

    # Phase 1 reads the curve for just this window's launches, to prove the
    # RPC path end-to-end. Deciding WHICH tokens are worth re-polling on
    # later ticks is Stage 1 of the filter - that's Phase 2, not here.
    #
    # Expect partial results here, and don't treat that as an error: measured
    # during validation, only 4 of 7 curves resolved when queried seconds
    # after creation, but all 7 resolved on a re-query ~2 minutes later. The
    # accounts simply aren't indexed by the RPC node yet that early. Phase 2
    # must therefore NOT judge a token on its first sighting - re-poll on a
    # later tick instead, which it does anyway to measure velocity.
    curve_keys = [t.get("bondingCurveKey") for t in new_tokens]
    curves = fetch_bonding_curves(curve_keys)
    for token in new_tokens:
        key = token.get("bondingCurveKey")
        parsed = curves.get(key)
        if parsed is None:
            continue
        entry = bonding_state.setdefault(key, {
            "mint": token.get("mint"),
            "name": token.get("name"),
            "symbol": token.get("symbol"),
            "creator": token.get("traderPublicKey"),
            "first_seen": timestamp,
            "history": [],
        })
        entry["history"].append({"ts": timestamp, **parsed})
        entry["last_seen"] = timestamp

    pruned = prune_dev_history(dev_history, datetime.now(timezone.utc))

    # Overwritten fresh each run (not appended) - the workflow uploads this as
    # a per-run artifact instead of committing an ever-growing file to git.
    with open(RUN_SNAPSHOT_PATH, "w") as f:
        for token in new_tokens:
            f.write(json.dumps({"row_type": "new_token", **token}) + "\n")
        for mig in migrations:
            f.write(json.dumps({"row_type": "migration", **mig}) + "\n")

    engine.save_json(DEV_HISTORY_PATH, dev_history)
    engine.save_json(BONDING_STATE_PATH, bonding_state)

    for notice in notices:
        if "errors" in notice:
            print(f"NOTE from PumpPortal: {notice['errors']}", file=sys.stderr)

    graduated_pct = [c["graduation_pct"] for c in curves.values()]
    print(f"Listened {LISTEN_WINDOW_SECONDS}s: {len(new_tokens)} new token(s), "
          f"{len(migrations)} migration(s) ({matched_migrations} matched to a launch we'd seen). "
          f"Read {len(curves)}/{len(curve_keys)} bonding curve(s); "
          f"max graduation progress this batch: {max(graduated_pct) if graduated_pct else 0:.2f}%. "
          f"Dev history: {len(dev_history)} creator(s) tracked, {pruned} pruned. "
          f"Bonding state: {len(bonding_state)} curve(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
