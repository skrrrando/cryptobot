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

CURRENT STATE - PHASES 1-2: DISCOVERY THROUGH SCORED CANDIDATES.
Listens, tracks launches, re-polls every tracked bonding curve each tick to
measure how fast SOL is flowing in, applies the staged filter funnel below,
runs a RugCheck security gate on the survivors, and writes a ranked candidate
list. Still NO Telegram alerts and NO positions - alerting and the manual
BUY/IGNORE loop are phase 3, so the filter can be watched against real data
before anything acts on it.

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
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import websocket  # the one third-party dep in this repo; see workflow's pip step

import engine  # reused only for load_json/save_json (atomic writes)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEV_HISTORY_PATH = os.path.join(DATA_DIR, "pumpfun_dev_history.json")
BONDING_STATE_PATH = os.path.join(DATA_DIR, "pumpfun_bonding_state.json")
SECURITY_CACHE_PATH = os.path.join(DATA_DIR, "pumpfun_security_cache.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "pumpfun_candidates.json")
ALERTS_PATH = os.path.join(DATA_DIR, "pumpfun_alerts.json")
OUTCOMES_PATH = os.path.join(DATA_DIR, "pumpfun_pending_outcomes.json")
PORTFOLIO_PATH = os.path.join(DATA_DIR, "pumpfun_portfolio.json")
TELEGRAM_OFFSET_PATH = os.path.join(DATA_DIR, "pumpfun_telegram_offset.json")
DECISION_LABELS_PATH = os.path.join(DATA_DIR, "pumpfun_decision_labels.jsonl")
RUN_SNAPSHOT_PATH = os.path.join(DATA_DIR, "pumpfun_run_snapshot.jsonl")  # per-run only, not committed
# Small pre-computed summary for the dashboard. The dashboard deliberately does
# NOT read bonding_state.json directly: at MAX_TRACKED_CURVES x
# BONDING_HISTORY_WINDOW that file is megabytes, and valuing a position needs
# the curve invariant anyway - which is this script's job, not the browser's.
DASHBOARD_PATH = os.path.join(DATA_DIR, "pumpfun_dashboard.json")
# Net worth over time for the homepage's weekly change - same shape and same
# reasoning as scan_memecoins.py's copy (duplicated, not shared, per this
# codebase's sleeve-independence convention).
NETWORTH_HISTORY_PATH = os.path.join(DATA_DIR, "pumpfun_networth_history.json")
NETWORTH_SNAPSHOT_INTERVAL_SECONDS = 3600
NETWORTH_HISTORY_MAX_POINTS = 2000

PUMPPORTAL_WS_URL = "wss://pumpportal.fun/api/data?api-key={api_key}"
# .strip() guards against a trailing newline/whitespace sneaking into the
# GitHub secret on copy-paste - confirmed live as the actual cause of a
# production failure: PumpPortal's nginx returned a flat "400 Bad Request"
# on the WS handshake (not an auth error), which is exactly what happens
# when a stray \n breaks the HTTP request line before it ever reaches the
# app that would check the key itself. Passed through urllib.parse.quote
# when building the URL too, in case the key ever contains characters that
# need escaping - cheap insurance, not a response to an observed failure.
PUMPPORTAL_API_KEY = (os.environ.get("PUMPPORTAL_API_KEY") or "").strip() or None

# Public endpoint by default. Confirmed working for getBalance/getAccountInfo
# from a local machine during validation, but NOT yet proven from a GitHub
# Actions runner IP (the public endpoint rate-limits datacenter IPs harder
# than it documents) - SOLANA_RPC_URL exists so a provider URL can be dropped
# in via secret without a code change if that turns out to be necessary.
# `or` rather than .get(key, default): the workflow references
# ${{ secrets.SOLANA_RPC_URL }} unconditionally, which GitHub sets to an
# EMPTY STRING (not an absent key) when that secret doesn't exist - confirmed
# live as the actual cause of a production crash (urllib rejecting "" as an
# unknown URL type). .get()'s default only fires on a missing key, not a
# present-but-empty one, so it silently let "" through.
SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL") or "https://api.mainnet-beta.solana.com"

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

# --- Staged filter funnel --------------------------------------------------
# Measured reality: ~49,000 launches/day, of which only ~3% ever graduate,
# and the target is at most 4 alerts/day. That is a ~12,000:1 funnel, so each
# stage only spends money/calls on what survived the cheaper stage before it.
# Every threshold below is an explicit starting guess, NOT tuned against
# outcome data yet - same caveat as every constant in scan_memecoins.py.
#
# Stage 0 (free, in-memory): is this worth tracking at all?
STAGE0_MAX_DEV_LAUNCHES_24H = 5   # serial deployers spraying tokens all day
#
# Stage 1 (cheap, 100 curves per RPC call): has it actually shown traction?
# This is where most of the precision comes from - it conditions on a token
# having already climbed, rather than trying to predict a winner at birth.
# Deliberately NOT applied on first sighting: validation showed curves queried
# seconds after creation often aren't RPC-indexed yet.
# Non-binding in normal production operation: with a ~240s listen window on a
# ~5-minute trigger, a token is already ~7 min old by its second curve read,
# which is the earliest a velocity measurement can exist anyway. It only bites
# in compressed local tests. Its real job is to stop a token being judged off
# a single sighting.
STAGE1_MIN_AGE_SECONDS = 180
STAGE1_MIN_GRADUATION_PCT = 2.0    # 2% of the 85-SOL path = ~1.7 SOL in
STAGE1_MIN_SOL_PER_MIN = 0.10      # still actively taking money in, not stalled
VELOCITY_SAMPLE_WINDOW = 4         # measure momentum over the last N samples only
#
# Stage 2 (expensive, one HTTP call per token): is it safe?
MAX_SECURITY_CHECKS_PER_TICK = 8   # same defensive cap as scan_memecoins.py
SECURITY_CACHE_TTL_SECONDS = 900   # 15 min - shorter than the memecoin sleeve's
                                    # 30 min because these tokens change fast
PUMPFUN_MAX_DEV_HOLDING_PCT = 5.0  # creator holding more than this can dump on us

# Largest single non-curve holder's share of everything actually bought so far.
# dev_holding_pct above only sees the CREATOR's own wallet; an insider buying
# through a different address is invisible to it. This catches that.
#
# Motivated by a real failure: "Biden Wif Hat" was alerted at 70.8% of the way
# to graduation and was at 0.05% five minutes later. On a bonding curve you
# cannot pull liquidity the way you can from an AMM - the curve holds the SOL -
# so that can only happen if someone sold nearly the entire bought supply back
# into it. A single wallet holding a large share of the circulating supply is
# exactly the precondition for that.
#
# CAVEAT, stated plainly: 20% is chosen from only 8 observed signals, where it
# happens to sit in a clean gap (13.7% vs 23.1%) that separates both winners
# (9.1%, 6.2%) from the two worst losses (23.1%, 33.8%). That is far too few
# points to call validated - it is a hypothesis with a plausible mechanism
# behind it, not a fitted rule. Re-check against real outcomes as data builds.
PUMPFUN_MAX_CIRCULATING_TOP_PCT = 20.0

RUGCHECK_REPORT_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"

# --- Outcome tracking ------------------------------------------------------
# Denser and shorter than scan_memecoins.py's 10m/30m/1h/3h/6h schedule,
# because these move far faster - live observation caught tokens doing 30-40%
# of the graduation path inside two minutes.
OUTCOME_CHECKPOINT_OFFSETS_MINUTES = [5, 15, 30, 60, 180]
# Stop watching a token that has neither graduated nor died by this point.
# Longer than TRACKED_MAX_AGE_SECONDS on purpose: alerted tokens are rare
# (<= DAILY_SIGNAL_CAP a day) so watching them longer costs almost nothing,
# and the whole question this half of the bot exists to answer - did the
# signal graduate? - can't be answered by giving up after two hours.
OUTCOME_MAX_AGE_SECONDS = 6 * 3600

# --- Stage 3: alerting -----------------------------------------------------
# The whole point of the funnel: few enough signals a day that each one can
# actually be researched by hand before deciding. Counted per UTC day.
DAILY_SIGNAL_CAP = 4
# Raised from the original untuned guess (25) after backtesting all 147
# resolved signals (11 real alerts + 136 shadow-tracked, at $10/signal, hold
# to graduation or final outcome, no stop-loss):
#
#   score >= 30: n=129  29% win rate  -11.9% return
#   score >= 75: n= 58  47% win rate   +1.4% return
#   score >= 90: n= 28  64% win rate  +14.2% return
#   score >= 95: n= 17  71% win rate   +9.1% return
#
# The win rate climbs smoothly with score rather than jumping at one lucky
# cutoff, which is what makes this look like a real gradient and not a
# curve-fit. Confirmed on the 135 shadow-only signals alone (never touched
# by the score formula's own tuning, so a genuine out-of-sample check):
# score >= 90 there gives +20.0% (n=24), score >= 95 gives +18.1% (n=13) -
# same shape, if anything stronger. A stop-loss on top of this filter makes
# it WORSE (+14.2% -> +7-9%), because it cuts off some of the eventual
# graduations along with the losers - once the entry filter is this
# selective, holding to the real outcome beats protecting the downside.
#
# Set at 90 rather than 95 for a larger, steadier sample at the same
# ballpark return. Caveat that has to travel with this number: it is drawn
# from ~3 days of real market data. Re-check once more signals accumulate -
# this could be a real edge or a temporary regime, and only more data (which
# keeps flowing regardless via shadow-tracking, unaffected by this bar) can
# tell those apart.
MIN_SCORE_TO_ALERT = 90.0
ALERT_COOLDOWN_SECONDS = 3600  # never re-alert the same mint within this window

# --- Moonshot paper portfolio ---------------------------------------------
# Completely separate from scan_memecoins.py's own $1000 - different file,
# different balance, different rules. This one NEVER auto-buys: a position is
# only ever opened by the user pressing BUY on a Telegram alert, and only ever
# closed by the user pressing SELL. That is the entire point of this half of
# the bot - it is a practice ground for making the call, not an autonomous
# trader.
MOONSHOT_STARTING_BALANCE_USD = 1000.0
# auto_buy_moonshot_signals() spends this on every signal that clears
# MIN_SCORE_TO_ALERT, fixed rather than balance-scaled - the user's own
# choice, made deliberately small next to MOONSHOT_STARTING_BALANCE_USD so
# a bad run of signals can't do much damage while this is still new.
MOONSHOT_AUTO_BUY_USD = 20.0
# The manual BUY-amount flow (poll_telegram_decisions) is no longer reachable
# for new signals now that alerting means buying, but the code is left in
# place - see auto_buy_moonshot_signals's docstring - so these still apply to
# a stale pre-automation alert someone might still answer, and to nothing
# else.
MOONSHOT_MIN_TRADE_USD = 5.0
MOONSHOT_SUGGESTED_TRADE_USD = 100.0  # shown in the prompt as a sensible default
MOONSHOT_QUICK_BUY_PCTS = [10, 15, 20]  # % of free balance, shown as one-tap buttons

# Pump.fun's own trading fee. Widely documented as 1% per trade; recorded as a
# named constant rather than folded into the maths so it's easy to correct.
PUMPFUN_TRADE_FEE_PCT = 1.0
# Solana transaction cost, both legs. Trivial next to slippage here, but it's
# real and the memecoin sleeve already learned not to pretend fills are free.
SOLANA_GAS_USD = 0.02

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
GECKOTERMINAL_POOL_URL = "https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}"
SOL_USDC_POOL_ADDRESS = "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"

# --- Tracked-set bounds ----------------------------------------------------
# Can't track 49k tokens/day forever. Drop anything that graduated, went dead,
# or simply got old without going anywhere.
TRACKED_MAX_AGE_SECONDS = 7200        # 2h without graduating = it isn't going to
BONDING_HISTORY_WINDOW = 20           # rolling samples per curve, like hot_state
MAX_TRACKED_CURVES = 1500             # hard ceiling on RPC work per tick

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
                PUMPPORTAL_WS_URL.format(api_key=urllib.parse.quote(api_key, safe="")),
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

def http_get_json(url, timeout=15):
    """Duplicated from scan_memecoins.py - see _as_float on why each sleeve
    keeps its own copy."""
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "cryptobot-pumpfun-scan/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def check_security_rugcheck(mint, bonding_curve_key, total_supply_raw):
    """RugCheck report for a pre-graduation pump.fun mint.

    IMPORTANT - why this is NOT a copy of scan_memecoins.py's
    check_security_solana(): that function fails a token when its top-10
    holders own more than SOLANA_MAX_TOP10_PCT (40%). Pre-graduation, the
    bonding curve account itself holds ~all the supply, so RugCheck reports
    top10_pct = 100.0 for EVERY pump.fun token - verified on two real mints.
    Reusing that gate would have silently rejected 100% of candidates forever.

    What IS meaningful pre-graduation, and is used instead:
      - mint/freeze authority still active (should be revoked)
      - RugCheck's own `rugged` flag and danger-level risks
      - `creatorBalance` as a share of supply - the actual dump risk, since
        the dev's own bag is the one thing that isn't locked in the curve
      - `totalHolders` - a real holder count, tracked over time for growth
    """
    try:
        data = http_get_json(RUGCHECK_REPORT_URL.format(mint=mint), timeout=15)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"WARN: RugCheck lookup failed for {mint}: {e}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        return None

    mint_authority_active = data.get("mintAuthority") is not None
    freeze_authority_active = data.get("freezeAuthority") is not None
    risks = data.get("risks") or []
    has_danger_risk = any(r.get("level") == "danger" for r in risks)
    rugged = bool(data.get("rugged"))

    dev_holding_pct = None
    if total_supply_raw:
        dev_holding_pct = round(_as_float(data.get("creatorBalance")) / total_supply_raw * 100.0, 3)

    # Concentration among real holders only - the curve is excluded by
    # matching the bonding-curve address we already know from the WS event.
    circulating_top_pct = None
    holders = data.get("topHolders") or []
    non_curve = [h for h in holders if h.get("owner") != bonding_curve_key]
    non_curve_total = sum(_as_float(h.get("pct")) for h in non_curve)
    if non_curve_total > 0:
        circulating_top_pct = round(max(_as_float(h.get("pct")) for h in non_curve) / non_curve_total * 100.0, 1)

    passed = (
        not mint_authority_active
        and not freeze_authority_active
        and not has_danger_risk
        and not rugged
        and (dev_holding_pct is None or dev_holding_pct <= PUMPFUN_MAX_DEV_HOLDING_PCT)
        # Unknown concentration is NOT treated as a failure: it's usually just
        # a very new token with too few holders to compute a share from, and
        # failing those outright would gut the whole point of catching launches
        # early. Only a known-and-too-high value rejects.
        and (circulating_top_pct is None or circulating_top_pct <= PUMPFUN_MAX_CIRCULATING_TOP_PCT)
    )
    return {
        "passed": passed,
        "mint_authority_active": mint_authority_active,
        "freeze_authority_active": freeze_authority_active,
        "has_danger_risk": has_danger_risk,
        "rugged": rugged,
        "dev_holding_pct": dev_holding_pct,
        "total_holders": data.get("totalHolders"),
        "total_market_liquidity_usd": _as_float(data.get("totalMarketLiquidity")),
        # Recorded but deliberately NOT gated on: RugCheck's own score scale
        # isn't documented well enough here to pick a threshold honestly.
        # Kept so analysis can tell us later whether it predicts anything.
        "rugcheck_score": data.get("score_normalised"),
        "circulating_top_pct": circulating_top_pct,
        "checked_at": time.time(),
    }


def check_security_cached(mint, bonding_curve_key, total_supply_raw, security_cache, checks_this_tick):
    cached = security_cache.get(mint)
    now = time.time()
    if cached and (now - cached.get("checked_at", 0) < SECURITY_CACHE_TTL_SECONDS):
        return cached
    if checks_this_tick[0] >= MAX_SECURITY_CHECKS_PER_TICK:
        return cached  # defer to a later tick; may be None
    checks_this_tick[0] += 1
    result = check_security_rugcheck(mint, bonding_curve_key, total_supply_raw)
    if result is None:
        return cached
    security_cache[mint] = result
    return result


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


# --- Telegram --------------------------------------------------------------

def telegram_call(method, payload, timeout=15):
    """One place for every Telegram Bot API call. Returns the parsed `result`
    or None - a Telegram failure must never take down a scan tick."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"WARN: TELEGRAM_* not set, skipping {method}. Payload: "
              f"{json.dumps(payload)[:400]}", file=sys.stderr)
        return None
    req = urllib.request.Request(
        TELEGRAM_API.format(token=TELEGRAM_TOKEN, method=method),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        if not body.get("ok"):
            print(f"ERROR: Telegram {method} returned not-ok: {str(body)[:300]}", file=sys.stderr)
            return None
        return body.get("result")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        print(f"ERROR: Telegram {method} failed: {e}", file=sys.stderr)
        return None


def fetch_sol_usd():
    """SOL price, for denominating the moonshot portfolio in dollars. Same
    free GeckoTerminal endpoint the other sleeve uses. None on failure -
    callers fall back rather than guessing a price."""
    url = GECKOTERMINAL_POOL_URL.format(network="solana", pool_address=SOL_USDC_POOL_ADDRESS)
    try:
        data = http_get_json(url, timeout=15)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"WARN: SOL price lookup failed: {e}", file=sys.stderr)
        return None
    attrs = (data.get("data") or {}).get("attributes") or {}
    price = _as_float(attrs.get("base_token_price_usd"))
    return price if price > 0 else None


def token_price_usd(curve, sol_usd):
    """Price of one token in USD, from the bonding curve's own reserves.

    price_in_sol = virtual_sol_reserves / virtual_token_reserves. Verified
    against RugCheck's independently-reported USD price for a real mint: the
    implied SOL price came out at ~103, matching the live SOL/USDC quote.
    """
    if not curve or not sol_usd:
        return None
    v_tok = curve.get("virtual_token_reserves") or 0
    v_sol = curve.get("virtual_sol_reserves") or 0
    if v_tok <= 0:
        return None
    return (v_sol / v_tok) * sol_usd


def simulate_curve_buy(curve, sol_in):
    """Tokens actually received for `sol_in` SOL, using the curve's own
    constant-product invariant - not an approximation.

    This matters far more here than on a normal DEX: a $100 order into a curve
    holding ~10 SOL is ~10% of the entire pool, so the price moves hard against
    the buyer within the single trade. Quoting the position at the pre-trade
    spot price would flatter every paper result, which defeats the point of
    this half of the bot being a realistic practice ground.

    Returns (tokens_out, effective_price_per_token, slippage_pct).
    """
    v_sol = _as_float((curve or {}).get("virtual_sol_reserves"))
    v_tok = _as_float((curve or {}).get("virtual_token_reserves"))
    if v_sol <= 0 or v_tok <= 0 or sol_in <= 0:
        return None, None, None
    sol_after_fee = sol_in * (1 - PUMPFUN_TRADE_FEE_PCT / 100.0)
    k = v_sol * v_tok
    tokens_out = v_tok - (k / (v_sol + sol_after_fee))
    if tokens_out <= 0:
        return None, None, None
    spot = v_sol / v_tok
    effective = sol_in / tokens_out
    slippage_pct = (effective / spot - 1) * 100.0
    return tokens_out, effective, round(slippage_pct, 2)


def simulate_curve_sell(curve, tokens_in):
    """SOL received for selling `tokens_in` back into the curve. Same exact
    invariant, same reason - exiting a thin curve costs just as much as
    entering one."""
    v_sol = _as_float((curve or {}).get("virtual_sol_reserves"))
    v_tok = _as_float((curve or {}).get("virtual_token_reserves"))
    if v_sol <= 0 or v_tok <= 0 or tokens_in <= 0:
        return None
    k = v_sol * v_tok
    sol_out = v_sol - (k / (v_tok + tokens_in))
    if sol_out <= 0:
        return None
    return sol_out * (1 - PUMPFUN_TRADE_FEE_PCT / 100.0)


def format_moonshot_alert(candidate, price_usd):
    """The one message type this project still sends to Telegram. Kept dense
    and scannable - the user reads these to decide, so every line has to earn
    its place."""
    sec = candidate.get("security") or {}
    dev = candidate.get("dev_stats")
    grad = candidate.get("graduation_pct") or 0
    vel = candidate.get("sol_per_min")
    holders = sec.get("total_holders")
    dev_pct = sec.get("dev_holding_pct")
    liq = sec.get("total_market_liquidity_usd")

    age_min = None
    try:
        age_min = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(candidate["first_seen"])).total_seconds() / 60.0
    except (KeyError, ValueError):
        pass

    lines = [
        f"MOONSHOT SIGNAL  |  score {candidate.get('score')}",
        f"{candidate.get('name')} (${candidate.get('symbol')})",
        "",
        f"Graduation:  {grad:.1f}% of the way ({candidate.get('real_sol_reserves', 0):.1f} / 85 SOL)",
        f"Speed:       {vel if vel is not None else '?'} SOL/min",
        f"Age:         {age_min:.0f} min" if age_min is not None else "Age:         ?",
        f"Holders:     {holders if holders is not None else '?'}",
        f"Dev holds:   {dev_pct if dev_pct is not None else '?'}%",
        f"Liquidity:   ${liq:,.0f}" if liq else "Liquidity:   ?",
    ]
    if dev:
        lines.append(f"Dev history: {dev['launches_seen']} launch(es) seen, "
                     f"{dev['graduated']} graduated ({dev['graduation_rate_pct']}%)")
    else:
        lines.append("Dev history: first time we've seen this creator")
    if price_usd:
        lines.append(f"Price:       ${price_usd:.10f}")
    lines += [
        "",
        f"mint: {candidate.get('mint')}",
        "",
        "BUY will ask how much to put in. Play money only.",
    ]
    return "\n".join(lines)


def alert_keyboard(alert_id, mint):
    """Inline keyboard. callback_data is capped at 64 bytes by Telegram, so it
    carries only a short alert id - the full context lives in the alerts state
    file, keyed by that id. OPEN is a plain url button (no callback plumbing,
    nothing to log) so the user can eyeball the chart before deciding."""
    return {"inline_keyboard": [[
        {"text": "BUY", "callback_data": f"buy:{alert_id}"},
        {"text": "IGNORE", "callback_data": f"ignore:{alert_id}"},
        {"text": "OPEN", "url": f"https://pump.fun/coin/{mint}"},
    ]]}


def amount_keyboard(alert_id):
    """Shown after BUY, so a decision is one tap instead of typing a number.
    Percentages are of free balance, computed fresh at press time (not baked
    into the button), so the amount is always right even if balance moved
    since the alert fired. callback_data is 'buypct:{alert_id}:{pct}' -
    alert_id is a short int-string, so this stays well under Telegram's
    64-byte cap even for three buttons."""
    return {"inline_keyboard": [[
        {"text": f"{pct}%", "callback_data": f"buypct:{alert_id}:{pct}"}
        for pct in MOONSHOT_QUICK_BUY_PCTS
    ]]}


def record_networth_snapshot(history, balance, positions_value, now_wall):
    """Append a net-worth point, at most once per NETWORTH_SNAPSHOT_INTERVAL_SECONDS.
    Duplicated from scan_memecoins.py per the sleeve-independence convention."""
    now_ts = now_wall.timestamp()
    if history:
        try:
            last_ts = datetime.fromisoformat(history[-1]["ts"]).timestamp()
            if now_ts - last_ts < NETWORTH_SNAPSHOT_INTERVAL_SECONDS:
                return history
        except (KeyError, ValueError):
            pass
    history.append({
        "ts": now_wall.isoformat(),
        "net_worth": round(balance + positions_value, 2),
        "balance": round(balance, 2),
        "positions_value": round(positions_value, 2),
    })
    return history[-NETWORTH_HISTORY_MAX_POINTS:]


def _parse_amount(text):
    """Read a dollar amount out of a chat reply. Tolerates '$50', '50 usd',
    '12,50' and stray spaces; returns None if there's no sane positive number
    in there, so a random chat message can't be mistaken for a trade size."""
    cleaned = (text or "").strip().lower()
    for junk in ("$", "usd", "dollar", "dollars", "€", "eur"):
        cleaned = cleaned.replace(junk, "")
    cleaned = cleaned.replace(",", ".").strip()
    try:
        amount = float(cleaned)
    except ValueError:
        return None
    if amount <= 0 or amount != amount or amount == float("inf"):
        return None
    return round(amount, 2)


def default_portfolio():
    return {"balance": MOONSHOT_STARTING_BALANCE_USD, "positions": {}, "closed": []}


def open_position(portfolio, candidate, curve, sol_usd, timestamp, amount_usd):
    """Only ever called from a user's BUY press, with the dollar amount they
    typed in reply to the alert. Fills against the curve's real invariant
    (fee + slippage), not the quoted spot price."""
    mint = candidate["mint"]
    if mint in portfolio["positions"]:
        return None, "already holding this one"
    if not sol_usd or sol_usd <= 0:
        return None, "no SOL price available"
    if amount_usd < MOONSHOT_MIN_TRADE_USD:
        return None, f"${amount_usd:.2f} is below the ${MOONSHOT_MIN_TRADE_USD:.0f} minimum"
    if amount_usd > portfolio["balance"]:
        return None, f"only ${portfolio['balance']:.2f} free"

    sol_in = amount_usd / sol_usd
    qty, effective_price_sol, slippage_pct = simulate_curve_buy(curve, sol_in)
    if qty is None:
        return None, "curve data unusable for pricing"

    portfolio["balance"] -= amount_usd
    spot = _as_float(curve.get("virtual_sol_reserves")) / _as_float(curve.get("virtual_token_reserves"))
    portfolio["positions"][mint] = {
        "mint": mint,
        "name": candidate.get("name"),
        "symbol": candidate.get("symbol"),
        "bonding_curve_key": candidate.get("bonding_curve_key"),
        "entry_ts": timestamp,
        "entry_price_usd": effective_price_sol * sol_usd,   # what we ACTUALLY paid
        "entry_spot_price_usd": spot * sol_usd,             # what it was quoted at
        "entry_slippage_pct": slippage_pct,
        "amount_usd": amount_usd,
        "qty": qty,
        "entry_graduation_pct": candidate.get("graduation_pct"),
        "entry_score": candidate.get("score"),
    }
    return portfolio["positions"][mint], None


def usable_curve_state(history):
    """The most recent curve reading in `history` that has real, sellable
    reserves - walking backward past any that don't.

    Exists because of a real production failure: a position that graduated
    was sold using the exact RPC read that reported complete=True, and that
    read had virtual_sol_reserves/virtual_token_reserves of zero - pump.fun's
    account gets its reserves cleared as part of the same state transition
    that flips the completion flag, sometimes before the funds are gone,
    sometimes not. simulate_curve_sell correctly refuses to price against
    zero reserves and returns None, but close_position then silently treated
    that None as "proceeds $0" - turning the sleeve's best possible outcome
    (an actual graduation) into a 100% loss. Confirmed on a real trade:
    "I'm Feeling Lucky", entered at 74.95% graduation, completed 9 minutes
    later, sold for $0.00 instead of the real payout.

    Used for both the automatic graduation exit and the manual SELL path -
    they share the same risk, since both ultimately price off the newest
    history entry."""
    for entry in reversed(history or []):
        if _as_float(entry.get("virtual_sol_reserves")) > 0 and _as_float(entry.get("virtual_token_reserves")) > 0:
            return entry
    return None


def close_position(portfolio, mint, curve, sol_usd, timestamp, reason="manual"):
    """Called from a user's SELL press, or automatically from
    check_moonshot_positions (reason="graduated"/"take_profit"/"timeout").
    Exits against the same invariant - selling back into a thin curve costs
    as much as entering it."""
    pos = portfolio["positions"].pop(mint, None)
    if pos is None:
        return None
    sol_out = simulate_curve_sell(curve, pos["qty"])
    proceeds = max(0.0, (sol_out or 0.0) * (sol_usd or 0.0) - SOLANA_GAS_USD)
    pnl = proceeds - pos["amount_usd"]
    portfolio["balance"] += proceeds
    closed = {
        **pos,
        "exit_ts": timestamp,
        "exit_price_usd": (proceeds / pos["qty"]) if pos["qty"] else None,
        "proceeds_usd": round(proceeds, 2),
        "pnl_usd": round(pnl, 2),
        "pnl_pct": round(pnl / pos["amount_usd"] * 100.0, 2) if pos["amount_usd"] else 0.0,
        "exit_reason": reason,
    }
    portfolio["closed"].append(closed)
    return closed


# Take-profit level for automated moonshot exits. Backtested against every
# resolved score>=90 signal (70 real + cross-checked on 62 shadow-only, both
# at $20/trade): every stop-loss level tried (15-50%) made the RESULT WORSE
# than no stop-loss at all, because it cuts off recoveries on a filter that's
# already this selective (60%+ win rate baseline). So there is deliberately
# no stop-loss here - downside is naturally capped anyway, a position can't
# lose more than its own stake. Take-profit alone, swept from 20% to 120%,
# forms a smooth plateau from roughly 30% to 85% (not a knife-edge one-point
# optimum), peaking around 55-80% (+4.1-4.4% vs +2.1% for "hold to
# graduation/timeout with no take-profit at all"). Picked 60% - the middle of
# that plateau, not its exact edge. Same caveat as every other backtested
# number in this file: real, but based on days not months of data - revisit
# once more signals accumulate.
MOONSHOT_TAKE_PROFIT_PCT = 60.0


def check_moonshot_positions(portfolio, bonding_state, sol_usd, now_wall):
    """Automated exit check for every open moonshot position, run once per
    tick. Three terminal conditions, highest-priority first:
      1. graduated - the curve's migration to a real pool is the actual
         payout this sleeve is built around; cash out immediately against the
         curve's final state rather than holding into the (unmodeled)
         post-graduation DEX pool.
      2. take-profit - MOONSHOT_TAKE_PROFIT_PCT above cost basis.
      3. timeout - OUTCOME_MAX_AGE_SECONDS since entry, same giving-up point
         process_outcome_checkpoints already uses to close out label
         tracking, so a position that never graduates and never hits
         take-profit doesn't sit open forever waiting for a SELL that isn't
         coming.
    A position with no fresh curve read yet this tick is left alone rather
    than guessed at - it'll get evaluated on a later tick instead. Same if
    the curve has no reading anywhere in history with real reserves left to
    price a sale against (see usable_curve_state) - deferred to the next
    tick rather than forced through at a bogus $0, on the chance a better
    read shows up (real accounts have come back with valid reserves on a
    later poll after one bad one).
    """
    closed = []
    for mint, pos in list(portfolio["positions"].items()):
        history = (bonding_state.get(pos.get("bonding_curve_key")) or {}).get("history") or []
        latest = history[-1] if history else None
        if latest is None:
            continue
        sellable = usable_curve_state(history)
        if sellable is None:
            continue

        if latest.get("complete"):
            c = close_position(portfolio, mint, sellable, sol_usd, now_wall.isoformat(), reason="graduated")
            if c is not None:
                closed.append(c)
            continue

        sol_out = simulate_curve_sell(sellable, pos["qty"])
        value_usd = max(0.0, (sol_out or 0.0) * (sol_usd or 0.0) - SOLANA_GAS_USD)
        if pos["amount_usd"] and value_usd >= pos["amount_usd"] * (1 + MOONSHOT_TAKE_PROFIT_PCT / 100.0):
            c = close_position(portfolio, mint, sellable, sol_usd, now_wall.isoformat(), reason="take_profit")
            if c is not None:
                closed.append(c)
            continue

        try:
            elapsed_seconds = (now_wall - datetime.fromisoformat(pos["entry_ts"])).total_seconds()
        except (KeyError, ValueError):
            elapsed_seconds = 0
        if elapsed_seconds >= OUTCOME_MAX_AGE_SECONDS:
            c = close_position(portfolio, mint, sellable, sol_usd, now_wall.isoformat(), reason="timeout")
            if c is not None:
                closed.append(c)

    return closed


# --- Staged filter ---------------------------------------------------------

def passes_stage0(token, dev_history, now_wall):
    """Free, in-memory gate on the WebSocket payload plus our own accumulated
    dev history. Decides only whether a token is worth TRACKING - it is
    deliberately permissive, because at this point the token is seconds old
    and there is genuinely almost nothing to judge it on. The real filtering
    is stage 1 (demonstrated traction). Rejecting hard here would just be
    guessing with extra steps."""
    if not token.get("mint") or not token.get("bondingCurveKey"):
        return False, "missing mint/curve"
    if not token.get("name") and not token.get("symbol"):
        return False, "no name or symbol"

    creator = token.get("traderPublicKey")
    entry = dev_history.get(creator)
    if entry:
        cutoff = now_wall.timestamp() - 24 * 3600
        recent = 0
        for launch in entry.get("launches", []):
            try:
                if datetime.fromisoformat(launch["ts"]).timestamp() >= cutoff:
                    recent += 1
            except (KeyError, ValueError):
                continue
        if recent > STAGE0_MAX_DEV_LAUNCHES_24H:
            return False, f"serial deployer ({recent} launches/24h)"
    return True, None


def compute_velocity(history, window=VELOCITY_SAMPLE_WINDOW):
    """SOL flowing into the curve per minute *recently*.

    Deliberately measured over only the last few samples, not the whole
    tracked history: a token that pumped hard in its first minutes and then
    went flat would otherwise keep reporting a healthy lifetime average for
    the next hour and a half (BONDING_HISTORY_WINDOW samples at ~5 min each),
    which is the opposite of what stage 1 is trying to detect. Observed live -
    tokens routinely do 30-40% of the graduation path in their first two
    minutes and then stall.

    Needs at least two samples spread over real time; a single sighting says
    nothing about direction.
    """
    if len(history) < 2:
        return None
    recent = history[-window:]
    try:
        first, last = recent[0], recent[-1]
        t0 = datetime.fromisoformat(first["ts"]).timestamp()
        t1 = datetime.fromisoformat(last["ts"]).timestamp()
    except (KeyError, ValueError):
        return None
    minutes = (t1 - t0) / 60.0
    if minutes <= 0:
        return None
    return round((last["real_sol_reserves"] - first["real_sol_reserves"]) / minutes, 4)


def passes_stage1(entry, now_wall):
    """Traction gate. The token must be old enough to have had a fair chance,
    far enough along the curve to matter, and still actively taking money in."""
    history = entry.get("history") or []
    if not history:
        return False, "no curve reads yet"
    try:
        age = now_wall.timestamp() - datetime.fromisoformat(entry["first_seen"]).timestamp()
    except (KeyError, ValueError):
        return False, "bad first_seen"
    if age < STAGE1_MIN_AGE_SECONDS:
        return False, "too young to judge"

    latest = history[-1]
    if latest.get("complete"):
        return False, "already graduated"
    if latest.get("graduation_pct", 0) < STAGE1_MIN_GRADUATION_PCT:
        return False, f"only {latest.get('graduation_pct', 0):.2f}% to graduation"

    velocity = compute_velocity(history)
    if velocity is None:
        return False, "not enough samples for velocity"
    if velocity < STAGE1_MIN_SOL_PER_MIN:
        return False, f"stalled ({velocity:.3f} SOL/min)"
    return True, None


def score_candidate(entry, security, dev=None):
    """Combine the surviving signals into one number.

    Still hand-weighted rather than fitted - there is nowhere near enough
    outcome data to fit anything, and pretending otherwise would repeat the
    momentum sleeve's original mistake. Its job is to rank the day's survivors
    against each other so the daily cap spends itself on the strongest few.

    Reweighted after reviewing the first 8 real signals, where the original
    weights turned out to reward the things that did NOT separate winners from
    losers, and to ignore the one thing that did:

      - progress, velocity and holder count all failed to discriminate. The
        single worst loss (-89%) had the HIGHEST velocity in the sample
        (6.78 SOL/min), a higher graduation % than one of the two winners,
        and more holders than either of them.
      - four different tokens all scored a maxed-out 100, of which only two
        graduated - so the top of the scale carried no information at all.
      - concentration was the only input that lined up with outcomes: both
        winners sat at 6.2% and 9.1%, while the two worst losses were the two
        highest in the sample at 23.1% and 33.8%.

    So concentration now carries real weight, and the caps on progress and
    velocity are lowered so that a single fast-moving token can no longer max
    the score on momentum alone. Deliberately NOT tuned to perfectly separate
    those 8 points: a rule that cleanly splits 8 observations is far more
    likely to be a curve-fit than an edge.

    dev's creator track record was added later, from dev_history - a much
    bigger sample (thousands of creators, not 8 signals) split three ways by
    launch position and prior outcome:
      - creator's first-ever observed launch: 7.89% graduate (n=5592)
      - repeat launch, creator has NEVER graduated before:  1.41% (n=7966)
      - repeat launch, creator HAS graduated before:         5.19% (n=1195)
    The gap between "never graduated despite repeat attempts" and either other
    group is large and highly significant (z=18.8 / z=8.9) - a wallet that has
    tried more than once and never once succeeded is a real, distinct risk
    signal, not noise. Deliberately NOT rewarding "first-ever" over "unknown"
    even though it measured highest (7.89%) - that number aggregates every
    creator's first attempt, including future serial failures whose first one
    just hasn't happened yet, and the gap to the proven-repeat-winner group is
    real but far weaker (z=3.2) than the failure signal. Only the two robust,
    unambiguous ends are scored: proven repeat winner gets rewarded, proven
    repeat failure gets penalized, everything else (including "no history at
    all", which is most candidates - see the plan's own cold-start caveat) is
    neutral.
    """
    history = entry.get("history") or []
    latest = history[-1] if history else {}
    velocity = compute_velocity(history) or 0.0

    progress = latest.get("graduation_pct", 0.0)          # 0-100
    progress_pts = min(progress * 0.5, 30.0)               # 60%+ maxes this out
    velocity_pts = min(velocity * 8.0, 20.0)               # 2.5 SOL/min maxes this out
    holders = _as_float((security or {}).get("total_holders"))
    holder_pts = min(holders / 40.0, 10.0)                 # 400+ holders maxes this out

    # Concentration: the strongest discriminator in the data so far, and the
    # one with a real mechanism behind it (a wallet holding a big share of the
    # bought supply is what makes a dump-to-zero possible at all). Full marks
    # under 10%, tapering to nothing at the PUMPFUN_MAX_CIRCULATING_TOP_PCT
    # gate, above which the token is rejected outright anyway.
    conc = (security or {}).get("circulating_top_pct")
    if conc is None:
        conc_pts = 12.0   # unknown: neither rewarded nor punished, mid-range
    elif conc <= 10.0:
        conc_pts = 30.0
    else:
        span = max(PUMPFUN_MAX_CIRCULATING_TOP_PCT - 10.0, 1.0)
        conc_pts = max(0.0, 30.0 * (1 - (conc - 10.0) / span))

    dev_pct = (security or {}).get("dev_holding_pct")
    dev_holding_pts = 10.0 if (dev_pct is not None and dev_pct <= 1.0) else 0.0

    dev_track_pts = 0.0
    if dev and dev.get("launches_seen", 0) >= 2:
        dev_track_pts = 15.0 if dev.get("graduated", 0) >= 1 else -20.0

    return round(progress_pts + velocity_pts + holder_pts + conc_pts
                 + dev_holding_pts + dev_track_pts, 1)


def prune_bonding_state(bonding_state, now_wall, protected_keys=frozenset()):
    """Drop graduated, dead, and stale curves so the tracked set (and the RPC
    work it implies) stays bounded.

    `protected_keys` are curves still under outcome watch after being alerted
    on. They are exempt from BOTH rules, and that exemption is load-bearing:
    without it a token would be dropped the instant it graduated, which is
    precisely the success outcome this sleeve exists to measure, and any
    checkpoint past TRACKED_MAX_AGE_SECONDS would find no curve to read.
    Alerted tokens are capped at DAILY_SIGNAL_CAP a day, so keeping them
    costs almost nothing.
    """
    cutoff = now_wall.timestamp() - TRACKED_MAX_AGE_SECONDS
    drop = []
    for key, entry in bonding_state.items():
        if key in protected_keys:
            continue
        history = entry.get("history") or []
        if history and history[-1].get("complete"):
            drop.append(key)
            continue
        try:
            first_seen = datetime.fromisoformat(entry["first_seen"]).timestamp()
        except (KeyError, ValueError):
            drop.append(key)
            continue
        if first_seen < cutoff:
            drop.append(key)
    for key in drop:
        del bonding_state[key]
    return len(drop)


def register_outcome_watch(pending_outcomes, alert, timestamp, alerted=True):
    """Start watching a token's outcome.

    Deliberately called when the ALERT is sent, not when the user decides -
    ignored tokens have to be followed exactly as closely as bought ones.
    Otherwise the only data ever collected would be about coins the user
    already liked, which cannot answer whether the ignores were right.

    `alerted=False` is used for shadow-tracking (see register_shadow_watches):
    a token that cleared the security gate but never made the daily cap or
    the score bar still gets its outcome recorded, just without ever
    reaching Telegram. The flag rides along on every label row this watch
    produces so dashboard stats and the alert history can stay scoped to
    real alerts, while the underlying label file still has the full picture
    for later analysis.
    """
    mint = alert["mint"]
    if mint in pending_outcomes:
        return
    pending_outcomes[mint] = {
        "mint": mint,
        "alert_id": alert.get("alert_id"),
        "name": alert.get("name"),
        "symbol": alert.get("symbol"),
        "bonding_curve_key": alert.get("bonding_curve_key"),
        "signal_ts": alert.get("signal_ts") or timestamp,
        "alert_score": alert.get("score"),
        "alert_graduation_pct": alert.get("graduation_pct"),
        "decision": None,          # filled in when the user presses a button
        "graduated": False,
        "checkpoints_done": {},
        "alerted": alerted,
    }


def register_shadow_watches(candidates, pending_outcomes, timestamp):
    """Track every stage-2 survivor's outcome, not just the ones that made
    today's DAILY_SIGNAL_CAP cut.

    Motivated by how slowly outcome data was accumulating: only ~4/day ever
    got tracked at all, because tracking was gated on being alerted. RugCheck
    already ran on every one of these (stage 2's own per-tick cap is what
    actually limits cost, unrelated to DAILY_SIGNAL_CAP) - so recording what
    happened to the ones that DIDN'T make the cut is free, and is exactly the
    data needed to tell whether the cap and the score floor are set in the
    right place, rather than guessing from 4 signals a day.

    Returns label rows with row_type "shadow", never "alert" - so
    build_dashboard_summary's alert-scoped stats (what the user actually saw
    and can be judged on) stay unaffected by candidates nobody was ever
    shown. register_outcome_watch's own "already tracked" guard means a
    candidate that IS alerted this same tick (send_moonshot_alerts must run
    first) is simply skipped here rather than downgraded.
    """
    label_rows = []
    for candidate in candidates:
        mint = candidate["mint"]
        if mint in pending_outcomes:
            continue
        register_outcome_watch(pending_outcomes, {
            "mint": mint,
            "name": candidate.get("name"),
            "symbol": candidate.get("symbol"),
            "bonding_curve_key": candidate.get("bonding_curve_key"),
            "signal_ts": timestamp,
            "score": candidate.get("score"),
            "graduation_pct": candidate.get("graduation_pct"),
        }, timestamp, alerted=False)
        label_rows.append({
            "row_type": "shadow",
            "mint": mint,
            "name": candidate.get("name"),
            "symbol": candidate.get("symbol"),
            "score": candidate.get("score"),
            "graduation_pct": candidate.get("graduation_pct"),
            "signal_ts": timestamp,
        })
    return label_rows


def record_decision_on_outcome(pending_outcomes, mint, decision, timestamp):
    watch = pending_outcomes.get(mint)
    if watch is not None and watch.get("decision") is None:
        watch["decision"] = decision
        watch["decided_ts"] = timestamp


def process_outcome_checkpoints(pending_outcomes, bonding_state, portfolio,
                                sol_usd, now_wall, timestamp):
    """Record what happened to every alerted token at fixed offsets, and
    detect the terminal outcome (graduated, or gave up).

    Graduation is the verdict this whole sleeve is judged on, so it is checked
    on every tick rather than only at checkpoint boundaries - a token can
    graduate between two offsets and must not be missed."""
    label_rows = []
    finished = []

    for mint, watch in pending_outcomes.items():
        history = (bonding_state.get(watch.get("bonding_curve_key")) or {}).get("history") or []
        latest = history[-1] if history else None
        try:
            signal_ts = datetime.fromisoformat(watch["signal_ts"])
        except (KeyError, ValueError):
            finished.append(mint)
            continue
        elapsed_min = (now_wall - signal_ts).total_seconds() / 60.0

        # --- terminal: graduated ------------------------------------------
        if latest and latest.get("complete") and not watch["graduated"]:
            watch["graduated"] = True
            label_rows.append({
                "row_type": "outcome",
                "outcome": "graduated",
                "minutes_to_graduate": round(elapsed_min, 1),
                "decision": watch.get("decision"),
                "mint": mint,
                "name": watch.get("name"),
                "alert_score": watch.get("alert_score"),
                "alert_graduation_pct": watch.get("alert_graduation_pct"),
                "signal_ts": watch["signal_ts"],
                "recorded_ts": timestamp,
                "alerted": watch.get("alerted", True),
            })
            finished.append(mint)
            continue

        # --- scheduled checkpoints ----------------------------------------
        if latest:
            for offset in OUTCOME_CHECKPOINT_OFFSETS_MINUTES:
                key = str(offset)
                if key in watch["checkpoints_done"] or elapsed_min < offset:
                    continue
                value_usd = None
                pos = portfolio.get("positions", {}).get(mint)
                if pos and sol_usd:
                    sol_out = simulate_curve_sell(latest, pos["qty"])
                    if sol_out:
                        value_usd = round(sol_out * sol_usd, 2)
                label_rows.append({
                    "row_type": "checkpoint",
                    "mint": mint,
                    "name": watch.get("name"),
                    "offset_minutes": offset,
                    "decision": watch.get("decision"),
                    "graduation_pct": latest.get("graduation_pct"),
                    "real_sol_reserves": latest.get("real_sol_reserves"),
                    "progress_since_alert_pct": round(
                        (latest.get("graduation_pct") or 0)
                        - (watch.get("alert_graduation_pct") or 0), 2),
                    # only present when the user actually bought it
                    "position_value_usd": value_usd,
                    "signal_ts": watch["signal_ts"],
                    "checkpoint_ts": timestamp,
                    "alerted": watch.get("alerted", True),
                })
                watch["checkpoints_done"][key] = timestamp

        # --- terminal: gave up --------------------------------------------
        if elapsed_min >= OUTCOME_MAX_AGE_SECONDS / 60.0:
            label_rows.append({
                "row_type": "outcome",
                "outcome": "did_not_graduate",
                "final_graduation_pct": (latest or {}).get("graduation_pct"),
                "decision": watch.get("decision"),
                "mint": mint,
                "name": watch.get("name"),
                "alert_score": watch.get("alert_score"),
                "alert_graduation_pct": watch.get("alert_graduation_pct"),
                "signal_ts": watch["signal_ts"],
                "recorded_ts": timestamp,
                "alerted": watch.get("alerted", True),
            })
            finished.append(mint)

    for mint in finished:
        del pending_outcomes[mint]
    return label_rows


def auto_buy_moonshot_signals(candidates, alerts_state, portfolio, pending_outcomes,
                              sol_usd, timestamp, now_wall):
    """Stage 3: buy at most DAILY_SIGNAL_CAP signals per UTC day, strongest
    first, fixed MOONSHOT_AUTO_BUY_USD each. Returns the label rows to append.

    Used to send a Telegram alert and wait for a BUY/IGNORE decision instead -
    switched to auto-buying because that manual step turned out to be the
    real bottleneck in production: a review of the first 3 days found 12
    alerts sent (including two real graduations) and zero completed
    purchases, because the user doesn't reliably watch Telegram. The
    BUY/IGNORE/quick-percent machinery in poll_telegram_decisions is left in
    place - unreachable for new signals now that alerting and buying happen
    at the same score bar, but /positions + SELL still uses the same
    close_position() path for a manual early exit, and it's a straightforward
    revert if manual mode is ever wanted back.
    """
    today = now_wall.strftime("%Y-%m-%d")
    daily = alerts_state.setdefault("daily", {"date": today, "count": 0})
    if daily.get("date") != today:
        daily["date"], daily["count"] = today, 0

    recent = alerts_state.setdefault("recent_mints", {})
    label_rows = []

    for candidate in candidates:
        if daily["count"] >= DAILY_SIGNAL_CAP:
            break
        if candidate["score"] < MIN_SCORE_TO_ALERT:
            break  # sorted by score, so everything after this is weaker too
        mint = candidate["mint"]
        if mint in portfolio["positions"]:
            continue
        last = recent.get(mint)
        if last:
            try:
                if now_wall.timestamp() - datetime.fromisoformat(last).timestamp() < ALERT_COOLDOWN_SECONDS:
                    continue
            except ValueError:
                pass

        curve = {
            "virtual_sol_reserves": candidate.get("virtual_sol_reserves"),
            "virtual_token_reserves": candidate.get("virtual_token_reserves"),
        }
        pos, err = open_position(portfolio, candidate, curve, sol_usd, timestamp, MOONSHOT_AUTO_BUY_USD)
        if pos is None:
            print(f"WARN: auto-buy skipped for {candidate.get('name')}: {err}", file=sys.stderr)
            continue

        alert_id = f"{int(time.time())}{daily['count']}"
        alert_row = {
            "alert_id": alert_id,
            "mint": mint,
            "name": candidate.get("name"),
            "symbol": candidate.get("symbol"),
            "bonding_curve_key": candidate.get("bonding_curve_key"),
            "score": candidate.get("score"),
            "graduation_pct": candidate.get("graduation_pct"),
            "sol_per_min": candidate.get("sol_per_min"),
            "signal_ts": timestamp,          # kept so a future tiered/delayed
                                              # release needs no schema change
            "alert_price_usd": pos["entry_price_usd"],
        }
        recent[mint] = timestamp
        daily["count"] += 1
        register_outcome_watch(pending_outcomes, alert_row, timestamp)
        record_decision_on_outcome(pending_outcomes, mint, "buy", timestamp)
        label_rows.append({"row_type": "alert", **alert_row})
        label_rows.append({"row_type": "decision", "decision": "buy", "decided_ts": timestamp,
                           "entry_price_usd": pos["entry_price_usd"],
                           "entry_slippage_pct": pos["entry_slippage_pct"], **alert_row})
        telegram_call("sendMessage", {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": (f"Auto-bought ${pos['amount_usd']:.2f} of {candidate.get('name')} "
                     f"(score {candidate.get('score')}, slippage {pos['entry_slippage_pct']:+.1f}%).\n"
                     f"https://pump.fun/coin/{mint}\n"
                     f"Balance ${portfolio['balance']:.2f}"),
        })

    return label_rows


def _finalize_buy_decision(portfolio, pending, awaiting, pending_outcomes,
                            alert_id, alert, pos, prompt_message_id, timestamp, label_rows):
    """Shared tail for a successful BUY, whichever path resolved the amount
    (a typed number or a quick-percent button) - retires both the original
    alert's buttons AND the "how much?" prompt's buttons (if it had any), so
    neither can be tapped again after the decision is already final."""
    awaiting.clear()
    pending.pop(alert_id, None)
    if alert.get("message_id"):
        telegram_call("editMessageReplyMarkup", {
            "chat_id": TELEGRAM_CHAT_ID, "message_id": alert["message_id"], "reply_markup": {},
        })
    if prompt_message_id:
        telegram_call("editMessageReplyMarkup", {
            "chat_id": TELEGRAM_CHAT_ID, "message_id": prompt_message_id, "reply_markup": {},
        })
    record_decision_on_outcome(pending_outcomes, alert["mint"], "buy", timestamp)
    label_rows.append({"row_type": "decision", "decision": "buy", "decided_ts": timestamp,
                       "entry_price_usd": pos["entry_price_usd"],
                       "entry_slippage_pct": pos["entry_slippage_pct"], **alert})
    telegram_call("sendMessage", {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": (f"Bought ${pos['amount_usd']:.2f} of {alert.get('name')} "
                 f"(slippage {pos['entry_slippage_pct']:+.1f}%).\n"
                 f"Balance ${portfolio['balance']:.2f}"),
    })


def poll_telegram_decisions(alerts_state, portfolio, bonding_state, pending_outcomes,
                            sol_usd, offset_state, timestamp):
    """Read button presses since the last tick and act on them.

    There is no always-on server here to receive a webhook - everything in
    this repo is a short scheduled job - so getUpdates with a persisted offset
    is the only mechanism that fits. Latency is bounded by the tick cadence,
    which is fine for a considered decision.
    """
    label_rows = []
    params = {"timeout": 0, "allowed_updates": ["callback_query", "message"]}
    if offset_state.get("offset"):
        params["offset"] = offset_state["offset"]
    updates = telegram_call("getUpdates", params)
    if not updates:
        return label_rows

    pending = alerts_state.setdefault("pending", {})

    for update in updates:
        # Advance the offset for EVERY update, including ones we ignore, or a
        # single unhandled message would be re-fetched forever.
        offset_state["offset"] = update["update_id"] + 1

        message = update.get("message")
        if message:
            text = (message.get("text") or "").strip()
            if text.lower().startswith("/positions"):
                send_positions_list(portfolio, bonding_state, sol_usd)
                continue
            # A bare number answers an outstanding "how much?" from a BUY press.
            awaiting = alerts_state.setdefault("awaiting_amount", {})
            if awaiting and text:
                amount = _parse_amount(text)
                alert_id = next(iter(awaiting))
                alert = pending.get(alert_id)
                if amount is None:
                    telegram_call("sendMessage", {
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": "That didn't look like a number. Send just the amount, e.g. 50",
                    })
                elif alert is None:
                    awaiting.clear()
                    telegram_call("sendMessage", {
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": "That signal is no longer pending.",
                    })
                else:
                    history = (bonding_state.get(alert.get("bonding_curve_key")) or {}).get("history") or []
                    curve = history[-1] if history else None
                    pos, err = open_position(portfolio, alert, curve, sol_usd, timestamp, amount)
                    if pos is None:
                        # Keep both the question and the alert open so the user
                        # can just send a different number.
                        telegram_call("sendMessage", {
                            "chat_id": TELEGRAM_CHAT_ID,
                            "text": f"Could not buy: {err}. Send another amount, or press IGNORE.",
                        })
                    else:
                        prompt_message_id = awaiting.get(alert_id, {}).get("prompt_message_id")
                        _finalize_buy_decision(portfolio, pending, awaiting, pending_outcomes,
                                                alert_id, alert, pos, prompt_message_id, timestamp, label_rows)
                continue
            continue

        cq = update.get("callback_query")
        if not cq:
            continue
        data = cq.get("data") or ""
        cq_id = cq.get("id")
        msg = cq.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        message_id = msg.get("message_id")

        action, _, key = data.partition(":")
        note = "Unknown action"
        buypct_completed = False  # set True below on a successful buypct purchase

        if action == "ignore":
            alert = pending.pop(key, None)
            if alert is None:
                note = "That signal has already been answered."
            else:
                alerts_state.get("awaiting_amount", {}).pop(key, None)
                record_decision_on_outcome(pending_outcomes, alert["mint"], "ignore", timestamp)
                label_rows.append({"row_type": "decision", "decision": "ignore",
                                   "decided_ts": timestamp, **alert})
                note = f"Ignored {alert.get('name')} - still tracking how it turns out."

        elif action == "buy":
            # BUY doesn't buy anything yet - it asks how much, via quick
            # percent-of-balance buttons or a typed number. The alert stays
            # pending until an amount arrives, so nothing is lost if the user
            # never answers.
            alert = pending.get(key)
            if alert is None:
                note = "That signal has already been answered."
            else:
                awaiting = alerts_state.setdefault("awaiting_amount", {})
                # Only one open "how much?" question at a time, otherwise a
                # bare number in the chat would be ambiguous.
                awaiting.clear()
                prompt = telegram_call("sendMessage", {
                    "chat_id": chat_id or TELEGRAM_CHAT_ID,
                    "text": (f"How much do you want to put into {alert.get('name')}?\n"
                             f"Tap a quick amount, or reply with a number, e.g. {MOONSHOT_SUGGESTED_TRADE_USD:.0f}\n\n"
                             f"Free balance: ${portfolio['balance']:.2f}  "
                             f"(min ${MOONSHOT_MIN_TRADE_USD:.0f})"),
                    "reply_markup": amount_keyboard(key),
                })
                awaiting[key] = {"alert_id": key, "asked_ts": timestamp,
                                  "prompt_message_id": (prompt or {}).get("message_id")}
                note = f"How much? Send a number or tap a quick amount."

        elif action == "buypct":
            # key is "{alert_id}:{pct}" - buypct's own callback_data shape,
            # distinct from every other action's plain alert_id key.
            alert_id, _, pct_str = key.partition(":")
            alert = pending.get(alert_id)
            if alert is None:
                note = "That signal has already been answered."
            else:
                try:
                    pct = float(pct_str)
                except ValueError:
                    pct = None
                amount = round(portfolio["balance"] * pct / 100.0, 2) if pct else None
                if not amount or amount < MOONSHOT_MIN_TRADE_USD:
                    note = f"That's below the ${MOONSHOT_MIN_TRADE_USD:.0f} minimum - send a number instead."
                else:
                    history = (bonding_state.get(alert.get("bonding_curve_key")) or {}).get("history") or []
                    curve = history[-1] if history else None
                    pos, err = open_position(portfolio, alert, curve, sol_usd, timestamp, amount)
                    if pos is None:
                        note = f"Could not buy: {err}. Send a number instead."
                    else:
                        awaiting = alerts_state.setdefault("awaiting_amount", {})
                        prompt_message_id = awaiting.get(alert_id, {}).get("prompt_message_id")
                        _finalize_buy_decision(portfolio, pending, awaiting, pending_outcomes,
                                                alert_id, alert, pos, prompt_message_id, timestamp, label_rows)
                        note = f"Bought ${pos['amount_usd']:.2f} of {alert.get('name')}."
                        buypct_completed = True

        elif action == "sell":
            held = portfolio["positions"].get(key) or {}
            history = (bonding_state.get(held.get("bonding_curve_key")) or {}).get("history") or []
            curve = usable_curve_state(history)
            if curve is None:
                note = "No current curve data - try again next tick."
            else:
                closed = close_position(portfolio, key, curve, sol_usd, timestamp)
                if closed is None:
                    note = "That position is already closed."
                else:
                    label_rows.append({"row_type": "sell", "decided_ts": timestamp, **closed})
                    note = (f"Sold {closed.get('name')}: {closed['pnl_pct']:+.1f}% "
                            f"(${closed['pnl_usd']:+.2f}). Balance ${portfolio['balance']:.2f}")

        # Always answer, or the user's client spins until it times out.
        telegram_call("answerCallbackQuery", {"callback_query_id": cq_id, "text": note[:200]})
        # BUY is not a decision yet - it only opens the "how much?" question,
        # which already sent its own detailed prompt. So leave its buttons
        # alone (the user must still be able to change their mind and press
        # IGNORE) and don't echo the note a second time. A successful buypct
        # is the same story in reverse: it's already fully final and already
        # sent its own confirmation (and retired both messages' buttons)
        # inside _finalize_buy_decision, so it must skip the generic tail
        # below too, or the confirmation gets sent twice.
        if action == "buy" or buypct_completed:
            continue
        # For a real decision: strip the buttons so it's visibly final and
        # re-clicks are inert by construction rather than by state checking.
        if chat_id and message_id and action in ("ignore", "sell"):
            telegram_call("editMessageReplyMarkup",
                          {"chat_id": chat_id, "message_id": message_id, "reply_markup": {}})
        if chat_id:
            telegram_call("sendMessage", {"chat_id": chat_id, "text": note})

    return label_rows


def send_positions_list(portfolio, bonding_state, sol_usd):
    """Reply to /positions with one SELL button per open moonshot position."""
    positions = portfolio.get("positions") or {}
    if not positions:
        telegram_call("sendMessage", {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"No open moonshot positions. Balance ${portfolio['balance']:.2f}",
        })
        return
    for mint, pos in positions.items():
        history = (bonding_state.get(pos.get("bonding_curve_key")) or {}).get("history") or []
        curve = history[-1] if history else None
        # Value the position at what it would ACTUALLY fetch if sold now
        # (fee + exit slippage), not at spot - same reason as on entry.
        sol_out = simulate_curve_sell(curve, pos["qty"]) if curve else None
        if sol_out and sol_usd:
            value = sol_out * sol_usd
            pnl_pct = (value / pos["amount_usd"] - 1) * 100.0
            line = (f"{pos.get('name')} (${pos.get('symbol')})\n"
                    f"in ${pos['amount_usd']:.0f} -> now ${value:.2f}  ({pnl_pct:+.1f}%)")
        else:
            line = f"{pos.get('name')} (${pos.get('symbol')})\nin ${pos['amount_usd']:.0f} -> price unavailable"
        telegram_call("sendMessage", {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": line,
            "reply_markup": {"inline_keyboard": [[{"text": "SELL", "callback_data": f"sell:{mint}"}]]},
        })


def build_dashboard_summary(portfolio, pending_outcomes, bonding_state, sol_usd, timestamp):
    """Compact, pre-computed view for the dashboard.

    Includes the IGNORED signals and how they went, not just the bought ones -
    seeing what passing on a coin actually cost (or saved) is the entire point
    of practising the decision.
    """
    positions = []
    positions_value = 0.0
    for mint, pos in (portfolio.get("positions") or {}).items():
        history = (bonding_state.get(pos.get("bonding_curve_key")) or {}).get("history") or []
        latest = history[-1] if history else None
        sellable = usable_curve_state(history)
        value = None
        if sellable and sol_usd:
            sol_out = simulate_curve_sell(sellable, pos["qty"])
            if sol_out:
                value = round(sol_out * sol_usd, 2)
        positions_value += value if value is not None else pos["amount_usd"]
        positions.append({
            "mint": mint,
            "name": pos.get("name"),
            "symbol": pos.get("symbol"),
            "entry_ts": pos.get("entry_ts"),
            "amount_usd": pos.get("amount_usd"),
            "value_usd": value,
            "pnl_pct": round((value / pos["amount_usd"] - 1) * 100.0, 2)
            if (value is not None and pos.get("amount_usd")) else None,
            "entry_slippage_pct": pos.get("entry_slippage_pct"),
            "entry_graduation_pct": pos.get("entry_graduation_pct"),
            "graduation_pct": (latest or {}).get("graduation_pct"),
            "graduated": bool((latest or {}).get("complete")),
        })

    watching = []
    shadow_tracked = 0
    for mint, watch in (pending_outcomes or {}).items():
        if not watch.get("alerted", True):
            shadow_tracked += 1
            continue  # not shown here - see build_dashboard_summary's shadow_stats
        history = (bonding_state.get(watch.get("bonding_curve_key")) or {}).get("history") or []
        latest = history[-1] if history else None
        watching.append({
            "mint": mint,
            "name": watch.get("name"),
            "symbol": watch.get("symbol"),
            "decision": watch.get("decision"),      # None until the user presses a button
            "signal_ts": watch.get("signal_ts"),
            "alert_score": watch.get("alert_score"),
            "alert_graduation_pct": watch.get("alert_graduation_pct"),
            "graduation_pct": (latest or {}).get("graduation_pct"),
        })
    watching.sort(key=lambda w: w.get("graduation_pct") or 0, reverse=True)

    # Verdicts so far, straight from the append-only decision log. Scoped to
    # real alerts only (alerted=True/missing) - this is what the user was
    # actually shown and can be judged on. Shadow-tracked candidates (cleared
    # the security gate but never made the daily cap/score bar) are counted
    # separately below in shadow_stats, so a wide, low-scoring background
    # sample can't quietly dilute the headline graduation rate.
    stats = {"alerts": 0, "graduated": 0, "did_not_graduate": 0,
             "buy_graduated": 0, "buy_total": 0,
             "ignore_graduated": 0, "ignore_total": 0}
    shadow_stats = {"tracked": shadow_tracked, "graduated": 0, "did_not_graduate": 0}
    # Every signal that has reached a verdict, so the dashboard can show the
    # full record rather than only what's still pending. Without this a signal
    # vanishes from the UI the moment it resolves, which reads as "it was never
    # tracked" even though the data was collected all along.
    alert_rows = {}
    history = []
    try:
        with open(DECISION_LABELS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("row_type") == "alert":
                    stats["alerts"] += 1
                    alert_rows[row.get("mint")] = row
                elif row.get("row_type") == "outcome":
                    graduated = row.get("outcome") == "graduated"
                    if not row.get("alerted", True):
                        shadow_stats["graduated" if graduated else "did_not_graduate"] += 1
                        continue
                    stats["graduated" if graduated else "did_not_graduate"] += 1
                    decision = row.get("decision")
                    if decision in ("buy", "ignore"):
                        stats[f"{decision}_total"] += 1
                        if graduated:
                            stats[f"{decision}_graduated"] += 1
                    alert = alert_rows.get(row.get("mint"), {})
                    history.append({
                        "mint": row.get("mint"),
                        "name": row.get("name") or alert.get("name"),
                        "symbol": alert.get("symbol"),
                        "outcome": row.get("outcome"),
                        "decision": decision,          # None if never answered
                        "signal_ts": row.get("signal_ts"),
                        "recorded_ts": row.get("recorded_ts"),
                        "alert_score": row.get("alert_score"),
                        "alert_graduation_pct": row.get("alert_graduation_pct"),
                        "final_graduation_pct": row.get("final_graduation_pct"),
                        "minutes_to_graduate": row.get("minutes_to_graduate"),
                    })
    except FileNotFoundError:
        pass
    history.reverse()  # newest verdict first
    decided = stats["graduated"] + stats["did_not_graduate"]
    stats["graduation_rate_pct"] = round(stats["graduated"] / decided * 100.0, 1) if decided else None
    shadow_decided = shadow_stats["graduated"] + shadow_stats["did_not_graduate"]
    shadow_stats["graduation_rate_pct"] = (
        round(shadow_stats["graduated"] / shadow_decided * 100.0, 1) if shadow_decided else None)

    return {
        "timestamp": timestamp,
        "balance": round(portfolio.get("balance", 0.0), 2),
        "positions_value": round(positions_value, 2),
        "net_worth": round(portfolio.get("balance", 0.0) + positions_value, 2),
        "starting_balance": MOONSHOT_STARTING_BALANCE_USD,
        "positions": positions,
        "closed": (portfolio.get("closed") or [])[-25:],
        "watching": watching,
        "history": history[:50],
        "stats": stats,
        "shadow_stats": shadow_stats,
        "daily_cap": DAILY_SIGNAL_CAP,
        "min_score_to_alert": MIN_SCORE_TO_ALERT,
        "auto_buy_usd": MOONSHOT_AUTO_BUY_USD,
        "take_profit_pct": MOONSHOT_TAKE_PROFIT_PCT,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not PUMPPORTAL_API_KEY:
        print("ERROR: PUMPPORTAL_API_KEY not set - cannot connect to the discovery stream.", file=sys.stderr)
        return 1

    dev_history = engine.load_json(DEV_HISTORY_PATH, {})
    bonding_state = engine.load_json(BONDING_STATE_PATH, {})
    security_cache = engine.load_json(SECURITY_CACHE_PATH, {})
    alerts_state = engine.load_json(ALERTS_PATH, {})
    portfolio = engine.load_json(PORTFOLIO_PATH, default_portfolio())
    offset_state = engine.load_json(TELEGRAM_OFFSET_PATH, {})
    pending_outcomes = engine.load_json(OUTCOMES_PATH, {})

    timestamp = datetime.now(timezone.utc).isoformat()
    now_wall = datetime.now(timezone.utc)
    new_tokens, migrations, notices = listen_pumpfun_events(PUMPPORTAL_API_KEY)

    record_launches(dev_history, new_tokens, timestamp)
    matched_migrations = record_migrations(dev_history, migrations)

    # ---- Stage 0: decide what's even worth tracking -----------------------
    stage0_rejects = 0
    for token in new_tokens:
        ok, _reason = passes_stage0(token, dev_history, now_wall)
        if not ok:
            stage0_rejects += 1
            continue
        key = token["bondingCurveKey"]
        if key in bonding_state:
            continue
        if len(bonding_state) >= MAX_TRACKED_CURVES:
            break  # ceiling reached; pruning below frees room for the next tick
        bonding_state[key] = {
            "mint": token.get("mint"),
            "name": token.get("name"),
            "symbol": token.get("symbol"),
            "creator": token.get("traderPublicKey"),
            "first_seen": timestamp,
            "history": [],
        }

    # ---- Re-poll EVERY tracked curve, not just this window's launches -----
    # This is what turns a single sighting into a velocity measurement, and it
    # also picks up the curves that weren't RPC-indexed yet on first sight
    # (validation: 4/7 resolved immediately, 7/7 about two minutes later).
    curves = fetch_bonding_curves(list(bonding_state.keys()))
    for key, parsed in curves.items():
        entry = bonding_state[key]
        entry["history"].append({"ts": timestamp, **parsed})
        entry["history"] = entry["history"][-BONDING_HISTORY_WINDOW:]
        entry["last_seen"] = timestamp

    # ---- Stage 1: traction ------------------------------------------------
    stage1_survivors = []
    for key, entry in bonding_state.items():
        ok, _reason = passes_stage1(entry, now_wall)
        if ok:
            stage1_survivors.append((key, entry))
    # Strongest first, so the per-tick security-check budget is spent on the
    # most promising tokens rather than whichever happened to be first.
    stage1_survivors.sort(key=lambda kv: kv[1]["history"][-1].get("graduation_pct", 0), reverse=True)

    # ---- Stage 2: security (expensive, capped per tick) -------------------
    checks_this_tick = [0]
    candidates = []
    for key, entry in stage1_survivors:
        latest = entry["history"][-1]
        total_supply_raw = latest.get("token_total_supply", 0) * TOKEN_DECIMALS
        security = check_security_cached(
            entry["mint"], key, total_supply_raw, security_cache, checks_this_tick
        )
        if security is None or not security.get("passed"):
            continue
        dev = dev_stats(dev_history, entry.get("creator"))
        candidates.append({
            "bonding_curve_key": key,
            "mint": entry["mint"],
            "name": entry.get("name"),
            "symbol": entry.get("symbol"),
            "creator": entry.get("creator"),
            "first_seen": entry["first_seen"],
            "timestamp": timestamp,
            "graduation_pct": latest.get("graduation_pct"),
            "real_sol_reserves": latest.get("real_sol_reserves"),
            # carried so alerts/positions can price off the curve directly
            "virtual_sol_reserves": latest.get("virtual_sol_reserves"),
            "virtual_token_reserves": latest.get("virtual_token_reserves"),
            "sol_per_min": compute_velocity(entry["history"]),
            "samples": len(entry["history"]),
            "security": security,
            "dev_stats": dev,
            "score": score_candidate(entry, security, dev),
        })

    # ---- Stage 3: rank, alert, and collect the user's decisions -----------
    candidates.sort(key=lambda c: c["score"], reverse=True)

    sol_usd = fetch_sol_usd()
    label_rows = []
    # Exits run first, so a position that just graduated or hit take-profit
    # frees its balance before this tick's own auto-buys spend it.
    closed_positions = check_moonshot_positions(portfolio, bonding_state, sol_usd, now_wall)
    for closed in closed_positions:
        label_rows.append({"row_type": "sell", "decided_ts": timestamp, **closed})
        telegram_call("sendMessage", {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": (f"Sold {closed.get('name')} ({closed.get('exit_reason')}): "
                     f"{closed['pnl_pct']:+.1f}% (${closed['pnl_usd']:+.2f}).\n"
                     f"Balance ${portfolio['balance']:.2f}"),
        })
    # Decisions on PREVIOUS (pre-automation) alerts are still read, so a stale
    # BUY/IGNORE someone answers doesn't get lost - see poll_telegram_decisions.
    label_rows += poll_telegram_decisions(
        alerts_state, portfolio, bonding_state, pending_outcomes,
        sol_usd, offset_state, timestamp)
    label_rows += auto_buy_moonshot_signals(
        candidates, alerts_state, portfolio, pending_outcomes,
        sol_usd, timestamp, now_wall)
    # Everything that cleared the security gate but didn't make today's cap
    # or score bar still gets its outcome tracked - see register_shadow_watches
    # for why. Must run after auto_buy_moonshot_signals so a candidate that WAS
    # bought this tick is already in pending_outcomes and gets skipped here
    # rather than downgraded to a shadow watch.
    label_rows += register_shadow_watches(candidates, pending_outcomes, timestamp)

    # ---- Phase 4: did the signals actually pan out? -----------------------
    # (label/analytics bookkeeping only now - the actual sale on graduation
    # already happened above, in check_moonshot_positions)
    outcome_rows = process_outcome_checkpoints(
        pending_outcomes, bonding_state, portfolio, sol_usd, now_wall, timestamp)
    label_rows += outcome_rows

    dev_pruned = prune_dev_history(dev_history, now_wall)
    # Curves under outcome watch are exempt - see prune_bonding_state.
    watched_curves = {w.get("bonding_curve_key") for w in pending_outcomes.values()}
    watched_curves |= {p.get("bonding_curve_key") for p in portfolio.get("positions", {}).values()}
    curves_pruned = prune_bonding_state(bonding_state, now_wall, protected_keys=watched_curves)

    # Overwritten fresh each run (not appended) - the workflow uploads this as
    # a per-run artifact instead of committing an ever-growing file to git.
    with open(RUN_SNAPSHOT_PATH, "w") as f:
        for token in new_tokens:
            f.write(json.dumps({"row_type": "new_token", **token}) + "\n")
        for mig in migrations:
            f.write(json.dumps({"row_type": "migration", **mig}) + "\n")

    # Append-only forever - the permanent record of every signal and every
    # call the user made on it, including the ignores. Analysis later needs
    # both halves: whether the BUYs went on to graduate AND whether the
    # IGNOREs correctly skipped losers.
    if label_rows:
        with open(DECISION_LABELS_PATH, "a") as f:
            for row in label_rows:
                f.write(json.dumps(row) + "\n")

    engine.save_json(DEV_HISTORY_PATH, dev_history)
    engine.save_json(BONDING_STATE_PATH, bonding_state)
    engine.save_json(SECURITY_CACHE_PATH, security_cache)
    engine.save_json(CANDIDATES_PATH, {"timestamp": timestamp, "candidates": candidates})
    engine.save_json(ALERTS_PATH, alerts_state)
    engine.save_json(PORTFOLIO_PATH, portfolio)
    engine.save_json(TELEGRAM_OFFSET_PATH, offset_state)
    engine.save_json(OUTCOMES_PATH, pending_outcomes)
    summary = build_dashboard_summary(portfolio, pending_outcomes, bonding_state, sol_usd, timestamp)
    engine.save_json(DASHBOARD_PATH, summary)
    # Reuses the summary's already-computed mark-to-market rather than
    # re-valuing every position a second time.
    engine.save_json(NETWORTH_HISTORY_PATH, record_networth_snapshot(
        engine.load_json(NETWORTH_HISTORY_PATH, []),
        summary["balance"], summary["positions_value"], now_wall))

    for notice in notices:
        if "errors" in notice:
            print(f"NOTE from PumpPortal: {notice['errors']}", file=sys.stderr)

    top = candidates[0] if candidates else None
    watched_alerted = sum(1 for w in pending_outcomes.values() if w.get("alerted", True))
    watched_shadow = len(pending_outcomes) - watched_alerted
    print(f"Listened {LISTEN_WINDOW_SECONDS}s: {len(new_tokens)} new ({stage0_rejects} rejected at stage 0), "
          f"{len(migrations)} migration(s) ({matched_migrations} matched). "
          f"Tracking {len(bonding_state)} curve(s), read {len(curves)} this tick. "
          f"Stage 1 survivors: {len(stage1_survivors)}; "
          f"security-checked {checks_this_tick[0]}; candidates: {len(candidates)}. "
          + (f"Top: {top['name']} score={top['score']} grad={top['graduation_pct']:.2f}% "
             f"vel={top['sol_per_min']} SOL/min. " if top else "")
          + f"Alerts today: {alerts_state.get('daily', {}).get('count', 0)}/{DAILY_SIGNAL_CAP}, "
          f"{len(alerts_state.get('pending', {}))} awaiting a decision. "
          f"Moonshot: ${portfolio['balance']:.2f} free, {len(portfolio['positions'])} open, "
          f"{len(portfolio['closed'])} closed. "
          f"Watching {watched_alerted} alerted + {watched_shadow} shadow token(s) for graduation. "
          f"Pruned {curves_pruned} curve(s), {dev_pruned} creator(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
