#!/usr/bin/env python3
"""
Solana wallet copy-trading tracker - a third sleeve, independent of both
engine.py's Crypto.com trading and of scan_memecoins.py/scan_pumpfun.py.

WHY THIS EXISTS: both other sleeves' own collected outcome data says their
approach doesn't work - scan_memecoins.py's median return is at/below 0% at
every checkpoint, and buying already-trending tokens or brand-new pump.fun
launches has been unprofitable across the board. The next hypothesis, based
on the user's own research: stop trying to pick winners ourselves, and
instead copy specific Solana wallets that are ALREADY demonstrably
profitable. This script is step 1 of that plan - observation, not trading.

WHY POLLING FREE RPC IS ENOUGH (no paid stream needed): the user's own
reasoning, confirmed technically during validation - any wallet's swaps
(Jupiter-routed or otherwise) are visible on-chain as plain token-balance
deltas in `getTransaction`'s preTokenBalances/postTokenBalances, regardless
of which DEX/router was used. That means detection doesn't need to decode
Jupiter/Raydium/pump.fun instruction formats, and a few minutes of polling
lag is fine for copy-trading - there is no need for the always-on paid
WebSocket stream (e.g. Solana Tracker's Premium tier) that would matter for
sub-second sniping. Validated live against a real wallet
(EBVyC1VakvQvWs1bPoSstS9AHhPux4tmB8yRpuSeuNfQ): both a sell and two buys were
correctly detected from token-balance deltas alone.

WHAT THIS SCRIPT DOES: tracks a fixed candidate pool of wallets (seeded once
via ad hoc browser research across Solana Tracker / Kolscan / uwuu.ai
leaderboards into data/wallet_candidates.json - see that file's
`generated_note`) and polls each one's recent transactions for real,
detected buy/sell activity, appending it to an append-only log. This script
does NOT discover new wallets and does NOT edit wallet_candidates.json - it
only tracks the pool it's given.

CURRENT PHASE: pure observation, per the user's own 3-step plan (find good
wallets -> track real activity for ~2 weeks -> only then connect a real
wallet). No auto-buy, no Phantom connection, no capital at risk of any kind -
this script's only output is a data log for that 2-week decision. Matches
the same discipline scan_pumpfun.py's moonshot half started with before its
own auto-buy phase was added.

RATE LIMITING: validation found the public Solana RPC endpoint returns
HTTP 429 after roughly 11 rapid unpaced requests. Every call in this script
is paced with a fixed sleep (RPC_CALL_PACING_SECONDS) and retried with
backoff on 429, the same treatment scan_memecoins.py already gives
GeckoTerminal's rate limit.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import engine  # reused only for load_json/save_json (atomic writes)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
# Static seed list, built once via browser research - see that file's own
# generated_note. This script reads it but never writes it.
WALLET_CANDIDATES_PATH = os.path.join(DATA_DIR, "wallet_candidates.json")
SCAN_STATE_PATH = os.path.join(DATA_DIR, "wallet_scan_state.json")
# Append-only forever. Unlike scan_pumpfun.py's run_snapshot (~49k
# launches/day, deliberately kept out of git), a tracked pool of ~50 wallets
# produces at most a few dozen real trade events a day - small enough to
# just commit directly, no artifact-upload dance needed.
ACTIVITY_LABELS_PATH = os.path.join(DATA_DIR, "wallet_activity_labels.jsonl")

# Same fallback + `or` (not .get default) reasoning as scan_pumpfun.py: the
# workflow references ${{ secrets.SOLANA_RPC_URL }} unconditionally, which
# GitHub sets to an EMPTY STRING when the secret doesn't exist, and only
# `or` (not a dict-default) treats that the same as "unset".
SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL") or "https://api.mainnet-beta.solana.com"

LAMPORTS_PER_SOL = 1e9

# --- Pacing / budget --------------------------------------------------------
# See module docstring: ~11 unpaced calls trips a 429 on the public endpoint.
# 1.2s keeps well clear of that even across a whole tick's worth of calls.
RPC_CALL_PACING_SECONDS = 1.2
# getSignaturesForAddress's own documented max. Used as the page size, not a
# "good enough" cap - see SIGNATURE_PAGES_PER_WALLET below for why a small
# fixed cap here would be unsafe.
SIGNATURES_PAGE_LIMIT = 1000
# Safety net on pagination depth per wallet per tick, in case a wallet is
# producing more than SIGNATURES_PAGE_LIMIT signatures between two ticks.
# Confirmed live during testing that this is a real risk, not hypothetical:
# one candidate wallet fired 25 transactions inside a single second, almost
# all failing with the same InstructionError (bot-like burst activity, not
# manual trading). A naive single small-limit fetch would have silently and
# PERMANENTLY dropped anything past the newest N - the next tick's cursor
# starts after whatever was seen, so unseen older signatures are never
# retried. Paginating with `before` until `until` is reached (or this cap)
# avoids that; if the cap is hit, the cursor only advances to the oldest
# fully-covered page, so the remainder is retried next tick instead of lost.
SIGNATURE_PAGES_PER_WALLET = 5
# Per-wallet cap on getTransaction calls (i.e. real, non-failed signatures)
# in one tick - the actually expensive step, unlike listing signatures.
# Bounds how much of the shared per-tick budget one hyperactive wallet (bot
# or genuine high-frequency trader) can consume; a wallet that exceeds this
# is picked back up next tick rather than blocking everyone else.
MAX_TX_FETCHES_PER_WALLET_PER_TICK = 30
# Hard ceiling on getTransaction calls in a single run, across all wallets
# combined - a safety net against one burst of simultaneous activity blowing
# the job's time budget. Wallets not yet reached this tick are simply
# deferred to the next one; their cursor is untouched so nothing is skipped.
MAX_TRANSACTION_FETCHES_PER_TICK = 80


def _as_float(v, default=0.0):
    """Duplicated from the other sleeves rather than imported - each stays
    standalone (only engine.load_json/save_json is ever shared)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_candidates():
    """The tracked pool, as a flat list of {address, names, tier}. Missing
    file or empty pool both come back as [] - main() decides that's fatal,
    this function just reports what it found."""
    data = engine.load_json(WALLET_CANDIDATES_PATH, {"wallets": []})
    out = []
    for w in data.get("wallets", []):
        addr = w.get("address")
        if not addr:
            continue
        out.append({"address": addr, "names": w.get("names") or [], "tier": w.get("tier")})
    return out


# --- Solana RPC --------------------------------------------------------------

def _rpc_post(payload, timeout=20):
    req = urllib.request.Request(
        SOLANA_RPC_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "cryptobot-wallet-scan/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _rpc_call_with_retry(payload, timeout=20, max_retries=2):
    """Same 429-retry-with-backoff treatment as scan_memecoins.py's
    geckoterminal_get_json - a 429 here is routine, not exceptional."""
    for attempt in range(max_retries + 1):
        try:
            return _rpc_post(payload, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == max_retries:
                raise
            wait = e.headers.get("Retry-After")
            default_delay = 3 * (attempt + 1)
            try:
                delay = max(float(wait), default_delay) if wait else default_delay
            except ValueError:
                delay = default_delay
            print(f"WARN: 429 from Solana RPC, retrying in {delay:.0f}s "
                  f"({attempt + 1}/{max_retries})...", file=sys.stderr)
            time.sleep(delay)


def fetch_newest_signature(wallet):
    """Just the single most recent signature - used only to seed a cold-start
    cursor, where no backfill is wanted (see main())."""
    resp = _rpc_call_with_retry({
        "jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
        "params": [wallet, {"limit": 1}],
    })
    result = (resp or {}).get("result") or []
    return result[0] if result else None


def fetch_signatures_since(wallet, until_sig):
    """Every signature newer than `until_sig`, newest-first, paginating with
    `before` past SIGNATURES_PAGE_LIMIT if needed.

    Returns (signatures, complete). complete=False means SIGNATURE_PAGES_PER_WALLET
    was exhausted before reaching `until_sig` - there may be more, older,
    still-unseen signatures beyond what's returned. Callers must NOT advance
    their cursor past the oldest signature actually returned in that case, or
    the ungathered remainder would be silently skipped forever (this is
    exactly the bug a fixed single-page fetch had - see module constants).
    """
    all_sigs = []
    before = None
    for _ in range(SIGNATURE_PAGES_PER_WALLET):
        params = {"limit": SIGNATURES_PAGE_LIMIT, "until": until_sig}
        if before:
            params["before"] = before
        resp = _rpc_call_with_retry({
            "jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
            "params": [wallet, params],
        })
        batch = (resp or {}).get("result") or []
        if not batch:
            return all_sigs, True   # nothing more - genuinely reached `until`
        all_sigs.extend(batch)
        if len(batch) < SIGNATURES_PAGE_LIMIT:
            return all_sigs, True   # short page = the RPC itself stopped at `until`
        before = batch[-1]["signature"]
        time.sleep(RPC_CALL_PACING_SECONDS)  # pace the extra page fetch too
    return all_sigs, False   # exhausted the page budget, `until` not confirmed reached


def fetch_transaction(signature):
    """maxSupportedTransactionVersion=1 (not 0) - validation hit
    'Transaction version (1) is not supported' with 0, because real modern
    wallets commonly use versioned (v0+) transactions with address lookup
    tables."""
    resp = _rpc_call_with_retry({
        "jsonrpc": "2.0", "id": 1, "method": "getTransaction",
        "params": [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 1}],
    })
    return (resp or {}).get("result")


def parse_wallet_trade_events(tx, wallet):
    """Every non-zero token-balance change belonging to `wallet` in this
    transaction, protocol-agnostic (no Jupiter/Raydium/pump.fun instruction
    decoding - validated during Phase 0 as sufficient on its own). A positive
    delta is a buy (tokens received), negative is a sell (tokens sent away).
    `sol_delta` is the wallet's native SOL balance change for the whole
    transaction, carried on every event as shared context (most of these
    trades are token<->SOL swaps, so it's usually the trade's other leg).

    A failed transaction (meta.err is not None) produced no real balance
    change and is skipped - it consumed a signature slot but nothing
    actually traded.
    """
    meta = (tx or {}).get("meta") or {}
    if meta.get("err") is not None:
        return []

    message = ((tx or {}).get("transaction") or {}).get("message") or {}
    account_keys = message.get("accountKeys") or []
    wallet_idx = None
    for i, ak in enumerate(account_keys):
        pubkey = ak.get("pubkey") if isinstance(ak, dict) else ak
        if pubkey == wallet:
            wallet_idx = i
            break

    sol_delta = None
    pre_bal = meta.get("preBalances") or []
    post_bal = meta.get("postBalances") or []
    if wallet_idx is not None and wallet_idx < len(pre_bal) and wallet_idx < len(post_bal):
        sol_delta = round((post_bal[wallet_idx] - pre_bal[wallet_idx]) / LAMPORTS_PER_SOL, 9)

    pre_tok, post_tok = {}, {}
    for entry in meta.get("preTokenBalances") or []:
        if entry.get("owner") != wallet:
            continue
        mint = entry.get("mint")
        amt = _as_float((entry.get("uiTokenAmount") or {}).get("uiAmount"))
        pre_tok[mint] = pre_tok.get(mint, 0.0) + amt
    for entry in meta.get("postTokenBalances") or []:
        if entry.get("owner") != wallet:
            continue
        mint = entry.get("mint")
        amt = _as_float((entry.get("uiTokenAmount") or {}).get("uiAmount"))
        post_tok[mint] = post_tok.get(mint, 0.0) + amt

    events = []
    for mint in set(pre_tok) | set(post_tok):
        delta = post_tok.get(mint, 0.0) - pre_tok.get(mint, 0.0)
        if abs(delta) < 1e-9:
            continue
        events.append({
            "mint": mint,
            "token_delta": round(delta, 9),
            "direction": "buy" if delta > 0 else "sell",
            "sol_delta": sol_delta,
        })
    return events


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    candidates = load_candidates()
    if not candidates:
        print("ERROR: data/wallet_candidates.json has no wallets to track.", file=sys.stderr)
        return 1

    scan_state = engine.load_json(SCAN_STATE_PATH, {})
    timestamp = datetime.now(timezone.utc).isoformat()

    new_rows = []
    tx_fetches_this_tick = 0
    wallets_checked = 0
    wallets_deferred = 0
    cold_started = 0
    errors = 0

    for wallet_info in candidates:
        wallet = wallet_info["address"]
        if tx_fetches_this_tick >= MAX_TRANSACTION_FETCHES_PER_TICK:
            wallets_deferred += 1
            continue

        state = scan_state.get(wallet)
        time.sleep(RPC_CALL_PACING_SECONDS)

        if state is None:
            # Cold start: seed the cursor at the current newest signature with
            # NO backfill. Some candidates trade hundreds of times a month
            # (Lynk: 1216 trades/30d) - backfilling all of that on first sight
            # would blow the RPC budget on day one for no real benefit, since
            # the whole point is watching what happens FROM NOW, not history.
            try:
                newest = fetch_newest_signature(wallet)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
                print(f"WARN: signature lookup failed for {wallet[:8]}...: {e}", file=sys.stderr)
                errors += 1
                continue
            wallets_checked += 1
            if newest is not None:
                scan_state[wallet] = {
                    "last_signature": newest["signature"],
                    "last_checked_ts": timestamp,
                    "first_seen_ts": timestamp,
                }
                cold_started += 1
            # else: wallet has no transaction history at all yet - nothing to
            # seed a cursor from; state stays absent and it's retried like new
            # next tick.
            continue

        try:
            sigs, complete = fetch_signatures_since(wallet, state["last_signature"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            print(f"WARN: signature lookup failed for {wallet[:8]}...: {e}", file=sys.stderr)
            errors += 1
            continue

        wallets_checked += 1
        if not sigs:
            scan_state[wallet] = {**state, "last_checked_ts": timestamp}
            continue

        if not complete:
            print(f"WARN: {wallet[:8]}... produced more than "
                  f"{SIGNATURE_PAGES_PER_WALLET * SIGNATURES_PAGE_LIMIT} signatures since "
                  f"the last check - likely bot/bursty, not a discretionary trader. The "
                  f"untraced older portion of this gap will not be retried (same accepted "
                  f"tradeoff as cold start's no-backfill).", file=sys.stderr)

        # Oldest-to-newest, so if a budget runs out partway through, the
        # cursor only advances past what was actually attempted.
        newest_processed = state["last_signature"]
        tx_fetches_this_wallet = 0
        for sig_info in reversed(sigs):
            if tx_fetches_this_tick >= MAX_TRANSACTION_FETCHES_PER_TICK:
                break
            if tx_fetches_this_wallet >= MAX_TX_FETCHES_PER_WALLET_PER_TICK:
                break
            newest_processed = sig_info["signature"]
            if sig_info.get("err") is not None:
                continue  # failed tx, nothing actually traded

            time.sleep(RPC_CALL_PACING_SECONDS)
            tx_fetches_this_tick += 1
            tx_fetches_this_wallet += 1
            try:
                tx = fetch_transaction(sig_info["signature"])
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
                print(f"WARN: getTransaction failed for {sig_info['signature'][:12]}...: {e}", file=sys.stderr)
                errors += 1
                continue
            if tx is None:
                continue

            for ev in parse_wallet_trade_events(tx, wallet):
                new_rows.append({
                    "row_type": "trade",
                    "wallet": wallet,
                    "names": wallet_info.get("names"),
                    "tier": wallet_info.get("tier"),
                    "signature": sig_info["signature"],
                    "block_time": sig_info.get("blockTime"),
                    "detected_ts": timestamp,
                    **ev,
                })

        scan_state[wallet] = {
            "last_signature": newest_processed,
            "last_checked_ts": timestamp,
            "first_seen_ts": state.get("first_seen_ts", timestamp),
        }

    if new_rows:
        with open(ACTIVITY_LABELS_PATH, "a") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")

    engine.save_json(SCAN_STATE_PATH, scan_state)

    print(f"Checked {wallets_checked}/{len(candidates)} wallet(s) "
          f"({cold_started} cold-started, {wallets_deferred} deferred to next tick, "
          f"{errors} error(s)). Fetched {tx_fetches_this_tick} transaction(s), "
          f"logged {len(new_rows)} trade event(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
