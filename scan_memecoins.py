#!/usr/bin/env python3
"""
Standalone memecoin trend-scanner - independent of the Crypto.com trading
sleeves in engine.py/broker.py. Data collection + heuristic screening +
Telegram alerts only, no trading, no capital at risk.

Price/volume/rank source: GeckoTerminal's public /networks/{network}/trending_pools
endpoint (no API key). Covers the same Solana/Base/BSC memecoins the FOMO app
surfaces, since FOMO's own trending data is gated behind login with no public API.

Security source (only for tokens that already pass the momentum pre-filter,
to conserve calls): RugCheck.xyz for Solana (mint/freeze authority, holder
concentration, risk flags), GoPlus Security API for Base/BSC (honeypot,
mintable, blacklist, owner/creator holding %).

This is v2, still a hand-written heuristic pre-filter + hard security gate,
not a learned model - there's no historical outcome data yet to train or
validate anything on. All thresholds below are deliberately explicit and
named so they're easy to find and retune once outcome data exists - see the
plan doc for why they must NOT be hand-tuned further by guessing.

Two kinds of historical record are kept, deliberately NOT the same file:

  1. data/memecoin_labels.jsonl - the actual training data. Every token that
     passes both filters gets ONE entry-row (T0: price/mcap/features at the
     moment it qualified) plus a handful of outcome-checkpoint rows at fixed
     offsets after entry (10m/30m/1h/3h/6h: price/mcap/return at that point).
     This directly IS the trade_labels table a pattern-learning pass needs -
     small, append-only, safe to commit to git forever.

  2. The full unfiltered per-tick snapshot (every trending pool, not just
     candidates) is NOT committed to git at all - at production tick density
     that would add several GB/month to a public repo. Instead each run
     writes its own data/memecoin_run_snapshot.jsonl (overwritten fresh each
     run, not appended) which the GitHub Actions workflow uploads as a
     per-run artifact (see memecoin_scan.yml) with its own retention window,
     so the raw firehose is still fully recoverable without living in git
     history forever.

Runs as a short-lived burst loop (not a long-running daemon): each GitHub
Actions invocation (triggered every 5 minutes, the practical floor for
scheduled Actions cron) takes a snapshot every TICK_INTERVAL_SECONDS for
LOOP_DURATION_SECONDS, then commits once. This approximates much denser
polling than a single per-invocation snapshot would give, without needing
a persistent server.
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
RUN_SNAPSHOT_PATH = os.path.join(DATA_DIR, "memecoin_run_snapshot.jsonl")  # per-run only, not committed - see workflow artifact upload
HOT_STATE_PATH = os.path.join(DATA_DIR, "memecoin_hot_state.json")
SECURITY_CACHE_PATH = os.path.join(DATA_DIR, "memecoin_security_cache.json")
ALERTED_PATH = os.path.join(DATA_DIR, "memecoin_alerted.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "memecoin_candidates.json")
PENDING_CHECKPOINTS_PATH = os.path.join(DATA_DIR, "memecoin_pending_checkpoints.json")
LABELS_JSONL_PATH = os.path.join(DATA_DIR, "memecoin_labels.jsonl")

NETWORKS = ["solana", "base", "bsc"]
GECKOTERMINAL_TRENDING_URL = "https://api.geckoterminal.com/api/v2/networks/{network}/trending_pools"
GECKOTERMINAL_POOL_URL = "https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}"
RUGCHECK_REPORT_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"
GOPLUS_TOKEN_SECURITY_URL = "https://api.gopluslabs.io/api/v1/token_security/{chain_id}?contract_addresses={address}"
EVM_CHAIN_IDS = {"base": "8453", "bsc": "56"}

# --- Timing ------------------------------------------------------------
TICK_INTERVAL_SECONDS = 30
LOOP_DURATION_SECONDS = 240  # ~4 min of ticks per 5-min cron invocation,
                              # leaving ~1 min headroom for setup/commit/push

# --- Momentum pre-filter (cheap, runs every tick on every trending pool) -
MIN_MARKET_CAP_USD = 50_000
HOT_STATE_WINDOW = 20          # keep last N ticks per pool (~10 min at 30s cadence)
HOT_STATE_MAX_AGE_SECONDS = 7200  # drop a pool from hot state if unseen for 2h
MIN_HISTORY_FOR_VELOCITY = 3   # need this many prior ticks before trusting
                                # rank_velocity/h1_accel; below that, fall back
                                # to the base momentum conditions only

# --- Security hard-filter (expensive, runs only on momentum candidates,
# result cached per address) ---------------------------------------------
SECURITY_CACHE_TTL_SECONDS = 1800   # 30 min
MAX_SECURITY_CHECKS_PER_TICK = 8    # defensive cap against a burst of new candidates
SOLANA_MAX_TOP10_PCT = 40.0         # combined top-10 holder %, RugCheck's own pct scale (0-100)
EVM_MAX_OWNER_CREATOR_FRACTION = 0.20  # combined owner+creator holding, GoPlus's 0-1 fraction scale

# --- Alerting -------------------------------------------------------------
ALERT_COOLDOWN_SECONDS = 7200  # don't re-alert the same token within 2h

# --- Outcome checkpoints (the actual training-data generator) -------------
CHECKPOINT_OFFSETS_MINUTES = [10, 30, 60, 180, 360]  # 10m, 30m, 1h, 3h, 6h after entry
MAX_CHECKPOINT_LOOKUPS_PER_TICK = 8  # defensive cap, same rationale as security checks

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def http_get_json(url, timeout=15):
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "cryptobot-memecoin-scan/2.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def geckoterminal_get_json(url, timeout=15, max_retries=2):
    """GeckoTerminal's free tier rate-limits tighter than its documented 10/min
    in practice (empirically confirmed - see plan doc). A 429 here is routine,
    not exceptional, so retry with backoff instead of just dropping the tick's
    data. Honors Retry-After when the API sends one."""
    for attempt in range(max_retries + 1):
        try:
            return http_get_json(url, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == max_retries:
                raise
            # GeckoTerminal sometimes sends "Retry-After: 0", which is useless -
            # floor it at our own escalating minimum instead of trusting it blindly.
            wait = e.headers.get("Retry-After")
            default_delay = 3 * (attempt + 1)
            try:
                delay = max(float(wait), default_delay) if wait else default_delay
            except ValueError:
                delay = default_delay
            print(f"WARN: 429 from GeckoTerminal, retrying in {delay:.0f}s ({attempt + 1}/{max_retries})...", file=sys.stderr)
            time.sleep(delay)


def send_telegram(text):
    """Duplicated from fetch_and_run.py rather than imported, so this sleeve
    stays fully decoupled from the Crypto.com-specific trading script."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("WARN: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set, skipping send. "
              "Message would have been:\n" + text, file=sys.stderr)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"ERROR sending Telegram message: {e}", file=sys.stderr)


def _as_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --- GeckoTerminal: trending pools + rank --------------------------------

def fetch_trending_pools(network):
    url = GECKOTERMINAL_TRENDING_URL.format(network=network)
    try:
        data = geckoterminal_get_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"WARN: failed to fetch trending pools for {network}: {e}", file=sys.stderr)
        return []
    pools = []
    for rank, entry in enumerate(data.get("data", []), start=1):
        attrs = entry.get("attributes", {})
        base_token_id = (
            entry.get("relationships", {}).get("base_token", {}).get("data", {}).get("id", "")
        )
        # base_token_id looks like "solana_<mint>" or "base_0x..." - strip the
        # network prefix to get the actual token address. This is the token
        # contract itself, NOT attrs["address"] (which is the pool/pair
        # address - security APIs need the token, not the pool, or they 400).
        base_token_address = base_token_id.split("_", 1)[1] if "_" in base_token_id else None
        pools.append({
            "network": network,
            "id": entry.get("id"),
            "address": attrs.get("address"),  # pool/pair address, for reference only
            "base_token_address": base_token_address,
            "name": attrs.get("name"),
            "rank": rank,  # GeckoTerminal's own trending order for this network
            "price_usd": attrs.get("base_token_price_usd"),
            "price_change_pct": attrs.get("price_change_percentage", {}) or {},
            "volume_usd": attrs.get("volume_usd", {}) or {},
            "transactions": attrs.get("transactions", {}) or {},
            "market_cap_usd": attrs.get("market_cap_usd"),
            "fdv_usd": attrs.get("fdv_usd"),
            "pool_created_at": attrs.get("pool_created_at"),
        })
    return pools


def fetch_pool_price(network, pool_address):
    """Single-pool lookup, used only at checkpoint time (not every tick) to
    get this specific token's current price/mcap without re-fetching the
    whole trending list."""
    url = GECKOTERMINAL_POOL_URL.format(network=network, pool_address=pool_address)
    try:
        data = geckoterminal_get_json(url, timeout=15)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"WARN: pool price lookup failed for {network}/{pool_address}: {e}", file=sys.stderr)
        return None
    attrs = (data.get("data") or {}).get("attributes") or {}
    return {
        "price_usd": attrs.get("base_token_price_usd"),
        "market_cap_usd": attrs.get("market_cap_usd"),
        "fdv_usd": attrs.get("fdv_usd"),
    }


# --- Hot state: bounded rolling window per pool, for velocity/acceleration -

def update_hot_state(hot_state, pool, timestamp):
    pool_id = pool["id"]
    entry = hot_state.setdefault(pool_id, {"name": pool["name"], "network": pool["network"], "history": []})
    txns_h1 = pool["transactions"].get("h1", {}) or {}
    entry["history"].append({
        "ts": timestamp,
        "rank": pool["rank"],
        "h1": _as_float(pool["price_change_pct"].get("h1")),
        "h6": _as_float(pool["price_change_pct"].get("h6")),
        "buys_h1": _as_float(txns_h1.get("buys")),
        "sells_h1": _as_float(txns_h1.get("sells")),
    })
    entry["history"] = entry["history"][-HOT_STATE_WINDOW:]
    entry["last_seen"] = timestamp

    history = entry["history"]
    features = {"rank_velocity": None, "h1_accel": None, "history_len": len(history)}
    if len(history) >= MIN_HISTORY_FOR_VELOCITY:
        baseline = history[-MIN_HISTORY_FOR_VELOCITY]
        latest = history[-1]
        # positive rank_velocity = climbing toward rank 1 (improving)
        features["rank_velocity"] = baseline["rank"] - latest["rank"]
        features["h1_accel"] = latest["h1"] - baseline["h1"]
    return features


def prune_hot_state(hot_state, now_wall):
    cutoff = now_wall.timestamp() - HOT_STATE_MAX_AGE_SECONDS
    stale = []
    for pool_id, entry in hot_state.items():
        last_seen = entry.get("last_seen")
        try:
            last_seen_ts = datetime.fromisoformat(last_seen).timestamp() if last_seen else 0
        except ValueError:
            last_seen_ts = 0
        if last_seen_ts < cutoff:
            stale.append(pool_id)
    for pool_id in stale:
        del hot_state[pool_id]


# --- Momentum pre-filter --------------------------------------------------

def is_momentum_candidate(pool, features):
    """Cheap pre-filter (not a learned model): positive momentum on both h1
    and h6, more buyers than sellers in the last hour, a market-cap floor,
    and - once enough history exists for this pool - rank improving and h1
    momentum still accelerating rather than fading."""
    price_change = pool["price_change_pct"]
    h1 = _as_float(price_change.get("h1"))
    h6 = _as_float(price_change.get("h6"))
    txns_h1 = pool["transactions"].get("h1", {}) or {}
    buys = _as_float(txns_h1.get("buys"))
    sells = _as_float(txns_h1.get("sells"))
    market_cap = _as_float(pool["market_cap_usd"]) or _as_float(pool["fdv_usd"])

    if not (h1 > 0 and h6 > 0 and buys > sells and market_cap >= MIN_MARKET_CAP_USD):
        return False
    if features["rank_velocity"] is not None and features["rank_velocity"] < 0:
        return False
    if features["h1_accel"] is not None and features["h1_accel"] < 0:
        return False
    return True


# --- Security hard-filter (Solana via RugCheck, Base/BSC via GoPlus) -----

def check_security_solana(address):
    url = RUGCHECK_REPORT_URL.format(mint=address)
    try:
        data = http_get_json(url, timeout=15)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"WARN: RugCheck lookup failed for {address}: {e}", file=sys.stderr)
        return None
    mint_authority_active = data.get("mintAuthority") is not None
    freeze_authority_active = data.get("freezeAuthority") is not None
    top_holders = data.get("topHolders") or []
    top10_pct = sum(_as_float(h.get("pct")) for h in top_holders[:10])
    risks = data.get("risks") or []
    has_danger_risk = any(r.get("level") == "danger" for r in risks)
    passed = (
        not mint_authority_active
        and not freeze_authority_active
        and top10_pct <= SOLANA_MAX_TOP10_PCT
        and not has_danger_risk
    )
    return {
        "passed": passed,
        "mint_authority_active": mint_authority_active,
        "freeze_authority_active": freeze_authority_active,
        "top10_pct": round(top10_pct, 1),
        "has_danger_risk": has_danger_risk,
    }


def check_security_evm(address, network):
    chain_id = EVM_CHAIN_IDS.get(network)
    if chain_id is None or not address:
        return None
    url = GOPLUS_TOKEN_SECURITY_URL.format(chain_id=chain_id, address=address.lower())
    try:
        data = http_get_json(url, timeout=15)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"WARN: GoPlus lookup failed for {address}: {e}", file=sys.stderr)
        return None
    result_map = data.get("result") or {}
    result = result_map.get(address.lower()) or result_map.get(address)
    if not result:
        return None
    owner_creator_pct = _as_float(result.get("owner_percent")) + _as_float(result.get("creator_percent"))
    passed = (
        result.get("is_honeypot") == "0"
        and result.get("is_mintable") == "0"
        and result.get("is_blacklisted") == "0"
        and result.get("cannot_buy") == "0"
        and owner_creator_pct <= EVM_MAX_OWNER_CREATOR_FRACTION
    )
    return {
        "passed": passed,
        "is_honeypot": result.get("is_honeypot") == "1",
        "is_mintable": result.get("is_mintable") == "1",
        "is_blacklisted": result.get("is_blacklisted") == "1",
        "owner_creator_pct": round(owner_creator_pct * 100, 1),
    }


def check_security_cached(pool, security_cache, checks_this_tick):
    address = pool["base_token_address"]
    if not address:
        return None
    cached = security_cache.get(address)
    now = time.time()
    if cached and (now - cached.get("checked_at", 0) < SECURITY_CACHE_TTL_SECONDS):
        return cached
    if checks_this_tick[0] >= MAX_SECURITY_CHECKS_PER_TICK:
        return cached  # defer fresh check to a later tick; may be None
    checks_this_tick[0] += 1

    if pool["network"] == "solana":
        result = check_security_solana(address)
    else:
        result = check_security_evm(address, pool["network"])
    if result is None:
        return None
    result["checked_at"] = now
    security_cache[address] = result
    return result


# --- Outcome checkpoints / labels table -----------------------------------

def register_pending_checkpoint(pending, pool, timestamp):
    """Called the first time a pool qualifies (passes momentum + security).
    Writes the T0 entry row straight to the labels buffer, and schedules
    future checkpoint lookups (10m/30m/1h/3h/6h) so we eventually learn what
    actually happened after entry - not just that entry looked good."""
    pool_id = pool["id"]
    if pool_id in pending:
        return None
    entry_price = _as_float(pool.get("price_usd"))
    entry_mcap = _as_float(pool["market_cap_usd"]) or _as_float(pool["fdv_usd"])
    pending[pool_id] = {
        "network": pool["network"],
        "pool_address": pool["address"],
        "base_token_address": pool["base_token_address"],
        "name": pool["name"],
        "entry_ts": timestamp,
        "entry_price_usd": entry_price,
        "entry_mcap_usd": entry_mcap,
        "checkpoints_done": {},
    }
    return {
        "row_type": "entry",
        "pool_id": pool_id,
        "network": pool["network"],
        "base_token_address": pool["base_token_address"],
        "name": pool["name"],
        "entry_ts": timestamp,
        "entry_price_usd": entry_price,
        "entry_mcap_usd": entry_mcap,
    }


def process_due_checkpoints(pending, now_wall, lookups_this_tick):
    """Runs once per tick over every token still awaiting outcome checkpoints.
    Only fetches a fresh price for checkpoints that are actually due, and
    only up to MAX_CHECKPOINT_LOOKUPS_PER_TICK - if there's a burst, the rest
    are simply checked on the next tick a little late, which is fine (we
    label against wall-clock elapsed time, not against tick boundaries)."""
    label_rows = []
    completed_pool_ids = []

    for pool_id, entry in pending.items():
        entry_ts = datetime.fromisoformat(entry["entry_ts"])
        elapsed_minutes = (now_wall - entry_ts).total_seconds() / 60.0
        done = entry["checkpoints_done"]

        for offset in CHECKPOINT_OFFSETS_MINUTES:
            key = str(offset)
            if key in done or elapsed_minutes < offset:
                continue
            if lookups_this_tick[0] >= MAX_CHECKPOINT_LOOKUPS_PER_TICK:
                break  # defer remaining due checkpoints (this token and others) to a later tick
            if lookups_this_tick[0] > 0:
                time.sleep(1.5)  # stagger checkpoint lookups too - they share GeckoTerminal's rate limit
            lookups_this_tick[0] += 1

            current = fetch_pool_price(entry["network"], entry["pool_address"])
            checkpoint_ts = now_wall.isoformat()
            if current is None or entry["entry_price_usd"] <= 0:
                done[key] = checkpoint_ts  # mark attempted so we don't retry forever on a dead pool
                continue

            price_now = _as_float(current.get("price_usd"))
            mcap_now = _as_float(current.get("market_cap_usd")) or _as_float(current.get("fdv_usd"))
            return_pct = (price_now - entry["entry_price_usd"]) / entry["entry_price_usd"] * 100.0

            label_rows.append({
                "row_type": "checkpoint",
                "pool_id": pool_id,
                "network": entry["network"],
                "base_token_address": entry["base_token_address"],
                "name": entry["name"],
                "entry_ts": entry["entry_ts"],
                "checkpoint_ts": checkpoint_ts,
                "offset_minutes": offset,
                "entry_price_usd": entry["entry_price_usd"],
                "price_usd": price_now,
                "return_pct": round(return_pct, 2),
                "entry_mcap_usd": entry["entry_mcap_usd"],
                "mcap_usd": mcap_now,
            })
            done[key] = checkpoint_ts

        if len(done) >= len(CHECKPOINT_OFFSETS_MINUTES):
            completed_pool_ids.append(pool_id)

    for pool_id in completed_pool_ids:
        del pending[pool_id]

    return label_rows


# --- Formatting -------------------------------------------------------------

def format_alert(pool, features, security):
    h1 = _as_float(pool["price_change_pct"].get("h1"))
    h24 = _as_float(pool["price_change_pct"].get("h24"))
    vol_h24 = _as_float(pool["volume_usd"].get("h24"))
    mcap = _as_float(pool["market_cap_usd"]) or _as_float(pool["fdv_usd"])
    lines = [
        f"[MEMECOIN SCAN] {pool['name']} ({pool['network']})",
        f"rank #{pool['rank']} (velocity {features['rank_velocity']:+d})" if features["rank_velocity"] is not None
        else f"rank #{pool['rank']} (velocity n/a, too new)",
        f"1h {h1:+.1f}% | 24h {h24:+.1f}% | vol24h ${vol_h24:,.0f} | mcap ${mcap:,.0f}",
    ]
    if pool["network"] == "solana":
        lines.append(f"security: top10 holders {security['top10_pct']:.1f}%, mint/freeze authority inactive, no danger-level risk flags")
    else:
        lines.append(f"security: owner+creator hold {security['owner_creator_pct']:.1f}%, not a honeypot, not mintable")
    lines.append(f"token: {pool['base_token_address']}")
    return "\n".join(lines)


# --- Main loop ---------------------------------------------------------------

def run_tick(hot_state, security_cache, alerted, pending_checkpoints, jsonl_buffer, candidates_out, label_rows_out):
    timestamp = datetime.now(timezone.utc).isoformat()
    checks_this_tick = [0]
    checkpoint_lookups_this_tick = [0]

    for i, network in enumerate(NETWORKS):
        if i > 0:
            time.sleep(3)  # stagger GeckoTerminal calls within a tick to avoid bursting its rate limit
        pools = fetch_trending_pools(network)
        jsonl_buffer.append(json.dumps({"timestamp": timestamp, "network": network, "pools": pools}))

        for pool in pools:
            features = update_hot_state(hot_state, pool, timestamp)
            if not is_momentum_candidate(pool, features):
                continue

            security = check_security_cached(pool, security_cache, checks_this_tick)
            if security is None or not security["passed"]:
                continue

            candidates_out.append({
                "timestamp": timestamp, "network": network, "name": pool["name"],
                "address": pool["address"], "base_token_address": pool["base_token_address"],
                "rank": pool["rank"], "features": features, "security": security,
                # display fields for the dashboard chart - not used by filter logic itself
                "price_change_pct": pool["price_change_pct"],
                "volume_usd": pool["volume_usd"],
                "market_cap_usd": _as_float(pool["market_cap_usd"]) or _as_float(pool["fdv_usd"]),
            })

            entry_row = register_pending_checkpoint(pending_checkpoints, pool, timestamp)
            if entry_row is not None:
                label_rows_out.append(entry_row)

            last_alert = alerted.get(pool["id"])
            if last_alert:
                try:
                    since = datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(last_alert).timestamp()
                except ValueError:
                    since = ALERT_COOLDOWN_SECONDS + 1
                if since < ALERT_COOLDOWN_SECONDS:
                    continue

            send_telegram(format_alert(pool, features, security))
            alerted[pool["id"]] = timestamp

    label_rows_out.extend(process_due_checkpoints(pending_checkpoints, datetime.now(timezone.utc), checkpoint_lookups_this_tick))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    hot_state = engine.load_json(HOT_STATE_PATH, {})
    security_cache = engine.load_json(SECURITY_CACHE_PATH, {})
    alerted = engine.load_json(ALERTED_PATH, {})
    pending_checkpoints = engine.load_json(PENDING_CHECKPOINTS_PATH, {})

    jsonl_buffer = []
    candidates_out = []
    label_rows = []

    loop_start = time.monotonic()
    ticks = 0
    while True:
        run_tick(hot_state, security_cache, alerted, pending_checkpoints, jsonl_buffer, candidates_out, label_rows)
        ticks += 1
        if time.monotonic() - loop_start >= LOOP_DURATION_SECONDS:
            break
        time.sleep(TICK_INTERVAL_SECONDS)

    prune_hot_state(hot_state, datetime.now(timezone.utc))

    # Overwritten fresh each run (not appended) - the workflow uploads this
    # as a per-run artifact instead of committing an ever-growing file to git.
    with open(RUN_SNAPSHOT_PATH, "w") as f:
        for line in jsonl_buffer:
            f.write(line + "\n")

    # Append-only forever - this is the small, permanent training-data file.
    with open(LABELS_JSONL_PATH, "a") as f:
        for row in label_rows:
            f.write(json.dumps(row) + "\n")

    engine.save_json(HOT_STATE_PATH, hot_state)
    engine.save_json(SECURITY_CACHE_PATH, security_cache)
    engine.save_json(ALERTED_PATH, alerted)
    engine.save_json(PENDING_CHECKPOINTS_PATH, pending_checkpoints)
    engine.save_json(CANDIDATES_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidates": candidates_out,
    })

    print(f"Completed {ticks} ticks over ~{int(time.monotonic() - loop_start)}s. "
          f"{len(candidates_out)} candidate observation(s) passed both filters, "
          f"{len(label_rows)} label row(s) written, {len(pending_checkpoints)} token(s) still awaiting outcome checkpoints.")


if __name__ == "__main__":
    main()
