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

Runs as a short-lived single-tick script, not a long-running daemon: each
invocation takes one snapshot across all networks, then commits once and
exits. It used to loop internally for ~4 minutes to fake density, back when
this ran on GitHub Actions' own `schedule:` trigger - which turned out to
be unreliable on this account (confirmed empirically: runs meant to be 5
minutes apart landed 2-4 HOURS apart). The fix was to stop relying on
`schedule:` at all: an external cron service (cron-job.org) now calls this
workflow's `workflow_dispatch` API every 5 minutes instead, which GitHub
does NOT throttle the way it throttles `schedule:`. With a genuinely
reliable external trigger, internal looping is redundant - it was only
ever a workaround for GitHub's unreliable scheduler, and looping for 4
minutes across 6 networks risked runs overlapping the next 5-minute
trigger and queueing up behind each other (concurrency group serializes,
doesn't run in parallel). One fast tick per invocation now IS the polling
density, at whatever cadence the external trigger actually delivers.
"""
import json
import os
import random
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
PORTFOLIO_PATH = os.path.join(DATA_DIR, "memecoin_portfolio.json")

NETWORKS = ["solana", "base", "bsc", "eth", "arbitrum", "polygon_pos"]
GECKOTERMINAL_TRENDING_URL = "https://api.geckoterminal.com/api/v2/networks/{network}/trending_pools"
GECKOTERMINAL_POOL_URL = "https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}"
RUGCHECK_REPORT_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"
GOPLUS_TOKEN_SECURITY_URL = "https://api.gopluslabs.io/api/v1/token_security/{chain_id}?contract_addresses={address}"
EVM_CHAIN_IDS = {"base": "8453", "bsc": "56", "eth": "1", "arbitrum": "42161", "polygon_pos": "137"}

# --- Timing ------------------------------------------------------------
# One tick per invocation now (see module docstring for why) - an external
# cron service triggers a fresh invocation every 5 minutes, so that IS the
# polling cadence. LOOP_DURATION_SECONDS just needs to be smaller than one
# tick's own runtime so the while-loop in main() never attempts a second
# tick; TICK_INTERVAL_SECONDS is dead code left in place only in case
# someone wants multi-tick behavior back for local testing.
TICK_INTERVAL_SECONDS = 30
LOOP_DURATION_SECONDS = 1

# --- Momentum pre-filter (cheap, runs every tick on every trending pool) -
# One "tick" = one invocation now (~5 min apart, set by the external cron -
# see Timing section above), not a sub-loop within a single run.
MIN_MARKET_CAP_USD = 50_000
HOT_STATE_WINDOW = 20          # keep last N ticks per pool (~100 min at ~5 min/tick)
HOT_STATE_MAX_AGE_SECONDS = 7200  # drop a pool from hot state if unseen for 2h
MIN_HISTORY_FOR_VELOCITY = 3   # need this many prior ticks (~15 min) before trusting
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

# --- Play-money paper portfolio ("mänguraha") -----------------------------
# Buys only the tightest tier - candidates that clear the same "recommended"
# bar the dashboard shows by default (>=2 good tags, 0 caution tags), not
# every base-filter candidate. Exits at the 6h checkpoint, reusing that
# price fetch instead of costing extra API calls. Purely play money - this
# never touches broker.py/engine.py or anything resembling real trading.
STARTING_BALANCE_USD = 1000.0
POSITION_SIZE_FRACTION = 0.20  # 20% of current balance per buy
MIN_TRADE_USD = 20.0            # below this, a trade is too small to matter even if it triples - skip it
MAX_CONCURRENT_POSITIONS = 6    # once full, hold off buying until a position closes and frees a slot -
                                 # without this, 20%-of-current-balance sizing shrinks every subsequent
                                 # buy geometrically (found in production: a real run reached $10 trades)
EXIT_OFFSET_MINUTES = 360  # sell at the 6h checkpoint
GOOD_CONCENTRATION_LOW = {"solana": 20.0, "_evm": 5.0}
GOOD_CONCENTRATION_HIGH = {"solana": 30.0, "_evm": 10.0}
RECOMMENDED_MIN_GOOD = 2

# Smart exit: open positions are re-checked once per run (not every tick -
# conserves API calls) against a stop-loss, a trailing-stop from the peak
# price seen, and a momentum-collapse signal. The fixed 6h checkpoint exit
# still applies as a fallback if none of these fire first. Starting
# thresholds, not tuned against outcome data yet - same caveat as the rest
# of this file's constants.
STOP_LOSS_PCT = 30.0            # exit if price is down this much from entry
TRAILING_STOP_PCT = 25.0        # exit if price pulls back this much from its peak (only once ever in profit)
MOMENTUM_COLLAPSE_H1_PCT = -15.0  # exit if 1h change turns this negative AND sell pressure exceeds buy pressure

# Averaging down: only on a moderate dip, well clear of the stop-loss zone,
# and only if the token still looks healthy right now - not just cheaper.
# Capped at one add per position, sized smaller than the original buy.
DCA_TRIGGER_MIN_PCT = -20.0      # must be down at least this much from entry...
DCA_TRIGGER_MAX_PCT = -10.0      # ...but not more than this (stay clear of the -30% stop-loss)
DCA_SIZE_FRACTION_OF_ORIGINAL = 0.5  # the add is half the size of the original position
DCA_MAX_ADDS = 1

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
    """Single-pool lookup, used at checkpoint/exit-check time (not every tick)
    to get this specific token's current price/momentum without re-fetching
    the whole trending list. Also returns price_change_pct/transactions (same
    response, no extra cost) so exit logic can see momentum, not just price."""
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
        "price_change_pct": attrs.get("price_change_percentage", {}) or {},
        "transactions": attrs.get("transactions", {}) or {},
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


# --- Recommendation tier + play-money portfolio ----------------------------
# Mirrors the dashboard's client-side getTags() logic exactly (same thresholds,
# same "good/caution" counts) so the coins the bot "buys" are exactly the ones
# the dashboard shows as today's recommendations - one definition, two places
# it's read, kept in sync deliberately rather than reimplemented differently.

def classify_recommendation(pool, features, security):
    good = 0
    caution = 0
    rv = features.get("rank_velocity")
    accel = features.get("h1_accel")
    h1 = _as_float(pool["price_change_pct"].get("h1"))
    h24 = _as_float(pool["price_change_pct"].get("h24"))
    mcap = _as_float(pool["market_cap_usd"]) or _as_float(pool["fdv_usd"])

    key = "solana" if pool["network"] == "solana" else "_evm"
    concentration = security.get("top10_pct") if pool["network"] == "solana" else security.get("owner_creator_pct")
    conc_low = GOOD_CONCENTRATION_LOW[key]
    conc_high = GOOD_CONCENTRATION_HIGH[key]

    if rv is not None and rv > 0:
        good += 1
    if accel is not None and accel > 0:
        good += 1
    if h24 >= 20:
        good += 1
    if concentration is not None and concentration < conc_low:
        good += 1
    if mcap >= 1_000_000:
        good += 1

    if concentration is not None and concentration > conc_high:
        caution += 1
    if 0 < mcap < 200_000:
        caution += 1
    if h24 < 0 and h1 > 0:
        caution += 1
    if rv is not None and rv == 0:
        caution += 1

    return good >= RECOMMENDED_MIN_GOOD and caution == 0


def default_portfolio():
    return {"balance": STARTING_BALANCE_USD, "positions": {}, "closed": []}


def maybe_buy(portfolio, pool, timestamp):
    pool_id = pool["id"]
    if pool_id in portfolio["positions"]:
        return None
    if len(portfolio["positions"]) >= MAX_CONCURRENT_POSITIONS:
        return None  # full - wait for a position to close before opening another
    price = _as_float(pool.get("price_usd"))
    if price <= 0:
        return None
    amount = portfolio["balance"] * POSITION_SIZE_FRACTION
    if amount < MIN_TRADE_USD:
        return None
    portfolio["balance"] -= amount
    position = {
        "network": pool["network"],
        "pool_address": pool["address"],
        "base_token_address": pool["base_token_address"],
        "name": pool["name"],
        "entry_ts": timestamp,
        "entry_price_usd": price,
        "peak_price_usd": price,
        "amount_usd": amount,
        "qty": amount / price,
    }
    portfolio["positions"][pool_id] = position
    return position


def maybe_sell(portfolio, pool_id, price_now, timestamp, reason="timeout"):
    pos = portfolio["positions"].pop(pool_id, None)
    if pos is None:
        return None
    proceeds = pos["qty"] * price_now if price_now and price_now > 0 else 0.0
    pnl_usd = proceeds - pos["amount_usd"]
    pnl_pct = (pnl_usd / pos["amount_usd"] * 100.0) if pos["amount_usd"] else 0.0
    portfolio["balance"] += proceeds
    closed = {
        **pos,
        "exit_ts": timestamp,
        "exit_price_usd": price_now,
        "proceeds_usd": proceeds,
        "pnl_usd": pnl_usd,
        "pnl_pct": round(pnl_pct, 2),
        "exit_reason": reason,
    }
    portfolio["closed"].append(closed)
    return closed


def maybe_average_down(portfolio, pos, current, security_cache, checks_this_tick, timestamp):
    """Adds to an existing position on a moderate dip - but only if the token
    still looks healthy right now (still-positive h1/h6 momentum, security
    still passes), not just "it got cheaper." See DCA_* constants for the
    exact band/sizing/cap. Mutates pos in place; returns a summary dict or
    None if nothing fired."""
    if pos.get("dca_count", 0) >= DCA_MAX_ADDS:
        return None
    price = _as_float(current.get("price_usd"))
    if price <= 0:
        return None
    change_pct = (price - pos["entry_price_usd"]) / pos["entry_price_usd"] * 100.0
    if not (DCA_TRIGGER_MIN_PCT <= change_pct <= DCA_TRIGGER_MAX_PCT):
        return None

    h1 = _as_float(current.get("price_change_pct", {}).get("h1"))
    h6 = _as_float(current.get("price_change_pct", {}).get("h6"))
    if not (h1 > 0 and h6 > 0):
        return None  # dipped, but not "still strong" - just cheaper

    pseudo_pool = {"network": pos["network"], "base_token_address": pos["base_token_address"]}
    security = check_security_cached(pseudo_pool, security_cache, checks_this_tick)
    if security is None or not security["passed"]:
        return None

    add_amount = pos["amount_usd"] * DCA_SIZE_FRACTION_OF_ORIGINAL
    if add_amount < MIN_TRADE_USD or add_amount > portfolio["balance"]:
        return None

    portfolio["balance"] -= add_amount
    pos["amount_usd"] += add_amount
    pos["qty"] += add_amount / price
    pos["dca_count"] = pos.get("dca_count", 0) + 1
    # cost basis is now the blended average - stop-loss/trailing-stop from
    # here on are measured against this, same as a real average-down would be
    pos["entry_price_usd"] = pos["amount_usd"] / pos["qty"]
    return {"name": pos["name"], "network": pos["network"], "add_amount": add_amount,
            "price": price, "new_entry_price": pos["entry_price_usd"], "timestamp": timestamp}


def check_open_positions(portfolio, security_cache, checks_this_tick, now_wall):
    """Smart exit + DCA: re-checks every open position once per run (not every
    tick - each check costs a GeckoTerminal call, and open-position count
    isn't bounded the way checkpoint lookups are). Priority order: a hard
    stop-loss/trailing-stop/momentum-collapse exit always wins over "should
    we average down" - never add to a position in the same breath as cutting
    it. The fixed 6h checkpoint exit (process_due_checkpoints) still applies
    underneath as a fallback if none of the exit conditions fire first."""
    closed_positions = []
    dca_events = []
    for i, (pool_id, pos) in enumerate(list(portfolio["positions"].items())):
        if i > 0:
            time.sleep(1.5)  # shares GeckoTerminal's rate limit with everything else
        current = fetch_pool_price(pos["network"], pos["pool_address"])
        if current is None:
            continue
        price = _as_float(current.get("price_usd"))
        if price <= 0:
            continue

        pos["peak_price_usd"] = max(pos.get("peak_price_usd", pos["entry_price_usd"]), price)
        was_in_profit = pos["peak_price_usd"] > pos["entry_price_usd"]
        loss_pct = (price - pos["entry_price_usd"]) / pos["entry_price_usd"] * 100.0
        drawdown_from_peak_pct = (price - pos["peak_price_usd"]) / pos["peak_price_usd"] * 100.0

        h1 = _as_float(current.get("price_change_pct", {}).get("h1"))
        txns_h1 = current.get("transactions", {}).get("h1", {}) or {}
        sell_pressure = _as_float(txns_h1.get("sells")) > _as_float(txns_h1.get("buys"))

        reason = None
        if loss_pct <= -STOP_LOSS_PCT:
            reason = "stop_loss"
        elif was_in_profit and drawdown_from_peak_pct <= -TRAILING_STOP_PCT:
            reason = "trailing_stop"
        elif h1 <= MOMENTUM_COLLAPSE_H1_PCT and sell_pressure:
            reason = "momentum_collapse"

        if reason is not None:
            closed = maybe_sell(portfolio, pool_id, price, now_wall.isoformat(), reason=reason)
            if closed is not None:
                closed_positions.append(closed)
            continue

        dca = maybe_average_down(portfolio, pos, current, security_cache, checks_this_tick, now_wall.isoformat())
        if dca is not None:
            dca_events.append(dca)
    return closed_positions, dca_events


# --- Outcome checkpoints / labels table -----------------------------------

def register_pending_checkpoint(pending, pool, timestamp, features=None, security=None):
    """Called the first time a pool qualifies (passes momentum + security).
    Writes the T0 entry row straight to the labels buffer, and schedules
    future checkpoint lookups (10m/30m/1h/3h/6h) so we eventually learn what
    actually happened after entry - not just that entry looked good.

    Persists features/security onto the entry row (not just price/mcap) so
    analyze_labels.py can actually test whether rank_velocity, h1_accel, and
    holder concentration predict outcomes - not just guess from the name."""
    pool_id = pool["id"]
    if pool_id in pending:
        return None
    entry_price = _as_float(pool.get("price_usd"))
    entry_mcap = _as_float(pool["market_cap_usd"]) or _as_float(pool["fdv_usd"])
    features = features or {}
    security = security or {}
    concentration = security.get("top10_pct") if pool["network"] == "solana" else security.get("owner_creator_pct")
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
        "rank_velocity": features.get("rank_velocity"),
        "h1_accel": features.get("h1_accel"),
        "concentration_pct": concentration,
    }


def process_due_checkpoints(pending, now_wall, lookups_this_tick, portfolio):
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

            if offset == EXIT_OFFSET_MINUTES:
                closed = maybe_sell(portfolio, pool_id, price_now, checkpoint_ts)
                if closed is not None:
                    send_telegram(format_sell_alert(closed, portfolio["balance"]))

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
# Playful on purpose (the user asked for it) - the numbers underneath are
# still exact and unembellished, only the framing has personality.

ALERT_INTROS = [
    "🚨 Radar piiksus!",
    "👀 Silm jäi millelegi pidama.",
    "🐸 Uus tulija areenile!",
    "🔍 Nuusutasime ringi ja leidsime midagi.",
    "📈 Trendib nagu hull.",
    "🎯 Bot nägi midagi ja ei jäänud ükskõikseks.",
]
BUY_INTROS = [
    "🛒 Bot ostis!",
    "💰 Portfell just kasvas.",
    "🤝 Käsi sügeles, panus tehtud.",
    "🎮 Mänguraha liigub!",
    "💸 CHA-CHING, ostuots sooritatud.",
    "🚀 Pardale astutud, lootuses to-the-moon'i.",
]
SELL_WIN_INTROS = [
    "🎉 CHA-CHING! Kasumiga väljas.",
    "🥳 Selline nädal võiks iga kord olla.",
    "💎🙌 Kassa kõliseb!",
    "🏆 Bot müüs plussiga, respekt.",
    "🌕 WAGMI — kasum kotti!",
]
SELL_LOSS_INTROS = [
    "😅 Noh, ei läinud plaanipäraselt.",
    "🩹 See oli valus, aga mänguraha ju.",
    "⚰️ RIP see trade, järgmine tuleb parem.",
    "📚 Jälle üks õppetund kirja saanud.",
    "🫠 NGMI see kord, aga oleme siin lõbu pärast.",
]
DCA_INTROS = [
    "📉➕ Dip osteti juurde.",
    "🛍️ Bot lisas positsiooni - hind meeldis veel rohkem.",
    "🧮 Keskmine sisenemishind just paranes.",
    "🎯 Ikka tugev, lihtsalt odavam. Lisasime.",
]


def format_alert(pool, features, security):
    h1 = _as_float(pool["price_change_pct"].get("h1"))
    h24 = _as_float(pool["price_change_pct"].get("h24"))
    vol_h24 = _as_float(pool["volume_usd"].get("h24"))
    mcap = _as_float(pool["market_cap_usd"]) or _as_float(pool["fdv_usd"])
    lines = [
        f"[MEMECOIN SCAN] {random.choice(ALERT_INTROS)}",
        f"{pool['name']} ({pool['network']})",
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


def format_buy_alert(position, balance_after):
    return "\n".join([
        f"[MÄNGURAHA] {random.choice(BUY_INTROS)}",
        f"{position['name']} ({position['network']})",
        f"Ostsime ${position['amount_usd']:.2f} eest hinnaga ${position['entry_price_usd']:.8f}",
        f"Ülejäänud saldo: ${balance_after:.2f}",
    ])


def format_dca_alert(dca, balance_after):
    return "\n".join([
        f"[MÄNGURAHA] {random.choice(DCA_INTROS)}",
        f"{dca['name']} ({dca['network']})",
        f"Lisasime ${dca['add_amount']:.2f} hinnaga ${dca['price']:.8f}",
        f"Uus keskmine sisenemishind: ${dca['new_entry_price']:.8f}",
        f"Ülejäänud saldo: ${balance_after:.2f}",
    ])


EXIT_REASON_LABELS = {
    "stop_loss": "stop-loss vallandus",
    "trailing_stop": "trailing-stop lukustas kasumi",
    "momentum_collapse": "hoog kadus, väljusime",
    "timeout": "6h hoidmise piir täis",
}


def format_sell_alert(closed, balance_after):
    win = closed["pnl_usd"] >= 0
    intro = random.choice(SELL_WIN_INTROS if win else SELL_LOSS_INTROS)
    sign = "+" if win else ""
    reason_label = EXIT_REASON_LABELS.get(closed.get("exit_reason"), closed.get("exit_reason", ""))
    return "\n".join([
        f"[MÄNGURAHA] {intro}",
        f"{closed['name']} ({closed['network']})",
        f"Tulemus: {sign}${closed['pnl_usd']:.2f} ({sign}{closed['pnl_pct']:.1f}%)",
        f"Põhjus: {reason_label}",
        f"Uus saldo: ${balance_after:.2f}",
    ])


# --- Main loop ---------------------------------------------------------------

def run_tick(hot_state, security_cache, alerted, pending_checkpoints, portfolio, jsonl_buffer, candidates_out, label_rows_out):
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
                "id": pool["id"],  # joins against memecoin_hot_state.json for the dashboard's sparklines
                "timestamp": timestamp, "network": network, "name": pool["name"],
                "address": pool["address"], "base_token_address": pool["base_token_address"],
                "rank": pool["rank"], "features": features, "security": security,
                # display fields for the dashboard chart - not used by filter logic itself
                "price_change_pct": pool["price_change_pct"],
                "volume_usd": pool["volume_usd"],
                "market_cap_usd": _as_float(pool["market_cap_usd"]) or _as_float(pool["fdv_usd"]),
            })

            entry_row = register_pending_checkpoint(pending_checkpoints, pool, timestamp, features=features, security=security)
            if entry_row is not None:
                label_rows_out.append(entry_row)

            # Checked every tick a pool is still a candidate, NOT gated on
            # entry_row (which only fires once, the first time this pool_id
            # is ever seen). A pool that didn't qualify on first sight but
            # improves later (rank climbs, momentum accelerates) must still
            # get a buy chance - maybe_buy's own "already holding" guard is
            # what prevents buying the same open position twice.
            if classify_recommendation(pool, features, security):
                position = maybe_buy(portfolio, pool, timestamp)
                if position is not None:
                    send_telegram(format_buy_alert(position, portfolio["balance"]))

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

    label_rows_out.extend(process_due_checkpoints(pending_checkpoints, datetime.now(timezone.utc), checkpoint_lookups_this_tick, portfolio))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    hot_state = engine.load_json(HOT_STATE_PATH, {})
    security_cache = engine.load_json(SECURITY_CACHE_PATH, {})
    alerted = engine.load_json(ALERTED_PATH, {})
    pending_checkpoints = engine.load_json(PENDING_CHECKPOINTS_PATH, {})
    portfolio = engine.load_json(PORTFOLIO_PATH, default_portfolio())

    jsonl_buffer = []
    candidates_out = []
    label_rows = []

    loop_start = time.monotonic()
    ticks = 0
    while True:
        run_tick(hot_state, security_cache, alerted, pending_checkpoints, portfolio, jsonl_buffer, candidates_out, label_rows)
        ticks += 1
        if time.monotonic() - loop_start >= LOOP_DURATION_SECONDS:
            break
        time.sleep(TICK_INTERVAL_SECONDS)

    prune_hot_state(hot_state, datetime.now(timezone.utc))

    exit_checks_this_tick = [0]
    closed_early, dca_events = check_open_positions(portfolio, security_cache, exit_checks_this_tick, datetime.now(timezone.utc))
    for closed in closed_early:
        send_telegram(format_sell_alert(closed, portfolio["balance"]))
    for dca in dca_events:
        send_telegram(format_dca_alert(dca, portfolio["balance"]))

    # Overwritten fresh each run (not appended) - the workflow uploads this
    # as a per-run artifact instead of committing an ever-growing file to git.
    with open(RUN_SNAPSHOT_PATH, "w") as f:
        for line in jsonl_buffer:
            f.write(line + "\n")

    # Append-only forever - this is the small, permanent training-data file.
    with open(LABELS_JSONL_PATH, "a") as f:
        for row in label_rows:
            f.write(json.dumps(row) + "\n")

    # A pool that's still a candidate several ticks in a row gets appended to
    # candidates_out once per tick - dedupe to the latest observation per
    # pool_id so the dashboard doesn't show the same token twice (it did:
    # BNBCAT/WKC each showed up as two separate "today's recommendations"
    # slots, wasting the RECOMMENDED_CAP on itself).
    deduped_candidates = list({c["id"]: c for c in candidates_out}.values())

    engine.save_json(HOT_STATE_PATH, hot_state)
    engine.save_json(SECURITY_CACHE_PATH, security_cache)
    engine.save_json(ALERTED_PATH, alerted)
    engine.save_json(PENDING_CHECKPOINTS_PATH, pending_checkpoints)
    engine.save_json(PORTFOLIO_PATH, portfolio)
    engine.save_json(CANDIDATES_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidates": deduped_candidates,
    })

    print(f"Completed {ticks} ticks over ~{int(time.monotonic() - loop_start)}s. "
          f"{len(candidates_out)} candidate observation(s) passed both filters, "
          f"{len(label_rows)} label row(s) written, {len(pending_checkpoints)} token(s) still awaiting outcome checkpoints. "
          f"Portfolio balance: ${portfolio['balance']:.2f}, {len(portfolio['positions'])} open position(s), {len(portfolio['closed'])} closed trade(s).")


if __name__ == "__main__":
    main()
