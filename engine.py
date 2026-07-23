#!/usr/bin/env python3
"""
Cryptobot scoring engine - v2 (self-learning).

Two-stage pipeline, run once per hour by the scheduled task:

  1. `screen`   - input: freshly fetched tickers (from Crypto.com) for the
                  watchlist. Computes a momentum+trend score per token,
                  applies noise filtering (needs history to confirm a trend),
                  and outputs the top candidates worth a closer look. Also
                  does small epsilon-greedy "exploration": occasionally lets
                  a slightly-below-threshold token through anyway, tagged,
                  purely so the model keeps learning near the boundary
                  instead of only ever confirming its own priors.

  2. `finalize` - input: the candidates from step 1 plus short hype notes
                  gathered via WebSearch, plus current prices for any
                  recommendations due for a 24h/7d follow-up check. Computes
                  a final 0-100 score that blends the hand-written heuristic
                  with a small online-learned logistic-regression model (pure
                  Python, no dependencies), applies cooldown/dedup, logs new
                  recommendations (storing the exact feature vector used, so
                  it can be trained on later), resolves due follow-ups,
                  trains the model one step per resolved outcome, nudges the
                  adaptive risk thresholds, computes feature/outcome
                  correlations, (re)writes dashboard.html, and prints the
                  chat summary (the notification).

The self-learning part: every scored recommendation stores the feature
vector that produced it. When we later find out whether it actually worked
(24h/7d price check), we do one step of online logistic-regression training
(gradient ascent on log-likelihood) so the model's weights shift toward
whatever actually predicted success - not just what we assumed would.
Progress is logged in plain language in state["model"]["learning_log"] so
it's visible on the dashboard, not just a black box.

The math behind "how the money is actually made" (v3):
  - Momentum is volatility-normalized (Sharpe-style): a % move is scored
    relative to the instrument's OWN recent noise level, not in isolation,
    via the square-root-of-time rule (hourly stdev -> expected 24h stdev).
    A 5% day is a strong signal on a calm major and a shrug on a meme coin
    that does that every afternoon.
  - Trend confirmation uses closed-form OLS regression (slope + R^2) over
    the recent price/volume history instead of a brittle "every hour must
    be higher than the last" check, so one noisy tick no longer erases an
    otherwise real trend.
  - A simple single-factor (CAPM-style) alpha term compares each token's
    move to BTC's move in the same run: r_token = beta*r_BTC + alpha (beta
    fixed at 1 for simplicity/transparency). Most altcoins are just
    leveraged BTC-beta in disguise; this rewards genuine idiosyncratic
    strength over "it went up because everything went up."
  - The heuristic/model blend uses actuarial credibility weighting
    (Z = n / (n + k)) instead of a hard on/off switch at a training-count
    threshold, so the model's influence grows smoothly with evidence.
  - Virtual position sizing uses a fractional Kelly criterion
    (f* = p - (1-p)/b, quarter-Kelly, clamped) once the model has earned
    enough evidence to be trusted with sizing - bigger, well-founded edges
    get bigger (but bounded) bets instead of everything getting the same
    flat 5%.

All state lives in data/state.json (persisted in the user's project folder,
NOT the ephemeral session scratchpad) so history/learning survives across
hourly runs.
"""
import json
import math
import os
import random
import sys
import time
import argparse
from datetime import datetime, timezone

import broker as broker_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
WATCHLIST_PATH = os.path.join(BASE_DIR, "watchlist.json")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")  # same content - lets GitHub Pages serve a clean root URL

HISTORY_KEEP = 48  # keep last 48 hourly snapshots per instrument (~2 days)
VOL_WINDOW = 24     # snapshots used for the rolling volatility estimate (~1 day at hourly cadence)
TREND_WINDOW = 6    # snapshots used for the OLS trend fit (~6h at hourly cadence)

# Feature order matters - it's the order weights/x vectors are stored in.
FEATURE_NAMES = ["bias", "momentum", "trend_bonus", "liquidity", "is_meme", "volatility", "hype_bonus", "alpha"]

# Initial weights: a reasonable hand-set prior (mirrors the original heuristic
# direction) that then gets refined by real evidence via train_step().
INITIAL_WEIGHTS = {
    "bias": -0.4,
    "momentum": 1.6,
    "trend_bonus": 1.1,
    "liquidity": 1.0,
    "is_meme": -0.7,
    "volatility": -0.8,
    "hype_bonus": 1.2,
    "alpha": 0.9,   # prior: genuine outperformance vs. BTC predicts follow-through better than raw noise
}

MODEL_MIN_TRAINING = 15   # need this many resolved outcomes before the model is trusted for POSITION SIZING (Kelly)
CREDIBILITY_K = 25        # actuarial credibility constant for SCORE BLENDING: Z = n/(n+k), grows smoothly
                          # instead of jumping from 0% to 50% model-influence at one exact data point
CREDIBILITY_CAP = 0.65    # even with unlimited data, the hand-written heuristic keeps >=35% say -
                          # it encodes real domain priors (liquidity, scam keywords) the model may never see enough of
LEARNING_RATE = 0.08
L2_LAMBDA = 0.015         # small ridge penalty - shrinks weights that aren't earning their keep back toward 0,
                          # so noise features don't drift just from finite-sample luck
EXPLORE_EPSILON = 0.15    # chance to let a near-miss candidate through anyway, for learning
EXPLORE_MARGIN = 10       # how far below the screen threshold still counts as "near"

# Virtual paper-trading portfolio - play money only, never touches anything real.
# Rule: when an alert fires (and there's room + cash), "buy" a position sized as
# a % of current balance (flat, or Kelly-sized once the model has earned trust -
# see kelly_position_pct); automatically "sell" it at the 24h mark (same moment
# the 24h follow-up outcome is already being checked), realizing whatever the
# actual price move was. This is a simple, transparent day-trade-style
# simulation - not a recommendation for how to actually trade.
PORTFOLIO_START_BALANCE = 1000.0
PORTFOLIO_POSITION_PCT = 0.05   # 5% of current balance - the flat fallback before Kelly sizing kicks in
PORTFOLIO_MAX_OPEN = 5          # at most 5 positions open at once (~25% max exposure at flat sizing)
PORTFOLIO_CATEGORY_CAP = {"meme": 2}   # never more than 2/5 open slots in high-variance meme names at
                                        # once, however good they score - basic diversification so one
                                        # hype wave rolling over can't wipe the whole simulated book

# Fractional Kelly criterion for position sizing: f* = p - (1-p)/b (p = model's
# win probability for this trade, b = historical avg-win/avg-loss payoff ratio).
# Using a QUARTER of full Kelly is a standard practitioner compromise - full
# Kelly maximizes long-run growth rate in theory, but its variance is brutal
# (it will happily size a position at 40%+ of the bankroll on a good edge);
# quarter-Kelly keeps roughly 3/4 of the growth rate with a fraction of the
# drawdown risk.
KELLY_FRACTION = 0.25
KELLY_MIN_PCT = 0.01   # never below 1% - keep participating/learning even on a marginal edge
KELLY_MAX_PCT = 0.15   # never above 15% on one idea, however confident the model looks

# "5 parimat, mitte 5 esimest": kui raamat on täis, võib uus, SELGELT tugevam
# kandidaat välja vahetada nõrgima MIINUSES oleva positsiooni. Hüsterees on
# teadlik fee-kaitse: üks vahetus maksab garanteeritult ~2x(fee+slippage)
# (~1.3% vaikeseadetega - müüd ühe ja ostad teise), nii et vahetame ainult
# siis, kui (a) senine positsioon on ise juba vee all (>=1% miinuses ehk
# "nõrgenenud", mitte lihtsalt aeglasem tõusja), (b) uus kandidaat on skoorilt
# selgelt parem (>= +12 punkti - rohkem kui tavaline tunnisisene müra) ja
# (c) positsiooni on hoitud vähemalt paar tundi (ei pinguta edasi-tagasi sama
# tunni sees). Max 1 vahetus tunnis, et churn ei sööks kasumit ära.
SWAP_MIN_SCORE_ADVANTAGE = 12
SWAP_LOSER_MAX_RETURN_PCT = -1.0   # holding must be at least this far under water
SWAP_MIN_HOLD_HOURS = 2
SWAP_MAX_PER_RUN = 1

DEFAULT_STATE = {
    "history": {},          # instrument -> [ {ts, price, change_24h, volume_value}, ... ]
    "alerted": {},          # instrument -> {last_score, last_ts, last_risk}
    "thresholds": {         # adaptive, per risk bucket, tuned by adapt_thresholds()
        "screen_score": 65,
        "green_hit_bar": 60,
        "yellow_hit_bar": 65,
        "red_hit_bar": 75
    },
    "adjustment_checkpoint": 0,  # len(completed) at last threshold adjustment - prevents re-applying
                                 # the same historical evidence over and over on every run (see adapt_thresholds)
    "model": {
        "weights": dict(INITIAL_WEIGHTS),
        "n_updates": 0,
        "learning_log": []   # [{ts, text}], plain-language "what I just learned"
    },
    "pending_followups": [],   # awaiting 24h and/or 7d outcome check
    "completed": [],          # followups fully resolved (both checks done or expired)
    "run_log": [],            # short history of each run (for the dashboard)
    "next_id": 1,
    "portfolio": {
        "starting_balance": PORTFOLIO_START_BALANCE,
        "balance": PORTFOLIO_START_BALANCE,
        "position_size_pct": PORTFOLIO_POSITION_PCT,
        "max_open_positions": PORTFOLIO_MAX_OPEN,
        "mode": "paper",         # which broker mode the last run used (paper/live)
        "open_positions": [],    # [{id, instrument, entry_price, quantity, size_usd, entry_fee_usd, entry_ts, risk, score, mode, stop_price, stop_order_id}]
        "closed_trades": [],     # [{...same + exit_price, exit_ts, pnl_usd, pnl_pct, exit_fee_usd, exit_reason}]
        "balance_history": [{"ts": 0, "balance": PORTFOLIO_START_BALANCE}],
    },
    "killswitch": {
        # Safety brake: when active, the bot stops OPENING new positions
        # (existing ones are still managed and closed normally) until reset.
        # Reset: run once with env KILLSWITCH_RESET=1, or edit this block.
        "active": False,
        "reason": "",
        "ts": 0,
        "consec_failures": 0   # consecutive failed live orders
    }
}


def now_ts():
    return datetime.now(timezone.utc).timestamp()


def load_json(path, default):
    if not os.path.exists(path):
        return json.loads(json.dumps(default))
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _deep_default(v):
    return json.loads(json.dumps(v))


def load_state():
    st = load_json(STATE_PATH, DEFAULT_STATE)
    for k, v in DEFAULT_STATE.items():
        st.setdefault(k, _deep_default(v))
    # in case an older state.json exists without the model block
    st["model"].setdefault("weights", dict(INITIAL_WEIGHTS))
    for fn in FEATURE_NAMES:
        st["model"]["weights"].setdefault(fn, INITIAL_WEIGHTS[fn])
    st["model"].setdefault("n_updates", 0)
    st["model"].setdefault("learning_log", [])
    st.setdefault("portfolio", _deep_default(DEFAULT_STATE["portfolio"]))
    for k, v in DEFAULT_STATE["portfolio"].items():
        st["portfolio"].setdefault(k, v)
    st.setdefault("killswitch", _deep_default(DEFAULT_STATE["killswitch"]))
    for k, v in DEFAULT_STATE["killswitch"].items():
        st["killswitch"].setdefault(k, v)
    return st


def save_state(state):
    save_json(STATE_PATH, state)


def load_watchlist():
    wl = load_json(WATCHLIST_PATH, {"majors": [], "meme_trend": []})
    cat = {}
    for sym in wl.get("majors", []):
        cat[sym] = "major"
    for sym in wl.get("meme_trend", []):
        cat[sym] = "meme"
    return cat


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Online logistic-regression model (pure Python, no dependencies)
# ---------------------------------------------------------------------------

def sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def build_features(momentum_score_, trend_bonus_, liquidity, category, change_24h, hype_bonus, alpha_val=0.0):
    liq_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
    return {
        "bias": 1.0,
        "momentum": momentum_score_ / 100.0,
        "trend_bonus": clamp((trend_bonus_ + 8) / 28.0, 0, 1),
        "liquidity": liq_map.get(liquidity, 0.0),
        "is_meme": 1.0 if category in ("meme", "unknown") else 0.0,
        "volatility": clamp(abs(change_24h) / 0.30, 0, 1),
        "hype_bonus": clamp((hype_bonus + 30) / 60.0, 0, 1),
        # alpha_val is a fraction (0.03 = 3pp outperformance vs BTC); -10pp..+15pp -> 0..1
        "alpha": clamp((alpha_val * 100 + 10) / 25.0, 0, 1),
    }


def model_predict(weights, features):
    z = sum(weights.get(k, 0.0) * features.get(k, 0.0) for k in FEATURE_NAMES)
    return sigmoid(z)


def train_step(state, features, hit):
    """One step of online logistic-regression training (gradient ascent on
    log-likelihood). Logs a plain-language note if weights moved meaningfully."""
    weights = state["model"]["weights"]
    old_weights = dict(weights)
    p = model_predict(weights, features)
    y = 1.0 if hit else 0.0
    error = y - p
    for k in FEATURE_NAMES:
        reg = 0.0 if k == "bias" else L2_LAMBDA * weights[k]  # don't regularize the bias term
        weights[k] = weights[k] + LEARNING_RATE * (error * features.get(k, 0.0) - reg)
    state["model"]["n_updates"] += 1

    moved = {k: weights[k] - old_weights[k] for k in FEATURE_NAMES if abs(weights[k] - old_weights[k]) >= 0.02}
    if moved:
        biggest = max(moved, key=lambda k: abs(moved[k]))
        direction = "tugevam" if moved[biggest] > 0 else "nõrgem"
        note = (f"Tulemus ({'tabas' if hit else 'ei tabanud'}, ennustus oli {p:.0%}) muutis kaalu "
                f"'{biggest}' {direction}maks ({old_weights[biggest]:.2f} -> {weights[biggest]:.2f}).")
        state["model"]["learning_log"].append({"ts": now_ts(), "text": note})
        state["model"]["learning_log"] = state["model"]["learning_log"][-80:]


def model_credibility(n_updates):
    """How much weight the learned model earns in the blended final score.
    Classic actuarial credibility formula (Z = n / (n + k)): grows smoothly
    with the amount of resolved evidence instead of jumping straight from 0%
    to 50% the instant a fixed training-count threshold is crossed. Capped
    below 1.0 - the hand-written heuristic encodes real domain knowledge
    (scam keywords, liquidity floors) the model may never see enough of to
    learn on its own, so it always keeps some say."""
    if n_updates <= 0:
        return 0.0
    return clamp(n_updates / (n_updates + CREDIBILITY_K), 0.0, CREDIBILITY_CAP)


def pearson_r(xs, ys):
    n = len(xs)
    if n < 5:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def feature_correlations(state):
    """Correlation between each raw feature and actual outcome (hit=1/0),
    computed over every resolved 24h result we have. Tells us, mathematically,
    which parts of the score actually predict success."""
    rows = []
    for rec in state["completed"] + state["pending_followups"]:
        if rec.get("result_24h") and rec.get("features"):
            rows.append((rec["features"], 1.0 if rec["result_24h"]["hit"] else 0.0))
    out = {}
    if len(rows) >= 5:
        for fn in FEATURE_NAMES:
            if fn == "bias":
                continue
            xs = [r[0].get(fn, 0.0) for r in rows]
            ys = [r[1] for r in rows]
            out[fn] = pearson_r(xs, ys)
    return out, len(rows)


# ---------------------------------------------------------------------------
# Virtual paper-trading portfolio (play money only)
# ---------------------------------------------------------------------------

def payoff_ratio(state):
    """b in the Kelly formula: average win size / average loss size
    (magnitudes, in % return) over every resolved 24h outcome so far.
    Falls back to 1.0 (breakeven-odds assumption) until there are at least a
    few of each so the ratio isn't just noise from one lucky/unlucky trade."""
    wins, losses = [], []
    for rec in state["completed"] + state["pending_followups"]:
        r = rec.get("result_24h")
        if not r:
            continue
        (wins if r["hit"] else losses).append(r["return_pct"] if r["hit"] else abs(r["return_pct"]))
    if len(wins) < 3 or len(losses) < 3:
        return 1.0
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    if avg_loss <= 0:
        return 1.0
    return avg_win / avg_loss


def kelly_position_pct(win_prob, b, base_pct):
    """Fractional Kelly criterion: f* = p - (1-p)/b is the bankroll fraction
    that maximizes long-run geometric growth for a bet with win probability p
    and payoff ratio b. We only ever use a quarter of that (KELLY_FRACTION)
    and clamp it to [KELLY_MIN_PCT, KELLY_MAX_PCT] - full Kelly is
    notoriously violent in practice (it will happily suggest 40%+ of the
    bankroll on a strong-looking edge). If the math says there's no edge
    (f* <= 0, or p<=0.5) we fall back to the flat base_pct rather than
    sizing to zero - an alert that clears every other bar still deserves a
    baseline-sized look."""
    if win_prob <= 0.5 or b <= 0:
        return base_pct
    f_star = win_prob - (1 - win_prob) / b
    if f_star <= 0:
        return base_pct
    return clamp(f_star * KELLY_FRACTION, KELLY_MIN_PCT, KELLY_MAX_PCT)


def _record_order_failure(state, what):
    """Count consecutive failed orders toward the kill-switch (live mode's
    'the API is misbehaving, stop before it gets expensive' brake)."""
    ks = state["killswitch"]
    ks["consec_failures"] = ks.get("consec_failures", 0) + 1
    print(f"WARN: order failed ({what}), consecutive failures: {ks['consec_failures']}", file=sys.stderr)


def maybe_open_position(state, rec_id, instrument, price, ts, risk, score, brk,
                        category="unknown", win_prob=None):
    """Called when a real alert fires. Opens a position THROUGH THE BROKER
    (paper simulation or real exchange order - same code path) if there's
    room (overall cap AND per-category diversification cap), cash, and the
    kill-switch is not engaged. Sized via fractional Kelly once the model
    has earned enough evidence (win_prob passed in); otherwise the flat
    PORTFOLIO_POSITION_PCT. A stop-loss is attached immediately: a real
    resting order in live mode, a simulated per-run check in paper mode."""
    pf = state["portfolio"]
    if state["killswitch"].get("active"):
        return False
    if len(pf["open_positions"]) >= pf["max_open_positions"]:
        return False
    cat_cap = PORTFOLIO_CATEGORY_CAP.get(category)
    if cat_cap is not None:
        cat_open = sum(1 for p in pf["open_positions"] if p.get("category") == category)
        if cat_open >= cat_cap:
            return False

    size_pct = pf["position_size_pct"]
    if win_prob is not None:
        size_pct = kelly_position_pct(win_prob, payoff_ratio(state), pf["position_size_pct"])

    size_usd = pf["balance"] * size_pct
    if size_usd < 1.0 or size_usd > pf["balance"]:
        return False

    try:
        fill = brk.buy(instrument, size_usd, price)
    except broker_mod.BrokerError as e:
        fill = None
        print(f"WARN: buy {instrument} failed: {e}", file=sys.stderr)
    if not fill:
        _record_order_failure(state, f"buy {instrument}")
        return False
    state["killswitch"]["consec_failures"] = 0

    stop_price = fill["price"] * (1 - broker_mod.STOP_LOSS_PCT)
    stop_order_id = brk.place_stop_loss(instrument, fill["quantity"], stop_price)

    pf["balance"] -= size_usd
    pf["open_positions"].append({
        "id": rec_id, "instrument": instrument, "entry_price": fill["price"],
        "quantity": fill["quantity"], "entry_fee_usd": fill["fee_usd"],
        "size_usd": round(size_usd, 2), "size_pct": round(size_pct * 100, 2),
        "entry_ts": ts, "risk": risk, "score": score, "category": category,
        "mode": brk.mode, "stop_price": stop_price, "stop_order_id": stop_order_id
    })
    return True


def _book_close(state, match, exit_price, proceeds_usd, exit_fee_usd, exit_ts, reason):
    """Shared bookkeeping for every way a position can close (24h timer,
    stop-loss, reconcile). P&L is NET of both sides' fees + slippage - the
    number that would actually land in the account."""
    pf = state["portfolio"]
    pf["open_positions"].remove(match)
    pnl_usd = proceeds_usd - match["size_usd"]
    pnl_pct = pnl_usd / match["size_usd"] * 100 if match["size_usd"] else 0.0
    pf["balance"] += proceeds_usd
    pf["closed_trades"].append({
        "id": match["id"], "instrument": match["instrument"], "entry_price": match["entry_price"],
        "exit_price": exit_price, "size_usd": match["size_usd"], "entry_ts": match["entry_ts"],
        "exit_ts": exit_ts, "pnl_usd": round(pnl_usd, 2), "pnl_pct": round(pnl_pct, 2),
        "entry_fee_usd": match.get("entry_fee_usd", 0.0), "exit_fee_usd": round(exit_fee_usd, 4),
        "exit_reason": reason, "mode": match.get("mode", "paper"),
        "risk": match["risk"], "score": match["score"], "category": match.get("category", "unknown")
    })
    pf["closed_trades"] = pf["closed_trades"][-300:]
    pf["balance_history"].append({"ts": exit_ts, "balance": round(pf["balance"], 2)})
    pf["balance_history"] = pf["balance_history"][-500:]


def close_position(state, rec_id, exit_price, exit_ts, brk, reason="24h"):
    """Called when that same recommendation's 24h result resolves. Sells the
    matching position (if one was opened and hasn't already been stopped
    out) through the broker and realizes the NET P&L."""
    pf = state["portfolio"]
    match = next((p for p in pf["open_positions"] if p["id"] == rec_id), None)
    if not match:
        return

    if "quantity" not in match:
        # Legacy position opened before the broker layer existed - close it
        # the old way (price-based, no fees) so history stays consistent.
        pf["open_positions"].remove(match)
        pnl_pct = (exit_price - match["entry_price"]) / match["entry_price"]
        pnl_usd = match["size_usd"] * pnl_pct
        pf["balance"] += match["size_usd"] + pnl_usd
        pf["closed_trades"].append({
            "id": rec_id, "instrument": match["instrument"], "entry_price": match["entry_price"],
            "exit_price": exit_price, "size_usd": match["size_usd"], "entry_ts": match["entry_ts"],
            "exit_ts": exit_ts, "pnl_usd": round(pnl_usd, 2), "pnl_pct": round(pnl_pct * 100, 2),
            "exit_reason": reason, "mode": "paper",
            "risk": match["risk"], "score": match["score"], "category": match.get("category", "unknown")
        })
        pf["closed_trades"] = pf["closed_trades"][-300:]
        pf["balance_history"].append({"ts": exit_ts, "balance": round(pf["balance"], 2)})
        pf["balance_history"] = pf["balance_history"][-500:]
        return

    if match.get("stop_order_id"):
        brk.cancel_order(match["instrument"], match["stop_order_id"])
    try:
        fill = brk.sell(match["instrument"], match["quantity"], exit_price)
    except broker_mod.BrokerError as e:
        fill = None
        print(f"WARN: sell {match['instrument']} failed: {e}", file=sys.stderr)
    if not fill:
        # Keep the position open; next hourly run will retry the close.
        _record_order_failure(state, f"sell {match['instrument']}")
        return
    state["killswitch"]["consec_failures"] = 0
    _book_close(state, match, fill["price"], fill["proceeds_usd"], fill["fee_usd"], exit_ts, reason)


def maybe_swap_position(state, rec_id, instrument, price, ts, risk, score, brk,
                        current_prices, category="unknown", win_prob=None):
    """Called when an alert fires but the book is already full. Finds the
    weakest eligible holding (broker-era position, held at least
    SWAP_MIN_HOLD_HOURS, currently >= 1% under water, and scored at least
    SWAP_MIN_SCORE_ADVANTAGE below the candidate), sells it, and opens the
    stronger candidate in its place. See the constants block above for why
    each condition exists (fee hysteresis). Returns (note, opened) - note
    is a summary line or None if no swap happened."""
    pf = state["portfolio"]
    if state["killswitch"].get("active"):
        return None, False
    if len(pf["open_positions"]) < pf["max_open_positions"]:
        return None, False
    if any(p["instrument"] == instrument for p in pf["open_positions"]):
        return None, False

    eligible = []
    for p in pf["open_positions"]:
        if "quantity" not in p:
            continue  # legacy pre-broker position - let its 24h timer handle it
        cp = current_prices.get(p["instrument"])
        if cp is None or p["entry_price"] <= 0:
            continue
        if (now_ts() - p["entry_ts"]) / 3600 < SWAP_MIN_HOLD_HOURS:
            continue
        unreal_pct = (cp - p["entry_price"]) / p["entry_price"] * 100
        if unreal_pct > SWAP_LOSER_MAX_RETURN_PCT:
            continue
        if score - p["score"] < SWAP_MIN_SCORE_ADVANTAGE:
            continue
        eligible.append((p, cp))
    if not eligible:
        return None, False

    weakest, cp = min(eligible, key=lambda x: x[0]["score"])

    # the per-category diversification cap must still hold AFTER the swap
    cat_cap = PORTFOLIO_CATEGORY_CAP.get(category)
    if cat_cap is not None:
        cat_open = sum(1 for p in pf["open_positions"]
                       if p.get("category") == category and p is not weakest)
        if cat_open >= cat_cap:
            return None, False

    if weakest.get("stop_order_id"):
        brk.cancel_order(weakest["instrument"], weakest["stop_order_id"])
    try:
        fill = brk.sell(weakest["instrument"], weakest["quantity"], cp)
    except broker_mod.BrokerError as e:
        fill = None
        print(f"WARN: swap-sell {weakest['instrument']} failed: {e}", file=sys.stderr)
    if not fill:
        _record_order_failure(state, f"swap-sell {weakest['instrument']}")
        return None, False
    state["killswitch"]["consec_failures"] = 0
    _book_close(state, weakest, fill["price"], fill["proceeds_usd"], fill["fee_usd"],
                now_ts(), "swap")
    old_trade = pf["closed_trades"][-1]

    opened = maybe_open_position(state, rec_id, instrument, price, ts, risk, score, brk,
                                 category=category, win_prob=win_prob)
    note = (f"🔄 VAHETUS: {weakest['instrument']} (skoor {weakest['score']}, {old_trade['pnl_pct']:+.1f}%) "
            f"→ {instrument} (skoor {score})")
    if not opened:
        note += " - uue ost ebaõnnestus, koht jäi vabaks"
    return note, opened


def check_stop_losses(state, current_prices, brk):
    """Once per run, for every open position: did the stop-loss trigger?
    Paper mode simulates the fill at the stop level; live mode checks the
    real resting order's status (and market-sells as a backstop if the
    resting order was never successfully placed). Returns notes for the
    summary message."""
    pf = state["portfolio"]
    notes = []
    for pos in list(pf["open_positions"]):
        fill = brk.check_stop_order(pos, current_prices.get(pos["instrument"]))
        if not fill:
            continue
        _book_close(state, pos, fill["price"], fill["proceeds_usd"], fill["fee_usd"],
                    now_ts(), "stop_loss")
        t = pf["closed_trades"][-1]
        notes.append(f"🛑 STOP-LOSS: {pos['instrument']} suleti {t['pnl_pct']:+.1f}% (${t['pnl_usd']:+.2f})")
    return notes


def update_killswitch(state, brk):
    """The circuit breaker. Engages (stops new opens) when:
      - portfolio balance dropped more than MAX_DAILY_LOSS_PCT in ~24h, or
      - MAX_CONSEC_FAILURES orders in a row failed (live API misbehaving).
    Manual reset: run once with env KILLSWITCH_RESET=1 (or edit state.json).
    Returns notes for the summary message."""
    ks = state["killswitch"]
    pf = state["portfolio"]
    notes = []

    if os.environ.get("KILLSWITCH_RESET") == "1" and ks["active"]:
        ks["active"] = False
        ks["reason"] = ""
        ks["consec_failures"] = 0
        notes.append("✅ Kill-switch käsitsi lähtestatud (KILLSWITCH_RESET=1) - bot avab jälle positsioone.")
        return notes

    if ks["active"]:
        notes.append(f"⛔ Kill-switch AKTIIVNE ({ks['reason']}) - uusi positsioone ei avata. "
                     "Lähtesta env muutujaga KILLSWITCH_RESET=1 kui oled olukorra üle vaadanud.")
        return notes

    cutoff = now_ts() - 24 * 3600
    baseline = pf["starting_balance"]
    for h in pf["balance_history"]:
        if h["ts"] <= cutoff:
            baseline = h["balance"]
    if baseline > 0:
        daily_loss_pct = (baseline - pf["balance"]) / baseline * 100
        if daily_loss_pct > broker_mod.MAX_DAILY_LOSS_PCT:
            ks["active"] = True
            ks["ts"] = now_ts()
            ks["reason"] = f"päevakaotus {daily_loss_pct:.1f}% > {broker_mod.MAX_DAILY_LOSS_PCT:.0f}% lubatud"
            notes.append(f"⛔ KILL-SWITCH RAKENDUS: {ks['reason']}. Uusi positsioone ei avata kuni lähtestamiseni.")
            return notes

    if ks.get("consec_failures", 0) >= broker_mod.MAX_CONSEC_FAILURES:
        ks["active"] = True
        ks["ts"] = now_ts()
        ks["reason"] = f"{ks['consec_failures']} järjestikust ebaõnnestunud orderit"
        notes.append(f"⛔ KILL-SWITCH RAKENDUS: {ks['reason']}. Kontrolli API võtit/börsi staatust.")
    return notes


# ---------------------------------------------------------------------------
# Portfolio performance math (Sharpe, drawdown, profit factor, expectancy)
# ---------------------------------------------------------------------------

def sharpe_ratio(returns_pct):
    """Annualized Sharpe ratio of the trade-return series (mean / stdev of
    returns, scaled by sqrt(365)). Trades resolve on a ~24h cadence (matched
    to the follow-up check), so treating the series like daily returns and
    applying the standard sqrt(365) annualization is a reasonable, clearly-
    labeled approximation - not a claim of true day-by-day granularity."""
    n = len(returns_pct)
    if n < 5:
        return None
    mean_r = sum(returns_pct) / n
    var = sum((r - mean_r) ** 2 for r in returns_pct) / (n - 1)
    std_r = math.sqrt(var)
    if std_r <= 0:
        return None
    return (mean_r / std_r) * math.sqrt(365)


def max_drawdown_pct(balance_series):
    """Largest peak-to-trough decline in the balance curve, as a % of the
    peak at the time - the standard way to size up 'how bad did this feel
    along the way', independent of where the balance ended up."""
    peak = None
    worst = 0.0
    for b in balance_series:
        if peak is None or b > peak:
            peak = b
        if peak and peak > 0:
            worst = max(worst, (peak - b) / peak)
    return worst * 100


def profit_factor(closed_trades):
    """Gross profit / gross loss. >1 means the winners outweigh the losers
    in dollar terms (not just in count) - the number professional trading
    desks actually watch, since a high win RATE with tiny wins and rare huge
    losses can still be a net loser."""
    gains = sum(t["pnl_usd"] for t in closed_trades if t["pnl_usd"] > 0)
    losses = -sum(t["pnl_usd"] for t in closed_trades if t["pnl_usd"] <= 0)
    if losses <= 0:
        return None
    return gains / losses


def expectancy_pct(closed_trades):
    """Average P&L per trade, in %. The single number that answers 'is
    doing this, on average, worth it' - positive expectancy is the whole
    point of any of the rest of this math."""
    if not closed_trades:
        return None
    return sum(t["pnl_pct"] for t in closed_trades) / len(closed_trades)


# ---------------------------------------------------------------------------
# Stage 1: screen
# ---------------------------------------------------------------------------

def realized_volatility(history):
    """Rolling stdev of hour-over-hour returns from up to the last VOL_WINDOW
    snapshots (sample stdev, Bessel-corrected). This is the token's OWN
    recent noise level, used to judge whether today's move is actually
    unusual for it. Returns None if there isn't enough history yet (a
    brand-new listing falls back to the plain, unnormalized momentum
    mapping - see momentum_score)."""
    if len(history) < 3:
        return None
    prices = [h["price"] for h in history[-VOL_WINDOW:]]
    rets = [(prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices)) if prices[i - 1] > 0]
    if len(rets) < 2:
        return None
    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def momentum_score(change_24h, hourly_vol=None):
    """Map 24h % change to a 0-100 score, blended half-and-half with a
    volatility-normalized (Sharpe-style) version once there's enough history
    to estimate the token's own noise level.

    Plain mapping: 0% -> 50, +30% -> 100, -30% -> 0 (same as v1, kept as a
    floor so a brand-new token with no history still scores sensibly).

    Risk-adjusted half: scale the hourly volatility up to an expected 24h
    volatility via the square-root-of-time rule (expected_24h ~= hourly *
    sqrt(24), valid for an approximately random-walk return series), then
    express the actual 24h move as a z-score against that. A +5% day on a
    token that normally barely moves is a much stronger signal than the
    same +5% on something that does that every afternoon - this is the same
    idea as a Sharpe ratio (return per unit of the asset's own risk),
    applied one day at a time instead of over a whole track record."""
    pct = clamp(change_24h, -0.30, 0.30)
    base = (pct + 0.30) / 0.60 * 100

    if hourly_vol and hourly_vol > 1e-6:
        expected_24h_vol = hourly_vol * math.sqrt(24)
        z = change_24h / expected_24h_vol
        risk_adjusted = clamp(50 + z * 12, 0, 100)  # z=0 -> 50, +-~4.2 sigma saturates the scale
        return round(0.5 * base + 0.5 * risk_adjusted, 1)
    return round(base, 1)


def linreg_slope_r2(ys):
    """Closed-form ordinary-least-squares fit of ys against the index
    0..n-1 (pure Python, no numpy needed for a handful of points). Returns
    (slope, r_squared). R^2 measures how well the points actually line up
    (1.0 = perfect line, 0.0 = scatter) - it's the "trust this trend" dial,
    continuous instead of a single noisy tick flipping a yes/no switch."""
    n = len(ys)
    if n < 2:
        return 0.0, 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    if n < 3:
        return slope, 0.0
    syy = sum((y - my) ** 2 for y in ys)
    if syy <= 0:
        return slope, 1.0 if abs(slope) < 1e-12 else 0.0
    r2 = clamp((sxy ** 2) / (sxx * syy), 0, 1)
    return slope, r2


def trend_bonus(history):
    """Fit a straight line (OLS) through the last TREND_WINDOW price points
    (normalized to % change from the window start, so slope is dimensionless)
    and through volume the same way. A confirmed uptrend needs a positive
    price slope AND a high R^2 (the points genuinely line up, not scattered)
    AND rising volume (participation, not a thin illiquid wobble). Bonus
    strength scales continuously with R^2 instead of being all-or-nothing -
    one noisy hour can no longer erase an otherwise real multi-hour trend,
    which is what the old strict-monotonic check was doing."""
    if len(history) < 2:
        return 0, "uus jälgimine, ajalugu veel liiga lühike trendi kinnitamiseks"

    recent = history[-TREND_WINDOW:]
    prices = [h["price"] for h in recent]
    vols = [h["volume_value"] for h in recent]

    p0 = prices[0] if prices[0] else 1e-9
    norm_prices = [(p - p0) / p0 for p in prices]
    price_slope, price_r2 = linreg_slope_r2(norm_prices)

    v0 = vols[0] if vols[0] else 1e-9
    norm_vols = [(v - v0) / v0 for v in vols]
    vol_slope, _ = linreg_slope_r2(norm_vols)

    n = len(recent)
    if n >= 4 and price_slope > 0 and price_r2 >= 0.5 and vol_slope > 0:
        bonus = round(8 + price_r2 * 12, 1)  # 8..20, scaled by how clean the fit is
        return bonus, f"trend kinnitatud (OLS R²={price_r2:.2f} üle {n} tunni, hind+maht mõlemad tõusuteel)"
    if price_slope > 0 and price_r2 >= 0.3:
        return 5.0, f"nõrk kinnitus - hind tõusuteel (R²={price_r2:.2f}), aga veel ebakindel"
    if price_slope < 0 and price_r2 >= 0.5:
        return -8.0, f"langustrend kinnitatud (R²={price_r2:.2f}) - ettevaatust"
    return -2.0, "ühekordne hüpe või müra, trend ei ole veel kinnitatud (madal R²)"


def alpha_bonus(change_24h, btc_change):
    """Single-factor (CAPM-style) alpha: r_token = beta*r_BTC + alpha, with
    beta fixed at 1 for transparency (a simple, explainable baseline rather
    than a fitted beta that would need its own history to estimate
    reliably). Most altcoins are largely leveraged BTC-beta in disguise; this
    isolates and rewards the part of the move BTC does NOT already explain -
    genuine idiosyncratic strength - instead of crediting a token for simply
    riding the whole market up. Returns (bonus_points, alpha_fraction, note)."""
    if btc_change is None:
        return 0.0, 0.0, "BTC võrdlust pole saadaval selles käivituses"
    alpha = change_24h - btc_change
    bonus = clamp(alpha * 100, -10, 15)
    if alpha > 0.03:
        note = f"alpha {alpha*100:+.1f}pp üle BTC-turu ({btc_change*100:+.1f}%) - reaalne oma jõud, mitte lihtsalt BTC laine"
    elif alpha < -0.03:
        note = f"alpha {alpha*100:+.1f}pp alla BTC-turu ({btc_change*100:+.1f}%) - nõrgem kui turg tervikuna"
    else:
        note = f"liigub suuresti koos BTC-turuga ({btc_change*100:+.1f}%), vähe oma alphat"
    return round(bonus, 1), alpha, note


def liquidity_bucket(volume_value):
    if volume_value >= 10_000_000:
        return "high"
    if volume_value >= 1_000_000:
        return "medium"
    return "low"


def screen(tickers_raw_path, out_path):
    """Stage 1. tickers_raw_path: JSON list of Crypto.com ticker dicts
    (as returned by get_tickers), one per watched instrument."""
    state = load_state()
    watchlist_cat = load_watchlist()
    tickers = load_json(tickers_raw_path, [])

    candidates = []
    near_misses = []
    ts = now_ts()
    threshold = state["thresholds"]["screen_score"]

    # Market benchmark for the alpha term: BTC's own 24h move in this same
    # run. Every altcoin's "how much of this move is genuinely its own" gets
    # measured against this one number (see alpha_bonus).
    btc_change = None
    for t in tickers:
        if t.get("instrument_name") == "BTCUSD":
            try:
                btc_change = float(t["change"])
            except (KeyError, ValueError, TypeError):
                btc_change = None
            break

    for t in tickers:
        inst = t["instrument_name"]
        try:
            price = float(t["last"])
            change_24h = float(t["change"])
            volume_value = float(t.get("volume_value", 0))
        except (KeyError, ValueError, TypeError):
            continue

        hist = state["history"].setdefault(inst, [])
        hourly_vol = realized_volatility(hist)
        m_score = momentum_score(change_24h, hourly_vol)
        bonus, trend_note = trend_bonus(hist)
        if inst == "BTCUSD":
            a_bonus, alpha_frac, alpha_note = 0.0, 0.0, "BTC ise on turu võrdlusalus"
        else:
            a_bonus, alpha_frac, alpha_note = alpha_bonus(change_24h, btc_change)

        # append this snapshot to history now (so next run can see it)
        hist.append({"ts": ts, "price": price, "change_24h": change_24h,
                      "volume_value": volume_value})
        state["history"][inst] = hist[-HISTORY_KEEP:]

        raw_score = clamp(m_score + bonus + a_bonus, 0, 100)
        category = watchlist_cat.get(inst, "unknown")

        entry = {
            "instrument": inst,
            "price": price,
            "change_24h": change_24h,
            "volume_value": volume_value,
            "category": category,
            "momentum_score": m_score,
            "trend_bonus": bonus,
            "trend_note": trend_note,
            "alpha_bonus": a_bonus,
            "alpha_pct": alpha_frac,
            "alpha_note": alpha_note,
            "raw_score": raw_score,
            "liquidity": liquidity_bucket(volume_value),
            "exploration": False,
        }

        if raw_score >= threshold:
            candidates.append(entry)
        elif raw_score >= threshold - EXPLORE_MARGIN:
            near_misses.append(entry)

    # Epsilon-greedy exploration: occasionally let a near-miss through anyway,
    # purely so the model keeps learning near the decision boundary instead
    # of only ever confirming what it already believes.
    explored = []
    for entry in near_misses:
        if random.random() < EXPLORE_EPSILON:
            entry["exploration"] = True
            explored.append(entry)

    candidates.sort(key=lambda c: c["raw_score"], reverse=True)
    all_out = (candidates + explored)[:10]
    save_state(state)
    save_json(out_path, all_out)
    print(f"STAGE1: {len(tickers)} jälgitavat, {len(candidates)} ületas läve ({threshold}), "
          f"{len(explored)} valiti eksperimentaalselt õppimiseks, {len(all_out)} saadetud edasi.")
    for c in all_out:
        tag = " [EXPLORE]" if c["exploration"] else ""
        print(f"  - {c['instrument']}: raw_score={c['raw_score']} "
              f"({c['momentum_score']}+{c['trend_bonus']}+{c['alpha_bonus']}) [{c['category']}/{c['liquidity']}]{tag}")


# ---------------------------------------------------------------------------
# Stage 2: finalize
# ---------------------------------------------------------------------------

SCAM_KEYWORDS = ["rug pull", "rugpull", "scam", "hack", "exploit", "delisted",
                 "delisting", "investigation", "lawsuit", "fraud", "hacked",
                 "exit scam", "ponzi"]


def hype_adjustment(note):
    """note: {'summary': str, 'sentiment': 'positive'|'neutral'|'negative'|'warning', 'found': bool}
    Returns (bonus, forced_risk_or_None, why_fragment)."""
    if not note or not note.get("found"):
        return 0, None, "veebist ei leidnud värsket kajastust - hinnang põhineb ainult turuandmetel"
    text = (note.get("summary") or "").lower()
    if note.get("sentiment") == "warning" or any(k in text for k in SCAM_KEYWORDS):
        return -30, "red", f"HOIATUS leitud veebist: {note.get('summary', '')[:200]}"
    if note.get("sentiment") == "positive":
        return 12, None, f"hype kinnitatud veebist: {note.get('summary', '')[:200]}"
    if note.get("sentiment") == "negative":
        return -10, None, f"negatiivne kajastus veebist: {note.get('summary', '')[:200]}"
    return 3, None, f"mainitud veebis, neutraalne toon: {note.get('summary', '')[:200]}"


def risk_label(category, liquidity, change_24h, forced_risk):
    if forced_risk:
        return forced_risk
    risk_points = 0
    if category == "meme" or category == "unknown":
        risk_points += 2
    if liquidity == "low":
        risk_points += 2
    elif liquidity == "medium":
        risk_points += 1
    if abs(change_24h) >= 0.20:
        risk_points += 2
    elif abs(change_24h) >= 0.10:
        risk_points += 1
    if risk_points >= 4:
        return "red"
    if risk_points >= 2:
        return "yellow"
    return "green"


def should_alert(state, inst, final_score, risk):
    prev = state["alerted"].get(inst)
    if not prev:
        return True
    if risk != prev.get("last_risk"):
        return True
    if final_score - prev.get("last_score", 0) >= 15:
        return True
    if now_ts() - prev.get("last_ts", 0) >= 12 * 3600:
        return True
    return False


def process_followups(state, current_prices, brk):
    """current_prices: {instrument: price}. Check pending recs that are due
    for their 24h or 7d outcome check, and train the model on each 24h
    result as soon as it resolves."""
    still_pending = []
    resolved_notes = []
    for rec in state["pending_followups"]:
        inst = rec["instrument"]
        age_h = (now_ts() - rec["ts"]) / 3600
        price_now = current_prices.get(inst)

        if not rec.get("result_24h") and age_h >= 24 and price_now:
            ret = (price_now - rec["price_at_call"]) / rec["price_at_call"] * 100
            hit = ret >= 3
            rec["result_24h"] = {"price": price_now, "return_pct": round(ret, 2), "hit": hit}
            resolved_notes.append(f"{inst} 24h: {ret:+.1f}%")
            if rec.get("features"):
                train_step(state, rec["features"], hit)
            close_position(state, rec["id"], price_now, now_ts(), brk, reason="24h")

        if not rec.get("result_7d") and age_h >= 24 * 7 and price_now:
            ret = (price_now - rec["price_at_call"]) / rec["price_at_call"] * 100
            rec["result_7d"] = {"price": price_now, "return_pct": round(ret, 2), "hit": ret >= 8}
            resolved_notes.append(f"{inst} 7d: {ret:+.1f}%")

        if rec.get("result_24h") and rec.get("result_7d"):
            state["completed"].append(rec)
        else:
            still_pending.append(rec)

    state["pending_followups"] = still_pending
    return resolved_notes


def adapt_thresholds(state):
    """Simple, explainable adaptive control loop on top of the learned model:
    look at hit-rate per risk bucket among RECENT completed recs (7d result
    when available, else 24h; last 40 records so it can track a changing
    market regime instead of being anchored to all-time history).
    Underperforming buckets get stricter; overperforming buckets get looser.

    Only acts once there's enough sample size to mean something, AND only
    when there is genuinely NEW evidence since the last adjustment (tracked
    via adjustment_checkpoint) - otherwise this would re-apply the exact same
    historical average as a fresh nudge on every single run and the
    thresholds would ratchet to their clamp limits within a day for no
    reason. One batch of new evidence -> at most one adjustment step.
    """
    completed = state["completed"]
    if len(completed) < 20:
        return None
    if len(completed) <= state.get("adjustment_checkpoint", 0):
        return None  # no new resolved evidence since we last adjusted
    state["adjustment_checkpoint"] = len(completed)

    recent = completed[-40:]
    by_risk = {"green": [], "yellow": [], "red": []}
    for rec in recent:
        result = rec.get("result_7d") or rec.get("result_24h")
        if result:
            by_risk.setdefault(rec["risk"], []).append(result["hit"])

    notes = []
    for risk, hits in by_risk.items():
        if len(hits) < 5:
            continue
        hit_rate = sum(hits) / len(hits)
        key = f"{risk}_hit_bar"
        if hit_rate < 0.40:
            state["thresholds"][key] = clamp(state["thresholds"].get(key, 65) + 5, 50, 90)
            notes.append(f"{risk}: tabamus {hit_rate:.0%} madal -> lävi tõstetud {state['thresholds'][key]}")
        elif hit_rate > 0.75:
            state["thresholds"][key] = clamp(state["thresholds"].get(key, 65) - 3, 50, 90)
            notes.append(f"{risk}: tabamus {hit_rate:.0%} kõrge -> lävi langetatud {state['thresholds'][key]}")

    overall_hits = [h for hs in by_risk.values() for h in hs]
    if overall_hits:
        overall_rate = sum(overall_hits) / len(overall_hits)
        if overall_rate < 0.35:
            state["thresholds"]["screen_score"] = clamp(state["thresholds"]["screen_score"] + 3, 50, 85)
            notes.append(f"üldine tabamus {overall_rate:.0%} madal -> screen_score tõstetud {state['thresholds']['screen_score']}")
        elif overall_rate > 0.65:
            state["thresholds"]["screen_score"] = clamp(state["thresholds"]["screen_score"] - 2, 50, 85)
            notes.append(f"üldine tabamus {overall_rate:.0%} kõrge -> screen_score langetatud {state['thresholds']['screen_score']}")
    return notes


def finalize(candidates_path, hype_notes_path, current_prices_path, out_summary_path):
    state = load_state()
    candidates = load_json(candidates_path, [])
    hype_notes = load_json(hype_notes_path, {})
    current_prices = load_json(current_prices_path, {})

    brk = broker_mod.get_broker()
    state["portfolio"]["mode"] = brk.mode
    reconcile_notes = brk.reconcile(state)
    stop_notes = check_stop_losses(state, current_prices, brk)
    followup_notes = process_followups(state, current_prices, brk)
    killswitch_notes = update_killswitch(state, brk)
    threshold_notes = adapt_thresholds(state) or []

    n_updates = state["model"]["n_updates"]
    model_ready = n_updates >= MODEL_MIN_TRAINING

    alerts = []
    swap_notes = []
    swaps_left = SWAP_MAX_PER_RUN
    ts = now_ts()

    for c in candidates:
        inst = c["instrument"]
        note = hype_notes.get(inst)
        bonus, forced_risk, why_hype = hype_adjustment(note)
        heuristic_score = clamp(c["raw_score"] + bonus, 0, 100)
        risk = risk_label(c["category"], c["liquidity"], c["change_24h"], forced_risk)

        features = build_features(c["momentum_score"], c["trend_bonus"], c["liquidity"],
                                   c["category"], c["change_24h"], bonus, c.get("alpha_pct", 0.0))
        model_p = model_predict(state["model"]["weights"], features)

        # Credibility-weighted blend (Z = n/(n+k)): the model's say grows
        # smoothly with evidence instead of jumping from 0% to 50% the
        # instant a fixed training-count is crossed.
        credibility = model_credibility(n_updates)
        final_score = round((1 - credibility) * heuristic_score + credibility * (model_p * 100), 1)
        if credibility >= 0.05:
            model_note = (f" Mudel (treenitud {n_updates} tulemuse pealt, praegu {credibility:.0%} "
                           f"kaaluga lõppskooris) hindab tabamise tõenäosuseks {model_p:.0%}.")
        else:
            model_note = f" Mudel alles õpib ({n_updates} tulemust kogutud, mõju hetkel alla 5%)."

        explore_note = " [EKSPERIMENTAALNE - allpool tavalävendit, kogutakse andmeid õppimiseks]" if c.get("exploration") else ""

        why = (f"Momentum {c['momentum_score']}/100 (24h {c['change_24h']*100:+.1f}%), "
               f"{c['trend_note']}. {c.get('alpha_note', '')} Likviidsus: {c['liquidity']} "
               f"(maht ${c['volume_value']:,.0f}). {why_hype}{model_note}{explore_note}")

        entry = {"instrument": inst, "score": final_score, "risk": risk, "why": why,
                 "price": c["price"], "category": c["category"], "features": features,
                 "exploration": c.get("exploration", False)}

        if should_alert(state, inst, final_score, risk):
            rec_id = state["next_id"]
            state["next_id"] += 1
            # Kelly sizing only once the model has earned the same minimum
            # evidence bar it always needed before being trusted at all -
            # sizing real (virtual) money is a higher bar than just display.
            win_prob = model_p if model_ready else None
            traded = maybe_open_position(state, rec_id, inst, c["price"], ts, risk, final_score,
                                          brk, category=c["category"], win_prob=win_prob)
            if not traded and swaps_left > 0:
                swap_note, traded = maybe_swap_position(state, rec_id, inst, c["price"], ts, risk,
                                                        final_score, brk, current_prices,
                                                        category=c["category"], win_prob=win_prob)
                if swap_note:
                    swaps_left -= 1
                    swap_notes.append(swap_note)
            entry["traded"] = traded
            alerts.append(entry)
            state["alerted"][inst] = {"last_score": final_score, "last_ts": ts, "last_risk": risk}
            state["pending_followups"].append({
                "id": rec_id, "instrument": inst, "ts": ts, "score": final_score,
                "risk": risk, "why": why, "price_at_call": c["price"], "features": features,
                "exploration": c.get("exploration", False), "traded": traded,
                "result_24h": None, "result_7d": None
            })

    alerts.sort(key=lambda a: a["score"], reverse=True)
    state["run_log"].append({
        "ts": ts, "n_candidates": len(candidates), "n_alerts": len(alerts),
        "followups_resolved": followup_notes, "threshold_adjustments": threshold_notes
    })
    state["run_log"] = state["run_log"][-200:]

    save_state(state)
    render_dashboard(state)

    # ---- chat summary (this text is what gets posted as the notification) ----
    mode_tag = "🔴 LIVE (PÄRIS RAHA)" if brk.mode == "live" else "📄 PAPER (mänguraha)"
    lines = [f"Režiim: {mode_tag}"] if brk.mode == "live" else []
    if killswitch_notes:
        lines.extend(killswitch_notes)
    if stop_notes:
        lines.extend(stop_notes)
    if swap_notes:
        lines.extend(swap_notes)
    if reconcile_notes:
        lines.extend(reconcile_notes)
    if alerts:
        lines.append(f"Cryptobot skann ({datetime.now(timezone.utc).strftime('%d.%m %H:%M')} UTC) - {len(alerts)} uut/muutunud signaali:")
        for a in alerts[:8]:
            emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[a["risk"]]
            tag = " 🧪" if a.get("exploration") else ""
            trade_tag = " 💰" if a.get("traded") else ""
            lines.append(f"{emoji}{tag}{trade_tag} {a['instrument']}: {a['score']}/100 - {a['why']}")
    else:
        wl = load_watchlist()
        lines.append(f"Cryptobot skann ({datetime.now(timezone.utc).strftime('%d.%m %H:%M')} UTC) - kontrolliti {len(wl)} tokenit, ükski ei ületanud praegu läve ({state['thresholds']['screen_score']}/100). Bot töötab, lihtsalt hetkel pole miski silma jäänud.")
    if followup_notes:
        lines.append("Tagasivaade: " + "; ".join(followup_notes))
    if threshold_notes:
        lines.append("Mudel kohandus: " + "; ".join(threshold_notes))
    if alerts:
        lines.append("Dashboard: https://skrrrando.github.io/cryptobot/")

    summary = "\n".join(lines)
    with open(out_summary_path, "w") as f:
        f.write(summary)
    print(summary)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "momentum": "Momentum (riskiga kohandatud)",
    "trend_bonus": "Trendi kinnitus (OLS R²)",
    "liquidity": "Likviidsus",
    "is_meme": "Meemi/trend-kategooria",
    "volatility": "Volatiilsus",
    "hype_bonus": "Veebi hype-kinnitus",
    "alpha": "Alpha (BTC-suhteline üleliikumine)",
}


def render_dashboard(state):
    completed = state["completed"]
    pending = state["pending_followups"]
    run_log = list(reversed(state["run_log"][-30:]))
    model = state["model"]
    thresholds = state["thresholds"]

    total_hits_24h_records = [r for r in (completed + pending) if r.get("result_24h")]
    total_hits_24h_records.sort(key=lambda r: r["ts"])
    hit_rate_24h = None
    if total_hits_24h_records:
        hits = [1 if r["result_24h"]["hit"] else 0 for r in total_hits_24h_records]
        hit_rate_24h = sum(hits) / len(hits) * 100

    # rolling hit-rate (window of 10) for the chart
    rolling_labels, rolling_values = [], []
    window = 10
    for i in range(len(total_hits_24h_records)):
        chunk = total_hits_24h_records[max(0, i - window + 1):i + 1]
        hr = sum(1 if r["result_24h"]["hit"] else 0 for r in chunk) / len(chunk) * 100
        rolling_labels.append(datetime.fromtimestamp(total_hits_24h_records[i]["ts"], tz=timezone.utc).strftime("%d.%m"))
        rolling_values.append(round(hr, 1))

    correlations, corr_n = feature_correlations(state)

    # Plain profit/loss breakdown (separate from the "hit" bar) - how many
    # resolved signals were simply above vs below their entry price, and by
    # how much on average. This is the "kui kasulik see päriselt on" view.
    returns_24h = [r["result_24h"]["return_pct"] for r in total_hits_24h_records]
    n_profit = sum(1 for x in returns_24h if x > 0)
    n_loss = sum(1 for x in returns_24h if x <= 0)
    avg_return = sum(returns_24h) / len(returns_24h) if returns_24h else None
    best = max(total_hits_24h_records, key=lambda r: r["result_24h"]["return_pct"]) if total_hits_24h_records else None
    worst = min(total_hits_24h_records, key=lambda r: r["result_24h"]["return_pct"]) if total_hits_24h_records else None

    # Trading portfolio (paper simulation or live mirror - see pf["mode"])
    pf = state["portfolio"]
    pf_mode = pf.get("mode", "paper")
    ks = state.get("killswitch", {})
    pf_return_pct = (pf["balance"] - pf["starting_balance"]) / pf["starting_balance"] * 100
    pf_closed = list(reversed(pf["closed_trades"][-40:]))
    pf_total_fees = (sum(t.get("entry_fee_usd", 0) + t.get("exit_fee_usd", 0) for t in pf["closed_trades"])
                     + sum(p.get("entry_fee_usd", 0) for p in pf["open_positions"]))
    pf_open = pf["open_positions"]
    pf_wins = sum(1 for t in pf["closed_trades"] if t["pnl_usd"] > 0)
    pf_losses = sum(1 for t in pf["closed_trades"] if t["pnl_usd"] <= 0)
    pf_chart_labels = [datetime.fromtimestamp(h["ts"], tz=timezone.utc).strftime("%d.%m %H:%M") if h["ts"] else "algus"
                        for h in pf["balance_history"]]
    pf_chart_values = [h["balance"] for h in pf["balance_history"]]

    # Real quant performance metrics - the "kas see matemaatika päriselt teenib raha" view.
    pf_sharpe = sharpe_ratio([t["pnl_pct"] for t in pf["closed_trades"]])
    pf_drawdown = max_drawdown_pct(pf_chart_values)
    pf_profit_factor = profit_factor(pf["closed_trades"])
    pf_expectancy = expectancy_pct(pf["closed_trades"])

    def risk_dot(risk):
        color = {"green": "#0ca30c", "yellow": "#fab219", "red": "#f4756f"}.get(risk, "#9a94b8")
        return f'<span class="dot" style="background:{color}"></span>{risk}'

    def info_badge(desc):
        """Small '?' badge next to a card heading - hover (desktop) or tap
        (mobile, via a click-toggle handled by the shared script at the
        bottom of the page) reveals the explanation, keeping the card itself
        uncluttered."""
        return f'<span class="info" tabindex="0">?<span class="info-pop">{desc}</span></span>'

    history_rows = ""
    all_recs = sorted(completed + pending, key=lambda r: r["ts"], reverse=True)[:40]
    for r in all_recs:
        r24 = r.get("result_24h")
        r7 = r.get("result_7d")
        r24_txt = f"{r24['return_pct']:+.1f}% {'✅' if r24['hit'] else '❌'}" if r24 else "ootel"
        r7_txt = f"{r7['return_pct']:+.1f}% {'✅' if r7['hit'] else '❌'}" if r7 else "ootel"
        dt = datetime.fromtimestamp(r["ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        explore_tag = ' <span class="tag">EXPLORE</span>' if r.get("exploration") else ""
        history_rows += f"""
        <tr>
          <td>{dt}</td>
          <td class="mono">{r['instrument']}{explore_tag}</td>
          <td>{r['score']}</td>
          <td>{risk_dot(r['risk'])}</td>
          <td>{r24_txt}</td>
          <td>{r7_txt}</td>
          <td class="why"><div class="clip">{r['why']}</div></td>
        </tr>"""

    run_rows = ""
    for rl in run_log:
        dt = datetime.fromtimestamp(rl["ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        run_rows += f"""
        <tr><td>{dt}</td><td>{rl['n_candidates']}</td><td>{rl['n_alerts']}</td>
        <td>{'; '.join(rl['followups_resolved']) or '-'}</td>
        <td>{'; '.join(rl['threshold_adjustments']) or '-'}</td></tr>"""

    weight_rows = ""
    for fn in FEATURE_NAMES:
        if fn == "bias":
            continue
        w = model["weights"].get(fn, 0.0)
        direction = "toetab tabamist" if w > 0.05 else ("vähendab tabamist" if w < -0.05 else "neutraalne")
        corr = correlations.get(fn)
        corr_txt = f"{corr:+.2f}" if corr is not None else "–"
        weight_rows += f"""
        <tr><td>{FEATURE_LABELS.get(fn, fn)}</td><td class="mono">{w:+.2f}</td>
        <td>{direction}</td><td class="mono">{corr_txt}</td></tr>"""

    learning_log_rows = ""
    for entry in reversed(model["learning_log"][-25:]):
        dt = datetime.fromtimestamp(entry["ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        learning_log_rows += f'<div class="log-entry"><span class="log-ts">{dt}</span> {entry["text"]}</div>'

    chart_script = ""
    if len(rolling_values) >= 2:
        chart_script = f"""
        <script>
          const ctx = document.getElementById('hitRateChart');
          new Chart(ctx, {{
            type: 'line',
            data: {{
              labels: {json.dumps(rolling_labels)},
              datasets: [{{
                label: 'Libisev tabamusprotsent (10 viimase pealt)',
                data: {json.dumps(rolling_values)},
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99,102,241,0.15)',
                tension: 0.3,
                fill: true,
                pointRadius: 2
              }}]
            }},
            options: {{
              responsive: true,
              scales: {{ y: {{ min: 0, max: 100, ticks: {{ color: '#8b94a8' }} }},
                         x: {{ ticks: {{ color: '#8b94a8' }} }} }},
              plugins: {{ legend: {{ labels: {{ color: '#e5e9f0' }} }} }}
            }}
          }});
        </script>"""

    model_status = (f"Mudel aktiivne (treenitud {model['n_updates']} tulemuse pealt, mõjutab 50% lõppskoorist)"
                     if model["n_updates"] >= MODEL_MIN_TRAINING
                     else f"Mudel õpib ({model['n_updates']}/{MODEL_MIN_TRAINING} tulemust) - skoor põhineb veel ainult käsitsi reeglitel")

    portfolio_chart_script = ""
    if len(pf_chart_values) >= 2:
        portfolio_chart_script = f"""
        <script>
          const ctx2 = document.getElementById('portfolioChart');
          new Chart(ctx2, {{
            type: 'line',
            data: {{
              labels: {json.dumps(pf_chart_labels)},
              datasets: [{{
                label: 'Virtuaalne saldo ($)',
                data: {json.dumps(pf_chart_values)},
                borderColor: {json.dumps('#22c55e' if pf['balance'] >= pf['starting_balance'] else '#ef4444')},
                backgroundColor: {json.dumps('rgba(34,197,94,0.12)' if pf['balance'] >= pf['starting_balance'] else 'rgba(239,68,68,0.12)')},
                tension: 0.25, fill: true, pointRadius: 2
              }}]
            }},
            options: {{
              responsive: true,
              scales: {{ y: {{ ticks: {{ color: '#8b94a8' }} }}, x: {{ ticks: {{ color: '#8b94a8' }} }} }},
              plugins: {{ legend: {{ labels: {{ color: '#e5e9f0' }} }} }}
            }}
          }});
        </script>"""

    pf_open_rows = ""
    for p in pf_open:
        dt = datetime.fromtimestamp(p["entry_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        size_pct_txt = f"{p['size_pct']:.1f}%" if "size_pct" in p else "–"
        stop_txt = f"{p['stop_price']:.6g}" if p.get("stop_price") else "–"
        pf_open_rows += f"""
        <tr><td>{dt}</td><td class="mono">{p['instrument']}</td><td>${p['size_usd']:.2f} <span class="tag" style="background:linear-gradient(120deg,var(--accent2),var(--pink))">{size_pct_txt}</span></td>
        <td class="mono">{p['entry_price']:.6g}</td><td class="mono">{stop_txt}</td><td>{risk_dot(p['risk'])}</td></tr>"""

    pf_closed_rows = ""
    for t in pf_closed:
        dt = datetime.fromtimestamp(t["exit_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        pnl_color = "#0ca30c" if t["pnl_usd"] > 0 else "#f4756f"
        reason_tag = {"stop_loss": ' <span class="tag" style="background:#f4756f">SL</span>',
                      "swap": ' <span class="tag" style="background:#f59e0b">SWAP</span>'}.get(t.get("exit_reason"), "")
        live_tag = ' <span class="tag" style="background:#dc2626">LIVE</span>' if t.get("mode") == "live" else ""
        pf_closed_rows += f"""
        <tr><td>{dt}</td><td class="mono">{t['instrument']}{reason_tag}{live_tag}</td><td>${t['size_usd']:.2f}</td>
        <td style="color:{pnl_color}">{t['pnl_pct']:+.1f}% (${t['pnl_usd']:+.2f})</td><td>{risk_dot(t['risk'])}</td></tr>"""

    mode_badge = ('<span class="tag" style="background:#dc2626">🔴 LIVE</span>' if pf_mode == "live"
                  else '<span class="tag">📄 PAPER</span>')
    killswitch_banner = ""
    if ks.get("active"):
        ks_dt = datetime.fromtimestamp(ks.get("ts", 0), tz=timezone.utc).strftime("%d.%m %H:%M") if ks.get("ts") else "?"
        killswitch_banner = f"""
  <div class="card" style="border-color:#f4756f;background:linear-gradient(160deg,#2a1520,#1c1220)">
    <h2>⛔ Kill-switch aktiivne (alates {ks_dt} UTC)</h2>
    <div class="desc" style="margin-bottom:0">Põhjus: {ks.get('reason', '?')}. Bot EI ava uusi positsioone
    (olemasolevaid haldab edasi), kuni käivitad ühe korra keskkonna­muutujaga <span class="mono">KILLSWITCH_RESET=1</span>.</div>
  </div>"""

    portfolio_desc = (
        ("PÄRIS RAHA - orderid lähevad Crypto.com börsile. " if pf_mode == "live"
         else "Mängu raha, mitte päris. ")
        + 'Kui bot alert annab ja on ruumi/raha, ostab positsiooni ja müüb 24h pärast automaatselt maha, '
        f"või varem, kui hind kukub stop-lossini (-{broker_mod.STOP_LOSS_PCT*100:.0f}% sisenemisest). "
        f"Kui raamat on täis, aga tuleb selgelt tugevam signaal (≥{SWAP_MIN_SCORE_ADVANTAGE}p kõrgem skoor), "
        f"vahetatakse nõrgim miinuses olev positsioon välja (max {SWAP_MAX_PER_RUN}/tunnis, et fee'd ei sööks kasumit). "
        f"P&L on NETO: sisaldab {broker_mod.FEE_PCT*100:.2f}% teenustasu mõlemal pool tehingut"
        + (f" ja {broker_mod.SLIPPAGE_PCT*100:.2f}% simuleeritud slippage'it" if pf_mode == "paper" else "")
        + ". "
        f"Suurus on kas fikseeritud {PORTFOLIO_POSITION_PCT*100:.0f}% (kuni mudel on piisavalt treenitud) "
        f"või pärast seda veerand-Kelly kriteeriumi järgi ({KELLY_MIN_PCT*100:.0f}–{KELLY_MAX_PCT*100:.0f}% vahemikus). "
        f"Algsaldo ${pf['starting_balance']:.0f}."
    )

    html = f"""<!DOCTYPE html>
<html lang="et">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#080b13">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Cryptobot Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.0/dist/chart.umd.js"></script>
<style>
  :root {{
    --bg: #080b13; --bg2: #0d1220; --card: #121826; --card2: #171f33; --border: #242f47;
    --text: #f1eefb; --text2: #c7c2e0; --muted: #9a94b8;
    --accent: #7c7ff2; --accent2: #a78bfa; --accent3: #22d3ee; --pink: #ec4899;
    --good: #0ca30c; --warning: #fab219; --critical: #f4756f;
  }}
  * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  html, body {{
    overflow-x: hidden; width: 100%; max-width: 100vw; background: var(--bg);
  }}
  html {{ scroll-behavior: smooth; min-height: 100%; }}
  body {{
    background:
      radial-gradient(900px 480px at 12% -8%, rgba(124,127,242,0.13) 0%, transparent 60%),
      radial-gradient(700px 420px at 100% 0%, rgba(34,211,238,0.08) 0%, transparent 55%),
      var(--bg);
    color: var(--text);
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    margin: 0; padding: 0; line-height: 1.55; -webkit-font-smoothing: antialiased;
    min-height: 100vh; min-height: 100dvh;
  }}
  .wrap {{ max-width: 1080px; width: 100%; margin: 0 auto; padding: 28px 20px 60px; }}
  .topbar {{
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    margin-bottom: 24px; flex-wrap: wrap;
  }}
  h1 {{
    font-size: clamp(20px, 5vw, 27px); margin: 0; font-weight: 800; letter-spacing: -.01em;
    display: flex; align-items: center; gap: 12px;
  }}
  h1 .title-text {{
    background: linear-gradient(100deg, var(--text) 30%, var(--accent2) 70%, var(--accent3) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .logo-badge {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 42px; height: 42px; border-radius: 13px; font-size: 21px; flex-shrink: 0;
    background: linear-gradient(140deg, var(--accent), var(--accent3) 60%, var(--pink) 120%);
    box-shadow: 0 0 0 1px rgba(255,255,255,.06) inset, 0 6px 18px -4px rgba(124,127,242,.55);
  }}
  .subtitle {{ color: var(--muted); font-size: 12.5px; margin-top: 4px; font-weight: 500; }}
  .method-btn {{
    background: linear-gradient(160deg, var(--card2), var(--card)); border: 1px solid var(--border);
    color: var(--text2); font: inherit; font-size: 12.5px; font-weight: 700;
    padding: 9px 15px; border-radius: 10px; cursor: pointer; flex-shrink: 0;
    transition: border-color .2s ease, color .2s ease, transform .2s ease;
  }}
  .method-btn:hover, .method-btn.open {{ border-color: rgba(124,127,242,.55); color: var(--text); transform: translateY(-1px); }}
  .method-panel {{
    display: none; margin: 0 0 20px; background: linear-gradient(160deg, var(--card2), var(--card));
    border: 1px solid var(--border); border-radius: 16px; padding: 18px 20px;
  }}
  .method-panel.open {{ display: block; }}
  .method-panel .desc {{ color: var(--muted); font-size: 12.5px; margin-bottom: 14px; }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px; margin-bottom: 20px; width: 100%;
  }}
  .stat-card {{
    background: linear-gradient(160deg, var(--card2), var(--card));
    border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; min-width: 0; position: relative; overflow: hidden;
    transition: transform .2s ease, border-color .2s ease;
  }}
  .stat-card::after {{
    content: ""; position: absolute; width: 80px; height: 80px; border-radius: 50%;
    top: -40px; right: -30px; filter: blur(24px); opacity: .22; pointer-events: none;
    background: var(--accent);
  }}
  .stat-card:hover {{ transform: translateY(-2px); border-color: rgba(124,127,242,.4); }}
  .stat-card .label {{
    color: var(--muted); font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; position: relative;
  }}
  .stat-card .value {{
    font-size: clamp(18px, 4vw, 23px); font-weight: 800; margin-top: 6px; letter-spacing: -.02em;
    position: relative; color: var(--text);
  }}
  .card {{
    background: linear-gradient(160deg, var(--card2), var(--card));
    border: 1px solid var(--border); border-radius: 16px;
    padding: 20px; margin-bottom: 16px; width: 100%;
    animation: fadeInUp .4s cubic-bezier(.16,.8,.4,1) both;
  }}
  .card > table, .card > .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; display: block; max-width: 100%; }}
  .card h2 {{ font-size: 14.5px; margin: 0 0 12px; color: var(--text); font-weight: 700; display: flex; align-items: center; gap: 7px; letter-spacing: -.005em; }}
  .card .desc {{ color: var(--muted); font-size: 12.5px; margin-bottom: 14px; }}
  .info {{
    position: relative; display: inline-flex; align-items: center; justify-content: center;
    width: 16px; height: 16px; border-radius: 50%; flex-shrink: 0;
    background: rgba(124,127,242,0.15); border: 1px solid rgba(124,127,242,0.4);
    color: var(--accent2); font-size: 10px; font-weight: 800; cursor: help; user-select: none;
  }}
  .info .info-pop {{
    position: absolute; top: calc(100% + 8px); left: 0; z-index: 40;
    width: max-content; max-width: min(300px, 78vw);
    background: #1c2440; border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; font-size: 12.5px; font-weight: 500; color: var(--text2);
    line-height: 1.55; box-shadow: 0 14px 30px -10px rgba(0,0,0,.7);
    opacity: 0; visibility: hidden; transform: translateY(-4px); pointer-events: none;
    transition: opacity .15s ease, transform .15s ease, visibility .15s ease;
  }}
  .info:hover .info-pop, .info:focus .info-pop, .info.open .info-pop {{
    opacity: 1; visibility: visible; transform: translateY(0); pointer-events: auto;
  }}
  .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 0 -4px; padding: 0 4px; max-width: 100%; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; min-width: 480px; }}
  th {{ text-align: left; color: var(--muted); font-weight: 700; padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; color: var(--text2); }}
  tr:last-child td {{ border-bottom: none; }}
  .mono {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; color: var(--text); }}
  .why {{ color: var(--muted); max-width: 420px; }}
  .why .clip {{
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; cursor: pointer;
  }}
  .why .clip.expanded {{ display: block; -webkit-line-clamp: unset; }}
  .why .clip:not(.expanded)::after {{ content: ""; }}
  details.card {{ padding: 0; }}
  details.card > summary {{
    list-style: none; cursor: pointer; padding: 16px 20px;
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
  }}
  details.card > summary::-webkit-details-marker {{ display: none; }}
  details.card > summary h2 {{ margin: 0; }}
  details.card > summary .chev {{
    color: var(--muted); font-size: 12px; flex-shrink: 0; transition: transform .2s ease;
  }}
  details.card[open] > summary .chev {{ transform: rotate(180deg); }}
  details.card > .body {{ padding: 0 20px 20px; }}
  .dot {{
    display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;
  }}
  .thresholds {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(125px, 1fr)); gap: 10px; font-size: 13px; color: var(--muted); width: 100%; }}
  .thresholds > div {{
    background: rgba(255,255,255,.02); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
  }}
  .thresholds b {{ color: var(--text); display: block; font-size: 17px; margin-top: 2px; font-weight: 700; }}
  .tag {{
    display: inline-block; font-size: 10px; background: linear-gradient(120deg, var(--accent), var(--accent3));
    color: white; padding: 2px 7px; border-radius: 6px; margin-left: 4px; font-weight: 700;
  }}
  .log-entry {{ font-size: 13px; color: var(--muted); padding: 10px 0; border-bottom: 1px solid var(--border); }}
  .log-entry:last-child {{ border-bottom: none; }}
  .log-ts {{ color: var(--text); font-family: ui-monospace, monospace; margin-right: 8px; font-weight: 700; }}
  .model-status {{
    font-size: 13px; padding: 12px 14px; background: rgba(124,127,242,0.10);
    border: 1px solid rgba(124,127,242,.35); border-radius: 10px; margin-bottom: 14px; color: var(--text2);
  }}
  canvas {{ max-width: 100%; }}
  ::-webkit-scrollbar {{ height: 8px; width: 8px; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 8px; }}
  @keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; }}
  }}
  @media (max-width: 640px) {{
    .wrap {{ padding: 18px 14px 48px; }}
    .card {{ padding: 16px; border-radius: 14px; }}
    details.card {{ padding: 0; }}
    details.card > summary {{ padding: 14px 16px; }}
    details.card > .body {{ padding: 0 16px 16px; }}
    .stats {{ gap: 8px; }}
    .stat-card {{ padding: 12px 12px; border-radius: 12px; }}
    table {{ font-size: 11.5px; min-width: 420px; }}
    th, td {{ padding: 6px 7px; }}
    .why {{ max-width: 190px; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div>
      <h1><span class="logo-badge">🤖</span><span class="title-text">Cryptobot Dashboard</span></h1>
      <div class="subtitle">Uuendatud {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M')} UTC · V4 · {mode_badge}</div>
    </div>
    <button type="button" class="method-btn" id="methodBtn">📐 Metoodika</button>
  </div>

  <div class="method-panel" id="methodPanel">
    <div class="desc">Iga number sellel lehel tuleb ühest neist viiest arvutusest.</div>
    <div class="table-scroll"><table>
      <tr><th>Mõõdik</th><th>Idee</th><th>Miks see loeb</th></tr>
      <tr><td class="mono">Riskiga momentum</td><td>24h liikumine jagatuna tokeni enda tavapärase kõikumisega (Sharpe-loogika, √24h-reegel)</td><td>5% tõus rahulikul BTC-l ≠ 5% tõus meemikoinil, mis teeb seda iga päev</td></tr>
      <tr><td class="mono">OLS trend + R²</td><td>Sirge joon läbi viimaste tundide hinna/mahu, R² = kui hästi punktid joonel püsivad</td><td>üks müratund enam ei kustuta terve trendi kinnitust</td></tr>
      <tr><td class="mono">Alpha vs BTC</td><td>r_token − r_BTC (lihtsustatud CAPM, beeta=1)</td><td>enamik altcoine on lihtsalt BTC laine - see leiab pärisliku oma jõu</td></tr>
      <tr><td class="mono">Credibility segamine</td><td>Z = n/(n+{CREDIBILITY_K}) - mudeli mõju kasvab sujuvalt kogemuse kasvades</td><td>skoor ei hüppa ühe andmepunkti pealt 15 protsendipunkti</td></tr>
      <tr><td class="mono">Veerand-Kelly</td><td>f* = p − (1−p)/b, kasutusel ¼ ulatuses, {KELLY_MIN_PCT*100:.0f}–{KELLY_MAX_PCT*100:.0f}% piirides</td><td>suurem, tõestatud edge → suurem (aga piiratud) panus, mitte alati sama 5%</td></tr>
    </table></div>
  </div>

{killswitch_banner}
  <div class="stats">
    <div class="stat-card"><div class="label">Aktiivseid soovitusi (ootel)</div><div class="value">{len(pending)}</div></div>
    <div class="stat-card"><div class="label">Lõpetatud soovitusi</div><div class="value">{len(completed)}</div></div>
    <div class="stat-card"><div class="label">24h tabamusprotsent</div><div class="value">{f'{hit_rate_24h:.0f}%' if hit_rate_24h is not None else '–'}</div></div>
    <div class="stat-card"><div class="label">Mudeli treeningsamme</div><div class="value">{model['n_updates']}</div></div>
  </div>

  <div class="card">
    <h2>Tulemuste kokkuvõte{info_badge("Kõik 24h tulemuse saanud soovitused, ilma tabamuslävendita - lihtsalt kas hind läks üles või alla.")}</h2>
    <div class="stats" style="margin-bottom:0">
      <div class="stat-card"><div class="label">Plussis</div><div class="value" style="color:var(--good)">{n_profit}</div></div>
      <div class="stat-card"><div class="label">Miinuses</div><div class="value" style="color:var(--critical)">{n_loss}</div></div>
      <div class="stat-card"><div class="label">Keskmine tootlus</div><div class="value">{f'{avg_return:+.1f}%' if avg_return is not None else '–'}</div></div>
      <div class="stat-card"><div class="label">Parim</div><div class="value" style="color:var(--good)">{f"{best['instrument']} {best['result_24h']['return_pct']:+.1f}%" if best else '–'}</div></div>
      <div class="stat-card"><div class="label">Halvim</div><div class="value" style="color:var(--critical)">{f"{worst['instrument']} {worst['result_24h']['return_pct']:+.1f}%" if worst else '–'}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>💰 Virtuaalne portfell{info_badge(portfolio_desc)}</h2>
    <div class="stats" style="margin-bottom:14px">
      <div class="stat-card"><div class="label">Praegune saldo</div><div class="value" style="color:{'var(--good)' if pf['balance']>=pf['starting_balance'] else 'var(--critical)'}">${pf['balance']:.2f}</div></div>
      <div class="stat-card"><div class="label">Tootlus algusest</div><div class="value" style="color:{'var(--good)' if pf_return_pct>=0 else 'var(--critical)'}">{pf_return_pct:+.1f}%</div></div>
      <div class="stat-card"><div class="label">Avatud positsioone</div><div class="value">{len(pf_open)}/{pf['max_open_positions']}</div></div>
      <div class="stat-card"><div class="label">Suletud kauplusi</div><div class="value">{pf_wins}✅ / {pf_losses}❌</div></div>
      <div class="stat-card"><div class="label">Sharpe (aastastatud)</div><div class="value">{f'{pf_sharpe:.2f}' if pf_sharpe is not None else '–'}</div></div>
      <div class="stat-card"><div class="label">Max languse sügavus</div><div class="value" style="color:var(--critical)">{f'-{pf_drawdown:.1f}%' if pf_drawdown else '0.0%'}</div></div>
      <div class="stat-card"><div class="label">Profit factor</div><div class="value">{f'{pf_profit_factor:.2f}' if pf_profit_factor is not None else '–'}</div></div>
      <div class="stat-card"><div class="label">Oodatav väärtus/kauplus</div><div class="value" style="color:{'var(--good)' if (pf_expectancy or 0)>=0 else 'var(--critical)'}">{f'{pf_expectancy:+.2f}%' if pf_expectancy is not None else '–'}</div></div>
      <div class="stat-card"><div class="label">Teenustasud kokku</div><div class="value" style="color:var(--warning)">${pf_total_fees:.2f}</div></div>
    </div>
    <canvas id="portfolioChart" height="70"></canvas>
    {portfolio_chart_script if portfolio_chart_script else '<div style="color:var(--muted)">Vaja on vähemalt paar suletud virtuaalset kauplust, enne kui graafik ilmub.</div>'}
    <h2 style="margin-top:20px">Avatud positsioonid</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal ostetud</th><th>Token</th><th>Suurus</th><th>Sisenemishind</th><th>Stop-loss</th><th>Risk</th></tr>
      {pf_open_rows or '<tr><td colspan="6" style="color:var(--muted)">Hetkel pole avatud positsioone.</td></tr>'}
    </table></div>
    <h2 style="margin-top:20px">Suletud kauplused (uusimad enne)</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal suletud</th><th>Token</th><th>Suurus</th><th>Tulemus</th><th>Risk</th></tr>
      {pf_closed_rows or '<tr><td colspan="5" style="color:var(--muted)">Veel pole ühtegi kauplust suletud.</td></tr>'}
    </table></div>
  </div>

  <div class="card">
    <h2>Õppiv mudel{info_badge("Iga kord kui üks soovitus saab tulemuse (24h hiljem), õpib see väike mudel sellest üht sammu - kaalud liiguvad selle poole, mis PÄRISELT ennustab tabamist, mitte selle poole, mida algul arvati.")}</h2>
    <div class="model-status">{model_status}</div>
    <div class="table-scroll"><table>
      <tr><th>Tunnus</th><th>Õpitud kaal</th><th>Mõju</th><th>Korrelatsioon tulemusega (n={corr_n})</th></tr>
      {weight_rows or '<tr><td colspan="4" style="color:var(--muted)">Veel andmeid pole.</td></tr>'}
    </table></div>
  </div>

  <div class="card">
    <h2>Tabamusprotsent üle aja{info_badge("Libisev tabamusprotsent (viimase 10 lahendatud soovituse pealt) - kui see joon aja jooksul tõuseb, õpib süsteem päriselt paremaks.")}</h2>
    <canvas id="hitRateChart" height="80"></canvas>
    {chart_script if chart_script else '<div style="color:var(--muted)">Vaja on vähemalt paar lahendatud tulemust, enne kui graafik ilmub.</div>'}
  </div>

  <details class="card">
    <summary><h2>📔 Õpipäevik{info_badge("Mida süsteem viimati enda kohta õppis, tavakeeles.")}</h2><span class="chev">▼</span></summary>
    <div class="body">
    {learning_log_rows or '<div style="color:var(--muted)">Veel pole midagi õppida olnud - vajab lahendatud tulemusi.</div>'}
    </div>
  </details>

  <details class="card">
    <summary><h2>🎚️ Riski-lävendid{info_badge("Kohandatakse automaatselt tagasiside põhjal.")}</h2><span class="chev">▼</span></summary>
    <div class="body">
    <div class="thresholds">
      <div>Skanni lävi: <b>{thresholds['screen_score']}</b></div>
      <div>🟢 roheline tabamuslävi: <b>{thresholds['green_hit_bar']}</b></div>
      <div>🟡 kollane tabamuslävi: <b>{thresholds['yellow_hit_bar']}</b></div>
      <div>🔴 punane tabamuslävi: <b>{thresholds['red_hit_bar']}</b></div>
    </div>
    </div>
  </details>

  <details class="card">
    <summary><h2>📜 Soovituste ajalugu{info_badge("Uusimad enne. Puuduta põhjendust, et seda täispikkuses lugeda. 🧪 EXPLORE = eksperimentaalne valik allpool tavalävendit, tehtud tahtlikult õppimise huvides.")}</h2><span class="chev">▼</span></summary>
    <div class="body">
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Token</th><th>Skoor</th><th>Risk</th><th>24h</th><th>7p</th><th>Põhjendus</th></tr>
      {history_rows or '<tr><td colspan="7" style="color:var(--muted)">Veel andmeid pole.</td></tr>'}
    </table></div>
    </div>
  </details>

  <details class="card">
    <summary><h2>⚙️ Käivituste logi</h2><span class="chev">▼</span></summary>
    <div class="body">
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Kandidaate</th><th>Alerte</th><th>Tagasivaade</th><th>Mudeli kohandused</th></tr>
      {run_rows or '<tr><td colspan="5" style="color:var(--muted)">Veel käivitusi pole.</td></tr>'}
    </table></div>
    </div>
  </details>
</div>
<script>
  document.querySelectorAll('.info').forEach(function(b) {{
    b.addEventListener('click', function(e) {{
      e.stopPropagation();
      e.preventDefault();  // don't also toggle the surrounding <details> section
      var wasOpen = b.classList.contains('open');
      document.querySelectorAll('.info.open').forEach(function(o) {{ o.classList.remove('open'); }});
      if (!wasOpen) b.classList.add('open');
    }});
  }});
  document.querySelectorAll('.why .clip').forEach(function(c) {{
    c.addEventListener('click', function() {{ c.classList.toggle('expanded'); }});
  }});
  var methodBtn = document.getElementById('methodBtn');
  var methodPanel = document.getElementById('methodPanel');
  if (methodBtn && methodPanel) {{
    methodBtn.addEventListener('click', function(e) {{
      e.stopPropagation();
      methodBtn.classList.toggle('open');
      methodPanel.classList.toggle('open');
    }});
  }}
  document.addEventListener('click', function() {{
    document.querySelectorAll('.info.open').forEach(function(o) {{ o.classList.remove('open'); }});
  }});
</script>
</body>
</html>"""
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    with open(INDEX_PATH, "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("screen")
    s1.add_argument("--tickers", required=True)
    s1.add_argument("--out", required=True)

    s2 = sub.add_parser("finalize")
    s2.add_argument("--candidates", required=True)
    s2.add_argument("--hype-notes", required=True)
    s2.add_argument("--current-prices", required=True)
    s2.add_argument("--out-summary", required=True)

    args = ap.parse_args()
    if args.cmd == "screen":
        screen(args.tickers, args.out)
    elif args.cmd == "finalize":
        finalize(args.candidates, args.hype_notes, args.current_prices, args.out_summary)
