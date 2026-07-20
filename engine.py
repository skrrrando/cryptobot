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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
WATCHLIST_PATH = os.path.join(BASE_DIR, "watchlist.json")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")  # same content - lets GitHub Pages serve a clean root URL

HISTORY_KEEP = 48  # keep last 48 hourly snapshots per instrument (~2 days)

# Feature order matters - it's the order weights/x vectors are stored in.
FEATURE_NAMES = ["bias", "momentum", "trend_bonus", "liquidity", "is_meme", "volatility", "hype_bonus"]

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
}

MODEL_MIN_TRAINING = 15   # need this many resolved outcomes before the model influences the score
LEARNING_RATE = 0.08
L2_LAMBDA = 0.015         # small ridge penalty - shrinks weights that aren't earning their keep back toward 0,
                          # so noise features don't drift just from finite-sample luck
EXPLORE_EPSILON = 0.15    # chance to let a near-miss candidate through anyway, for learning
EXPLORE_MARGIN = 10       # how far below the screen threshold still counts as "near"

# Virtual paper-trading portfolio - play money only, never touches anything real.
# Rule: when an alert fires (and there's room + cash), "buy" a position sized as
# a % of current balance; automatically "sell" it at the 24h mark (same moment
# the 24h follow-up outcome is already being checked), realizing whatever the
# actual price move was. This is a simple, transparent day-trade-style
# simulation - not a recommendation for how to actually trade.
PORTFOLIO_START_BALANCE = 1000.0
PORTFOLIO_POSITION_PCT = 0.05   # 5% of current balance per trade
PORTFOLIO_MAX_OPEN = 5          # at most 5 positions open at once (~25% max exposure)

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
        "open_positions": [],    # [{id, instrument, entry_price, size_usd, entry_ts, risk, score}]
        "closed_trades": [],     # [{...same + exit_price, exit_ts, pnl_usd, pnl_pct}]
        "balance_history": [{"ts": 0, "balance": PORTFOLIO_START_BALANCE}],
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


def build_features(momentum_score_, trend_bonus_, liquidity, category, change_24h, hype_bonus):
    liq_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
    return {
        "bias": 1.0,
        "momentum": momentum_score_ / 100.0,
        "trend_bonus": clamp((trend_bonus_ + 5) / 20.0, 0, 1),
        "liquidity": liq_map.get(liquidity, 0.0),
        "is_meme": 1.0 if category in ("meme", "unknown") else 0.0,
        "volatility": clamp(abs(change_24h) / 0.30, 0, 1),
        "hype_bonus": clamp((hype_bonus + 30) / 60.0, 0, 1),
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

def maybe_open_position(state, rec_id, instrument, price, ts, risk, score):
    """Called when a real alert fires. Opens a virtual position sized as a %
    of current balance, if there's room (max open positions) and cash. If
    not, the alert still stands - it just isn't "traded" in the simulation."""
    pf = state["portfolio"]
    if len(pf["open_positions"]) >= pf["max_open_positions"]:
        return False
    size_usd = pf["balance"] * pf["position_size_pct"]
    if size_usd < 1.0 or size_usd > pf["balance"]:
        return False
    pf["balance"] -= size_usd
    pf["open_positions"].append({
        "id": rec_id, "instrument": instrument, "entry_price": price,
        "size_usd": round(size_usd, 2), "entry_ts": ts, "risk": risk, "score": score
    })
    return True


def close_position(state, rec_id, exit_price, exit_ts):
    """Called when that same recommendation's 24h result resolves. Closes the
    matching virtual position (if one was opened) and realizes the P&L."""
    pf = state["portfolio"]
    match = next((p for p in pf["open_positions"] if p["id"] == rec_id), None)
    if not match:
        return
    pf["open_positions"].remove(match)
    pnl_pct = (exit_price - match["entry_price"]) / match["entry_price"]
    pnl_usd = match["size_usd"] * pnl_pct
    pf["balance"] += match["size_usd"] + pnl_usd
    pf["closed_trades"].append({
        "id": rec_id, "instrument": match["instrument"], "entry_price": match["entry_price"],
        "exit_price": exit_price, "size_usd": match["size_usd"], "entry_ts": match["entry_ts"],
        "exit_ts": exit_ts, "pnl_usd": round(pnl_usd, 2), "pnl_pct": round(pnl_pct * 100, 2),
        "risk": match["risk"], "score": match["score"]
    })
    pf["closed_trades"] = pf["closed_trades"][-300:]
    pf["balance_history"].append({"ts": exit_ts, "balance": round(pf["balance"], 2)})
    pf["balance_history"] = pf["balance_history"][-500:]


# ---------------------------------------------------------------------------
# Stage 1: screen
# ---------------------------------------------------------------------------

def momentum_score(change_24h):
    """Map 24h % change (fractional, e.g. 0.05 = +5%) to a 0-100 score.
    0% -> 50, +30% -> 100, -30% -> 0."""
    pct = clamp(change_24h, -0.30, 0.30)
    return round((pct + 0.30) / 0.60 * 100, 1)


def trend_bonus(history):
    """Look at up to the last 3 snapshots. If price AND volume have been
    rising each step (not just a single-hour spike), award a confirmation
    bonus. This is the main noise filter: a lone spike gets no bonus."""
    if len(history) < 2:
        return 0, "uus jälgimine, ajalugu veel liiga lühike trendi kinnitamiseks"
    recent = history[-3:]
    prices = [h["price"] for h in recent]
    vols = [h["volume_value"] for h in recent]
    price_rising = all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))
    vol_rising = all(vols[i] <= vols[i + 1] * 1.05 for i in range(len(vols) - 1))  # allow small noise
    if len(recent) >= 3 and price_rising and vol_rising:
        return 15, f"trend kinnitatud - hind ja maht tõusnud järjest {len(recent)} tunni jooksul"
    if len(recent) >= 2 and prices[-1] > prices[0]:
        return 5, "nõrk kinnitus - hind tõusuteel, aga vähe andmepunkte"
    return -5, "ühekordne hüpe, trend ei ole veel kinnitatud (võib olla müra)"


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

    for t in tickers:
        inst = t["instrument_name"]
        try:
            price = float(t["last"])
            change_24h = float(t["change"])
            volume_value = float(t.get("volume_value", 0))
        except (KeyError, ValueError, TypeError):
            continue

        hist = state["history"].setdefault(inst, [])
        m_score = momentum_score(change_24h)
        bonus, trend_note = trend_bonus(hist)

        # append this snapshot to history now (so next run can see it)
        hist.append({"ts": ts, "price": price, "change_24h": change_24h,
                      "volume_value": volume_value})
        state["history"][inst] = hist[-HISTORY_KEEP:]

        raw_score = clamp(m_score + bonus, 0, 100)
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
              f"({c['momentum_score']}+{c['trend_bonus']}) [{c['category']}/{c['liquidity']}]{tag}")


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


def process_followups(state, current_prices):
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
            close_position(state, rec["id"], price_now, now_ts())

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

    followup_notes = process_followups(state, current_prices)
    threshold_notes = adapt_thresholds(state) or []

    n_updates = state["model"]["n_updates"]
    model_ready = n_updates >= MODEL_MIN_TRAINING

    alerts = []
    ts = now_ts()

    for c in candidates:
        inst = c["instrument"]
        note = hype_notes.get(inst)
        bonus, forced_risk, why_hype = hype_adjustment(note)
        heuristic_score = clamp(c["raw_score"] + bonus, 0, 100)
        risk = risk_label(c["category"], c["liquidity"], c["change_24h"], forced_risk)

        features = build_features(c["momentum_score"], c["trend_bonus"], c["liquidity"],
                                   c["category"], c["change_24h"], bonus)
        model_p = model_predict(state["model"]["weights"], features)

        if model_ready:
            final_score = round(0.5 * heuristic_score + 0.5 * (model_p * 100), 1)
            model_note = f" Mudel (treenitud {n_updates} tulemuse pealt) hindab tabamise tõenäosuseks {model_p:.0%}."
        else:
            final_score = heuristic_score
            model_note = f" Mudel alles õpib ({n_updates}/{MODEL_MIN_TRAINING} tulemust kogutud enne kui see skoori mõjutama hakkab)."

        explore_note = " [EKSPERIMENTAALNE - allpool tavalävendit, kogutakse andmeid õppimiseks]" if c.get("exploration") else ""

        why = (f"Momentum {c['momentum_score']}/100 (24h {c['change_24h']*100:+.1f}%), "
               f"{c['trend_note']}. Likviidsus: {c['liquidity']} "
               f"(maht ${c['volume_value']:,.0f}). {why_hype}{model_note}{explore_note}")

        entry = {"instrument": inst, "score": final_score, "risk": risk, "why": why,
                 "price": c["price"], "category": c["category"], "features": features,
                 "exploration": c.get("exploration", False)}

        if should_alert(state, inst, final_score, risk):
            rec_id = state["next_id"]
            state["next_id"] += 1
            traded = maybe_open_position(state, rec_id, inst, c["price"], ts, risk, final_score)
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
    lines = []
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
    "momentum": "Momentum (hinnaliikumine)",
    "trend_bonus": "Trendi kinnitus",
    "liquidity": "Likviidsus",
    "is_meme": "Meemi/trend-kategooria",
    "volatility": "Volatiilsus",
    "hype_bonus": "Veebi hype-kinnitus",
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

    # Virtual paper-trading portfolio
    pf = state["portfolio"]
    pf_return_pct = (pf["balance"] - pf["starting_balance"]) / pf["starting_balance"] * 100
    pf_closed = list(reversed(pf["closed_trades"][-40:]))
    pf_open = pf["open_positions"]
    pf_wins = sum(1 for t in pf["closed_trades"] if t["pnl_usd"] > 0)
    pf_losses = sum(1 for t in pf["closed_trades"] if t["pnl_usd"] <= 0)
    pf_chart_labels = [datetime.fromtimestamp(h["ts"], tz=timezone.utc).strftime("%d.%m %H:%M") if h["ts"] else "algus"
                        for h in pf["balance_history"]]
    pf_chart_values = [h["balance"] for h in pf["balance_history"]]

    def risk_dot(risk):
        color = {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444"}.get(risk, "#94a3b8")
        return f'<span class="dot" style="background:{color}"></span>{risk}'

    history_rows = ""
    all_recs = sorted(completed + pending, key=lambda r: r["ts"], reverse=True)[:60]
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
          <td class="why">{r['why']}</td>
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
        pf_open_rows += f"""
        <tr><td>{dt}</td><td class="mono">{p['instrument']}</td><td>${p['size_usd']:.2f}</td>
        <td class="mono">{p['entry_price']:.6g}</td><td>{risk_dot(p['risk'])}</td></tr>"""

    pf_closed_rows = ""
    for t in pf_closed:
        dt = datetime.fromtimestamp(t["exit_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        pnl_color = "#22c55e" if t["pnl_usd"] > 0 else "#ef4444"
        pf_closed_rows += f"""
        <tr><td>{dt}</td><td class="mono">{t['instrument']}</td><td>${t['size_usd']:.2f}</td>
        <td style="color:{pnl_color}">{t['pnl_pct']:+.1f}% (${t['pnl_usd']:+.2f})</td><td>{risk_dot(t['risk'])}</td></tr>"""

    html = f"""<!DOCTYPE html>
<html lang="et">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#080b13">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Cryptobot Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.0/dist/chart.umd.js"></script>
<style>
  :root {{
    --bg: #080b13; --bg2: #0d1220; --card: #121826; --card2: #171f33; --border: #242f47;
    --text: #edf1f9; --muted: #8b94a8; --accent: #7c7ff2; --accent2: #a78bfa; --accent3: #22d3ee;
    --pink: #ec4899; --green: #22c55e; --yellow: #eab308; --red: #ef4444;
  }}
  * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  html, body {{
    overflow-x: hidden; width: 100%; max-width: 100vw; background: var(--bg);
  }}
  html {{ scroll-behavior: smooth; min-height: 100%; }}
  body {{
    background:
      radial-gradient(900px 480px at 12% -8%, rgba(124,127,242,0.18) 0%, transparent 60%),
      radial-gradient(700px 420px at 100% 0%, rgba(34,211,238,0.12) 0%, transparent 55%),
      radial-gradient(600px 500px at 90% 60%, rgba(236,72,153,0.08) 0%, transparent 55%),
      var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
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
    background: linear-gradient(100deg, var(--text) 30%, var(--accent2) 65%, var(--accent3) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    background-size: 200% auto; animation: shimmer 6s ease-in-out infinite;
  }}
  .logo-badge {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 42px; height: 42px; border-radius: 13px; font-size: 21px; flex-shrink: 0;
    background: linear-gradient(140deg, var(--accent), var(--accent3) 60%, var(--pink) 120%);
    box-shadow: 0 0 0 1px rgba(255,255,255,.06) inset, 0 6px 18px -4px rgba(124,127,242,.6);
    animation: glowpulse 3.2s ease-in-out infinite;
  }}
  .subtitle {{ color: var(--muted); font-size: 12.5px; margin-top: 4px; }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px; margin-bottom: 20px; width: 100%;
  }}
  .stat-card {{
    background: linear-gradient(160deg, var(--card2), var(--card));
    border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; min-width: 0; position: relative; overflow: hidden;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
  }}
  .stat-card::before {{
    content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
    opacity: .0; transition: opacity .25s ease;
  }}
  .stat-card::after {{
    content: ""; position: absolute; width: 90px; height: 90px; border-radius: 50%;
    top: -45px; right: -35px; filter: blur(22px); opacity: .35; pointer-events: none;
    background: var(--accent);
  }}
  .stat-card:nth-child(4n+1)::after {{ background: var(--accent); }}
  .stat-card:nth-child(4n+2)::after {{ background: var(--accent3); }}
  .stat-card:nth-child(4n+3)::after {{ background: var(--pink); }}
  .stat-card:nth-child(4n+4)::after {{ background: var(--green); }}
  .stat-card:hover {{ transform: translateY(-3px); border-color: rgba(124,127,242,.5); box-shadow: 0 10px 24px -12px rgba(124,127,242,.35); }}
  .stat-card:hover::before {{ opacity: 1; }}
  .stat-card .label {{
    color: var(--muted); font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; position: relative;
  }}
  .stat-card .value {{
    font-size: clamp(18px, 4vw, 24px); font-weight: 800; margin-top: 6px; letter-spacing: -.02em;
    position: relative;
    background: linear-gradient(100deg, var(--text), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .card {{
    background: linear-gradient(160deg, var(--card2), var(--card));
    border: 1px solid var(--border); border-radius: 16px;
    padding: 20px; margin-bottom: 20px; width: 100%;
    box-shadow: 0 8px 24px -12px rgba(0,0,0,.55);
    animation: fadeInUp .5s cubic-bezier(.16,.8,.4,1) both;
    transition: border-color .3s ease, box-shadow .3s ease;
  }}
  .card:hover {{ border-color: rgba(124,127,242,.35); box-shadow: 0 12px 32px -14px rgba(124,127,242,.25), 0 8px 24px -12px rgba(0,0,0,.55); }}
  .card:nth-of-type(1) {{ animation-delay: .02s; }}
  .card:nth-of-type(2) {{ animation-delay: .06s; }}
  .card:nth-of-type(3) {{ animation-delay: .10s; }}
  .card:nth-of-type(4) {{ animation-delay: .14s; }}
  .card:nth-of-type(5) {{ animation-delay: .18s; }}
  .card:nth-of-type(6) {{ animation-delay: .22s; }}
  .card:nth-of-type(n+7) {{ animation-delay: .26s; }}
  .card > table, .card > .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; display: block; max-width: 100%; }}
  .card h2 {{ font-size: 15px; margin: 0 0 4px; color: var(--text); font-weight: 650; display: flex; align-items: center; gap: 8px; }}
  .card .desc {{ color: var(--muted); font-size: 12.5px; margin-bottom: 14px; }}
  .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 0 -4px; padding: 0 4px; max-width: 100%; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; min-width: 480px; }}
  th {{ text-align: left; color: var(--muted); font-weight: 600; padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; font-size: 11.5px; text-transform: uppercase; letter-spacing: .03em; }}
  td {{ padding: 9px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr {{ transition: background .2s ease; }}
  tr:hover td {{ background: rgba(124,127,242,0.06); }}
  .mono {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  .why {{ color: var(--muted); max-width: 420px; }}
  .dot {{
    display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;
    box-shadow: 0 0 8px currentColor; animation: pulse 2.4s ease-in-out infinite;
  }}
  .thresholds {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(125px, 1fr)); gap: 12px; font-size: 13px; color: var(--muted); width: 100%; }}
  .thresholds > div {{
    background: rgba(255,255,255,.02); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
    transition: transform .25s ease, border-color .25s ease;
  }}
  .thresholds > div:hover {{ transform: translateY(-2px); border-color: rgba(124,127,242,.4); }}
  .thresholds b {{ color: var(--text); display: block; font-size: 17px; margin-top: 2px; }}
  .tag {{
    display: inline-block; font-size: 10px; background: linear-gradient(120deg, var(--accent), var(--accent3));
    color: white; padding: 2px 7px; border-radius: 6px; margin-left: 4px; font-weight: 700;
  }}
  .log-entry {{ font-size: 13px; color: var(--muted); padding: 10px 0; border-bottom: 1px solid var(--border); }}
  .log-entry:last-child {{ border-bottom: none; }}
  .log-ts {{ color: var(--text); font-family: ui-monospace, monospace; margin-right: 8px; font-weight: 600; }}
  .model-status {{
    font-size: 13px; padding: 12px 14px; background: rgba(124,127,242,0.12);
    border: 1px solid rgba(124,127,242,.4); border-radius: 10px; margin-bottom: 14px;
  }}
  canvas {{ max-width: 100%; }}
  ::-webkit-scrollbar {{ height: 8px; width: 8px; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 8px; }}
  @keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes shimmer {{
    0%, 100% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: .55; }}
  }}
  @keyframes glowpulse {{
    0%, 100% {{ box-shadow: 0 0 0 1px rgba(255,255,255,.06) inset, 0 6px 18px -4px rgba(124,127,242,.6); }}
    50% {{ box-shadow: 0 0 0 1px rgba(255,255,255,.06) inset, 0 6px 26px -2px rgba(124,127,242,.9); }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; }}
  }}
  @media (max-width: 640px) {{
    .wrap {{ padding: 18px 14px 48px; }}
    .card {{ padding: 16px; border-radius: 14px; }}
    .stats {{ gap: 8px; }}
    .stat-card {{ padding: 12px 12px; border-radius: 12px; }}
    table {{ font-size: 12.5px; }}
    .why {{ max-width: 220px; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div>
      <h1><span class="logo-badge">🤖</span><span class="title-text">Cryptobot Dashboard</span></h1>
      <div class="subtitle">Uuendatud {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M')} UTC · V2 (momentum + trend + veebi hype-kinnitus + isikliku õppiva mudeliga)</div>
    </div>
  </div>

  <div class="stats">
    <div class="stat-card"><div class="label">Aktiivseid soovitusi (ootel)</div><div class="value">{len(pending)}</div></div>
    <div class="stat-card"><div class="label">Lõpetatud soovitusi</div><div class="value">{len(completed)}</div></div>
    <div class="stat-card"><div class="label">24h tabamusprotsent</div><div class="value">{f'{hit_rate_24h:.0f}%' if hit_rate_24h is not None else '–'}</div></div>
    <div class="stat-card"><div class="label">Mudeli treeningsamme</div><div class="value">{model['n_updates']}</div></div>
  </div>

  <div class="card">
    <h2>Tulemuste kokkuvõte - kui kasulik see päriselt on</h2>
    <div class="desc">Kõik 24h tulemuse saanud soovitused, ilma tabamuslävendita - lihtsalt kas hind läks üles või alla.</div>
    <div class="stats" style="margin-bottom:0">
      <div class="stat-card"><div class="label">Plussis</div><div class="value" style="color:#22c55e">{n_profit}</div></div>
      <div class="stat-card"><div class="label">Miinuses</div><div class="value" style="color:#ef4444">{n_loss}</div></div>
      <div class="stat-card"><div class="label">Keskmine tootlus</div><div class="value">{f'{avg_return:+.1f}%' if avg_return is not None else '–'}</div></div>
      <div class="stat-card"><div class="label">Parim</div><div class="value" style="color:#22c55e">{f"{best['instrument']} {best['result_24h']['return_pct']:+.1f}%" if best else '–'}</div></div>
      <div class="stat-card"><div class="label">Halvim</div><div class="value" style="color:#ef4444">{f"{worst['instrument']} {worst['result_24h']['return_pct']:+.1f}%" if worst else '–'}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>💰 Virtuaalne portfell (mängu raha, mitte päris)</h2>
    <div class="desc">Kui bot alert annab ja on ruumi/raha, "ostab" see virtuaalselt {PORTFOLIO_POSITION_PCT*100:.0f}% praegusest saldost, ja "müüb" 24h pärast automaatselt maha. Algsaldo ${pf['starting_balance']:.0f}. See EI OLE päris raha ega päris kauplemine.</div>
    <div class="stats" style="margin-bottom:14px">
      <div class="stat-card"><div class="label">Praegune saldo</div><div class="value" style="color:{'#22c55e' if pf['balance']>=pf['starting_balance'] else '#ef4444'}">${pf['balance']:.2f}</div></div>
      <div class="stat-card"><div class="label">Tootlus algusest</div><div class="value" style="color:{'#22c55e' if pf_return_pct>=0 else '#ef4444'}">{pf_return_pct:+.1f}%</div></div>
      <div class="stat-card"><div class="label">Avatud positsioone</div><div class="value">{len(pf_open)}/{pf['max_open_positions']}</div></div>
      <div class="stat-card"><div class="label">Suletud kauplusi</div><div class="value" style="background:none;-webkit-text-fill-color:initial;color:var(--text)">{pf_wins}✅ / {pf_losses}❌</div></div>
    </div>
    <canvas id="portfolioChart" height="70"></canvas>
    {portfolio_chart_script if portfolio_chart_script else '<div style="color:var(--muted)">Vaja on vähemalt paar suletud virtuaalset kauplust, enne kui graafik ilmub.</div>'}
    <h2 style="margin-top:20px">Avatud positsioonid</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal ostetud</th><th>Token</th><th>Suurus</th><th>Sisenemishind</th><th>Risk</th></tr>
      {pf_open_rows or '<tr><td colspan="5" style="color:var(--muted)">Hetkel pole avatud positsioone.</td></tr>'}
    </table></div>
    <h2 style="margin-top:20px">Suletud kauplused (uusimad enne)</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal suletud</th><th>Token</th><th>Suurus</th><th>Tulemus</th><th>Risk</th></tr>
      {pf_closed_rows or '<tr><td colspan="5" style="color:var(--muted)">Veel pole ühtegi kauplust suletud.</td></tr>'}
    </table></div>
  </div>

  <div class="card">
    <h2>Õppiv mudel</h2>
    <div class="desc">Iga kord kui üks soovitus saab tulemuse (24h hiljem), õpib see väike mudel sellest üht sammu - kaalud liiguvad selle poole, mis PÄRISELT ennustab tabamist, mitte selle poole, mida ma algul arvasin.</div>
    <div class="model-status">{model_status}</div>
    <div class="table-scroll"><table>
      <tr><th>Tunnus</th><th>Õpitud kaal</th><th>Mõju</th><th>Korrelatsioon tulemusega (n={corr_n})</th></tr>
      {weight_rows or '<tr><td colspan="4" style="color:var(--muted)">Veel andmeid pole.</td></tr>'}
    </table></div>
  </div>

  <div class="card">
    <h2>Tabamusprotsent üle aja</h2>
    <div class="desc">Libisev tabamusprotsent (viimase 10 lahendatud soovituse pealt) - kui see joon aja jooksul tõuseb, õpib süsteem päriselt paremaks.</div>
    <canvas id="hitRateChart" height="80"></canvas>
    {chart_script if chart_script else '<div style="color:var(--muted)">Vaja on vähemalt paar lahendatud tulemust, enne kui graafik ilmub.</div>'}
  </div>

  <div class="card">
    <h2>Õpipäevik</h2>
    <div class="desc">Mida süsteem viimati enda kohta õppis, tavakeeles.</div>
    {learning_log_rows or '<div style="color:var(--muted)">Veel pole midagi õppida olnud - vajab lahendatud tulemusi.</div>'}
  </div>

  <div class="card">
    <h2>Praegused riski-lävendid (kohandatud automaatselt tagasiside põhjal)</h2>
    <div class="thresholds">
      <div>Skanni lävi: <b>{thresholds['screen_score']}</b></div>
      <div>🟢 roheline tabamuslävi: <b>{thresholds['green_hit_bar']}</b></div>
      <div>🟡 kollane tabamuslävi: <b>{thresholds['yellow_hit_bar']}</b></div>
      <div>🔴 punane tabamuslävi: <b>{thresholds['red_hit_bar']}</b></div>
    </div>
  </div>

  <div class="card">
    <h2>Soovituste ajalugu (uusimad enne)</h2>
    <div class="desc">🧪 EXPLORE = eksperimentaalne valik allpool tavalävendit, tehtud tahtlikult õppimise huvides.</div>
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Token</th><th>Skoor</th><th>Risk</th><th>24h</th><th>7p</th><th>Põhjendus</th></tr>
      {history_rows or '<tr><td colspan="7" style="color:var(--muted)">Veel andmeid pole.</td></tr>'}
    </table></div>
  </div>

  <div class="card">
    <h2>Käivituste logi</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Kandidaate</th><th>Alerte</th><th>Tagasivaade</th><th>Mudeli kohandused</th></tr>
      {run_rows or '<tr><td colspan="5" style="color:var(--muted)">Veel käivitusi pole.</td></tr>'}
    </table></div>
  </div>
</div>
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
