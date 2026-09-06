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
# This legacy Crypto.com sleeve no longer owns the site root. index.html is now
# a hand-written landing page (the moonshot + cryptobot balances and links to
# each), so rewriting it hourly from here would silently clobber that page.
# The legacy dashboard is still generated and reachable at /dashboard.html.
INDEX_PATH = None

HISTORY_KEEP = 48  # keep last 48 hourly snapshots per instrument (~2 days)
VOL_WINDOW = 24     # snapshots used for the rolling volatility estimate (~1 day at hourly cadence)
TREND_WINDOW = 6    # snapshots used for the OLS trend fit (~6h at hourly cadence)

# Experimental knob, NOT wired to any env var - production always runs at
# 1.0 (unchanged behavior). A live data review found trend_bonus's learned
# model weight had gone NEGATIVE over a full backtested year (confirmed
# "breakouts" predicting worse outcomes, not better - consistent with
# chasing exhaustion tops rather than genuine continuations). Rather than
# hand-editing the heuristic on a hunch, this multiplier lets an experiment
# script (see experiment_trend_bonus.py) test alternate hypotheses -
# disabling (0.0) or inverting (negative) the bonus's contribution to
# raw_score - validated on a held-out test period, not fit to the whole
# sample. Only ever changed by an experiment harness, never by engine.py
# itself.
TREND_BONUS_MULTIPLIER = 1.0

# Feature order matters - it's the order weights/x vectors are stored in.
FEATURE_NAMES = ["bias", "momentum", "trend_bonus", "liquidity", "is_meme", "volatility",
                 "hype_bonus", "alpha", "book_imbalance", "market_fng", "onchain_activity"]

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
    "book_imbalance": 0.5,  # prior: visible buy pressure in the order book supports follow-through
    "market_fng": 0.0,      # no prior - let the data decide what market mood is worth
    "onchain_activity": 0.0,  # no prior - BTC-only proxy signal, let the data decide what it's worth
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
PORTFOLIO_CORR_ALPHA_THRESHOLD = 0.02   # |alpha_pct| below this = "moves like BTC", not idiosyncratic
PORTFOLIO_CORR_MAX_LOW_ALPHA = 3        # at most 3/5 open slots may be low-alpha (highly BTC-correlated) at once -
                                         # otherwise "5 diversified positions" can secretly be one leveraged BTC bet
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

# Upside management. Old behavior was asymmetric: protected below (-8% stop)
# but blind above - a position could ride +15% and give it all back before
# the fixed 24h exit. Two mechanisms fix that:
#   - Partial take-profit: at +TAKE_PROFIT_PCT sell TAKE_PROFIT_FRACTION of
#     the position (bank real profit), let the rest keep running.
#   - Trailing stop: once the high-water mark is +TRAIL_ARM_PCT above entry,
#     the stop starts following TRAIL_DIST_PCT below the highest price seen,
#     locking in gains instead of waiting passively for the 24h bell.
# The 24h exit still closes whatever remains.
TAKE_PROFIT_PCT = 8.0
TAKE_PROFIT_FRACTION = 0.5
TRAIL_ARM_PCT = 4.0
TRAIL_DIST_PCT = 3.0

# ---------------------------------------------------------------------------
# Funding-rate arbitrage - a market-neutral sleeve, separate from the
# momentum book. Idea: go long spot + short the equivalent notional on the
# perpetual (delta-neutral - price moves roughly cancel between the two
# legs), and collect the funding payment perpetual shorts earn when funding
# is positive. This does not care whether the market goes up or down, so it
# is a genuinely different, uncorrelated return source from the momentum
# strategy - not "more of the same" trend-following.
# ---------------------------------------------------------------------------
FUNDING_ARB_START_BALANCE = 300.0   # deliberately smaller than the momentum
                                     # sleeve's $1000 - a real deployment would
                                     # split capital across strategies, not
                                     # give each one the full pot
FUNDING_ARB_MAX_OPEN = 3
FUNDING_ARB_POSITION_PCT = 0.25     # fraction of the funding-arb balance risked per pair (both legs use this notional)
FUNDING_MAX_HOLD_HOURS = 45 * 24    # 45 days

# All four legs (spot buy, perp short, spot sell, perp close) are placed as
# POST_ONLY maker orders - this sleeve is not time-critical, so it can
# afford to rest in the book and simply retry next hour if a leg doesn't
# fill. That choice is what makes the strategy viable at all, and the
# numbers are stark. Measured breakeven hold time, funding accruing on ONE
# leg's notional against FOUR legs of cost:
#     taker (0.5% fee + 0.15% slippage): 95d @10% APR, 47d @20% APR
#     maker (0.1% fee, no slippage):     15d @10% APR,  7d @20% APR
# At taker rates the trade cannot break even inside its own 45-day cap at
# ANY realistic funding rate - an earlier version of this sleeve was
# therefore mathematically incapable of profit, which is exactly the sort
# of thing that only shows up when you compute the breakeven instead of
# assuming it. At maker rates a 10% APR threshold breaks even in ~15 days,
# leaving a month of margin inside the cap.
#
# Threshold set from real measured data, not a guess: across ~2700 hours of
# funding history over 8 instruments, 18.5% of hours were at/above 10% APR
# while only 3.9% reached 20% - a 20% bar leaves the sleeve idle almost
# always, while 10% is both frequently reachable and comfortably profitable
# under maker economics.
FUNDING_MIN_APR_ENTER = 10.0
FUNDING_MIN_APR_EXIT = 1.0          # close if the rate has decayed below this - no longer worth the drag


def funding_arb_state_default():
    return {
        "starting_balance": FUNDING_ARB_START_BALANCE,
        "balance": FUNDING_ARB_START_BALANCE,
        "open_positions": [],   # [{id, instrument, spot_qty, spot_entry, perp_qty, perp_entry,
                                 #   notional_usd, entry_ts, entry_apr, funding_collected_usd, mode}]
        "closed_trades": [],
        "balance_history": [{"ts": 0, "balance": FUNDING_ARB_START_BALANCE}],
        "next_id": 1,
    }


def scan_funding_opportunities(state, funding_rates, current_prices, brk):
    """Open a delta-neutral pair (spot long + perp short) for any instrument
    whose recent funding rate pays enough (>= FUNDING_MIN_APR_ENTER
    annualized) to be worth the two-legged entry cost, while there's room
    and cash in the funding-arb sleeve. Independent of the momentum book -
    this never touches state["portfolio"]."""
    fa = state.setdefault("funding_arb", funding_arb_state_default())
    if state["killswitch"].get("active"):
        return []
    if len(fa["open_positions"]) >= FUNDING_ARB_MAX_OPEN:
        return []

    notes = []
    already_open = {p["instrument"] for p in fa["open_positions"]}
    candidates = sorted(
        ((inst, apr) for inst, apr in funding_rates.items()
         if apr >= FUNDING_MIN_APR_ENTER and inst not in already_open and inst in current_prices),
        key=lambda x: x[1], reverse=True)

    for inst, apr in candidates:
        if len(fa["open_positions"]) >= FUNDING_ARB_MAX_OPEN:
            break
        notional = fa["balance"] * FUNDING_ARB_POSITION_PCT
        if notional < 5.0 or notional > fa["balance"]:
            continue
        price = current_prices[inst]
        try:
            spot_fill = brk.buy(inst, notional, price, maker=True)
        except broker_mod.BrokerError as e:
            print(f"WARN: funding-arb spot leg open {inst} failed: {e}", file=sys.stderr)
            continue
        if not spot_fill:
            continue  # maker order didn't fill - retry next run, nothing committed
        try:
            perp_fill = brk.open_short(inst, notional, price, maker=True)
        except broker_mod.BrokerError as e:
            perp_fill = None
            print(f"WARN: funding-arb perp leg open {inst} failed ({e}) - unwinding the spot "
                  f"leg that already filled, so we don't leave an unhedged position", file=sys.stderr)
        if not perp_fill:
            # Spot leg is real (live mode) or a no-op fill (paper mode) - either
            # way, unwind it immediately rather than leaving a directional,
            # un-hedged spot position this sleeve never intended to hold.
            try:
                # Unwind as a TAKER: we need this filled now, not "maybe
                # next hour" - an unhedged position is exactly what we're
                # trying not to hold, so paying the spread to be rid of it
                # is the right trade.
                brk.sell(inst, spot_fill["quantity"], price)
            except broker_mod.BrokerError as e2:
                print(f"CRITICAL: funding-arb spot unwind FAILED for {inst} after the perp leg "
                      f"couldn't open: {e2} - a real unhedged spot position may be sitting on the "
                      f"exchange right now, check the account manually.", file=sys.stderr)
            continue

        fa["balance"] -= (notional + notional)  # both legs' notional leave the funding-arb cash pool
        rec_id = fa["next_id"]
        fa["next_id"] += 1
        fa["open_positions"].append({
            "id": rec_id, "instrument": inst,
            "spot_qty": spot_fill["quantity"], "spot_entry": spot_fill["price"],
            "perp_qty": perp_fill["quantity"], "perp_entry": perp_fill["price"],
            "notional_usd": notional, "entry_ts": now_ts(), "entry_apr": round(apr, 1),
            "funding_collected_usd": 0.0,
            "entry_fee_usd": round(spot_fill["fee_usd"] + perp_fill["fee_usd"], 4),
            "mode": brk.mode,
        })
        notes.append(f"💹 FUNDING-ARB AVATUD: {inst} (funding {apr:+.1f}% aastas, "
                    f"${notional:.2f} kummalgi jalal - spot pikk + perp lühike)")
    return notes


def manage_funding_positions(state, funding_rates, current_prices, brk):
    """Each run: accrue this hour's funding payment on every open pair
    (a short perpetual position earns funding when the rate is positive -
    approximated here as notional * current hourly rate, paid every run,
    which is how the exchange itself settles it), and close out (both legs)
    when the rate has decayed below FUNDING_MIN_APR_EXIT or the max hold
    time is reached."""
    fa = state.setdefault("funding_arb", funding_arb_state_default())
    notes = []
    for pos in list(fa["open_positions"]):
        inst = pos["instrument"]
        apr = funding_rates.get(inst)
        if apr is not None:
            hourly_rate = apr / 100 / 24 / 365
            funding_payment = pos["notional_usd"] * hourly_rate
            pos["funding_collected_usd"] = pos.get("funding_collected_usd", 0.0) + funding_payment
            fa["balance"] += funding_payment

        age_h = (now_ts() - pos["entry_ts"]) / 3600
        should_close = (apr is not None and apr < FUNDING_MIN_APR_EXIT) or age_h >= FUNDING_MAX_HOLD_HOURS
        if not should_close:
            continue
        price = current_prices.get(inst)
        if price is None:
            continue

        # Close the spot leg first, but track completion on the position
        # itself (pos["spot_closed"]) so a perp-leg failure AFTER the spot
        # already sold doesn't re-sell (nonexistent) spot again next run -
        # it just retries the remaining perp leg until that succeeds too.
        if not pos.get("spot_closed"):
            try:
                spot_fill = brk.sell(inst, pos["spot_qty"], price, maker=True)
            except broker_mod.BrokerError as e:
                print(f"WARN: funding-arb spot close {inst} failed: {e}", file=sys.stderr)
                continue
            if not spot_fill:
                continue  # maker exit didn't fill - position stays intact, retry next run
            pos["spot_closed"] = True
            pos["spot_exit_fill"] = spot_fill
        else:
            spot_fill = pos["spot_exit_fill"]

        try:
            # Taker on this leg: the spot side is already sold, so the pair
            # is unhedged until this completes - speed beats fee here.
            perp_fill = brk.close_short(inst, pos["perp_qty"], price)
        except broker_mod.BrokerError as e:
            print(f"WARN: funding-arb perp close {inst} failed - spot leg is ALREADY SOLD, "
                  f"position is temporarily unhedged, will retry the perp leg next run: {e}",
                  file=sys.stderr)
            continue
        if not perp_fill:
            continue

        fa["open_positions"].remove(pos)
        # Standard short P&L (profit when price fell) for the perp leg, plain
        # sell-vs-cost-basis for the spot leg. funding_collected_usd is NOT
        # added again here - it was already credited to cash incrementally,
        # run by run, in the accrual step above; re-adding it would double-count.
        spot_pnl_usd = spot_fill["proceeds_usd"] - pos["notional_usd"]
        short_pnl_usd = (pos["perp_entry"] - perp_fill["price"]) * pos["perp_qty"] - perp_fill["fee_usd"]
        basis_pnl_usd = spot_pnl_usd + short_pnl_usd
        pnl_usd = basis_pnl_usd + pos["funding_collected_usd"]  # for DISPLAY: total economic result of the trade
        pnl_pct = pnl_usd / (pos["notional_usd"] * 2) * 100 if pos["notional_usd"] else 0.0
        fa["balance"] += pos["notional_usd"] * 2 + basis_pnl_usd
        fa["closed_trades"].append({
            "id": pos["id"], "instrument": inst, "notional_usd": pos["notional_usd"],
            "entry_ts": pos["entry_ts"], "exit_ts": now_ts(), "entry_apr": pos["entry_apr"],
            "funding_collected_usd": round(pos["funding_collected_usd"], 4),
            "basis_pnl_usd": round(basis_pnl_usd, 2),
            "pnl_usd": round(pnl_usd, 2), "pnl_pct": round(pnl_pct, 2), "mode": pos.get("mode", "paper"),
        })
        fa["closed_trades"] = fa["closed_trades"][-300:]
        reason = "funding kahanes alla läve" if (apr is not None and apr < FUNDING_MIN_APR_EXIT) else "max hoiuaeg täis"
        notes.append(f"💹 FUNDING-ARB SULETUD: {inst} ({reason}) - kogutud funding "
                    f"${pos['funding_collected_usd']:+.2f}, netotulem {pnl_pct:+.1f}% (${pnl_usd:+.2f})")

    fa["balance_history"].append({"ts": now_ts(), "balance": round(fa["balance"], 2)})
    fa["balance_history"] = fa["balance_history"][-500:]
    return notes


# ---------------------------------------------------------------------------
# Mean-reversion / grid sleeve - a second, deliberately DIFFERENT return
# source from momentum. Momentum only trades when trend_bonus confirms a
# real breakout (strong R²) - meaning most hours, on most instruments, it
# does nothing (the market is just chopping sideways, which is most of the
# time in crypto). This sleeve does the opposite: weak trend + price near
# the bottom of its recent range on a small set of liquid majors -> buy the
# dip, take a modest fast profit near the top of the range. Classic grid
# trading, monetizing the sideways hours momentum has to sit out. Separate
# ledger, separate capital - never touches state["portfolio"].
# ---------------------------------------------------------------------------
GRID_START_BALANCE = 300.0
GRID_INSTRUMENTS = ["BTCUSD", "ETHUSD"]   # liquid majors only - grid needs tight spreads to survive fees
GRID_MAX_OPEN = 3
GRID_POSITION_PCT = 0.20
GRID_LOWER_BAND_PCT = 0.35    # buy when price sits in the bottom 35% of its recent range
GRID_UPPER_BAND_PCT = 0.75    # take profit once price reaches the top 25% of the range
GRID_MAX_TREND_R2 = 0.35      # only trade when there's NO strong confirmed trend (above this is momentum's territory)
GRID_TAKE_PROFIT_PCT = 3.0
GRID_STOP_LOSS_PCT = 6.0      # protective stop if the range breaks down instead of reverting


def grid_state_default():
    return {
        "starting_balance": GRID_START_BALANCE,
        "balance": GRID_START_BALANCE,
        "open_positions": [],
        "closed_trades": [],
        "balance_history": [{"ts": 0, "balance": GRID_START_BALANCE}],
        "next_id": 1,
    }


def range_signal(candles):
    """(position_in_range 0..1, trend_r2) from recent 15m candles - the grid
    sleeve's entry test. None if there isn't enough history yet."""
    recent = [c for c in candles if c.get("c")][-96:]
    if len(recent) < 20:
        return None
    closes = [c["c"] for c in recent]
    highs = [c.get("h", c["c"]) for c in recent]
    lows = [c.get("l", c["c"]) for c in recent]
    range_high, range_low = max(highs), min(lows)
    if range_high <= range_low:
        return None
    position = (closes[-1] - range_low) / (range_high - range_low)
    p0 = closes[0] if closes[0] else 1e-9
    norm = [(p - p0) / p0 for p in closes]
    _, r2 = linreg_slope_r2(norm)
    return clamp(position, 0, 1), r2


def manage_grid(state, current_prices, candles_all, brk):
    """One pass per run: check exits/stops on open grid positions, then look
    for new range-bound entries on GRID_INSTRUMENTS. Gated by the same
    kill-switch as momentum (an account-level circuit breaker should stop
    ALL new risk, not just one sleeve)."""
    grid = state.setdefault("grid", grid_state_default())
    notes = []

    for pos in list(grid["open_positions"]):
        inst = pos["instrument"]
        cp = current_prices.get(inst)
        if cp is None:
            continue
        cd = candles_all.get(inst) or []
        hour_low = min((c["l"] for c in cd[-4:] if c.get("l")), default=cp)
        take_profit_price = pos["entry_price"] * (1 + GRID_TAKE_PROFIT_PCT / 100)
        hit_stop = hour_low <= pos["stop_price"] or cp <= pos["stop_price"]
        hit_take_profit = cp >= take_profit_price
        sig = range_signal(cd) if cd else None
        hit_upper_band = sig is not None and sig[0] >= GRID_UPPER_BAND_PCT
        if not (hit_stop or hit_take_profit or hit_upper_band):
            continue
        sell_price = min(cp, pos["stop_price"]) if hit_stop else cp
        try:
            fill = brk.sell(inst, pos["quantity"], sell_price)
        except broker_mod.BrokerError as e:
            print(f"WARN: grid sell {inst} failed: {e}", file=sys.stderr)
            continue
        if not fill:
            continue
        grid["open_positions"].remove(pos)
        pnl_usd = fill["proceeds_usd"] - pos["size_usd"]
        pnl_pct = pnl_usd / pos["size_usd"] * 100 if pos["size_usd"] else 0.0
        grid["balance"] += fill["proceeds_usd"]
        reason = "stop_loss" if hit_stop else ("take_profit" if hit_take_profit else "upper_band")
        grid["closed_trades"].append({
            "id": pos["id"], "instrument": inst, "entry_price": pos["entry_price"],
            "exit_price": fill["price"], "size_usd": pos["size_usd"], "entry_ts": pos["entry_ts"],
            "exit_ts": now_ts(), "pnl_usd": round(pnl_usd, 2), "pnl_pct": round(pnl_pct, 2),
            "exit_reason": reason, "mode": pos.get("mode", "paper"),
        })
        grid["closed_trades"] = grid["closed_trades"][-300:]
        notes.append(f"🔲 GRID SULETUD: {inst} ({reason}) {pnl_pct:+.1f}% (${pnl_usd:+.2f})")

    if not state["killswitch"].get("active") and len(grid["open_positions"]) < GRID_MAX_OPEN:
        already_open = {p["instrument"] for p in grid["open_positions"]}
        for inst in GRID_INSTRUMENTS:
            if len(grid["open_positions"]) >= GRID_MAX_OPEN or inst in already_open:
                continue
            cp = current_prices.get(inst)
            cd = candles_all.get(inst)
            if cp is None or not cd:
                continue
            sig = range_signal(cd)
            if sig is None:
                continue
            position, r2 = sig
            if position > GRID_LOWER_BAND_PCT or r2 > GRID_MAX_TREND_R2:
                continue  # not near the bottom of its range, or a real trend is running - momentum's job, not grid's
            size_usd = grid["balance"] * GRID_POSITION_PCT
            if size_usd < 5.0 or size_usd > grid["balance"]:
                continue
            try:
                fill = brk.buy(inst, size_usd, cp)
            except broker_mod.BrokerError as e:
                print(f"WARN: grid buy {inst} failed: {e}", file=sys.stderr)
                continue
            if not fill:
                continue
            grid["balance"] -= size_usd
            rec_id = grid["next_id"]
            grid["next_id"] += 1
            grid["open_positions"].append({
                "id": rec_id, "instrument": inst, "entry_price": fill["price"],
                "quantity": fill["quantity"], "size_usd": round(size_usd, 2),
                "entry_fee_usd": fill["fee_usd"], "entry_ts": now_ts(),
                "stop_price": fill["price"] * (1 - GRID_STOP_LOSS_PCT / 100), "mode": brk.mode,
            })
            notes.append(f"🔲 GRID AVATUD: {inst} vahemiku põhjas (positsioon {position:.0%}, R²={r2:.2f}) - ${size_usd:.2f}")

    grid["balance_history"].append({"ts": now_ts(), "balance": round(portfolio_equity(grid, current_prices), 2)})
    grid["balance_history"] = grid["balance_history"][-500:]
    return notes


# ---------------------------------------------------------------------------
# Regime-gated mean-reversion SHORT sleeve.
#
# Built in direct response to a live data review that asked "our hit rate is
# only 12.5% - wouldn't reversing every signal give >80% accuracy?" Checked
# against the full 184-record history: reversing EVERY long signal gives a
# raw edge of only +0.40%/trade (184 records: 12.5% would have hit long,
# 22.3% would have hit short at a symmetric -3% bar, the other 65% moved
# less than 3% either way) - nowhere near enough to clear the ~1.3%
# round-trip cost of a real buy+sell cycle. A BLIND full reversal does not
# survive contact with real trading costs.
#
# So this is deliberately narrower than "flip the signal": it only shorts
# when TWO conditions both hold -
#   (a) the broader market is confirmed risk-off (BTC itself down over the
#       window AND Fear&Greed in fear territory) - shorting into strength
#       during a genuine uptrend is exactly the "catching a falling knife
#       in reverse" mistake the long side already makes;
#   (b) the SPECIFIC candidate looks overextended - high momentum (it
#       pumped) but a WEAK confirmed trend (trend_bonus <= 5, i.e. the
#       existing scoring already couldn't confirm a genuine breakout) -
#       a spike more likely to be exhaustion than continuation.
# Separate ledger, separate small capital, gated by the same kill-switch -
# this is an experimental sleeve to validate, not a replacement strategy.
# ---------------------------------------------------------------------------
SHORT_START_BALANCE = 200.0
SHORT_MAX_OPEN = 2
SHORT_POSITION_PCT = 0.15
SHORT_STOP_LOSS_PCT = 6.0
SHORT_TAKE_PROFIT_PCT = 5.0
SHORT_MAX_HOLD_HOURS = 24
SHORT_MIN_MOMENTUM_SCORE = 60   # candidate must have genuinely pumped (high momentum score)
SHORT_MAX_TREND_BONUS = 5.0     # but NOT have a strongly confirmed trend (that's the long side's territory)
SHORT_BTC_BEAR_CHANGE_PCT = -3.0   # BTC's own 24h change must be at/below this
SHORT_FNG_FEAR_MAX = 40            # Fear & Greed at/below this (fear territory)


def short_state_default():
    return {
        "starting_balance": SHORT_START_BALANCE,
        "balance": SHORT_START_BALANCE,
        "open_positions": [],   # [{id, instrument, quantity, entry_price, size_usd, entry_ts, stop_price, entry_fee_usd, mode}]
        "closed_trades": [],
        "balance_history": [{"ts": 0, "balance": SHORT_START_BALANCE}],
        "next_id": 1,
    }


def short_sleeve_equity(sh, current_prices):
    """Unlike portfolio_equity() (built for LONG holdings, where quantity*
    price is the payoff), a short's mark-to-market value is the reserved
    capital plus/minus its unrealized P&L (positive when price has fallen
    below entry, negative when it has risen) - reusing the long-side
    formula here would silently double-count the position's notional."""
    equity = sh["balance"]
    for p in sh["open_positions"]:
        price = current_prices.get(p["instrument"], p["entry_price"])
        unrealized_pnl = (p["entry_price"] - price) * p["quantity"]
        equity += p["size_usd"] + unrealized_pnl
    return equity


def market_bearish_regime(btc_change_24h, fng_value):
    """The regime gate: is the BROADER market confirmed risk-off right now?
    Both conditions must hold - a single fearful FNG reading during an
    otherwise-fine market, or a BTC dip without genuine fear, isn't enough."""
    if btc_change_24h is None or fng_value is None:
        return False
    return btc_change_24h <= SHORT_BTC_BEAR_CHANGE_PCT / 100 and fng_value <= SHORT_FNG_FEAR_MAX


def manage_short_sleeve(state, candidates, btc_change_24h, fng_value, current_prices, brk):
    """One pass per run: check exits on open shorts, then look for new
    overextended-pump-in-a-bearish-regime entries among this run's already-
    scored candidates (no separate scan needed - reuses screen()'s output)."""
    sh = state.setdefault("short_reversal", short_state_default())
    notes = []

    for pos in list(sh["open_positions"]):
        inst = pos["instrument"]
        cp = current_prices.get(inst)
        if cp is None:
            continue
        age_h = (now_ts() - pos["entry_ts"]) / 3600
        hit_stop = cp >= pos["stop_price"]
        hit_take_profit = cp <= pos["entry_price"] * (1 - SHORT_TAKE_PROFIT_PCT / 100)
        hit_max_hold = age_h >= SHORT_MAX_HOLD_HOURS
        if not (hit_stop or hit_take_profit or hit_max_hold):
            continue
        try:
            fill = brk.close_short(inst, pos["quantity"], cp)
        except broker_mod.BrokerError as e:
            print(f"WARN: short-sleeve close {inst} failed: {e}", file=sys.stderr)
            continue
        if not fill:
            continue
        sh["open_positions"].remove(pos)
        # Standard short P&L: profit when price fell below entry.
        pnl_usd = (pos["entry_price"] - fill["price"]) * pos["quantity"] - fill["fee_usd"]
        pnl_pct = pnl_usd / pos["size_usd"] * 100 if pos["size_usd"] else 0.0
        sh["balance"] += pos["size_usd"] + pnl_usd
        reason = "stop_loss" if hit_stop else ("take_profit" if hit_take_profit else "max_hold")
        sh["closed_trades"].append({
            "id": pos["id"], "instrument": inst, "entry_price": pos["entry_price"],
            "exit_price": fill["price"], "size_usd": pos["size_usd"], "entry_ts": pos["entry_ts"],
            "exit_ts": now_ts(), "pnl_usd": round(pnl_usd, 2), "pnl_pct": round(pnl_pct, 2),
            "exit_reason": reason, "mode": pos.get("mode", "paper"),
        })
        sh["closed_trades"] = sh["closed_trades"][-300:]
        notes.append(f"📉 SHORT SULETUD: {inst} ({reason}) {pnl_pct:+.1f}% (${pnl_usd:+.2f})")

    regime_ok = market_bearish_regime(btc_change_24h, fng_value)
    if regime_ok and not state["killswitch"].get("active") and len(sh["open_positions"]) < SHORT_MAX_OPEN:
        already_open = {p["instrument"] for p in sh["open_positions"]}
        ranked = sorted(
            (c for c in candidates
             if c["instrument"] not in already_open
             and c["momentum_score"] >= SHORT_MIN_MOMENTUM_SCORE
             and c["trend_bonus"] <= SHORT_MAX_TREND_BONUS
             and c["change_24h"] > 0
             and c["instrument"] in current_prices),
            key=lambda c: c["momentum_score"], reverse=True)
        for c in ranked:
            if len(sh["open_positions"]) >= SHORT_MAX_OPEN:
                break
            inst = c["instrument"]
            size_usd = sh["balance"] * SHORT_POSITION_PCT
            if size_usd < 5.0 or size_usd > sh["balance"]:
                continue
            price = current_prices[inst]
            try:
                fill = brk.open_short(inst, size_usd, price)
            except broker_mod.BrokerError as e:
                print(f"WARN: short-sleeve open {inst} failed: {e}", file=sys.stderr)
                continue
            if not fill:
                continue
            sh["balance"] -= size_usd
            rec_id = sh["next_id"]
            sh["next_id"] += 1
            sh["open_positions"].append({
                "id": rec_id, "instrument": inst, "entry_price": fill["price"],
                "quantity": fill["quantity"], "size_usd": round(size_usd, 2),
                "entry_fee_usd": fill["fee_usd"], "entry_ts": now_ts(),
                "stop_price": fill["price"] * (1 + SHORT_STOP_LOSS_PCT / 100), "mode": brk.mode,
            })
            notes.append(f"📉 SHORT AVATUD: {inst} üleostetud (momentum {c['momentum_score']:.0f}, "
                        f"trend nõrk R²-ga kinnitamata) turu hirmu-režiimis - ${size_usd:.2f}")

    sh["balance_history"].append({"ts": now_ts(), "balance": round(short_sleeve_equity(sh, current_prices), 2)})
    sh["balance_history"] = sh["balance_history"][-500:]
    return notes


# Go-live readiness criteria (mirrors RUNBOOK.md "Go-live checklist" exactly -
# this is a REPORTING feature only, never touches TRADING_MODE itself. The
# switch to live money stays a deliberate manual step on GitHub; this just
# tells you, honestly, when the paper track record has earned it.
GO_LIVE_MIN_PROFIT_FACTOR = 1.3
GO_LIVE_MAX_DRAWDOWN_PCT = 15.0
GO_LIVE_MIN_CLOSED_TRADES = 60
GO_LIVE_MIN_EXPECTANCY_PCT = 0.0

DEFAULT_STATE = {
    "history": {},          # instrument -> [ {ts, price, change_24h, volume_value}, ... ]
    "alerted": {},          # instrument -> {last_score, last_ts, last_risk}
    "thresholds": {         # adaptive, per risk bucket, tuned by adapt_thresholds()
        "screen_score": 65,
        "green_hit_bar": 60,
        "yellow_hit_bar": 65,
        "red_hit_bar": 75,
        "screen_score_raise_streak": 0,  # consecutive stuck-below-35%-hit-rate raise cycles - see SCREEN_SCORE_STAGNANT_STREAK_LIMIT
    },
    "adjustment_checkpoint": 0,  # len(completed) at last threshold adjustment - prevents re-applying
                                 # the same historical evidence over and over on every run (see adapt_thresholds)
    "model": {
        "weights": dict(INITIAL_WEIGHTS),
        "n_updates": 0,
        "last_full_retrain_n": 0,   # resolved-outcome count at the last full batch retrain
        "learning_log": []   # [{ts, text}], plain-language "what I just learned"
    },
    "pending_followups": [],   # awaiting 24h and/or 7d outcome check
    "completed": [],          # followups fully resolved (both checks done or expired)
    "go_live_alert_sent": False,      # one-time celebratory Telegram alert, fires once when all criteria first met
    "go_live_last_progress_day": "",  # UTC date string - throttles the daily progress line to once/day
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
    st["model"].setdefault("last_full_retrain_n", 0)
    st["model"].setdefault("learning_log", [])
    st.setdefault("thresholds", _deep_default(DEFAULT_STATE["thresholds"]))
    for k, v in DEFAULT_STATE["thresholds"].items():
        st["thresholds"].setdefault(k, v)
    st.setdefault("portfolio", _deep_default(DEFAULT_STATE["portfolio"]))
    for k, v in DEFAULT_STATE["portfolio"].items():
        st["portfolio"].setdefault(k, v)
    st.setdefault("killswitch", _deep_default(DEFAULT_STATE["killswitch"]))
    for k, v in DEFAULT_STATE["killswitch"].items():
        st["killswitch"].setdefault(k, v)
    st.setdefault("go_live_alert_sent", False)
    st.setdefault("go_live_last_progress_day", "")
    st.setdefault("funding_arb", funding_arb_state_default())
    for k, v in funding_arb_state_default().items():
        st["funding_arb"].setdefault(k, v)
    st.setdefault("grid", grid_state_default())
    for k, v in grid_state_default().items():
        st["grid"].setdefault(k, v)
    st.setdefault("short_reversal", short_state_default())
    for k, v in short_state_default().items():
        st["short_reversal"].setdefault(k, v)
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


def build_features(momentum_score_, trend_bonus_, liquidity, category, change_24h, hype_bonus,
                   alpha_val=0.0, book_imbalance=0.0, fng_value=None, onchain_ratio=None):
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
        # order book notional imbalance -1..+1 -> 0..1 (0.5 = balanced/unknown)
        "book_imbalance": clamp((book_imbalance + 1) / 2, 0, 1),
        # Fear & Greed 0..100 -> 0..1 (0.5 = neutral/unknown)
        "market_fng": (fng_value / 100.0) if fng_value is not None else 0.5,
        # BTC on-chain activity vs its own 7d average, ratio 0.5x..2x -> 0..1 (0.5 = normal/unknown)
        "onchain_activity": clamp((onchain_ratio - 0.5) / 1.5, 0, 1) if onchain_ratio is not None else 0.5,
    }


def model_predict(weights, features):
    z = sum(weights.get(k, 0.0) * features.get(k, 0.0) for k in FEATURE_NAMES)
    return sigmoid(z)


# A trade that missed by a mile (or won by a mile) should teach the model
# more than one that barely crossed or barely missed the +3% hit bar - the
# old scheme weighted a +3.01% "hit" the same as a +28% "hit", and a -0.5%
# "miss" the same as a -10% "miss", which is exactly the "optimizes for
# quota, not profit" gap a live data review flagged. OUTCOME_SCALE_PCT is
# the |return_pct| that counts as "a normal, decisive outcome" (weight
# 1.0); the clamp keeps one freak 28% mover from swamping everything else.
OUTCOME_SCALE_PCT = 5.0
OUTCOME_WEIGHT_MIN = 0.3
OUTCOME_WEIGHT_MAX = 3.0


def outcome_magnitude_weight(return_pct):
    if return_pct is None:
        return 1.0
    return clamp(abs(return_pct) / OUTCOME_SCALE_PCT, OUTCOME_WEIGHT_MIN, OUTCOME_WEIGHT_MAX)


def train_step(state, features, hit, return_pct=None):
    """One step of online logistic-regression training (gradient ascent on
    log-likelihood), scaled by outcome_magnitude_weight() so decisive
    trades move the weights more than marginal ones. Logs a plain-language
    note if weights moved meaningfully."""
    weights = state["model"]["weights"]
    old_weights = dict(weights)
    p = model_predict(weights, features)
    y = 1.0 if hit else 0.0
    error = y - p
    lr = LEARNING_RATE * outcome_magnitude_weight(return_pct)
    for k in FEATURE_NAMES:
        reg = 0.0 if k == "bias" else L2_LAMBDA * weights[k]  # don't regularize the bias term
        weights[k] = weights[k] + lr * (error * features.get(k, 0.0) - reg)
    state["model"]["n_updates"] += 1

    moved = {k: weights[k] - old_weights[k] for k in FEATURE_NAMES if abs(weights[k] - old_weights[k]) >= 0.02}
    if moved:
        biggest = max(moved, key=lambda k: abs(moved[k]))
        direction = "tugevam" if moved[biggest] > 0 else "nõrgem"
        note = (f"Tulemus ({'tabas' if hit else 'ei tabanud'}, ennustus oli {p:.0%}) muutis kaalu "
                f"'{biggest}' {direction}maks ({old_weights[biggest]:.2f} -> {weights[biggest]:.2f}).")
        state["model"]["learning_log"].append({"ts": now_ts(), "text": note})
        state["model"]["learning_log"] = state["model"]["learning_log"][-80:]


RETRAIN_EVERY = 20     # full batch retrain after this many NEW resolved outcomes
RETRAIN_EPOCHS = 150


def full_retrain(state):
    """Periodic full batch retraining alongside the per-outcome online steps.

    Online learning (one gradient step per result, in arrival order) is
    cheap but noisy: with less than ~100 samples the weights depend on the
    ORDER outcomes happened to arrive in, and one weird streak can drag
    them somewhere a full look at the data wouldn't. So after every
    RETRAIN_EVERY new resolved outcomes, refit from the prior on the ENTIRE
    resolved history: multiple shuffled epochs (fixed seed, deterministic)
    with a decaying learning rate and the same L2 penalty. The batch result
    replaces the online-accumulated weights; online learning then continues
    from this better-calibrated base until the next full retrain."""
    rows = [(r["features"], 1.0 if r["result_24h"]["hit"] else 0.0, r["result_24h"].get("return_pct"))
            for r in state["completed"] + state["pending_followups"]
            if r.get("result_24h") and r.get("features")]
    n = len(rows)
    if n < RETRAIN_EVERY or n - state["model"].get("last_full_retrain_n", 0) < RETRAIN_EVERY:
        return None

    weights = dict(INITIAL_WEIGHTS)
    rng = random.Random(42)
    order = list(range(n))
    for epoch in range(RETRAIN_EPOCHS):
        rng.shuffle(order)
        base_lr = LEARNING_RATE * (1 - 0.9 * epoch / RETRAIN_EPOCHS)  # decay to 10%
        for i in order:
            feats, y, ret = rows[i]
            p = model_predict(weights, feats)
            error = y - p
            lr = base_lr * outcome_magnitude_weight(ret)
            for k in FEATURE_NAMES:
                reg = 0.0 if k == "bias" else L2_LAMBDA * weights[k]
                weights[k] = weights[k] + lr * (error * feats.get(k, 0.0) - reg)

    correct = sum(1 for feats, y, _ret in rows
                  if (model_predict(weights, feats) >= 0.5) == (y == 1.0))
    old_weights = state["model"]["weights"]
    biggest = max(FEATURE_NAMES, key=lambda k: abs(weights.get(k, 0) - old_weights.get(k, 0)))
    state["model"]["weights"] = weights
    state["model"]["last_full_retrain_n"] = n
    note = (f"Täis-treening: mudel treeniti nullist uuesti kogu {n} tulemuse peal "
            f"({RETRAIN_EPOCHS} epohhi, segatud järjekord) - stabiilsemad kaalud kui "
            f"ükshaaval õppides. Treeningtäpsus {correct / n:.0%}, suurim muutus "
            f"'{biggest}' ({old_weights.get(biggest, 0):+.2f} -> {weights.get(biggest, 0):+.2f}).")
    state["model"]["learning_log"].append({"ts": now_ts(), "text": note})
    state["model"]["learning_log"] = state["model"]["learning_log"][-80:]
    return note


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
    (f* <= 0) we fall back to the flat base_pct rather than sizing to zero -
    an alert that clears every other bar still deserves a baseline look.

    IMPORTANT: the breakeven win probability is 1/(1+b), NOT 0.5 - an
    earlier version of this function hard-gated on win_prob <= 0.5, which
    is only correct for an even-money payoff (b=1). With this strategy's
    real payoff ratio (avg win well above avg loss, since losses are capped
    by the stop-loss but wins can run), the true breakeven sits well below
    50% - a live check found breakeven at ~23% against a payoff ratio of
    ~3.3, while the model's best-scoring tercile of candidates has a real
    empirical hit rate of ~26%. The hard 0.5 gate meant Kelly sizing NEVER
    activated - every position used the flat base_pct regardless of how
    much the model's own ranking favored it. f_star's own sign is already
    the correct, complete gate; do not add a redundant win_prob threshold
    on top of it."""
    if b <= 0:
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
                        category="unknown", win_prob=None, alpha_pct=0.0):
    """Called when a real alert fires. Opens a position THROUGH THE BROKER
    (paper simulation or real exchange order - same code path) if there's
    room (overall cap, per-category diversification cap, AND correlation
    cap), cash, and the kill-switch is not engaged. Sized via fractional
    Kelly once the model has earned enough evidence (win_prob passed in);
    otherwise the flat PORTFOLIO_POSITION_PCT. A stop-loss is attached
    immediately: a real resting order in live mode, a simulated per-run
    check in paper mode."""
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
    # Correlation cap: most altcoins are just leveraged BTC-beta (see
    # alpha_bonus) - without this check, "5 diversified positions" can
    # quietly be one oversized BTC bet if all 5 happen to be low-alpha.
    # Only meaningfully idiosyncratic moves (|alpha| above the threshold)
    # count as "independent" for this check.
    if abs(alpha_pct) < PORTFOLIO_CORR_ALPHA_THRESHOLD:
        low_alpha_open = sum(1 for p in pf["open_positions"]
                             if abs(p.get("alpha_pct", 0.0)) < PORTFOLIO_CORR_ALPHA_THRESHOLD)
        if low_alpha_open >= PORTFOLIO_CORR_MAX_LOW_ALPHA:
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
        "mode": brk.mode, "stop_price": stop_price, "stop_order_id": stop_order_id,
        "alpha_pct": alpha_pct
    })
    return True


def _book_close(state, match, exit_price, proceeds_usd, exit_fee_usd, exit_ts, reason):
    """Shared bookkeeping for every way a position can close (24h timer,
    stop-loss, reconcile). P&L is NET of both sides' fees + slippage - the
    number that would actually land in the account.

    Does NOT touch balance_history - that used to append raw pf["balance"]
    (cash) right here, which understates true account value mid-run
    whenever OTHER positions are still open (their value isn't in cash yet).
    record_equity_snapshot() is the single place balance_history gets
    written now, once per run, after all of a run's closes are settled."""
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
                        current_prices, category="unknown", win_prob=None, alpha_pct=0.0):
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
                                 category=category, win_prob=win_prob, alpha_pct=alpha_pct)
    note = (f"🔄 VAHETUS: {weakest['instrument']} (skoor {weakest['score']}, {old_trade['pnl_pct']:+.1f}%) "
            f"→ {instrument} (skoor {score})")
    if not opened:
        note += " - uue ost ebaõnnestus, koht jäi vabaks"
    return note, opened


def _book_partial_close(state, pos, fraction, fill, exit_ts, reason):
    """Realize a FRACTION of a position (partial take-profit): the sold part
    becomes its own closed trade, the position shrinks proportionally and
    stays open. P&L for the sold part is net of both sides' fees."""
    pf = state["portfolio"]
    part_size = round(pos["size_usd"] * fraction, 2)
    pnl_usd = fill["proceeds_usd"] - part_size
    pnl_pct = pnl_usd / part_size * 100 if part_size else 0.0
    pf["balance"] += fill["proceeds_usd"]
    pf["closed_trades"].append({
        "id": pos["id"], "instrument": pos["instrument"], "entry_price": pos["entry_price"],
        "exit_price": fill["price"], "size_usd": part_size, "entry_ts": pos["entry_ts"],
        "exit_ts": exit_ts, "pnl_usd": round(pnl_usd, 2), "pnl_pct": round(pnl_pct, 2),
        "entry_fee_usd": round(pos.get("entry_fee_usd", 0.0) * fraction, 4),
        "exit_fee_usd": round(fill["fee_usd"], 4),
        "exit_reason": reason, "mode": pos.get("mode", "paper"), "partial": True,
        "risk": pos["risk"], "score": pos["score"], "category": pos.get("category", "unknown")
    })
    pf["closed_trades"] = pf["closed_trades"][-300:]
    remain = 1 - fraction
    pos["quantity"] *= remain
    pos["size_usd"] = round(pos["size_usd"] * remain, 2)
    pos["entry_fee_usd"] = round(pos.get("entry_fee_usd", 0.0) * remain, 4)
    pos["partial_taken"] = True


def manage_positions(state, current_prices, candles_all, brk):
    """Once per run, for every open position, in this order:
      1. update the high-water mark (uses the last hour's candle HIGH when
         available, not just the once-an-hour snapshot price);
      2. partial take-profit at +TAKE_PROFIT_PCT (sell half, bank it);
      3. trailing stop: once armed (+TRAIL_ARM_PCT), raise the stop to
         TRAIL_DIST_PCT below the high-water mark - never lowered;
      4. stop trigger check - paper simulates (candle-LOW aware, so an
         intra-hour dip through the stop counts); live checks/replaces the
         real resting order on the exchange.
    Returns notes for the summary message."""
    pf = state["portfolio"]
    notes = []
    for pos in list(pf["open_positions"]):
        if "quantity" not in pos:
            continue  # legacy pre-broker position - only the 24h timer applies
        inst = pos["instrument"]
        cp = current_prices.get(inst)
        cd = candles_all.get(inst) or []
        hour_high = max((c["h"] for c in cd[-4:] if c.get("h")), default=None)
        hour_low = min((c["l"] for c in cd[-4:] if c.get("l")), default=None)

        highs = [x for x in (cp, hour_high) if x is not None]
        if highs:
            pos["high_water"] = max(pos.get("high_water", pos["entry_price"]), *highs)
        entry = pos["entry_price"]
        hw = pos.get("high_water", entry)
        stop_needs_replace = False

        # 2) partial take-profit
        if cp and not pos.get("partial_taken"):
            unreal_pct = (cp - entry) / entry * 100
            if unreal_pct >= TAKE_PROFIT_PCT:
                sell_qty = pos["quantity"] * TAKE_PROFIT_FRACTION
                try:
                    fill = brk.sell(inst, sell_qty, cp)
                except broker_mod.BrokerError as e:
                    fill = None
                    print(f"WARN: take-profit sell {inst} failed: {e}", file=sys.stderr)
                if fill:
                    _book_partial_close(state, pos, TAKE_PROFIT_FRACTION, fill, now_ts(), "take_profit")
                    t = pf["closed_trades"][-1]
                    notes.append(f"💰 KASUMIVÕTT: {inst} pool positsioonist müüdud {t['pnl_pct']:+.1f}% (${t['pnl_usd']:+.2f}), teine pool jookseb edasi")
                    stop_needs_replace = True  # resting stop covers the old, larger quantity

        # 3) trailing stop (never moves down)
        raised_this_run = False
        if hw >= entry * (1 + TRAIL_ARM_PCT / 100):
            new_stop = hw * (1 - TRAIL_DIST_PCT / 100)
            if new_stop > pos.get("stop_price", 0):
                pos["stop_price"] = new_stop
                pos["trailing"] = True
                stop_needs_replace = True
                raised_this_run = True

        if stop_needs_replace and brk.mode == "live":
            if pos.get("stop_order_id"):
                brk.cancel_order(inst, pos["stop_order_id"])
            pos["stop_order_id"] = brk.place_stop_loss(inst, pos["quantity"], pos["stop_price"])

        # 4) stop trigger check. Paper mode counts the intra-hour candle LOW
        # as a trigger too (a dip through the stop mid-hour is a real fill) -
        # EXCEPT against a stop that was only just raised this run: that
        # hour's low may well have happened BEFORE the high that raised the
        # stop, and we can't know the order within the hour. A freshly
        # raised stop is only compared against the current price; from the
        # next run onward the candle low counts normally.
        if raised_this_run:
            lows = [cp] if cp is not None else []
        else:
            lows = [x for x in (cp, hour_low) if x is not None]
        trigger_price = min(lows) if lows else None
        fill = brk.check_stop_order(pos, trigger_price)
        if fill:
            reason = "trailing_stop" if pos.get("trailing") and pos["stop_price"] > entry else "stop_loss"
            _book_close(state, pos, fill["price"], fill["proceeds_usd"], fill["fee_usd"],
                        now_ts(), reason)
            t = pf["closed_trades"][-1]
            label = "📈🛑 TRAILING-STOP (kasum lukus)" if reason == "trailing_stop" else "🛑 STOP-LOSS"
            notes.append(f"{label}: {inst} suleti {t['pnl_pct']:+.1f}% (${t['pnl_usd']:+.2f})")
    return notes


def portfolio_equity(pf, current_prices):
    """Total account value: cash + mark-to-market value of every open
    position. Buying a position immediately deducts its cost from cash -
    that capital is DEPLOYED, not lost. Any check that cares about real
    gains/losses (kill-switch, drawdown) must use this, not raw pf["balance"]
    - otherwise opening positions during a normal trading day looks
    indistinguishable from an actual loss."""
    equity = pf["balance"]
    for p in pf["open_positions"]:
        if "quantity" not in p:
            continue  # legacy pre-broker position, no quantity tracked
        price = current_prices.get(p["instrument"], p["entry_price"])
        equity += p["quantity"] * price
    return equity


def record_equity_snapshot(state, current_prices):
    """Append one equity data point per run (in addition to the points
    trade-closes already append via _book_close/_book_partial_close). This
    keeps the balance curve, drawdown, and kill-switch check accurate even
    during quiet hours with no closes, and - critically - means the
    kill-switch's 24h comparison is always measuring real account value,
    not a cash balance that dips the instant a position opens."""
    pf = state["portfolio"]
    equity = portfolio_equity(pf, current_prices)
    pf["balance_history"].append({"ts": now_ts(), "balance": round(equity, 2)})
    pf["balance_history"] = pf["balance_history"][-500:]


def update_killswitch(state, brk):
    """The circuit breaker. Engages (stops new opens) when:
      - TOTAL EQUITY (cash + open positions' current value) dropped more
        than MAX_DAILY_LOSS_PCT in ~24h, or
      - MAX_CONSEC_FAILURES orders in a row failed (live API misbehaving).
    Must run AFTER record_equity_snapshot() in the same call, so the latest
    balance_history entry is this run's true equity, not stale cash.
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
    current_equity = pf["balance_history"][-1]["balance"] if pf["balance_history"] else pf["balance"]
    if baseline > 0:
        daily_loss_pct = (baseline - current_equity) / baseline * 100
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


def go_live_readiness(state):
    """Check the RUNBOOK go-live criteria against the current paper track
    record. Purely informational - returns (criteria dict, all_met bool).
    Never touches TRADING_MODE; the switch to real money stays a deliberate
    manual step in GitHub. This just answers 'have I actually earned it, by
    the numbers I said mattered' instead of 'does it feel like it's working'."""
    pf = state["portfolio"]
    closed = pf["closed_trades"]
    n_trades = len(closed)
    pf_factor = profit_factor(closed)
    balances = [h["balance"] for h in pf["balance_history"]]
    drawdown = max_drawdown_pct(balances) if balances else 0.0
    expectancy = expectancy_pct(closed)

    criteria = {
        "profit_factor": {"label": "Profit factor", "value": pf_factor,
                          "target_txt": f"> {GO_LIVE_MIN_PROFIT_FACTOR}",
                          "value_txt": f"{pf_factor:.2f}" if pf_factor is not None else "–",
                          "met": pf_factor is not None and pf_factor > GO_LIVE_MIN_PROFIT_FACTOR},
        "drawdown": {"label": "Max langus", "value": drawdown,
                    "target_txt": f"< {GO_LIVE_MAX_DRAWDOWN_PCT:.0f}%",
                    "value_txt": f"{drawdown:.1f}%",
                    "met": drawdown < GO_LIVE_MAX_DRAWDOWN_PCT},
        "n_trades": {"label": "Suletud kauplusi", "value": n_trades,
                    "target_txt": f">= {GO_LIVE_MIN_CLOSED_TRADES}",
                    "value_txt": str(n_trades),
                    "met": n_trades >= GO_LIVE_MIN_CLOSED_TRADES},
        "expectancy": {"label": "Oodatav väärtus/kauplus", "value": expectancy,
                      "target_txt": "> 0%",
                      "value_txt": f"{expectancy:+.2f}%" if expectancy is not None else "–",
                      "met": expectancy is not None and expectancy > GO_LIVE_MIN_EXPECTANCY_PCT},
    }
    all_met = all(c["met"] for c in criteria.values())
    return criteria, all_met


def go_live_readiness_notes(state):
    """Telegram-facing notes for the current run: a one-time celebratory
    alert the moment ALL criteria are first met (with a reminder that
    flipping TRADING_MODE is still a deliberate manual GitHub step), plus a
    compact once-a-day progress line otherwise so you get a regular pulse
    without needing to check the dashboard or build a two-way Telegram bot."""
    criteria, all_met = go_live_readiness(state)
    notes = []
    if all_met:
        if not state.get("go_live_alert_sent"):
            state["go_live_alert_sent"] = True
            lines = ["🎯 GO-LIVE VALMIDUS SAAVUTATUD - kõik RUNBOOK kriteeriumid on täidetud:"]
            for c in criteria.values():
                lines.append(f"  ✅ {c['label']}: {c['value_txt']} ({c['target_txt']})")
            lines.append("See on ikkagi Sinu enda otsus - live-režiim lülitub käsitsi GitHubis "
                        "(RUNBOOK.md 'Go-live checklist'), mitte automaatselt.")
            notes.append("\n".join(lines))
    else:
        state["go_live_alert_sent"] = False  # allow re-alerting if it regresses and re-qualifies later
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.get("go_live_last_progress_day") != today:
            state["go_live_last_progress_day"] = today
            parts = [f"{c['label']} {c['value_txt']} {'✅' if c['met'] else '⏳'}" for c in criteria.values()]
            notes.append("📊 Go-live valmidus: " + ", ".join(parts))
    return notes


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


def candle_hourly_vol(candles):
    """Realized volatility from 15m candle closes, scaled up to hourly via
    the square-root-of-time rule (sqrt(4) - four 15m bars per hour). With
    ~96 return samples per day this is a far better noise estimate than the
    6..24 sparse hourly snapshots realized_volatility() has to work with.
    Returns None if there aren't enough candles (caller falls back)."""
    closes = [c["c"] for c in candles if c.get("c")]
    if len(closes) < 20:
        return None
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(rets) < 10:
        return None
    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(4)


def trend_bonus_from_candles(candles):
    """Same contract as trend_bonus() (bonus, note) but fitted on the last
    6h of 15m closes - 24 real data points instead of 6 hourly snapshots,
    so one odd hour can't fake or hide a trend. Returns None when there
    aren't enough candles (caller falls back to snapshot math)."""
    recent = [c for c in candles if c.get("c")][-24:]
    if len(recent) < 12:
        return None
    closes = [c["c"] for c in recent]
    vols = [c.get("v", 0.0) for c in recent]

    p0 = closes[0] if closes[0] else 1e-9
    norm_prices = [(p - p0) / p0 for p in closes]
    price_slope, price_r2 = linreg_slope_r2(norm_prices)

    v0 = vols[0] if vols[0] else 1e-9
    norm_vols = [(v - v0) / v0 for v in vols]
    vol_slope, _ = linreg_slope_r2(norm_vols)

    n = len(recent)
    if price_slope > 0 and price_r2 >= 0.5 and vol_slope > 0:
        bonus = round(8 + price_r2 * 12, 1)
        return bonus, f"trend kinnitatud küünaldelt (OLS R²={price_r2:.2f}, {n}×15m punkti / 6h, hind+maht tõusuteel)"
    if price_slope > 0 and price_r2 >= 0.3:
        return 5.0, f"nõrk kinnitus küünaldelt - hind tõusuteel (R²={price_r2:.2f}), aga veel ebakindel"
    if price_slope < 0 and price_r2 >= 0.5:
        return -8.0, f"langustrend kinnitatud küünaldelt (R²={price_r2:.2f}) - ettevaatust"
    return -2.0, "küünlad ei kinnita trendi - ühekordne hüpe või müra (madal R²)"


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


# Hard liquidity floor - a data-driven fix, not a guess: analysis of live
# paper-trading results found many candidates at $166-$60,000 24h volume
# with spreads up to 0.95%, exactly the instruments where fees/slippage eat
# any edge and where a single actor can move the price. Below this, a
# candidate never even reaches scoring - no score is high enough to
# compensate for a market this thin and this easy to manipulate.
MIN_CANDIDATE_VOLUME_USD = 500_000
# Spread wide enough that no score should override it (the softer -12pt
# penalty in finalize() still applies below this, for 0.3-2% spreads).
MAX_CANDIDATE_SPREAD_PCT = 2.0


def screen(tickers_raw_path, out_path, candles_path=None):
    """Stage 1. tickers_raw_path: JSON list of Crypto.com ticker dicts
    (as returned by get_tickers), one per watched instrument. candles_path
    (optional): {instrument: [{t,o,h,l,c,v}, ...]} of 15m candles - when
    present, volatility and trend come from real intra-hour data instead of
    sparse hourly snapshots; when absent everything falls back to the old
    snapshot math, so the Cowork/manual path keeps working unchanged."""
    state = load_state()
    watchlist_cat = load_watchlist()
    tickers = load_json(tickers_raw_path, [])
    candles_all = load_json(candles_path, {}) if candles_path else {}

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

        if volume_value < MIN_CANDIDATE_VOLUME_USD:
            continue  # too thin to trade honestly - see MIN_CANDIDATE_VOLUME_USD

        hist = state["history"].setdefault(inst, [])
        cd = candles_all.get(inst)
        hourly_vol = candle_hourly_vol(cd) if cd else None
        if hourly_vol is None:
            hourly_vol = realized_volatility(hist)
        m_score = momentum_score(change_24h, hourly_vol)
        candle_trend = trend_bonus_from_candles(cd) if cd else None
        if candle_trend is not None:
            bonus, trend_note = candle_trend
        else:
            bonus, trend_note = trend_bonus(hist)
        if inst == "BTCUSD":
            a_bonus, alpha_frac, alpha_note = 0.0, 0.0, "BTC ise on turu võrdlusalus"
        else:
            a_bonus, alpha_frac, alpha_note = alpha_bonus(change_24h, btc_change)

        # append this snapshot to history now (so next run can see it)
        hist.append({"ts": ts, "price": price, "change_24h": change_24h,
                      "volume_value": volume_value})
        state["history"][inst] = hist[-HISTORY_KEEP:]

        raw_score = clamp(m_score + bonus * TREND_BONUS_MULTIPLIER + a_bonus, 0, 100)
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
    Returns (bonus, forced_risk_or_None, why_fragment, excluded). A confirmed
    security warning (scam/hack/governance-attack keywords) is a HARD
    exclusion, not just a score penalty - a data review found candidates
    carrying live security warnings that still scored high enough to trade
    despite a -30 penalty. No momentum score should override "this token
    has an active exploit/scam warning right now"."""
    if not note or not note.get("found"):
        return 0, None, "veebist ei leidnud värsket kajastust - hinnang põhineb ainult turuandmetel", False
    text = (note.get("summary") or "").lower()
    if note.get("sentiment") == "warning" or any(k in text for k in SCAM_KEYWORDS):
        return -30, "red", f"VÄLJA JÄETUD - turvahoiatus veebist: {note.get('summary', '')[:200]}", True
    if note.get("sentiment") == "positive":
        return 12, None, f"hype kinnitatud veebist: {note.get('summary', '')[:200]}", False
    if note.get("sentiment") == "negative":
        return -10, None, f"negatiivne kajastus veebist: {note.get('summary', '')[:200]}", False
    return 3, None, f"mainitud veebis, neutraalne toon: {note.get('summary', '')[:200]}", False


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
                train_step(state, rec["features"], hit, return_pct=ret)
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


SCREEN_SCORE_STAGNANT_STREAK_LIMIT = 3


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
        streak = state["thresholds"].get("screen_score_raise_streak", 0)
        if overall_rate < 0.35:
            # Raising the bar repeatedly while the hit rate STAYS stuck below
            # 35% means the system is just trading less, not trading smarter
            # - a live data review found exactly this pattern (screen_score
            # 65->85 over several cycles with no improvement). Past
            # SCREEN_SCORE_STAGNANT_STREAK_LIMIT consecutive raise-cycles
            # still below the bar, stop auto-raising and say so loudly - the
            # fix belongs in the scoring logic, not in trading less often.
            if streak >= SCREEN_SCORE_STAGNANT_STREAK_LIMIT:
                notes.append(
                    f"⚠️ DIAGNOSTIKA: screen_score on tõstetud {streak} korda järjest (praegu "
                    f"{state['thresholds']['screen_score']}), aga tabamus ({overall_rate:.0%}) ei "
                    f"parane. Viga on tõenäoliselt SIGNAALIS endas, mitte lävendis - täiendav "
                    f"automaatne tõstmine peatatud. Vaja on strateegiat ennast üle vaadata."
                )
            else:
                state["thresholds"]["screen_score"] = clamp(state["thresholds"]["screen_score"] + 3, 50, 85)
                state["thresholds"]["screen_score_raise_streak"] = streak + 1
                notes.append(f"üldine tabamus {overall_rate:.0%} madal -> screen_score tõstetud "
                            f"{state['thresholds']['screen_score']} (järjestikune tõus {streak + 1}/{SCREEN_SCORE_STAGNANT_STREAK_LIMIT})")
        elif overall_rate > 0.65:
            state["thresholds"]["screen_score"] = clamp(state["thresholds"]["screen_score"] - 2, 50, 85)
            state["thresholds"]["screen_score_raise_streak"] = 0
            notes.append(f"üldine tabamus {overall_rate:.0%} kõrge -> screen_score langetatud {state['thresholds']['screen_score']}")
        else:
            state["thresholds"]["screen_score_raise_streak"] = 0
    return notes


def finalize(candidates_path, hype_notes_path, current_prices_path, out_summary_path,
             candles_path=None, book_path=None, funding_path=None):
    state = load_state()
    candidates = load_json(candidates_path, [])
    hype_notes = load_json(hype_notes_path, {})
    current_prices = load_json(current_prices_path, {})
    candles_all = load_json(candles_path, {}) if candles_path else {}
    book_notes = load_json(book_path, {}) if book_path else {}
    funding_rates = load_json(funding_path, {}) if funding_path else {}
    regime = load_json(os.path.join(DATA_DIR, "market_regime.json"), {})
    fng_value = regime.get("value")
    onchain = load_json(os.path.join(DATA_DIR, "onchain_latest.json"), {})
    onchain_ratio = onchain.get("ratio_vs_7d_avg")

    brk = broker_mod.get_broker()
    state["portfolio"]["mode"] = brk.mode
    reconcile_notes = brk.reconcile(state)
    stop_notes = manage_positions(state, current_prices, candles_all, brk)
    followup_notes = process_followups(state, current_prices, brk)
    retrain_note = full_retrain(state)
    record_equity_snapshot(state, current_prices)
    killswitch_notes = update_killswitch(state, brk)
    threshold_notes = adapt_thresholds(state) or []

    # Independent sleeves - each manages its own separate ledger and never
    # touches state["portfolio"] (the momentum book). Both are still gated
    # by the momentum kill-switch (a real account-level circuit breaker
    # should stop ALL new risk-taking, not just one strategy).
    funding_close_notes = manage_funding_positions(state, funding_rates, current_prices, brk)
    funding_open_notes = scan_funding_opportunities(state, funding_rates, current_prices, brk)
    grid_notes = manage_grid(state, current_prices, candles_all, brk)
    btc_hist = state["history"].get("BTCUSD", [])
    btc_change_24h = btc_hist[-1]["change_24h"] if btc_hist else None
    short_notes = manage_short_sleeve(state, candidates, btc_change_24h, fng_value, current_prices, brk)
    if retrain_note:
        threshold_notes = threshold_notes + [retrain_note]

    n_updates = state["model"]["n_updates"]
    model_ready = n_updates >= MODEL_MIN_TRAINING

    alerts = []
    swap_notes = []
    swaps_left = SWAP_MAX_PER_RUN
    ts = now_ts()

    excluded_notes = []
    for c in candidates:
        inst = c["instrument"]
        note = hype_notes.get(inst)
        bonus, forced_risk, why_hype, hype_excluded = hype_adjustment(note)
        if hype_excluded:
            excluded_notes.append(f"🚫 {inst} VÄLJA JÄETUD: {why_hype[:150]}")
            continue

        # Order book check: thin/wide books are where market orders get hurt,
        # and top-of-book imbalance is real-time buy/sell pressure. A spread
        # this wide is a hard exclusion (data review found candidates with up
        # to 0.95% spread still trading on score alone) - below that, only a
        # score penalty.
        book = book_notes.get(inst) or {}
        imbalance = book.get("imbalance", 0.0)
        spread_pct = book.get("spread_pct")
        if spread_pct is not None and spread_pct > MAX_CANDIDATE_SPREAD_PCT:
            excluded_notes.append(f"🚫 {inst} VÄLJA JÄETUD: spread {spread_pct:.2f}% > {MAX_CANDIDATE_SPREAD_PCT:.1f}% lubatud")
            continue
        book_adj = 0
        if spread_pct is None:
            book_note = "orderiraamatu andmeid pole selles käivituses"
        elif spread_pct > 1.0:
            book_adj = -12
            book_note = f"HOIATUS: õhuke raamat (spread {spread_pct:.2f}%) - turuorder saab siin valusalt pihta"
        elif imbalance >= 0.3:
            book_adj = 6
            book_note = f"ostusurve raamatus (imbalance {imbalance:+.2f}, spread {spread_pct:.2f}%)"
        elif imbalance <= -0.3:
            book_adj = -6
            book_note = f"müügisurve raamatus (imbalance {imbalance:+.2f}, spread {spread_pct:.2f}%)"
        else:
            book_note = f"raamat tasakaalus (imbalance {imbalance:+.2f}, spread {spread_pct:.2f}%)"

        heuristic_score = clamp(c["raw_score"] + bonus + book_adj, 0, 100)
        risk = risk_label(c["category"], c["liquidity"], c["change_24h"], forced_risk)

        features = build_features(c["momentum_score"], c["trend_bonus"], c["liquidity"],
                                   c["category"], c["change_24h"], bonus, c.get("alpha_pct", 0.0),
                                   book_imbalance=imbalance, fng_value=fng_value, onchain_ratio=onchain_ratio)
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
               f"(maht ${c['volume_value']:,.0f}). {book_note}. {why_hype}{model_note}{explore_note}")

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
            alpha_pct_val = c.get("alpha_pct", 0.0)
            traded = maybe_open_position(state, rec_id, inst, c["price"], ts, risk, final_score,
                                          brk, category=c["category"], win_prob=win_prob,
                                          alpha_pct=alpha_pct_val)
            if not traded and swaps_left > 0:
                swap_note, traded = maybe_swap_position(state, rec_id, inst, c["price"], ts, risk,
                                                        final_score, brk, current_prices,
                                                        category=c["category"], win_prob=win_prob,
                                                        alpha_pct=alpha_pct_val)
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

    # Computed BEFORE save_state() - it sets go_live_alert_sent /
    # go_live_last_progress_day flags on state that must be persisted.
    readiness_notes = go_live_readiness_notes(state)

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
    if funding_close_notes:
        lines.extend(funding_close_notes)
    if funding_open_notes:
        lines.extend(funding_open_notes)
    if grid_notes:
        lines.extend(grid_notes)
    if short_notes:
        lines.extend(short_notes)
    if excluded_notes:
        lines.extend(excluded_notes)
    if reconcile_notes:
        lines.extend(reconcile_notes)
    if fng_value is not None and (fng_value <= 25 or fng_value >= 75):
        mood = "äärmuslik HIRM - turg paanikas, momentum petlik" if fng_value <= 25 \
            else "äärmuslik AHNUS - turg ülekuumenenud, ettevaatust"
        lines.append(f"🌡️ Turu meeleolu: {regime.get('classification', '')} ({fng_value}/100) - {mood}")
    if onchain_ratio is not None and (onchain_ratio >= 1.8 or onchain_ratio <= 0.5):
        direction = "ebatavaliselt kõrge" if onchain_ratio >= 1.8 else "ebatavaliselt madal"
        lines.append(f"⛓️ BTC on-chain aktiivsus {direction} ({onchain_ratio:.2f}x 7p keskmisest)")
    if readiness_notes:
        lines.extend(readiness_notes)
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
    "book_imbalance": "Orderiraamatu ostu/müügisurve",
    "market_fng": "Turu meeleolu (Fear & Greed)",
    "onchain_activity": "BTC on-chain aktiivsus",
}


def render_dashboard(state):
    regime = load_json(os.path.join(DATA_DIR, "market_regime.json"), {})
    fng_txt = (f"{regime['value']} · {regime.get('classification', '')}"
               if regime.get("value") is not None else "–")
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
    # Total equity (cash + mark-to-market open positions), not raw cash -
    # the latest balance_history point is always an equity snapshot (see
    # record_equity_snapshot). Cash alone understates account value while
    # positions are open and would make active trading look like a loss.
    pf_equity = pf["balance_history"][-1]["balance"] if pf["balance_history"] else pf["balance"]
    pf_return_pct = (pf_equity - pf["starting_balance"]) / pf["starting_balance"] * 100
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

    def sleeve_stats(ledger):
        closed = ledger["closed_trades"]
        equity = ledger["balance_history"][-1]["balance"] if ledger["balance_history"] else ledger["balance"]
        return_pct = (equity - ledger["starting_balance"]) / ledger["starting_balance"] * 100 if ledger["starting_balance"] else 0.0
        wins = sum(1 for t in closed if t["pnl_usd"] > 0)
        return {
            "equity": equity, "return_pct": return_pct, "n_open": len(ledger["open_positions"]),
            "n_closed": len(closed), "wins": wins, "losses": len(closed) - wins,
            "profit_factor": profit_factor(closed), "expectancy": expectancy_pct(closed),
        }

    fa = state.get("funding_arb", funding_arb_state_default())
    fa_stats = sleeve_stats(fa)
    fa_rows = ""
    for t in reversed(fa["closed_trades"][-20:]):
        dt = datetime.fromtimestamp(t["exit_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        pnl_color = "#0ca30c" if t["pnl_usd"] > 0 else "#f4756f"
        fa_rows += f"""
        <tr><td>{dt}</td><td class="mono">{t['instrument']}</td><td>{t['entry_apr']:+.1f}%</td>
        <td>${t.get('funding_collected_usd',0):+.2f}</td>
        <td style="color:{pnl_color}">{t['pnl_pct']:+.1f}% (${t['pnl_usd']:+.2f})</td></tr>"""
    fa_open_rows = ""
    for p in fa["open_positions"]:
        dt = datetime.fromtimestamp(p["entry_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        fa_open_rows += f"""
        <tr><td>{dt}</td><td class="mono">{p['instrument']}</td><td>{p['entry_apr']:+.1f}%</td>
        <td>${p['notional_usd']:.2f}/jalg</td><td>${p.get('funding_collected_usd',0):+.2f}</td></tr>"""

    grid = state.get("grid", grid_state_default())
    grid_stats = sleeve_stats(grid)
    grid_rows = ""
    for t in reversed(grid["closed_trades"][-20:]):
        dt = datetime.fromtimestamp(t["exit_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        pnl_color = "#0ca30c" if t["pnl_usd"] > 0 else "#f4756f"
        grid_rows += f"""
        <tr><td>{dt}</td><td class="mono">{t['instrument']}</td><td>{t.get('exit_reason','')}</td>
        <td style="color:{pnl_color}">{t['pnl_pct']:+.1f}% (${t['pnl_usd']:+.2f})</td></tr>"""
    grid_open_rows = ""
    for p in grid["open_positions"]:
        dt = datetime.fromtimestamp(p["entry_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        grid_open_rows += f"""
        <tr><td>{dt}</td><td class="mono">{p['instrument']}</td><td class="mono">{p['entry_price']:.6g}</td>
        <td>${p['size_usd']:.2f}</td></tr>"""

    short_sleeve = state.get("short_reversal", short_state_default())
    short_stats = sleeve_stats(short_sleeve)
    short_rows = ""
    for t in reversed(short_sleeve["closed_trades"][-20:]):
        dt = datetime.fromtimestamp(t["exit_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        pnl_color = "#0ca30c" if t["pnl_usd"] > 0 else "#f4756f"
        short_rows += f"""
        <tr><td>{dt}</td><td class="mono">{t['instrument']}</td><td>{t.get('exit_reason','')}</td>
        <td style="color:{pnl_color}">{t['pnl_pct']:+.1f}% (${t['pnl_usd']:+.2f})</td></tr>"""
    short_open_rows = ""
    for p in short_sleeve["open_positions"]:
        dt = datetime.fromtimestamp(p["entry_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        short_open_rows += f"""
        <tr><td>{dt}</td><td class="mono">{p['instrument']}</td><td class="mono">{p['entry_price']:.6g}</td>
        <td>${p['size_usd']:.2f}</td></tr>"""

    readiness_criteria, readiness_all_met = go_live_readiness(state)
    readiness_rows = ""
    for c in readiness_criteria.values():
        mark = "✅" if c["met"] else "⏳"
        row_color = "var(--good)" if c["met"] else "var(--muted)"
        readiness_rows += f"""
        <tr><td>{c['label']}</td><td class="mono" style="color:{row_color}">{c['value_txt']}</td>
        <td class="mono">{c['target_txt']}</td><td style="text-align:center">{mark}</td></tr>"""

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
        if p.get("trailing"):
            stop_txt += ' <span class="tag" style="background:#22d3ee">järgneb</span>'
        if p.get("partial_taken"):
            stop_txt += ' <span class="tag" style="background:#22c55e">TP½ võetud</span>'
        pf_open_rows += f"""
        <tr><td>{dt}</td><td class="mono">{p['instrument']}</td><td>${p['size_usd']:.2f} <span class="tag" style="background:linear-gradient(120deg,var(--accent2),var(--pink))">{size_pct_txt}</span></td>
        <td class="mono">{p['entry_price']:.6g}</td><td class="mono">{stop_txt}</td><td>{risk_dot(p['risk'])}</td></tr>"""

    pf_closed_rows = ""
    for t in pf_closed:
        dt = datetime.fromtimestamp(t["exit_ts"], tz=timezone.utc).strftime("%d.%m %H:%M")
        pnl_color = "#0ca30c" if t["pnl_usd"] > 0 else "#f4756f"
        reason_tag = {"stop_loss": ' <span class="tag" style="background:#f4756f">SL</span>',
                      "swap": ' <span class="tag" style="background:#f59e0b">SWAP</span>',
                      "take_profit": ' <span class="tag" style="background:#22c55e">TP½</span>',
                      "trailing_stop": ' <span class="tag" style="background:#22d3ee">TSL</span>'}.get(t.get("exit_reason"), "")
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
    <div class="stat-card"><div class="label">Turu meeleolu (F&G)</div><div class="value">{fng_txt}</div></div>
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
      <div class="stat-card"><div class="label">Omakapital (raha+avatud)</div><div class="value" style="color:{'var(--good)' if pf_equity>=pf['starting_balance'] else 'var(--critical)'}">${pf_equity:.2f}</div></div>
      <div class="stat-card"><div class="label">Tootlus algusest</div><div class="value" style="color:{'var(--good)' if pf_return_pct>=0 else 'var(--critical)'}">{pf_return_pct:+.1f}%</div></div>
      <div class="stat-card"><div class="label">Vaba sularaha</div><div class="value">${pf['balance']:.2f}</div></div>
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

  <div class="card" style="{'border-color:var(--good)' if readiness_all_met else ''}">
    <h2>🎯 Go-live valmidus{info_badge("RUNBOOK.md checklisti neli kriteeriumi. Ainult info - live-režiimi lülitamine jääb alati käsitsi GitHubi sammuks, see kaart lihtsalt ütleb ausalt, millal number seda õigustab.")}</h2>
    {'<div class="model-status" style="border-color:var(--good);background:rgba(34,197,94,0.10)">✅ Kõik kriteeriumid täidetud - vaata RUNBOOK.md "Go-live checklist" järgmisi samme.</div>' if readiness_all_met else ''}
    <div class="table-scroll"><table>
      <tr><th>Kriteerium</th><th>Praegu</th><th>Vajalik</th><th>Staatus</th></tr>
      {readiness_rows}
    </table></div>
  </div>

  <details class="card">
    <summary><h2>💹 Funding-arb sahtel{info_badge("Turuneutraalne strateegia, eraldi kapitaliga (algsaldo $" + f'{fa["starting_balance"]:.0f}' + "): ostab spot + avab võrdse suurusega lühikese perpetual-positsiooni, teenib ainult funding-makseid, ei sõltu turusuunast. Sisenemine kui funding ≥" + f'{FUNDING_MIN_APR_ENTER:.0f}' + "% aastas, väljumine kui langeb alla " + f'{FUNDING_MIN_APR_EXIT:.0f}' + "% või max " + f'{FUNDING_MAX_HOLD_HOURS//24}' + " päeva täis.")}</h2><span class="chev">▼</span></summary>
    <div class="body">
    <div class="stats" style="margin-bottom:14px">
      <div class="stat-card"><div class="label">Omakapital</div><div class="value" style="color:{'var(--good)' if fa_stats['equity']>=fa['starting_balance'] else 'var(--critical)'}">${fa_stats['equity']:.2f}</div></div>
      <div class="stat-card"><div class="label">Tootlus</div><div class="value" style="color:{'var(--good)' if fa_stats['return_pct']>=0 else 'var(--critical)'}">{fa_stats['return_pct']:+.1f}%</div></div>
      <div class="stat-card"><div class="label">Avatud/suletud</div><div class="value">{fa_stats['n_open']}/{fa_stats['n_closed']}</div></div>
      <div class="stat-card"><div class="label">Profit factor</div><div class="value">{f"{fa_stats['profit_factor']:.2f}" if fa_stats['profit_factor'] is not None else '–'}</div></div>
    </div>
    <h2 style="margin-top:10px;font-size:13px">Avatud positsioonid</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Token</th><th>Funding (aastas)</th><th>Notional</th><th>Kogutud funding</th></tr>
      {fa_open_rows or '<tr><td colspan="5" style="color:var(--muted)">Hetkel pole avatud positsioone.</td></tr>'}
    </table></div>
    <h2 style="margin-top:14px;font-size:13px">Suletud (uusimad enne)</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Token</th><th>Sisenemis-APR</th><th>Kogutud funding</th><th>Netotulem</th></tr>
      {fa_rows or '<tr><td colspan="5" style="color:var(--muted)">Veel pole ühtegi suletud.</td></tr>'}
    </table></div>
    </div>
  </details>

  <details class="card">
    <summary><h2>🔲 Grid/range sahtel{info_badge("Mean-reversion strateegia likviidsetel majoritel (BTC, ETH), eraldi kapitaliga (algsaldo $" + f'{grid["starting_balance"]:.0f}' + "): ostab kui hind on oma 24h vahemiku põhjas JA trend on nõrk (madal R² - momentumi vastand), müüb +" + f'{GRID_TAKE_PROFIT_PCT:.0f}' + "% juures või vahemiku tipus. Monetiseerib tunde, mil momentum-strateegia lihtsalt ootab.")}</h2><span class="chev">▼</span></summary>
    <div class="body">
    <div class="stats" style="margin-bottom:14px">
      <div class="stat-card"><div class="label">Omakapital</div><div class="value" style="color:{'var(--good)' if grid_stats['equity']>=grid['starting_balance'] else 'var(--critical)'}">${grid_stats['equity']:.2f}</div></div>
      <div class="stat-card"><div class="label">Tootlus</div><div class="value" style="color:{'var(--good)' if grid_stats['return_pct']>=0 else 'var(--critical)'}">{grid_stats['return_pct']:+.1f}%</div></div>
      <div class="stat-card"><div class="label">Avatud/suletud</div><div class="value">{grid_stats['n_open']}/{grid_stats['n_closed']}</div></div>
      <div class="stat-card"><div class="label">Profit factor</div><div class="value">{f"{grid_stats['profit_factor']:.2f}" if grid_stats['profit_factor'] is not None else '–'}</div></div>
    </div>
    <h2 style="margin-top:10px;font-size:13px">Avatud positsioonid</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Token</th><th>Sisenemishind</th><th>Suurus</th></tr>
      {grid_open_rows or '<tr><td colspan="4" style="color:var(--muted)">Hetkel pole avatud positsioone.</td></tr>'}
    </table></div>
    <h2 style="margin-top:14px;font-size:13px">Suletud (uusimad enne)</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Token</th><th>Põhjus</th><th>Netotulem</th></tr>
      {grid_rows or '<tr><td colspan="4" style="color:var(--muted)">Veel pole ühtegi suletud.</td></tr>'}
    </table></div>
    </div>
  </details>

  <details class="card">
    <summary><h2>📉 Short-pöörduse sahtel (eksperimentaalne){info_badge("Režiimist sõltuv mean-reversion, eraldi kapitaliga (algsaldo $" + f'{short_sleeve["starting_balance"]:.0f}' + "). Andmepõhine leid: pime 'pööra kõik signaalid ümber' andis kogu ajaloo peal ainult +0.4%/tehing toorest edge'i - alla kauplemiskulu. Seetõttu shortitakse AINULT kui BTC 24h muutus ≤" + f'{SHORT_BTC_BEAR_CHANGE_PCT:.0f}' + "% JA Fear&Greed ≤" + f'{SHORT_FNG_FEAR_MAX}' + " (kinnitatud turu-hirm) JA kandidaat on üleostetud (momentum ≥" + f'{SHORT_MIN_MOMENTUM_SCORE:.0f}' + ", trend nõrgalt kinnitatud). Väike, ettevaatlik katse, mitte peamine strateegia.")}</h2><span class="chev">▼</span></summary>
    <div class="body">
    <div class="stats" style="margin-bottom:14px">
      <div class="stat-card"><div class="label">Omakapital</div><div class="value" style="color:{'var(--good)' if short_stats['equity']>=short_sleeve['starting_balance'] else 'var(--critical)'}">${short_stats['equity']:.2f}</div></div>
      <div class="stat-card"><div class="label">Tootlus</div><div class="value" style="color:{'var(--good)' if short_stats['return_pct']>=0 else 'var(--critical)'}">{short_stats['return_pct']:+.1f}%</div></div>
      <div class="stat-card"><div class="label">Avatud/suletud</div><div class="value">{short_stats['n_open']}/{short_stats['n_closed']}</div></div>
      <div class="stat-card"><div class="label">Profit factor</div><div class="value">{f"{short_stats['profit_factor']:.2f}" if short_stats['profit_factor'] is not None else '–'}</div></div>
    </div>
    <h2 style="margin-top:10px;font-size:13px">Avatud positsioonid</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Token</th><th>Sisenemishind</th><th>Suurus</th></tr>
      {short_open_rows or '<tr><td colspan="4" style="color:var(--muted)">Hetkel pole avatud positsioone.</td></tr>'}
    </table></div>
    <h2 style="margin-top:14px;font-size:13px">Suletud (uusimad enne)</h2>
    <div class="table-scroll"><table>
      <tr><th>Millal</th><th>Token</th><th>Põhjus</th><th>Netotulem</th></tr>
      {short_rows or '<tr><td colspan="4" style="color:var(--muted)">Veel pole ühtegi suletud.</td></tr>'}
    </table></div>
    </div>
  </details>

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
    if INDEX_PATH:  # None in production - see the constant for why
        with open(INDEX_PATH, "w") as f:
            f.write(html)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("screen")
    s1.add_argument("--tickers", required=True)
    s1.add_argument("--out", required=True)
    s1.add_argument("--candles", default=None)

    s2 = sub.add_parser("finalize")
    s2.add_argument("--candidates", required=True)
    s2.add_argument("--hype-notes", required=True)
    s2.add_argument("--current-prices", required=True)
    s2.add_argument("--out-summary", required=True)
    s2.add_argument("--candles", default=None)
    s2.add_argument("--book", default=None)

    args = ap.parse_args()
    if args.cmd == "screen":
        screen(args.tickers, args.out, candles_path=args.candles)
    elif args.cmd == "finalize":
        finalize(args.candidates, args.hype_notes, args.current_prices, args.out_summary,
                 candles_path=args.candles, book_path=args.book)
