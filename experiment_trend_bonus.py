#!/usr/bin/env python3
"""
Train/test experiment: does trend_bonus help or hurt raw_score?

Why this exists: a full-year backtest (see backtest.py) found the model's
own learned weight for trend_bonus went NEGATIVE - confirmed "breakouts"
predicted WORSE outcomes, not better. That's a real, data-driven signal,
but hand-picking new parameters to fit that same year of data would be
overfitting (see the project discussion this script came out of) - a
config tuned to match history exactly has no reason to generalize.

This script tests the hypothesis properly instead:
  1. Fetch history ONCE, split chronologically into TRAIN (first ~8
     months) and TEST (last ~4 months, held out - never used to pick
     anything).
  2. On TRAIN only, run the full backtest replay once per candidate
     TREND_BONUS_MULTIPLIER value (1.0=current, 0.5, 0.0=disabled,
     -0.5, -1.0=fully inverted), each from a clean state.
  3. Pick whichever multiplier had the best expectancy on TRAIN.
  4. Re-run BOTH that winner AND the current baseline (1.0) on TEST -
     data neither ever touched during selection - and report both
     side by side.

If the "winning" multiplier doesn't beat the 1.0 baseline on TEST, that
is a real, honest result: the trend_bonus signal doesn't generalize
cleanly enough to act on with a simple multiplier, and the answer is NOT
to keep searching until something looks good on this same year of data.

Usage: python3 experiment_trend_bonus.py --months 12 --train-fraction 0.67
"""
import argparse
import json
import os
import sys
from datetime import datetime

import engine
import backtest as bt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE_DIR, "experiment_data")

DEFAULT_INSTRUMENTS = bt.DEFAULT_INSTRUMENTS
CANDIDATE_MULTIPLIERS = [1.0, 0.0, -1.0]  # baseline / disabled / fully inverted -
                                          # kept short since each is a full TRAIN-period replay


def run_replay(all_ts_slice, histories, btc_closes_by_ts, volume_24h_by_symbol, state_path, multiplier):
    """One clean replay over a slice of the timeline, at a given
    TREND_BONUS_MULTIPLIER, starting from an empty state. Returns the
    final state dict."""
    if os.path.exists(state_path):
        os.remove(state_path)
    engine.STATE_PATH = state_path
    engine.TREND_BONUS_MULTIPLIER = multiplier

    tickers_path = os.path.join(EXP_DIR, "tickers.json")
    cands_path = os.path.join(EXP_DIR, "cands.json")
    hype_path = os.path.join(EXP_DIR, "hype.json")
    prices_path = os.path.join(EXP_DIR, "prices.json")
    summary_path = os.path.join(EXP_DIR, "summary.txt")
    with open(hype_path, "w") as f:
        json.dump({}, f)

    def _simulated_now():
        return _simulated_now.ts_seconds
    _simulated_now.ts_seconds = all_ts_slice[0] / 1000.0 if all_ts_slice else 0.0
    engine.now_ts = _simulated_now

    for ts in all_ts_slice:
        _simulated_now.ts_seconds = ts / 1000.0
        tickers = []
        for sym, hist in histories.items():
            c = hist.get(ts)
            if not c:
                continue
            past_ts = ts - 24 * 3600 * 1000
            past_c = hist.get(past_ts)
            if not past_c:
                earliest = min(hist.keys())
                past_c = hist[earliest]
            change = (c["c"] - past_c["c"]) / past_c["c"] if past_c["c"] else 0.0
            vol_24h = volume_24h_by_symbol[sym].get(ts, c["v"] * c["c"])
            tickers.append({"instrument_name": sym, "last": c["c"], "change": change,
                            "volume_value": vol_24h})
        engine.save_json(tickers_path, tickers)
        engine.screen(tickers_path, cands_path)
        current_prices = {t["instrument_name"]: t["last"] for t in tickers}
        engine.save_json(prices_path, current_prices)
        engine.finalize(cands_path, hype_path, prices_path, summary_path)

    return engine.load_state()


def report(label, state):
    closed = state["portfolio"]["closed_trades"]
    if not closed:
        print(f"  {label}: 0 kauplust")
        return None
    pf = engine.profit_factor(closed)
    exp = engine.expectancy_pct(closed)
    wins = sum(1 for t in closed if t["pnl_usd"] > 0)
    pf_txt = f"{pf:.2f}" if pf is not None else "–"
    exp_txt = f"{exp:+.2f}%" if exp is not None else "–"
    print(f"  {label}: {len(closed)} kauplust, {wins}/{len(closed)} võitu ({wins/len(closed)*100:.0f}%), "
          f"PF={pf_txt}, expectancy={exp_txt}")
    return {"n": len(closed), "pf": pf, "expectancy": exp, "wins": wins}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--train-fraction", type=float, default=0.67,
                    help="Fraction of the timeline used for TRAIN (selection); rest is held-out TEST")
    ap.add_argument("--instruments", type=str, default=None)
    args = ap.parse_args()
    instruments = args.instruments.split(",") if args.instruments else DEFAULT_INSTRUMENTS

    os.makedirs(EXP_DIR, exist_ok=True)
    engine.DATA_DIR = EXP_DIR
    engine.DASHBOARD_PATH = os.path.join(EXP_DIR, "dashboard.html")
    engine.INDEX_PATH = os.path.join(EXP_DIR, "index.html")

    print(f"Laen {args.months} kuu ajalugu {len(instruments)} instrumendi kohta ühe korra...")
    histories = {}
    for sym in instruments:
        real = bt._real_instrument_name(sym)
        if not real:
            continue
        candles = bt.fetch_full_history(real, args.months)
        if len(candles) < 200:
            print(f"  {sym}: liiga vähe andmeid ({len(candles)}), jäetakse vahele")
            continue
        histories[sym] = {c["t"]: c for c in candles}
        print(f"  {sym} ({real}): {len(candles)} tunniküünalt")

    all_ts = sorted(set.intersection(*[set(h.keys()) for h in histories.values()]))
    split_idx = int(len(all_ts) * args.train_fraction)
    train_ts = all_ts[:split_idx]
    test_ts = all_ts[split_idx:]
    print(f"\nKokku {len(all_ts)}h. TRAIN: {len(train_ts)}h "
          f"({datetime.utcfromtimestamp(train_ts[0]/1000).strftime('%Y-%m-%d')} -> "
          f"{datetime.utcfromtimestamp(train_ts[-1]/1000).strftime('%Y-%m-%d')}), "
          f"TEST (nägematu): {len(test_ts)}h "
          f"({datetime.utcfromtimestamp(test_ts[0]/1000).strftime('%Y-%m-%d')} -> "
          f"{datetime.utcfromtimestamp(test_ts[-1]/1000).strftime('%Y-%m-%d')})\n")

    volume_24h_by_symbol = {}
    for sym, hist in histories.items():
        sorted_ts = sorted(hist.keys())
        notionals = [hist[t]["v"] * hist[t]["c"] for t in sorted_ts]
        rolling, window_sum, window = {}, 0.0, []
        for t, notional in zip(sorted_ts, notionals):
            window.append(notional)
            window_sum += notional
            if len(window) > 24:
                window_sum -= window.pop(0)
            rolling[t] = window_sum
        volume_24h_by_symbol[sym] = rolling

    # --- Phase 1: TRAIN - try each multiplier, pick the best by expectancy ---
    print("=" * 70)
    print("FAAS 1: TRAIN periood - iga trend_bonus multiplier eraldi puhtalt olekult")
    print("=" * 70)
    train_results = {}
    for mult in CANDIDATE_MULTIPLIERS:
        state_path = os.path.join(EXP_DIR, f"state_train_{mult}.json")
        st = run_replay(train_ts, histories, None, volume_24h_by_symbol, state_path, mult)
        r = report(f"multiplier={mult:+.1f}", st)
        train_results[mult] = r

    scored = [(m, r) for m, r in train_results.items() if r is not None and r["n"] >= 10 and r["expectancy"] is not None]
    if not scored:
        print("\nMitte ühelgi multiplier'il ei tulnud TRAIN peal piisavalt kauplusi (>=10) - katkestan.")
        return
    best_mult, best_train_r = max(scored, key=lambda x: x[1]["expectancy"])
    print(f"\nParim TRAIN peal: multiplier={best_mult:+.1f} (expectancy {best_train_r['expectancy']:+.2f}%, n={best_train_r['n']})")

    # --- Phase 2: TEST - validate the winner AND the baseline on unseen data ---
    print()
    print("=" * 70)
    print("FAAS 2: TEST periood (NÄGEMATU andmestik, ei mõjutanud valikut)")
    print("=" * 70)
    state_path_baseline = os.path.join(EXP_DIR, "state_test_baseline.json")
    st_baseline = run_replay(test_ts, histories, None, volume_24h_by_symbol, state_path_baseline, 1.0)
    baseline_test_r = report("Baseline (multiplier=1.0, praegune tootmiskood)", st_baseline)

    if best_mult != 1.0:
        state_path_winner = os.path.join(EXP_DIR, f"state_test_winner_{best_mult}.json")
        st_winner = run_replay(test_ts, histories, None, volume_24h_by_symbol, state_path_winner, best_mult)
        winner_test_r = report(f"Võitja TRAIN pealt (multiplier={best_mult:+.1f}) TEST peal", st_winner)
    else:
        winner_test_r = baseline_test_r
        print("  (Võitja OLI baseline - pole eraldi midagi valideerida.)")

    print()
    print("=" * 70)
    print("JÄRELDUS")
    print("=" * 70)
    winner_exp = winner_test_r["expectancy"] if winner_test_r else None
    baseline_exp = baseline_test_r["expectancy"] if baseline_test_r else None
    if best_mult == 1.0:
        print("Praegune trend_bonus käitumine (multiplier=1.0) oli juba parim ka TRAIN peal - "
              "andmed ei toeta muutust.")
    elif winner_exp is None or baseline_exp is None:
        print("Üks pool ei andnud TEST peal piisavalt kauplusi usaldusväärseks võrdluseks.")
    elif winner_exp > baseline_exp:
        print(f"Multiplier={best_mult:+.1f} võitis KA nägematul TEST perioodil "
              f"({winner_exp:+.2f}% vs baseline'i {baseline_exp:+.2f}%) - "
              f"see on reaalne, mitte üle sobitatud signaal. Väärt tootmisse viimist, ikkagi ettevaatlikult.")
    else:
        print(f"Multiplier={best_mult:+.1f} NÄGI TRAIN peal hea välja, aga EI PIDANUD TEST peal vastu "
              f"({winner_exp:+.2f}% vs baseline'i {baseline_exp:+.2f}%). "
              f"See on täpselt üle sobitamise näide - AUS tulemus, mitte midagi tootmisse viia.")


if __name__ == "__main__":
    main()
