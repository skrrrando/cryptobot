#!/usr/bin/env python3
"""
Offline backtest over data/memecoin_labels.jsonl - answers "if we'd sold on
a fixed take-profit/stop-loss rule instead of whatever we actually did,
which threshold combination would have made the most money?"

NOT part of the live scanner and never run by the GitHub Actions workflow -
a manual tool (`python3 backtest_exit_rules.py`) for checking whether
scan_memecoins.py's actual exit thresholds (TAKE_PROFIT_MIN_PCT,
STOP_LOSS_PCT, TRAILING_STOP_*) are in a reasonable neighborhood, or whether
the data points somewhere else entirely.

Every candidate is measured with the exact same ruler: the same rule is
simulated against every entry's checkpoint sequence, not tuned per-token -
that's the whole point of a backtest over cherry-picking winners after the
fact.

Important limitation: we only have 5 fixed checkpoints per entry (10m, 30m,
1h, 3h, 6h), not continuous price data, so a simulated exit can only ever
land ON one of those 5 points - "exit at +18%" really means "exit at the
first checkpoint where return was >= 18%", which could overshoot the target
by however much the price moved between checkpoints. This is a coarse
approximation, not a precise backtest, and does NOT include gas/slippage/tax
(see GAS_COST_USD/estimate_slippage_pct in scan_memecoins.py for that model)
- these are raw price returns only. Same "don't declare a pattern found on
too little data" caution as analyze_labels.py applies here too.

Profit factor and max drawdown are both equal-weighted, non-compounding
approximations (every trade sized the same, no reinvestment) - a rough
robustness signal, not a real portfolio simulation. Drawdown in particular
depends on trade ORDER, which we only have to entry-timestamp resolution.
"""
import json
import statistics
from collections import defaultdict

LABELS_PATH = "data/memecoin_labels.jsonl"
CHECKPOINT_OFFSETS = [10, 30, 60, 180, 360]

MIN_TRUSTWORTHY_N = 30

# What scan_memecoins.py actually does right now - shown as a reference
# point so the sweep results below can be read as "better/worse than live",
# not just as numbers in a vacuum.
LIVE_TAKE_PROFIT_PCT = 15.0
LIVE_STOP_LOSS_PCT = 30.0

TAKE_PROFIT_SWEEP = [5, 10, 15, 20, 25, 30, 40, 50]
STOP_LOSS_SWEEP = [10, 15, 20, 25, 30, 40]


def load_rows():
    rows = []
    try:
        with open(LABELS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return rows


def build_sequences(rows):
    """pool_id -> {"entry_ts": str, "checkpoints": [(offset_minutes, return_pct), ...]}
    sorted by offset. Entries with no checkpoints yet are skipped entirely -
    nothing to simulate an exit against. entry_ts is kept so trades can be
    ordered chronologically for the drawdown calc below."""
    entry_ts_by_pool = {r["pool_id"]: r.get("entry_ts") for r in rows if r.get("row_type") == "entry"}
    by_pool = defaultdict(list)
    for r in rows:
        if r.get("row_type") != "checkpoint" or r.get("return_pct") is None:
            continue
        by_pool[r["pool_id"]].append((r["offset_minutes"], r["return_pct"]))
    sequences = {}
    for pool_id, checkpoints in by_pool.items():
        checkpoints.sort(key=lambda t: t[0])
        sequences[pool_id] = {"entry_ts": entry_ts_by_pool.get(pool_id) or "", "checkpoints": checkpoints}
    return sequences


def simulate(sequences, take_profit_pct, stop_loss_pct):
    """For each token: walk its checkpoints in time order, exit at the
    first one that crosses take_profit_pct or -stop_loss_pct. If neither
    ever triggers, "hold" to the last checkpoint we actually have (a stand-in
    for the real bot's 6h-timeout fallback - slightly generous, since a real
    6h checkpoint may not exist yet for very recent entries). Returns a list
    of (entry_ts, exit_return_pct) tuples, unsorted."""
    results = []
    for pool_id, data in sequences.items():
        exit_return = None
        for _offset, ret in data["checkpoints"]:
            if ret >= take_profit_pct or ret <= -stop_loss_pct:
                exit_return = ret
                break
        if exit_return is None:
            exit_return = data["checkpoints"][-1][1]
        results.append((data["entry_ts"], exit_return))
    return results


def profit_factor(returns):
    gains = sum(r for r in returns if r > 0)
    losses = sum(-r for r in returns if r < 0)
    if losses == 0:
        return None  # undefined (no losing trades in this sample) - not "infinite edge"
    return gains / losses


def max_drawdown(entry_ts_and_returns):
    """Equal-weighted, non-compounding equity curve (each trade contributes
    its return_pct at 1x size, summed in chronological order by entry_ts).
    Returns the largest peak-to-trough decline in percentage-points, or None
    if there's nothing to order."""
    ordered = sorted((r for r in entry_ts_and_returns if r[0]), key=lambda t: t[0])
    if not ordered:
        return None
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for _ts, ret in ordered:
        equity += ret
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return -worst  # report as a positive magnitude


def summarize(entry_ts_and_returns):
    returns = [r for _ts, r in entry_ts_and_returns]
    if not returns:
        return None
    n = len(returns)
    return {
        "n": n,
        "avg": statistics.mean(returns),
        "median": statistics.median(returns),
        "win_rate": sum(1 for r in returns if r > 0) / n * 100.0,
        "profit_factor": profit_factor(returns),
        "max_drawdown": max_drawdown(entry_ts_and_returns),
    }


def fmt_row(tp, sl, stat, is_live=False):
    flag = "" if stat["n"] >= MIN_TRUSTWORTHY_N else "  (< %d samples - too thin to trust)" % MIN_TRUSTWORTHY_N
    marker = "  <- current live setting" if is_live else ""
    pf = f"{stat['profit_factor']:.2f}" if stat["profit_factor"] is not None else "  n/a"
    dd = f"{stat['max_drawdown']:.1f}%" if stat["max_drawdown"] is not None else "n/a"
    return (f"  TP={tp:>4.0f}%  SL={sl:>4.0f}%   n={stat['n']:<4} "
            f"avg={stat['avg']:+6.1f}%  median={stat['median']:+6.1f}%  "
            f"win_rate={stat['win_rate']:5.1f}%  profit_factor={pf}  max_drawdown={dd}{flag}{marker}")


def main():
    rows = load_rows()
    sequences = build_sequences(rows)
    print(f"Loaded {len(rows)} label rows, {len(sequences)} tokens with at least one checkpoint outcome.\n")
    if not sequences:
        print("No checkpoint outcomes yet - nothing to backtest.")
        return

    # ---- Full grid: take-profit x stop-loss ----
    print("=== Take-profit / stop-loss grid (avg return per rule, all tokens) ===")
    print("Every token is walked with the SAME rule - not cherry-picked per-token.")
    print("Profit factor = gross gains / gross losses (>1 means winners outweigh losers in $ terms).")
    print("Max drawdown = worst peak-to-trough dip of an equal-weighted, non-compounding equity curve.\n")
    results = []
    for tp in TAKE_PROFIT_SWEEP:
        for sl in STOP_LOSS_SWEEP:
            stat = summarize(simulate(sequences, tp, sl))
            if stat is not None:
                results.append((tp, sl, stat))

    live_stat = summarize(simulate(sequences, LIVE_TAKE_PROFIT_PCT, LIVE_STOP_LOSS_PCT))
    print("Current live rule:")
    print(fmt_row(LIVE_TAKE_PROFIT_PCT, LIVE_STOP_LOSS_PCT, live_stat, is_live=True))
    print()

    # ---- Top 10 by average return, trustworthy sample sizes only ----
    trustworthy = [r for r in results if r[2]["n"] >= MIN_TRUSTWORTHY_N]
    ranked = sorted(trustworthy if trustworthy else results, key=lambda r: -r[2]["avg"])
    print(f"Top {min(10, len(ranked))} combinations by average return"
          + ("" if trustworthy else " (NONE reach the trustworthy sample size yet - showing best of what exists)") + ":")
    for tp, sl, stat in ranked[:10]:
        is_live = (tp == LIVE_TAKE_PROFIT_PCT and sl == LIVE_STOP_LOSS_PCT)
        print(fmt_row(tp, sl, stat, is_live=is_live))
    print()

    # ---- Take-profit alone (stop-loss fixed at live 30%) ----
    print(f"=== Take-profit sweep alone (stop-loss fixed at live {LIVE_STOP_LOSS_PCT:.0f}%) ===")
    for tp in TAKE_PROFIT_SWEEP:
        stat = summarize(simulate(sequences, tp, LIVE_STOP_LOSS_PCT))
        if stat is not None:
            print(fmt_row(tp, LIVE_STOP_LOSS_PCT, stat, is_live=(tp == LIVE_TAKE_PROFIT_PCT)))
    print()

    # ---- Stop-loss alone (take-profit fixed at live 15%) ----
    print(f"=== Stop-loss sweep alone (take-profit fixed at live {LIVE_TAKE_PROFIT_PCT:.0f}%) ===")
    for sl in STOP_LOSS_SWEEP:
        stat = summarize(simulate(sequences, LIVE_TAKE_PROFIT_PCT, sl))
        if stat is not None:
            print(fmt_row(LIVE_TAKE_PROFIT_PCT, sl, stat, is_live=(sl == LIVE_STOP_LOSS_PCT)))
    print()

    print("Reminder: no gas/slippage/tax in these numbers (raw price returns only), "
          "and every 'exit' can only land on one of the 5 fixed checkpoints (10m/30m/1h/3h/6h), "
          "not the exact threshold price - so treat this as a rough direction-finder, not a precise backtest.")


if __name__ == "__main__":
    main()
