#!/usr/bin/env python3
"""
Offline analysis over data/memecoin_labels.jsonl - answers "which of our
hand-picked 'good'/'caution' tags actually predict a positive outcome?"

NOT part of the live scanner and never run by the GitHub Actions workflow -
this is a manual tool you run yourself (`python3 analyze_labels.py`) when
you want to check whether the heuristic in scan_memecoins.py's
classify_recommendation()/getTags() (dashboard) is still just a guess or
has started to earn its keep.

With only a handful of labeled trades this will be noisy - that's expected,
not a bug. The point of this script is to make it obvious WHEN you have
enough data to trust a number, not to produce a verdict on day one. Same
lesson as the original momentum-signal work: don't declare a pattern found
on a training set with nothing held out to check it against.
"""
import json
import statistics
import sys
from collections import defaultdict

LABELS_PATH = "data/memecoin_labels.jsonl"
CHECKPOINT_OFFSETS = [10, 30, 60, 180, 360]
OFFSET_LABEL = {10: "10min", 30: "30min", 60: "1h", 180: "3h", 360: "6h"}

# Minimum sample size before a stat is shown with any confidence framing -
# below this, print the number but flag it as too thin to trust.
MIN_TRUSTWORTHY_N = 30


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


def summarize(returns):
    if not returns:
        return None
    n = len(returns)
    avg = statistics.mean(returns)
    win_rate = sum(1 for r in returns if r > 0) / n * 100.0
    median = statistics.median(returns)
    return {"n": n, "avg": avg, "median": median, "win_rate": win_rate}


def fmt(stat, label):
    if stat is None:
        return f"  {label:<28} no data yet"
    flag = "" if stat["n"] >= MIN_TRUSTWORTHY_N else "  (too few samples to trust - want >= %d)" % MIN_TRUSTWORTHY_N
    return (f"  {label:<28} n={stat['n']:<5} avg={stat['avg']:+6.1f}%  "
            f"median={stat['median']:+6.1f}%  win_rate={stat['win_rate']:5.1f}%{flag}")


def main():
    rows = load_rows()
    entries = {r["pool_id"]: r for r in rows if r.get("row_type") == "entry"}
    checkpoints = [r for r in rows if r.get("row_type") == "checkpoint"]

    print(f"Loaded {len(entries)} entries, {len(checkpoints)} checkpoint rows from {LABELS_PATH}\n")
    if not checkpoints:
        print("No checkpoint outcomes yet - nothing to analyze. Check back once trades have had time to mature (checkpoints land at 10min/30min/1h/3h/6h after entry).")
        return

    # ---- Overall outcome distribution per checkpoint offset ----
    print("=== Overall outcome by checkpoint offset (every candidate, not just recommended) ===")
    by_offset = defaultdict(list)
    for c in checkpoints:
        if c.get("return_pct") is not None:
            by_offset[c["offset_minutes"]].append(c["return_pct"])
    for off in CHECKPOINT_OFFSETS:
        print(fmt(summarize(by_offset.get(off, [])), OFFSET_LABEL[off]))
    print()

    # ---- Feature-importance-lite: does each tag actually help? ----
    # Re-derives the same conditions classify_recommendation() checks, split
    # by whether the entry had that feature, then compares forward returns.
    # This is exactly the "let the data tell you which good/caution tags are
    # real" step the plan called for once there's enough history.
    print("=== Per-feature conditional outcome (1h checkpoint) ===")
    print("Compares average 1h return WHEN a feature was present at entry vs when it wasn't.\n")

    h1_by_pool = {c["pool_id"]: c["return_pct"] for c in checkpoints
                  if c["offset_minutes"] == 60 and c.get("return_pct") is not None}

    def split_by(predicate, label):
        with_feature, without_feature = [], []
        for pool_id, entry in entries.items():
            ret = h1_by_pool.get(pool_id)
            if ret is None:
                continue
            (with_feature if predicate(entry) else without_feature).append(ret)
        print(f"[{label}]")
        print(fmt(summarize(with_feature), "  present"))
        print(fmt(summarize(without_feature), "  absent"))
        print()

    split_by(lambda e: (e.get("entry_mcap_usd") or 0) >= 1_000_000, "mcap >= $1M at entry")
    split_by(lambda e: (e.get("entry_mcap_usd") or 0) < 200_000, "mcap < $200K at entry")
    split_by(lambda e: (e.get("rank_velocity") or 0) > 0, "rank_velocity > 0 (rank climbing)")
    split_by(lambda e: (e.get("h1_accel") or 0) > 0, "h1_accel > 0 (momentum accelerating)")
    split_by(lambda e: e.get("concentration_pct") is not None and e["concentration_pct"] < 20, "concentration < 20% (Solana top10 / EVM owner+creator scale differ - see scan_memecoins.py)")
    split_by(lambda e: e.get("buys_per_buyer_h1") is not None and e["buys_per_buyer_h1"] >= 3.5, "buys_per_buyer_h1 >= 3.5 (wash-trade risk: few unique wallets behind the buy volume)")

    old_rows = [e for e in entries.values() if "rank_velocity" not in e]
    if old_rows:
        print(f"({len(old_rows)} entries predate the rank_velocity/h1_accel/concentration fields being "
              f"saved on entry rows - they're skipped in those splits, not counted as 'absent'.)")


if __name__ == "__main__":
    main()
