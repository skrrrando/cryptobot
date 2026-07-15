#!/usr/bin/env python3
"""
Cryptobot scoring engine.

Two-stage pipeline, run once per hour by the scheduled task:

  1. `screen`   - input: freshly fetched tickers (from Crypto.com) for the
                  watchlist. Computes a momentum+trend score per token,
                  applies noise filtering (needs >=1 prior snapshot to
                  confirm a trend), and outputs the top candidates that are
                  worth a closer (WebSearch) look.

  2. `finalize` - input: the candidates from step 1 plus short hype notes
                  gathered via WebSearch for each candidate, plus current
                  prices for any recommendations that are due for a 24h/7d
                  follow-up check. Computes final 0-100 score + risk label
                  + "why" explanation, applies cooldown/dedup so the same
                  token isn't re-alerted every hour for no reason, logs new
                  recommendations, scores due follow-ups, nudges the
                  adaptive thresholds, (re)writes dashboard.html, and
                  prints a short chat summary (this is the notification).

All state lives in data/state.json (persisted in the user's project folder,
NOT the ephemeral session scratchpad) so history/learning survives across
hourly runs.
"""
import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
WATCHLIST_PATH = os.path.join(BASE_DIR, "watchlist.json")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")

HISTORY_KEEP = 48  # keep last 48 hourly snapshots per instrument (~2 days)

DEFAULT_STATE = {
    "history": {},          # instrument -> [ {ts, price, change_24h, volume_value}, ... ]
    "alerted": {},          # instrument -> {last_score, last_ts, last_risk}
    "thresholds": {         # adaptive, per risk bucket, tuned by adapt_thresholds()
        "screen_score": 65,
        "green_hit_bar": 60,
        "yellow_hit_bar": 65,
        "red_hit_bar": 75
    },
    "pending_followups": [],   # awaiting 24h and/or 7d outcome check
    "completed": [],          # followups fully resolved (both checks done or expired)
    "run_log": [],            # short history of each run (for the dashboard)
    "next_id": 1
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


def load_state():
    st = load_json(STATE_PATH, DEFAULT_STATE)
    for k, v in DEFAULT_STATE.items():
        st.setdefault(k, json.loads(json.dumps(v)))
    return st


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
    ts = now_ts()

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

        if raw_score >= state["thresholds"]["screen_score"]:
            candidates.append({
                "instrument": inst,
                "price": price,
                "change_24h": change_24h,
                "volume_value": volume_value,
                "category": category,
                "momentum_score": m_score,
                "trend_bonus": bonus,
                "trend_note": trend_note,
                "raw_score": raw_score,
                "liquidity": liquidity_bucket(volume_value)
            })

    candidates.sort(key=lambda c: c["raw_score"], reverse=True)
    save_state(state)
    save_json(out_path, candidates[:10])  # cap: only deep-dive the top 10
    print(f"STAGE1: {len(tickers)} jälgitavat, {len(candidates)} ületas läve "
          f"({state['thresholds']['screen_score']}), top {min(10, len(candidates))} saadetud edasi.")
    for c in candidates[:10]:
        print(f"  - {c['instrument']}: raw_score={c['raw_score']} "
              f"({c['momentum_score']}+{c['trend_bonus']}) [{c['category']}/{c['liquidity']}]")


def save_state(state):
    save_json(STATE_PATH, state)


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
    for their 24h or 7d outcome check."""
    still_pending = []
    resolved_notes = []
    for rec in state["pending_followups"]:
        inst = rec["instrument"]
        age_h = (now_ts() - rec["ts"]) / 3600
        price_now = current_prices.get(inst)

        if not rec.get("result_24h") and age_h >= 24 and price_now:
            ret = (price_now - rec["price_at_call"]) / rec["price_at_call"] * 100
            bar = state["thresholds"].get(rec["risk"] + "_hit_bar", 65)
            rec["result_24h"] = {"price": price_now, "return_pct": round(ret, 2),
                                  "hit": ret * 100 >= (bar - 100) if False else ret >= 3}
            resolved_notes.append(f"{inst} 24h: {ret:+.1f}%")

        if not rec.get("result_7d") and age_h >= 24 * 7 and price_now:
            ret = (price_now - rec["price_at_call"]) / rec["price_at_call"] * 100
            rec["result_7d"] = {"price": price_now, "return_pct": round(ret, 2),
                                 "hit": ret >= 8}
            resolved_notes.append(f"{inst} 7d: {ret:+.1f}%")

        if rec.get("result_24h") and rec.get("result_7d"):
            state["completed"].append(rec)
        else:
            still_pending.append(rec)

    state["pending_followups"] = still_pending
    return resolved_notes


def adapt_thresholds(state):
    """Simple, explainable adaptive control loop: look at hit-rate per risk
    bucket among completed recs (using the 7d result when available, else
    24h). If a bucket is underperforming, require a higher score next time
    (raise its hit bar / raise screen threshold slightly). If it's
    overperforming, loosen slightly. Only acts once there's enough sample
    size to mean something."""
    completed = state["completed"]
    if len(completed) < 20:
        return None
    by_risk = {"green": [], "yellow": [], "red": []}
    for rec in completed:
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

    alerts = []
    all_scored = []
    ts = now_ts()

    for c in candidates:
        inst = c["instrument"]
        note = hype_notes.get(inst)
        bonus, forced_risk, why_hype = hype_adjustment(note)
        final_score = clamp(c["raw_score"] + bonus, 0, 100)
        risk = risk_label(c["category"], c["liquidity"], c["change_24h"], forced_risk)

        why = (f"Momentum {c['momentum_score']}/100 (24h {c['change_24h']*100:+.1f}%), "
               f"{c['trend_note']}. Likviidsus: {c['liquidity']} "
               f"(maht ${c['volume_value']:,.0f}). {why_hype}")

        entry = {"instrument": inst, "score": final_score, "risk": risk,
                 "why": why, "price": c["price"], "category": c["category"]}
        all_scored.append(entry)

        if should_alert(state, inst, final_score, risk):
            alerts.append(entry)
            state["alerted"][inst] = {"last_score": final_score, "last_ts": ts, "last_risk": risk}
            rec_id = state["next_id"]
            state["next_id"] += 1
            state["pending_followups"].append({
                "id": rec_id, "instrument": inst, "ts": ts, "score": final_score,
                "risk": risk, "why": why, "price_at_call": c["price"],
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
            lines.append(f"{emoji} {a['instrument']}: {a['score']}/100 - {a['why']}")
    else:
        wl = load_watchlist()
        lines.append(f"Cryptobot skann ({datetime.now(timezone.utc).strftime('%d.%m %H:%M')} UTC) - kontrolliti {len(wl)} tokenit, ükski ei ületanud praegu läve ({state['thresholds']['screen_score']}/100). Bot töötab, lihtsalt hetkel pole miski silma jäänud.")
    if followup_notes:
        lines.append("Tagasivaade: " + "; ".join(followup_notes))
    if threshold_notes:
        lines.append("Mudel kohandus: " + "; ".join(threshold_notes))

    summary = "\n".join(lines)
    with open(out_summary_path, "w") as f:
        f.write(summary)
    print(summary)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(state):
    completed = state["completed"]
    pending = state["pending_followups"]
    run_log = list(reversed(state["run_log"][-30:]))

    total_hits_24h = [r["result_24h"]["hit"] for r in completed if r.get("result_24h")]
    total_hits_24h += [r["result_24h"]["hit"] for r in pending if r.get("result_24h")]
    hit_rate_24h = (sum(total_hits_24h) / len(total_hits_24h) * 100) if total_hits_24h else None

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
        history_rows += f"""
        <tr>
          <td>{dt}</td>
          <td class="mono">{r['instrument']}</td>
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

    thresholds = state["thresholds"]

    html = f"""<!DOCTYPE html>
<html lang="et">
<head>
<meta charset="UTF-8">
<title>Cryptobot Dashboard</title>
<style>
  :root {{
    --bg: #0b0f19; --card: #131a2a; --border: #232d42; --text: #e5e9f0;
    --muted: #8b94a8; --accent: #6366f1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 32px; line-height: 1.5;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: 28px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .stat-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px 20px; min-width: 140px;
  }}
  .stat-card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
  .stat-card .value {{ font-size: 26px; font-weight: 600; margin-top: 4px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; margin-bottom: 24px; overflow-x: auto;
  }}
  .card h2 {{ font-size: 15px; margin: 0 0 14px; color: var(--text); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th {{ text-align: left; color: var(--muted); font-weight: 500; padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  .mono {{ font-family: ui-monospace, monospace; }}
  .why {{ color: var(--muted); max-width: 420px; }}
  .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }}
  .thresholds {{ display: flex; gap: 20px; flex-wrap: wrap; font-size: 13px; color: var(--muted); }}
  .thresholds b {{ color: var(--text); }}
</style>
</head>
<body>
  <h1>🤖 Cryptobot Dashboard</h1>
  <div class="subtitle">Uuendatud {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M')} UTC · V1 (momentum + trend + veebiotsingu hype-kinnitus, kohandub tagasiside põhjal)</div>

  <div class="stats">
    <div class="stat-card"><div class="label">Aktiivseid soovitusi (ootel)</div><div class="value">{len(pending)}</div></div>
    <div class="stat-card"><div class="label">Lõpetatud soovitusi</div><div class="value">{len(completed)}</div></div>
    <div class="stat-card"><div class="label">24h tabamusprotsent</div><div class="value">{f'{hit_rate_24h:.0f}%' if hit_rate_24h is not None else '–'}</div></div>
    <div class="stat-card"><div class="label">Käivitusi kokku</div><div class="value">{len(state['run_log'])}</div></div>
  </div>

  <div class="card">
    <h2>Praegused mudeli lävendid (kohandatud automaatselt tagasiside põhjal)</h2>
    <div class="thresholds">
      <div>Skanni lävi: <b>{thresholds['screen_score']}</b></div>
      <div>🟢 roheline tabamuslävi: <b>{thresholds['green_hit_bar']}</b></div>
      <div>🟡 kollane tabamuslävi: <b>{thresholds['yellow_hit_bar']}</b></div>
      <div>🔴 punane tabamuslävi: <b>{thresholds['red_hit_bar']}</b></div>
    </div>
  </div>

  <div class="card">
    <h2>Soovituste ajalugu (uusimad enne)</h2>
    <table>
      <tr><th>Millal</th><th>Token</th><th>Skoor</th><th>Risk</th><th>24h</th><th>7p</th><th>Põhjendus</th></tr>
      {history_rows or '<tr><td colspan="7" style="color:var(--muted)">Veel andmeid pole.</td></tr>'}
    </table>
  </div>

  <div class="card">
    <h2>Käivituste logi</h2>
    <table>
      <tr><th>Millal</th><th>Kandidaate</th><th>Alerte</th><th>Tagasivaade</th><th>Mudeli kohandused</th></tr>
      {run_rows or '<tr><td colspan="5" style="color:var(--muted)">Veel käivitusi pole.</td></tr>'}
    </table>
  </div>
</body>
</html>"""
    with open(DASHBOARD_PATH, "w") as f:
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
