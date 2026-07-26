#!/usr/bin/env python3
"""
Broker layer - the ONE interface engine.py trades through, with two
interchangeable implementations selected by the TRADING_MODE env var:

  paper (default) - simulated fills against the given price hint, with
        taker fees + slippage applied so the simulated results are NOT
        prettier than reality. Stop-losses are simulated at each hourly
        run (close at the stop level if price fell through it). No network
        calls, no keys, no real money.

  live  - real orders on Crypto.com Exchange via its private REST API
        (HMAC-SHA256 signed requests). Requires CRYPTO_API_KEY and
        CRYPTO_API_SECRET env vars. Put them in GitHub Secrets, NEVER in
        the repo (the repo is public - GitHub Pages serves from it). The
        API key must have TRADE permission only, withdrawals DISABLED.
        Stop-losses are placed as real STOP_LOSS orders on the exchange
        right after the buy fills, so protection lives server-side and
        does not depend on the hourly cron actually running on time.

Both modes run through the exact same engine.py code paths - that is the
whole point: a month of paper results validates the very same system that
later trades real money, not a lookalike.

IMPORTANT: the LiveBroker request/response handling follows Crypto.com
Exchange API v1 docs but has not been exercised against a real funded
account yet. Before ever flipping TRADING_MODE=live, do one supervised
smoke test with a tiny amount (see RUNBOOK.md "Go-live checklist").
"""
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
from decimal import Decimal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTRUMENT_MAP_PATH = os.path.join(BASE_DIR, "data", "instrument_map.json")

EXCHANGE_REST_BASE = "https://api.crypto.com/exchange/v1/"

# ---------------------------------------------------------------------------
# Configuration (env-overridable so paper/live can be tuned without edits)
# ---------------------------------------------------------------------------

TRADING_MODE = os.environ.get("TRADING_MODE", "paper").strip().lower()

# Taker fee per side. Default 0.5% = Crypto.com Exchange base tier taker fee.
# If your account tier / CRO staking gives you a lower fee, set
# TRADING_FEE_PCT accordingly so paper results match your real cost.
FEE_PCT = float(os.environ.get("TRADING_FEE_PCT", "0.5")) / 100.0

# Simulated slippage per side for paper fills (market orders never fill at
# the exact last price). Live fills use the real executed price instead.
SLIPPAGE_PCT = float(os.environ.get("TRADING_SLIPPAGE_PCT", "0.15")) / 100.0

# Stop-loss distance below entry. 8% is wide enough that normal hourly noise
# on majors rarely triggers it, tight enough to cap a single meme-coin
# faceplant well below "wipe out the month".
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "8")) / 100.0

# Kill-switch: if the portfolio balance drops more than this % within 24h,
# or this many consecutive live orders fail, the bot stops OPENING new
# positions (existing ones are still managed/closed) until manually reset.
MAX_DAILY_LOSS_PCT = float(os.environ.get("MAX_DAILY_LOSS_PCT", "5"))
MAX_CONSEC_FAILURES = int(os.environ.get("MAX_CONSEC_FAILURES", "3"))


def _fmt_num(x):
    """Format a number for the exchange API without float artifacts or
    scientific notation ('0.000021' not '2.1e-05')."""
    d = Decimal(str(x))
    s = format(d.normalize(), "f")
    return s


def _round_down_to_tick(value, tick):
    if not tick:
        return value
    v = Decimal(str(value))
    t = Decimal(str(tick))
    if t <= 0:
        return value
    return float((v // t) * t)


# ---------------------------------------------------------------------------
# Paper broker - simulation with honest costs
# ---------------------------------------------------------------------------

class PaperBroker:
    mode = "paper"

    def buy(self, instrument, usd_amount, price_hint):
        """Simulate a market buy spending usd_amount total (fee comes out of
        it, same as a real notional order). Returns a fill dict or None."""
        if not price_hint or price_hint <= 0 or usd_amount <= 0:
            return None
        exec_price = price_hint * (1 + SLIPPAGE_PCT)
        fee = usd_amount * FEE_PCT
        quantity = (usd_amount - fee) / exec_price
        return {"price": exec_price, "quantity": quantity,
                "fee_usd": round(fee, 4), "notional_usd": usd_amount}

    def sell(self, instrument, quantity, price_hint):
        """Simulate a market sell. Returns net proceeds after fee, or None."""
        if not price_hint or price_hint <= 0 or quantity <= 0:
            return None
        exec_price = price_hint * (1 - SLIPPAGE_PCT)
        gross = quantity * exec_price
        fee = gross * FEE_PCT
        return {"price": exec_price, "proceeds_usd": gross - fee,
                "fee_usd": round(fee, 4)}

    def place_stop_loss(self, instrument, quantity, stop_price):
        """No real order to place - paper stops are simulated at each run
        via check_stop_order(). Returns None (no order id)."""
        return None

    def cancel_order(self, instrument, order_id):
        pass

    def check_stop_order(self, position, current_price):
        """Called once per hourly run for every open position. If price has
        fallen to/under the stop level since last run, simulate the stop
        filling at the stop price (or lower, if price gapped below it -
        stops are not a guaranteed floor in reality either)."""
        stop = position.get("stop_price")
        qty = position.get("quantity")
        if not stop or not qty or current_price is None:
            return None
        if current_price > stop:
            return None
        fill_hint = min(stop, current_price)
        return self.sell(position["instrument"], qty, fill_hint)

    def reconcile(self, state):
        """Nothing external to reconcile against in paper mode."""
        return []

    def open_short(self, instrument, usd_amount, price_hint):
        """Simulate opening (or adding to) a perpetual-futures short sized at
        usd_amount notional. Selling into the bid -> adverse slippage vs. a
        long entry, same fee rate."""
        if not price_hint or price_hint <= 0 or usd_amount <= 0:
            return None
        exec_price = price_hint * (1 - SLIPPAGE_PCT)
        fee = usd_amount * FEE_PCT
        quantity = (usd_amount - fee) / exec_price
        return {"price": exec_price, "quantity": quantity,
                "fee_usd": round(fee, 4), "notional_usd": usd_amount}

    def close_short(self, instrument, quantity, price_hint):
        """Simulate buying back `quantity` to close a short. Buying at the
        ask -> adverse slippage, same fee rate."""
        if not price_hint or price_hint <= 0 or quantity <= 0:
            return None
        exec_price = price_hint * (1 + SLIPPAGE_PCT)
        cost = quantity * exec_price
        fee = cost * FEE_PCT
        return {"price": exec_price, "cost_usd": cost + fee, "fee_usd": round(fee, 4)}


# ---------------------------------------------------------------------------
# Live broker - Crypto.com Exchange private REST API
# ---------------------------------------------------------------------------

class BrokerError(Exception):
    pass


class LiveBroker:
    mode = "live"
    MAX_SIG_LEVEL = 3

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self._id = int(time.time() * 1000) % 1_000_000
        # watchlist symbol (BTCUSD) -> real exchange instrument (BTC_USD),
        # written each run by fetch_and_run.py from what actually resolved.
        try:
            with open(INSTRUMENT_MAP_PATH) as f:
                self.instrument_map = json.load(f)
        except (OSError, ValueError):
            self.instrument_map = {}
        self._specs = None  # lazy: instrument -> {qty_tick_size, min_quantity}

    # -- plumbing ----------------------------------------------------------

    def _params_to_str(self, obj, level=0):
        """Crypto.com's documented signature scheme: params flattened by
        sorted key, values concatenated, nested structures recursed."""
        if level >= self.MAX_SIG_LEVEL:
            return str(obj)
        out = ""
        for key in sorted(obj):
            out += key
            v = obj[key]
            if v is None:
                out += "null"
            elif isinstance(v, bool):
                out += str(v).lower()
            elif isinstance(v, list):
                for sub in v:
                    out += self._params_to_str(sub, level + 1)
            elif isinstance(v, dict):
                out += self._params_to_str(v, level + 1)
            else:
                out += str(v)
        return out

    def _request(self, method, params=None):
        params = params or {}
        self._id += 1
        nonce = int(time.time() * 1000)
        payload_str = method + str(self._id) + self.api_key + self._params_to_str(params) + str(nonce)
        sig = hmac.new(self.api_secret.encode(), payload_str.encode(), hashlib.sha256).hexdigest()
        body = json.dumps({
            "id": self._id, "method": method, "api_key": self.api_key,
            "params": params, "nonce": nonce, "sig": sig,
        }).encode()
        req = urllib.request.Request(
            EXCHANGE_REST_BASE + method, data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode()[:300]
            except Exception:
                detail = str(e)
            raise BrokerError(f"{method} HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            raise BrokerError(f"{method} failed: {e}")
        if data.get("code") not in (0, "0", None):
            raise BrokerError(f"{method} rejected: code={data.get('code')} {data.get('message', '')}")
        return data.get("result") or {}

    def _public_get(self, path):
        try:
            with urllib.request.urlopen(EXCHANGE_REST_BASE + path, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            return data.get("result") or {}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            raise BrokerError(f"public {path} failed: {e}")

    def _instrument_specs(self):
        if self._specs is None:
            specs = {}
            result = self._public_get("public/get-instruments")
            for d in result.get("data", []):
                name = d.get("symbol") or d.get("instrument_name")
                if not name:
                    continue
                specs[name] = {
                    "qty_tick_size": d.get("qty_tick_size"),
                    "price_tick_size": d.get("price_tick_size"),
                }
            self._specs = specs
        return self._specs

    def _real_name(self, instrument):
        real = self.instrument_map.get(instrument)
        if not real:
            raise BrokerError(f"no exchange instrument mapping for {instrument} "
                              f"(data/instrument_map.json missing or stale)")
        return real

    def _perp_name(self, instrument):
        """Perpetual futures naming is uniform (no separate lookup file
        needed like spot's BASE_USD/BASE_USDT ambiguity): '{SYMBOL}-PERP'."""
        if not instrument.endswith("USD"):
            raise BrokerError(f"{instrument}: can't derive a perpetual name (expected a *USD symbol)")
        return f"{instrument}-PERP"

    def _round_qty(self, real_name, quantity):
        spec = self._instrument_specs().get(real_name) or {}
        return _round_down_to_tick(quantity, spec.get("qty_tick_size"))

    def _round_price(self, real_name, price):
        spec = self._instrument_specs().get(real_name) or {}
        return _round_down_to_tick(price, spec.get("price_tick_size"))

    def _wait_filled(self, order_id, tries=8, delay=1.5):
        last = None
        for _ in range(tries):
            last = self._request("private/get-order-detail", {"order_id": str(order_id)})
            status = (last.get("status") or "").upper()
            if status == "FILLED":
                return last
            if status in ("REJECTED", "CANCELED", "EXPIRED"):
                raise BrokerError(f"order {order_id} ended {status}")
            time.sleep(delay)
        raise BrokerError(f"order {order_id} not filled after {tries * delay:.0f}s "
                          f"(last status: {last.get('status') if last else 'unknown'})")

    # -- trading interface (same shape as PaperBroker) ---------------------

    def buy(self, instrument, usd_amount, price_hint):
        real = self._real_name(instrument)
        result = self._request("private/create-order", {
            "instrument_name": real, "side": "BUY", "type": "MARKET",
            "notional": _fmt_num(round(usd_amount, 2)),
        })
        detail = self._wait_filled(result.get("order_id"))
        avg_price = float(detail.get("avg_price") or 0)
        qty = float(detail.get("cumulative_quantity") or 0)
        if avg_price <= 0 or qty <= 0:
            raise BrokerError(f"buy {real}: filled but no price/qty in order detail")
        fee = float(detail.get("cumulative_fee") or 0) or usd_amount * FEE_PCT
        return {"price": avg_price, "quantity": qty,
                "fee_usd": round(fee, 4), "notional_usd": usd_amount,
                "order_id": str(result.get("order_id"))}

    def sell(self, instrument, quantity, price_hint):
        real = self._real_name(instrument)
        qty = self._round_qty(real, quantity)
        if qty <= 0:
            raise BrokerError(f"sell {real}: quantity {quantity} rounds to 0")
        result = self._request("private/create-order", {
            "instrument_name": real, "side": "SELL", "type": "MARKET",
            "quantity": _fmt_num(qty),
        })
        detail = self._wait_filled(result.get("order_id"))
        avg_price = float(detail.get("avg_price") or 0)
        filled_qty = float(detail.get("cumulative_quantity") or qty)
        gross = avg_price * filled_qty
        fee = float(detail.get("cumulative_fee") or 0) or gross * FEE_PCT
        return {"price": avg_price, "proceeds_usd": gross - fee,
                "fee_usd": round(fee, 4)}

    def place_stop_loss(self, instrument, quantity, stop_price):
        """Real server-side protection: a STOP_LOSS sell resting on the
        exchange, triggered at stop_price. Failure to place it is reported
        (returns None) but does NOT unwind the buy - the hourly
        check_stop_order() pass still acts as a slower backstop."""
        try:
            real = self._real_name(instrument)
            qty = self._round_qty(real, quantity)
            trigger = self._round_price(real, stop_price)
            result = self._request("private/create-order", {
                "instrument_name": real, "side": "SELL", "type": "STOP_LOSS",
                "quantity": _fmt_num(qty), "ref_price": _fmt_num(trigger),
            })
            return str(result.get("order_id"))
        except BrokerError as e:
            print(f"WARN: stop-loss order for {instrument} failed: {e}", file=sys.stderr)
            return None

    def cancel_order(self, instrument, order_id):
        if not order_id:
            return
        try:
            self._request("private/cancel-order", {"order_id": str(order_id)})
        except BrokerError as e:
            # Usually means it already filled/canceled - reconcile handles that.
            print(f"WARN: cancel {order_id} for {instrument}: {e}", file=sys.stderr)

    def check_stop_order(self, position, current_price):
        """If the resting stop order filled since last run, return its fill.
        If there is no resting stop (placement failed earlier) and price is
        under the stop level, market-sell now as a backstop."""
        stop = position.get("stop_price")
        qty = position.get("quantity")
        if not stop or not qty:
            return None
        order_id = position.get("stop_order_id")
        if order_id:
            try:
                detail = self._request("private/get-order-detail", {"order_id": str(order_id)})
            except BrokerError as e:
                print(f"WARN: stop status check {order_id}: {e}", file=sys.stderr)
                return None
            status = (detail.get("status") or "").upper()
            if status == "FILLED":
                avg_price = float(detail.get("avg_price") or stop)
                filled_qty = float(detail.get("cumulative_quantity") or qty)
                gross = avg_price * filled_qty
                fee = float(detail.get("cumulative_fee") or 0) or gross * FEE_PCT
                return {"price": avg_price, "proceeds_usd": gross - fee,
                        "fee_usd": round(fee, 4)}
            return None
        if current_price is not None and current_price <= stop:
            try:
                return self.sell(position["instrument"], qty, current_price)
            except BrokerError as e:
                print(f"WARN: backstop sell for {position['instrument']} failed: {e}", file=sys.stderr)
                return None
        return None

    def reconcile(self, state):
        """Sanity pass at the start of each run: does the exchange agree we
        exist? Currently a light check - fetch the account balance so auth
        problems surface as a loud note instead of a mysterious failed order
        mid-run. Returns list of human-readable notes."""
        notes = []
        try:
            result = self._request("private/user-balance")
            data = (result.get("data") or [{}])[0]
            total = data.get("total_available_balance")
            if total is not None:
                notes.append(f"LIVE saldo börsil: ${float(total):.2f} vaba")
        except BrokerError as e:
            notes.append(f"⚠️ LIVE reconcile ebaõnnestus (API võti/ühendus?): {e}")
        return notes

    def open_short(self, instrument, usd_amount, price_hint):
        """Open (or add to) a short on the perpetual - a plain SELL market
        order on the *-PERP instrument. Requires the account to have
        derivatives/margin trading enabled (separate eligibility from spot
        on some exchanges/jurisdictions - see RUNBOOK before funding-arb
        go-live)."""
        real = self._perp_name(instrument)
        result = self._request("private/create-order", {
            "instrument_name": real, "side": "SELL", "type": "MARKET",
            "notional": _fmt_num(round(usd_amount, 2)),
        })
        detail = self._wait_filled(result.get("order_id"))
        avg_price = float(detail.get("avg_price") or 0)
        qty = float(detail.get("cumulative_quantity") or 0)
        if avg_price <= 0 or qty <= 0:
            raise BrokerError(f"open_short {real}: filled but no price/qty in order detail")
        fee = float(detail.get("cumulative_fee") or 0) or usd_amount * FEE_PCT
        return {"price": avg_price, "quantity": qty,
                "fee_usd": round(fee, 4), "notional_usd": usd_amount}

    def close_short(self, instrument, quantity, price_hint):
        """Close a short - a plain BUY market order on the *-PERP instrument."""
        real = self._perp_name(instrument)
        qty = self._round_qty(real, quantity)
        if qty <= 0:
            raise BrokerError(f"close_short {real}: quantity {quantity} rounds to 0")
        result = self._request("private/create-order", {
            "instrument_name": real, "side": "BUY", "type": "MARKET",
            "quantity": _fmt_num(qty),
        })
        detail = self._wait_filled(result.get("order_id"))
        avg_price = float(detail.get("avg_price") or 0)
        filled_qty = float(detail.get("cumulative_quantity") or qty)
        cost = avg_price * filled_qty
        fee = float(detail.get("cumulative_fee") or 0) or cost * FEE_PCT
        return {"price": avg_price, "cost_usd": cost + fee, "fee_usd": round(fee, 4)}


# ---------------------------------------------------------------------------

def get_broker():
    """Pick the broker for this run. live requires both key env vars; if
    they are missing we loudly fall back to paper rather than half-run."""
    if TRADING_MODE == "live":
        key = os.environ.get("CRYPTO_API_KEY")
        secret = os.environ.get("CRYPTO_API_SECRET")
        if key and secret:
            return LiveBroker(key, secret)
        print("WARN: TRADING_MODE=live but CRYPTO_API_KEY/CRYPTO_API_SECRET "
              "not set - falling back to PAPER mode.", file=sys.stderr)
    return PaperBroker()
