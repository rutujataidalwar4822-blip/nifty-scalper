"""
Fill and cost model (design §15, R14) — pure functions, no I/O, shared by
paper trading and replay. Every rate comes from TradingParams.

The governing rule: entries may be refused, exits may not.
  * Entry fills at ask × (1 + quote_slippage_pct/100), rounded UP to the ₹0.05
    tick, when the quote is valid ('ASK'). Without a valid quote the candidate
    should already have been rejected at selection; the LTP fallback
    ltp × (1 + slippage_pct/100) ('LTP_FALLBACK') remains for that path.
  * Exits always fill (an open position must always be closable), per the
    decision table in ``exit_fill``.

Statutory charges, NSE index options (buyer). Sources (checked 2026-10-06;
verify periodically — rates change with the Union Budget and exchange
circulars):
  https://zerodha.com/charges/                       (F&O - Options column)
  https://support.fyers.in/portal/en/kb/articles/what-are-the-statutory-charges-on-trades-at-fyers-stt-gst-sebi-turnover-fee-ipft-etc
  STT 0.15% of premium, sell side only; NSE exchange transaction charge
  0.03553% of turnover; SEBI ₹10 / crore of turnover; stamp duty 0.003% on the
  buy side; GST 18% on (brokerage + exchange txn + SEBI fee). IPFT
  (₹0.01/crore) is ignored as negligible. These are approximations for paper
  P&L; a broker contract note is the source of truth for real trades.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

TICK = 0.05
EXIT_FALLBACK_SLIPPAGE = 0.001      # trigger × (1 − 0.1%) when no valid bid / TARGET limit
# 0.30 / 0.15 == 1.9999999999999998 in IEEE-754; without this epsilon the 1:2 gate
# would reject every signal in the system (design §10.1, §24).
RR_EPSILON = 1e-6


def tick_age_seconds(tick: Optional[dict], now: datetime) -> Optional[float]:
    """Age of a Redis tick from its ``ts`` field at *now*. The live format is
    aware UTC ISO (``…+00:00``, TickData.received_at); a naive ISO string
    (legacy Redis values, replay) is read as UTC. None when unknown. Pure: the
    caller supplies the ctx clock reading."""
    if not tick:
        return None
    ts = tick.get("ts")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (now - t).total_seconds()


def rr_ok(sl_pct: float, target_pct: float, min_rr: float) -> tuple[bool, Optional[float]]:
    """(passes, un-stretched rr). A non-positive stop distance has no defined
    R:R: it passes here and is refused as INITIAL_RISK_NON_POSITIVE at sizing."""
    if sl_pct is None or sl_pct <= 0:
        return True, None
    rr = target_pct / sl_pct
    return rr + RR_EPSILON >= min_rr, rr


def floor_tick(x: float) -> float:
    return round(math.floor(round(x / TICK, 6)) * TICK, 2)


def ceil_tick(x: float) -> float:
    return round(math.ceil(round(x / TICK, 6)) * TICK, 2)


def quote_valid(bid, ask, age_s: Optional[float], params) -> bool:
    """Fresh (age <= stale_tick_seconds), uncrossed (ask >= bid, both > 0) and
    no wider than quote_max_spread_pct of mid."""
    try:
        bid = float(bid or 0)
        ask = float(ask or 0)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in (bid, ask)) or bid <= 0 or ask <= 0 or ask < bid:
        return False
    if age_s is None or not math.isfinite(age_s) or not 0 <= age_s <= params.stale_tick_seconds:
        return False
    mid = (bid + ask) / 2
    return (ask - bid) / mid * 100 <= params.quote_max_spread_pct + 1e-9


def entry_fill(ltp: float, bid, ask, age_s: Optional[float], params) -> tuple[float, str]:
    """(price, entry_fill_source)."""
    if quote_valid(bid, ask, age_s, params):
        return ceil_tick(float(ask) * (1 + params.quote_slippage_pct / 100)), "ASK"
    return round(ltp * (1 + (params.slippage_pct or 0) / 100), 2), "LTP_FALLBACK"


def exit_fill(trigger_price: float, ltp: Optional[float], bid, ask, age_s: Optional[float],
              reason: str, params) -> tuple[float, str]:
    """(price, exit_fill_source). *ltp* is the observed price that triggered the
    exit (None only for a no-tick close). Never better than the stop for an SL.

    Decision table, first matching row wins:
      TARGET                                → trigger × (1 − 0.1%)        TRIGGER_SLIPPED
      valid quote                           → floor_tick(bid × (1 − qsp)) BID
      reason ends in _STALE and ltp is None → trigger × (1 − 0.1%)        LAST_KNOWN
      otherwise                             → trigger × (1 − 0.1%)        TRIGGER_SLIPPED
    Gap-through, after the table, for SL / SL_STALE with an observed ltp:
      price = min(base, ltp); LTP_GAP when ltp is the lower of the two.
    """
    slipped = round(trigger_price * (1 - EXIT_FALLBACK_SLIPPAGE), 2)
    if reason == "TARGET":
        price, source = slipped, "TRIGGER_SLIPPED"
    elif quote_valid(bid, ask, age_s, params):
        price, source = floor_tick(float(bid) * (1 - params.quote_slippage_pct / 100)), "BID"
    elif reason.endswith("_STALE") and ltp is None:
        price, source = slipped, "LAST_KNOWN"
    else:
        price, source = slipped, "TRIGGER_SLIPPED"

    if reason.startswith("SL") and ltp is not None and float(ltp) < price:
        price, source = round(float(ltp), 2), "LTP_GAP"
    return price, source


def entry_levels(actual_entry: float, sl_pct: float, target_pct: float,
                 min_rr: float) -> tuple[float, float, float, bool]:
    """(stop, target, realised_rr, target_stretched) on the fill price.

    The stop rounds DOWN and the target UP to the ₹0.05 tick, so rounding can
    only improve the R:R (design §10.1). The signal has already passed the 1:2
    gate on its own percentages; if tick rounding of an off-tick fill still
    leaves the realised R:R a hair under the floor, the target is lifted to
    meet it and the position is flagged ``target_stretched``. That is the only
    place a target is ever moved, and only after the gate has passed.
    """
    sl = floor_tick(actual_entry * (1.0 - sl_pct))
    tgt = ceil_tick(actual_entry * (1.0 + target_pct))
    risk = actual_entry - sl
    stretched = False
    if risk > 0:
        rr = (tgt - actual_entry) / risk
        if rr + RR_EPSILON < min_rr:
            tgt = ceil_tick(actual_entry + risk * min_rr)
            rr = (tgt - actual_entry) / risk
            stretched = True
    else:
        rr = 0.0
    return sl, tgt, rr, stretched


def brokerage(lots: int, params) -> float:
    """Per leg: brokerage_per_lot_per_leg × lots."""
    return round(params.brokerage_per_lot_per_leg * max(int(lots or 0), 0), 2)


def statutory_charges(buy_value: float, sell_value: float, brokerage_amount: float,
                      params) -> float:
    """STT + exchange txn + SEBI + stamp + GST for the legs given (brokerage itself
    excluded; *brokerage_amount* is the GST base's brokerage)."""
    buy_value = max(buy_value or 0.0, 0.0)
    sell_value = max(sell_value or 0.0, 0.0)
    turnover = buy_value + sell_value
    stt = sell_value * params.stt_sell_pct / 100
    exch = turnover * params.exchange_txn_pct / 100
    sebi = turnover * params.sebi_per_crore / 1e7
    stamp = buy_value * params.stamp_buy_pct / 100
    gst = ((brokerage_amount or 0.0) + exch + sebi) * params.gst_pct / 100
    return round(stt + exch + sebi + stamp + gst, 2)


def net_pnl(gross: float, brokerage_amount: float, charges: float) -> float:
    """The identity every position and summary satisfies: gross − brokerage − charges."""
    return round(gross - brokerage_amount - charges, 2)


def expected_round_trip_cost(entry_price: float, quantity: int, lots: int, params) -> float:
    """Brokerage on both legs plus statutory charges, exiting at the entry price."""
    brok = 2 * brokerage(lots, params)
    value = entry_price * quantity
    return round(brok + statutory_charges(value, value, brok, params), 2)
