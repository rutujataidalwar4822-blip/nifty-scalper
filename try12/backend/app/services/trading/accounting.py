"""Position accounting without changing stored field meanings.

quantity is the open remainder (or final-leg quantity on legacy closed rows).
gross_pnl is realized gross only; brokerage includes entry and executed exit
legs; statutory_charges books both legs of each exited tranche. partial_pnl
is a legacy tranche figure, NOT an additional component of position net P&L.
"""
from __future__ import annotations

import math

from app.services.trading import fills


def valid_price(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def remaining_quantity(pos) -> int:
    return int(pos.quantity or 0) if pos.is_open else 0


def open_pnl(pos, mark: float, p) -> dict:
    qty = remaining_quantity(pos)
    entry = float(pos.actual_entry)
    if not valid_price(mark) or not valid_price(entry) or qty < 0:
        raise ValueError("invalid position mark, entry or remaining quantity")
    realized = float(pos.gross_pnl or 0.0)
    unrealized = (mark - entry) * qty
    booked = float(pos.brokerage or 0.0) + float(pos.statutory_charges or 0.0)
    lots = qty // pos.lot_size if pos.lot_size else 0
    exit_brokerage = fills.brokerage(lots, p)
    # Entry brokerage is already booked, but entry statutory charges for the
    # open remainder are only persisted when that tranche exits.
    pending = exit_brokerage + fills.statutory_charges(
        entry * qty, mark * qty, 2 * fills.brokerage(lots, p), p)
    total = round(realized + unrealized, 2)
    return dict(realized_gross_pnl=realized, unrealized_gross_pnl=round(unrealized, 2),
                total_gross_pnl=total, booked_costs=round(booked, 2),
                estimated_pending_costs=round(pending, 2),
                estimated_net_pnl=round(total - booked - pending, 2))


def planned_risk(entry: float, stop: float, quantity: int, lot_size: int, p) -> dict:
    if (not valid_price(entry) or not valid_price(stop) or stop >= entry
            or isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0
            or isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size <= 0
            or quantity % lot_size):
        raise ValueError("entry > stop > 0 and positive whole-lot quantity required")
    lots = quantity // lot_size
    stop_loss = (entry - stop) * quantity
    costs = fills.expected_round_trip_cost(entry, quantity, lots, p)
    gap = entry * quantity * p.stop_gap_buffer_pct / 100.0
    return dict(stop_risk=round(stop_loss, 2), estimated_costs=costs,
                gap_buffer=round(gap, 2), planned_risk=round(stop_loss + costs + gap, 2))
