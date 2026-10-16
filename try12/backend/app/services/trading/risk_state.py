"""
Derived risk state (design §9.2-§9.4). Everything here is computed from
persisted rows — ``paper_positions`` and ``risk_halts`` — so a process restart
cannot reset a cap, a halt or a streak (bugs 7, 8, 9). Nothing reads an
aggregate table: ``daily_pnl_summary`` can miss a close (PERF_WRITE_FAILED), and
a cap that reads a table that can silently miss a loss is not a cap.

Every function takes a session and a ``run_id`` (or reads it from ``ctx()``), so
replay runs the identical code under its own run id and clock.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select

from app.db.models.paper_trade import DailyPnLSummary, PaperPosition
from app.db.models.risk import RiskHalt
from app.runtime.context import ctx, market, now_utc, params, trading_day
from app.services.trading import fills, no_trade
from app.services.trading.session_times import trading_day_bounds_utc

logger = logging.getLogger(__name__)

HALT_DAILY_LOSS = "DAILY_LOSS"
HALT_LOSING_STREAK = "LOSING_STREAK"
HALT_STRATEGY_STREAK = "STRATEGY_STREAK"

REALISED_MISMATCH_TOLERANCE = 0.01


# ── trade count and capital in use ───────────────────────────────────────────

async def derived_trade_count(db, run_id: str, day: date) -> int:
    """Positions inserted for *run_id* on IST trading day *day*. A rejected or
    rolled-back entry never inserted a row, so it never consumed a slot."""
    start, end = trading_day_bounds_utc(day)
    return int((await db.execute(
        select(func.count(PaperPosition.id)).where(
            PaperPosition.run_id == run_id,
            PaperPosition.entry_time >= start,
            PaperPosition.entry_time < end,
        )
    )).scalar_one() or 0)


async def open_capital_used(db, run_id: str) -> float:
    """Capital held by this run's open positions. The only definition of
    capital in use: every writer of ``trade_amount_used`` keeps it equal to the
    capital still deployed (insert sets it, a partial close reduces it, the
    final close clears ``is_open``)."""
    return float((await db.execute(
        select(func.coalesce(func.sum(PaperPosition.trade_amount_used), 0.0))
        .where(PaperPosition.run_id == run_id, PaperPosition.is_open == True)  # noqa: E712
    )).scalar_one())


async def open_position_count(db, run_id: str) -> int:
    return int((await db.execute(
        select(func.count(PaperPosition.id))
        .where(PaperPosition.run_id == run_id, PaperPosition.is_open == True)  # noqa: E712
    )).scalar_one() or 0)


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    free_capital: float
    open_exposure: float
    exposure_cap: float
    exposure_headroom: float
    open_planned_risk: float
    mtm_degraded: bool


async def account_snapshot(db, run_id: str) -> AccountSnapshot:
    from app.services.trading.accounting import open_pnl, planned_risk
    cfg, p = ctx().cfg, params()
    closed_net = float((await db.execute(select(func.coalesce(func.sum(PaperPosition.net_pnl), 0.0))
        .where(PaperPosition.run_id == run_id, PaperPosition.is_open.is_(False)))).scalar_one())
    rows = (await db.execute(select(PaperPosition).where(
        PaperPosition.run_id == run_id, PaperPosition.is_open.is_(True)))).scalars().all()
    exposure = await open_capital_used(db, run_id)
    cash = cfg.capital + closed_net
    equity = cash
    risk, degraded = 0.0, False
    for pos in rows:
        mark, bad, _, _ = await _mark(pos)
        values = open_pnl(pos, mark, p)
        equity += values["estimated_net_pnl"]
        cash += values["realized_gross_pnl"] - values["booked_costs"]
        degraded = degraded or bad
        # Reserve original stop risk even after trailing: never assume a
        # guaranteed fill at a tightened stop, or silently free that risk.
        stop = pos.initial_stop_loss or pos.stop_loss
        risk += planned_risk(float(pos.actual_entry), float(stop),
                             int(pos.quantity), int(pos.lot_size), p)["planned_risk"]
    cap = min(cfg.capital, max(equity, 0.0)) * cfg.open_exposure_cap_pct / 100.0
    return AccountSnapshot(round(equity, 2), round(max(cash - exposure, 0.0), 2),
                           exposure, round(cap, 2), round(max(cap - exposure, 0.0), 2),
                           round(risk, 2), degraded)


# ── daily loss (realised + staleness-aware MTM, ratcheted) ───────────────────

@dataclass(frozen=True)
class DailyLoss:
    realised: float                 # derived from closed paper_positions
    realised_cached: float          # the daily_pnl_summary figure, for display only
    realised_mismatch: bool
    unrealised: float               # MTM of open positions, net of exit charges
    combined: float
    mtm_degraded: bool
    degraded_positions: list[int]
    threshold_abs: float
    threshold_pct: float
    breached: bool
    breach_source: Optional[str]    # 'ABS' | 'PCT' | 'RATCHET'
    halt_engaged_at: Optional[datetime] = None


async def _last_known_price(pos_id: int) -> Optional[float]:
    """In-memory last fresh LTP, else the value persisted to pos:{id} in Redis."""
    from app.services.trading.paper_engine import paper_engine   # lazy: import cycle
    known = paper_engine._last_known.get(pos_id)
    if known:
        return float(known[0])
    try:
        cached = await market().get(f"pos:{pos_id}")
    except Exception:
        cached = None
    if isinstance(cached, dict) and cached.get("last_ltp"):
        return float(cached["last_ltp"])
    return None


async def _mark(pos) -> tuple[float, bool, Optional[str], Optional[float]]:
    """(mark price, degraded, mark_source, tick age). Fresh valid quote → bid,
    fresh LTP → ltp; stale or missing → last known; none → entry (zero
    unrealised, still flagged degraded)."""
    p = params()
    tick, ltp = None, None
    try:
        tick = await market().get_tick(pos.token)
        ltp = await market().get_ltp(pos.token)
    except Exception as exc:
        logger.debug("MTM price read failed for pos %s: %s", pos.id, exc)
    age = fills.tick_age_seconds(tick, now_utc()) if tick else None
    from app.services.trading.accounting import valid_price
    fresh = valid_price(ltp) and age is not None and 0 <= age <= p.stale_tick_seconds
    if fresh:
        bid, ask = (tick or {}).get("best_bid"), (tick or {}).get("best_ask")
        if fills.quote_valid(bid, ask, age, p):
            return float(bid), False, "BID", age
        return float(ltp), False, "LTP", age
    known = await _last_known_price(pos.id)
    if valid_price(known):
        return known, True, "LAST_KNOWN", age
    return float(pos.actual_entry or pos.entry_price or 0.0), True, "ENTRY", age


def _exit_charges(mark: float, quantity: int, lots: int) -> float:
    p = params()
    brok = fills.brokerage(lots, p)
    return brok + fills.statutory_charges(0.0, mark * quantity, brok, p)


async def realised_today(db, run_id: str, day: date) -> float:
    start, end = trading_day_bounds_utc(day)
    return float((await db.execute(
        select(func.coalesce(func.sum(PaperPosition.net_pnl), 0.0)).where(
            PaperPosition.run_id == run_id,
            PaperPosition.is_open == False,  # noqa: E712
            PaperPosition.exit_time >= start,
            PaperPosition.exit_time < end,
        )
    )).scalar_one())


async def _active_halt(db, run_id: str, day: date, halt_type: str,
                       scope: Optional[str] = None) -> Optional[RiskHalt]:
    q = select(RiskHalt).where(RiskHalt.run_id == run_id, RiskHalt.trading_day == day,
                               RiskHalt.halt_type == halt_type)
    q = q.where(RiskHalt.scope.is_(None) if scope is None else RiskHalt.scope == scope)
    return (await db.execute(q.order_by(RiskHalt.id).limit(1))).scalar_one_or_none()


async def _engage(db, run_id: str, day: date, halt_type: str, *, scope: Optional[str],
                  reason: str, context: dict) -> None:
    """Write the halt row once per (run, day, type, scope); the caller commits."""
    if await _active_halt(db, run_id, day, halt_type, scope) is not None:
        return
    db.add(RiskHalt(run_id=run_id, trading_day=day, halt_type=halt_type, scope=scope,
                    reason=reason, context=context, engaged_at=now_utc()))
    await db.flush()


async def daily_loss_figure(db, *, commit: bool = True) -> DailyLoss:
    """Realised (derived) + staleness-aware MTM, against the frozen RunConfig
    thresholds, with the per-day ratchet persisted in ``risk_halts``.

    Open positions keep running their own exits; this only decides whether new
    entries are allowed. One-sided: a positive figure can never breach.
    """
    rc = ctx()
    run_id, day = rc.run_id, trading_day()
    cfg = rc.cfg

    realised = await realised_today(db, run_id, day)
    cached = float((await db.execute(
        select(func.coalesce(func.sum(DailyPnLSummary.net_pnl), 0.0)).where(
            DailyPnLSummary.run_id == run_id, DailyPnLSummary.date == day)
    )).scalar_one())

    open_rows = (await db.execute(
        select(PaperPosition).where(PaperPosition.run_id == run_id,
                                    PaperPosition.is_open == True)  # noqa: E712
    )).scalars().all()

    unrealised, degraded = 0.0, []
    degradations = []
    for pos in open_rows:
        mark, is_degraded, source, age = await _mark(pos)
        from app.services.trading.accounting import open_pnl
        unrealised += open_pnl(pos, mark, params())["estimated_net_pnl"]
        if is_degraded:
            degraded.append(pos.id)
            degradations.append((pos, source, age))
    unrealised = round(unrealised, 2)

    for pos, source, age in (degradations if commit else []):
        await no_trade.record_degradation_once(
            run_id, no_trade.ReasonCode.MTM_DEGRADED, position_id=pos.id,
            underlying=pos.underlying,
            reason=f"pos {pos.id}: no fresh price, MTM marked at {source}",
            context=no_trade.as_context(last_valid_age_s=age, mark_source=source),
        )

    combined = round(realised + unrealised, 2)
    breach_source: Optional[str] = None
    if combined <= -abs(cfg.max_daily_loss):
        breach_source = "ABS"
    elif combined < 0 and abs(combined) / cfg.capital * 100.0 >= cfg.max_daily_loss_pct:
        breach_source = "PCT"

    halt = await _active_halt(db, run_id, day, HALT_DAILY_LOSS)
    if halt is not None and halt.released_at is None:
        breach_source, engaged_at = "RATCHET", halt.engaged_at
    elif halt is not None:                         # operator released it for the day
        breach_source, engaged_at = None, halt.engaged_at
    elif breach_source is not None:
        await _engage(db, run_id, day, HALT_DAILY_LOSS, scope=None,
                      reason=f"daily loss {combined:.2f} breached ({breach_source})",
                      context=no_trade.as_context(realised=realised, unrealised=unrealised,
                                                  combined=combined, source=breach_source))
        if commit:
            await db.commit()
        engaged_at = now_utc()
        logger.warning("DAILY_LOSS halt engaged for %s %s: combined %.2f (%s)",
                       run_id, day, combined, breach_source)
    else:
        engaged_at = None

    return DailyLoss(
        realised=round(realised, 2), realised_cached=round(cached, 2),
        realised_mismatch=abs(realised - cached) > REALISED_MISMATCH_TOLERANCE,
        unrealised=unrealised, combined=combined, mtm_degraded=bool(degraded),
        degraded_positions=degraded, threshold_abs=float(cfg.max_daily_loss),
        threshold_pct=float(cfg.max_daily_loss_pct),
        breached=breach_source is not None, breach_source=breach_source,
        halt_engaged_at=engaged_at,
    )


# ── losing streaks and the post-loss cooldown ────────────────────────────────

@dataclass(frozen=True)
class StreakState:
    consecutive_losses: int
    per_strategy_losses: dict[str, int]
    last_loss_exit_at: Optional[datetime]
    cooldown_until: Optional[datetime]
    global_halt: bool
    halted_strategies: frozenset[str]
    triggering_position_ids: list[int] = field(default_factory=list)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


async def streak_state(db, run_id: str, day: date) -> StreakState:
    """Fold the day's closed positions, ordered by exit time. A win resets the
    global and its strategy's count; a loss increments both; a flat trade
    (abs(net) <= flat_epsilon, ₹1) is neutral. A breakeven exit that cost money
    is a LOSS here (it cost money), though R-flat for Kelly (§9.5)."""
    p = params()
    start, end = trading_day_bounds_utc(day)
    rows = (await db.execute(
        select(PaperPosition.id, PaperPosition.strategy_name, PaperPosition.net_pnl,
               PaperPosition.exit_time)
        .where(PaperPosition.run_id == run_id, PaperPosition.is_open == False,  # noqa: E712
               PaperPosition.exit_time >= start, PaperPosition.exit_time < end)
        .order_by(PaperPosition.exit_time, PaperPosition.id)
    )).all()

    eps = p.flat_epsilon
    consecutive, per, last_loss, trig = 0, {}, None, []
    for pid, strat, net, exit_time in rows:
        net = float(net or 0.0)
        if net > eps:
            consecutive = 0
            per[strat] = 0
            trig = []
        elif net < -eps:
            consecutive += 1
            per[strat] = per.get(strat, 0) + 1
            last_loss = _aware(exit_time)
            trig.append(pid)
        # flat: neither increments nor resets

    global_halt = consecutive >= p.losing_streak_halt_n
    halted = frozenset(s for s, n in per.items() if n >= p.strategy_streak_halt_n)

    released = (await db.execute(
        select(RiskHalt.halt_type, RiskHalt.scope).where(
            RiskHalt.run_id == run_id, RiskHalt.trading_day == day,
            RiskHalt.released_at.is_not(None),
            RiskHalt.halt_type.in_([HALT_LOSING_STREAK, HALT_STRATEGY_STREAK]))
    )).all()
    if any(t == HALT_LOSING_STREAK for t, _ in released):
        global_halt = False
    halted = frozenset(s for s in halted
                       if not any(t == HALT_STRATEGY_STREAK and sc == s for t, sc in released))

    cooldown_until = (last_loss + timedelta(minutes=p.post_loss_cooldown_minutes)
                      if last_loss is not None else None)
    return StreakState(
        consecutive_losses=consecutive, per_strategy_losses=per, last_loss_exit_at=last_loss,
        cooldown_until=cooldown_until, global_halt=global_halt, halted_strategies=halted,
        triggering_position_ids=trig,
    )


async def engage_streak_halts(db, run_id: str, day: date, state: StreakState) -> None:
    """Record the halts the derived state implies (audit trail and the hook an
    operator release attaches to). The derived state stays authoritative."""
    wrote = False
    if state.global_halt:
        before = await _active_halt(db, run_id, day, HALT_LOSING_STREAK)
        if before is None:
            await _engage(db, run_id, day, HALT_LOSING_STREAK, scope=None,
                          reason=f"{state.consecutive_losses} consecutive losses",
                          context={"position_ids": state.triggering_position_ids})
            wrote = True
    for strat in sorted(state.halted_strategies):
        if await _active_halt(db, run_id, day, HALT_STRATEGY_STREAK, strat) is None:
            await _engage(db, run_id, day, HALT_STRATEGY_STREAK, scope=strat,
                          reason=f"{state.per_strategy_losses.get(strat)} consecutive losses",
                          context={})
            wrote = True
    if wrote:
        await db.commit()
