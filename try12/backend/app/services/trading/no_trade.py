"""
Explicit no-trade decisions (design §12, R12.1-R12.8).

Two closed enumerations and the recorder. Every "why didn't it trade" answer
is a ``no_trade_decisions`` row with a ``Stage`` and a ``ReasonCode``; the
reason text is untruncated. ``cycle_stats`` is the positive evidence that the
per-candle cycle ran even when nothing was rejected.

Observability must not break trading: the isolated recorders open their own
short-lived session, log a write failure at WARNING and return None.

Imports are deliberately narrow (app.runtime.context, app.db.models.risk, the
stdlib, sqlalchemy) so app/strategies/base.py can import this module without
an import cycle. A static test enforces it.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterator, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models.risk import CycleStats, NoTradeDecision
from app.runtime.context import ctx, sessions

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    CYCLE = "CYCLE"                 # per-candle gate, no strategy evaluated
    STRATEGY_GATE = "STRATEGY_GATE" # strategy disabled / cooldown
    SIGNAL_GATE = "SIGNAL_GATE"     # aggregator, regime direction
    RISK_GATE = "RISK_GATE"         # caps, halts, cooldowns
    SELECTION = "SELECTION"         # option selection and its filters
    SIZING = "SIZING"
    ENTRY = "ENTRY"                 # post-sizing aborts (capital race, cap race)
    DEGRADATION = "DEGRADATION"     # not a rejection; a recorded data/stat gap


class ReasonCode(str, Enum):
    # CYCLE
    OUTSIDE_SESSION_WINDOW = "OUTSIDE_SESSION_WINDOW"
    AFTER_ENTRY_CUTOFF = "AFTER_ENTRY_CUTOFF"
    INSUFFICIENT_CANDLES = "INSUFFICIENT_CANDLES"
    ATR_BELOW_FLOOR = "ATR_BELOW_FLOOR"
    VIX_ABOVE_MAX = "VIX_ABOVE_MAX"
    VIX_BELOW_MIN = "VIX_BELOW_MIN"
    NO_PREV_DAY_SNAPSHOT = "NO_PREV_DAY_SNAPSHOT"
    REGIME_UNAVAILABLE = "REGIME_UNAVAILABLE"
    REGIME_NO_TRADE = "REGIME_NO_TRADE"
    TRANSITION_COOLDOWN = "TRANSITION_COOLDOWN"
    FEED_UNAVAILABLE = "FEED_UNAVAILABLE"
    # STRATEGY_GATE
    STRATEGY_USER_DISABLED = "STRATEGY_USER_DISABLED"
    STRATEGY_REGIME_DISABLED = "STRATEGY_REGIME_DISABLED"
    STRATEGY_DAILY_HALT = "STRATEGY_DAILY_HALT"
    EXPIRY_DAY_SL_TOO_TIGHT = "EXPIRY_DAY_SL_TOO_TIGHT"
    STRATEGY_COOLDOWN = "STRATEGY_COOLDOWN"
    # SIGNAL_GATE
    AGGREGATOR_SAME_DIRECTION = "AGGREGATOR_SAME_DIRECTION"
    AGGREGATOR_CONFLICT = "AGGREGATOR_CONFLICT"
    REGIME_DIRECTION_MISMATCH = "REGIME_DIRECTION_MISMATCH"
    REGIME_STATE_UNAVAILABLE = "REGIME_STATE_UNAVAILABLE"
    RR_INSUFFICIENT = "RR_INSUFFICIENT"
    # RISK_GATE
    NOT_PAPER_MODE = "NOT_PAPER_MODE"
    KILL_SWITCH = "KILL_SWITCH"
    DAILY_LOSS_CAP = "DAILY_LOSS_CAP"
    LOSING_STREAK_HALT = "LOSING_STREAK_HALT"
    POST_LOSS_COOLDOWN = "POST_LOSS_COOLDOWN"
    DAILY_TRADE_CAP = "DAILY_TRADE_CAP"
    MAX_OPEN_POSITIONS = "MAX_OPEN_POSITIONS"
    EXPOSURE_CAP = "EXPOSURE_CAP"
    REGIME_MULTIPLIER_ZERO = "REGIME_MULTIPLIER_ZERO"
    CONFIG_INVALID = "CONFIG_INVALID"
    # SELECTION
    NO_SUITABLE_EXPIRY = "NO_SUITABLE_EXPIRY"
    NO_INSTRUMENTS = "NO_INSTRUMENTS"
    TICK_UNAVAILABLE = "TICK_UNAVAILABLE"
    QUOTE_UNAVAILABLE = "QUOTE_UNAVAILABLE"
    QUOTE_STALE = "QUOTE_STALE"
    OI_UNAVAILABLE = "OI_UNAVAILABLE"
    VOLUME_UNAVAILABLE = "VOLUME_UNAVAILABLE"
    LIQUIDITY_TOO_LOW = "LIQUIDITY_TOO_LOW"
    VOLUME_INSUFFICIENT = "VOLUME_INSUFFICIENT"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    SLIPPAGE_TOO_HIGH = "SLIPPAGE_TOO_HIGH"
    PREMIUM_ABOVE_AFFORDABLE = "PREMIUM_ABOVE_AFFORDABLE"
    CONTRACT_UNSUITABLE = "CONTRACT_UNSUITABLE"
    SCORE_BELOW_MIN = "SCORE_BELOW_MIN"
    NO_OPTION_AVAILABLE = "NO_OPTION_AVAILABLE"
    OPTION_MOMENTUM_FAIL = "OPTION_MOMENTUM_FAIL"
    ATR_VS_SL_DISTANCE = "ATR_VS_SL_DISTANCE"
    # SIZING
    KELLY_NO_EDGE = "KELLY_NO_EDGE"
    SIZED_TO_ZERO = "SIZED_TO_ZERO"
    LOT_UNAFFORDABLE = "LOT_UNAFFORDABLE"
    INSUFFICIENT_CAPITAL = "INSUFFICIENT_CAPITAL"
    INITIAL_RISK_NON_POSITIVE = "INITIAL_RISK_NON_POSITIVE"
    COSTS_EXCEED_EDGE = "COSTS_EXCEED_EDGE"
    PLANNED_RISK_CAP = "PLANNED_RISK_CAP"
    AGGREGATE_RISK_CAP = "AGGREGATE_RISK_CAP"
    INVALID_CONTRACT = "INVALID_CONTRACT"
    # ENTRY
    DUPLICATE_OPEN_POSITION = "DUPLICATE_OPEN_POSITION"
    CAPITAL_RACE_LOST = "CAPITAL_RACE_LOST"
    DAILY_TRADE_CAP_RACE = "DAILY_TRADE_CAP_RACE"
    HALT_ENGAGED_MID_ENTRY = "HALT_ENGAGED_MID_ENTRY"
    UNEXPECTED_ENTRY_ERROR = "UNEXPECTED_ENTRY_ERROR"
    # DEGRADATION
    PERF_WRITE_FAILED = "PERF_WRITE_FAILED"
    MTM_DEGRADED = "MTM_DEGRADED"
    FALLBACK_FILL = "FALLBACK_FILL"
    DATA_GAP = "DATA_GAP"
    DETECTOR_ERROR = "DETECTOR_ERROR"
    REGIME_MULTIPLIER_ERROR = "REGIME_MULTIPLIER_ERROR"


def _v(x) -> str:
    return getattr(x, "value", x)


# ── per-cycle tally (cycle_stats counters) ───────────────────────────────────

@dataclass
class CycleTally:
    """Counters for one run_cycle. Recorders running inside the cycle's task
    (SignalBus.publish, paper_engine) bump it through the ContextVar."""
    strategies_evaluated: int = 0
    signals_emitted: int = 0
    rejections_recorded: int = 0
    oi_iv_context_missing: int = 0
    pending: list[dict] = field(default_factory=list)   # rows the runner writes at cycle end


_tally: ContextVar[Optional[CycleTally]] = ContextVar("no_trade_cycle_tally", default=None)


@contextmanager
def cycle_tally() -> Iterator[CycleTally]:
    t = CycleTally()
    token = _tally.set(t)
    try:
        yield t
    finally:
        _tally.reset(token)


def current_tally() -> Optional[CycleTally]:
    return _tally.get()


def note_signal_emitted() -> None:
    t = _tally.get()
    if t is not None:
        t.signals_emitted += 1


def _count_rejection(stage) -> None:
    t = _tally.get()
    if t is not None and _v(stage) != Stage.DEGRADATION.value:
        t.rejections_recorded += 1


# ── recorder ─────────────────────────────────────────────────────────────────

async def record(db, rc, *, stage: Stage, code: ReasonCode, reason: str,
                 context: dict, underlying: Optional[str] = None,
                 strategy_name: Optional[str] = None, signal_id: Optional[int] = None,
                 position_id: Optional[int] = None) -> int:
    """Write one row on *db* (the caller commits) and return its id.
    *rc* is the RuntimeContext: run_id, trading day and timestamp come from it."""
    row = NoTradeDecision(
        run_id=rc.run_id,
        trading_day=rc.clock.trading_day(),
        decided_at=rc.clock.now_utc(),
        stage=_v(stage),
        code=_v(code),
        underlying=underlying,
        strategy_name=strategy_name,
        signal_id=signal_id,
        position_id=position_id,
        reason=reason or _v(code),
        context=dict(context or {}),
    )
    db.add(row)
    await db.flush()
    _count_rejection(stage)
    return row.id


async def record_isolated(*, stage: Stage, code: ReasonCode, reason: str,
                          context: Optional[dict] = None, underlying: Optional[str] = None,
                          strategy_name: Optional[str] = None, signal_id: Optional[int] = None,
                          position_id: Optional[int] = None) -> Optional[int]:
    """record() in its own short-lived session. Never raises: a write failure
    is a WARNING and returns None."""
    try:
        rc = ctx()
        async with sessions()() as db:
            rid = await record(db, rc, stage=stage, code=code, reason=reason,
                               context=context or {}, underlying=underlying,
                               strategy_name=strategy_name, signal_id=signal_id,
                               position_id=position_id)
            await db.commit()
            return rid
    except Exception as exc:
        logger.warning("no-trade record %s/%s failed (non-fatal): %s",
                       _v(stage), _v(code), exc)
        return None


async def record_signal_gate(*, code: ReasonCode, reason: str, underlying: Optional[str],
                             strategy_name: Optional[str]) -> Optional[int]:
    """SIGNAL_GATE row in its own session; WARNING and None on write failure.
    The only recorder callable from app/strategies/ (SignalBus.publish)."""
    return await record_isolated(stage=Stage.SIGNAL_GATE, code=code, reason=reason,
                                 underlying=underlying, strategy_name=strategy_name)


async def record_degradation(run_id: str, code: ReasonCode, *, position_id: Optional[int] = None,
                             context: Optional[dict] = None, underlying: Optional[str] = None,
                             reason: Optional[str] = None) -> Optional[int]:
    """One DEGRADATION row per call (bounded by its call sites, §12)."""
    try:
        rc = ctx()
        async with sessions()() as db:
            rid = await _insert_degradation(db, rc, run_id, code, position_id=position_id,
                                            context=context, underlying=underlying,
                                            reason=reason)
            await db.commit()
            return rid
    except Exception as exc:
        logger.warning("no-trade degradation %s failed (non-fatal): %s", _v(code), exc)
        return None


async def record_degradation_once(run_id: str, code: ReasonCode, *, position_id: Optional[int],
                                  context: Optional[dict] = None,
                                  underlying: Optional[str] = None,
                                  reason: Optional[str] = None) -> Optional[int]:
    """Dedupes on (run_id, trading_day, code, position_id). Returns None on a repeat.
    The pre-check is the mechanism; the partial unique index is the backstop."""
    try:
        rc = ctx()
        day = rc.clock.trading_day()
        async with sessions()() as db:
            q = select(NoTradeDecision.id).where(
                NoTradeDecision.run_id == run_id,
                NoTradeDecision.trading_day == day,
                NoTradeDecision.code == _v(code),
                NoTradeDecision.stage == Stage.DEGRADATION.value,
            )
            q = q.where(NoTradeDecision.position_id.is_(None) if position_id is None
                        else NoTradeDecision.position_id == position_id)
            if (await db.execute(q.limit(1))).first() is not None:
                return None
            rid = await _insert_degradation(db, rc, run_id, code, position_id=position_id,
                                            context=context, underlying=underlying,
                                            reason=reason)
            await db.commit()
            return rid
    except IntegrityError:
        return None                          # a concurrent writer got there first
    except Exception as exc:
        logger.warning("no-trade degradation %s failed (non-fatal): %s", _v(code), exc)
        return None


async def _insert_degradation(db, rc, run_id: str, code: ReasonCode, *, position_id, context,
                              underlying, reason) -> int:
    row = NoTradeDecision(
        run_id=run_id,
        trading_day=rc.clock.trading_day(),
        decided_at=rc.clock.now_utc(),
        stage=Stage.DEGRADATION.value,
        code=_v(code),
        underlying=underlying,
        position_id=position_id,
        reason=reason or _v(code),
        context=dict(context or {}),
    )
    db.add(row)
    await db.flush()
    return row.id


# ── cycle_stats ──────────────────────────────────────────────────────────────

async def upsert_cycle_stats(db, rc, *, underlying: str, trading_day: date, at: datetime,
                             candles: int, strategies: int, signals: int, rejections: int,
                             oi_iv_missing: int) -> None:
    """Add one cycle's counters to the (run_id, trading_day, underlying) row."""
    row = (await db.execute(select(CycleStats).where(
        CycleStats.run_id == rc.run_id,
        CycleStats.trading_day == trading_day,
        CycleStats.underlying == underlying,
    ))).scalar_one_or_none()
    if row is None:
        row = CycleStats(run_id=rc.run_id, trading_day=trading_day, underlying=underlying,
                         candles_evaluated=0, strategies_evaluated=0, signals_emitted=0,
                         rejections_recorded=0, oi_iv_context_missing=0, first_cycle_at=at)
        db.add(row)
    row.candles_evaluated = (row.candles_evaluated or 0) + candles
    row.strategies_evaluated = (row.strategies_evaluated or 0) + strategies
    row.signals_emitted = (row.signals_emitted or 0) + signals
    row.rejections_recorded = (row.rejections_recorded or 0) + rejections
    row.oi_iv_context_missing = (row.oi_iv_context_missing or 0) + oi_iv_missing
    row.last_cycle_at = at
    await db.flush()


def as_context(**values: Any) -> dict:
    """JSON-safe context dict (drops None, stringifies non-primitive values)."""
    out = {}
    for k, v in values.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool, list, dict)) else str(v)
    return out
