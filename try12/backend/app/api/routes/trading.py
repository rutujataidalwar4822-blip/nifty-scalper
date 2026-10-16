"""
Trading API routes:
  GET  /trading/mode                    — current mode
  POST /trading/mode/paper              — switch to paper
  POST /trading/mode/live               — switch to live (requires confirm)
  GET  /trading/config                  — get full config
  PUT  /trading/config                  — update config
  PUT  /trading/config/strategy-flag    — enable/disable a strategy
  GET  /trading/signals                 — recent signals
  GET  /trading/signals/diagnostics     — why signals are (not) firing / converting
  GET  /trading/positions               — open + closed positions
  POST /trading/positions/{id}/close    — manual close a position
  GET  /trading/pnl                     — daily P&L summary
  GET  /trading/pnl/history             — multi-day P&L
  GET  /trading/capital                 — capital breakdown
  PUT  /trading/capital/trade-amount    — set ₹ per trade
  PUT  /trading/capital/set             — set total capital
  POST /trading/capital/reset-daily     — reset daily trade counter
  GET  /trading/stats                   — daily stats
  POST /trading/option-selector         — best option for a direction
  GET  /trading/market-study/intraday   — intraday features
  GET  /trading/market-study/daily      — daily snapshots
  GET  /trading/market-study/vix        — VIX history
  POST /trading/kill-switch             — halt new entries + square off all (paper)
  GET  /trading/kill-switch             — kill switch status
  POST /trading/kill-switch/release     — lift the halt (admin key)
  POST /trading/halt                    — engage a MANUAL halt (key only if configured)
  POST /trading/halt/release            — release a halt for today (admin key)
  WS   /trading/ws/signals              — live signal stream

Authentication (design §17.4) is asymmetric: engaging a halt is always allowed;
releasing one, resetting the daily counter and clearing a run's trades need
X-API-Key and return 503 until ADMIN_API_KEY is configured. Every query against
a run-shared table takes ``run_id`` (default PAPER). Other routes stay
unauthenticated (requirements, out of scope 10).
"""
import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.guards import (
    LIVE_RUN_ID, optional_admin_key, require_admin_key, run_id_query, validate_run_id,
)
from app.db.base import get_db
from app.db.models.paper_trade import (
    StrategySignal, PaperPosition, DailyPnLSummary, TradingConfig,
    TradingMode, SignalStatus,
)
from app.services.trading.mode_manager import ConfigValidationError
from app.db.models.market_study import IntradayMarketFeature, MarketDailySnapshot, VIXData
from app.schemas.diagnostics import SignalDiagnosticsResponse
from app.services.trading.mode_manager import mode_manager
from app.services.trading.paper_engine import paper_engine
from app.services.trading.option_selector import option_selector
from app.strategies.base import Direction
from app.strategies.names import ALL_STRATEGY_NAMES
from app.core.config import settings
from app.runtime.context import params as runtime_params

router = APIRouter(prefix="/trading", tags=["trading"])
logger = logging.getLogger(__name__)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ConfigUpdate(BaseModel):
    max_daily_loss:       Optional[float]   = None
    max_daily_loss_pct:   Optional[float]   = None
    max_open_positions:   Optional[int]     = None
    max_lots_per_trade:   Optional[int]     = None
    capital:              Optional[float]   = None
    slippage_pct:         Optional[float]   = None
    square_off_time:      Optional[str]     = None
    # ── new capital/trade-sizing fields ──
    trade_amount:         Optional[float]   = None   # ₹ per trade
    max_trades_per_day:   Optional[int]     = None
    study_phase:          Optional[bool]    = None   # flip to False to start paper trading
    sizing_policy:        Optional[str] = None
    planned_risk_pct:     Optional[float] = None
    aggregate_risk_pct:   Optional[float] = None
    stop_gap_buffer_pct:  Optional[float] = None

class StrategyFlagUpdate(BaseModel):
    strategy_name: str
    enabled:       bool

class OptionSelectorRequest(BaseModel):
    direction:   str          # BUY or SELL
    underlying:  str          # NIFTY or BANKNIFTY
    spot_price:  float
    expiry:      Optional[str] = None   # YYYY-MM-DD
    # The setup the strategies actually trade (0.15 / 0.30 = 1:2), so the
    # diagnostic ranks candidates against the same R:R gate as production.
    sl_pct:      float = 0.15
    target_pct:  float = 0.30


# ── Mode ──────────────────────────────────────────────────────────────────────

@router.get("/mode")
async def get_mode():
    cfg = await mode_manager.get_config()
    return {
        "mode":                cfg.mode if cfg else "PAPER",
        "live_mode_confirmed": cfg.live_mode_confirmed if cfg else False,
    }

@router.post("/mode/paper")
async def switch_paper():
    return await mode_manager.switch_to_paper()

@router.post("/mode/live")
async def switch_live(confirm: bool = Query(False)):
    result = await mode_manager.switch_to_live(confirm=confirm)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── Config ────────────────────────────────────────────────────────────────────

# RunConfig keys are frozen at process start (design §14.1); everything else that
# overlays TradingParams is re-read within params_cache_ttl_s.
RUN_CONFIG_KEYS = ("capital", "max_trades_per_day", "max_open_positions", "max_lots_per_trade",
                   "max_daily_loss", "max_daily_loss_pct", "open_exposure_cap_pct",
                   "square_off_time", "entry_cutoff_buffer_min")
# Kept in the payload for compatibility, read by nothing (design §5, R18.4).
DEPRECATED_KEYS = {"regime_transition_window": True}


def _effective_from(keys) -> dict:
    from app.runtime.context import params as _params
    ttl = _params().params_cache_ttl_s
    return {k: ("next_process_start" if k in RUN_CONFIG_KEYS else f"within_{ttl}s")
            for k in keys}


def _pending_restart(row) -> dict:
    """Every RunConfig key whose stored value differs from the running ctx().cfg."""
    from app.runtime.context import ctx
    cfg = ctx().cfg
    out = {}
    for k in RUN_CONFIG_KEYS:
        stored = getattr(row, k, None)
        if stored is None:
            continue
        running = getattr(cfg, k)
        differs = (str(stored) != str(running)) if isinstance(running, str) \
            else abs(float(stored) - float(running)) > 1e-9
        if differs:
            out[k] = {"stored": stored, "running": running}
    return out


def _convert_trade_amount(updates: dict, row) -> None:
    """trade_amount is a display cache since 0013; the stored knob is
    fixed_fraction_pct = 100 × trade_amount / capital (design §14.1 table)."""
    if updates.get("trade_amount") is None:
        return
    capital = float(updates.get("capital") or row.capital)
    if capital <= 0:
        raise HTTPException(status_code=400, detail="capital must be > 0")
    fraction = 100.0 * float(updates["trade_amount"]) / capital
    updates["fixed_fraction_pct"] = round(fraction, 6)
    updates["trade_amount"] = round(fraction / 100.0 * capital, 2)


def _write_error(exc: ValueError) -> HTTPException:
    if isinstance(exc, ConfigValidationError):
        return HTTPException(status_code=400, detail={"violations": exc.violations})
    return HTTPException(status_code=409, detail=str(exc))


@router.get("/config")
async def get_config():
    import dataclasses as _dc
    from app.runtime.context import ctx, params as _params

    cfg = await mode_manager.get_config()
    if not cfg:
        raise HTTPException(status_code=404, detail="Config not found")
    capital, used, daily_trades = await mode_manager.capital_figures()
    p = _params()
    violations = p.validate()
    port = ctx().params
    columns = {c.key: getattr(cfg, c.key) for c in TradingConfig.__table__.columns
               if c.key not in ("used_capital", "daily_trades_count", "mode", "updated_at",
                                "halted_at")}
    return {
        **columns,
        "mode":                cfg.mode,
        "max_daily_loss":      cfg.max_daily_loss,
        "max_daily_loss_pct":  cfg.max_daily_loss_pct,
        "max_open_positions":  cfg.max_open_positions,
        "max_lots_per_trade":  cfg.max_lots_per_trade,
        "capital":             cfg.capital,
        "trade_amount":        cfg.trade_amount,
        "used_capital":        round(used, 2),
        "available_capital":   round(capital - used, 2),
        "slippage_pct":        cfg.slippage_pct,
        "square_off_time":     cfg.square_off_time,
        "strategy_flags":      cfg.strategy_flags,
        "live_mode_confirmed": cfg.live_mode_confirmed,
        "study_phase":         cfg.study_phase,
        "max_trades_per_day":  cfg.max_trades_per_day,
        "daily_trades_count":  daily_trades,
        "trading_halted":      cfg.trading_halted,
        "halted_at":           cfg.halted_at.isoformat() if cfg.halted_at else None,
        "params": {
            "resolved": json.loads(json.dumps(_dc.asdict(p), default=str)),
            "cache_age_s": round(getattr(port, "cache_age_s", 0.0), 1),
        },
        "run_config": json.loads(json.dumps(_dc.asdict(ctx().cfg), default=str)),
        "pending_restart": _pending_restart(cfg),
        "config_valid": not violations,
        "config_violations": violations,
        "deprecated": DEPRECATED_KEYS,
    }

@router.put("/config")
async def update_config(body: ConfigUpdate):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    row = await mode_manager.get_config()
    if row is not None:
        _convert_trade_amount(updates, row)
    try:
        cfg = await mode_manager.update_config(**updates)
    except ValueError as exc:
        raise _write_error(exc) from exc
    return {"status": "updated", "mode": cfg.mode,
            "effective_from": _effective_from(updates),
            "pending_restart": _pending_restart(cfg)}

@router.put("/config/strategy-flag")
async def update_strategy_flag(body: StrategyFlagUpdate):
    from app.runtime.context import params as _params
    if body.strategy_name not in ALL_STRATEGY_NAMES:
        raise HTTPException(status_code=400,
                            detail=f"strategy_name must be one of {sorted(ALL_STRATEGY_NAMES)}")
    await mode_manager.update_strategy_flag(body.strategy_name, body.enabled)
    return {"status": "updated", "strategy": body.strategy_name, "enabled": body.enabled,
            "effective_within_s": _params().params_cache_ttl_s}


# ── Capital endpoints ─────────────────────────────────────────────────────────

@router.get("/capital")
async def get_capital():
    """
    Full capital breakdown:
      capital, used_capital, available_capital,
      trade_amount, realised_pnl, utilisation_pct,
      circuit_breaker status, daily trade count.
    """
    return await mode_manager.get_capital_summary()


class TradeAmountUpdate(BaseModel):
    trade_amount: float       # ₹ per trade, e.g. 10000
    capital:      Optional[float] = None   # total funds (optional, update if provided)


@router.put("/capital/trade-amount")
async def set_trade_amount(body: TradeAmountUpdate):
    """
    Set the ₹ amount to deploy per trade.
    Optionally update total capital at the same time.

    Example: trade_amount=10000 with Nifty option at ₹150 (lot 50):
      cost_per_lot = 150 × 50 = ₹7,500
      lots = floor(10000 / 7500) = 1
      capital_deployed = ₹7,500
    """
    from app.runtime.context import params as _params

    if body.trade_amount <= 0:
        raise HTTPException(status_code=400, detail="trade_amount must be > 0")

    updates: dict = {"trade_amount": body.trade_amount}
    if body.capital is not None:
        if body.capital <= 0:
            raise HTTPException(status_code=400, detail="capital must be > 0")
        updates["capital"] = body.capital

    row = await mode_manager.get_config()
    if row is None:
        raise HTTPException(status_code=404, detail="Config not found")
    _convert_trade_amount(updates, row)       # → fixed_fraction_pct, the stored knob
    try:
        cfg = await mode_manager.update_config(**updates)
    except ValueError as exc:
        raise _write_error(exc) from exc

    # Preview: calculate lots for current ATM option price (approximate)
    preview = _trade_amount_preview(body.trade_amount, cfg.max_lots_per_trade)
    capital, used, _n = await mode_manager.capital_figures()

    return {
        "status":           "updated",
        "trade_amount":     cfg.trade_amount,
        "capital":          cfg.capital,
        "available_capital": round(capital - used, 2),
        "preview":          preview,
        "fixed_fraction_pct": cfg.fixed_fraction_pct,
        "effective_from":   _effective_from(updates),
        "effective_within_s": _params().params_cache_ttl_s,
    }


@router.put("/capital/set")
async def set_capital(capital: float = Query(..., gt=0, description="Total paper trading capital in ₹")):
    """Update total capital (funds available for trading). Capital is a RunConfig
    key: the running process keeps its value until the next start (design §14.1)."""
    try:
        cfg = await mode_manager.update_config(capital=capital)
    except ValueError as exc:
        raise _write_error(exc) from exc
    running, used, _n = await mode_manager.capital_figures()
    return {
        "status":          "updated",
        "capital":         cfg.capital,
        "used_capital":    round(used, 2),
        "available":       round(running - used, 2),
        "trade_amount":    cfg.trade_amount,
        "effective_from":  "next_process_start",
        "pending":         {"capital": cfg.capital, "running_capital": running},
    }


@router.post("/capital/reset-daily")
async def reset_daily_counter(operator: str = Depends(require_admin_key)):
    """Reset the daily trades DISPLAY counter. The trade cap itself is derived from
    persisted positions and cannot be reset here (design §9.2). Admin key required."""
    await mode_manager.reset_daily_counter()
    logger.warning("Daily trade display counter reset by operator %s", operator)
    return {"status": "reset", "daily_trades_count": 0}


def _trade_amount_preview(trade_amount: float, max_lots: int) -> list[dict]:
    """
    Preview how many lots ₹trade_amount buys at various typical option prices.
    Useful for the frontend to show the user what they'd get.
    """
    from app.services.trading.paper_engine import _calc_lots
    result = []
    for ltp in [50, 80, 100, 150, 200, 300, 500]:
        for lot_size in [50, 15]:   # Nifty=50, BankNifty=15 (revised Nov 2024)
            cost_per_lot = ltp * lot_size
            # 0 lots = unaffordable → paper engine would SKIP such a signal
            lots = min(max_lots, _calc_lots(trade_amount, ltp, lot_size)[0])
            result.append({
                "ltp":            ltp,
                "lot_size":       lot_size,
                "lots":           lots,
                "capital_used":   round(lots * cost_per_lot, 2),
                "capital_needed": cost_per_lot,
            })
    return result


# ── Daily stats ───────────────────────────────────────────────────────────────

@router.get("/stats")
async def daily_stats():
    return await mode_manager.get_daily_stats()


# ── Signals ───────────────────────────────────────────────────────────────────

@router.get("/signals")
async def get_signals(
    strategy:    Optional[str]  = None,
    underlying:  Optional[str]  = None,
    status:      Optional[str]  = None,
    today_only:  bool           = Query(True,  description="Only return today's signals (default true)"),
    limit:       int            = Query(50, ge=1, le=500),
    run_id:      str            = Depends(run_id_query),
    db: AsyncSession = Depends(get_db),
):
    q = select(StrategySignal).where(StrategySignal.run_id == run_id) \
        .order_by(desc(StrategySignal.signal_time))
    if strategy:
        q = q.where(StrategySignal.strategy_name == strategy)
    if underlying:
        q = q.where(StrategySignal.underlying == underlying.upper())
    if status:
        try:
            status_enum = SignalStatus(status.upper())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"status must be one of: {', '.join(s.value for s in SignalStatus)}",
            )
        q = q.where(StrategySignal.status == status_enum)
    if today_only:
        # Filter to today's IST date using timezone-aware bounds.
        # signal_time is stored as UTC with timezone — convert today IST to UTC bounds.
        from datetime import timezone as _tz, timedelta as _td
        IST = _tz(_td(hours=5, minutes=30))
        today_ist = datetime.now(IST).date()
        day_start_utc = datetime.combine(today_ist, datetime.min.time()).replace(
            tzinfo=IST
        ).astimezone(_tz.utc)
        q = q.where(StrategySignal.signal_time >= day_start_utc)
    q = q.limit(limit)
    result = await db.execute(q)
    rows = result.scalars().all()
    return [
        {
            "id":               r.id,
            "strategy":         r.strategy_name,
            "underlying":       r.underlying,
            "direction":        r.direction,
            "status":           r.status,
            "spot":             r.spot_price,
            "vwap":             r.vwap,
            "strike":           r.suggested_strike,
            "symbol":           r.suggested_symbol,
            "suggested_token":  r.suggested_token,
            "entry_ref":        r.entry_price_ref,
            "sl_ref":           r.stop_loss_ref,
            "target_ref":       r.target_ref,
            "rr":               r.risk_reward,
            "mode":             r.mode,
            "time":             r.signal_time.isoformat() if r.signal_time else None,
            "indicators":       r.indicators,
            "skip_reason":      r.skip_reason,
            "run_id":           r.run_id,
            "no_trade_decision_id": r.no_trade_decision_id,
            "expiry_band":      r.expiry_band,
            "selection_rejections": r.selection_rejections,
        }
        for r in rows
    ]


# ── Signal pipeline diagnostics ───────────────────────────────────────────────

class _SkipProbe(Exception):
    """Internal: a probe skipped because its dependency already failed the preflight."""


@router.get("/signals/diagnostics", response_model=SignalDiagnosticsResponse)
async def get_signal_diagnostics():
    """
    Read-only explanation of why signals are (not) firing / converting.
    Total by construction: every probe is individually wrapped; a failing probe
    becomes a warning blocker and the endpoint still returns 200.
    """
    # Function-local imports: gates/constants are read from the runner itself so
    # the reported thresholds can never drift from the ones the loop applies.
    from app.services.trading import strategy_runner as _sr
    from app.services.trading.signal_diagnostics import (
        SESSION_WINDOW_NOTE, derive_blockers, format_session_window, session_window_open, vix_gate,
    )
    from app.services.trading.regime_rules import get_regime_rule
    from app.services.trading.strategy_orchestrator import strategy_orchestrator
    from app.services.cache.redis_client import redis_cache
    from app.services.market_data.levels import IST, parse_candle_ts
    from app.db.base import AsyncSessionLocal
    from app.db.models.instrument import Instrument, InstrumentType, OptionType
    from app.services.broker.instrument_sync import get_active_expiries
    from app.strategies.indicators import compute_all
    from sqlalchemy import func as _func

    now_ist = datetime.now(IST)
    start, end = settings.SESSION_START_IST, settings.SESSION_END_IST
    probe_errors: list[dict] = []

    def _probe_failed(code: str, exc: Exception) -> None:
        logger.warning("diagnostics probe %s failed: %s", code, exc)
        probe_errors.append({"code": code, "detail": str(exc)})

    # Redis preflight: with Redis down every call costs seconds (connect retries), so
    # one bounded ping decides whether the Redis-backed probes run at all.
    redis_ok = True
    try:
        await asyncio.wait_for(redis_cache._r().ping(), timeout=1.5)
    except Exception as exc:
        redis_ok = False
        _probe_failed("REDIS_UNAVAILABLE", exc if str(exc) else TimeoutError("redis ping timed out after 1.5 s"))

    # Runner
    runner_running = None
    try:
        runner_running = _sr.strategy_runner.is_running
    except Exception as exc:
        _probe_failed("RUNNER_PROBE", exc)

    # Feed + broker session
    feed = {"connected": None, "subscriptions": None, "session_valid": None}
    try:
        from app.services.market_data.feed import market_feed
        feed["connected"] = bool(market_feed.is_connected)
    except Exception as exc:
        _probe_failed("FEED_PROBE", exc)
    try:
        from app.services.market_data.feed import market_feed
        feed["subscriptions"] = int(market_feed.subscription_count)
    except Exception as exc:
        _probe_failed("FEED_SUBSCRIPTIONS_PROBE", exc)
    try:
        from app.services.broker.angel_client import _session
        feed["session_valid"] = bool(_session.is_valid())
    except Exception as exc:
        _probe_failed("SESSION_PROBE", exc)

    # Config (mode, study phase, strategy flags)
    mode = study_phase = None
    strategy_flags: dict = {}
    try:
        cfg = await mode_manager.get_config()
        if cfg:
            mode = cfg.mode.value if hasattr(cfg.mode, "value") else str(cfg.mode)
            study_phase = bool(cfg.study_phase)
            strategy_flags = cfg.strategy_flags or {}
    except Exception as exc:
        _probe_failed("CONFIG_PROBE", exc)

    # DB: signals + positions (DB_UNAVAILABLE if unreachable — counts stay null)
    db_available = True
    signals = {"today_count": None, "by_status": {}, "last_signal_at": None, "last_signal_age_secs": None}
    positions = {"open": None, "today_total": None}
    day_start_utc = datetime.combine(now_ist.date(), datetime.min.time()).replace(tzinfo=IST).astimezone(timezone.utc)
    try:
        async with AsyncSessionLocal() as db:
            # live paper diagnostics: scoped to the live run (design §17.2)
            rows = (await db.execute(
                select(StrategySignal.status, _func.count())
                .where(StrategySignal.run_id == LIVE_RUN_ID,
                       StrategySignal.signal_time >= day_start_utc)
                .group_by(StrategySignal.status)
            )).all()
            by_status = {(s.value if hasattr(s, "value") else str(s)): int(n) for s, n in rows}
            signals["by_status"] = by_status
            signals["today_count"] = sum(by_status.values())
            last = (await db.execute(select(_func.max(StrategySignal.signal_time))
                                     .where(StrategySignal.run_id == LIVE_RUN_ID))).scalar()
            if last is not None:
                signals["last_signal_at"] = last.astimezone(IST).isoformat()
                signals["last_signal_age_secs"] = max(0, int((now_ist - last).total_seconds()))
            positions["open"] = int((await db.execute(
                select(_func.count()).select_from(PaperPosition).where(
                    PaperPosition.run_id == LIVE_RUN_ID, PaperPosition.is_open == True)  # noqa: E712
            )).scalar() or 0)
            positions["today_total"] = int((await db.execute(
                select(_func.count()).select_from(PaperPosition).where(
                    PaperPosition.run_id == LIVE_RUN_ID, PaperPosition.entry_time >= day_start_utc)
            )).scalar() or 0)
    except Exception as exc:
        logger.error("diagnostics: database unreachable: %s", exc)
        db_available = False

    # Capital
    capital = {k: None for k in ("capital", "available", "trade_amount", "daily_trades", "max_trades_per_day",
                                 "open_positions", "max_open_positions", "circuit_breaker", "circuit_reason")}
    if db_available:
        try:
            cs = await mode_manager.get_capital_summary()
            if cs:
                capital.update(
                    capital=cs.get("capital"), available=cs.get("available_capital"),
                    trade_amount=cs.get("trade_amount"), daily_trades=cs.get("daily_trades"),
                    max_trades_per_day=cs.get("max_trades_per_day"), open_positions=cs.get("open_positions"),
                    max_open_positions=cs.get("max_open_positions"),
                    circuit_breaker=bool(cs.get("circuit_breaker")), circuit_reason=cs.get("circuit_reason") or "",
                )
        except Exception as exc:
            _probe_failed("CAPITAL_PROBE", exc)

    # VIX
    vix_ltp = None
    if redis_ok:
        try:
            vix_ltp = await redis_cache.get_ltp(settings.VIX_TOKEN)
        except Exception as exc:
            _probe_failed("VIX_PROBE", exc)
    vix = {"ltp": vix_ltp, "min": _sr.VIX_MIN, "max": _sr.VIX_MAX,
           "gate": vix_gate(vix_ltp, _sr.VIX_MIN, _sr.VIX_MAX)}

    # Per underlying
    underlyings: list[dict] = []
    session_start_mins = start.hour * 60 + start.minute
    for u in ("NIFTY", "BANKNIFTY"):
        token = settings.NIFTY_TOKEN if u == "NIFTY" else settings.BANKNIFTY_TOKEN
        info: dict = {
            "underlying": u, "spot": None, "candles_today": None,
            "candles_required": _sr.MIN_TODAY_CANDLES,
            "atr": None, "atr_min": _sr.ATR_MIN_NIFTY if u == "NIFTY" else _sr.ATR_MIN_BANKNIFTY,
            "last_candle_at": None, "regime": None, "regime_allows_trading": None,
        }
        extra: dict = {}
        try:
            if not redis_ok:
                raise _SkipProbe
            info["spot"] = await redis_cache.get_ltp(token)
            all_c = await redis_cache.get_candles(token, "ONE_MINUTE", 500)
            today_c = []
            for c in all_c:
                try:
                    ts = parse_candle_ts(c.get("timestamp")).astimezone(IST)
                except Exception:
                    continue
                if ts.date() == now_ist.date() and ts.hour * 60 + ts.minute >= session_start_mins:
                    today_c.append(c)
            info["candles_today"] = len(today_c)
            if all_c:
                info["last_candle_at"] = str(all_c[-1].get("timestamp"))
            if len(today_c) >= _sr.MIN_TODAY_CANDLES:
                atr = compute_all(today_c).get("atr")
                info["atr"] = round(float(atr), 4) if atr is not None else None
        except _SkipProbe:
            pass
        except Exception as exc:
            _probe_failed(f"CANDLES_PROBE_{u}", exc)
        try:
            if redis_ok:
                raw = await redis_cache._r().get(f"regime:{u}:current")
                if raw:
                    info["regime"] = json.loads(raw).get("regime")
                    info["regime_allows_trading"] = bool(get_regime_rule(info["regime"]).enabled_strategies)
            extra["in_cooldown"] = bool(strategy_orchestrator.is_in_cooldown(u))
        except Exception as exc:
            _probe_failed(f"REGIME_PROBE_{u}", exc)
        if db_available:
            try:
                async with AsyncSessionLocal() as db:
                    snap_date = (await db.execute(
                        select(MarketDailySnapshot.date)
                        .where(MarketDailySnapshot.underlying == u, MarketDailySnapshot.date < now_ist.date())
                        .order_by(desc(MarketDailySnapshot.date)).limit(1)
                    )).scalar()
                    extra["prev_day_snapshot"] = snap_date.isoformat() if snap_date else None
                    info["prev_day_snapshot"] = extra["prev_day_snapshot"]

                    # CAPITAL_BELOW_LOT_COST: nearest-expiry ATM CE/PE premium x lot size
                    if info["spot"]:
                        step = 50 if u == "NIFTY" else 100
                        atm = round(info["spot"] / step) * step
                        expiries = await get_active_expiries(db, u, as_of=now_ist.date())
                        insts = []
                        if expiries:
                            insts = (await db.execute(
                                select(Instrument).where(
                                    Instrument.name == u,
                                    Instrument.instrument_type == InstrumentType.OPTIDX,
                                    Instrument.expiry == expiries[0],
                                    Instrument.strike == atm,
                                    Instrument.option_type.in_([OptionType.CE, OptionType.PE]),
                                    Instrument.is_active == True,  # noqa: E712
                                )
                            )).scalars().all()
                        chain_ltp: dict = {}
                        if expiries:
                            chain = await redis_cache.get_option_chain(u, str(expiries[0]))
                            for s in (chain or {}).get("strikes", []):
                                if s.get("token") and s.get("ltp"):
                                    chain_ltp[str(s["token"])] = float(s["ltp"])
                        best = None
                        for inst in insts:
                            prem = chain_ltp.get(inst.token) or await redis_cache.get_ltp(inst.token)
                            if prem and prem > 0:
                                cost = prem * inst.lot_size
                                if best is None or cost < best[0]:
                                    best = (cost, prem, inst.lot_size)
                        if best:
                            extra.update(lot_cost=round(best[0], 2), atm_premium=best[1], lot_size=best[2])
                        else:
                            extra["lot_cost_unavailable"] = True
                    else:
                        extra["lot_cost_unavailable"] = True
            except Exception as exc:
                _probe_failed(f"LOT_COST_PROBE_{u}", exc)
        underlyings.append(info | {"_extra": extra})

    # Strategies
    strategies = {"total": None, "enabled": None, "disabled": []}
    try:
        names = list(ALL_STRATEGY_NAMES)    # canonical universe, never strategy_flags.keys()
        if names:
            disabled = [n for n in names if strategy_flags.get(n, True) is False]
            strategies = {"total": len(names), "enabled": len(names) - len(disabled), "disabled": disabled}
    except Exception as exc:
        _probe_failed("STRATEGIES_PROBE", exc)

    blockers = derive_blockers({
        "now_ist": now_ist, "session_start": start, "session_end": end,
        "db_available": db_available, "runner_running": runner_running, "study_phase": study_phase,
        "feed": feed, "vix": vix, "capital": capital, "strategies": strategies,
        "underlyings": [{**{k: v for k, v in i.items() if k != "_extra"}, **i["_extra"]} for i in underlyings],
        "probe_errors": probe_errors,
    })

    return SignalDiagnosticsResponse(
        as_of=now_ist.isoformat(),
        session_window_open=session_window_open(now_ist, start, end),
        session_window=format_session_window(start, end),
        session_window_note=SESSION_WINDOW_NOTE,
        runner_running=runner_running, mode=mode, study_phase=study_phase,
        feed=feed, signals=signals, positions=positions,
        underlyings=[{k: v for k, v in i.items() if k != "_extra"} for i in underlyings],
        vix=vix, capital=capital, strategies=strategies, blockers=blockers,
    )


# ── Positions ─────────────────────────────────────────────────────────────────

@router.get("/positions")
async def get_positions(
    is_open:    Optional[bool] = None,
    strategy:   Optional[str] = None,
    underlying: Optional[str] = None,
    limit:      int = Query(100, ge=1, le=500),
    run_id:     str = Depends(run_id_query),
    db: AsyncSession = Depends(get_db),
):
    q = select(PaperPosition).where(PaperPosition.run_id == run_id) \
        .order_by(desc(PaperPosition.entry_time))
    if is_open is not None:
        q = q.where(PaperPosition.is_open == is_open)
    if strategy:
        q = q.where(PaperPosition.strategy_name == strategy)
    if underlying:
        q = q.where(PaperPosition.underlying == underlying.upper())
    q = q.limit(limit)
    result = await db.execute(q)
    rows = result.scalars().all()
    return [
        {
            "id":               r.id,
            "strategy":         r.strategy_name,
            "token":            r.token,
            "symbol":           r.symbol,
            "underlying":       r.underlying,
            "strike":           r.strike,
            "option_type":      r.option_type,
            "expiry":           r.expiry.isoformat() if r.expiry else None,
            "direction":        r.direction,
            "lots":             r.lots,
            "lot_size":         r.lot_size,
            "qty":              r.quantity,
            "entry":            r.actual_entry,
            "sl":               r.stop_loss,
            "initial_sl":       r.initial_stop_loss,
            "target":           r.target,
            "trailing_sl":      r.trailing_sl,
            "exit":             r.actual_exit,
            "exit_reason":      r.exit_reason,
            "capital_used":     r.trade_amount_used,
            "gross_pnl":    r.gross_pnl,
            "realized_gross_pnl": float(r.gross_pnl or 0.0),
            "original_qty": r.original_quantity or (r.quantity + (r.partial_exit_qty or 0)),
            "brokerage": r.brokerage,
            "planned_risk": (r.sizing_detail or {}).get("planned_risk"),
            "realized_loss": max(0.0, -float(r.net_pnl or 0.0)) if not r.is_open else None,
            "quote_max_age_seconds": runtime_params().stale_tick_seconds,
            "charges":      r.statutory_charges,
            "net_pnl":      r.net_pnl,
            "is_open":      r.is_open,
            "mode":         r.mode,
            "partial_closed":     r.partial_closed,
            "partial_exit_price": r.partial_exit_price,
            "partial_exit_qty":   r.partial_exit_qty,
            "partial_pnl":        r.partial_pnl,
            "remaining_qty":      r.quantity if r.is_open else 0,
            "entry_time":   r.entry_time.isoformat() if r.entry_time else None,
            "exit_time":    r.exit_time.isoformat() if r.exit_time else None,
        }
        for r in rows
    ]

@router.post("/positions/{pos_id}/close")
async def close_position(pos_id: int):
    ok = await paper_engine.manual_close(pos_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Position not found or already closed")
    return {"status": "closed", "id": pos_id}


# ── Kill switch (paper only; unauthenticated — see module docstring) ──────────

class KillSwitchRequest(BaseModel):
    reason: Optional[str] = None


@router.post("/kill-switch")
async def kill_switch(body: Optional[KillSwitchRequest] = None):
    """
    Halt all new entries (persisted, survives restart) and square off every
    open PAPER position at the best available price. Fails safe.
    No authentication yet.
    """
    reason = (body.reason if body and body.reason else None) or "manual kill switch"
    persisted = await mode_manager.set_trading_halted(True, reason)
    result = await paper_engine.square_off_all("KILL_SWITCH")
    return {"halted": True, "reason": reason, "persisted": persisted, **result}


@router.get("/kill-switch")
async def kill_switch_status():
    """Current kill switch state. No authentication yet."""
    return await mode_manager.get_halt_status()


@router.post("/kill-switch/release")
async def kill_switch_release(operator: str = Depends(require_admin_key)):
    """Lift the kill switch so paper entries can resume. Resuming trading needs the
    admin key (design §17.4); this is the same release as POST /halt/release MANUAL."""
    res = await _release_halt(HaltRelease(halt_type="MANUAL"), operator)
    return {"halted": res["halted"]}          # the pre-existing response shape


# ── Halt control (design §17.4): engaging only stops trading, releasing resumes it ──

HALT_TYPES = ("MANUAL", "DAILY_LOSS", "LOSING_STREAK", "STRATEGY_STREAK")


class HaltEngage(BaseModel):
    reason: Optional[str] = None


class HaltRelease(BaseModel):
    halt_type: str = "MANUAL"
    scope: Optional[str] = None          # strategy name for STRATEGY_STREAK
    note: Optional[str] = None


async def _halt_state() -> dict:
    from app.runtime.context import ctx
    from app.db.models.risk import RiskHalt
    from app.runtime.context import sessions
    rc = ctx()
    async with sessions()() as db:
        rows = (await db.execute(select(RiskHalt).where(
            RiskHalt.run_id == rc.run_id, RiskHalt.trading_day == rc.clock.trading_day())
            .order_by(RiskHalt.id))).scalars().all()
    halted, why = await mode_manager.kill_switch_state()
    return {
        "kill_switch": {"halted": halted, "reason": why or None},
        "halts": [{"halt_type": h.halt_type, "scope": h.scope, "reason": h.reason,
                   "engaged_at": h.engaged_at.isoformat() if h.engaged_at else None,
                   "released_at": h.released_at.isoformat() if h.released_at else None,
                   "released_by": h.released_by, "release_note": h.release_note}
                  for h in rows],
    }


@router.post("/halt")
async def engage_halt(body: Optional[HaltEngage] = None,
                      operator: str = Depends(optional_admin_key)):
    """Engage a MANUAL halt: blocks new entries for the trading day (open positions
    keep their own exits). Allowed even when no admin key is configured."""
    from app.db.models.risk import RiskHalt
    from app.runtime.context import ctx, sessions
    reason = (body.reason if body and body.reason else None) or "manual halt"
    persisted = await mode_manager.set_trading_halted(True, reason)
    rc = ctx()
    day = rc.clock.trading_day()
    async with sessions()() as db:
        row = (await db.execute(select(RiskHalt).where(
            RiskHalt.run_id == rc.run_id, RiskHalt.trading_day == day,
            RiskHalt.halt_type == "MANUAL", RiskHalt.scope.is_(None)))).scalar_one_or_none()
        if row is None:
            db.add(RiskHalt(run_id=rc.run_id, trading_day=day, halt_type="MANUAL", scope=None,
                            reason=reason, context={"engaged_by": operator},
                            engaged_at=rc.clock.now_utc()))
        else:                                  # re-engage after a release, same day
            row.reason, row.engaged_at = reason, rc.clock.now_utc()
            row.released_at = row.released_by = row.release_note = None
        await db.commit()
    logger.warning("HALT ENGAGED by %s: %s", operator, reason)
    return {"halted": True, "reason": reason, "persisted": persisted, **(await _halt_state())}


async def _release_halt(body: HaltRelease, operator: str) -> dict:
    from app.db.models.risk import RiskHalt
    from app.runtime.context import ctx, sessions
    halt_type = (body.halt_type or "MANUAL").upper()
    if halt_type not in HALT_TYPES:
        raise HTTPException(status_code=400, detail=f"halt_type must be one of {HALT_TYPES}")
    if halt_type == "STRATEGY_STREAK" and body.scope not in ALL_STRATEGY_NAMES:
        raise HTTPException(status_code=400, detail="scope must name a strategy for STRATEGY_STREAK")
    scope = body.scope if halt_type == "STRATEGY_STREAK" else None
    rc = ctx()
    day = rc.clock.trading_day()
    if halt_type == "MANUAL":
        if not await mode_manager.set_trading_halted(False, None):
            raise HTTPException(status_code=500, detail="Failed to persist kill switch release")
    async with sessions()() as db:
        q = select(RiskHalt).where(RiskHalt.run_id == rc.run_id, RiskHalt.trading_day == day,
                                   RiskHalt.halt_type == halt_type)
        q = q.where(RiskHalt.scope.is_(None) if scope is None else RiskHalt.scope == scope)
        row = (await db.execute(q)).scalar_one_or_none()
        if row is None and halt_type != "MANUAL":
            raise HTTPException(status_code=404, detail=f"no {halt_type} halt engaged today")
        if row is not None:
            row.released_at = rc.clock.now_utc()
            row.released_by = operator
            row.release_note = body.note
            await db.commit()
    state = await _halt_state()
    logger.warning("HALT RELEASED by %s: %s%s — state now %s", operator, halt_type,
                   f"/{scope}" if scope else "", state["kill_switch"])
    return {"released": halt_type, "scope": scope, "halted": state["kill_switch"]["halted"],
            **state}


@router.post("/halt/release")
async def release_halt(body: Optional[HaltRelease] = None,
                       operator: str = Depends(require_admin_key)):
    """Release a halt for today. Requires the admin key; 503 when none is configured."""
    return await _release_halt(body or HaltRelease(), operator)


class ClearTestTrades(BaseModel):
    run_id: Optional[str] = None


@router.post("/clear-test-trades")
async def clear_test_trades(body: Optional[ClearTestTrades] = None,
                            operator: str = Depends(require_admin_key),
                            db: AsyncSession = Depends(get_db)):
    """Delete ONE run's positions, signals and daily P&L summaries (design §17.2).
    ``run_id`` is required, with no default: wiping every run must not be the
    accidental case. Admin key required."""
    from sqlalchemy import delete, update
    from app.db.models.order import Order
    from app.db.models.trade import Trade
    from app.services.cache.redis_client import redis_cache

    if body is None or not body.run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    run_id = validate_run_id(body.run_id)
    r_pos = await db.execute(delete(PaperPosition).where(PaperPosition.run_id == run_id))
    r_sig = await db.execute(delete(StrategySignal).where(StrategySignal.run_id == run_id))
    r_pnl = await db.execute(delete(DailyPnLSummary).where(DailyPnLSummary.run_id == run_id))
    # Order / Trade are legacy paper-order bookkeeping, not run-shared: no replay
    # path writes either, so these two deletes are unscoped by design (§17.2).
    r_orders = await db.execute(delete(Order).where(Order.is_paper == True))
    r_trades = await db.execute(delete(Trade).where(Trade.is_paper == True))
    logger.warning("clear-test-trades run_id=%s by %s", run_id, operator)

    await db.commit()
    # Recompute the display caches from what is left (design §9.2); the old
    # unconditional used_capital = 0 write was only right when every run was deleted.
    from app.runtime.context import ctx
    await mode_manager.reconcile_used_capital(session_factory=ctx().sessions)

    # Clear redis keys
    try:
        r = redis_cache._r()   # uses the guarded accessor — raises if not connected
        pos_keys = await r.keys("pos:*")
        if pos_keys:
            await r.delete(*pos_keys)
        await redis_cache.delete("positions")
        await redis_cache.delete("order_book")
    except Exception as e:
        logger.warning(f"Error clearing redis keys: {e}")

    return {
        "status": "cleared",
        "run_id": run_id,
        "positions_deleted": r_pos.rowcount,
        "signals_deleted": r_sig.rowcount,
        "pnl_summaries_deleted": r_pnl.rowcount,
        "orders_deleted": r_orders.rowcount,
        "trades_deleted": r_trades.rowcount,
    }



# ── P&L ───────────────────────────────────────────────────────────────────────

@router.get("/pnl")
async def get_pnl_today(run_id: str = Depends(run_id_query),
                        db: AsyncSession = Depends(get_db)):
    from app.runtime.context import trading_day
    today = trading_day()
    result = await db.execute(
        select(DailyPnLSummary).where(DailyPnLSummary.run_id == run_id,
                                      DailyPnLSummary.date == today)
        .order_by(desc(DailyPnLSummary.net_pnl))
    )
    rows = result.scalars().all()
    total_pnl = sum(r.net_pnl for r in rows)
    return {
        "date":       str(today),
        "total_pnl":  round(total_pnl, 2),
        "strategies": [
            {
                "strategy":      r.strategy_name,
                "mode":          r.mode,
                "trades":        r.total_trades,
                "wins":          r.winning_trades,
                "losses":        r.losing_trades,
                "win_rate":      r.win_rate,
                "gross_pnl":     r.gross_pnl,
                "net_pnl":       r.net_pnl,
                "brokerage":     r.brokerage,
                # renamed from max_drawdown (R15.4): largest single-trade loss
                "worst_trade_loss":    r.worst_trade_loss,
                "max_equity_drawdown": r.max_equity_drawdown,
                "avg_rr":        r.avg_rr,
            }
            for r in rows
        ],
    }

@router.get("/pnl/history")
async def get_pnl_history(
    days: int = Query(22, ge=1, le=90),
    run_id: str = Depends(run_id_query),
    db: AsyncSession = Depends(get_db),
):
    from app.runtime.context import trading_day
    since = trading_day() - timedelta(days=days)
    result = await db.execute(
        select(DailyPnLSummary)
        .where(DailyPnLSummary.run_id == run_id, DailyPnLSummary.date >= since)
        .order_by(DailyPnLSummary.date)
    )
    rows = result.scalars().all()

    # Aggregate by date
    by_date: dict[str, dict] = {}
    for r in rows:
        d = str(r.date)
        if d not in by_date:
            by_date[d] = {"date": d, "net_pnl": 0, "trades": 0, "wins": 0,
                          "statutory_charges": 0.0,
                          "worst_trade_loss": None, "max_equity_drawdown": None}
        by_date[d]["net_pnl"]  = round(by_date[d]["net_pnl"] + r.net_pnl, 2)
        by_date[d]["statutory_charges"] = round(
            by_date[d]["statutory_charges"] + (r.statutory_charges or 0.0), 2)
        by_date[d]["trades"]  += r.total_trades
        by_date[d]["wins"]    += r.winning_trades
        # per-strategy maxima for the day (R15.4: worst_trade_loss is the
        # largest single-trade loss; max_equity_drawdown the deepest intraday fall)
        for key in ("worst_trade_loss", "max_equity_drawdown"):
            v = getattr(r, key)
            if v is not None:
                prev = by_date[d][key]
                by_date[d][key] = round(max(prev, v) if prev is not None else v, 2)

    # Cumulative P&L
    cum = 0.0
    result_list = []
    for d in sorted(by_date.keys()):
        cum += by_date[d]["net_pnl"]
        by_date[d]["cumulative_pnl"] = round(cum, 2)
        result_list.append(by_date[d])
    return result_list


# ── Option Selector ───────────────────────────────────────────────────────────

@router.post("/option-selector")
async def select_option(body: OptionSelectorRequest, db: AsyncSession = Depends(get_db)):
    try:
        direction = Direction(body.direction.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="direction must be BUY or SELL")

    expiry = date.fromisoformat(body.expiry) if body.expiry else None

    from app.runtime.context import ctx, params as _params
    from app.services.trading.risk_gate import round_to_500

    p = _params()
    result = await option_selector.select_top_n(
        db=db, direction=direction,
        underlying=body.underlying.upper(),
        spot_price=body.spot_price, n=3,
        sl_pct=body.sl_pct, target_pct=body.target_pct, params=p,
        # the diagnostic previews the 1.0x budget: fixed fraction of capital
        base_amount=round_to_500(p.fixed_fraction_pct / 100.0 * ctx().cfg.capital),
        regime_multiplier=1.0,
    )
    top3 = result.ranked
    if not top3:
        raise HTTPException(
            status_code=404,
            detail=f"No suitable options found: rejections={result.rejections} "
                   f"expiry={result.expiry.reason}")

    return [
        {
            "rank":        i + 1,
            "token":       o.token,
            "symbol":      o.symbol,
            "strike":      o.strike,
            "option_type": o.option_type,
            "expiry":      str(o.expiry),
            "ltp":         o.ltp,
            "bid":         o.bid,
            "ask":         o.ask,
            "spread_pct":  o.spread_pct,
            "oi":          o.oi,
            "volume":      o.volume,
            "delta":       o.delta_est,
            "iv":          o.iv,
            "score":       o.score,
            "lot_size":    o.lot_size,
        }
        for i, o in enumerate(top3)
    ]


# ── Market Study ──────────────────────────────────────────────────────────────

@router.get("/market-study/intraday")
async def get_intraday_features(
    underlying: str = "NIFTY",
    date_str:   Optional[str] = Query(None, alias="date"),
    limit:      int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    target_date = date.fromisoformat(date_str) if date_str else date.today()
    # Use IST-aware bounds so the query matches snapshot_time (timezone-aware UTC column)
    IST = timezone(timedelta(hours=5, minutes=30))
    day_start = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=IST)
    day_end   = datetime.combine(target_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=IST)
    result = await db.execute(
        select(IntradayMarketFeature)
        .where(
            IntradayMarketFeature.underlying == underlying.upper(),
            IntradayMarketFeature.snapshot_time >= day_start,
            IntradayMarketFeature.snapshot_time < day_end,
        )
        .order_by(IntradayMarketFeature.snapshot_time)
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        {
            "time":            r.snapshot_time.isoformat(),
            "spot":            r.spot_ltp,
            "vwap":            r.vwap,
            "spot_vs_vwap":    r.spot_vs_vwap_pct,
            "atm":             r.atm_strike,
            "ce_ltp":          r.atm_ce_ltp,
            "pe_ltp":          r.atm_pe_ltp,
            "ce_iv":           r.atm_ce_iv,
            "pe_iv":           r.atm_pe_iv,
            "straddle":        r.atm_straddle_val,
            "pcr":             r.pcr_oi,
            "ce_oi_chg":       r.ce_oi_change,
            "pe_oi_chg":       r.pe_oi_change,
            "ema9":            r.ema9,
            "ema21":           r.ema21,
            "rsi":             r.rsi14,
            "supertrend":      r.supertrend,
            "st_dir":          r.supertrend_dir,
            "bb_upper":        r.bb_upper,
            "bb_lower":        r.bb_lower,
            "orb_high":        r.orb_high,
            "orb_low":         r.orb_low,
            "orb_broken":      r.orb_broken,
            "vix":             r.vix_snapshot,
            "5m_ret":          r.spot_5m_return,
            "15m_ret":         r.spot_15m_return,
        }
        for r in rows
    ]

@router.get("/market-study/daily")
async def get_daily_snapshots(
    underlying: str = "NIFTY",
    days:       int = Query(30, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    from app.runtime.context import trading_day
    since = trading_day() - timedelta(days=days)
    result = await db.execute(
        select(MarketDailySnapshot)
        .where(
            MarketDailySnapshot.underlying == underlying.upper(),
            MarketDailySnapshot.date >= since,
        )
        .order_by(MarketDailySnapshot.date)
    )
    rows = result.scalars().all()
    return [
        {
            "date":        str(r.date),
            "underlying":  r.underlying,
            "open":        r.open,
            "high":        r.high,
            "low":         r.low,
            "close":       r.close,
            "change_pct":  r.change_pct,
            "vix":         r.india_vix,
            "atm_iv":      r.atm_iv,
            "iv_pct":      r.iv_percentile,
            "pcr_oi":      r.pcr_oi,
            "max_pain":    r.max_pain,
            "max_ce_oi":   r.max_oi_ce_strike,
            "max_pe_oi":   r.max_oi_pe_strike,
            "trend":       r.trend,
            "orb_high":    r.orb_high,
            "orb_low":     r.orb_low,
            "vwap_close":  r.vwap_close,
            "pdh":         r.pdh,
            "pdl":         r.pdl,
            "dte":         r.days_to_expiry,
        }
        for r in rows
    ]

@router.get("/market-study/vix")
async def get_vix_history(days: int = Query(30), db: AsyncSession = Depends(get_db)):
    from app.runtime.context import trading_day
    since = trading_day() - timedelta(days=days)
    result = await db.execute(
        select(VIXData).where(VIXData.date >= since).order_by(VIXData.date)
    )
    rows = result.scalars().all()
    return [{"date": str(r.date), "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "change_pct": r.change_pct}
            for r in rows]


# ── WebSocket — live signal stream ────────────────────────────────────────────

@router.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    """Streams new signals and position events to the frontend in real-time."""
    await websocket.accept()
    import redis.asyncio as aioredis

    pubsub_client = aioredis.from_url(
        settings.REDIS_URL, encoding="utf-8", decode_responses=True
    )
    pubsub = pubsub_client.pubsub()
    await pubsub.subscribe("signals:paper_trade", "signals:live_trade")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    await websocket.send_text(message["data"])
                except WebSocketDisconnect:
                    break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        await pubsub.unsubscribe("signals:paper_trade", "signals:live_trade")
        await pubsub_client.aclose()
