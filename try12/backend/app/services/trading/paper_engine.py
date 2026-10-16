"""
Paper Trading Engine (PAPER ONLY — nothing here talks to the broker).
Simulates order fills, tracks open positions, monitors SL/target every second,
handles EOD square-off at the configured square_off_time (default 15:15 IST).

Lot sizing logic (risk_gate.sizing_decision, design §9.5):
  budget       = round_to_500(fraction × capital) × regime multiplier (₹)
  cost_per_lot = fill_price × lot_size     (e.g. 40 × 65 = ₹2,600)
  lots = floor(budget / cost_per_lot)  — may be 0 → signal SKIPPED
         ("Lot unaffordable"); a lot is never forced above the budget.
  actual_capital_used = lots × cost_per_lot

Fills (fills.py, design §15):
  entry at ask × (1 + quote_slippage_pct) when the quote is valid (bid > 0,
  ask ≥ bid, spread ≤ 5% of mid, tick fresh), else LTP × (1 + slippage_pct);
  exits per the fills.exit_fill decision table (BID / TRIGGER_SLIPPED /
  LAST_KNOWN / LTP_GAP) — an open position can always be closed.

Exit ladder (trailing.py, design §8.1-8.3): SL / target first, then the
partial (once, ≥ 2 lots), then trailing.next_stop on the reduced position. A
stale price only ever runs SL / EOD exits.

R accounting (design §8.4): initial_stop_loss, initial_risk and r_source are
written once at insert; exit_value_weighted, r_multiple (gross) and
r_multiple_net are written once at the final close.

Capital tracking: nothing is reserved or released. Capital in use is derived per
run_id from open positions (risk_state.open_capital_used) and checked, together with
the daily trade cap, under the (run_id, trading_day) lock at entry (design §9.2).
"""
import asyncio
import logging
import math
import time as _time
from datetime import datetime, date, time as dtime, timezone, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func

from app.db.models.paper_trade import (
    PaperPosition, StrategySignal, DailyPnLSummary,
    TradingMode, SignalDirection, SignalStatus,
)
from app.db.models.risk import PositionEvent
from app.runtime.context import ctx, market, now_ist, now_utc, params, sessions, trading_day
from app.services.trading import fills, no_trade, risk_gate, risk_state, trailing
from app.services.trading.no_trade import Stage
from app.services.trading.option_selector import SelectedOption, option_selector
from app.services.trading.strategy_orchestrator import strategy_orchestrator
from app.services.trading.trade_outcome import TradeOutcome, build_outcome
from app.services.trading.session_times import (
    DEFAULT_SQUARE_OFF, entry_cutoff, is_after_entry_cutoff,
    is_market_hours, parse_square_off,
)
from app.strategies.base import SignalEvent, Direction

logger = logging.getLogger(__name__)

# Fills, charges and the trailing ladder live in fills.py and trailing.py
# (design §8.1, §15): one implementation shared with replay, every rate a
# TradingParams field. There is no default-target fallback ladder.

# ── Price staleness ───────────────────────────────────────────────────────────
# The staleness threshold is params().stale_tick_seconds (default 60 s).
LAST_PRICE_PERSIST_SECS  = 15    # throttle for writing last_ltp into pos:{id}
STALE_WARN_INTERVAL_SECS = 60    # rate-limit stale/missing-price warnings per position
STALE_DEFER_INTERVAL_SECS = 60   # at most one STALE_DEFER position event per minute

# Strong references to detached observability tasks (SIGNAL_RECORDED publishes) so
# a pending task cannot be garbage-collected mid-await.
_bg_tasks: set[asyncio.Task] = set()


_calc_lots = risk_gate.calc_lots     # one implementation (risk_gate), kept for importers


def _tick_age_seconds(tick: Optional[dict]) -> Optional[float]:
    """Age of a Redis tick from its ``ts`` field. Aware UTC ISO is the live
    format; naive ISO (legacy Redis values, replay) is read as UTC.
    None when unknown. Reads the ctx clock; there is no `now` parameter, because
    two sources of 'now' are how the replay clock gets bypassed (design §14.2)."""
    return fills.tick_age_seconds(tick, now_utc())


def _plain_indicators(indicators: Optional[dict]) -> dict:
    """Primitive-valued copy of the signal's indicator snapshot (entry_indicators)."""
    out = {}
    for k, v in (indicators or {}).items():
        try:
            if isinstance(v, (bool, int, float, str)) or v is None:
                out[str(k)] = v
            else:
                out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _quote(tick: Optional[dict]) -> tuple:
    """(best_bid, best_ask, age_s) of a tick, for fills.*"""
    tick = tick or {}
    return tick.get("best_bid"), tick.get("best_ask"), _tick_age_seconds(tick or None)


class PaperEngine:
    """Full lifecycle: signal → lot sizing → fill → SL/target monitor → exit."""

    def __init__(self):
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        # pos_id → (last fresh LTP, UTC time seen) — EOD/kill-switch fallback price
        self._last_known: dict[int, tuple[float, datetime]] = {}
        self._last_persist: dict[int, float] = {}    # pos_id → monotonic ts of last Redis write
        self._stale_warned: dict[int, float] = {}    # pos_id → monotonic ts of last warning
        self._stale_deferred: dict[int, datetime] = {}   # pos_id → ctx-clock ts of last STALE_DEFER
        self._square_off_cache: Optional[tuple[dtime, float]] = None

    async def start(self):
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("PaperEngine started.")

    async def stop(self):
        self._running = False
        task = self._monitor_task
        self._monitor_task = None
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ── Signal handler ────────────────────────────────────────────────────────

    async def handle_signal(self, signal: SignalEvent) -> bool:
        """
        Called by SignalBus for every fired strategy signal.
        Returns True only when a position is actually opened — SignalBus uses
        this to decide whether to consume the aggregator cooldown.

        The order is fixed (design §9.1, §20.3):
          strategy/study/regime gates
          → risk_gate.evaluate_pre_selection   (checks 0-11b: every cap and halt)
          → option_selector.select(base_amount, regime_multiplier)
          → option momentum, risk_gate.evaluate_sizing (check 12), ATR-vs-SL
          → risk_gate.enter  (lock → 13a trade cap → 13b capital → insert → commit)
        Nothing is reserved, so every refusal leaves nothing to release. Every
        refusal records one no_trade_decisions row and a SKIPPED signal.
        """
        from app.services.trading.mode_manager import mode_manager

        # study_phase comes through ctx().control: the live row (an operator
        # toggle applies at once) or, in replay, the frozen RunConfig — never the
        # shared row. Every cap and the capital come from ctx().cfg.
        study_phase = await ctx().control.study_phase()

        # Strategy enabled? Re-check the orchestrator's last merged flags (regime
        # rule AND user flag AND today's streak halt) rather than trusting
        # cfg.strategy_flags (§6.4).
        enabled, why = strategy_orchestrator.is_strategy_enabled(
            signal.underlying, signal.strategy_name)
        if not enabled:
            code = {
                "user_disabled": no_trade.ReasonCode.STRATEGY_USER_DISABLED,
                "regime_unavailable": no_trade.ReasonCode.REGIME_UNAVAILABLE,
                "daily_strategy_halt": no_trade.ReasonCode.STRATEGY_DAILY_HALT,
            }.get(why, no_trade.ReasonCode.STRATEGY_REGIME_DISABLED)
            await self._reject(signal, Stage.STRATEGY_GATE, code,
                               f"{signal.strategy_name} not enabled for {signal.underlying}: {why}")
            return False

        # Study phase: record signal but do not open any position.
        if study_phase:
            await self._save_signal(signal, SignalStatus.SKIPPED, reason="study_phase=True — observing only")
            logger.info(
                "[PaperEngine] STUDY PHASE — signal recorded, no trade: %s %s %s",
                signal.strategy_name, signal.underlying, signal.direction,
            )
            return False

        # ── Regime signal filter ──────────────────────────────────────────────────
        try:
            regime_state = await strategy_orchestrator.get_current_regime(signal.underlying)
            allowed, regime_code, regime_reason = strategy_orchestrator.should_allow_signal(
                signal, regime_state)
            if not allowed:
                await self._reject(
                    signal, Stage.SIGNAL_GATE, regime_code, regime_reason,
                    context=no_trade.as_context(
                        regime=getattr(regime_state, "regime", None),
                        direction=signal.direction.value),
                    status_reason=f"[Regime] {regime_reason}")
                return False
        except Exception as _re:
            # Fail closed (§20.3): a regime filter that cannot run admits nothing.
            logger.error("[PaperEngine] Regime filter error — signal rejected: %s", _re,
                         exc_info=True)
            await self._reject(
                signal, Stage.SIGNAL_GATE, no_trade.ReasonCode.REGIME_STATE_UNAVAILABLE,
                f"[Regime] regime state unreadable: {type(_re).__name__}: {_re}")
            return False

        # ── Phase one: checks 0-11b ───────────────────────────────────────────
        async with sessions()() as db:
            gate = await risk_gate.evaluate_pre_selection(db, signal, regime_state)
        if not gate.allowed:
            await self._reject(signal, gate.stage, gate.code, gate.reason, context=gate.context)
            return False
        fraction, regime_multiplier = gate.fraction, gate.regime_multiplier
        p = params()

        # ── Select the best option (no capital is touched) ────────────────────
        try:
            async with sessions()() as db:
                result = await option_selector.select(
                    db=db,
                    direction=signal.direction,
                    underlying=signal.underlying,
                    spot_price=signal.spot_price,
                    sl_pct=signal.sl_pct,
                    target_pct=signal.target_pct,
                    params=p,
                    base_amount=fraction.base_amount,
                    regime_multiplier=regime_multiplier,
                )
        except Exception as exc:
            logger.error("[PaperEngine] option selection failed: %s", exc, exc_info=True)
            await no_trade.record_degradation(
                ctx().run_id, no_trade.ReasonCode.UNEXPECTED_ENTRY_ERROR,
                underlying=signal.underlying,
                reason=f"option selection raised {type(exc).__name__}: {exc}")
            await self._save_signal(signal, SignalStatus.SKIPPED,
                                    reason=f"Option selection error: {exc}")
            return False

        selected = result.candidate
        if selected is None:
            await self._reject(
                signal, Stage.SELECTION, no_trade.ReasonCode.NO_OPTION_AVAILABLE,
                "No option available", selection=result,
                context=no_trade.as_context(
                    rejections=dict(result.rejections), evaluated=result.evaluated,
                    expiry_reason=result.expiry.reason),
                status_reason="No option available")
            return False

        lot_size = selected.lot_size

        # ── Option momentum confirmation (the candidate's own tick, §10.3) ────
        # The option's LTP must not be falling at entry: buying into weakness is
        # the biggest cause of immediate SL hits. Skipped when the tick carried
        # no open; fails closed if the check itself errors.
        try:
            opt_open = float(getattr(selected, "tick_open", 0) or 0)
            opt_now = float(selected.ltp or 0)
            if opt_open > 0 and opt_now > 0:
                falling_pct = (opt_open - opt_now) / opt_open * 100
                if falling_pct > 2.0:
                    await self._reject(
                        signal, Stage.SELECTION, no_trade.ReasonCode.OPTION_MOMENTUM_FAIL,
                        f"Option momentum check FAILED — LTP falling {falling_pct:.1f}% "
                        f"this candle (open={opt_open:.2f} now={opt_now:.2f}). "
                        f"Skipping to avoid buying into weakness.",
                        selection=result)
                    return False
        except Exception as _e:
            logger.error("[PaperEngine] Option momentum check error — signal rejected: %s", _e,
                         exc_info=True)
            await self._reject(signal, Stage.SELECTION,
                               no_trade.ReasonCode.OPTION_MOMENTUM_FAIL,
                               f"momentum check errored: {_e}", selection=result)
            return False

        # ── Phase two: check 12, lots from the fill and the lot size ──────────
        async with sessions()() as db:
            sized = await risk_gate.evaluate_sizing(
                db, signal, selected, fraction=fraction, regime_multiplier=regime_multiplier)
        if not sized.allowed:
            await self._reject(signal, sized.stage, sized.code, sized.reason,
                               context=sized.context, selection=result)
            return False
        sizing = sized.sizing
        actual_entry = sizing.expected_entry_fill
        entry_fill_source = sized.context.get("fill_source", "LTP_FALLBACK")
        lots, capital_used = sizing.lots, sizing.capital_used
        ltp = selected.ltp
        slippage = round(actual_entry - ltp, 2)

        # ── SL / target on the ACTUAL entry; initial risk is written once ─────
        sl, tgt, rr, target_stretched = fills.entry_levels(
            actual_entry, signal.sl_pct, signal.target_pct, p.min_rr)
        initial_risk = round(actual_entry - sl, 2)
        rr = round(rr, 2)

        qty = lots * lot_size
        brokerage = fills.brokerage(lots, p)

        # ── Spot ATR vs option SL distance check ──────────────────────────────
        try:
            spot_atr = signal.indicators.get("atr") or 0.0
            if spot_atr > 0:
                sl_distance_pts = actual_entry * signal.sl_pct
                approx_delta = getattr(selected, "delta_est", None)
                approx_delta = abs(approx_delta) if approx_delta else 0.0
                if approx_delta < 0.15:
                    approx_delta = 0.35
                spot_move_needed = sl_distance_pts / approx_delta
                min_atr_needed = spot_move_needed / 2.0
                if signal.underlying == "BANKNIFTY":
                    min_atr_needed = min(min_atr_needed, 25.0)
                else:
                    min_atr_needed = min(min_atr_needed, 7.5)
                if spot_atr < min_atr_needed:
                    await self._reject(
                        signal, Stage.SELECTION, no_trade.ReasonCode.ATR_VS_SL_DISTANCE,
                        f"Spot ATR={spot_atr:.1f} < {min_atr_needed:.1f} needed "
                        f"(sl_dist=₹{sl_distance_pts:.1f}, δ≈{approx_delta:.2f}). "
                        f"Market too quiet to reach option target.", selection=result)
                    return False
        except Exception as _e:
            logger.error("[PaperEngine] ATR/SL check error — signal rejected: %s", _e,
                         exc_info=True)
            await self._reject(signal, Stage.SELECTION,
                               no_trade.ReasonCode.ATR_VS_SL_DISTANCE,
                               f"ATR/SL check errored: {_e}", selection=result)
            return False

        # Kill switch may have fired while this signal was being processed
        if mode_manager.is_halted():
            await self._reject(signal, Stage.RISK_GATE, no_trade.ReasonCode.KILL_SWITCH,
                               "Kill switch activated during entry", selection=result)
            return False

        # ── Entry section: lock → 13a → 13b → insert → commit ─────────────────
        pos = PaperPosition(
            strategy_name   = signal.strategy_name,
            entry_regime    = getattr(regime_state, 'regime', None) if regime_state else None,
            token           = selected.token,
            symbol          = selected.symbol,
            underlying      = signal.underlying,
            strike          = selected.strike,
            option_type     = selected.option_type,
            expiry          = selected.expiry,
            direction       = SignalDirection.BUY,
            lots            = lots,
            lot_size        = lot_size,
            quantity        = qty,
            entry_price     = ltp,
            entry_slippage  = slippage,
            actual_entry    = actual_entry,
            stop_loss       = sl,
            target          = tgt,
            max_loss_per_lot= round((actual_entry - sl) * lot_size, 2),
            brokerage       = brokerage,
            statutory_charges = 0.0,
            trade_amount_used = capital_used,
            is_open         = True,
            mode            = TradingMode.PAPER,
            entry_time      = now_utc(),
            run_id          = ctx().run_id,
            entry_fill_source = entry_fill_source,
            target_stretched  = target_stretched,
            dte_at_entry      = result.expiry.dte,
            expiry_band       = result.expiry.band,
            entry_regime_multiplier = regime_multiplier,
            sizing_detail     = risk_gate.jsonable(sizing),
            entry_indicators  = _plain_indicators(signal.indicators),
            # Write-once at insert (§8.4): never in any later update()
            initial_stop_loss = sl,
            initial_risk      = initial_risk,
            r_source          = "ENTRY",
            original_quantity = qty,
            original_lots     = lots,
        )

        async def _insert(db: AsyncSession) -> int:
            sig_id = await self._save_signal(
                signal, SignalStatus.SELECTED, selected=selected, sl=sl, tgt=tgt, rr=rr,
                db=db, commit=False, selection=result)
            pos.signal_id = sig_id
            db.add(pos)
            await db.flush()
            await db.execute(update(StrategySignal).where(StrategySignal.id == sig_id)
                             .values(status=SignalStatus.ENTERED, trade_id=pos.id))
            return pos.id

        try:
            async with sessions()() as db:
                entered = await risk_gate.enter(
                    db, run_id=ctx().run_id, trading_day=trading_day(),
                    capital_used=capital_used,
                    advisory_used=float(sized.context.get("open_capital_used", 0.0)),
                    duplicate_token=selected.token,
                    duplicate_symbol=selected.symbol,
                    underlying=signal.underlying,
                    planned_risk_amount=sizing.planned_risk,
                    insert=_insert)
        except Exception as exc:
            # The transaction rolled back: no row exists and nothing is held.
            logger.error(
                "[PaperEngine] Unexpected error opening %s (₹%.0f not deployed): %s",
                signal.strategy_name, capital_used, exc, exc_info=True)
            await no_trade.record_degradation(
                ctx().run_id, no_trade.ReasonCode.UNEXPECTED_ENTRY_ERROR,
                underlying=signal.underlying,
                reason=f"entry insert raised {type(exc).__name__}: {exc}",
                context=no_trade.as_context(amount=capital_used))
            await self._save_signal(signal, SignalStatus.SKIPPED,
                                    reason=f"Entry error: {exc}", selection=result)
            return False

        if not entered.allowed:
            await self._reject(signal, entered.stage, entered.code, entered.reason,
                               context=entered.context, selection=result)
            return False

        pos_id = entered.result
        logger.info(
            "[PaperEngine] ✅ OPENED %s | %s %s %.0f%s | "
            "entry=%.2f SL=%.2f TGT=%.2f R:R=%.2f | "
            "lots=%d cost=₹%.0f (%s base ₹%.0f × regime %.1f)",
            signal.strategy_name, signal.underlying, selected.symbol,
            selected.strike, selected.option_type,
            actual_entry, sl, tgt, rr,
            lots, capital_used, fraction.method, fraction.base_amount, regime_multiplier
        )

        # An entry filled without a valid quote is a recorded degradation (§12)
        if entry_fill_source != "ASK":
            await no_trade.record_degradation(
                ctx().run_id, no_trade.ReasonCode.FALLBACK_FILL, position_id=pos_id,
                underlying=signal.underlying,
                reason=f"entry filled at {entry_fill_source} (no valid quote)",
                context=no_trade.as_context(leg="ENTRY", source=entry_fill_source))

        # Post-commit bookkeeping must never undo or hide an opened position
        try:
            await market().set(
                f"pos:{pos_id}",
                {
                    "id":       pos_id,
                    "token":    selected.token,
                    "entry":    actual_entry,
                    "sl":       sl,
                    "tgt":      tgt,
                    "qty":      qty,
                    "strategy": signal.strategy_name,
                    "capital":  capital_used,
                },
                ttl=86400,
            )
            async with sessions()() as db:
                used = await risk_state.open_capital_used(db, ctx().run_id)
            await market().publish_signal("paper_trade", {
                "event":         "POSITION_OPENED",
                "pos_id":        pos_id,
                "symbol":        selected.symbol,
                "strategy":      signal.strategy_name,
                "direction":     signal.direction.value,
                "lots":          lots,
                "entry":         actual_entry,
                "sl":            sl,
                "target":        tgt,
                "capital_used":  capital_used,
                "available":     round(ctx().cfg.capital - used, 2),
            })
        except Exception as exc:
            logger.error("[PaperEngine] pos %d opened but cache/notify failed: %s",
                         pos_id, exc, exc_info=True)

        # Race guard: kill switch fired after the pre-insert check but
        # before this position was visible to square_off_all → close now.
        if mode_manager.is_halted():
            try:
                await no_trade.record_isolated(
                    stage=Stage.ENTRY, code=no_trade.ReasonCode.HALT_ENGAGED_MID_ENTRY,
                    reason="kill switch engaged after insert — closing immediately",
                    underlying=signal.underlying, strategy_name=signal.strategy_name,
                    position_id=pos_id)
                price, stale = await self._resolve_exit_price(pos)
                await self._close_position(
                    pos_id, price or actual_entry,
                    "KILL_SWITCH_STALE" if stale else "KILL_SWITCH",
                )
            except Exception as _ke:
                logger.error("[PaperEngine] Kill-switch close of new pos %d failed: %s", pos_id, _ke)
        return True  # position opened successfully

    async def _reject(self, signal: SignalEvent, stage: Stage, code, reason: str, *,
                      context: Optional[dict] = None, selection=None,
                      status_reason: Optional[str] = None) -> None:
        """One no_trade_decisions row plus a SKIPPED signal linked to it. The
        recorders never raise: observability must not break trading."""
        ntd_id = await no_trade.record_isolated(
            stage=stage, code=code, reason=reason, context=context or {},
            underlying=signal.underlying, strategy_name=signal.strategy_name)
        await self._save_signal(signal, SignalStatus.SKIPPED, reason=status_reason or reason,
                                no_trade_decision_id=ntd_id, selection=selection)
        logger.info("[PaperEngine] SKIPPED %s: %s [%s]", signal.strategy_name, reason,
                    getattr(code, "value", code))

    # ── Position monitor loop ─────────────────────────────────────────────────

    async def _monitor_loop(self):
        while self._running:
            try:
                await asyncio.sleep(1)
                await self.monitor_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitor loop error: %s", e, exc_info=True)

    async def monitor_once(self) -> None:
        """One monitor pass over every open position (design §14.2). Shared by
        the live loop and replay. No time parameter: it reads the ctx clock,
        since two sources of "now" invite divergence."""
        now_ist = _ist_time()
        async with sessions()() as db:
            result = await db.execute(
                select(PaperPosition).where(PaperPosition.run_id == ctx().run_id,
                                            PaperPosition.is_open == True)  # noqa: E712
            )
            open_positions = result.scalars().all()
        if not open_positions:
            return
        square_off = await self._get_square_off()
        for pos in open_positions:
            await self._check_position(pos, now_ist, square_off)

    async def _get_square_off(self) -> dtime:
        """Configured square-off time, cached for 30 s (fallback 15:15)."""
        now_m = _time.monotonic()
        if self._square_off_cache and now_m - self._square_off_cache[1] < 30:
            return self._square_off_cache[0]
        try:
            sq = parse_square_off(await ctx().control.square_off_time())
        except Exception as exc:
            logger.debug("square_off_time lookup failed (%s) — using default", exc)
            sq = DEFAULT_SQUARE_OFF
        self._square_off_cache = (sq, now_m)
        return sq

    def _warn_price(self, pos_id: int, msg: str, *args) -> None:
        """Rate-limited (1 / 60 s per position) WARNING for missing/stale prices."""
        now_m = _time.monotonic()
        last = self._stale_warned.get(pos_id)
        if last is None or now_m - last >= STALE_WARN_INTERVAL_SECS:
            self._stale_warned[pos_id] = now_m
            logger.warning(msg, *args)

    def _forget(self, pos_id: int) -> None:
        self._last_known.pop(pos_id, None)
        self._last_persist.pop(pos_id, None)
        self._stale_warned.pop(pos_id, None)
        self._stale_deferred.pop(pos_id, None)

    async def _remember_price(self, pos_id: int, ltp: float) -> None:
        """Track the last fresh LTP in memory and (throttled) in Redis pos:{id}."""
        seen_at = now_utc()
        self._last_known[pos_id] = (float(ltp), seen_at)
        now_m = _time.monotonic()
        last = self._last_persist.get(pos_id)
        if last is not None and now_m - last < LAST_PRICE_PERSIST_SECS:
            return
        self._last_persist[pos_id] = now_m
        try:
            key = f"pos:{pos_id}"
            cached = await market().get(key)
            cached = cached if isinstance(cached, dict) else {"id": pos_id}
            cached["last_ltp"] = float(ltp)
            cached["last_ltp_ts"] = seen_at.isoformat()
            await market().set(key, cached, ttl=86400)
        except Exception as exc:
            logger.debug("last_ltp persist failed for pos %d: %s", pos_id, exc)

    async def _resolve_exit_price(self, pos: PaperPosition) -> tuple[Optional[float], bool]:
        """
        Best available exit price → (price, is_stale).
        Order: fresh Redis LTP → stale Redis LTP → in-memory last known →
        Redis pos:{id}.last_ltp → entry price (logged at ERROR) → (None, True).
        """
        ltp = None
        try:
            ltp = await market().get_ltp(pos.token)
        except Exception as exc:
            logger.debug("get_ltp failed for pos %d: %s", pos.id, exc)
        from app.services.trading.accounting import valid_price
        if valid_price(ltp):
            tick = None
            try:
                tick = await market().get_tick(pos.token)
            except Exception:
                tick = None
            age = _tick_age_seconds(tick)
            # An LTP without a verifiable timestamp is not fresh. Treat it as
            # stale so it cannot trigger target, trailing, partial, or timeout
            # logic as though it were a current market quote.
            return float(ltp), (age is None or not 0 <= age <= params().stale_tick_seconds)

        known = self._last_known.get(pos.id)
        if known:
            return known[0], True

        try:
            cached = await market().get(f"pos:{pos.id}")
            if isinstance(cached, dict) and cached.get("last_ltp"):
                return float(cached["last_ltp"]), True
        except Exception as exc:
            logger.debug("pos:%d cache read failed: %s", pos.id, exc)

        fallback = pos.actual_entry or pos.entry_price
        if fallback:
            logger.error(
                "[PaperEngine] No price at all for pos %d token %s — closing at ENTRY "
                "price %.2f, P&L unknown",
                pos.id, pos.token, fallback,
            )
            return float(fallback), True
        return None, True

    async def square_off_all(self, reason: str) -> dict:
        """
        Close every open position at the best available price (EOD and kill switch).
        Positions closed on a stale/fallback price get reason "<reason>_STALE".
        """
        async with sessions()() as db:
            result = await db.execute(
                select(PaperPosition).where(PaperPosition.run_id == ctx().run_id,
                                            PaperPosition.is_open == True)  # noqa: E712
            )
            open_positions = result.scalars().all()

        closed, stale_ids, failed_ids = 0, [], []
        for pos in open_positions:
            try:
                price, is_stale = await self._resolve_exit_price(pos)
                if price is None:
                    logger.error("[PaperEngine] %s: cannot close pos %d — no price", reason, pos.id)
                    failed_ids.append(pos.id)
                    continue
                exit_reason = f"{reason}_STALE" if is_stale else reason
                await self._close_position(pos.id, price, exit_reason)
                closed += 1
                if is_stale:
                    stale_ids.append(pos.id)
                self._forget(pos.id)
            except Exception as exc:
                logger.error("[PaperEngine] %s close error for pos %d: %s", reason, pos.id, exc)
                failed_ids.append(pos.id)

        logger.info(
            "[PaperEngine] square_off_all(%s): closed=%d stale=%s failed=%s",
            reason, closed, stale_ids, failed_ids,
        )
        return {"closed": closed, "stale": stale_ids, "failed": failed_ids}

    async def _check_position(
        self, pos: PaperPosition, now_ist: dtime, square_off: dtime = DEFAULT_SQUARE_OFF,
    ):
        """One monitor pass for one position, in the §8.2 order:
          1. exits — EOD, SL (both allowed on a stale price, tagged _STALE), TARGET;
          2. the partial exit (once, >= 2 lots; a single lot records the skip once);
          3. trailing.next_stop on the post-partial position;
          4. TIMEOUT from params().max_hold_minutes.
        A stale price defers 2-4 and TARGET (R5.10, R14.3)."""
        p = params()
        ltp = await market().get_ltp(pos.token)
        tick = None
        try:
            tick = await market().get_tick(pos.token)
        except Exception:
            tick = None
        age = _tick_age_seconds(tick)
        # A price without a timestamp cannot be proven fresh. Treat it as stale
        # so targets, trailing, partials, and timeouts never act on an unknown-age quote.
        from app.services.trading.accounting import valid_price
        if not valid_price(ltp):
            ltp = None
        stale = ltp is None or age is None or not 0 <= age <= p.stale_tick_seconds
        in_session = is_market_hours(now_ist)

        if not ltp:
            # Never skip silently: warn, and still square off at EOD using the
            # last known price instead of leaving the position open.
            known = self._last_known.get(pos.id)
            if in_session:
                self._warn_price(
                    pos.id, "[PaperEngine] No LTP for pos %d token %s; last known %s",
                    pos.id, pos.token, f"{known[0]:.2f}" if known else "none",
                )
            if now_ist >= square_off:
                price, _ = await self._resolve_exit_price(pos)
                if price is not None:
                    await self._close_position(pos.id, price, "EOD_STALE")
                    self._forget(pos.id)
            return

        if not stale:
            await self._remember_price(pos.id, ltp)
        elif in_session and now_ist < square_off:
            self._warn_price(
                pos.id,
                "[PaperEngine] Stale price for pos %d token %s: tick %.0fs old "
                "(> %ds) — only SL/EOD exits evaluated",
                pos.id, pos.token, age or 0, p.stale_tick_seconds,
            )

        # Re-read the position: the monitor's snapshot may predate a trailing
        # advance or the partial written on a previous tick.
        async with sessions()() as _db:
            fresh = (await _db.execute(
                select(PaperPosition).where(PaperPosition.id == pos.id)
            )).scalar_one_or_none()
        if fresh is None or not fresh.is_open:
            return
        pos = fresh
        live_sl: Optional[float] = pos.trailing_sl or pos.stop_loss
        entry: float = pos.actual_entry or 0.0
        sfx = "_STALE" if stale else ""

        # ── 1. exits ─────────────────────────────────────────────────────────
        if now_ist >= square_off:
            await self._close_position(pos.id, ltp, "EOD" + sfx)
            self._forget(pos.id)
            return
        if live_sl and ltp <= live_sl:
            # fills.exit_fill applies the gap-through rule (min(base, ltp))
            await self._close_position(pos.id, live_sl, "SL" + sfx)
            self._forget(pos.id)
            return
        if stale:
            # Never book a target, advance the stop, take the partial or time
            # out on an old quote — wait for a fresh tick.
            await self._stale_defer(pos, ltp, age)
            return
        if pos.target and ltp >= pos.target:
            await self._close_position(pos.id, pos.target, "TARGET")
            self._forget(pos.id)
            return

        # ── 2-3. partial, then the ladder on the post-partial position ───────
        try:
            progress = trailing.reward_progress(entry, pos.target, ltp)
        except ValueError as exc:
            logger.error("[PaperEngine] pos %d has no usable target (%s) — stop left at %s",
                         pos.id, exc, live_sl)
            await no_trade.record_degradation_once(
                ctx().run_id, no_trade.ReasonCode.DATA_GAP, position_id=pos.id,
                underlying=pos.underlying, reason=f"pos {pos.id}: {exc}",
                context=no_trade.as_context(what="position_target", entry=entry,
                                            target=pos.target),
            )
            progress = None

        if progress is not None:
            if (p.partial_exit_enabled and progress >= p.partial_exit_reward_frac
                    and not pos.partial_closed and not pos.partial_skipped):
                if (pos.lots or 0) >= 2:
                    if await self._partial_close(pos, ltp, tick):
                        pos.partial_closed = True
                else:
                    await self._skip_partial(pos, ltp)

            res = trailing.next_stop(entry, pos.target, ltp, live_sl, p.trail_tiers)
            if res is None:
                logger.debug("pos %d: no stop advance at progress %.3f", pos.id, progress)
            else:
                new_sl, label = res
                async with sessions()() as db:
                    await db.execute(
                        update(PaperPosition)
                        .where(PaperPosition.id == pos.id, PaperPosition.run_id == ctx().run_id,
                               PaperPosition.is_open.is_(True), PaperPosition.stop_loss < new_sl)
                        .values(stop_loss=new_sl, trailing_sl=new_sl)
                    )
                    db.add(PositionEvent(
                        position_id=pos.id, run_id=ctx().run_id, event_type="TRAIL_ADVANCE",
                        occurred_at=now_utc(), price=float(ltp), from_value=live_sl,
                        to_value=new_sl, tier_label=label,
                        detail={"progress": round(progress, 6)},
                    ))
                    await db.commit()
                logger.debug("Trailing SL %s → %.2f (%s, progress=%.2f) for pos %d",
                             live_sl, new_sl, label, progress, pos.id)

        # ── 4. max-hold-time exit (params().max_hold_minutes, §8.3) ──────────
        hold_limit = p.max_hold_minutes.get(pos.strategy_name, p.max_hold_minutes["DEFAULT"])
        if pos.entry_time:
            entry_utc = pos.entry_time
            if entry_utc.tzinfo is None:
                entry_utc = entry_utc.replace(tzinfo=timezone.utc)
            hold_mins = (now_utc() - entry_utc).total_seconds() / 60
            floor = entry * (1 + p.timeout_exit_min_gain_pct / 100)
            if hold_mins > hold_limit and ltp < floor:
                logger.info(
                    "[PaperEngine] MAX-HOLD-TIME exit pos %d (%s) — held %.0f min "
                    "(limit %d), LTP=%.2f < %.2f",
                    pos.id, pos.strategy_name, hold_mins, hold_limit, ltp, floor,
                )
                await self._close_position(pos.id, ltp, "TIMEOUT")
                self._forget(pos.id)

    async def _stale_defer(self, pos: PaperPosition, ltp: float, age: Optional[float]) -> None:
        """One STALE_DEFER position event per position per minute (ctx clock)."""
        now = now_utc()
        last = self._stale_deferred.get(pos.id)
        if last is not None and (now - last).total_seconds() < STALE_DEFER_INTERVAL_SECS:
            return
        self._stale_deferred[pos.id] = now
        async with sessions()() as db:
            db.add(PositionEvent(
                position_id=pos.id, run_id=ctx().run_id, event_type="STALE_DEFER",
                occurred_at=now, price=float(ltp),
                detail=no_trade.as_context(
                    tick_age_s=age, target=pos.target,
                    target_reached=bool(pos.target and ltp >= pos.target)),
            ))
            await db.commit()

    async def _skip_partial(self, pos: PaperPosition, ltp: float) -> None:
        """Single-lot position at the partial trigger: record it once (R5.6)."""
        async with sessions()() as db:
            upd = await db.execute(
                update(PaperPosition)
                .where(PaperPosition.id == pos.id, PaperPosition.partial_skipped == False)  # noqa: E712
                .values(partial_skipped=True)
            )
            if upd.rowcount == 1:
                db.add(PositionEvent(
                    position_id=pos.id, run_id=ctx().run_id,
                    event_type="PARTIAL_SKIPPED_SINGLE_LOT", occurred_at=now_utc(),
                    price=float(ltp), detail={"lots": pos.lots},
                ))
            await db.commit()
        pos.partial_skipped = True

    # ── Partial close ─────────────────────────────────────────────────────────

    async def _partial_close(self, pos: PaperPosition, ltp: float,
                             tick: Optional[dict] = None) -> bool:
        """
        Close partial_exit_fraction of the lots (at least one, never all) when
        reward progress reaches partial_exit_reward_frac. Fires at most once:
        the UPDATE is conditional on partial_closed = false, and rowcount 0
        means another tick already did it (capital is then not released again).

        The remainder's stop is NOT written here: _check_position runs
        trailing.next_stop on the reduced position right after, so the stop is
        the ladder's tier (LOCK_40 at the default 0.75 trigger). quantity and
        lots shrink; original_quantity / original_lots, initial_stop_loss,
        initial_risk and target are untouched.

        Returns True when this call performed the partial.
        """
        p = params()
        if not pos.actual_entry or not pos.lot_size:
            return False

        partial_lots = max(1, int(pos.lots * p.partial_exit_fraction))
        if partial_lots >= pos.lots:
            return False
        partial_qty  = partial_lots * pos.lot_size
        remain_qty   = pos.quantity - partial_qty

        bid, ask, age = _quote(tick)
        fill, _ = fills.exit_fill(ltp, ltp, bid, ask, age, "PARTIAL", p)

        partial_gross = round((fill - pos.actual_entry) * partial_qty, 2)
        partial_brok  = fills.brokerage(partial_lots, p)
        # Statutory charges on both legs of the closed quantity (GST base =
        # entry-leg + exit-leg brokerage of the closed lots)
        partial_chg   = fills.statutory_charges(
            pos.actual_entry * partial_qty, fill * partial_qty, 2 * partial_brok, p,
        )
        partial_net   = fills.net_pnl(partial_gross, partial_brok, partial_chg)

        # Capital released for the closed portion, against the ORIGINAL size
        original_qty = pos.original_quantity or pos.quantity
        capital_released = round((pos.trade_amount_used or 0) * partial_qty / original_qty, 2)

        async with sessions()() as db:
            upd = await db.execute(
                update(PaperPosition)
                .where(PaperPosition.id == pos.id, PaperPosition.run_id == ctx().run_id,
                       PaperPosition.is_open.is_(True), PaperPosition.partial_closed.is_(False),
                       PaperPosition.quantity == pos.quantity)
                .values(
                    partial_closed      = True,
                    partial_exit_price  = fill,
                    partial_exit_qty    = partial_qty,
                    partial_pnl         = partial_net,
                    remaining_quantity  = remain_qty,
                    quantity            = remain_qty,
                    lots                = pos.lots - partial_lots,
                    gross_pnl           = round((pos.gross_pnl or 0) + partial_gross, 2),
                    net_pnl             = fills.net_pnl(
                        (pos.gross_pnl or 0) + partial_gross,
                        (pos.brokerage or 0) + partial_brok,
                        (pos.statutory_charges or 0) + partial_chg),
                    brokerage           = round((pos.brokerage or 0) + partial_brok, 2),
                    statutory_charges   = round((pos.statutory_charges or 0) + partial_chg, 2),
                    trade_amount_used   = round((pos.trade_amount_used or 0) - capital_released, 2),
                )
            )
            if upd.rowcount != 1:
                await db.rollback()
                logger.debug("[PaperEngine] pos %d partial already taken — skipping", pos.id)
                return False
            db.add(PositionEvent(
                position_id=pos.id, run_id=ctx().run_id, event_type="PARTIAL_EXIT",
                occurred_at=now_utc(), price=fill, from_value=float(pos.quantity),
                to_value=float(remain_qty),
                detail={"lots_closed": partial_lots, "pnl": partial_net,
                        "capital_released": capital_released},
            ))
            await db.commit()

        pos.quantity, pos.lots = remain_qty, pos.lots - partial_lots

        logger.info(
            "[PaperEngine] ⚡ PARTIAL CLOSE pos %d | closed %d qty @ %.2f | "
            "partial_pnl=₹%.2f | remain=%d qty",
            pos.id, partial_qty, fill, partial_net, remain_qty,
        )

        # Notify frontend
        await market().publish_signal("paper_trade", {
            "event":        "PARTIAL_CLOSE",
            "pos_id":       pos.id,
            "partial_qty":  partial_qty,
            "exit_price":   fill,
            "partial_pnl":  partial_net,
            "remain_qty":   remain_qty,
            "new_sl":       pos.stop_loss,
        })
        return True

    # ── Full close ────────────────────────────────────────────────────────────

    async def _close_position(self, pos_id: int, exit_price: float, reason: str, _retry: int = 0):
        """Final close. *exit_price* is the trigger (stop, target, or the price
        the caller observed); the fill comes from fills.exit_fill with the tick
        read here, so the observed LTP drives the gap-through rule and a
        no-tick close is LAST_KNOWN."""
        p = params()
        async with sessions()() as db:
            result = await db.execute(
                select(PaperPosition).where(PaperPosition.id == pos_id,
                                            PaperPosition.run_id == ctx().run_id)
            )
            pos = result.scalar_one_or_none()
            if not pos or not pos.is_open:
                return
            # Detached: the guarded UPDATE below is the only write to this row,
            # and a stats rollback cannot expire the values read here.
            db.expunge(pos)

            tick = None
            observed = None
            try:
                observed = await market().get_ltp(pos.token)
                tick = await market().get_tick(pos.token)
            except Exception:
                tick = None
            bid, ask, age = _quote(tick)
            actual_exit, exit_source = fills.exit_fill(
                exit_price, observed, bid, ask, age, reason, p)
            exit_slip = round(exit_price - actual_exit, 2)

            # Remaining-lot P&L (pos.quantity already reflects the partial)
            close_qty = pos.quantity
            close_gross = round((actual_exit - pos.actual_entry) * close_qty, 2)

            # Brokerage for the lots closed NOW; pos.brokerage already holds the
            # entry leg and any partial's exit leg.
            remain_lots = (close_qty // pos.lot_size) if pos.lot_size else pos.lots
            remain_lots = max(remain_lots or 0, 1)
            close_brok_now = fills.brokerage(remain_lots, p)
            total_brok  = round((pos.brokerage or 0.0) + close_brok_now, 2)

            # Statutory charges for the quantity closed now (both legs; GST base
            # = entry-leg + exit-leg brokerage of these lots)
            charges = fills.statutory_charges(
                pos.actual_entry * close_qty, actual_exit * close_qty,
                2 * close_brok_now, p,
            )
            total_charges = round((pos.statutory_charges or 0.0) + charges, 2)

            # Accumulate on top of any partial-close P&L already stored
            total_gross = round((pos.gross_pnl or 0) + close_gross, 2)
            total_net   = fills.net_pnl(total_gross, total_brok, total_charges)

            # Quantity-weighted exit across the partial and final legs, and R
            # against the initial risk only (§8.4). Gross R is the glossary's
            # R_Multiple; r_multiple_net subtracts costs per original unit.
            original_qty = pos.original_quantity or (close_qty + (pos.partial_exit_qty or 0))
            evw = ((pos.partial_exit_qty or 0) * (pos.partial_exit_price or 0.0)
                   + close_qty * actual_exit) / original_qty
            r_multiple = r_multiple_net = None
            if pos.initial_stop_loss is not None and pos.initial_risk is not None \
                    and pos.initial_risk > 0:
                r_multiple = (evw - pos.actual_entry) / pos.initial_risk
                cost_per_unit = (total_brok + total_charges) / original_qty
                r_multiple_net = (evw - pos.actual_entry - cost_per_unit) / pos.initial_risk
            exit_time = now_utc()
            entry_utc = pos.entry_time
            if entry_utc is not None and entry_utc.tzinfo is None:
                entry_utc = entry_utc.replace(tzinfo=timezone.utc)
            hold_minutes = ((exit_time - entry_utc).total_seconds() / 60
                            if entry_utc is not None else None)

            close_values = dict(
                is_open      = False,
                exit_price   = exit_price,
                exit_slippage= exit_slip,
                actual_exit  = actual_exit,
                exit_reason  = reason,
                exit_time    = exit_time,
                gross_pnl    = total_gross,
                brokerage    = total_brok,
                statutory_charges = total_charges,
                net_pnl      = total_net,
                exit_fill_source    = exit_source,
                exit_value_weighted = evw,
                r_multiple          = r_multiple,
                r_multiple_net      = r_multiple_net,
                hold_minutes        = hold_minutes,
            )
            close_stmt = (update(PaperPosition)
                          .where(PaperPosition.id == pos_id, PaperPosition.run_id == ctx().run_id,
                                 PaperPosition.is_open.is_(True), PaperPosition.quantity == close_qty,
                                 PaperPosition.partial_closed == pos.partial_closed)
                          .values(**close_values))
            # Conditional on is_open so two concurrent closers (monitor loop,
            # EOD, kill switch) can never double-release capital.
            upd = await db.execute(close_stmt)
            if upd.rowcount != 1:
                await db.rollback()
                if _retry < 2:
                    return await self._close_position(pos_id, exit_price, reason, _retry + 1)
                from app.services.trading.mode_manager import mode_manager
                await mode_manager.set_trading_halted(True, f"Exit contention for position {pos_id}")
                raise RuntimeError(f"Exit reconciliation required for position {pos_id}")

            # One transaction: the close and its statistics (design §7). The
            # final values are stamped on *pos* so the outcome reads them.
            for k, v in close_values.items():
                setattr(pos, k, v)
            outcome = build_outcome(pos, run_id=ctx().run_id, trading_day=trading_day(),
                                    params=p)
            try:
                await self._record_outcome(db, pos, outcome)
                await db.commit()
                stats_ok = True
            except Exception as exc:
                await db.rollback()
                stats_ok = False
                stats_error = exc

        if not stats_ok:
            # The position is never left open because of a statistics failure:
            # close-only retry, then a visible degradation row. reconcile_performance
            # rebuilds the aggregates from paper_positions.
            async with sessions()() as db2:
                upd = await db2.execute(close_stmt)
                await db2.commit()
            logger.error(
                "[PaperEngine] stats write failed for pos %d — closed without stats, "
                "run reconcile_performance to backfill: %s", pos_id, stats_error,
                exc_info=stats_error,
            )
            await no_trade.record_degradation(
                ctx().run_id, no_trade.ReasonCode.PERF_WRITE_FAILED, position_id=pos_id,
                underlying=pos.underlying,
                reason=f"stats write failed for pos {pos_id}: "
                       f"{type(stats_error).__name__}: {stats_error}",
                context=no_trade.as_context(strategy_name=pos.strategy_name,
                                            regime=pos.entry_regime),
            )
            if upd.rowcount != 1:
                if _retry < 2:
                    return await self._close_position(pos_id, exit_price, reason, _retry + 1)
                from app.services.trading.mode_manager import mode_manager
                await mode_manager.set_trading_halted(True, f"Exit contention for position {pos_id}")
                raise RuntimeError(f"Exit reconciliation required for position {pos_id}")

        logger.info(
            "[PaperEngine] 🔒 CLOSED pos %d | %s | exit=%.2f (%s) PnL=₹%.2f R=%s",
            pos_id, reason, actual_exit, exit_source, total_net,
            f"{r_multiple:.3f}" if r_multiple is not None else "n/a",
        )

        # An exit not filled off a valid bid (other than a TARGET limit) is a recorded
        # fallback (§12); exits are never refused, so this is the visible cost.
        if exit_source == "LAST_KNOWN" or (exit_source == "TRIGGER_SLIPPED" and reason != "TARGET"):
            await no_trade.record_degradation(
                ctx().run_id, no_trade.ReasonCode.FALLBACK_FILL, position_id=pos_id,
                underlying=pos.underlying,
                reason=f"exit ({reason}) filled at {exit_source} (no valid quote)",
                context=no_trade.as_context(leg="EXIT", source=exit_source, reason=reason))

        # Nothing to release: the row just left open_capital_used's sum (is_open = false).

        # Cleanup Redis
        await market().delete(f"pos:{pos_id}")

        # Notify frontend
        await market().publish_signal("paper_trade", {
            "event":   "POSITION_CLOSED",
            "pos_id":  pos_id,
            "reason":  reason,
            "net_pnl": total_net,
            "exit":    exit_price,
        })

    # ── Performance aggregates (design §7) ────────────────────────────────────

    async def _record_outcome(self, db: AsyncSession, pos: PaperPosition,
                              outcome: TradeOutcome) -> None:
        """regime_performance + strategy_perf + daily_pnl_summary, all on the
        close's session. Any failure propagates to _close_position."""
        from app.services.trading.mode_manager import mode_manager

        if outcome.entry_regime:
            await strategy_orchestrator.record_trade_result(
                db,
                regime=outcome.entry_regime,
                underlying=outcome.underlying,
                strategy_name=outcome.strategy_name,
                outcome=outcome,
            )
        await mode_manager.update_strategy_perf(db, outcome.strategy_name, outcome)
        await self._update_daily_pnl(db, pos, outcome)

    async def _update_daily_pnl(self, db: AsyncSession, pos: PaperPosition,
                                outcome: TradeOutcome) -> None:
        """Per strategy, trading day and mode (R4.4): exact sums of the four
        P&L components (so gross − brokerage − charges == net holds on the
        summary, R14.5), the R buckets, worst_trade_loss, and the day's
        peak-to-trough max_equity_drawdown over this strategy's closes."""
        from app.services.trading.session_times import trading_day_bounds_utc
        from app.services.trading.trade_outcome import accumulate

        today = outcome.trading_day
        run_id = outcome.run_id
        summary = (await db.execute(
            select(DailyPnLSummary).where(
                DailyPnLSummary.date          == today,
                DailyPnLSummary.strategy_name == outcome.strategy_name,
                DailyPnLSummary.mode          == pos.mode,
                DailyPnLSummary.run_id        == run_id,
            )
        )).scalar_one_or_none()
        if summary is None:
            summary = DailyPnLSummary(run_id=run_id, date=today,
                                      strategy_name=outcome.strategy_name, mode=pos.mode)
            db.add(summary)

        accumulate(summary, outcome)
        summary.gross_pnl = float(summary.gross_pnl or 0.0) + outcome.gross_pnl
        summary.brokerage = float(summary.brokerage or 0.0) + outcome.brokerage
        summary.statutory_charges = float(summary.statutory_charges or 0.0) + outcome.statutory_charges
        summary.net_pnl = float(summary.net_pnl or 0.0) + outcome.net_pnl
        summary.total_lots = (summary.total_lots or 0) + outcome.lots
        summary.avg_rr = summary.expectancy_r
        if outcome.net_pnl < 0:
            summary.worst_trade_loss = max(float(summary.worst_trade_loss or 0.0), -outcome.net_pnl)

        # Peak-to-trough of the day's cumulative net for this strategy. The
        # close above is flushed in this transaction, so it is included.
        start, end = trading_day_bounds_utc(today)
        nets = (await db.execute(
            select(PaperPosition.net_pnl)
            .where(PaperPosition.run_id == run_id,
                   PaperPosition.strategy_name == outcome.strategy_name,
                   PaperPosition.mode == pos.mode,
                   PaperPosition.is_open == False,  # noqa: E712
                   PaperPosition.exit_time >= start, PaperPosition.exit_time < end)
            .order_by(PaperPosition.exit_time, PaperPosition.id)
        )).scalars().all()
        cum = peak = dd = 0.0
        for n in nets:
            cum += float(n or 0.0)
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        summary.max_equity_drawdown = dd
        await db.flush()

    # ── Signal persistence ────────────────────────────────────────────────────

    async def _save_signal(
        self,
        signal:   SignalEvent,
        status:   SignalStatus,
        selected: Optional[SelectedOption] = None,
        sl:       Optional[float] = None,
        tgt:      Optional[float] = None,
        rr:       Optional[float] = None,
        reason:   str = "",
        db:       Optional[AsyncSession] = None,
        no_trade_decision_id: Optional[int] = None,
        commit: bool = True,
        selection=None,
    ) -> Optional[int]:
        """Persist one StrategySignal. ``commit=False`` flushes on *db* and leaves
        the commit to the caller, so the entry section can keep the signal row and
        the position row in one transaction under the advisory lock."""
        from app.services.trading.mode_manager import mode_manager
        cfg  = await mode_manager.get_config()
        mode = TradingMode(cfg.mode) if cfg else TradingMode.PAPER

        sig = StrategySignal(
            strategy_name    = signal.strategy_name,
            underlying       = signal.underlying,
            direction        = SignalDirection(signal.direction.value),
            status           = status,
            spot_price       = signal.spot_price,
            vwap             = signal.vwap,
            signal_time      = signal.signal_time,
            suggested_strike = selected.strike if selected else signal.suggested_strike,
            suggested_token  = selected.token  if selected else None,
            suggested_symbol = selected.symbol if selected else None,
            entry_price_ref  = selected.ltp    if selected else None,
            stop_loss_ref    = sl,
            target_ref       = tgt,
            risk_reward      = rr,
            indicators       = signal.indicators,
            mode             = mode,
            skip_reason      = reason[:300] if reason else None,   # untruncated copy: no_trade_decisions
            no_trade_decision_id = no_trade_decision_id,
            run_id           = ctx().run_id,
        )
        if selection is not None:
            sig.expiry_band = selection.expiry.band
            sig.expiry_reason = (selection.expiry.reason or "")[:200]
            sig.selection_rejections = dict(selection.rejections)
            sig.score_breakdown = selection.score_table

        async def _publish_recorded(channel: str, payload: dict) -> None:
            try:
                await market().publish_signal(channel, payload)
            except Exception as exc:        # observability must never break persistence
                logger.debug("Signal publish failed (non-fatal): %s", exc)

        async def _persist(session: AsyncSession) -> int:
            session.add(sig)
            if commit:
                await session.commit()
                await session.refresh(sig)
            else:
                await session.flush()
            payload = {                     # built before scheduling — never touches an expired instance
                "event":       "SIGNAL_RECORDED",
                "signal_id":   sig.id,
                "strategy":    sig.strategy_name,
                "underlying":  sig.underlying,
                "direction":   sig.direction.value,
                "status":      sig.status.value,
                "time":        sig.signal_time.isoformat() if sig.signal_time else None,
                "skip_reason": sig.skip_reason,
            }
            channel = "live_trade" if mode == TradingMode.LIVE else "paper_trade"
            task = asyncio.create_task(_publish_recorded(channel, payload))   # fire-and-forget, never awaited
            _bg_tasks.add(task)
            task.add_done_callback(_bg_tasks.discard)
            return sig.id

        if db:
            return await _persist(db)
        async with sessions()() as session:
            return await _persist(session)

    # ── Manual close ──────────────────────────────────────────────────────────

    async def manual_close(self, pos_id: int) -> bool:
        async with sessions()() as db:
            result = await db.execute(
                select(PaperPosition).where(PaperPosition.id == pos_id,
                                            PaperPosition.run_id == ctx().run_id)
            )
            pos = result.scalar_one_or_none()
            if not pos or not pos.is_open:
                return False
        ltp = await market().get_ltp(pos.token) or pos.entry_price
        await self._close_position(pos_id, ltp, "MANUAL")
        return True


# ── IST time helper ───────────────────────────────────────────────────────────

def _ist_time() -> dtime:
    """Current IST wall time from the ctx clock (naive)."""
    return now_ist().time().replace(tzinfo=None)


# Singleton
paper_engine = PaperEngine()
