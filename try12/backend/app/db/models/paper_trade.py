"""
Paper trading tables — simulated orders, positions, P&L.
Identical schema to live trading so switching modes is seamless.
"""
from sqlalchemy import (
    Column, BigInteger, Integer, String, Float, Boolean,
    DateTime, Date, Index, Enum as SAEnum, JSON, UniqueConstraint,
)
from sqlalchemy.sql import func, false
import enum
from app.db.base import Base


class TradingMode(str, enum.Enum):
    PAPER = "PAPER"
    LIVE  = "LIVE"


class SignalDirection(str, enum.Enum):
    BUY  = "BUY"
    SELL = "SELL"


class SignalStatus(str, enum.Enum):
    GENERATED = "GENERATED"   # Signal fired, not yet acted on
    SELECTED  = "SELECTED"    # Option selected for this signal
    ENTERED   = "ENTERED"     # Position taken
    SL_HIT    = "SL_HIT"
    TARGET_HIT = "TARGET_HIT"
    EXPIRED   = "EXPIRED"     # Signal timed out before entry
    SKIPPED   = "SKIPPED"     # Skipped by risk/mode check


class StrategySignal(Base):
    """
    Every signal fired by any strategy.
    Recorded regardless of whether a trade is taken — crucial for study phase.
    """
    __tablename__ = "strategy_signals"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    strategy_name    = Column(String(50),  nullable=False)
    underlying       = Column(String(20),  nullable=False)
    direction        = Column(SAEnum(SignalDirection), nullable=False)
    status           = Column(SAEnum(SignalStatus), nullable=False, default=SignalStatus.GENERATED)

    # Signal context
    spot_price       = Column(Float,      nullable=False)
    vwap             = Column(Float,      nullable=True)
    signal_time      = Column(DateTime(timezone=True), nullable=False)
    expiry           = Column(Date,       nullable=True)

    # Entry parameters
    suggested_strike = Column(Float,      nullable=True)
    suggested_token  = Column(String(20), nullable=True)
    suggested_symbol = Column(String(60), nullable=True)
    entry_price_ref  = Column(Float,      nullable=True)   # option LTP at signal time
    stop_loss_ref    = Column(Float,      nullable=True)   # option SL
    target_ref       = Column(Float,      nullable=True)   # option target

    # Risk/reward
    risk_reward      = Column(Float,      nullable=True)
    max_loss_pts     = Column(Float,      nullable=True)

    # Human-readable reason when status=SKIPPED (study_phase, risk check, no option, etc.)
    skip_reason      = Column(String(300), nullable=True)

    # Indicator values at signal time (for study)
    indicators       = Column(JSON,       nullable=True)

    mode             = Column(SAEnum(TradingMode), nullable=False, default=TradingMode.PAPER)
    trade_id         = Column(BigInteger, nullable=True)   # FK to paper_positions.id

    # ── migration 0011 ──────────────────────────────────────────────────────
    run_id               = Column(String(32), nullable=False, default="PAPER", server_default="PAPER")
    expiry_band          = Column(String(16),  nullable=True)
    expiry_reason        = Column(String(200), nullable=True)
    selection_rejections = Column(JSON,        nullable=True)   # {ReasonCode value: count}
    score_breakdown      = Column(JSON,        nullable=True)
    no_trade_decision_id = Column(BigInteger,  nullable=True)   # → no_trade_decisions.id

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_sig_strategy_time",   "strategy_name", "signal_time"),
        Index("ix_sig_underlying_time", "underlying",    "signal_time"),
    )


class PaperPosition(Base):
    """
    Open and closed paper trading positions.
    Simulates real order fills with configurable slippage.
    """
    __tablename__ = "paper_positions"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    strategy_name    = Column(String(50),  nullable=True)
    signal_id        = Column(BigInteger,  nullable=True)

    # Instrument
    token            = Column(String(20),  nullable=False)
    symbol           = Column(String(60),  nullable=False)
    underlying       = Column(String(20),  nullable=False)
    strike           = Column(Float,       nullable=True)
    option_type      = Column(String(5),   nullable=True)   # CE / PE
    expiry           = Column(Date,        nullable=True)
    exchange         = Column(String(10),  nullable=False, default="NFO")

    # Position
    direction        = Column(SAEnum(SignalDirection), nullable=False)
    lots             = Column(Integer,  nullable=False, default=1)
    lot_size         = Column(Integer,  nullable=False, default=50)
    quantity         = Column(Integer,  nullable=False)

    # Fill prices (simulated)
    entry_price      = Column(Float,  nullable=False)
    entry_slippage   = Column(Float,  nullable=False, default=0.0)
    actual_entry     = Column(Float,  nullable=False)   # entry + slippage

    # Budget used for this trade (trade_amount from config at entry time)
    trade_amount_used = Column(Float, nullable=True)    # actual ₹ deployed

    # Exit
    exit_price       = Column(Float,  nullable=True)
    exit_slippage    = Column(Float,  nullable=True, default=0.0)
    actual_exit      = Column(Float,  nullable=True)
    exit_reason      = Column(String(20), nullable=True)   # SL/TARGET/MANUAL/EXPIRY/EOD

    # Risk management
    stop_loss        = Column(Float,  nullable=True)   # current SL (trailing updates overwrite it)
    target           = Column(Float,  nullable=True)
    trailing_sl      = Column(Float,  nullable=True)
    max_loss_per_lot = Column(Float,  nullable=True)
    # SL at entry, never modified — achieved R:R / Kelly use this (migration 0010)
    initial_stop_loss = Column(Float, nullable=True)
    # STT + exchange txn + SEBI + stamp + GST booked on closed quantity (migration 0010)
    statutory_charges = Column(Float, nullable=True, default=0.0, server_default="0")

    # Partial exit tracking
    # When progress to target reaches 75% (PARTIAL_CLOSE_PROGRESS), 50% of lots
    # are closed at market.
    # The remaining lots run to the full target with SL moved to breakeven.
    partial_closed        = Column(Boolean, nullable=False, default=False)  # has partial exit fired?
    partial_exit_price    = Column(Float,   nullable=True)   # price at partial close
    partial_exit_qty      = Column(Integer, nullable=True)   # qty closed in partial exit
    partial_pnl           = Column(Float,   nullable=True)   # P&L from partial close
    remaining_quantity    = Column(Integer, nullable=True)   # qty still open after partial exit

    # P&L
    gross_pnl        = Column(Float,  nullable=True)
    brokerage        = Column(Float,  nullable=True, default=0.0)
    net_pnl          = Column(Float,  nullable=True)

    # Regime at entry — set from RegimeState when position is opened
    entry_regime     = Column(String(50), nullable=True)

    # State
    is_open          = Column(Boolean, nullable=False, default=True)
    mode             = Column(SAEnum(TradingMode), nullable=False, default=TradingMode.PAPER)

    entry_time       = Column(DateTime(timezone=True), nullable=False)
    exit_time        = Column(DateTime(timezone=True), nullable=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())

    # ── migration 0011: run scoping and R accounting (design §8.4, §11) ─────
    run_id                  = Column(String(32), nullable=False, default="PAPER", server_default="PAPER")
    original_quantity       = Column(Integer, nullable=True)   # survives partial close
    original_lots           = Column(Integer, nullable=True)
    initial_risk            = Column(Float,   nullable=True)   # actual_entry − initial_stop_loss
    # 'ENTRY' when written at insert; 'BACKFILL_UNVERIFIED' for pre-0011 rows.
    # Anything but 'ENTRY' is excluded from R statistics (R6.7, R15.7).
    r_source                = Column(String(20), nullable=True)
    exit_value_weighted     = Column(Float,   nullable=True)
    r_multiple              = Column(Float,   nullable=True)   # gross of costs
    r_multiple_net          = Column(Float,   nullable=True)   # net of brokerage + charges
    target_stretched        = Column(Boolean, nullable=False, default=False, server_default=false())
    entry_fill_source       = Column(String(16), nullable=True)   # ASK / LTP_FALLBACK
    exit_fill_source        = Column(String(16), nullable=True)   # BID / TRIGGER_SLIPPED / LAST_KNOWN / LTP_GAP
    dte_at_entry            = Column(Integer, nullable=True)
    expiry_band             = Column(String(16), nullable=True)
    entry_regime_multiplier = Column(Float,   nullable=True)
    partial_skipped         = Column(Boolean, nullable=False, default=False, server_default=false())
    hold_minutes            = Column(Float,   nullable=True)
    sizing_detail           = Column(JSON,    nullable=True)   # SizingDecision record
    entry_indicators        = Column(JSON,    nullable=True)   # indicator snapshot at entry

    __table_args__ = (
        Index("ix_pp_strategy_open",   "strategy_name", "is_open"),
        Index("ix_pp_token_open",      "token", "is_open"),
        Index("ix_pp_underlying_date", "underlying", "entry_time"),
        Index("ix_pp_run_entry",       "run_id", "entry_time"),
    )


class DailyPnLSummary(Base):
    """Aggregated daily P&L per strategy — for performance dashboard."""
    __tablename__ = "daily_pnl_summary"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    date           = Column(Date,       nullable=False)
    strategy_name  = Column(String(50), nullable=False)
    mode           = Column(SAEnum(TradingMode), nullable=False)

    total_trades   = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades  = Column(Integer, nullable=False, default=0)
    win_rate       = Column(Float,   nullable=True)

    gross_pnl      = Column(Float,   nullable=False, default=0.0)
    brokerage      = Column(Float,   nullable=False, default=0.0)
    net_pnl        = Column(Float,   nullable=False, default=0.0)

    # Largest single-trade net loss that day, stored positive. Renamed from
    # max_drawdown in migration 0013 (R15.4): it was never an equity drawdown.
    worst_trade_loss = Column(Float, nullable=True)
    avg_rr         = Column(Float,   nullable=True)   # sum_r / r_sample_size (legacy name)
    total_lots     = Column(Integer, nullable=True)

    # ── migration 0011 ──────────────────────────────────────────────────────
    run_id              = Column(String(32), nullable=False, default="PAPER", server_default="PAPER")
    statutory_charges   = Column(Float,   nullable=False, default=0.0, server_default="0")
    flat_trades         = Column(Integer, nullable=False, default=0, server_default="0")
    sum_r               = Column(Float,   nullable=False, default=0.0, server_default="0")
    sum_r_win           = Column(Float,   nullable=False, default=0.0, server_default="0")
    sum_r_loss          = Column(Float,   nullable=False, default=0.0, server_default="0")
    r_sample_size       = Column(Integer, nullable=False, default=0, server_default="0")
    expectancy_r        = Column(Float,   nullable=True)
    expectancy_inr      = Column(Float,   nullable=True)
    gross_profit        = Column(Float,   nullable=False, default=0.0, server_default="0")
    gross_loss          = Column(Float,   nullable=False, default=0.0, server_default="0")
    max_equity_drawdown = Column(Float,   nullable=True)   # peak-to-trough of the day's cumulative net

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_dpnl_date_strategy", "date", "strategy_name", "mode", "run_id", unique=True),
    )


class TradingConfig(Base):
    """
    Runtime configuration: mode, risk limits, strategy enable/disable.
    Single row — updated via API.
    """
    __tablename__ = "trading_config"

    id                    = Column(Integer, primary_key=True, default=1)
    mode                  = Column(SAEnum(TradingMode), nullable=False, default=TradingMode.PAPER)

    # Risk limits (per day)
    max_daily_loss        = Column(Float,   nullable=False, default=5000.0)
    max_daily_loss_pct    = Column(Float,   nullable=False, default=2.0)    # % of capital
    max_open_positions    = Column(Integer, nullable=False, default=6)
    max_lots_per_trade    = Column(Integer, nullable=False, default=5)
    capital               = Column(Float,   nullable=False, default=100000.0)

    # Slippage simulation (paper mode)
    slippage_pct          = Column(Float,   nullable=False, default=0.2)    # 0.2% of LTP

    # ── Capital / trade sizing ──────────────────────────────────────────────
    # Demoted to a nullable display cache in migration 0013 (design §9.5):
    # the per-trade budget is fixed_fraction_pct × capital. A stored value
    # that diverges from that by more than ₹500 fails startup check 6.
    trade_amount          = Column(Float,   nullable=True)
    # Running total of capital currently locked in open positions
    used_capital          = Column(Float,   nullable=False, default=0.0)
    # Hard cap on trades per day (resets midnight)
    max_trades_per_day    = Column(Integer, nullable=False, default=10)
    # Counter incremented on every new trade entry, reset daily
    daily_trades_count    = Column(Integer, nullable=False, default=0)

    # Strategy enable flags (JSON: {strategy_name: bool})
    strategy_flags        = Column(JSON,    nullable=True)

    # Safety: auto square-off time
    square_off_time       = Column(String(5), nullable=False, default="15:15")

    # Live mode guard: must be explicitly set to True to go live
    live_mode_confirmed   = Column(Boolean,  nullable=False, default=False)

    # study_phase: always False — paper trading is active immediately.
    # Kept in schema for API compatibility; cannot be set to True via normal config updates.
    study_phase           = Column(Boolean,  nullable=False, default=False)

    # Kill switch (migration 0010): persisted so a halt survives restarts.
    # ensure_config()/update_config() never touch it implicitly.
    trading_halted        = Column(Boolean,  nullable=False, default=False, server_default=false())
    halted_reason         = Column(String(200), nullable=True)
    halted_at             = Column(DateTime(timezone=True), nullable=True)
    # IST trading date that daily_trades_count belongs to (migration 0010)
    daily_trades_date     = Column(Date, nullable=True)

    # ── migration 0013: operator-facing risk numbers (overlay TradingParams) ─
    open_exposure_cap_pct      = Column(Float,   nullable=False, default=40.0, server_default="40.0")
    fixed_fraction_pct         = Column(Float,   nullable=False, default=10.0, server_default="10.0")
    kelly_multiplier           = Column(Float,   nullable=False, default=0.25, server_default="0.25")
    kelly_fraction_cap_pct     = Column(Float,   nullable=False, default=15.0, server_default="15.0")
    kelly_min_trades           = Column(Integer, nullable=False, default=30, server_default="30")
    losing_streak_halt_n       = Column(Integer, nullable=False, default=3, server_default="3")
    strategy_streak_halt_n     = Column(Integer, nullable=False, default=2, server_default="2")
    post_loss_cooldown_minutes = Column(Integer, nullable=False, default=15, server_default="15")
    min_rr                     = Column(Float,   nullable=False, default=2.0, server_default="2.0")
    stale_tick_seconds         = Column(Integer, nullable=False, default=60, server_default="60")
    entry_cutoff_buffer_min    = Column(Integer, nullable=False, default=10, server_default="10")

    sizing_policy = Column(String(16), nullable=False, default="ONE_LOT", server_default="ONE_LOT")
    planned_risk_pct = Column(Float, nullable=False, default=0.5, server_default="0.5")
    aggregate_risk_pct = Column(Float, nullable=False, default=1.0, server_default="1.0")
    stop_gap_buffer_pct = Column(Float, nullable=False, default=0.2, server_default="0.2")

    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())


class StrategyPerf(Base):
    """
    Rolling performance stats per strategy — the inputs to Kelly sizing.

    Updated after every closed position, from one TradeOutcome (design §7).

    Kelly (risk_gate.fraction_decision, design §9.5), over R-multiples:
        decided_r = winning_trades_r + losing_trades_r   (R-valid, non-R-flat)
        p         = winning_trades_r / decided_r
        b         = avg_win_r / avg_loss_r      (NOT the mean R: that was bug 6)
        f         = min(kelly_multiplier × (p·b − q) / b,
                        min(kelly_fraction_cap, fixed_fraction))
    base_amount = round_to_500(f × capital); no floor — a strategy with no edge
    is skipped (KELLY_NO_EDGE). Below kelly_min_trades decided trades the
    fixed fraction applies. ``kelly_fraction`` here is only a cache of f.
    """
    __tablename__ = "strategy_perf"

    id            = Column(Integer,     primary_key=True, autoincrement=True)
    strategy_name = Column(String(50),  nullable=False)

    # Rolling counters (all-time, not daily)
    total_trades   = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)

    # Running averages
    win_rate       = Column(Float, nullable=True)   # 0–100
    avg_rr         = Column(Float, nullable=True)   # average achieved R:R

    # Computed half-Kelly fraction (cached — avoids recomputing on every signal)
    kelly_fraction = Column(Float, nullable=True)   # 0.0–0.25 (half-Kelly, capped)

    # ── migration 0011: exact sums (design §7, §9.5) ────────────────────────
    run_id          = Column(String(32), nullable=False, default="PAPER", server_default="PAPER")
    losing_trades   = Column(Integer, nullable=False, default=0, server_default="0")
    flat_trades     = Column(Integer, nullable=False, default=0, server_default="0")
    sum_r           = Column(Float,   nullable=False, default=0.0, server_default="0")
    sum_r_win       = Column(Float,   nullable=False, default=0.0, server_default="0")
    sum_r_loss      = Column(Float,   nullable=False, default=0.0, server_default="0")
    r_sample_size   = Column(Integer, nullable=False, default=0, server_default="0")
    # R-valid outcome counts keyed on sign(r_multiple): the only Kelly denominators
    winning_trades_r = Column(Integer, nullable=False, default=0, server_default="0")
    losing_trades_r  = Column(Integer, nullable=False, default=0, server_default="0")
    avg_win_r       = Column(Float,   nullable=True)
    avg_loss_r      = Column(Float,   nullable=True)   # mean magnitude of negative R
    expectancy_r    = Column(Float,   nullable=True)
    expectancy_inr  = Column(Float,   nullable=True)
    gross_profit    = Column(Float,   nullable=False, default=0.0, server_default="0")
    gross_loss      = Column(Float,   nullable=False, default=0.0, server_default="0")

    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("strategy_name", "run_id", name="uq_strategy_perf_name_run"),
        Index("ix_strategy_perf_name", "strategy_name"),
    )
