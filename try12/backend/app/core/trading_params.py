"""
TradingParams — every operator-owned number a trading decision may read
(design §19.1 defaults, §6.3 check 6 validation, R18.1/R18.2).

Live paper trading resolves it through ``DbTradingParams`` (code defaults
overlaid by the ``trading_config`` columns that carry the same name, TTL
cached). Replay resolves it from ``--params-file`` overlaid on the same
defaults. Both call ``validate()``; a non-empty result is a violation list.

Two keys are deliberately absent and rejected if supplied (design §3
deviations): ``fraction_floor`` (D5 — a floor sizes *up* a negative-edge
strategy; the no-edge path skips instead) and ``breakeven_buffer_reward``
(review finding 8 — it never entered the stop arithmetic; a negative tier
lock expresses the same thing).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from app.services.trading.trailing import (
    DEFAULT_TRAIL_TIERS,
    PARTIAL_EXIT_FRACTION,
    PARTIAL_EXIT_REWARD_FRAC,
    TrailTier,
    validate_ladder,
)

FORBIDDEN_KEYS: dict[str, str] = {
    "fraction_floor": "the Kelly fraction floor was removed (D5); a no-edge strategy is skipped",
    "breakeven_buffer_reward": "the breakeven buffer was removed; use a negative tier lock_reward_frac",
}

TRADE_AMOUNT_TOLERANCE = 500.0     # ₹ — display cache vs fixed_fraction × capital
MAX_HOLD_MINUTES_MAX = 374         # one session is 375 one-minute candles
MAX_FIXED_FRACTION_PCT = 40.0      # % of capital one trade may be sized on (the only bound)


def _default_max_hold() -> dict[str, int]:
    return {"VWAP_REVERSAL": 25, "RSI_VWAP": 25, "VWAP_PULLBACK": 25, "DEFAULT": 45}


def _default_dte_bands() -> dict[str, tuple[tuple[int, int], ...]]:
    return {"NIFTY": ((2, 7), (8, 14)), "BANKNIFTY": ((2, 7), (8, 14), (15, 30))}


def _default_premium_band() -> dict[str, tuple[float, float]]:
    return {"NIFTY": (10.0, 700.0), "BANKNIFTY": (10.0, 1500.0)}


class TradingParamsError(ValueError):
    """Raised by ``TradingParams.from_mapping`` when the mapping is invalid."""

    def __init__(self, violations: list[str]):
        self.violations = list(violations)
        super().__init__("; ".join(self.violations))


@dataclass(frozen=True)
class TradingParams:
    # ── capital and caps (trading_config) ────────────────────────────────────
    capital: float = 200_000.0
    fixed_fraction_pct: float = 10.0
    trade_amount: Optional[float] = None          # demoted display cache (§9.5)
    open_exposure_cap_pct: float = 40.0
    max_open_positions: int = 3
    max_lots_per_trade: int = 3
    max_trades_per_day: int = 6
    max_daily_loss: float = 6_000.0
    max_daily_loss_pct: float = 3.0
    sizing_policy: str = "ONE_LOT"
    planned_risk_pct: float = 0.5
    aggregate_risk_pct: float = 1.0
    stop_gap_buffer_pct: float = 0.2

    # ── halts (trading_config) ───────────────────────────────────────────────
    losing_streak_halt_n: int = 3
    strategy_streak_halt_n: int = 2
    post_loss_cooldown_minutes: int = 15

    # ── Kelly (trading_config) ───────────────────────────────────────────────
    kelly_multiplier: float = 0.25
    kelly_fraction_cap_pct: float = 15.0
    kelly_min_trades: int = 30

    # ── caches / regime ──────────────────────────────────────────────────────
    cooldown_mirror_ttl_s: int = 120
    params_cache_ttl_s: int = 30
    transition_cooldown_candles: int = 3

    # ── entry gate ───────────────────────────────────────────────────────────
    min_rr: float = 2.0
    stale_tick_seconds: int = 60
    entry_cutoff_buffer_min: int = 10
    sl_pct: float = 0.15
    target_pct: float = 0.30

    # ── exit ladder (§8.1, §8.2) ─────────────────────────────────────────────
    trail_tiers: tuple[TrailTier, ...] = DEFAULT_TRAIL_TIERS
    partial_exit_enabled: bool = True
    partial_exit_reward_frac: float = PARTIAL_EXIT_REWARD_FRAC
    partial_exit_fraction: float = PARTIAL_EXIT_FRACTION
    max_hold_minutes: dict[str, int] = field(default_factory=_default_max_hold)
    timeout_exit_min_gain_pct: float = 5.0

    # ── option selection (§10) ───────────────────────────────────────────────
    dte_bands: dict[str, tuple[tuple[int, int], ...]] = field(default_factory=_default_dte_bands)
    dte1_cutoff_ist: str = "12:00"
    max_spread_pct: float = 3.0
    min_oi: int = 50_000
    min_volume: int = 1_000
    max_slippage_pct: float = 2.0
    min_score: float = 40.0
    premium_band: dict[str, tuple[float, float]] = field(default_factory=_default_premium_band)

    # ── fills and costs (§15) ────────────────────────────────────────────────
    slippage_pct: float = 0.2                     # LTP fallback only
    quote_slippage_pct: float = 0.05              # every valid-quote fill
    quote_max_spread_pct: float = 5.0
    allow_ltp_only_fill: bool = False
    brokerage_per_lot_per_leg: float = 20.0
    stt_sell_pct: float = 0.15
    exchange_txn_pct: float = 0.03553
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.003
    gst_pct: float = 18.0
    max_cost_fraction_of_edge: float = 0.33

    # ── accounting / reporting ───────────────────────────────────────────────
    r_flat_epsilon: float = 0.01
    flat_epsilon: float = 1.0
    min_display_sample: int = 20
    search_trail_overlap_threshold: float = 0.5

    # ── replay (§14.3) ───────────────────────────────────────────────────────
    replay_min_strikes: int = 8
    replay_max_bracket_gap_s: int = 120
    replay_min_option_tick_minutes: int = 300

    # ── derived ──────────────────────────────────────────────────────────────

    @property
    def kelly_cap_effective_pct(self) -> float:
        """The Kelly fraction's ceiling: Kelly can only size DOWN from the fixed
        fraction, never above it, whatever kelly_fraction_cap_pct says."""
        return min(self.kelly_fraction_cap_pct, self.fixed_fraction_pct)

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_mapping(cls, overrides: Mapping[str, Any]) -> "TradingParams":
        """Defaults overlaid by *overrides*. Raises TradingParamsError on any violation."""
        violations = _key_violations(overrides)
        if violations:
            raise TradingParamsError(violations)
        params = dataclasses.replace(cls(), **_coerce(dict(overrides)))
        violations = params.validate()
        if violations:
            raise TradingParamsError(violations)
        return params

    # ── validation (§6.3 check 6) ────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Every violation, each naming its key. Empty list = valid."""
        from app.services.trading.regime_detector import (
            TRANSITION_COOLDOWN_CANDLES_MAX,
            _history_len,
        )
        from app.strategies.names import ALL_STRATEGY_NAMES

        v: list[str] = []

        # ladder monotonicity and reachability
        v += validate_ladder(self.trail_tiers, self.partial_exit_reward_frac,
                             self.partial_exit_fraction, self.sl_pct, self.target_pct)

        if self.min_rr < 2.0:
            v.append(f"min_rr={self.min_rr} must be >= 2.0")

        if self.capital <= 0:
            v.append(f"capital={self.capital} must be > 0")
        if not (0 < self.max_daily_loss <= self.capital):
            v.append(f"max_daily_loss={self.max_daily_loss} must be in (0, capital={self.capital}]")
        if not (0 < self.max_daily_loss_pct <= 100):
            v.append(f"max_daily_loss_pct={self.max_daily_loss_pct} must be in (0, 100]")

        if not (0 < self.fixed_fraction_pct <= MAX_FIXED_FRACTION_PCT):
            v.append(f"fixed_fraction_pct={self.fixed_fraction_pct} must be in "
                     f"(0, {MAX_FIXED_FRACTION_PCT:g}]")
        # kelly_fraction_cap_pct may sit below fixed_fraction_pct: Kelly is capped at
        # kelly_cap_effective_pct = min(both), so it can only size down.
        if not (0 < self.kelly_multiplier <= 0.5):
            v.append(f"kelly_multiplier={self.kelly_multiplier} must be in (0, 0.5]")
        if not (0 < self.kelly_fraction_cap_pct <= 25.0):
            v.append(f"kelly_fraction_cap_pct={self.kelly_fraction_cap_pct} must be in (0, 25.0]")

        v += self._dte_band_violations()

        for key in ("brokerage_per_lot_per_leg", "stt_sell_pct", "exchange_txn_pct",
                    "sebi_per_crore", "stamp_buy_pct", "gst_pct", "slippage_pct"):
            val = getattr(self, key)
            if val < 0:
                v.append(f"{key}={val} must be >= 0")

        import math
        if self.sizing_policy not in ("ONE_LOT", "BUDGET"):
            v.append("sizing_policy must be ONE_LOT or BUDGET")
        for key in ("planned_risk_pct", "aggregate_risk_pct"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 < value <= 100:
                v.append(f"{key} must be finite and in (0, 100]")
        if not math.isfinite(self.stop_gap_buffer_pct) or not 0 <= self.stop_gap_buffer_pct <= 100:
            v.append("stop_gap_buffer_pct must be finite and in [0, 100]")
        for key in ("max_lots_per_trade", "max_open_positions", "max_trades_per_day"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                v.append(f"{key} must be a positive integer")
        if not 0 < self.open_exposure_cap_pct <= 100:
            v.append("open_exposure_cap_pct must be in (0, 100]")

        c = self.transition_cooldown_candles
        if not (1 <= c <= TRANSITION_COOLDOWN_CANDLES_MAX):
            v.append(
                f"transition_cooldown_candles={c} must be in "
                f"[1, {TRANSITION_COOLDOWN_CANDLES_MAX}]"
            )
        elif not (c < _history_len(c) - 1):
            v.append(
                f"transition_cooldown_candles={c} must be < history length "
                f"{_history_len(c)} - 1"
            )

        allowed = set(ALL_STRATEGY_NAMES) | {"DEFAULT"}
        mh = self.max_hold_minutes
        if "DEFAULT" not in mh:
            v.append("max_hold_minutes must contain a 'DEFAULT' key")
        unknown = sorted(set(mh) - allowed)
        if unknown:
            v.append(f"max_hold_minutes has unknown strategy keys {unknown}")
        for k, val in mh.items():
            if not (1 <= val <= MAX_HOLD_MINUTES_MAX):
                v.append(f"max_hold_minutes[{k!r}]={val} must be in [1, {MAX_HOLD_MINUTES_MAX}]")

        if not (0.0 <= self.timeout_exit_min_gain_pct < self.target_pct * 100):
            v.append(
                f"timeout_exit_min_gain_pct={self.timeout_exit_min_gain_pct} must be in "
                f"[0, target_pct×100={self.target_pct * 100:g})"
            )

        if not (0 <= self.quote_slippage_pct <= self.slippage_pct):
            v.append(
                f"quote_slippage_pct={self.quote_slippage_pct} must be in "
                f"[0, slippage_pct={self.slippage_pct}]"
            )

        if self.trade_amount is not None:
            derived = self.fixed_fraction_pct / 100.0 * self.capital
            if abs(self.trade_amount - derived) > TRADE_AMOUNT_TOLERANCE:
                v.append(
                    f"trade_amount={self.trade_amount} must be within ₹{TRADE_AMOUNT_TOLERANCE:g} "
                    f"of fixed_fraction_pct×capital={derived:g}"
                )
        return v

    def _dte_band_violations(self) -> list[str]:
        out: list[str] = []
        for und, bands in self.dte_bands.items():
            prev_hi: Optional[int] = None
            for lo, hi in bands:
                if lo < 0 or lo > hi:
                    out.append(f"dte_bands[{und!r}] band ({lo}, {hi}) must satisfy 0 <= lo <= hi")
                if prev_hi is not None and lo <= prev_hi:
                    out.append(
                        f"dte_bands[{und!r}] band ({lo}, {hi}) must start after the previous "
                        f"band's end {prev_hi} (ordered, non-overlapping)"
                    )
                prev_hi = hi
        return out


def _key_violations(overrides: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    names = TradingParams.field_names()
    for key in overrides:
        if key in FORBIDDEN_KEYS:
            out.append(f"{key} is not a TradingParams key: {FORBIDDEN_KEYS[key]}")
        elif key not in names:
            out.append(f"{key} is not a TradingParams key")
    return out


def _coerce(overrides: dict[str, Any]) -> dict[str, Any]:
    """JSON-shaped values → the dataclass's tuple shapes (tiers, bands)."""
    if "trail_tiers" in overrides:
        overrides["trail_tiers"] = tuple(
            t if isinstance(t, TrailTier) else (
                TrailTier(**t) if isinstance(t, Mapping) else TrailTier(*t)
            )
            for t in overrides["trail_tiers"]
        )
    if "dte_bands" in overrides:
        overrides["dte_bands"] = {
            k: tuple(tuple(b) for b in bands) for k, bands in overrides["dte_bands"].items()
        }
    if "premium_band" in overrides:
        overrides["premium_band"] = {k: tuple(b) for k, b in overrides["premium_band"].items()}
    return overrides


def validate_overrides(overrides: Mapping[str, Any]) -> list[str]:
    """Violations for defaults overlaid by *overrides* (unknown/forbidden keys included)."""
    violations = _key_violations(overrides)
    if violations:
        return violations
    try:
        params = dataclasses.replace(TradingParams(), **_coerce(dict(overrides)))
    except (TypeError, ValueError) as exc:
        return [f"invalid TradingParams value: {exc}"]
    return params.validate()
