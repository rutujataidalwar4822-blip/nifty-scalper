"""
Option Selector Engine — Hard Filters + Multi-Factor Scoring Model
===================================================================

Architecture:
  Signal arrives (already past the risk gate's phase one)
       ↓
  EXPIRY POLICY  (per-underlying DTE bands: DTE 0 never, DTE 1 only before noon)
       ↓
  HARD FILTERS  (binary — any fail = this contract is out, ReasonCode tallied)
  ─────────────────────────────────────────────────────────────
  R:R         un-stretched target_pct / sl_pct (RR_EPSILON), floor = params().min_rr
  Tick        no tick / no LTP                         → TICK_UNAVAILABLE
  Premium     LTP outside the per-underlying band      → CONTRACT_UNSUITABLE
              LTP above what the budget can buy        → PREMIUM_ABOVE_AFFORDABLE
  Quote       missing / crossed bid-ask                → QUOTE_UNAVAILABLE
              older than stale_tick_seconds            → QUOTE_STALE
  Spread      > max_spread_pct of mid                  → SPREAD_TOO_WIDE
  OI          missing (or the -1 sentinel)             → OI_UNAVAILABLE
              < min_oi                                 → LIQUIDITY_TOO_LOW
  Volume      missing (or the -1 sentinel)             → VOLUME_UNAVAILABLE
              < min_volume                             → VOLUME_INSUFFICIENT
  Slippage    half-spread > max_slippage_pct           → SLIPPAGE_TOO_HIGH
  ─────────────────────────────────────────────────────────────
       ↓
  SOFT SCORING  (0–100 composite, 8 factors)
       ↓
  SELECT best scoring candidate (if score ≥ params().min_score)
       ↓
  NO TRADE if all candidates filtered/scored-out

Missing data is a rejection, never a skipped filter. Every threshold comes from
TradingParams; there are no module-level R:R, premium or liquidity constants.

Principle:
  Great NIFTY signal + Bad option → NO TRADE
  Mediocre NIFTY signal + Great option → STILL NO TRADE (signal gate upstream)

Scoring factors (weights sum to 1.0):
  Factor           Weight   Measures
  ──────────────────────────────────────────────────────────────────
  Liquidity         0.20    OI + Volume — clean entry/exit
  Spread            0.18    Bid-ask % — execution cost
  Delta             0.18    0.50–0.65 ideal for scalping (ATM/1 OTM)
  Moneyness         0.15    1–2 strikes OTM preferred
  OI change         0.12    Rising OI = fresh positioning = conviction
  Volume trend      0.08    vs recent average
  Expiry            0.06    2–7 DTE sweet spot
  Price behavior    0.03    LTP momentum direction
  ──────────────────────────────────────────────────────────────────
"""
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.trading_params import TradingParams
from app.db.models.instrument import Instrument, InstrumentType, OptionType
from app.runtime.context import ctx, market, now_ist, now_utc, params, trading_day
from app.services.trading import fills
from app.services.trading.fills import RR_EPSILON
from app.services.trading.no_trade import ReasonCode
from app.strategies.base import Direction

logger = logging.getLogger(__name__)

# ── Scoring weights ────────────────────────────────────────────────────────────
W_LIQUIDITY    = 0.20
W_SPREAD       = 0.18
W_DELTA        = 0.18
W_MONEYNESS    = 0.15
W_OI_CHANGE    = 0.12
W_VOLUME_TREND = 0.08
W_EXPIRY       = 0.06
W_PRICE_BEHAV  = 0.03

assert abs(W_LIQUIDITY + W_SPREAD + W_DELTA + W_MONEYNESS +
           W_OI_CHANGE + W_VOLUME_TREND + W_EXPIRY + W_PRICE_BEHAV - 1.0) < 1e-9

# ── Soft-scoring scales (not thresholds: nothing is accepted or rejected on these) ─
OI_GOOD          = 5_00_000   # 5 lakh OI = excellent liquidity
OI_OK            = 1_00_000   # 1 lakh OI = acceptable
VOL_GOOD         = 30_000     # 30k contracts = good volume
NIFTY_STEP       = 50         # Nifty strike gap
BANKNIFTY_STEP   = 100        # BankNifty strike gap

SCORE_TABLE_ROWS = 8          # rows of the scoring table persisted with the signal


@dataclass
class FilterResult:
    """Result of hard filter check — passed or rejected with reason."""
    passed:  bool
    reason:  str = ""          # populated when passed=False
    filter:  str = ""          # the ReasonCode value that triggered the rejection


def _reject(code: ReasonCode, reason: str) -> FilterResult:
    return FilterResult(passed=False, filter=code.value, reason=reason)


def _quote_usable(bid, ask) -> bool:
    try:
        return bid is not None and ask is not None \
            and math.isfinite(float(bid)) and math.isfinite(float(ask)) \
            and float(bid) > 0 and float(ask) >= float(bid)
    except (TypeError, ValueError):
        return False


def _hard_filter(
    ltp:        float,
    bid:        Optional[float],
    ask:        Optional[float],
    oi:         Optional[int],
    volume:     Optional[int],
    sl_pct:     float,
    target_pct: float,
    underlying: str = "NIFTY",
    *,
    min_rr:     float,
    tick_age_s: Optional[float] = None,
    max_premium: Optional[float] = None,
    tp:         Optional[TradingParams] = None,
) -> FilterResult:
    """
    Run every hard filter in order; the first failure wins and names a ReasonCode.

    ``min_rr`` is a required argument: the floor has one source,
    ``params().min_rr``, read by the caller at the point of use (design §10.1,
    startup check 7). ``oi``/``volume`` of None (or the legacy -1 sentinel) and a
    missing quote are rejections, not skipped filters (§10.2). ``max_premium`` is
    the affordability ceiling ``base_amount × regime_multiplier / lot_size``.

    These are BINARY gates — if any fails the option is rejected
    regardless of how well it scores on other factors.
    """
    tp = tp or params()

    # ── Filter 0: R:R on the UN-stretched setup ───────────────────────────────
    # The old gate stretched the target to the floor before testing it, which
    # made it identically true. A setup below 1:2 is rejected as it stands.
    if (sl_pct is None or target_pct is None
            or not all(math.isfinite(v) for v in (sl_pct, target_pct, min_rr))
            or not 0 < sl_pct < 1 or target_pct <= 0 or min_rr <= 0):
        return _reject(ReasonCode.RR_INSUFFICIENT,
                       f"R:R undefined: sl_pct={sl_pct} target_pct={target_pct}")
    rr = target_pct / sl_pct
    if rr + RR_EPSILON < min_rr:
        return _reject(ReasonCode.RR_INSUFFICIENT,
                       f"R:R {rr:.3f} < minimum {min_rr} (sl_pct={sl_pct}, "
                       f"target_pct={target_pct}); the target is never stretched to fit")

    # ── Filter 1: Contract suitability (premium band) and affordability ───────
    if not math.isfinite(ltp) or ltp <= 0:
        return _reject(ReasonCode.TICK_UNAVAILABLE, f"Invalid LTP: {ltp}")
    lo, hi = tp.premium_band.get((underlying or "NIFTY").upper(), tp.premium_band["NIFTY"])
    if ltp < lo:
        return _reject(ReasonCode.CONTRACT_UNSUITABLE,
                       f"LTP ₹{ltp:.2f} < minimum ₹{lo} — too cheap, "
                       f"gamma risk too high near expiry")
    if ltp > hi:
        return _reject(ReasonCode.CONTRACT_UNSUITABLE,
                       f"LTP ₹{ltp:.2f} > maximum ₹{hi} — too expensive, "
                       f"can't achieve 2× target within reasonable move")
    if max_premium is not None and ltp > max_premium:
        return _reject(ReasonCode.PREMIUM_ABOVE_AFFORDABLE,
                       f"LTP ₹{ltp:.2f} > affordable ceiling ₹{max_premium:.2f} "
                       f"(budget cannot buy one lot)")

    # ── Filter 2: quote presence and freshness ────────────────────────────────
    has_quote = _quote_usable(bid, ask)
    if not has_quote and not tp.allow_ltp_only_fill:
        return _reject(ReasonCode.QUOTE_UNAVAILABLE,
                       f"no usable bid/ask (bid={bid} ask={ask}) — entries fill off the "
                       f"ask, so a quote-less contract is refused")
    if (tick_age_s is None or not math.isfinite(tick_age_s)
            or tick_age_s < 0 or tick_age_s > tp.stale_tick_seconds):
        age = "unknown" if tick_age_s is None else f"{tick_age_s:.0f}s"
        return _reject(ReasonCode.QUOTE_STALE,
                       f"quote age {age} > {tp.stale_tick_seconds}s")

    # ── Filter 3: bid-ask spread ──────────────────────────────────────────────
    if has_quote and float(ask) > float(bid):
        mid = (float(bid) + float(ask)) / 2.0
        spread_pct = (float(ask) - float(bid)) / mid * 100.0
        if spread_pct > tp.max_spread_pct:
            return _reject(ReasonCode.SPREAD_TOO_WIDE,
                           f"Spread {spread_pct:.2f}% > limit {tp.max_spread_pct}% — "
                           f"bid={bid} ask={ask}. Entry cost too high.")

    # ── Filter 4: OI (liquidity) — missing is a rejection ─────────────────────
    if oi is None or oi < 0:
        return _reject(ReasonCode.OI_UNAVAILABLE, f"OI not received from feed (oi={oi})")
    if oi < tp.min_oi:
        return _reject(ReasonCode.LIQUIDITY_TOO_LOW,
                       f"OI {oi:,} < minimum {tp.min_oi:,} — insufficient "
                       f"open interest, exit may be difficult")

    # ── Filter 5: Volume — missing is a rejection ─────────────────────────────
    if volume is None or volume < 0:
        return _reject(ReasonCode.VOLUME_UNAVAILABLE,
                       f"volume not received from feed (volume={volume})")
    if volume < tp.min_volume:
        return _reject(ReasonCode.VOLUME_INSUFFICIENT,
                       f"Volume {volume:,} < minimum {tp.min_volume:,} — "
                       f"low participation, wide fills expected")

    # ── Filter 6: Expected slippage (half the spread as one-way impact) ───────
    if has_quote and float(ask) > float(bid):
        mid = (float(bid) + float(ask)) / 2.0
        expected_slippage_pct = ((float(ask) - mid) / mid) * 100.0
        if expected_slippage_pct > tp.max_slippage_pct:
            return _reject(ReasonCode.SLIPPAGE_TOO_HIGH,
                           f"Expected slippage {expected_slippage_pct:.2f}% > "
                           f"limit {tp.max_slippage_pct}% — market impact will erode edge")

    return FilterResult(passed=True)


@dataclass
class ScoredCandidate:
    """One scored option candidate with all factor breakdown."""
    token:        str
    symbol:       str
    strike:       float
    option_type:  str
    expiry:       date
    ltp:          float
    bid:          float
    ask:          float
    spread_pct:   float
    oi:           int
    oi_change:    int
    volume:       int
    delta_est:    float
    iv:           float
    lot_size:     int

    # Score breakdown
    score:           float
    score_liquidity: float = 0.0
    score_spread:    float = 0.0
    score_delta:     float = 0.0
    score_moneyness: float = 0.0
    score_oi_change: float = 0.0
    score_vol_trend: float = 0.0
    score_expiry:    float = 0.0
    score_price_beh: float = 0.0

    # The single tick read that produced every price above (design §10.3): the
    # hard filter, the score, the fill and the momentum check all judge THIS quote.
    tick_age_s:   Optional[float] = None
    tick_open:    float = 0.0

    # Hard filter result (always passed=True for candidates that reach scoring)
    filter_result: FilterResult = None

    @property
    def selected(self) -> bool:
        return self.score >= params().min_score


# Keep SelectedOption as alias for backward compatibility
SelectedOption = ScoredCandidate


@dataclass(frozen=True)
class ExpiryChoice:
    """The expiry the policy picked, or why none qualified (design §10.4)."""
    expiry:   Optional[date]
    dte:      Optional[int]
    band:     Optional[str]           # PREFERRED | FALLBACK_n | DTE1_MORNING | EXPLICIT
    reason:   str                     # rationale, persisted with the signal
    cause:    Optional[str] = None    # NO_INSTRUMENT_MASTER | NO_BANDED_EXPIRY when expiry is None
    observed: tuple = ()              # every expiry in the universe, with its DTE


@dataclass(frozen=True)
class SelectionResult:
    candidate:          Optional[ScoredCandidate]
    rejections:         dict          # ReasonCode value -> count
    evaluated:          int
    expiry:             ExpiryChoice
    ltp_only_fallbacks: int
    score_table:        list          # top rows, persisted with the signal
    ranked:             list = field(default_factory=list)   # every passing candidate, best first


# ── Expiry policy ─────────────────────────────────────────────────────────────

def _band_label(i: int) -> str:
    return "PREFERRED" if i == 0 else f"FALLBACK_{i}"


def _expiry_date(value) -> Optional[date]:
    """Normalize DB/feed expiry values before doing calendar-day arithmetic.

    Instrument masters may return a ``date``, a timezone-aware ``datetime``, or
    an ISO string. Comparing those forms directly can make a valid expiry look
    absent, especially around the UTC/IST day boundary.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def choose_expiry(expiries: list[date], *, today: date, now_t: dtime, underlying: str,
                  tp: TradingParams) -> ExpiryChoice:
    """Walk ``tp.dte_bands[underlying]`` in order and return the first expiry whose
    DTE falls in a band. DTE 0 is never traded; DTE 1 only before
    ``dte1_cutoff_ist`` and only when no banded expiry exists. Pure."""
    normalized = [e for raw in expiries if (e := _expiry_date(raw)) is not None]
    distinct = sorted(set(normalized))
    observed = tuple((e.isoformat(), (e - today).days) for e in distinct)
    if not distinct:
        return ExpiryChoice(None, None, None,
                            f"NO_INSTRUMENT_MASTER: no instruments for {underlying}",
                            cause="NO_INSTRUMENT_MASTER", observed=observed)
    bands = tp.dte_bands.get(underlying.upper(), ())
    for i, (lo, hi) in enumerate(bands):
        for e in distinct:
            dte = (e - today).days
            if dte >= 2 and lo <= dte <= hi:
                return ExpiryChoice(e, dte, _band_label(i),
                                    f"{_band_label(i)} band [{lo},{hi}] DTE {dte}",
                                    observed=observed)
    hh, mm = (int(x) for x in tp.dte1_cutoff_ist.split(":")[:2])
    one = [e for e in distinct if (e - today).days == 1]
    if one and now_t < dtime(hh, mm):
        return ExpiryChoice(one[0], 1, "DTE1_MORNING",
                            f"DTE 1 before {tp.dte1_cutoff_ist} and no banded expiry",
                            observed=observed)
    return ExpiryChoice(
        None, None, None,
        f"NO_BANDED_EXPIRY: expiries {[o[0] for o in observed]} (DTE "
        f"{[o[1] for o in observed]}) fall outside bands {list(bands)}; nearest DTE "
        f"{observed[0][1]}",
        cause="NO_BANDED_EXPIRY", observed=observed)


# ── Black-Scholes delta approximation ─────────────────────────���───────────────

def _estimate_delta(spot: float, strike: float, option_type: str,
                    dte: int, iv: float = 0.20) -> float:
    """
    Approximate B-S delta using normal CDF.
    iv is already in decimal (0.20 = 20%).
    """
    if dte <= 0 or iv <= 0:
        if option_type == "CE":
            return 1.0 if spot > strike else 0.05
        else:
            return -1.0 if spot < strike else -0.05

    T   = max(dte / 365.0, 1 / 365.0)
    sqT = math.sqrt(T)
    try:
        d1 = (math.log(spot / strike) + (0.5 * iv ** 2) * T) / (iv * sqT)
    except (ValueError, ZeroDivisionError):
        return 0.5

    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    delta = _ncdf(d1) if option_type == "CE" else _ncdf(d1) - 1.0
    return round(delta, 4)


# ── Individual factor scorers (each returns 0–100) ────────────────────────────

def _score_liquidity(oi: int, volume: int) -> float:
    """
    Nifty options need real liquidity to enter and exit cleanly.
    Scale:
      OI ≥ 5L  → 100     (very liquid)
      OI ≥ 1L  → 60-100
      OI < 1L  → 0-60    (illiquid, penalised)
    Volume adds a smaller boost.
    """
    oi_score  = min(oi  / OI_GOOD,  1.0) * 100
    vol_score = min(volume / VOL_GOOD, 1.0) * 100
    return round(oi_score * 0.65 + vol_score * 0.35, 2)


def _score_spread(bid: float, ask: float, ltp: float) -> float:
    """
    Tight spread = lower execution cost = better score.
    Scale:
      spread < 0.3%  → 100
      spread = 1%    → 72
      spread = 2%    → 44
      spread ≥ 5%    → 0
    Formula: score = 100 - (spread_pct / 0.05) capped at 100
    """
    if not bid or not ask or ask <= bid:
        return 40.0   # unknown spread — moderate penalty
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 40.0
    spread_pct = (ask - bid) / mid * 100.0
    score = max(0.0, 100.0 - spread_pct * 20.0)   # 5% spread → 0
    return round(score, 2)


def _score_delta(spot: float, strike: float, option_type: str,
                 dte: int, iv: float) -> tuple[float, float]:
    """
    Ideal delta for scalping: 0.50–0.65
      - Too low (<0.35): deeply OTM — slow price action, lottery ticket
      - Too high (>0.75): deep ITM — expensive, low leverage
      - 0.50–0.65: ATM/1 OTM — good gamma (delta acceleration) and theta profit
    Returns (score, delta_value).
    """
    delta = abs(_estimate_delta(spot, strike, option_type, dte, iv))
    if 0.50 <= delta <= 0.65:
        # Perfect score for ATM/1 OTM sweet spot
        score = 100.0
    elif 0.35 <= delta < 0.50:
        # Linear interpolation 0→100 as delta approaches 0.50 from below
        score = ((delta - 0.35) / 0.15) * 100.0
    elif 0.65 < delta <= 0.75:
        # Linear decay 100→60 from 0.65 to 0.75 (slightly ITM)
        score = 100.0 - ((delta - 0.65) / 0.10) * 40.0
    elif delta > 0.75:
        # Further decay for deep ITM
        score = max(0.0, 60.0 - ((delta - 0.75) / 0.25) * 60.0)
    else:
        # delta < 0.35: steep penalty for deeply OTM
        score = max(0.0, (delta / 0.35) * 50.0)
    return round(score, 2), round(delta, 4)


def _score_moneyness(strike: float, spot: float, step: int,
                     option_type: str = "CE") -> float:
    """
    For scalping we want strikes that are:
      • 1 strike OTM:  100  (sweet spot — good delta + leverage)
      • ATM:           85   (too expensive, lower leverage)
      • 2 strikes OTM: 80
      • 3 strikes OTM: 55
      • 4+ strikes OTM: decaying rapidly

    OTM direction depends on option type:
      CE: strike > spot is OTM  (higher strike = more OTM)
      PE: strike < spot is OTM  (lower strike = more OTM)
    """
    if option_type == "CE":
        strikes_otm = (strike - spot) / step   # positive = OTM for CE
    else:
        strikes_otm = (spot - strike) / step   # positive = OTM for PE

    strikes_otm_abs = abs(strikes_otm)
    # ITM (strikes_otm < 0): penalise — expensive, low leverage
    if strikes_otm < 0:
        return max(0.0, 60.0 + strikes_otm * 15.0)   # deeper ITM → lower score

    if strikes_otm_abs == 0:         # ATM
        return 85.0
    elif strikes_otm_abs <= 1:
        return 100.0
    elif strikes_otm_abs <= 2:
        return 80.0
    elif strikes_otm_abs <= 3:
        return 55.0
    elif strikes_otm_abs <= 4:
        return 30.0
    else:
        return max(0.0, 30.0 - (strikes_otm_abs - 4) * 10)


def _score_oi_change(oi: int, oi_change: int) -> float:
    """
    Rising OI = fresh money entering = conviction.
    Falling OI = unwinding = weaker signal.

    Score:
      OI increasing >10%  → 100
      OI stable           → 60
      OI declining >10%   → 20
    """
    if oi <= 0:
        return 50.0   # no data
    pct_change = (oi_change / oi) * 100.0 if oi > 0 else 0.0

    if pct_change >= 10.0:
        return 100.0
    elif pct_change >= 2.0:
        return 60.0 + (pct_change - 2.0) / 8.0 * 40.0
    elif pct_change >= 0:
        return 60.0
    elif pct_change >= -5.0:
        return 60.0 + pct_change * 4.0   # 60 → 40
    else:
        return max(20.0, 40.0 + (pct_change + 5.0) * 4.0)


def _score_volume_trend(volume: int, avg_volume: float) -> float:
    """
    Elevated volume = increased participation = better fill quality.
    Score based on volume / average:
      2× average → 100
      1× average → 70
      0.5× average → 40
      0× → 0
    """
    if avg_volume <= 0:
        return 50.0
    ratio = volume / avg_volume
    if ratio >= 2.0:
        return 100.0
    elif ratio >= 1.0:
        return 70.0 + (ratio - 1.0) * 30.0
    else:
        return max(0.0, ratio * 70.0)


def _score_expiry(expiry: date) -> float:
    """
    Weekly expiry 2-7 DTE is ideal for scalping:
      - Max theta decay per day
      - Biggest intraday move per point of NIFTY

    DTE 1:     25  (expiry day AM only — not afternoon)
    DTE 2-3:   90
    DTE 4-7:   100 (sweet spot)
    DTE 8-14:  75
    DTE 15-30: 55
    DTE >30:   30  (monthly expiry — slow theta)
    """
    dte = max((expiry - trading_day()).days, 0)
    if dte == 0:
        return 10.0   # expiry day — avoid unless before noon
    elif dte == 1:
        return 25.0
    elif dte <= 3:
        return 90.0
    elif dte <= 7:
        return 100.0
    elif dte <= 14:
        return 75.0
    elif dte <= 30:
        return 55.0
    else:
        return 30.0


def _score_price_behavior(ltp: float, ltp_history: list[float]) -> float:
    """
    Is the option price trending in our favour?
    For CE: we want LTP rising (underlying going up).
    We compare last LTP to average of recent ticks.

    ltp_history: list of recent LTPs (last 3-5 ticks), newest last.
    """
    if not ltp_history or len(ltp_history) < 2:
        return 50.0   # no data

    recent_avg = sum(ltp_history[:-1]) / len(ltp_history[:-1])
    if recent_avg <= 0:
        return 50.0

    pct_change = (ltp - recent_avg) / recent_avg * 100.0
    # Positive = option rising = momentum
    if pct_change > 1.0:
        return 100.0
    elif pct_change > 0:
        return 50.0 + pct_change * 50.0
    elif pct_change >= -1.0:
        return 50.0 + pct_change * 30.0
    else:
        return max(0.0, 50.0 + pct_change * 30.0)


# ── Main scorer ─────────────────────────��──────────────────────────────────────

def score_candidate(
    inst:        Instrument,
    spot:        float,
    tick:        dict,
    expiry:      date,
    direction:   Direction,
    step:        int,
    ltp_history: list[float] = None,
    prev_oi:     Optional[int] = None,
    avg_volume:  float = 0.0,   # rolling average volume from Redis; 0 = unknown
    *,
    sl_pct:      float,
    target_pct:  float,
    tp:          Optional[TradingParams] = None,
    max_premium: Optional[float] = None,
) -> tuple[Optional[ScoredCandidate], FilterResult]:
    """
    Run the hard filters on ONE tick, then score the survivors.
    Returns (candidate, FilterResult). ``candidate`` is None when a hard filter
    rejected the contract; the FilterResult names the ReasonCode.
    """
    tp = tp or params()
    if (not isinstance(inst.lot_size, int) or isinstance(inst.lot_size, bool)
            or inst.lot_size <= 0 or getattr(inst, "tick_size", fills.TICK) != fills.TICK):
        return None, _reject(ReasonCode.INVALID_CONTRACT,
                             "Master lot size invalid or tick size unsupported")
    ltp    = float(tick.get("ltp")       or 0)
    _bid   = tick.get("best_bid")
    _ask   = tick.get("best_ask")
    bid    = float(_bid or 0)
    ask    = float(_ask or 0)
    _oi_raw     = tick.get("oi")
    _vol_raw    = tick.get("volume")
    oi          = int(_oi_raw)    if _oi_raw    is not None else None
    volume      = int(_vol_raw)   if _vol_raw   is not None else None
    iv_raw = float(tick.get("iv")        or 0)
    # Normalise IV to percentage form (20.0 for 20%).
    # Tick data from WebSocket may be decimal (0.20); chain_iv is already percent.
    iv = iv_raw * 100 if (0 < iv_raw < 1.0) else iv_raw
    age = fills.tick_age_seconds(tick, now_utc())

    opt_type_str = inst.option_type.value if inst.option_type else "CE"

    # ── STEP 1: Hard filters — binary pass/fail ───────────────────────────────
    hf = _hard_filter(
        ltp=ltp, bid=_bid, ask=_ask, oi=oi, volume=volume,
        sl_pct=sl_pct, target_pct=target_pct,
        underlying=inst.name or "NIFTY", min_rr=tp.min_rr,
        tick_age_s=age, max_premium=max_premium, tp=tp,
    )
    if not hf.passed:
        logger.debug(
            "[HardFilter] REJECTED %s %.0f%s  ltp=%.2f bid=%.2f ask=%.2f "
            "oi=%s vol=%s | [%s] %s",
            inst.symbol, inst.strike, opt_type_str, ltp, bid, ask, oi, volume,
            hf.filter, hf.reason,
        )
        return None, hf

    entry, _ = fills.entry_fill(ltp, bid, ask, age, tp)
    if max_premium is not None and entry > max_premium + 1e-9:
        return None, _reject(ReasonCode.PREMIUM_ABOVE_AFFORDABLE,
                             f"Executable fill {entry:.2f} exceeds ceiling {max_premium:.2f}")

    # ── STEP 2: Soft scoring ──────────────────────────────────────────────────
    iv_decimal = (iv or 20.0) / 100.0
    dte        = max((expiry - trading_day()).days, 0)
    oi_score   = oi        # both are real, non-negative values past the hard filter
    vol_score  = volume
    has_previous_oi = prev_oi is not None and prev_oi > 0
    oi_change  = oi_score - prev_oi if has_previous_oi else 0

    # ── Compute all 8 factor scores ─────────────────────────────────────────
    s_liq    = _score_liquidity(oi_score, vol_score)
    s_spread = _score_spread(bid, ask, ltp)
    s_delta, delta_val = _score_delta(spot, inst.strike, opt_type_str, dte, iv_decimal)
    s_money  = _score_moneyness(inst.strike, spot, step, opt_type_str)
    s_oi_chg = _score_oi_change(oi_score, oi_change) if has_previous_oi else 50.0
    s_voltrd = _score_volume_trend(vol_score, avg_volume)
    s_expiry = _score_expiry(expiry)
    s_pbehav = _score_price_behavior(ltp, ltp_history or [])

    # ── Weighted composite ──────────────────────────────────────────────────
    composite = (
        W_LIQUIDITY    * s_liq    +
        W_SPREAD       * s_spread +
        W_DELTA        * s_delta  +
        W_MONEYNESS    * s_money  +
        W_OI_CHANGE    * s_oi_chg +
        W_VOLUME_TREND * s_voltrd +
        W_EXPIRY       * s_expiry +
        W_PRICE_BEHAV  * s_pbehav
    )

    mid = (bid + ask) / 2.0 if bid and ask else ltp
    spread_pct = round((ask - bid) / mid * 100, 2) if mid > 0 and ask > bid else 0.0

    cand = ScoredCandidate(
        token        = inst.token,
        symbol       = inst.symbol,
        strike       = inst.strike,
        option_type  = opt_type_str,
        expiry       = expiry,
        ltp          = round(ltp, 2),
        bid          = round(bid, 2),
        ask          = round(ask, 2),
        spread_pct   = spread_pct,
        oi           = oi_score,
        oi_change    = oi_change,
        volume       = vol_score,
        delta_est    = delta_val,
        iv           = round(iv or 20.0, 2),
        lot_size     = inst.lot_size,
        score        = round(composite, 2),
        score_liquidity = round(s_liq,    2),
        score_spread    = round(s_spread, 2),
        score_delta     = round(s_delta,  2),
        score_moneyness = round(s_money,  2),
        score_oi_change = round(s_oi_chg, 2),
        score_vol_trend = round(s_voltrd, 2),
        score_expiry    = round(s_expiry, 2),
        score_price_beh = round(s_pbehav, 2),
        tick_age_s      = age,
        tick_open       = float(tick.get("open") or 0),
        filter_result   = hf,
    )
    return cand, hf


# ── OptionSelector class ───────────────────────────────────────────────────────

# Rejection codes that describe the expiry/universe rather than a contract; they do
# not count toward ``evaluated``.
_EXPIRY_CODES = {ReasonCode.NO_SUITABLE_EXPIRY.value, ReasonCode.NO_INSTRUMENTS.value}


def _tally(rejections: dict, code: str) -> None:
    rejections[code] = rejections.get(code, 0) + 1


def _score_row(c: ScoredCandidate) -> dict:
    return {
        "strike": c.strike, "option_type": c.option_type, "ltp": c.ltp, "score": c.score,
        "delta": c.delta_est, "oi": c.oi, "volume": c.volume, "spread_pct": c.spread_pct,
        "liquidity": c.score_liquidity, "spread": c.score_spread, "delta_score": c.score_delta,
        "moneyness": c.score_moneyness, "oi_change": c.score_oi_change,
        "vol_trend": c.score_vol_trend, "expiry": c.score_expiry, "price": c.score_price_beh,
    }


def _candidate_sort_key(candidate: ScoredCandidate) -> tuple:
    """Rank equal-score contracts by execution quality, not strike order.

    Instruments are loaded in strike order, so sorting only by composite score
    silently preferred the lowest strike whenever two candidates rounded to the
    same score.  The tie-breakers preserve the primary score while preferring
    tighter execution, stronger liquidity, and delta nearer the scalping band.
    """
    delta_distance = min(abs(abs(candidate.delta_est) - 0.50),
                         abs(abs(candidate.delta_est) - 0.65))
    return (
        candidate.score,
        candidate.score_spread,
        candidate.score_liquidity,
        candidate.score_delta,
        -delta_distance,
        candidate.oi,
        candidate.volume,
        -candidate.spread_pct,
        -candidate.strike,
    )


class OptionSelector:
    """
    Evaluates all CE/PE candidates for a given signal and selects the best.
    Logs the full scoring table so you can audit every decision.
    """

    async def select(
        self,
        db:          AsyncSession,
        direction:   Direction,
        underlying:  str,
        spot_price:  float,
        *,
        sl_pct:      float,
        target_pct:  float,
        params:      TradingParams,
        base_amount: float,
        regime_multiplier: float,
        expiry:      Optional[date] = None,
    ) -> SelectionResult:
        """
        Main entry point — returns a SelectionResult whose ``candidate`` is the
        single best option, or None (NO TRADE) with the per-ReasonCode tally of why.
        direction = BUY → CE,  SELL → PE.

        ``base_amount`` is pre-multiplier: the affordability ceiling applies the
        regime multiplier here, and sizing applies it to the deployed amount, each
        exactly once.

        Every call is tallied once in the per-day rejection counter (item 7);
        ``select_top_n`` (diagnostics only) is not.
        """
        res = await self._select(
            db, direction, underlying, spot_price, sl_pct=sl_pct, target_pct=target_pct,
            params=params, base_amount=base_amount, regime_multiplier=regime_multiplier,
            expiry=expiry)
        _REJECTION_COUNTER.record(underlying, res)
        return res

    async def _select(
        self,
        db:          AsyncSession,
        direction:   Direction,
        underlying:  str,
        spot_price:  float,
        *,
        sl_pct:      float,
        target_pct:  float,
        params:      TradingParams,
        base_amount: float,
        regime_multiplier: float,
        expiry:      Optional[date] = None,
    ) -> SelectionResult:
        candidates, rejections, expiry_choice = await self._build_candidates(
            db, direction, underlying, spot_price, sl_pct=sl_pct, target_pct=target_pct,
            params=params, base_amount=base_amount, regime_multiplier=regime_multiplier,
            expiry=expiry,
        )
        n_fallback = sum(1 for c in candidates
                         if not fills.quote_valid(c.bid, c.ask, c.tick_age_s, params))
        evaluated = len(candidates) + sum(
            n for k, n in rejections.items() if k not in _EXPIRY_CODES)

        if not candidates:
            logger.info(
                "[OptionSelector] ❌ NO TRADE — %s %s spot=%.1f | %d evaluated, all rejected: %s "
                "(expiry: %s)", underlying, direction, spot_price, evaluated, rejections,
                expiry_choice.reason,
            )
            return SelectionResult(None, rejections, evaluated, expiry_choice, n_fallback, [])

        candidates.sort(key=_candidate_sort_key, reverse=True)
        self._log_table(candidates, underlying, sum(rejections.values()))

        best = candidates[0]
        logger.info(
            "[OptionSelector] ✅ SELECTED %s %.0f%s | "
            "score=%.1f/100  ltp=%.2f  δ=%.3f  iv=%.1f%%  "
            "spread=%.2f%%  oi=%s  (%d passed / %d rejected filters)",
            underlying, best.strike, best.option_type,
            best.score, best.ltp, best.delta_est, best.iv,
            best.spread_pct, f"{best.oi:,}",
            len(candidates), sum(rejections.values()),
        )
        return SelectionResult(
            best, rejections, evaluated, expiry_choice, n_fallback,
            [_score_row(c) for c in candidates[:SCORE_TABLE_ROWS]], ranked=list(candidates))

    async def select_top_n(
        self,
        db:         AsyncSession,
        direction:  Direction,
        underlying: str,
        spot_price: float,
        *,
        n:          int,
        sl_pct:     float,
        target_pct: float,
        params:     TradingParams,
        base_amount: float,
        regime_multiplier: float,
    ) -> SelectionResult:
        """Top N scored candidates (frontend option selector widget): the same
        SelectionResult as ``select``, with ``ranked`` truncated to *n*."""
        res = await self._select(
            db, direction, underlying, spot_price, sl_pct=sl_pct, target_pct=target_pct,
            params=params, base_amount=base_amount, regime_multiplier=regime_multiplier)
        return SelectionResult(
            res.candidate, res.rejections, res.evaluated, res.expiry, res.ltp_only_fallbacks,
            res.score_table, ranked=res.ranked[:n])

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _resolve_expiry(
        self, db: AsyncSession, underlying: str, option_type: OptionType, now: datetime,
        tp: TradingParams,
    ) -> ExpiryChoice:
        """Deliberate expiry choice (design §10.4) over the instrument universe
        for the ctx trading day."""
        universe = [
            i for i in await ctx().instruments.instruments_for(db, underlying.upper())
            if i.instrument_type == InstrumentType.OPTIDX and i.option_type == option_type
            and i.expiry is not None
        ]
        return choose_expiry([i.expiry for i in universe], today=trading_day(),
                             now_t=now.time().replace(tzinfo=None), underlying=underlying, tp=tp)

    async def _build_candidates(
        self,
        db:         AsyncSession,
        direction:  Direction,
        underlying: str,
        spot_price: float,
        *,
        sl_pct:     float,
        target_pct: float,
        params:     TradingParams,
        base_amount: float,
        regime_multiplier: float,
        expiry:     Optional[date] = None,
    ) -> tuple[list[ScoredCandidate], dict, ExpiryChoice]:
        """
        Build and score all candidates. EVERY path returns the same 3-tuple
        ``(candidates, rejections, expiry_choice)``; ``rejections`` maps
        ReasonCode values to counts.
        """
        tp = params
        rejections: dict[str, int] = {}
        option_type_enum = OptionType.CE if direction == Direction.BUY else OptionType.PE
        step = NIFTY_STEP if underlying.upper() == "NIFTY" else BANKNIFTY_STEP

        # Universe through the InstrumentPort (design §10.4, §14.2): live =
        # active rows with expiry >= the ctx trading day; replay = the day's
        # traded tokens. No inline is_active / wall-clock filter here.
        universe = [
            i for i in await ctx().instruments.instruments_for(db, underlying.upper())
            if i.instrument_type == InstrumentType.OPTIDX and i.option_type == option_type_enum
        ]

        today = trading_day()
        if expiry is not None:
            choice = ExpiryChoice(expiry, (expiry - today).days, "EXPLICIT",
                                  f"explicit expiry {expiry}")
        else:
            choice = choose_expiry([i.expiry for i in universe if i.expiry], today=today,
                                   now_t=now_ist().time().replace(tzinfo=None),
                                   underlying=underlying, tp=tp)
        if choice.expiry is None:
            _tally(rejections, ReasonCode.NO_INSTRUMENTS.value
                   if choice.cause == "NO_INSTRUMENT_MASTER"
                   else ReasonCode.NO_SUITABLE_EXPIRY.value)
            return [], rejections, choice
        expiry = choice.expiry

        # Fetch strikes: ATM ± 5 strikes
        atm  = round(spot_price / step) * step
        low  = atm - 5 * step
        high = atm + 5 * step
        instruments = sorted(
            (i for i in universe
             if _expiry_date(i.expiry) == expiry and i.strike is not None and low <= i.strike <= high),
            key=lambda i: i.strike,
        )
        logger.info("OPTION_UNIVERSE underlying=%s side=%s contracts=%d expiry=%s "
                    "expiry_survivors=%d strike_survivors=%d",
                    underlying, option_type_enum.value, len(universe), expiry,
                    sum(_expiry_date(i.expiry) == expiry for i in universe), len(instruments))
        if not instruments:
            _tally(rejections, ReasonCode.NO_INSTRUMENTS.value)
            return [], rejections, choice

        # Fetch IV from option chain cache (best-effort)
        chain_iv: dict[str, float] = {}
        chain = await market().get_option_chain(underlying, str(expiry))
        if chain:
            for s in chain.get("strikes", []):
                if s.get("token") and s.get("iv"):
                    raw_iv = float(s["iv"])
                    # Normalise to percentage form (e.g. 20.0 for 20%).
                    chain_iv[s["token"]] = raw_iv * 100 if raw_iv < 1.0 else raw_iv

        # Fetch previous OI snapshot from Redis for OI-change scoring
        prev_oi_map: dict[str, int] = {}
        prev_snap_key = f"prev_oi:{underlying}:{expiry}"
        prev_snap = await market().get(prev_snap_key)
        if prev_snap and isinstance(prev_snap, dict):
            prev_oi_map = {k: int(v) for k, v in prev_snap.items()}

        # Fetch rolling average volume from Redis for volume-trend scoring
        avg_vol_key = f"avg_vol:{underlying}:{expiry}"
        avg_vol_snap = await market().get(avg_vol_key)
        avg_vol_map: dict[str, float] = {}
        if avg_vol_snap and isinstance(avg_vol_snap, dict):
            avg_vol_map = {k: float(v) for k, v in avg_vol_snap.items()}

        candidates: list[ScoredCandidate] = []
        ticks: dict[str, dict] = {}          # the ONE tick read per instrument (§10.3)

        for inst in instruments:
            tick = await market().get_tick(inst.token)
            try:
                if not isinstance(tick, dict):
                    raise ValueError("tick is not an object")
                for key in ("ltp", "best_bid", "best_ask", "oi", "volume", "iv", "open"):
                    value = tick.get(key)
                    if value is not None and not math.isfinite(float(value)):
                        raise ValueError(f"non-finite {key}")
                if float(tick.get("ltp") or 0) <= 0:
                    raise ValueError("non-positive LTP")
                for key in ("oi", "volume"):
                    if tick.get(key) is not None and float(tick[key]) != int(tick[key]):
                        raise ValueError(f"non-integer {key}")
            except (TypeError, ValueError, OverflowError) as exc:
                _tally(rejections, ReasonCode.TICK_UNAVAILABLE.value)
                logger.info("CANDIDATE_REJECT token=%s reason=%s", inst.token, exc)
                continue
            ticks[inst.token] = tick

            # Inject IV from chain snapshot if tick doesn't have it
            if not tick.get("iv") and inst.token in chain_iv:
                tick = dict(tick)
                tick["iv"] = chain_iv[inst.token]

            # affordable_max_premium: the regime multiplier is applied ONCE, here
            lot = inst.lot_size or 0
            from app.services.trading.risk_gate import effective_budget
            max_premium = effective_budget(base_amount, regime_multiplier) / lot if lot > 0 else 0.0

            cand, hf = score_candidate(
                inst        = inst,
                spot        = spot_price,
                tick        = tick,
                expiry      = expiry,
                direction   = direction,
                step        = step,
                ltp_history = [],
                prev_oi     = prev_oi_map.get(inst.token),
                avg_volume  = avg_vol_map.get(inst.token, 0.0),
                sl_pct      = sl_pct,
                target_pct  = target_pct,
                tp          = tp,
                max_premium = max_premium,
            )
            if cand is None:
                _tally(rejections, hf.filter)
                logger.info("CANDIDATE_REJECT symbol=%s lot_size=%s budget=%.2f code=%s reason=%s",
                            inst.symbol, lot, effective_budget(base_amount, regime_multiplier), hf.filter, hf.reason)
            elif cand.score < tp.min_score:
                _tally(rejections, ReasonCode.SCORE_BELOW_MIN.value)
                logger.debug("[SoftFilter] REJECTED %s %.0f%s score=%.1f < %.1f",
                             inst.symbol, inst.strike, cand.option_type, cand.score, tp.min_score)
            else:
                candidates.append(cand)

        # Save current OI snapshot for next cycle's OI-change scoring and update the
        # rolling average volume (EMA α=0.3) — from the ticks already read above.
        # CE and PE scans share these keys; retain the side not scanned this cycle.
        new_snap: dict[str, int] = dict(prev_oi_map)
        new_avg_vol: dict[str, float] = dict(avg_vol_map)
        for token, tick in ticks.items():
            new_snap[token] = int(tick.get("oi") or 0)
            prev_avg = avg_vol_map.get(token, 0.0)
            cur_volume = float(tick.get("volume") or 0)
            new_avg_vol[token] = cur_volume if prev_avg <= 0 else \
                round(0.3 * cur_volume + 0.7 * prev_avg, 1)
        if new_snap:
            await market().set(prev_snap_key, new_snap, ttl=3600)
        if new_avg_vol:
            await market().set(avg_vol_key, new_avg_vol, ttl=28800)

        return candidates, rejections, choice

    async def log_startup_summary(self, db: AsyncSession) -> None:
        """Observed expiries, nearest DTE and the premium ceiling per underlying
        (design §10.2, §10.4), so an operator sees the effective bands at boot."""
        tp = params()
        for und in ("NIFTY", "BANKNIFTY"):
            rows = [i for i in await ctx().instruments.instruments_for(db, und)
                    if i.instrument_type == InstrumentType.OPTIDX and i.expiry is not None]
            choice = choose_expiry([i.expiry for i in rows], today=trading_day(),
                                   now_t=dtime(9, 15), underlying=und, tp=tp)
            lot = next((i.lot_size for i in rows if i.lot_size), None)
            # the ceiling sizing applies: base = round_to_500(fraction × capital)
            from app.services.trading.risk_gate import round_to_500
            ceiling = (round(round_to_500(tp.fixed_fraction_pct / 100.0 * ctx().cfg.capital)
                             / lot, 2) if lot else None)
            logger.info(
                "[OptionSelector] %s: expiries=%s nearest_dte=%s bands=%s -> %s | "
                "lot=%s premium ceiling (1.0x) ₹%s, band %s",
                und, [o[0] for o in choice.observed],
                choice.observed[0][1] if choice.observed else None,
                list(tp.dte_bands.get(und, ())), choice.reason, lot, ceiling,
                tp.premium_band.get(und),
            )

    def _log_table(self, candidates: list[ScoredCandidate], underlying: str, rejected: int = 0):
        """Log the full scoring table with pass/reject summary."""
        if not candidates:
            return
        logger.info(
            "[OptionSelector] %s — %d candidates passed, %d rejected by filters",
            underlying, len(candidates), rejected,
        )
        header = (
            f"  {'Strike':>8}  {'LTP':>7}  {'Delta':>6}  "
            f"{'IV':>6}  {'OI':>9}  {'Spread':>7}  "
            f"{'Score':>9}  | {'Liq':>5}  {'Sprd':>5}  "
            f"{'Δ':>5}  {'Mny':>5}  {'OIΔ':>5}"
        )
        logger.info(header)
        logger.info("  " + "─" * 112)
        for i, c in enumerate(candidates[:SCORE_TABLE_ROWS]):
            marker = "  ← SELECTED" if i == 0 else ""
            logger.info(
                "  %8.0f  %7.2f  %6.3f  %5.1f%%  %9s  %6.2f%%  "
                "%7.1f/100  | %5.1f  %5.1f  %5.1f  %5.1f  %5.1f%s",
                c.strike, c.ltp, c.delta_est, c.iv, f"{c.oi:,}",
                c.spread_pct, c.score,
                c.score_liquidity, c.score_spread, c.score_delta,
                c.score_moneyness, c.score_oi_change,
                marker,
            )


# ── Rejection visibility (blockers-plan item 7; thresholds unchanged) ─────────

class SelectionRejectionCounter:
    """Per (trading_day, underlying) tally of select() outcomes: evaluated,
    selected and a ReasonCode Counter. Logs one structured INFO line
    ``SELECTION_REJECTIONS {json}`` on the first rejection of the day and then at
    most once per LOG_INTERVAL_S per underlying (ctx clock). Days other than the
    current one are dropped on roll. Observability only: never raises."""

    LOG_INTERVAL_S = 300

    def __init__(self) -> None:
        self._tallies: dict[tuple[date, str], dict] = {}

    def reset(self) -> None:
        self._tallies.clear()

    def get(self, day: date, underlying: str) -> Optional[dict]:
        t = self._tallies.get((day, underlying.upper()))
        if t is None:
            return None
        return {"evaluated": t["evaluated"], "selected": t["selected"],
                "by_code": dict(t["by_code"])}

    def record(self, underlying: str, result: SelectionResult) -> None:
        try:
            self._record(underlying.upper(), result)
        except Exception as exc:
            logger.debug("selection rejection counter failed: %s", exc)

    def _record(self, und: str, result: SelectionResult) -> None:
        day, now = trading_day(), now_utc()
        for key in [k for k in self._tallies if k[0] != day]:
            del self._tallies[key]
        t = self._tallies.setdefault((day, und), {
            "evaluated": 0, "selected": 0, "by_code": Counter(), "last_logged_at": None})
        t["evaluated"] += int(result.evaluated or 0)
        t["selected"] += 1 if result.candidate is not None else 0
        t["by_code"].update({k: int(n) for k, n in (result.rejections or {}).items()})
        last = t["last_logged_at"]
        if last is None:
            due = bool(t["by_code"])                       # the first rejection of the day
        else:
            due = (now - last).total_seconds() >= self.LOG_INTERVAL_S
        if not due:
            return
        t["last_logged_at"] = now
        logger.info("SELECTION_REJECTIONS %s", json.dumps({
            "run_id": ctx().run_id, "trading_day": day.isoformat(), "underlying": und,
            "evaluated": t["evaluated"], "selected": t["selected"],
            "by_code": dict(t["by_code"]),
        }, sort_keys=True))


_REJECTION_COUNTER = SelectionRejectionCounter()


def rejection_counters() -> SelectionRejectionCounter:
    return _REJECTION_COUNTER


# ── Singleton ──────────────────────────────────────────────────────────────────
option_selector = OptionSelector()
