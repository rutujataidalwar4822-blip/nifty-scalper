export interface Signal {
  id:           number;
  strategy:     string;
  underlying:   string;
  direction:    "BUY" | "SELL";
  status:       string;
  spot:         number;
  vwap?:        number;
  strike?:      number;
  symbol?:      string;
  /** Option token the signal suggested (null for SKIPPED rows). */
  suggested_token?: string | null;
  entry_ref?:   number;
  sl_ref?:      number;
  target_ref?:  number;
  rr?:          number;
  mode:         string;
  time:         string;
  indicators?:  Record<string, number>;
  skip_reason?: string;
}

export interface Position {
  id:           number;
  strategy:     string;
  /** Instrument token of the traded contract (chart overlay / live P&L). */
  token:        string;
  symbol:       string;
  underlying:   string;
  strike?:      number;
  option_type?: string;
  expiry?:      string;
  direction:    string;
  lots:         number;
  lot_size?:    number;
  /** Open remainder; on legacy closed rows this is the final exit leg. */
  qty:          number;
  original_qty?: number;
  realized_gross_pnl?: number;
  brokerage?: number;
  planned_risk?: number | null;
  realized_loss?: number | null;
  quote_max_age_seconds?: number;
  entry:        number;
  sl?:          number;
  target?:      number;
  /** Stop at entry; never moved by trailing (backend `initial_stop_loss`). */
  initial_sl?:  number | null;
  trailing_sl?: number;
  exit?:        number;
  exit_reason?: string;
  gross_pnl?:   number;
  /** Statutory charges deducted from gross (backend `statutory_charges`). */
  charges?:     number | null;
  net_pnl?:     number;
  capital_used?: number;
  is_open:      boolean;
  mode:         string;
  partial_closed?:     boolean | null;
  partial_exit_price?: number | null;
  partial_exit_qty?:   number | null;
  partial_pnl?:        number | null;
  remaining_qty?:      number | null;
  entry_time:   string;
  exit_time?:   string;
}

export interface DailyStats {
  date:            string;
  mode:            string;
  total_pnl:       number;
  total_trades:    number;
  win_rate:        number;
  open_positions:  number;
  loss_limit:      number;
  capital:         number;
  trade_amount:    number;
  used_capital:    number;
  available:       number;
  loss_pct:        number;
  circuit_breaker_triggered: boolean;
  circuit_reason:  string;
  strategy_breakdown: StrategyBreakdown[];
}

export interface StrategyBreakdown {
  strategy: string;
  trades:   number;
  wins:     number;
  net_pnl:  number;
  win_rate: number;
}

export interface PnLHistory {
  date:            string;
  net_pnl:         number;
  trades:          number;
  wins:            number;
  cumulative_pnl:  number;
}

export interface OptionCandidate {
  rank:        number;
  token:       string;
  symbol:      string;
  strike:      number;
  option_type: string;
  expiry:      string;
  ltp:         number;
  bid:         number;
  ask:         number;
  spread_pct:  number;
  oi:          number;
  volume:      number;
  delta:       number;
  iv:          number;
  score:       number;
  lot_size:    number;
}

export interface TradingConfig {
  mode:                string;
  max_daily_loss:      number;
  max_daily_loss_pct:  number;
  max_open_positions:  number;
  max_lots_per_trade:  number;
  capital:             number;
  trade_amount:        number;
  used_capital:        number;
  available_capital:   number;
  slippage_pct:        number;
  square_off_time:     string;
  strategy_flags:      Record<string, boolean>;
  live_mode_confirmed: boolean;
  study_phase:         boolean;
  max_trades_per_day:  number;
  daily_trades_count:  number;
  /** Persisted kill-switch halt (GET /trading/kill-switch for the reason). */
  trading_halted?:     boolean;
}

export interface CapitalSummary {
  capital:            number;
  used_capital:       number;
  available_capital:  number;
  trade_amount:       number;
  realised_pnl:       number;
  net_capital:        number;
  utilisation_pct:    number;
  max_daily_loss:     number;
  max_daily_loss_pct: number;
  daily_trades:       number;
  max_trades_per_day: number;
  open_positions:     number;
  max_open_positions: number;
  circuit_breaker:    boolean;
  circuit_reason:     string;
  mode:               string;
}

export interface TradePreviewRow {
  ltp:            number;
  lot_size:       number;
  lots:           number;
  capital_used:   number;
  capital_needed: number;
}
