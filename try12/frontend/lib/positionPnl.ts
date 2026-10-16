import type { Position } from "../types/trading";
import type { Tick } from "../types/market";

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function estimatePositionPnl(pos: Position, tick: Tick | undefined, now: number) {
  const qty = pos.remaining_qty ?? pos.qty;
  const realized = pos.realized_gross_pnl ?? pos.gross_pnl ?? 0;
  if (!finite(qty) || qty < 0 || !Number.isInteger(qty) || !finite(realized)) return null;
  const booked = (pos.brokerage ?? 0) + (pos.charges ?? 0);
  if (!finite(booked)) return null;
  if (qty === 0) return { gross: realized, unrealized: 0, bookedNet: realized - booked, ltp: null };
  const raw = tick?.ts;
  const ts = raw ? Date.parse(/(?:Z|[+-]\d{2}:\d{2})$/i.test(raw) ? raw : `${raw}Z`) : NaN;
  const age = (now - ts) / 1000;
  if (!Number.isFinite(age) || age < 0 || age > (pos.quote_max_age_seconds ?? 60)
      || !finite(tick?.ltp) || tick.ltp <= 0 || !finite(pos.entry) || pos.entry <= 0) return null;
  const change = pos.direction === "SELL" ? pos.entry - tick.ltp : tick.ltp - pos.entry;
  const unrealized = change * qty;
  const gross = realized + unrealized;
  return { gross, unrealized, bookedNet: gross - booked, ltp: tick.ltp };
}
