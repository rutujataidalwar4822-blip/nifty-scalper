"use client";
import { useEffect, useState } from "react";
import { estimatePositionPnl } from "@/lib/positionPnl";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { getPositions, closePosition, clearTestTrades } from "@/lib/tradingApi";
import { ApiError, errorMessage } from "@/lib/http";
import { queryKeys } from "@/lib/queryKeys";
import { useMarketStore, selectTick } from "@/lib/store";
import { useTickFeed } from "@/hooks/useTickFeed";
import { cn } from "@/lib/utils";
import { formatIst } from "@/lib/time";
import { Panel } from "@/components/ui/Panel";
import { DataSurface } from "@/components/ui/DataSurface";
import { TableSkeleton } from "@/components/ui/Skeleton";
import { Badge, type BadgeIntent } from "@/components/ui/Badge";
import { PriceText, DeltaText } from "@/components/ui/PriceText";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { STRATEGY_META } from "./strategyMeta";
import type { Position } from "@/types/trading";

const EXIT_INTENT: Record<string, BadgeIntent> = {
  TARGET: "positive", SL: "negative", EOD: "neutral", MANUAL: "info", TIMEOUT: "warning",
};
const STRATEGY_LABEL: Record<string, string> = Object.fromEntries(STRATEGY_META.map((s) => [s.key, s.label]));
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const failText = (e: unknown) => (e instanceof ApiError && e.detail ? e.detail : errorMessage(e));

/** Open-position P&L from the live tick of the traded contract (field selector per row). */
function LivePnl({ pos }: { pos: Position }) {
  useTickFeed(pos.token || null);
  const tick = useMarketStore(selectTick(pos.token));
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1_000);
    return () => clearInterval(timer);
  }, []);
  const estimate = estimatePositionPnl(pos, tick, now);
  if (!estimate) {
    return <span className="text-table text-text-2">awaiting fresh tick</span>;
  }
  return (
    <span className="inline-flex flex-col items-end">
      <DeltaText value={estimate.gross} prefix="₹" decimals={0} />
      <span className="font-mono text-micro text-text-2">
        {estimate.ltp === null ? "No remaining quantity" : `LTP ${estimate.ltp.toFixed(2)} · gross est.`}
      </span>
      <span className="font-mono text-micro text-text-2" title="Excludes estimated charges for the remaining exit">
        After booked costs: ₹{estimate.bookedNet.toFixed(2)}
      </span>
    </span>
  );
}

function PositionRow({ p, closed, onClose, closing }: { p: Position; closed: boolean; onClose: (id: number) => void; closing: boolean }) {
  return (
    <tr data-position-id={p.id} className="odd:bg-surface-2">
      <td className="font-medium text-text-1">{STRATEGY_LABEL[p.strategy] ?? p.strategy?.replace(/_/g, " ")}</td>
      <td>
        <div className="font-mono text-text-1">{p.symbol}</div>
        {num(p.capital_used) !== null && <div className="font-mono text-micro text-text-2">₹{Math.round(p.capital_used as number).toLocaleString("en-IN")}</div>}
      </td>
      <td className="num text-text-1"><PriceText value={num(p.entry)} /></td>
      <td className="num text-red"><PriceText value={num(p.sl)} /></td>
      <td className="num text-yellow"><PriceText value={num(p.trailing_sl)} /></td>
      <td className="num text-green"><PriceText value={num(p.target)} /></td>
      {closed && <td className="num"><PriceText value={num(p.exit)} /></td>}
      {closed && (
        <td>
          <Badge intent={EXIT_INTENT[p.exit_reason ?? ""] ?? "neutral"} status={p.exit_reason ?? undefined}>{p.exit_reason || "—"}</Badge>
        </td>
      )}
      <td className="num">
        {closed ? <DeltaText value={num(p.net_pnl)} prefix="₹" decimals={0} /> : <LivePnl pos={p} />}
      </td>
      <td className="num font-mono">{p.lots}</td>
      <td className="whitespace-nowrap font-mono text-text-2">
        {formatIst(closed ? p.exit_time ?? p.entry_time : p.entry_time, { seconds: false, withDate: closed })}
      </td>
      {!closed && (
        <td>
          <button
            type="button"
            onClick={() => onClose(p.id)}
            disabled={closing}
            aria-label={`Close position ${p.symbol}`}
            className="rounded border border-red/40 bg-red/10 px-2.5 py-1 text-micro font-semibold text-red hover:bg-red/20 disabled:opacity-50"
          >
            Close
          </button>
        </td>
      )}
    </tr>
  );
}

export function PositionsTable() {
  const qc = useQueryClient();
  const [closed, setClosed] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  const params = { is_open: !closed, limit: 100 };
  const query = useQuery({
    queryKey: queryKeys.positions(params),
    queryFn: () => getPositions(params),
    refetchInterval: 5_000,
  });

  const invalidateDesk = () => {
    for (const key of [
      queryKeys.positionsAll, queryKeys.capital, queryKeys.stats, queryKeys.pnlToday,
      queryKeys.pnlHistoryAll, queryKeys.diagnostics, queryKeys.config,
    ]) void qc.invalidateQueries({ queryKey: key });
  };

  const close = useMutation({
    mutationFn: closePosition,
    onSuccess: () => { invalidateDesk(); toast.success("Position closed"); },
    onError: (e: unknown) => { console.error("[positions] close failed", e); toast.error(`Close failed: ${failText(e)}`); },
  });

  const clearTrades = useMutation({
    mutationFn: clearTestTrades,
    onSuccess: () => {
      invalidateDesk();
      void qc.invalidateQueries({ queryKey: queryKeys.signalsAll });
      setConfirmClear(false);
      toast.success("All test trades cleared");
    },
    onError: (e: unknown) => { console.error("[positions] clear failed", e); toast.error(`Clear failed: ${failText(e)}`); },
  });

  const rows = Array.isArray(query.data) ? query.data : [];
  const totalNet = rows.reduce((s, p) => s + (num(p.net_pnl) ?? 0), 0);

  return (
    <Panel
      title="Positions"
      meta={query.data !== undefined && <Badge intent="neutral">{rows.length}</Badge>}
      actions={
        <>
          {closed && query.data !== undefined && rows.length > 0 && (
            <span aria-live="polite" className="text-table text-text-2">
              net <DeltaText value={totalNet} prefix="₹" decimals={0} />
            </span>
          )}
          <div role="group" aria-label="Position filter" className="inline-flex overflow-hidden rounded-md border border-border">
            {([false, true] as const).map((c) => (
              <button
                key={String(c)}
                type="button"
                aria-pressed={closed === c}
                onClick={() => setClosed(c)}
                className={cn(
                  "px-2.5 py-1 text-table font-semibold",
                  closed === c ? "bg-accent/30 text-text-1" : "bg-surface-2 text-text-2 hover:text-text-1",
                )}
              >
                {c ? "Closed" : "Open"}
              </button>
            ))}
          </div>
          {closed && rows.length > 0 && (
            <button
              type="button"
              onClick={() => setConfirmClear(true)}
              disabled={clearTrades.isPending}
              className="rounded border border-red/40 bg-red/10 px-2 py-1 text-micro font-semibold text-red hover:bg-red/20 disabled:opacity-50"
            >
              Clear test trades
            </button>
          )}
        </>
      }
      bodyClassName="px-4 pb-4 pt-3"
    >
      <DataSurface
        query={query}
        isEmpty={(d) => !Array.isArray(d) || d.length === 0}
        emptyTitle={closed ? "No closed positions" : "No open positions"}
        emptyHint={closed ? "Closed paper/live positions appear here with their exit reason and net P&L." : "Positions open when a strategy signal passes every risk gate."}
        skeleton={<TableSkeleton rows={3} cols={10} />}
        hideFreshness
      >
        {(data) => (
          <div className="-mx-4 -mb-4 max-h-[320px] overflow-auto">
            <table className="data-table">
              <caption className="sr-only">{closed ? "Closed positions, most recent first" : "Open positions with live estimated P&L"}</caption>
              <thead>
                <tr>
                  <th scope="col">Strategy</th>
                  <th scope="col">Contract</th>
                  <th scope="col" className="num">Entry</th>
                  <th scope="col" className="num">SL</th>
                  <th scope="col" className="num">Trail SL</th>
                  <th scope="col" className="num">Target</th>
                  {closed && <th scope="col" className="num">Exit</th>}
                  {closed && <th scope="col">Reason</th>}
                  <th scope="col" className="num">P&amp;L</th>
                  <th scope="col" className="num">Lots</th>
                  <th scope="col">{closed ? "Exited" : "Entered"}</th>
                  {!closed && <th scope="col"><span className="sr-only">Actions</span></th>}
                </tr>
              </thead>
              <tbody>
                {data.map((p) => (
                  <PositionRow key={p.id} p={p} closed={closed} onClose={(id) => close.mutate(id)} closing={close.isPending} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </DataSurface>

      <ConfirmDialog
        open={confirmClear}
        title="Clear all test trades?"
        confirmLabel="Clear everything"
        danger
        pending={clearTrades.isPending}
        onConfirm={() => clearTrades.mutate()}
        onCancel={() => setConfirmClear(false)}
      >
        This deletes every position, every recorded signal, the daily P&amp;L summaries and paper orders/trades,
        and resets the daily trade counter and used capital. It cannot be undone.
      </ConfirmDialog>
    </Panel>
  );
}
