import { test } from 'node:test';
import assert from 'node:assert/strict';
import { estimatePositionPnl } from './positionPnl.ts';

const now = Date.parse('2026-10-16T06:00:00Z');
const pos = { direction: 'BUY', entry: 100, qty: 65, remaining_qty: 65,
  gross_pnl: 650, brokerage: 60, charges: 15 };
const tick = { ltp: 110, ts: '2026-10-16T06:00:00Z' };

test('adds partial realized gross once on the remainder and deducts booked costs once', () => {
  assert.deepEqual(estimatePositionPnl(pos, tick, now),
    { gross: 1300, unrealized: 650, bookedNet: 1225, ltp: 110 });
});
test('losing remainder offsets realized profit', () => {
  assert.equal(estimatePositionPnl(pos, { ...tick, ltp: 80 }, now).gross, -650);
});
test('zero remainder needs no tick and never falls back to original quantity', () => {
  assert.equal(estimatePositionPnl({ ...pos, remaining_qty: 0 }, undefined, now).gross, 650);
});
test('stale, future, missing and invalid marks are unavailable', () => {
  for (const value of [undefined, { ...tick, ltp: NaN }, { ...tick, ltp: Infinity },
    { ...tick, ltp: -1 }, { ...tick, ts: undefined }, { ...tick, ts: 'invalid' },
    { ...tick, ts: '2026-10-16T05:58:59Z' }, { ...tick, ts: '2026-10-16T06:00:01Z' }]) {
    assert.equal(estimatePositionPnl(pos, value, now), null);
  }
});
test('timestamp boundary and legacy naive UTC are explicit', () => {
  assert.notEqual(estimatePositionPnl(pos, { ...tick, ts: '2026-10-16T05:59:00' }, now), null);
  assert.equal(estimatePositionPnl(pos, tick, now + 60001), null);
});
