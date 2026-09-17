import { holderFlow } from './form13f';

test('BKNG flow uses split-adjusted shares, not the $4.4bn apparent increase', () => {
  expect(holderFlow({ shares: 25536950, shares_prev: 997498, shares_prev_adjusted: 24937450, value: 4551705968 }))
    .toBeCloseTo(106854880);
});

test('legacy responses fall back, but explicitly unknown adjustments do not', () => {
  expect(holderFlow({ shares: 100, shares_prev: 90, value: 1000 })).toBe(100);
  expect(holderFlow({ shares: 100, shares_prev: 90, shares_prev_adjusted: null, value: 1000 })).toBeNull();
  expect(holderFlow({ shares: 100, shares_prev: 90, shares_prev_adjusted: 0, value: 1000 })).toBe(1000);
});

test('missing shares or current price cannot produce a flow estimate', () => {
  expect(holderFlow({ shares: 0, shares_prev: 100, value: 0 })).toBeNull();
  expect(holderFlow({ shares: 100, shares_prev: null, value: 1000 })).toBeNull();
  expect(holderFlow({ shares: 100, shares_prev: 100, value: NaN })).toBeNull();
});

test.each([100, 2500, null])('closed flow uses prior value with adjusted shares %s', (adjusted) => {
  expect(holderFlow({ shares: 0, value: 0, shares_prev: 100, shares_prev_adjusted: adjusted, value_prev: 10000 }))
    .toBe(-10000);
});
