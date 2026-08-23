// Deterministic maths behind the Projection page: the closed-form future value
// used by the goal seeker, and the Monte Carlo itself. Kept out of the component
// so it can be unit-tested — a TWRR unit bug shipped here once already.

// mulberry32 — small deterministic PRNG.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box–Muller standard normal over a uniform source.
export function makeRandn(rand) {
  return () => {
    const u = 1 - rand();
    const v = 1 - rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

// Fraction of a sorted sample at or above `target` (binary search).
export function fractionAtOrAbove(sortedValues, target) {
  let lo = 0;
  let hi = sortedValues.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedValues[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return 1 - lo / sortedValues.length;
}

// Percent-string input → decimal, tolerating partial typing.
export function parsePctInput(raw, fallback) {
  if (raw.trim() === '') return fallback;
  const v = Number(raw);
  return Number.isFinite(v) ? v / 100 : fallback;
}

// Nominal rate equivalent to a real rate under `inflation` (Fisher).
export function toNominal(real, inflation) {
  return (1 + real) * (1 + inflation) - 1;
}

// Real rate equivalent to a nominal rate under `inflation` (Fisher, inverted).
export function toReal(nominal, inflation) {
  return (1 + nominal) / (1 + inflation) - 1;
}

/**
 * Restate a typed percent string when the display mode flips, preserving the
 * forecast the way the benchmark does.
 *
 * @returns {string|null} the restated percent, or null when `raw` isn't a number
 *   yet (mid-typing) and the box should be left alone.
 */
export function convertRateForMode(raw, toMode, inflation) {
  if (raw.trim() === '') return null;
  const typed = Number(raw);
  if (!Number.isFinite(typed)) return null;
  const converted = toMode === 'nominal'
    ? toNominal(typed / 100, inflation)
    : toReal(typed / 100, inflation);
  return (converted * 100).toFixed(1);
}

/**
 * Future value of a starting pot plus annual contributions.
 *
 * Contributions are paid at the start of each year (so they compound for the
 * full year) and run only for the first S of T years, growing at g.
 */
export function futureValue(V0, contribution, r, g, T, S = T) {
  const rT = (1 + r) ** T;
  const years = Math.min(S, T);
  const q = (1 + g) / (1 + r);
  const annuity = Math.abs(r - g) < 1e-9 ? years : (1 - q ** years) / (1 - q);
  return rT * (V0 + contribution * annuity);
}

// Bisection on r such that FV == target. Returns null if target unreachable in [-50%, +50%].
export function solveRequiredCagr(V0, contribution, g, T, target, S = T) {
  let lo = -0.5;
  let hi = 0.5;
  const fvLo = futureValue(V0, contribution, lo, g, T, S);
  const fvHi = futureValue(V0, contribution, hi, g, T, S);
  if (target < fvLo || target > fvHi) return null;
  for (let i = 0; i < 80; i += 1) {
    const mid = (lo + hi) / 2;
    const fv = futureValue(V0, contribution, mid, g, T, S);
    if (fv < target) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// Required annual contribution to hit `target` given an assumed CAGR `r`.
// Returns null when contributions can't move the outcome (S = 0).
export function solveRequiredContribution(V0, r, g, T, target, S = T) {
  const remaining = target - futureValue(V0, 0, r, g, T, S);
  if (remaining <= 0) return 0;
  const multiplier = futureValue(0, 1, r, g, T, S);
  if (multiplier <= 0) return null;
  return remaining / multiplier;
}

/**
 * Monte Carlo under geometric Brownian motion, annual steps.
 *
 * Contributions are paid at the start of years 1..contributionYears; withdrawals
 * come off at the start of each year after withdrawalStartYear and are floored at
 * zero, so a depleted pot stays depleted. The drawdown phase compounds at
 * drawdownReturn with drawdownVolatility rather than expectedReturn/volatility.
 *
 * @returns {{years: number[], percentiles: {p10: number[], p25: number[],
 *   p50: number[], p75: number[], p90: number[]}, deterministic: number[],
 *   invested: number[], ruin: number[], sortedPerYear: Float64Array[]}}
 *   All series are length T+1. `deterministic` is the zero-volatility CAGR path,
 *   `invested` is capital in with no returns, `ruin` the share of depleted paths,
 *   and `sortedPerYear` the ascending samples behind the percentiles.
 */
export function runSimulation({
  startingValue,
  contribution,
  contributionGrowth,
  contributionYears = Infinity,
  withdrawalAnnual = 0,
  withdrawalStartYear = null,
  withdrawalGrowth = 0,
  expectedReturn,
  drawdownReturn = expectedReturn,
  volatility,
  drawdownVolatility = volatility,
  horizonYears,
  paths,
  seed,
}) {
  const T = horizonYears;
  const N = paths;
  // Inputs are geometric (CAGR) rates, so the log-drift is ln(1+r) with no -σ²/2
  // term; the median path then compounds at exactly the input rate.
  const muAccum = Math.log(1 + expectedReturn);
  const muDrawdown = Math.log(1 + drawdownReturn);
  const randn = makeRandn(mulberry32(seed));

  // Simulate all paths year-by-year. Store results per year as a flat typed array
  // so sort-based percentile extraction stays fast.
  const perYear = [];
  for (let t = 0; t <= T; t += 1) perYear.push(new Float64Array(N));

  const withdrawing = (t) => withdrawalStartYear != null && t > withdrawalStartYear;

  for (let i = 0; i < N; i += 1) {
    let value = startingValue;
    let contrib = contribution;
    let wdraw = withdrawalAnnual;
    perYear[0][i] = value;
    for (let t = 1; t <= T; t += 1) {
      const z = randn();
      const inDrawdown = withdrawing(t);
      const mu = inDrawdown ? muDrawdown : muAccum;
      const r = Math.exp(mu + (inDrawdown ? drawdownVolatility : volatility) * z) - 1;
      if (t <= contributionYears) value += contrib;
      if (withdrawing(t)) value = Math.max(0, value - wdraw);
      value *= 1 + r;
      perYear[t][i] = value;
      contrib *= 1 + contributionGrowth;
      // Indexed every year (not just while withdrawing) so the first withdrawal
      // is already in that year's money under nominal display.
      wdraw *= 1 + withdrawalGrowth;
    }
  }

  // Sort each year in place (used for percentiles, goal probability and ruin).
  const sortedPerYear = perYear.map((arr) => arr.sort());
  const percentiles = { p10: [], p25: [], p50: [], p75: [], p90: [] };
  const ruin = [];
  for (let t = 0; t <= T; t += 1) {
    const sorted = sortedPerYear[t];
    percentiles.p10.push(sorted[Math.floor(0.10 * N)]);
    percentiles.p25.push(sorted[Math.floor(0.25 * N)]);
    percentiles.p50.push(sorted[Math.floor(0.50 * N)]);
    percentiles.p75.push(sorted[Math.floor(0.75 * N)]);
    percentiles.p90.push(sorted[Math.floor(0.90 * N)]);
    ruin.push(1 - fractionAtOrAbove(sorted, 1));
  }

  // Zero-volatility path at the expected CAGR, and a "total invested" baseline
  // showing capital in (starting value + cumulative contributions, no returns).
  const deterministic = [];
  const invested = [];
  let detValue = startingValue;
  let investedTotal = startingValue;
  let detContrib = contribution;
  let detWdraw = withdrawalAnnual;
  deterministic.push(detValue);
  invested.push(investedTotal);
  for (let t = 1; t <= T; t += 1) {
    if (t <= contributionYears) {
      detValue += detContrib;
      investedTotal += detContrib;
    }
    if (withdrawing(t)) detValue = Math.max(0, detValue - detWdraw);
    detValue *= 1 + (withdrawing(t) ? drawdownReturn : expectedReturn);
    deterministic.push(detValue);
    invested.push(investedTotal);
    detContrib *= 1 + contributionGrowth;
    detWdraw *= 1 + withdrawalGrowth;
  }

  const years = Array.from({ length: T + 1 }, (_, i) => i);
  return { years, percentiles, deterministic, invested, ruin, sortedPerYear };
}
