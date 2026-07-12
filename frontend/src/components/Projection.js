import React, { useState, useEffect, useMemo } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { portfolioAPI } from '../services/api';
import { useHideAmounts, MASK } from '../context/HideAmountsContext';
import './Projection.css';

// Defaults used when the API can't provide a value.
const DEFAULT_CONTRIBUTION = 20000; // Full UK ISA allowance
const DEFAULT_INFLATION = 0.025; // 2.5% long-run UK CPI
const DEFAULT_REAL_RETURN = 0.065; // MSCI World real return, matches backend benchmark
const DEFAULT_VOLATILITY = 0.16;
const DEFAULT_HORIZON_YEARS = 30;
const DEFAULT_MONTE_CARLO_PATHS = 10000;
const ISA_ANNUAL_ALLOWANCE = 20000;
const DEFAULT_SWR = 0.04; // 4% rule
const DEFAULT_DRAWDOWN_YEARS = 30;
const MAX_GOAL_YEARS = 50;
// Hard cap on simulated years (accumulation + drawdown); also the horizon slider max.
const MAX_SIM_YEARS = 60;
// Fixed seed keeps percentiles stable across re-renders, so tweaking one
// assumption doesn't jitter every number on the page.
const RNG_SEED = 0x5eed;
// CPI observations older than this get a staleness warning in the sidebar
// (monthly data normally lags ~7 weeks).
const CPI_STALE_DAYS = 120;
// Traffic-light thresholds for the goal-seeker probability.
const GOAL_PROB_GOOD = 0.7;
const GOAL_PROB_POOR = 0.4;

// mulberry32 — small deterministic PRNG.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box–Muller standard normal over a uniform source.
function makeRandn(rand) {
  return () => {
    const u = 1 - rand();
    const v = 1 - rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

// Fraction of a sorted sample at or above `target` (binary search).
function fractionAtOrAbove(sortedValues, target) {
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
function parsePctInput(raw, fallback) {
  if (raw.trim() === '') return fallback;
  const v = Number(raw);
  return Number.isFinite(v) ? v / 100 : fallback;
}

// Future value of starting pot plus annual contributions (paid at start of year,
// compounded for the full year), where contributions run only for the first S
// years. Supports a growing annuity (g = contribution growth).
function futureValue(V0, contribution, r, g, T, S = T) {
  const rT = (1 + r) ** T;
  const years = Math.min(S, T);
  const q = (1 + g) / (1 + r);
  const annuity = Math.abs(r - g) < 1e-9 ? years : (1 - q ** years) / (1 - q);
  return rT * (V0 + contribution * annuity);
}

// Bisection on r such that FV == target. Returns null if target unreachable in [-50%, +50%].
function solveRequiredCagr(V0, contribution, g, T, target, S = T) {
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
function solveRequiredContribution(V0, r, g, T, target, S = T) {
  const remaining = target - futureValue(V0, 0, r, g, T, S);
  if (remaining <= 0) return 0;
  const multiplier = futureValue(0, 1, r, g, T, S);
  if (multiplier <= 0) return null;
  return remaining / multiplier;
}

// Monte Carlo simulation under geometric Brownian motion, annual steps.
// Contributions are paid at the start of years 1..contributionYears; withdrawals
// (if any) come off at the start of each year after withdrawalStartYear, floored
// at zero — a depleted pot stays depleted. The drawdown phase compounds at
// drawdownReturn (alpha isn't assumed to persist into retirement).
// Returns { years: number[T+1],
//           percentiles: { p10, p25, p50, p75, p90: number[T+1] },
//           deterministic: number[T+1],  // expected path (no volatility)
//           invested:      number[T+1],  // cumulative capital in, no returns
//           sortedPerYear: Float64Array[T+1] } // sorted samples, for P(≥ target)
function runSimulation({
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
  horizonYears,
  paths,
}) {
  const T = horizonYears;
  const N = paths;
  // Returns are geometric (CAGR) inputs — TWRR and the long-run benchmark are
  // compound rates — so the log-drift is ln(1+r) with no -σ²/2 adjustment;
  // the median simulated path then compounds at exactly the input rate.
  const muAccum = Math.log(1 + expectedReturn);
  const muDrawdown = Math.log(1 + drawdownReturn);
  const sigma = volatility;
  const randn = makeRandn(mulberry32(RNG_SEED));

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
      const r = Math.exp((withdrawing(t) ? muDrawdown : muAccum) + sigma * z) - 1;
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

  // Sort each year in place (used for both percentiles and goal-probability lookups).
  const sortedPerYear = perYear.map((arr) => arr.sort());
  const percentiles = { p10: [], p25: [], p50: [], p75: [], p90: [] };
  for (let t = 0; t <= T; t += 1) {
    const sorted = sortedPerYear[t];
    percentiles.p10.push(sorted[Math.floor(0.10 * N)]);
    percentiles.p25.push(sorted[Math.floor(0.25 * N)]);
    percentiles.p50.push(sorted[Math.floor(0.50 * N)]);
    percentiles.p75.push(sorted[Math.floor(0.75 * N)]);
    percentiles.p90.push(sorted[Math.floor(0.90 * N)]);
  }

  // Deterministic line at the expected CAGR, and a "total invested" baseline
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
  return { years, percentiles, deterministic, invested, sortedPerYear };
}

// Full-precision variant for figures where "£3k" is too coarse (e.g. a monthly
// contribution the user would actually set up as a standing order).
const formatGbpFull = (value) => {
  if (value == null || !Number.isFinite(value)) return '—';
  return `£${Math.round(value).toLocaleString()}`;
};

const formatGbp = (value) => {
  if (value == null || !Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1e8) return `£${(value / 1e6).toFixed(0)}M`;
  if (abs >= 1e7) return `£${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e6) return `£${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `£${(value / 1e3).toFixed(0)}k`;
  return formatGbpFull(value);
};

const formatPct = (value, digits = 1) => {
  if (value == null || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
};

const probabilityClass = (p) => {
  if (p == null) return 'goal-value';
  if (p >= GOAL_PROB_GOOD) return 'goal-value prob-good';
  if (p >= GOAL_PROB_POOR) return 'goal-value prob-mid';
  return 'goal-value prob-poor';
};

// Custom tooltip that reads absolute percentile values from the original datum
// (Recharts' default passes stacked-area deltas, which are misleading here).
const FanTooltip = ({ active, payload, label, mask }) => {
  if (!active || !payload || payload.length === 0) return null;
  const d = payload[0].payload;
  const rows = [
    { key: 'det', label: 'Expected path', value: d.det, color: '#ef4444' },
    { key: 'p90', label: 'P90 (optimistic)', value: d.p90, color: '#93c5fd' },
    { key: 'p75', label: 'P75', value: d.p75, color: '#60a5fa' },
    { key: 'p50', label: 'Median', value: d.p50, color: '#1d4ed8', strong: true },
    { key: 'p25', label: 'P25', value: d.p25, color: '#60a5fa' },
    { key: 'p10', label: 'P10 (pessimistic)', value: d.p10, color: '#93c5fd' },
    { key: 'invested', label: 'Total invested', value: d.invested, color: '#6b7280' },
  ];
  return (
    <div className="fan-tooltip">
      <div className="fan-tooltip-header">Year {label}</div>
      {rows.map((r) => (
        <div key={r.key} className={r.strong ? 'fan-tooltip-row strong' : 'fan-tooltip-row'}>
          <span className="fan-tooltip-swatch" style={{ background: r.color }} />
          <span className="fan-tooltip-label">{r.label}</span>
          <span className="fan-tooltip-value">{mask(r.value)}</span>
        </div>
      ))}
    </div>
  );
};

const Projection = () => {
  const { hideAmounts } = useHideAmounts();
  const [inputs, setInputs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // User-controlled assumptions.
  const [displayMode, setDisplayMode] = useState('real'); // 'real' | 'nominal'
  const [returnSource, setReturnSource] = useState('blend'); // 'twrr' | 'benchmark' | 'blend' | 'custom'
  // Percent inputs are kept as raw strings so reformatting doesn't fight typing.
  const [customReturnPct, setCustomReturnPct] = useState('7.0');
  const [customVolPct, setCustomVolPct] = useState('16.0');
  const [contribution, setContribution] = useState(DEFAULT_CONTRIBUTION);
  const [contributionGrowsWithInflation, setContributionGrowsWithInflation] = useState(true);
  const [horizonYears, setHorizonYears] = useState(DEFAULT_HORIZON_YEARS);
  const [startingValueInput, setStartingValueInput] = useState(null); // raw string; null = use API value
  const [goalMode, setGoalMode] = useState('amount'); // 'amount' | 'income'
  const [targetAmount, setTargetAmount] = useState(500000);
  const [targetIncome, setTargetIncome] = useState(2000); // desired monthly income
  const [swrPct, setSwrPct] = useState('4.0'); // safe withdrawal rate, %
  const [targetYear, setTargetYear] = useState(new Date().getFullYear() + 15);
  const [drawdownYears, setDrawdownYears] = useState(DEFAULT_DRAWDOWN_YEARS);

  useEffect(() => {
    let cancelled = false;
    portfolioAPI.getProjectionInputs()
      .then((data) => {
        if (cancelled) return;
        setInputs(data);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || 'Failed to load projection inputs');
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  // Derived: inflation rate (decimal). Prefer 10y avg, fall back to 12m, then default.
  const inflation = useMemo(() => {
    if (!inputs) return DEFAULT_INFLATION;
    return inputs.inflation?.uk_cpi_10y_avg
        ?? inputs.inflation?.uk_cpi_12m
        ?? DEFAULT_INFLATION;
  }, [inputs]);

  // Benchmark return under the current display mode (real vs nominal).
  const benchReturn = useMemo(() => {
    if (!inputs) return DEFAULT_REAL_RETURN;
    return displayMode === 'real'
      ? inputs.benchmark.real_return
      : inputs.benchmark.nominal_return;
  }, [inputs, displayMode]);

  const customReturn = useMemo(
    () => parsePctInput(customReturnPct, DEFAULT_REAL_RETURN),
    [customReturnPct],
  );
  const customVolatility = useMemo(
    () => parsePctInput(customVolPct, DEFAULT_VOLATILITY),
    [customVolPct],
  );

  // Effective expected return based on returnSource + displayMode.
  const expectedReturn = useMemo(() => {
    if (!inputs) return DEFAULT_REAL_RETURN;

    // User's TWRR is nominal (PortfolioDaily TWRR is a nominal return). Convert to real
    // by deflating if we're in real mode.
    const rawTwrr = inputs.portfolio?.twrr;
    const userReturn = rawTwrr == null
      ? null
      : (displayMode === 'real' ? (1 + rawTwrr) / (1 + inflation) - 1 : rawTwrr);

    if (returnSource === 'custom') return customReturn;
    if (returnSource === 'benchmark') return benchReturn;
    if (returnSource === 'twrr') return userReturn ?? benchReturn;
    // 'blend': 50/50 user TWRR + benchmark. Falls back to benchmark if TWRR missing.
    return userReturn == null ? benchReturn : 0.5 * userReturn + 0.5 * benchReturn;
  }, [inputs, returnSource, customReturn, displayMode, inflation, benchReturn]);

  // Volatility: always benchmark long-run σ (user's realized σ is too noisy for a
  // ~1 year track record). Exposed as custom for power users.
  const volatility = useMemo(() => {
    if (!inputs) return customVolatility;
    return returnSource === 'custom' ? customVolatility : inputs.benchmark.volatility;
  }, [inputs, returnSource, customVolatility]);

  // Contribution growth. When the "keep real purchasing power" toggle is on the
  // contribution is held flat in real terms (grows with inflation in nominal mode,
  // zero growth in real mode). When off it stays nominal-constant, which means
  // real contributions decay by inflation.
  const contributionGrowth = useMemo(() => {
    if (displayMode === 'real') {
      return contributionGrowsWithInflation ? 0 : -inflation;
    }
    return contributionGrowsWithInflation ? inflation : 0;
  }, [displayMode, contributionGrowsWithInflation, inflation]);

  // Empty or invalid override falls back to the live API value so the chart
  // never blanks out mid-edit.
  const startingValue = useMemo(() => {
    const apiValue = inputs?.portfolio?.starting_value ?? 0;
    if (startingValueInput == null || startingValueInput.trim() === '') return apiValue;
    const parsed = Number(startingValueInput);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : apiValue;
  }, [inputs, startingValueInput]);

  const swr = useMemo(() => parsePctInput(swrPct, DEFAULT_SWR), [swrPct]);

  // Target pot: direct amount, or the pot that sustains the desired monthly
  // income at the chosen withdrawal rate (4% rule by default).
  const effectiveTarget = useMemo(() => {
    if (goalMode !== 'income') return targetAmount;
    return swr > 0 ? (targetIncome * 12) / swr : null;
  }, [goalMode, targetAmount, targetIncome, swr]);

  const currentYear = new Date().getFullYear();
  const goalYears = targetYear - currentYear;

  const drawdownActive = goalMode === 'income' && goalYears > 0;
  // Contributions run for the whole horizon, except in income mode where they
  // stop at retirement (the goal year).
  const contributionYears = drawdownActive ? goalYears : Infinity;
  // Withdrawals keep purchasing power regardless of the contribution checkbox.
  const withdrawalGrowth = displayMode === 'real' ? 0 : inflation;
  const drawdownEndYears = goalYears + drawdownYears;
  // Post-retirement regime: alpha isn't assumed to persist into drawdown, so
  // returns fall back to the benchmark unless the user chose a custom rate.
  const drawdownReturn = returnSource === 'custom' ? customReturn : benchReturn;

  // Simulate far enough to score the goal — and the drawdown phase — even when
  // they sit beyond the chart horizon.
  const simTarget = drawdownActive ? drawdownEndYears : goalYears;
  const simHorizon = Math.max(horizonYears, Math.min(Math.max(simTarget, 0), MAX_SIM_YEARS));

  // The Monte Carlo — memoized on inputs so slider tweaks only rerun when needed.
  const simulation = useMemo(() => {
    if (!inputs || startingValue <= 0 || simHorizon <= 0) return null;
    return runSimulation({
      startingValue,
      contribution,
      contributionGrowth,
      contributionYears,
      withdrawalAnnual: drawdownActive ? targetIncome * 12 : 0,
      withdrawalStartYear: drawdownActive ? goalYears : null,
      withdrawalGrowth,
      expectedReturn,
      drawdownReturn,
      volatility,
      horizonYears: simHorizon,
      paths: DEFAULT_MONTE_CARLO_PATHS,
    });
  }, [inputs, startingValue, contribution, contributionGrowth, contributionYears,
    drawdownActive, targetIncome, goalYears, withdrawalGrowth,
    expectedReturn, drawdownReturn, volatility, simHorizon]);

  // Shape for Recharts: one datum per year with all percentile bands flattened
  // into stacked deltas so AreaChart can draw the fan.
  const chartData = useMemo(() => {
    if (!simulation) return [];
    // The simulation may run past the chart horizon (goal year); clip for display.
    return simulation.years.slice(0, horizonYears + 1).map((year, i) => ({
      year,
      p10: simulation.percentiles.p10[i],
      p25: simulation.percentiles.p25[i],
      p50: simulation.percentiles.p50[i],
      p75: simulation.percentiles.p75[i],
      p90: simulation.percentiles.p90[i],
      det: simulation.deterministic[i],
      invested: simulation.invested[i],
      // Stacked area deltas (for fan rendering):
      band_lo: simulation.percentiles.p10[i],
      band_p25: simulation.percentiles.p25[i] - simulation.percentiles.p10[i],
      band_p50: simulation.percentiles.p50[i] - simulation.percentiles.p25[i],
      band_p75: simulation.percentiles.p75[i] - simulation.percentiles.p50[i],
      band_p90: simulation.percentiles.p90[i] - simulation.percentiles.p75[i],
    }));
  }, [simulation, horizonYears]);

  // Explicit ticks so long horizons get clean marks (and Y0 is never dropped).
  const xTicks = useMemo(() => {
    const step = horizonYears > 20 ? 5 : horizonYears > 10 ? 2 : 1;
    const ticks = [];
    for (let y = 0; y <= horizonYears; y += step) ticks.push(y);
    return ticks;
  }, [horizonYears]);

  // Summary table at 5-year checkpoints (clipped to horizon).
  const summaryCheckpoints = useMemo(() => {
    if (!simulation) return [];
    const marks = [5, 10, 15, 20, 25, 30, 35, 40].filter((m) => m <= horizonYears);
    return marks.map((year) => ({
      year,
      p10: simulation.percentiles.p10[year],
      p50: simulation.percentiles.p50[year],
      p90: simulation.percentiles.p90[year],
    }));
  }, [simulation, horizonYears]);

  // Goal seeker outputs.
  const goalSeeker = useMemo(() => {
    const years = goalYears;
    if (!inputs || startingValue <= 0 || years <= 0
        || effectiveTarget == null || effectiveTarget <= 0) {
      return null;
    }
    const requiredCagr = solveRequiredCagr(
      startingValue,
      contribution,
      contributionGrowth,
      years,
      effectiveTarget,
    );
    const requiredContribution = solveRequiredContribution(
      startingValue,
      expectedReturn,
      contributionGrowth,
      years,
      effectiveTarget,
    );
    // Share of simulated paths at or above the target in the goal year.
    const probability = simulation && years <= simHorizon
      ? fractionAtOrAbove(simulation.sortedPerYear[years], effectiveTarget)
      : null;
    // Drawdown scoring: paths still solvent at the end of the withdrawal phase
    // (a depleted pot is exactly 0), and the median residue.
    let survival = null;
    let medianEnd = null;
    if (drawdownActive && simulation && drawdownEndYears <= simHorizon) {
      survival = fractionAtOrAbove(simulation.sortedPerYear[drawdownEndYears], 1);
      medianEnd = simulation.percentiles.p50[drawdownEndYears];
    }
    return { years, requiredCagr, requiredContribution, probability, survival, medianEnd };
  }, [inputs, startingValue, contribution, contributionGrowth,
    expectedReturn, effectiveTarget, goalYears, simulation, simHorizon,
    drawdownActive, drawdownEndYears]);

  if (loading) {
    return (
      <div className="page-fixed projection-container">
        <div className="loading">Loading projection inputs...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-fixed projection-container">
        <div className="error">{error}</div>
      </div>
    );
  }

  const benchLabel = inputs.benchmark.label;
  const userTwrr = inputs.portfolio?.twrr;
  const trackRecordYears = inputs.portfolio?.track_record_years;
  const maskableValue = (v) => (hideAmounts ? MASK : formatGbp(v));
  const cpiAsOf = inputs.inflation?.uk_cpi_as_of;
  const cpiAsOfLabel = cpiAsOf
    ? new Date(cpiAsOf).toLocaleDateString('en-GB', { month: 'short', year: 'numeric' })
    : null;
  const cpiStale = cpiAsOf != null
    && (Date.now() - new Date(cpiAsOf).getTime()) / 86400000 > CPI_STALE_DAYS;
  // Hide the target label (but keep the line) when the target is dwarfed by the
  // chart scale — it collides with the X-axis labels down there.
  const chartPeak = chartData.reduce((m, d) => Math.max(m, d.p90), 0);
  const showTargetLabel = effectiveTarget != null && effectiveTarget >= 0.03 * chartPeak;

  return (
    <div className="page-fixed projection-container">
      <div className="projection-header">
        <h1>Portfolio Projection</h1>
        <p className="projection-subtitle">
          Monte Carlo simulation of your ISA over the next {horizonYears} years.
          All figures are in {displayMode === 'real' ? 'today\'s money (CPI-adjusted)' : 'future GBP (nominal)'}.
        </p>
      </div>

      <div className="projection-layout">
        <aside className="projection-assumptions">
          <h2>Assumptions</h2>

          <div className="assumption">
            <span className="assumption-label">Display mode</span>
            <div className="segmented" role="radiogroup" aria-label="Display mode">
              <button
                type="button"
                className={displayMode === 'real' ? 'segmented-btn active' : 'segmented-btn'}
                onClick={() => setDisplayMode('real')}
              >
                Real (CPI-adjusted)
              </button>
              <button
                type="button"
                className={displayMode === 'nominal' ? 'segmented-btn active' : 'segmented-btn'}
                onClick={() => setDisplayMode('nominal')}
              >
                Nominal
              </button>
            </div>
          </div>

          <div className="assumption">
            <label htmlFor="startingValue">Starting portfolio</label>
            {hideAmounts ? (
              <div className="assumption-input-masked" aria-label="Starting portfolio (hidden)">
                {MASK}
              </div>
            ) : (
              <>
                <input
                  id="startingValue"
                  type="number"
                  min="0"
                  step="1000"
                  value={startingValueInput ?? Math.round(inputs.portfolio?.starting_value ?? 0)}
                  onChange={(e) => setStartingValueInput(e.target.value)}
                />
                {startingValueInput != null && (
                  <button
                    type="button"
                    className="input-reset"
                    onClick={() => setStartingValueInput(null)}
                  >
                    Reset to live value
                  </button>
                )}
              </>
            )}
          </div>

          <div className="assumption">
            <label htmlFor="contribution">Annual contribution</label>
            <input
              id="contribution"
              type="number"
              min="0"
              step="500"
              value={contribution}
              onChange={(e) => setContribution(Number(e.target.value))}
            />
            <label className="assumption-inline">
              <input
                type="checkbox"
                checked={contributionGrowsWithInflation}
                onChange={(e) => setContributionGrowsWithInflation(e.target.checked)}
              />
              <span>Contributions keep real purchasing power</span>
            </label>
            <div className="assumption-readout">
              Invested as a start-of-year lump sum
              {drawdownActive ? `, until retirement (${targetYear}).` : '.'}
            </div>
            {contribution > ISA_ANNUAL_ALLOWANCE && (
              <div className="warning-note">
                Above the £{ISA_ANNUAL_ALLOWANCE.toLocaleString()} annual ISA allowance.
              </div>
            )}
          </div>

          <div className="assumption">
            <label htmlFor="horizon">Horizon (years)</label>
            <input
              id="horizon"
              type="range"
              min="5"
              max={MAX_SIM_YEARS}
              step="1"
              value={horizonYears}
              onChange={(e) => setHorizonYears(Number(e.target.value))}
            />
            <div className="assumption-readout">{horizonYears} years</div>
          </div>

          <div className="assumption">
            <label htmlFor="returnSource">Expected return source</label>
            <select
              id="returnSource"
              value={returnSource}
              onChange={(e) => setReturnSource(e.target.value)}
            >
              <option value="blend">Blend (50% your TWRR + 50% benchmark)</option>
              <option value="twrr">Your TWRR only</option>
              <option value="benchmark">Benchmark only ({benchLabel})</option>
              <option value="custom">Custom</option>
            </select>
            {returnSource === 'custom' && (
              <div className="custom-return-row">
                <label htmlFor="customReturn" className="custom-return-label">Return %</label>
                <input
                  id="customReturn"
                  type="number"
                  step="0.1"
                  value={customReturnPct}
                  onChange={(e) => setCustomReturnPct(e.target.value)}
                />
                <label htmlFor="customVol" className="custom-return-label">σ %</label>
                <input
                  id="customVol"
                  type="number"
                  step="0.1"
                  value={customVolPct}
                  onChange={(e) => setCustomVolPct(e.target.value)}
                />
              </div>
            )}
            <div className="assumption-readout">
              {formatPct(expectedReturn)} return · σ {formatPct(volatility, 0)}
              {drawdownActive && drawdownReturn !== expectedReturn
                && ` · ${formatPct(drawdownReturn)} from ${targetYear}`}
            </div>
          </div>

          <div className="assumption-summary">
            <div>Your TWRR: <strong>{formatPct(userTwrr)}</strong>
              {trackRecordYears != null && <span className="muted"> · {trackRecordYears.toFixed(1)}y history</span>}
            </div>
            <div>Benchmark ({displayMode}): <strong>{formatPct(benchReturn)}</strong></div>
            <div>UK CPI (10y avg): <strong>{formatPct(inflation)}</strong>
              <span className="muted">
                {cpiAsOfLabel ? ` · to ${cpiAsOfLabel}` : ' · default assumption'}
              </span>
            </div>
            {cpiStale && (
              <div className="warning-note">
                CPI data ends {cpiAsOfLabel} — the ONS feed looks stale; inflation
                figures may be out of date.
              </div>
            )}
            {userTwrr == null && (
              <div className="warning-note">
                No annualised TWRR available yet — projection is using the benchmark only.
              </div>
            )}
            {userTwrr != null && trackRecordYears != null && trackRecordYears < 3 && (
              <div className="warning-note">
                Your track record is short ({trackRecordYears.toFixed(1)}y). The blended default
                dampens that noise; prefer it over &quot;TWRR only&quot; for long projections.
              </div>
            )}
          </div>
        </aside>

        <section className="projection-main">
          <div className="card chart-card">
            <div className="chart-header">
              <h2>Projected value</h2>
              <div className="chart-legend">
                <span className="legend-item"><span className="legend-swatch swatch-fan-wide" /> 10–90th percentile</span>
                <span className="legend-item"><span className="legend-swatch swatch-fan-narrow" /> 25–75th percentile</span>
                <span className="legend-item"><span className="legend-swatch swatch-median" /> Median path</span>
                <span className="legend-item"><span className="legend-swatch swatch-det" /> Expected path (no volatility)</span>
                <span className="legend-item"><span className="legend-swatch swatch-invested" /> Total invested</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={420}>
              <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6e9ef" />
                <XAxis
                  dataKey="year"
                  ticks={xTicks}
                  tickFormatter={(y) => `Y${y}`}
                  stroke="#6b7280"
                />
                <YAxis
                  tickFormatter={(v) => (hideAmounts ? MASK : formatGbp(v))}
                  stroke="#6b7280"
                  width={72}
                />
                <Tooltip content={<FanTooltip mask={maskableValue} />} />
                <Area type="monotone" dataKey="band_lo" stackId="fan" stroke="none" fill="transparent" />
                <Area type="monotone" dataKey="band_p25" stackId="fan" stroke="none" fill="#93c5fd" fillOpacity={0.35} name="p25" />
                <Area type="monotone" dataKey="band_p50" stackId="fan" stroke="none" fill="#60a5fa" fillOpacity={0.55} name="p50" />
                <Area type="monotone" dataKey="band_p75" stackId="fan" stroke="none" fill="#60a5fa" fillOpacity={0.55} name="p75" />
                <Area type="monotone" dataKey="band_p90" stackId="fan" stroke="none" fill="#93c5fd" fillOpacity={0.35} name="p90" />
                <Line type="monotone" dataKey="invested" stroke="#6b7280" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="p50" stroke="#1d4ed8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="det" stroke="#ef4444" strokeDasharray="5 4" strokeWidth={2} dot={false} />
                {goalSeeker && (
                  <ReferenceLine
                    y={effectiveTarget}
                    stroke="#059669"
                    strokeDasharray="4 4"
                    ifOverflow="extendDomain"
                    label={showTargetLabel
                      ? { value: 'Target', position: 'insideTopRight', fill: '#059669', fontSize: 11 }
                      : undefined}
                  />
                )}
                {goalSeeker && goalSeeker.years <= horizonYears && (
                  <ReferenceLine
                    x={goalSeeker.years}
                    stroke="#059669"
                    strokeDasharray="4 4"
                    label={{ value: `${targetYear}`, position: 'insideTop', fill: '#059669', fontSize: 11 }}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="card summary-card">
            <h2>
              Checkpoints
              {drawdownActive && (
                <span className="muted-note"> · net of withdrawals from {targetYear}</span>
              )}
            </h2>
            <table className="summary-table">
              <thead>
                <tr>
                  <th>Years</th>
                  <th className="num">Pessimistic (P10)</th>
                  <th className="num">Median (P50)</th>
                  <th className="num">Optimistic (P90)</th>
                </tr>
              </thead>
              <tbody>
                {summaryCheckpoints.map((row) => (
                  <tr key={row.year}>
                    <td>{row.year}y</td>
                    <td className="num">{maskableValue(row.p10)}</td>
                    <td className="num strong">{maskableValue(row.p50)}</td>
                    <td className="num">{maskableValue(row.p90)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card goal-card">
            <div className="goal-header">
              <h2>Goal seeker</h2>
              <div className="segmented goal-mode" role="radiogroup" aria-label="Goal type">
                <button
                  type="button"
                  className={goalMode === 'amount' ? 'segmented-btn active' : 'segmented-btn'}
                  onClick={() => setGoalMode('amount')}
                >
                  Target pot
                </button>
                <button
                  type="button"
                  className={goalMode === 'income' ? 'segmented-btn active' : 'segmented-btn'}
                  onClick={() => setGoalMode('income')}
                >
                  Monthly income
                </button>
              </div>
            </div>
            <div className="goal-inputs">
              {goalMode === 'amount' ? (
                <label>
                  Target ({displayMode === 'real' ? 'today\'s money' : 'nominal'})
                  <input
                    type="number"
                    min="0"
                    step="10000"
                    value={targetAmount}
                    onChange={(e) => setTargetAmount(Number(e.target.value))}
                  />
                </label>
              ) : (
                <>
                  <label>
                    Monthly income ({displayMode === 'real' ? 'today\'s money' : 'nominal'})
                    <input
                      type="number"
                      min="0"
                      step="100"
                      value={targetIncome}
                      onChange={(e) => setTargetIncome(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Withdrawal rate %
                    <input
                      type="number"
                      min="1"
                      max="10"
                      step="0.1"
                      value={swrPct}
                      onChange={(e) => setSwrPct(e.target.value)}
                    />
                  </label>
                  <label>
                    Draw for (years)
                    <input
                      type="number"
                      min="5"
                      max="40"
                      step="1"
                      value={drawdownYears}
                      onChange={(e) => setDrawdownYears(Number(e.target.value))}
                    />
                  </label>
                </>
              )}
              <label>
                By year
                <input
                  type="number"
                  min={currentYear + 1}
                  max={currentYear + MAX_GOAL_YEARS}
                  step="1"
                  value={targetYear}
                  onChange={(e) => setTargetYear(Number(e.target.value))}
                />
              </label>
            </div>
            {goalMode === 'income' && effectiveTarget != null && (
              <div className="goal-implied">
                Drawing £{targetIncome.toLocaleString()}/mo at {formatPct(swr, 1)} needs
                a pot of <strong>{formatGbp(effectiveTarget)}</strong>.
              </div>
            )}
            {goalSeeker && simTarget > horizonYears && (
              <div className="goal-extend">
                Chart shows the first {horizonYears} years —{' '}
                <button
                  type="button"
                  className="link-button"
                  onClick={() => setHorizonYears(Math.min(simTarget, MAX_SIM_YEARS))}
                >
                  extend to {currentYear + Math.min(simTarget, MAX_SIM_YEARS)}
                </button>
              </div>
            )}
            {goalSeeker && (
              <div className="goal-outputs">
                <div className="goal-output">
                  <div className="goal-label">Chance of hitting it</div>
                  <div className={probabilityClass(goalSeeker.probability)}>
                    {goalSeeker.probability == null
                      ? '—'
                      : formatPct(goalSeeker.probability, 0)}
                  </div>
                  <div className="goal-note">
                    of {DEFAULT_MONTE_CARLO_PATHS.toLocaleString()} simulated paths by {targetYear}
                  </div>
                </div>
                {goalMode === 'income' && (
                  <>
                    <div className="goal-output">
                      <div className="goal-label">Pot survives drawdown</div>
                      <div className={probabilityClass(goalSeeker.survival)}>
                        {goalSeeker.survival == null
                          ? '—'
                          : formatPct(goalSeeker.survival, 0)}
                      </div>
                      <div className="goal-note">
                        drawing £{targetIncome.toLocaleString()}/mo until {targetYear + drawdownYears}
                      </div>
                    </div>
                    <div className="goal-output">
                      <div className="goal-label">Median pot left</div>
                      <div className="goal-value">
                        {goalSeeker.medianEnd == null ? '—' : maskableValue(goalSeeker.medianEnd)}
                      </div>
                      <div className="goal-note">
                        in {targetYear + drawdownYears}
                        {displayMode === 'real' ? ", today's money" : ''}
                      </div>
                    </div>
                  </>
                )}
                <div className="goal-output">
                  <div className="goal-label">Required CAGR</div>
                  <div className="goal-value">
                    {goalSeeker.requiredCagr == null
                      ? 'out of range'
                      : formatPct(goalSeeker.requiredCagr)}
                  </div>
                  <div className="goal-note">at £{contribution.toLocaleString()}/yr contribution</div>
                </div>
                <div className="goal-output">
                  <div className="goal-label">Required monthly contribution</div>
                  <div className="goal-value">
                    {goalSeeker.requiredContribution == null
                      ? '—'
                      : (hideAmounts ? MASK : formatGbpFull(goalSeeker.requiredContribution / 12))}
                  </div>
                  <div className="goal-note">at {formatPct(expectedReturn)} assumed return</div>
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};

export default Projection;
