import React, { useState, useEffect, useMemo } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
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
const DEFAULT_HORIZON_YEARS = 30;
const DEFAULT_MONTE_CARLO_PATHS = 10000;

// Box–Muller standard normal.
function randn() {
  const u = 1 - Math.random();
  const v = 1 - Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Future value of starting pot plus annual contributions (paid at start of year,
// compounded for the full year). Supports a growing annuity (g = contribution growth).
function futureValue(V0, contribution, r, g, T) {
  const rT = (1 + r) ** T;
  if (Math.abs(r - g) < 1e-9) {
    return V0 * rT + contribution * T * rT;
  }
  return V0 * rT + contribution * (1 + r) * (rT - (1 + g) ** T) / (r - g);
}

// Bisection on r such that FV == target. Returns null if target unreachable in [-50%, +50%].
function solveRequiredCagr(V0, contribution, g, T, target) {
  let lo = -0.5;
  let hi = 0.5;
  const fvLo = futureValue(V0, contribution, lo, g, T);
  const fvHi = futureValue(V0, contribution, hi, g, T);
  if (target < fvLo || target > fvHi) return null;
  for (let i = 0; i < 80; i += 1) {
    const mid = (lo + hi) / 2;
    const fv = futureValue(V0, contribution, mid, g, T);
    if (fv < target) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// Required annual contribution to hit `target` given an assumed CAGR `r`.
function solveRequiredContribution(V0, r, g, T, target) {
  const rT = (1 + r) ** T;
  const compoundV0 = V0 * rT;
  const remaining = target - compoundV0;
  if (remaining <= 0) return 0;
  const multiplier = Math.abs(r - g) < 1e-9
    ? T * rT
    : (1 + r) * (rT - (1 + g) ** T) / (r - g);
  return remaining / multiplier;
}

// Monte Carlo simulation under geometric Brownian motion, annual steps.
// Returns { years: number[T+1],
//           percentiles: { p10, p25, p50, p75, p90: number[T+1] },
//           deterministic: number[T+1],  // expected path (no volatility)
//           invested:      number[T+1] } // cumulative capital in, no returns
function runSimulation({
  startingValue,
  contribution,
  contributionGrowth,
  expectedReturn,
  volatility,
  horizonYears,
  paths,
}) {
  const T = horizonYears;
  const N = paths;
  const mu = Math.log(1 + expectedReturn);
  const sigma = volatility;

  // Simulate all paths year-by-year. Store results per year as a flat typed array
  // so sort-based percentile extraction stays fast.
  const perYear = [];
  for (let t = 0; t <= T; t += 1) perYear.push(new Float64Array(N));

  for (let i = 0; i < N; i += 1) {
    let value = startingValue;
    let contrib = contribution;
    perYear[0][i] = value;
    for (let t = 1; t <= T; t += 1) {
      const z = randn();
      const r = Math.exp(mu - 0.5 * sigma * sigma + sigma * z) - 1;
      value = (value + contrib) * (1 + r);
      perYear[t][i] = value;
      contrib *= 1 + contributionGrowth;
    }
  }

  const percentiles = { p10: [], p25: [], p50: [], p75: [], p90: [] };
  for (let t = 0; t <= T; t += 1) {
    const sorted = perYear[t].slice().sort();
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
  deterministic.push(detValue);
  invested.push(investedTotal);
  for (let t = 1; t <= T; t += 1) {
    detValue = (detValue + detContrib) * (1 + expectedReturn);
    investedTotal += detContrib;
    deterministic.push(detValue);
    invested.push(investedTotal);
    detContrib *= 1 + contributionGrowth;
  }

  const years = Array.from({ length: T + 1 }, (_, i) => i);
  return { years, percentiles, deterministic, invested };
}

const formatGbp = (value) => {
  if (value == null || !Number.isFinite(value)) return '—';
  if (Math.abs(value) >= 1e6) return `£${(value / 1e6).toFixed(2)}M`;
  if (Math.abs(value) >= 1e3) return `£${(value / 1e3).toFixed(0)}k`;
  return `£${Math.round(value).toLocaleString()}`;
};

const formatPct = (value, digits = 1) => {
  if (value == null || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
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
  const [customReturn, setCustomReturn] = useState(0.07);
  const [customVolatility, setCustomVolatility] = useState(0.16);
  const [contribution, setContribution] = useState(DEFAULT_CONTRIBUTION);
  const [contributionGrowsWithInflation, setContributionGrowsWithInflation] = useState(true);
  const [horizonYears, setHorizonYears] = useState(DEFAULT_HORIZON_YEARS);
  const [startingValueOverride, setStartingValueOverride] = useState(null);
  const [targetAmount, setTargetAmount] = useState(500000);
  const [targetYear, setTargetYear] = useState(new Date().getFullYear() + 15);

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

  const startingValue = useMemo(() => {
    if (startingValueOverride != null) return startingValueOverride;
    return inputs?.portfolio?.starting_value ?? 0;
  }, [inputs, startingValueOverride]);

  // The Monte Carlo — memoized on inputs so slider tweaks only rerun when needed.
  const simulation = useMemo(() => {
    if (!inputs || startingValue <= 0 || horizonYears <= 0) return null;
    return runSimulation({
      startingValue,
      contribution,
      contributionGrowth,
      expectedReturn,
      volatility,
      horizonYears,
      paths: DEFAULT_MONTE_CARLO_PATHS,
    });
  }, [inputs, startingValue, contribution, contributionGrowth, expectedReturn, volatility, horizonYears]);

  // Shape for Recharts: one datum per year with all percentile bands flattened
  // into stacked deltas so AreaChart can draw the fan.
  const chartData = useMemo(() => {
    if (!simulation) return [];
    return simulation.years.map((year, i) => ({
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
  }, [simulation]);

  // Summary table at 5/10/15/20/25/30 year checkpoints (clipped to horizon).
  const summaryCheckpoints = useMemo(() => {
    if (!simulation) return [];
    const marks = [5, 10, 15, 20, 25, 30].filter((m) => m <= horizonYears);
    return marks.map((year) => ({
      year,
      p10: simulation.percentiles.p10[year],
      p50: simulation.percentiles.p50[year],
      p90: simulation.percentiles.p90[year],
    }));
  }, [simulation, horizonYears]);

  // Goal seeker outputs.
  const goalSeeker = useMemo(() => {
    const years = targetYear - new Date().getFullYear();
    if (!inputs || startingValue <= 0 || years <= 0 || targetAmount <= 0) return null;
    const requiredCagr = solveRequiredCagr(
      startingValue,
      contribution,
      contributionGrowth,
      years,
      targetAmount,
    );
    const requiredContribution = solveRequiredContribution(
      startingValue,
      expectedReturn,
      contributionGrowth,
      years,
      targetAmount,
    );
    return { years, requiredCagr, requiredContribution };
  }, [inputs, startingValue, contribution, contributionGrowth, expectedReturn, targetAmount, targetYear]);

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
            <label htmlFor="displayMode">Display mode</label>
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
              <input
                id="startingValue"
                type="number"
                min="0"
                step="1000"
                value={startingValueOverride ?? Math.round(inputs.portfolio?.starting_value ?? 0)}
                onChange={(e) => setStartingValueOverride(Number(e.target.value))}
              />
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
          </div>

          <div className="assumption">
            <label htmlFor="horizon">Horizon (years)</label>
            <input
              id="horizon"
              type="range"
              min="5"
              max="40"
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
                  value={(customReturn * 100).toFixed(1)}
                  onChange={(e) => setCustomReturn(Number(e.target.value) / 100)}
                />
                <label htmlFor="customVol" className="custom-return-label">σ %</label>
                <input
                  id="customVol"
                  type="number"
                  step="0.1"
                  value={(customVolatility * 100).toFixed(1)}
                  onChange={(e) => setCustomVolatility(Number(e.target.value) / 100)}
                />
              </div>
            )}
            <div className="assumption-readout">
              {formatPct(expectedReturn)} return · σ {formatPct(volatility, 0)}
            </div>
          </div>

          <div className="assumption-summary">
            <div>Your TWRR: <strong>{formatPct(userTwrr)}</strong>
              {trackRecordYears != null && <span className="muted"> · {trackRecordYears.toFixed(1)}y history</span>}
            </div>
            <div>Benchmark ({displayMode}): <strong>{formatPct(benchReturn)}</strong></div>
            <div>UK CPI (10y avg): <strong>{formatPct(inflation)}</strong></div>
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
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="card summary-card">
            <h2>Checkpoints</h2>
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
            <h2>Goal seeker</h2>
            <div className="goal-inputs">
              <label>
                Target
                <input
                  type="number"
                  min="0"
                  step="10000"
                  value={targetAmount}
                  onChange={(e) => setTargetAmount(Number(e.target.value))}
                />
              </label>
              <label>
                By year
                <input
                  type="number"
                  min={new Date().getFullYear() + 1}
                  max={new Date().getFullYear() + 50}
                  step="1"
                  value={targetYear}
                  onChange={(e) => setTargetYear(Number(e.target.value))}
                />
              </label>
            </div>
            {goalSeeker && (
              <div className="goal-outputs">
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
                    {maskableValue(goalSeeker.requiredContribution / 12)}
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
