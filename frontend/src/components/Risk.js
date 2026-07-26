import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { portfolioAPI } from '../services/api';
import { useHideAmounts, MASK } from '../context/HideAmountsContext';
import './Risk.css';

const formatGbp = (value) => {
  if (value == null || !Number.isFinite(value)) return '—';
  return `£${Math.round(value).toLocaleString()}`;
};

const formatPct = (value, digits = 1) => {
  if (value == null || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
};

// Signed percentage-point delta, e.g. +1.9pp / -0.4pp
const formatPp = (value) => {
  if (value == null || !Number.isFinite(value)) return '—';
  const pp = value * 100;
  return `${pp > 0 ? '+' : ''}${pp.toFixed(1)}pp`;
};

// "Risk-heavy" only when contribution is both materially (>1pp) and
// proportionally (>1.5x) above the value weight.
const isRiskHeavy = (h) =>
  h.risk_contribution > h.weight + 0.01 && h.risk_contribution > h.weight * 1.5;

// The diverging chart shows the largest |risk - weight| gaps; the rest are noise.
const CHART_TOP_N = 20;

// Cell color thresholds. Screener scores run roughly -10..60 (normalizer 50 on
// the Holdings page); +£1k impacts beyond ~15bp materially concentrate risk.
const SCREENER_GOOD = 25;
const VOL_DELTA_BAD_BP = 15;

const rcDeltaClass = (v) => {
  if (v > 0.01) return 'neg';
  if (v < -0.01) return 'pos';
  return '';
};

const volDeltaClass = (v) => {
  if (v <= 0) return 'pos';
  if (v > VOL_DELTA_BAD_BP) return 'neg';
  return '';
};

const screenerClass = (v) => {
  if (v == null) return '';
  if (v >= SCREENER_GOOD) return 'pos';
  if (v < 0) return 'neg';
  return '';
};

// Volatility vs benchmark: growth portfolios run hot, so "good" is relative.
const volLevel = (vol, benchVol) => {
  const ratio = benchVol ? vol / benchVol : vol / 0.16;
  if (ratio <= 1.5) return 'good';
  if (ratio <= 2.5) return 'mid';
  return 'bad';
};

// Effective bets relative to position count: below 20% means a few names
// dominate the risk regardless of how long the holdings list is.
const betsLevel = (bets, n) => {
  const ratio = bets / n;
  if (ratio >= 0.35) return 'good';
  if (ratio >= 0.2) return 'mid';
  return 'bad';
};

const StockLink = ({ symbol }) => (
  <Link className="risk-symbol" to={`/stock/${encodeURIComponent(symbol)}`}>
    {symbol}
  </Link>
);

const SortableTh = ({ label, colKey, sort, setSort, defaultDir = 'desc', className = '' }) => {
  const active = sort.key === colKey;
  const arrow = active ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : '';
  const onClick = () =>
    setSort(active ? { key: colKey, dir: sort.dir === 'asc' ? 'desc' : 'asc' } : { key: colKey, dir: defaultDir });
  return (
    <th className={`sortable ${className}`} onClick={onClick}>
      {label}
      {arrow}
    </th>
  );
};

const sortRows = (rows, { key, dir }) => {
  const sorted = [...rows].sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    if (typeof av === 'string' || typeof bv === 'string') {
      return String(av ?? '').localeCompare(String(bv ?? ''));
    }
    return (av ?? -Infinity) - (bv ?? -Infinity);
  });
  return dir === 'desc' ? sorted.reverse() : sorted;
};

const DivergingTooltip = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;
  const h = payload[0].payload;
  return (
    <div className="risk-chart-tooltip">
      <div className="risk-chart-tooltip-title">{h.symbol}</div>
      <div>Value weight: {formatPct(h.weight)}</div>
      <div>Risk contribution: {formatPct(h.rc)}</div>
      <div>Difference: {formatPp(h.delta)}</div>
    </div>
  );
};

const Risk = () => {
  const [data, setData] = useState(null);
  const [scores, setScores] = useState(null);
  const [error, setError] = useState(null);
  const [buySort, setBuySort] = useState({ key: 'next_1k_vol_delta_bp', dir: 'asc' });
  const [detailSort, setDetailSort] = useState({ key: 'risk_contribution', dir: 'desc' });
  const { hideAmounts } = useHideAmounts();

  useEffect(() => {
    portfolioAPI
      .getRisk()
      .then(setData)
      .catch((e) => setError(e.message || 'Failed to load risk data'));
    // Screener scores come from the holdings endpoint; the page works without them.
    portfolioAPI
      .getCurrentHoldings()
      .then((res) => {
        const map = {};
        (res.holdings || []).forEach((h) => {
          // ETFs can't pass equity screeners — their 0 means "not applicable"
          if (h.yahoo_symbol) map[h.yahoo_symbol] = h.quote_type === 'ETF' ? null : h.screener_score;
        });
        setScores(map);
      })
      .catch(() => setScores({}));
  }, []);

  if (error) {
    return (
      <div className="page-fixed risk-container">
        <div className="risk-error">Error: {error}</div>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="page-fixed risk-container">
        <div className="risk-loading">Building risk model…</div>
      </div>
    );
  }

  const { portfolio, window: win, holdings, clusters, excluded, warnings, groups } = data;
  const riskAdjusted = data.risk_adjusted;
  const maskable = (v) => (hideAmounts ? MASK : formatGbp(v));

  const dailyMove = portfolio.annual_volatility / Math.sqrt(252);
  const severity = warnings?.severity;

  // Enrich rows once so sorting can use screener_score like any other column.
  const rows = holdings.map((h) => ({
    ...h,
    screener_score: scores?.[h.symbol] ?? null,
    rc_minus_weight: h.risk_contribution - h.weight,
  }));

  const chartData = rows
    .map((h) => ({
      symbol: h.symbol,
      weight: h.weight,
      rc: h.risk_contribution,
      delta: h.rc_minus_weight,
    }))
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, CHART_TOP_N)
    .sort((a, b) => b.delta - a.delta);

  const scoreCell = (v) => (v == null ? '—' : Number(v).toFixed(1));

  return (
    <div className="page-fixed risk-container">
      <div className="risk-header">
        <h1>Risk</h1>
        <p className="risk-subtitle">
          Where portfolio risk is concentrated, and where new money would diversify it. Model:{' '}
          {win.n_obs} days of GBP daily returns ({win.start} → {win.end}), Ledoit-Wolf shrunk
          covariance (δ = {portfolio.shrinkage.toFixed(2)}). As of {data.as_of}. Covers this ISA
          only — holdings in other accounts are not modelled, so concentration measured here
          understates total-wealth exposure.
        </p>
      </div>

      {severity === 'high' && (
        <div className="risk-warning-banner">
          <strong>Model inputs look unreliable</strong> — the numbers below are likely inflated.
          {warnings.gap_lumped.length > 0 && (
            <div>
              Price holes bridged by forward-fill (multi-day moves counted as one day):{' '}
              {warnings.gap_lumped.map((w) => `${w.symbol} (${w.max_gap_days}d)`).join(', ')}
            </div>
          )}
          {warnings.extreme_returns.length > 0 && (
            <div>
              Single-day GBP returns beyond ±60%:{' '}
              {warnings.extreme_returns
                .map((w) => `${w.symbol} ${formatPct(w.value, 0)} on ${w.date}`)
                .join(', ')}
            </div>
          )}
          <div>
            Fix: run <code>scripts/update_data.py</code>,{' '}
            <code>scripts/backfill_currency_rates.py</code> and{' '}
            <code>scripts/fix_split_prices.py</code>, then restart the API.
          </div>
        </div>
      )}
      {severity === 'info' && (
        <div className="risk-info-banner">
          Notable single-day moves in the window:{' '}
          {warnings.extreme_returns
            .map((w) => `${w.symbol} ${formatPct(w.value, 0)} on ${w.date}`)
            .join(', ')}
          . These look like genuine market events (isolated names, no data gaps) — they raise that
          name&apos;s measured volatility, but the model is trustworthy.
        </div>
      )}

      <div className="risk-cards">
        <div className="risk-card">
          <div className="risk-card-label">Annualised volatility</div>
          <div className={`risk-card-value level-${volLevel(portfolio.annual_volatility, portfolio.benchmark_volatility)}`}>
            {formatPct(portfolio.annual_volatility)}
          </div>
          <div className="risk-card-note">
            ≈ ±{formatPct(dailyMove)} on a typical day
            {portfolio.benchmark_volatility != null &&
              ` · ${(portfolio.annual_volatility / portfolio.benchmark_volatility).toFixed(1)}× ${portfolio.benchmark_symbol} (${formatPct(portfolio.benchmark_volatility)})`}
          </div>
        </div>
        <div className="risk-card">
          <div className="risk-card-label">Effective number of bets</div>
          <div className={`risk-card-value level-${betsLevel(portfolio.effective_bets, win.n_assets)}`}>
            {portfolio.effective_bets.toFixed(1)}
          </div>
          <div className="risk-card-note">
            {win.n_assets} positions behave like ~{Math.round(portfolio.effective_bets)}{' '}
            independent ones — see the groups below
          </div>
        </div>
        {riskAdjusted?.sharpe_ratio != null && (
          <div className="risk-card">
            <div className="risk-card-label">Risk-adjusted return</div>
            <div className="risk-card-value">{riskAdjusted.sharpe_ratio.toFixed(2)}</div>
            <div className="risk-card-note">
              Sharpe
              {riskAdjusted.sortino_ratio != null && ` · Sortino ${riskAdjusted.sortino_ratio.toFixed(2)}`}
              {riskAdjusted.jensens_alpha != null && ` · alpha ${riskAdjusted.jensens_alpha.toFixed(1)}%`}
              {riskAdjusted.beta != null && ` · beta ${riskAdjusted.beta.toFixed(2)}`}
              {' — whether the volatility above is being paid for'}
            </div>
          </div>
        )}
        <div className="risk-card">
          <div className="risk-card-label">Modelled value</div>
          <div className="risk-card-value">{maskable(portfolio.model_value_gbp)}</div>
          <div className="risk-card-note">
            {formatPct(portfolio.model_value_gbp / portfolio.total_value_gbp)} of this ISA
          </div>
        </div>
      </div>

      {groups?.correlated?.length > 0 && (
        <div className="risk-section">
          <h2>Your actual bets: correlated groups</h2>
          <p className="risk-section-note">
            Positions with average pairwise correlation above{' '}
            {formatPct(groups.min_corr, 0)} move largely as one bet — this is why{' '}
            {win.n_assets} positions only add up to ~{Math.round(portfolio.effective_bets)}{' '}
            effective ones. {groups.independent_count} positions sit outside any group.
          </p>
          <div className="risk-groups">
            {groups.correlated.map((g) => (
              <div className="risk-group" key={g.symbols.join('-')}>
                <div className="risk-group-header">
                  <span className="risk-group-share">{formatPct(g.risk_share)} of risk</span>
                  <span className="risk-group-meta">
                    {formatPct(g.value_weight)} of value · avg corr {g.avg_corr.toFixed(2)}
                  </span>
                </div>
                <div className="risk-group-chips">
                  {g.symbols.map((s) => (
                    <Link
                      key={s}
                      className="risk-chip"
                      to={`/stock/${encodeURIComponent(s)}`}
                    >
                      {s}
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="risk-section">
        <h2>Theme clusters</h2>
        <p className="risk-section-note">
          Two separate readings. The bar is share of portfolio <em>risk</em>, filling toward the
          cluster&apos;s alert threshold — red means risk is disproportionate to the money in it.
          &quot;Buyable&quot; is the <em>value</em> weight the buy gate tests, which excludes ETFs
          from thematic clusters; &quot;closed&quot; means the cluster is at its cap and takes no
          new money. Tags overlap, so shares don&apos;t sum to 100%.
        </p>
        <div className="risk-cluster-list">
          {clusters.map((c) => {
            // Two independent signals. `capped` is a policy state — the cluster is
            // closed to new money on the same basis the agent uses. `alert` is a
            // warning that risk is disproportionate to the money, and blocks nothing.
            const capped = c.gated_weight >= c.soft_cap;
            const alert = c.risk_alert_breached;
            return (
              <div className="risk-cluster-row" key={c.tag}>
                <div className="risk-cluster-name">{c.tag}</div>
                <div className="risk-cluster-bar">
                  {/* Fills toward the risk alert, not the value cap — the two are
                      measured on different quantities and the ratio of one to the
                      other means nothing. The cap is reported as text instead. */}
                  <div
                    className={`risk-cluster-fill ${alert ? 'over' : ''}`}
                    style={{
                      width: `${Math.min((c.risk_share / (c.risk_alert ?? c.soft_cap)) * 100, 100)}%`,
                    }}
                  />
                </div>
                <div className={`risk-cluster-values ${alert ? 'over-text' : ''}`}>
                  risk {formatPct(c.risk_share)}
                  {c.risk_alert != null && ` / alert ${formatPct(c.risk_alert, 0)}`}
                  <span className="risk-cluster-value-weight">
                    {' · '}buyable {formatPct(c.gated_weight)} / cap {formatPct(c.soft_cap, 0)}
                  </span>
                  {capped && <span className="risk-cluster-capped">closed</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="risk-section">
        <h2>Who punches above their weight</h2>
        <p className="risk-section-note">
          Risk contribution minus value weight, top {CHART_TOP_N} positions. Red bars carry more
          risk than their share of money; green bars are risk-diluters. The rest of the portfolio
          sits within ±1pp.
        </p>
        <ResponsiveContainer width="100%" height={chartData.length * 28 + 50}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tickFormatter={(v) => formatPp(v)} tick={{ fontSize: 12 }} />
            <YAxis type="category" dataKey="symbol" width={70} interval={0} tick={{ fontSize: 12 }} />
            <Tooltip content={<DivergingTooltip />} />
            <ReferenceLine x={0} stroke="#9ca3af" />
            {/* Unanimated per PortfolioChart: Recharts 3 + React 19 leaves marks
                at zero extent when the mount animation doesn't fire. */}
            <Bar dataKey="delta" barSize={14} isAnimationActive={false}>
              {chartData.map((d) => (
                <Cell key={d.symbol} fill={d.delta > 0 ? '#ef4444' : '#10b981'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="risk-section">
        <h2>Where should the next £{portfolio.new_money_gbp.toLocaleString()} go?</h2>
        <p className="risk-section-note">
          Volatility impact of adding £{portfolio.new_money_gbp.toLocaleString()} to each position
          (negative = diversifies), alongside the screener score — buy candidates should look good
          on both.
        </p>
        <table className="risk-table">
          <thead>
            <tr>
              <SortableTh label="Symbol" colKey="symbol" sort={buySort} setSort={setBuySort} defaultDir="asc" />
              <SortableTh label="Name" colKey="name" sort={buySort} setSort={setBuySort} defaultDir="asc" />
              <SortableTh label="Δ vol (bp)" colKey="next_1k_vol_delta_bp" sort={buySort} setSort={setBuySort} defaultDir="asc" className="num" />
              <SortableTh label="Screener" colKey="screener_score" sort={buySort} setSort={setBuySort} className="num" />
              <SortableTh label="Weight" colKey="weight" sort={buySort} setSort={setBuySort} className="num" />
            </tr>
          </thead>
          <tbody>
            {/* Blocked names sort last rather than being hidden — a cap breach is
                information, and silently dropping rows reads as a bug. Filtering
                happens before the slice so they can't crowd out fundable ones. */}
            {sortRows(rows, buySort)
              .sort((a, b) => (a.blocked_tags?.length ? 1 : 0) - (b.blocked_tags?.length ? 1 : 0))
              .slice(0, 12)
              .map((h) => (
                <tr key={h.symbol} className={h.blocked_tags?.length ? 'risk-row-blocked' : undefined}>
                  <td>
                    <StockLink symbol={h.symbol} />
                    {h.blocked_tags?.length > 0 && (
                      <span
                        className="risk-flag risk-flag-blocked"
                        title={`Cluster at or over its soft cap: ${h.blocked_tags.join(', ')} — closed to new money`}
                      >
                        capped
                      </span>
                    )}
                  </td>
                  <td className="risk-name-cell">{h.name}</td>
                  <td className={`num ${volDeltaClass(h.next_1k_vol_delta_bp)}`}>
                    {h.next_1k_vol_delta_bp.toFixed(2)}
                  </td>
                  <td className={`num ${screenerClass(h.screener_score)}`}>
                    {scoreCell(h.screener_score)}
                  </td>
                  <td className="num">{formatPct(h.weight)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      <div className="risk-section">
        <h2>Holdings detail</h2>
        <table className="risk-table">
          <thead>
            <tr>
              <SortableTh label="Symbol" colKey="symbol" sort={detailSort} setSort={setDetailSort} defaultDir="asc" />
              <SortableTh label="Name" colKey="name" sort={detailSort} setSort={setDetailSort} defaultDir="asc" />
              <SortableTh label="Value" colKey="value_gbp" sort={detailSort} setSort={setDetailSort} className="num" />
              <SortableTh label="Weight" colKey="weight" sort={detailSort} setSort={setDetailSort} className="num" />
              <SortableTh label="Risk contrib." colKey="risk_contribution" sort={detailSort} setSort={setDetailSort} className="num" />
              <SortableTh label="Risk − weight" colKey="rc_minus_weight" sort={detailSort} setSort={setDetailSort} className="num" />
              <SortableTh label="+£1k Δvol (bp)" colKey="next_1k_vol_delta_bp" sort={detailSort} setSort={setDetailSort} defaultDir="asc" className="num" />
              <SortableTh label="Screener" colKey="screener_score" sort={detailSort} setSort={setDetailSort} className="num" />
            </tr>
          </thead>
          <tbody>
            {sortRows(rows, detailSort).map((h) => (
              <tr key={h.symbol}>
                <td>
                  <StockLink symbol={h.symbol} />
                  {isRiskHeavy(h) && (
                    <span className="risk-flag" title="Risk contribution well above value weight">
                      risk-heavy
                    </span>
                  )}
                </td>
                <td className="risk-name-cell">{h.name}</td>
                <td className="num">{maskable(h.value_gbp)}</td>
                <td className="num">{formatPct(h.weight)}</td>
                <td className="num">{formatPct(h.risk_contribution)}</td>
                <td className={`num ${rcDeltaClass(h.rc_minus_weight)}`}>
                  {formatPp(h.rc_minus_weight)}
                </td>
                <td className={`num ${volDeltaClass(h.next_1k_vol_delta_bp)}`}>
                  {h.next_1k_vol_delta_bp.toFixed(2)}
                </td>
                <td className={`num ${screenerClass(h.screener_score)}`}>
                  {scoreCell(h.screener_score)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {excluded.length > 0 && (
          <p className="risk-excluded">
            Excluded from the model:{' '}
            {excluded.map((e) => `${e.symbol} (${e.reason})`).join('; ')}. Their value is still
            counted in cluster value weights.
          </p>
        )}
      </div>

      <details className="risk-section risk-hrp-details">
        <summary>
          <h2>Reference: Hierarchical Risk Parity allocation</h2>
          <span className="risk-section-note">
            What a pure risk-balancing algorithm would hold, given only the covariance of your
            positions. Context, not a recommendation — it knows nothing about quality or growth.
          </span>
        </summary>
        <table className="risk-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Name</th>
              <th className="num">Your weight</th>
              <th className="num">HRP weight</th>
              <th className="num">Difference</th>
            </tr>
          </thead>
          <tbody>
            {[...rows]
              .sort((a, b) => b.hrp_weight - a.hrp_weight)
              .map((h) => (
                <tr key={h.symbol}>
                  <td>
                    <StockLink symbol={h.symbol} />
                  </td>
                  <td className="risk-name-cell">{h.name}</td>
                  <td className="num">{formatPct(h.weight)}</td>
                  <td className="num">{formatPct(h.hrp_weight)}</td>
                  <td className="num">{formatPp(h.weight - h.hrp_weight)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </details>
    </div>
  );
};

export default Risk;
