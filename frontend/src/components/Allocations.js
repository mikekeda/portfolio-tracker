import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { portfolioAPI } from '../services/api';
import { renderCountryWithFlag } from '../utils/countryUtils';
import SharedTooltip from './SharedTooltip';
import './Allocations.css';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Group holdings by a key (e.g. 'sector', 'country') and aggregate P&L. */
function groupPerformance(holdings, key) {
  const groups = {};
  for (const h of holdings) {
    const name = h[key];
    if (!name) continue;
    if (!groups[name]) groups[name] = { value: 0, profit: 0, fxPnl: 0, count: 0 };
    groups[name].value  += h.market_value || 0;
    groups[name].profit += h.profit       || 0;
    groups[name].fxPnl  += h.fx_ppl       || 0;
    groups[name].count  += 1;
  }
  const result = {};
  for (const [name, g] of Object.entries(groups)) {
    const cost = g.value - g.profit;
    result[name] = {
      pnl:       g.profit,
      fxPnl:     g.fxPnl,
      returnPct: cost > 0 ? (g.profit / cost) * 100 : null,
      count:     g.count,
    };
  }
  return result;
}

const fmtPct  = (v)  => v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;
const fmtPnl  = (v)  => v == null ? '—' : `${v >= 0 ? '+' : ''}£${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
const fmtGBP  = (v)  => v == null ? '—' : `£${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

// ── Subcomponents ─────────────────────────────────────────────────────────────

const SummaryCard = ({ label, name, returnPct, pnl }) => {
  const positive = returnPct == null || returnPct >= 0;
  return (
    <div className={`alloc-summary-card ${positive ? 'positive' : 'negative'}`}>
      <div className="alloc-summary-label">{label}</div>
      <div className="alloc-summary-name">{name ?? '—'}</div>
      <div className="alloc-summary-return">{fmtPct(returnPct)}</div>
      {pnl != null && <div className="alloc-summary-pnl">{fmtPnl(pnl)}</div>}
    </div>
  );
};

const SortToggle = ({ value, onChange }) => (
  <div className="alloc-sort-toggle">
    <span className="alloc-sort-label">Sort:</span>
    <button
      className={`alloc-sort-btn ${value === 'alloc' ? 'active' : ''}`}
      onClick={() => onChange('alloc')}
    >
      Allocation
    </button>
    <button
      className={`alloc-sort-btn ${value === 'return' ? 'active' : ''}`}
      onClick={() => onChange('return')}
    >
      Return
    </button>
  </div>
);

/** Floating popover listing holdings within a sector or country group. */
const HoldingsPopover = ({ popover, onMouseEnter, onMouseLeave }) => {
  if (!popover) return null;
  return (
    <div
      className="alloc-popover"
      style={{ top: popover.top, left: popover.left }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="alloc-popover-title">{popover.groupName}</div>
      <table className="alloc-popover-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Return</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {popover.items.map(h => {
            const ret = h.return_pct ?? null;
            return (
              <tr key={h.yahoo_symbol || h.t212_code}>
                <td className="pop-symbol">{h.yahoo_symbol}</td>
                <td className={ret == null ? '' : ret >= 0 ? 'color-green' : 'color-red'}>
                  {fmtPct(ret)}
                </td>
                <td>{fmtGBP(h.market_value)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

/** A row for sector/country performance tables. */
const PerfRow = ({ name, renderName, allocPct, returnPct, pnl, count, maxAlloc, maxReturn, onCountEnter, onCountLeave }) => {
  const positive = returnPct == null || returnPct >= 0;
  const allocBarPct = Math.min((allocPct / maxAlloc) * 100, 100);
  const returnBarPct = returnPct == null ? 0 : Math.min(Math.abs(returnPct) / maxReturn * 100, 100);

  return (
    <tr>
      <td className="perf-name-cell">{renderName ? renderName(name) : name}</td>
      <td className="perf-count-cell">
        <span
          className={`count-badge${count > 0 ? ' count-badge-hover' : ''}`}
          onMouseEnter={count > 0 ? onCountEnter : undefined}
          onMouseLeave={count > 0 ? onCountLeave : undefined}
        >
          {count}
        </span>
      </td>
      <td className="perf-alloc-cell">
        <div className="perf-bar-wrap">
          <div className="perf-alloc-bar" style={{ width: `${allocBarPct}%` }} />
          <span className="perf-bar-label">{allocPct.toFixed(1)}%</span>
        </div>
      </td>
      <td className="perf-return-cell">
        <div className="perf-bar-wrap">
          <div
            className={`perf-return-bar ${positive ? 'pos' : 'neg'}`}
            style={{ width: `${returnBarPct}%` }}
          />
          <span className={`perf-bar-label ${positive ? 'color-green' : 'color-red'}`}>
            {fmtPct(returnPct)}
          </span>
        </div>
      </td>
      <td className={`perf-pnl-cell ${positive ? 'color-green' : 'color-red'}`}>
        {fmtPnl(pnl)}
      </td>
    </tr>
  );
};

// ── Main component ────────────────────────────────────────────────────────────

const TIME_RANGES = [
  { value: '1m',  label: '1M',  days: 30  },
  { value: '3m',  label: '3M',  days: 90  },
  { value: '6m',  label: '6M',  days: 180 },
  { value: '1y',  label: '1Y',  days: 365 },
  { value: 'all', label: 'All', days: null },
];

const CHART_COLORS = [
  '#6366f1', '#f59e0b', '#3b82f6', '#ec4899', '#10b981',
  '#8b5cf6', '#14b8a6', '#f97316', '#84cc16', '#06b6d4',
  '#a3a3a3', '#ef4444',
];

const Allocations = () => {
  const [chartData,        setChartData]        = useState(null);
  const [allocations,      setAllocations]      = useState(null);
  const [holdings,         setHoldings]         = useState([]);
  const [loading,          setLoading]          = useState(true);
  const [error,            setError]            = useState(null);
  const [timeRange,        setTimeRange]        = useState('1m');
  const [sectorSort,       setSectorSort]       = useState('alloc');
  const [countrySort,      setCountrySort]      = useState('alloc');
  const [showAllCountries, setShowAllCountries] = useState(false);
  const [popover,          setPopover]          = useState(null);
  const [chartLoading,     setChartLoading]     = useState(false);
  const popoverTimer = useRef(null);

  useEffect(() => () => clearTimeout(popoverTimer.current), []);

  // ── Data fetching ───────────────────────────────────────────────────────────

  // Fetch static data once on mount (allocations + holdings)
  useEffect(() => {
    const fetchStaticData = async () => {
      try {
        setLoading(true);
        const [allocationsData, holdingsData] = await Promise.all([
          portfolioAPI.getAllocations(),
          portfolioAPI.getCurrentHoldings(false),
        ]);
        setAllocations(allocationsData);
        setHoldings(holdingsData.holdings || []);
        setError(null);
      } catch (err) {
        setError('Failed to fetch allocation data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchStaticData();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-fetch only chart/history data when time range changes
  useEffect(() => {
    const fetchChartData = async () => {
      try {
        setChartLoading(true);
        const selected = TIME_RANGES.find(r => r.value === timeRange) || TIME_RANGES[0];
        const historyData = await portfolioAPI.getHistory(selected.days);
        if (historyData.history?.length > 0) {
          const processedData = historyData.history
            .map(item => {
              const out = {
                date:     new Date(item.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                fullDate: item.date,
              };
              ['country_allocation', 'sector_allocation', 'currency_allocation', 'etf_equity_split'].forEach(type => {
                const prefix = type === 'etf_equity_split' ? 'etf_' : `${type.split('_')[0]}_`;
                Object.entries(item[type] || {}).forEach(([k, v]) => { out[`${prefix}${k}`] = v; });
              });
              return out;
            })
            .sort((a, b) => new Date(a.fullDate) - new Date(b.fullDate));
          setChartData(processedData);
        }
      } catch (err) {
        console.error('Failed to fetch chart data:', err);
      } finally {
        setChartLoading(false);
      }
    };
    fetchChartData();
  }, [timeRange]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Performance aggregations ────────────────────────────────────────────────

  const sectorPerf  = useMemo(() => groupPerformance(holdings, 'sector'),  [holdings]);
  const countryPerf = useMemo(() => groupPerformance(holdings, 'country'), [holdings]);

  const currencyFxPnl = useMemo(() => {
    const groups = {};
    for (const h of holdings) {
      if (!h.currency) continue;
      groups[h.currency] = (groups[h.currency] || 0) + (h.fx_ppl || 0);
    }
    return groups;
  }, [holdings]);

  // Best / worst per dimension
  const summaryStats = useMemo(() => {
    const pick = (entries, fn) => entries.length ? entries.reduce(fn) : null;
    const gt   = (a, b) => b[1].returnPct > a[1].returnPct ? b : a;
    const lt   = (a, b) => b[1].returnPct < a[1].returnPct ? b : a;

    const se = Object.entries(sectorPerf ).filter(([, g]) => g.returnPct != null);
    const ce = Object.entries(countryPerf).filter(([, g]) => g.returnPct != null);

    return {
      bestSector:   pick(se, gt),
      worstSector:  pick(se, lt),
      bestCountry:  pick(ce, gt),
      worstCountry: pick(ce, lt),
    };
  }, [sectorPerf, countryPerf]);

  // ── Sorted table rows ───────────────────────────────────────────────────────

  const sectorRows = useMemo(() => {
    if (!allocations?.sector_allocation) return [];
    return Object.entries(allocations.sector_allocation)
      .map(([sector, allocPct]) => ({
        name: sector, allocPct,
        ...sectorPerf[sector],
      }))
      .sort((a, b) => sectorSort === 'return'
        ? (b.returnPct ?? -Infinity) - (a.returnPct ?? -Infinity)
        : b.allocPct - a.allocPct);
  }, [allocations, sectorPerf, sectorSort]);

  const countryRows = useMemo(() => {
    if (!allocations?.country_allocation) return [];
    return Object.entries(allocations.country_allocation)
      .map(([country, allocPct]) => ({
        name: country, allocPct,
        ...countryPerf[country],
      }))
      .sort((a, b) => countrySort === 'return'
        ? (b.returnPct ?? -Infinity) - (a.returnPct ?? -Infinity)
        : b.allocPct - a.allocPct);
  }, [allocations, countryPerf, countrySort]);

  // ── Chart helpers ───────────────────────────────────────────────────────────

  const getTopKeys = (prefix, limit = 8) => {
    if (!chartData?.length) return [];
    const latest = chartData[chartData.length - 1];
    return Object.entries(latest)
      .filter(([k, v]) => k.startsWith(prefix) && typeof v === 'number' && !isNaN(v))
      .sort(([, a], [, b]) => b - a)
      .slice(0, limit)
      .map(([k]) => k.replace(prefix, ''));
  };

  const topSectors    = getTopKeys('sector_');
  const topCountries  = getTopKeys('country_');
  const topCurrencies = getTopKeys('currency_');
  const etfKeys       = getTopKeys('etf_');

  // ── Popover handlers ────────────────────────────────────────────────────────

  const showPopover = (e, groupName, groupKey) => {
    clearTimeout(popoverTimer.current);
    const items = holdings
      .filter(h => h[groupKey] === groupName)
      .sort((a, b) => (b.market_value || 0) - (a.market_value || 0));
    if (!items.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const popoverWidth = 260;
    const left = rect.right + 8 + popoverWidth > window.innerWidth
      ? rect.left - popoverWidth - 8
      : rect.right + 8;
    setPopover({ groupName, items, top: rect.top, left });
  };

  const scheduleHidePopover = () => {
    popoverTimer.current = setTimeout(() => setPopover(null), 150);
  };

  const cancelHidePopover = () => {
    clearTimeout(popoverTimer.current);
  };

  // ── Guard states ────────────────────────────────────────────────────────────

  if (loading) return <div className="allocations-container"><div className="loading">Loading…</div></div>;
  if (error)   return <div className="allocations-container"><div className="error">{error}</div></div>;
  if (!allocations) return <div className="allocations-container"><div className="error">No allocation data available</div></div>;

  const maxSectorAlloc  = Math.max(...sectorRows.map(r => r.allocPct),  1);
  const maxCountryAlloc = Math.max(...countryRows.map(r => r.allocPct), 1);
  const maxSectorReturn  = Math.max(...sectorRows.map(r => Math.abs(r.returnPct  || 0)), 10);
  const maxCountryReturn = Math.max(...countryRows.map(r => Math.abs(r.returnPct || 0)), 10);

  const displayedCountryRows = showAllCountries
    ? countryRows
    : countryRows.filter(r => (r.count || 0) > 0);

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="allocations-container">
      <h2 className="alloc-page-title">Portfolio Allocations</h2>

      {/* ── Summary cards ──────────────────────────────────────────────────── */}
      <div className="alloc-summary-cards">
        <SummaryCard
          label="Best Sector"
          name={summaryStats.bestSector?.[0]}
          returnPct={summaryStats.bestSector?.[1]?.returnPct}
          pnl={summaryStats.bestSector?.[1]?.pnl}
        />
        <SummaryCard
          label="Weakest Sector"
          name={summaryStats.worstSector?.[0]}
          returnPct={summaryStats.worstSector?.[1]?.returnPct}
          pnl={summaryStats.worstSector?.[1]?.pnl}
        />
        <SummaryCard
          label="Best Country"
          name={summaryStats.bestCountry?.[0]}
          returnPct={summaryStats.bestCountry?.[1]?.returnPct}
          pnl={summaryStats.bestCountry?.[1]?.pnl}
        />
        <SummaryCard
          label="Weakest Country"
          name={summaryStats.worstCountry?.[0]}
          returnPct={summaryStats.worstCountry?.[1]?.returnPct}
          pnl={summaryStats.worstCountry?.[1]?.pnl}
        />
      </div>

      {/* ── Sector + Country performance tables ────────────────────────────── */}
      <div className="alloc-perf-section">

        {/* Sector */}
        <div className="alloc-perf-panel">
          <div className="alloc-perf-header">
            <h3>Sector Allocation &amp; Performance</h3>
            <SortToggle value={sectorSort} onChange={setSectorSort} />
          </div>
          <table className="perf-table">
            <thead>
              <tr>
                <th>Sector</th>
                <th title="Number of holdings">Pos</th>
                <th>Allocation</th>
                <th>Return</th>
                <th>P&amp;L (£)</th>
              </tr>
            </thead>
            <tbody>
              {sectorRows.map(row => (
                <PerfRow
                  key={row.name}
                  name={row.name}
                  allocPct={row.allocPct}
                  returnPct={row.returnPct ?? null}
                  pnl={row.pnl ?? null}
                  count={row.count ?? 0}
                  maxAlloc={maxSectorAlloc}
                  maxReturn={maxSectorReturn}
                  onCountEnter={(e) => showPopover(e, row.name, 'sector')}
                  onCountLeave={scheduleHidePopover}
                />
              ))}
            </tbody>
          </table>
        </div>

        {/* Country */}
        <div className="alloc-perf-panel">
          <div className="alloc-perf-header">
            <h3>Country Allocation &amp; Performance</h3>
            <SortToggle value={countrySort} onChange={setCountrySort} />
          </div>
          <table className="perf-table">
            <thead>
              <tr>
                <th>Country</th>
                <th title="Number of holdings">Pos</th>
                <th>Allocation</th>
                <th>Return</th>
                <th>P&amp;L (£)</th>
              </tr>
            </thead>
            <tbody>
              {displayedCountryRows.map(row => (
                <PerfRow
                  key={row.name}
                  name={row.name}
                  renderName={renderCountryWithFlag}
                  allocPct={row.allocPct}
                  returnPct={row.returnPct ?? null}
                  pnl={row.pnl ?? null}
                  count={row.count ?? 0}
                  maxAlloc={maxCountryAlloc}
                  maxReturn={maxCountryReturn}
                  onCountEnter={(e) => showPopover(e, row.name, 'country')}
                  onCountLeave={scheduleHidePopover}
                />
              ))}
            </tbody>
          </table>
          {(() => {
            const zeroCount = countryRows.filter(r => (r.count || 0) === 0).length;
            return zeroCount > 0 && (
              <button
                className="alloc-show-more-btn"
                onClick={() => setShowAllCountries(v => !v)}
              >
                {showAllCountries
                  ? 'Hide zero-position countries'
                  : `Show ${zeroCount} more (ETF domiciles)`}
              </button>
            );
          })()}
        </div>
      </div>

      {/* ── Currency + ETF/Equity ───────────────────────────────────────────── */}
      <div className="alloc-secondary-section">

        <div className="allocation-table">
          <h3>Currency Allocation</h3>
          <table>
            <thead>
              <tr>
                <th>Currency</th>
                <th>Allocation</th>
                <th title="FX contribution to P&L in GBP. Negative = strong GBP hurt returns.">FX P&amp;L</th>
              </tr>
            </thead>
            <tbody>
              {allocations.currency_allocation && Object.entries(allocations.currency_allocation)
                .sort(([, a], [, b]) => b - a)
                .map(([currency, pct]) => {
                  const fx = currencyFxPnl[currency];
                  const isHome = currency === 'GBP';
                  const fxDisplay = isHome ? '—' : (fx != null ? fmtPnl(fx) : '—');
                  const fxPositive = isHome || fx == null || fx >= 0;
                  return (
                    <tr key={currency}>
                      <td>
                        <span className="currency-code">{currency}</span>
                        {currency === 'USD' && ' 🇺🇸'}
                        {currency === 'GBP' && ' 🇬🇧'}
                        {currency === 'EUR' && ' 🇪🇺'}
                        {currency === 'CAD' && ' 🇨🇦'}
                        {currency === 'JPY' && ' 🇯🇵'}
                      </td>
                      <td style={{ '--bar-width': `${Math.min(pct, 100)}%` }}>
                        <span>{pct.toFixed(2)}%</span>
                      </td>
                      <td className={fxPositive ? 'color-green' : 'color-red'}>
                        {fxDisplay}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>

        <div className="allocation-table">
          <h3>ETF / Equity Split</h3>
          <table>
            <thead>
              <tr>
                <th>Type</th>
                <th>Allocation</th>
                <th>Value (£)</th>
              </tr>
            </thead>
            <tbody>
              {allocations.etf_equity_split && Object.entries(allocations.etf_equity_split)
                .sort(([, a], [, b]) => b - a)
                .map(([type, pct]) => (
                  <tr key={type}>
                    <td>{type}</td>
                    <td style={{ '--bar-width': `${Math.min(pct, 100)}%` }}>
                      <span>{pct.toFixed(2)}%</span>
                    </td>
                    <td>{fmtGBP(allocations.total_value * pct / 100)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Time-series charts ──────────────────────────────────────────────── */}
      {chartData?.length > 0 && (
        <div className={`alloc-charts-section${chartLoading ? ' alloc-charts-loading' : ''}`}>
          <div className="alloc-charts-header">
            <h3>Allocation Over Time</h3>
            <div className="time-range-selector">
              {TIME_RANGES.map(r => (
                <button
                  key={r.value}
                  className={`time-range-btn ${timeRange === r.value ? 'active' : ''}`}
                  onClick={() => setTimeRange(r.value)}
                >
                  {r.label}
                </button>
              ))}
            </div>
          </div>

          <div className="allocation-charts">
            {[
              { title: 'Sector',   keys: topSectors,    prefix: 'sector_'   },
              { title: 'Country',  keys: topCountries,  prefix: 'country_'  },
              { title: 'Currency', keys: topCurrencies, prefix: 'currency_' },
              { title: 'ETF vs Equity', keys: etfKeys,  prefix: 'etf_'      },
            ].map(({ title, keys, prefix }) => (
              <div key={title} className="chart-panel allocation-panel">
                <h3>{title}</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                    <Tooltip content={<SharedTooltip prefix={prefix} valueFormatter={v => `${v.toFixed(1)}%`} />} />
                    <Legend />
                    {keys.map((k, i) => (
                      <Line
                        key={k}
                        type="monotone"
                        dataKey={`${prefix}${k}`}
                        name={k}
                        stroke={CHART_COLORS[i % CHART_COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </div>
      )}

      <HoldingsPopover
        popover={popover}
        onMouseEnter={cancelHidePopover}
        onMouseLeave={scheduleHidePopover}
      />
    </div>
  );
};

export default Allocations;
