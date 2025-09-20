import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { portfolioAPI } from '../services/api';
import SharedTooltip from './SharedTooltip';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import './Stock.css';

// Constants
const CHART_DAYS_OPTIONS = [
  { value: 30, label: '30d' },
  { value: 90, label: '90d' },
  { value: 180, label: '6m' },
  { value: 365, label: '1y' },
  { value: 1827, label: '5y' },
  { value: 3652, label: '10y' }
];

const PRICE_METRICS = [
  { value: 'price', label: 'Price' },
  { value: 'price_pct_change', label: 'Price % Change' }
];

// Utility functions
const formatDate = (dateString) => {
  return new Date(dateString).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: '2-digit'
  });
};

const formatShort = (n) => {
  if (n === null || n === undefined || isNaN(n)) return '-';
  const abs = Math.abs(n);
  if (abs >= 1e12) return (n / 1e12).toFixed(2) + 'T';
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return (n / 1e3).toFixed(2) + 'K';
  return String(n);
};

const formatRatio = (n) => (n === null || n === undefined || isNaN(n) ? '-' : Number(n).toFixed(2));

const Stock = () => {
  const { symbol } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showFullSummary, setShowFullSummary] = useState(false);
  const [chartDays, setChartDays] = useState(() =>
    Number(localStorage.getItem('stock_chart_days') || 365)
  );
  const [priceMetric, setPriceMetric] = useState(() =>
    localStorage.getItem('stock_price_metric') || 'price'
  );

  // Event handlers
  const handlePriceMetricChange = (e) => {
    const value = e.target.value;
    setPriceMetric(value);
    localStorage.setItem('stock_price_metric', value);
  };

  const handleChartDaysChange = (e) => {
    const value = Number(e.target.value);
    setChartDays(value);
    localStorage.setItem('stock_chart_days', String(value));
  };

  const toggleSummary = () => {
    setShowFullSummary(prev => !prev);
  };

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const res = await portfolioAPI.getInstrument(symbol, chartDays);
        setData(res);
      } catch (e) {
        console.error(e);
        setError('Failed to load instrument');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [symbol, chartDays]);

  // Process splits data
  const splitsData = useMemo(() => {
    const splits = data?.splits || {};
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'];
    return Object.entries(splits)
      .map(([date, ratio], index) => ({
        originalDate: date,
        date: formatDate(date),
        ratio: ratio,
        label: `${ratio}:1 Split`,
        color: colors[index % colors.length]
      }))
      .sort((a, b) => new Date(a.originalDate) - new Date(b.originalDate));
  }, [data?.splits]);

  const epsSeries = useMemo(() => {
    const earnings = data?.earnings || {};
    return Object.entries(earnings)
      .map(([date, v]) => ({
        date,
        eps_est: v['EPS Estimate'],
        eps_rep: v['Reported EPS'],
        surprise: v['Surprise(%)'],
      }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }, [data]);

  const cashflowSeries = useMemo(() => {
    const cf = data?.cashflow || {};
    return Object.entries(cf)
      .map(([date, v]) => ({
        date,
        ocf: v['Operating Cash Flow'],
        capex: v['Capital Expenditure'] ?? v['Purchase Of PPE'],
        fcf: v['Free Cash Flow'],
      }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }, [data]);

  const recommendationsSeries = useMemo(() => {
    const recs = data?.recommendations || {};
    return Object.entries(recs)
      .map(([date, v]) => ({
        date: formatDate(date),
        strongBuy: v.strongBuy || 0,
        buy: v.buy || 0,
        hold: v.hold || 0,
        sell: v.sell || 0,
        strongSell: v.strongSell || 0,
        total: (v.strongBuy || 0) + (v.buy || 0) + (v.hold || 0) + (v.sell || 0) + (v.strongSell || 0)
      }))
      .sort((a, b) => new Date(a.date) - new Date(b.date));
  }, [data]);

  // Process price data from the new API response
  const priceData = useMemo(() => {
    const prices = data?.prices || {};
    const entries = Object.entries(prices).map(([date, value]) => ({ date, value }));

    if (priceMetric === 'price_pct_change') {
      const sorted = [...entries].sort((a, b) => new Date(a.date) - new Date(b.date));
      if (sorted.length === 0) return [];
      const base = sorted[0].value;
      return sorted.map(p => ({
        originalDate: p.date,
        date: formatDate(p.date),
        value: base > 0 ? ((p.value - base) / base) * 100 : 0
      }));
    } else {
      return entries.map(p => ({
        originalDate: p.date,
        date: formatDate(p.date),
        value: p.value
      }));
    }
  }, [data?.prices, priceMetric]);

  // Process PE history data from the new API response
  const peData = useMemo(() => {
    const peHistory = data?.pe_history || {};
    return Object.entries(peHistory)
      .map(([date, pe]) => ({
        originalDate: date,
        date: formatDate(date),
        pe: pe
      }))
      .sort((a, b) => new Date(a.originalDate) - new Date(b.originalDate));
  }, [data?.pe_history]);

  // Calculate smart Y-axis domain for PE chart
  const peDomain = useMemo(() => {
    if (peData.length === 0) return [0, 100];

    const peValues = peData.map(d => d.pe).filter(v => v != null && !isNaN(v));
    if (peValues.length === 0) return [0, 100];

    const minPe = Math.min(...peValues);
    const maxPe = Math.max(...peValues);
    const currentPe = peValues[peValues.length - 1]; // Most recent PE

    // Calculate padding as percentage of range
    const range = maxPe - minPe;
    const padding = Math.max(range * 0.1, 5); // 10% padding, minimum 5 points

    // Ensure current PE is always visible with some padding
    const domainMin = Math.max(0, Math.min(minPe - padding, currentPe - padding * 2));
    const domainMax = Math.max(maxPe + padding, currentPe + padding * 2);

    // Round to nice numbers for better readability
    const niceMin = Math.floor(domainMin / 5) * 5;
    const niceMax = Math.ceil(domainMax / 5) * 5;

    return [niceMin, niceMax];
  }, [peData]);

  if (loading) return <div className="stock-container">Loading...</div>;
  if (error) return <div className="stock-container error">{error}</div>;
  if (!data) return null;

  const i = data.instrument;
  const f = data.fundamentals || {};

  const kpis = [
    { label: 'Market Cap', value: formatShort(f.marketCap) },
    { label: 'PE', value: f.peRatio ?? '-' },
    { label: 'PEG', value: f.pegRatio ?? '-' },
    { label: 'Dividend', value: f.dividendYield ? `${f.dividendYield.toFixed(2)}%` : '-' },
    { label: 'Beta', value: f.beta ?? '-' },
    { label: 'Debt', value: formatShort(f.totalDebt) },
    { label: 'Cash', value: formatShort(f.totalCash) },
    { label: 'FCF Yield', value: (f.freeCashflow && f.marketCap && f.marketCap > 0)
      ? `${((f.freeCashflow / f.marketCap) * 100).toFixed(2)}%` : '-' },
  ];

  const valuation = [
    { label: 'EV', value: formatShort(f.enterpriseValue) },
    { label: 'EV/EBITDA', value: formatRatio(f.enterpriseToEbitda) },
    { label: 'EV/Sales', value: formatRatio(f.enterpriseToRevenue) },
    { label: 'P/S (TTM)', value: formatRatio(f.priceToSalesTtm) },
    { label: 'P/B', value: formatRatio(f.priceToBook) },
    { label: 'Net Debt', value: (f.totalDebt !== undefined && f.totalCash !== undefined)
      ? formatShort((f.totalDebt || 0) - (f.totalCash || 0)) : '-' },
    { label: 'Net Debt/EBITDA', value: (f.ebitda && f.ebitda !== 0 &&
      (f.totalDebt !== undefined && f.totalCash !== undefined))
      ? formatRatio(((f.totalDebt || 0) - (f.totalCash || 0)) / f.ebitda) : '-' },
    { label: 'Price/FCF', value: (f.marketCap && f.freeCashflow)
      ? formatRatio(f.marketCap / f.freeCashflow) : '-' },
  ];

  return (
    <div className="stock-container">
      <div className="stock-header">
        <h2>{i.symbol} — {i.name}</h2>
        <div className="stock-kpis">
          {kpis.map(k => (
            <div key={k.label} className="kpi"><div className="kpi-label">{k.label}</div><div className="kpi-value">{k.value}</div></div>
          ))}
        </div>
        <div className="stock-meta"><span>{i.sector || '-'}</span><span>· {i.country || '-'}</span><span>· {i.currency}</span></div>
      </div>

      {i.business_summary && (
        <div className="stock-summary">
          <h3>Business Summary</h3>
          <p className={!showFullSummary ? 'clamped' : ''}>{i.business_summary}</p>
          <button className="summary-toggle" onClick={toggleSummary}>
            {showFullSummary ? 'Show less' : 'Show more'}
          </button>
        </div>
      )}

      <div className="stock-panels">
        <div className="panel">
          <h3>Valuation Multiples</h3>
          <div className="valuation-grid">
            {valuation.map(v => (
              <div key={v.label} className="kpi">
                <div className="kpi-label">{v.label}</div>
                <div className="kpi-value">{v.value}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="panel">
          <h3>Price {priceMetric === 'price_pct_change' ? '(%)' : ''}</h3>
          <div className="inline-controls">
            <label>Metric: </label>
            <select aria-label="Price metric" value={priceMetric} onChange={handlePriceMetricChange}>
              {PRICE_METRICS.map(metric => (
                <option key={metric.value} value={metric.value}>{metric.label}</option>
              ))}
            </select>
            <label style={{ marginLeft: 8 }}>Range: </label>
            <select aria-label="Chart range" value={chartDays} onChange={handleChartDaysChange}>
              {CHART_DAYS_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
          {priceData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={priceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis
                  tickFormatter={(v) =>
                    priceMetric === 'price_pct_change'
                      ? `${v.toFixed?.(1)}%`
                      : (typeof v === 'number'
                          ? v.toLocaleString(undefined, { maximumFractionDigits: 2 })
                          : v)
                  }
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      const displayName = priceMetric === 'price_pct_change' ? 'Price %' : 'Price';
                      const formattedValue = priceMetric === 'price_pct_change'
                        ? `${data.value.toFixed(2)}%`
                        : (typeof data.value === 'number'
                            ? data.value.toLocaleString(undefined, { maximumFractionDigits: 2 })
                            : data.value);

                      return (
                        <div className="custom-tooltip">
                          <p className="tooltip-label">{label}</p>
                          <p className="tooltip-item" style={{ color: '#6f42c1' }}>
                            <span className="color-indicator" style={{ backgroundColor: '#6f42c1' }}></span>
                            {displayName}: {formattedValue}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="value"
                  name={priceMetric === 'price_pct_change' ? 'Price %' : 'Price'}
                  stroke="#6f42c1"
                  strokeWidth={2}
                  dot={false}
                  activeDot={false}
                />
                {/* Add split markers as ReferenceLines */}
                {splitsData.map((split, index) => (
                  <ReferenceLine
                    key={`price-split-${index}`}
                    x={split.date}
                    stroke={split.color}
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    label={{
                      value: `${split.ratio}:1`,
                      position: "top",
                      style: { fontSize: 10, fill: split.color }
                    }}
                  />
                ))}
                {/* Add invisible lines for legend */}
                {splitsData.map((split, index) => (
                  <Line
                    key={`price-legend-${index}`}
                    type="monotone"
                    dataKey={() => null}
                    data={[]}
                    stroke={split.color}
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    activeDot={false}
                    name={`${split.ratio}:1 Split`}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : <div className="empty">No price data</div>}
        </div>

        <div className="panel">
          <h3>PE Ratio</h3>
          <div className="inline-controls">
            <label>Range: </label>
            <select aria-label="PE chart range" value={chartDays} onChange={handleChartDaysChange}>
              {CHART_DAYS_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
          {peData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={peData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis
                  domain={peDomain}
                  tickFormatter={(v) => typeof v === 'number' ? v.toFixed(1) : v}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;

                      return (
                        <div className="custom-tooltip">
                          <p className="tooltip-label">{label}</p>
                          <p className="tooltip-item" style={{ color: '#e74c3c' }}>
                            <span className="color-indicator" style={{ backgroundColor: '#e74c3c' }}></span>
                            PE Ratio: {typeof data.pe === 'number' ? data.pe.toFixed(2) : data.pe}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="pe"
                  name="PE Ratio"
                  stroke="#e74c3c"
                  strokeWidth={2}
                  dot={false}
                  activeDot={false}
                />
                {/* Add split markers as ReferenceLines */}
                {splitsData.map((split, index) => (
                  <ReferenceLine
                    key={`pe-split-${index}`}
                    x={split.date}
                    stroke={split.color}
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    label={{
                      value: `${split.ratio}:1`,
                      position: "top",
                      style: { fontSize: 10, fill: split.color }
                    }}
                  />
                ))}
                {/* Add invisible lines for legend */}
                {splitsData.map((split, index) => (
                  <Line
                    key={`pe-legend-${index}`}
                    type="monotone"
                    dataKey={() => null}
                    data={[]}
                    stroke={split.color}
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    activeDot={false}
                    name={`${split.ratio}:1 Split`}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : <div className="empty">No PE data</div>}
        </div>

        <div className="panel">
          <h3>EPS: Estimate vs Reported (with Surprise%)</h3>
          {epsSeries.length > 0 ? (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={epsSeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis yAxisId="left" />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tickFormatter={(v) => `${v?.toFixed?.(0)}%`}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="custom-tooltip">
                          <p className="tooltip-label">{label}</p>
                          {payload.map((entry, index) => (
                            <p key={index} className="tooltip-item" style={{ color: entry.color }}>
                              <span className="color-indicator" style={{ backgroundColor: entry.color }}></span>
                              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
                            </p>
                          ))}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend />
                <Bar yAxisId="left" dataKey="eps_est" name="EPS Estimate" fill="#adb5bd" />
                <Bar yAxisId="left" dataKey="eps_rep" name="Reported EPS" fill="#2a9d8f" />
                <Line yAxisId="right" type="monotone" dataKey="surprise" name="Surprise %" stroke="#f4a261" strokeWidth={3} dot={false} />
                <ReferenceLine yAxisId="right" y={0} stroke="#ccc" />
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="empty">No earnings data</div>}
        </div>

        <div className="panel">
          <h3>Cash Flow (OCF, CapEx, FCF)</h3>
          {cashflowSeries.length > 0 ? (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={cashflowSeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis
                  tickFormatter={(v) =>
                    typeof v === 'number' ? (v / 1e9).toFixed(1) + 'B' : v
                  }
                />
                <Tooltip
                  content={
                    <SharedTooltip
                      valueFormatter={(v) =>
                        typeof v === 'number' ? (v / 1e9).toFixed(2) + 'B' : v
                      }
                    />
                  }
                />
                <Legend />
                <ReferenceLine y={0} stroke="#ccc" />
                <Bar dataKey="ocf" name="Operating CF" fill="#0d6efd" />
                <Bar dataKey="capex" name="CapEx" fill="#ff7f0e" />
                <Line type="monotone" dataKey="fcf" name="Free CF" stroke="#28a745" strokeWidth={3} dot={false} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="empty">No cash flow data</div>}
        </div>

        <div className="panel">
          <h3>Analyst Recommendations</h3>
          {recommendationsSeries.length > 0 ? (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={recommendationsSeries} stackOffset="expand">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  content={<SharedTooltip
                    valueFormatter={(v) => `${v} analysts`}
                    nameMap={{
                      strongBuy: 'Strong Buy',
                      buy: 'Buy',
                      hold: 'Hold',
                      sell: 'Sell',
                      strongSell: 'Strong Sell'
                    }}
                    sortOrder={['strongBuy', 'buy', 'hold', 'sell', 'strongSell']}
                  />}
                />
                <Legend />
                <Bar dataKey="strongSell" name="Strong Sell" stackId="a" fill="#dc3545" />
                <Bar dataKey="sell" name="Sell" stackId="a" fill="#fd7e14" />
                <Bar dataKey="hold" name="Hold" stackId="a" fill="#ffc107" />
                <Bar dataKey="buy" name="Buy" stackId="a" fill="#20c997" />
                <Bar dataKey="strongBuy" name="Strong Buy" stackId="a" fill="#28a745" />
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="empty">No recommendations data</div>}
        </div>
      </div>
    </div>
  );
};

export default Stock;


