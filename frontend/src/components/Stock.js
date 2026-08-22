import React, {useEffect, useMemo, useRef, useState} from 'react';
import PropTypes from 'prop-types';
import {Link, useNavigate, useParams} from 'react-router-dom';
import {portfolioAPI} from '../services/api';
import SharedTooltip from './SharedTooltip';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { renderCountryWithFlag } from '../utils/countryUtils';
import { ytdDays } from '../utils/dates';
import { useHideAmounts, MASK } from '../context/HideAmountsContext';
import { getPbThresholds, getPsThresholds } from '../utils/valuationUtils';
import { computeComposite, compositeTooltip, screenerRatio } from '../utils/compositeScore';
import './Stock.css';

// Constants
const CHART_DAYS_OPTIONS = [
  { value: 1, label: '1D' },
  { value: 5, label: '5D' },
  { value: 30, label: '1M' },
  { value: 90, label: '3M' },
  { value: 180, label: '6M' },
  { value: 'ytd', label: 'YTD' },
  { value: 365, label: '1Y' },
  { value: 731, label: '2Y' },
  { value: 1825, label: '5Y' },
  { value: 3652, label: '10Y' }
];

// Quarterly and monthly series need a year of history to read as a trend, so
// shorter chart ranges floor here rather than rendering one or two bars.
const EVENT_SERIES_MIN_DAYS = 365;

// Recharts mounts a component per data point, so a 10Y daily series costs
// ~2700 of them per render. Past this count the series is thinned.
const MAX_CHART_POINTS = 800;

const PRICE_METRICS = [
  { value: 'price', label: 'Price' },
  { value: 'price_pct_change', label: 'Price % Change' }
];

const formatActionName = (action) => {
  return action
    .replace('MARKET_', '')
    .replace('LIMIT_', '')
    .replace('_(DIVIDEND)', '')
    .replace('DIVIDEND', 'Dividend');
};

const SMA_50_COLOR = '#3b82f6';   // blue
const SMA_200_COLOR = '#ef4444';  // red
const BENCH_COLORS = { 'VUAG.L': '#94a3b8', 'XNAS.L': '#0ea5e9' };
const BENCH_FALLBACK_COLOR = '#94a3b8';

const FiftyTwoWeekBar = ({ low, high, current }) => {
  if (low == null || high == null || current == null || high <= low) return null;
  const pct = Math.min(100, Math.max(0, ((current - low) / (high - low)) * 100));
  const fromLowPct = (((current - low) / low) * 100).toFixed(1);
  const fromHighPct = (((current - high) / high) * 100).toFixed(1);
  const fmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
  const signLow = Number(fromLowPct) >= 0 ? '+' : '';
  return (
    <div
      className="ftw-bar-wrap"
      title={`${signLow}${fromLowPct}% from 52W low · ${fromHighPct}% from 52W high`}
    >
      <span className="ftw-bar-bound">{fmt(low)}</span>
      <div className="ftw-bar-track">
        <div className="ftw-bar-fill" style={{ width: `${pct}%` }} />
        <div className="ftw-bar-dot" style={{ left: `${pct}%` }} />
      </div>
      <span className="ftw-bar-bound">{fmt(high)}</span>
    </div>
  );
};

FiftyTwoWeekBar.propTypes = {
  low: PropTypes.number,
  high: PropTypes.number,
  current: PropTypes.number,
};

// Independent fair-value estimates drawn on one price axis, with the current
// price as a tick in every row — overlap means consensus, spread means the
// fair value is contested.
const FairValueBand = ({ rows, current }) => {
  const values = rows
    .flatMap(r => [r.low, r.mid, r.high])
    .concat([current])
    .filter(v => v != null && Number.isFinite(v));
  if (rows.length === 0 || current == null || values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max <= min) return null;
  const pad = (max - min) * 0.06;
  const lo = Math.max(0, min - pad);
  const hi = max + pad;
  const x = (v) => ((v - lo) / (hi - lo)) * 100;
  const fmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });

  return (
    <div className="fv-band">
      {rows.map(r => {
        const mid = r.mid ?? ((r.low != null && r.high != null) ? (r.low + r.high) / 2 : null);
        const diff = mid != null ? (mid / current - 1) * 100 : null;
        return (
          <div key={r.label} className={`fv-row${r.muted ? ' fv-row-muted' : ''}`} title={r.tooltip}>
            <span className="fv-row-label">{r.label}</span>
            <div className="fv-row-track">
              {r.low != null && r.high != null && (
                <div
                  className="fv-row-range"
                  style={{ left: `${x(r.low)}%`, width: `${Math.max(0.5, x(r.high) - x(r.low))}%` }}
                />
              )}
              {mid != null && <div className="fv-row-mid" style={{ left: `${x(mid)}%` }} />}
              <div className="fv-row-price" style={{ left: `${x(current)}%` }} />
            </div>
            <span className={`fv-row-diff ${diff > 0 ? 'positive' : diff < 0 ? 'negative' : ''}`}>
              {diff != null ? `${diff >= 0 ? '+' : ''}${Math.round(diff)}%` : ''}
            </span>
          </div>
        );
      })}
      <div className="fv-row fv-scale-row">
        <span className="fv-row-label" />
        <div className="fv-scale-track">
          <span>{fmt(lo)}</span>
          <span className="fv-scale-price" style={{ left: `${x(current)}%` }}>{fmt(current)}</span>
          <span>{fmt(hi)}</span>
        </div>
        <span className="fv-row-diff" />
      </div>
    </div>
  );
};

FairValueBand.propTypes = {
  rows: PropTypes.arrayOf(PropTypes.shape({
    label: PropTypes.string.isRequired,
    low: PropTypes.number,
    mid: PropTypes.number,
    high: PropTypes.number,
    tooltip: PropTypes.string,
    muted: PropTypes.bool,
  })).isRequired,
  current: PropTypes.number,
};

const PriceTooltip = ({ active, payload, label, priceMetric, benchSymbols }) => {
  if (active && payload && payload.length) {
    const data = payload[0]?.payload || {};
    const displayName = priceMetric === 'price_pct_change' ? 'Price %' : 'Price';
    const formatVal = (v) =>
      priceMetric === 'price_pct_change'
        ? `${v?.toFixed(2)}%`
        : typeof v === 'number'
          ? v.toLocaleString(undefined, { maximumFractionDigits: 2 })
          : v;
    const isOrder = data.order !== undefined;

    return (
      <div className="custom-tooltip">
        <p className="tooltip-label">{label}</p>
        {isOrder && (
          <p className="tooltip-item" style={{ color: data.order.color, fontWeight: 'bold' }}>
            <span className="color-indicator" style={{ backgroundColor: data.order.color }}></span>
            {formatActionName(data.order.action)}: £{data.order.total.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </p>
        )}
        <p className="tooltip-item" style={{ color: '#6f42c1' }}>
          <span className="color-indicator" style={{ backgroundColor: '#6f42c1' }}></span>
          {displayName}: {formatVal(data.value)}
        </p>
        {data.sma50 != null && (
          <p className="tooltip-item" style={{ color: SMA_50_COLOR }}>
            <span className="color-indicator" style={{ backgroundColor: SMA_50_COLOR }}></span>
            SMA 50: {formatVal(data.sma50)}
          </p>
        )}
        {data.sma200 != null && (
          <p className="tooltip-item" style={{ color: SMA_200_COLOR }}>
            <span className="color-indicator" style={{ backgroundColor: SMA_200_COLOR }}></span>
            SMA 200: {formatVal(data.sma200)}
          </p>
        )}
        {(benchSymbols || []).map(sym => data[sym] != null && (
          <p key={sym} className="tooltip-item" style={{ color: BENCH_COLORS[sym] || BENCH_FALLBACK_COLOR }}>
            <span className="color-indicator" style={{ backgroundColor: BENCH_COLORS[sym] || BENCH_FALLBACK_COLOR }}></span>
            {sym}: {formatVal(data[sym])}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

PriceTooltip.propTypes = {
  active: PropTypes.bool,
  payload: PropTypes.arrayOf(
    PropTypes.shape({
      payload: PropTypes.shape({
        value: PropTypes.number,
        sma50: PropTypes.number,
        sma200: PropTypes.number,
        order: PropTypes.shape({
          action: PropTypes.string,
          total: PropTypes.number,
          color: PropTypes.string,
        }),
      }),
    })
  ),
  label: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  priceMetric: PropTypes.string.isRequired,
  benchSymbols: PropTypes.arrayOf(PropTypes.string),
};

PriceTooltip.defaultProps = {
  active: false,
  payload: [],
  label: '',
  benchSymbols: [],
};

// Chart panel subhead: the span actually plotted, plus how the range was applied.
// Renders nothing without a span — a lone note would describe an empty chart.
const PanelSpan = ({ span, note }) =>
  span ? <p className="panel-span">{span}{note && ` · ${note}`}</p> : null;

PanelSpan.propTypes = {
  span: PropTypes.string,
  note: PropTypes.string,
};

PanelSpan.defaultProps = {
  span: null,
  note: null,
};

// Transaction categories for color coding
const TRANSACTION_CATEGORIES = {
  BUY: ['Market buy', 'Limit buy'],
  SELL: ['Market sell', 'Limit sell'],
  DIVIDEND: ['Dividend (Dividend)', 'Dividend (Property income distribution)', 'Dividend (Tax exempted)'],
  CASH: ['Deposit', 'Withdrawal', 'Interest on cash'],
  ADMIN: ['Stock split open', 'Stock split close', 'Result adjustment']
};

// Transaction colors
const TRANSACTION_COLORS = {
  BUY: '#28a745',      // Green
  SELL: '#dc3545',     // Red
  DIVIDEND: '#fd7e14', // Orange
  CASH: '#17a2b8',     // Blue
  ADMIN: '#6c757d'     // Gray
};

// Common dot properties for legends
const LEGEND_DOT_PROPS = {
  r: 8,
  fillOpacity: 0.8,
  stroke: '#fff',
  strokeWidth: 2,
  strokeOpacity: 0.8
};

// Utility functions
const formatDate = (dateString) => {
  return new Date(dateString).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: '2-digit'
  });
};

const formatDateShort = (dateString) => {
  if (!dateString) return '';
  const d = new Date(dateString + 'T12:00:00');
  const day = String(d.getDate()).padStart(2, '0');
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const year = d.getFullYear();
  return `${day}/${month}/${year}`;
};

const getTransactionCategory = (action) => {
  if (TRANSACTION_CATEGORIES.BUY.includes(action)) return 'BUY';
  if (TRANSACTION_CATEGORIES.SELL.includes(action)) return 'SELL';
  if (TRANSACTION_CATEGORIES.DIVIDEND.includes(action)) return 'DIVIDEND';
  if (TRANSACTION_CATEGORIES.CASH.includes(action)) return 'CASH';
  if (TRANSACTION_CATEGORIES.ADMIN.includes(action)) return 'ADMIN';
  return 'ADMIN'; // Unknown actions render neutral, not as sells
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

// First displayed date for a chart range ('ytd' or a day count)
const getRangeCutoff = (chartDays) => {
  if (chartDays === 'ytd') return new Date(new Date().getFullYear(), 0, 1);
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - Number(chartDays));
  return cutoff;
};

// Covered span of a chronologically sorted series, for the panel subheads
const formatSpan = (rows) => {
  if (rows.length === 0) return null;
  const fmt = (r) => new Date(r.originalDate ?? r.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  return `${fmt(rows[0])} – ${fmt(rows[rows.length - 1])}`;
};

/**
 * Thin a dense daily series to roughly MAX_CHART_POINTS for rendering.
 *
 * Keeps each bucket's high and low so the envelope survives, plus every point
 * carrying an order marker. Series at or under the cap are returned untouched.
 */
const thinSeries = (rows) => {
  if (rows.length <= MAX_CHART_POINTS) return rows;
  const stride = Math.ceil(rows.length / Math.floor(MAX_CHART_POINTS / 2));
  const keep = new Set([0, rows.length - 1]);
  for (let i = 0; i < rows.length; i += stride) {
    let lo = i;
    let hi = i;
    for (let j = i; j < Math.min(i + stride, rows.length); j++) {
      if (rows[j].value < rows[lo].value) lo = j;
      if (rows[j].value > rows[hi].value) hi = j;
    }
    keep.add(lo);
    keep.add(hi);
  }
  rows.forEach((r, i) => {
    if (r.order) keep.add(i);
  });
  return [...keep].sort((a, b) => a - b).map((i) => rows[i]);
};

// An empty chart is worse than an out-of-range one, so a window that predates
// every point falls back to the full series.
const withinCutoff = (rows, cutoff) => {
  const kept = rows.filter((r) => new Date(r.originalDate ?? r.date) >= cutoff);
  return kept.length > 0 ? kept : rows;
};


// Convert markdown to HTML (handles headers, bold, lists)
const markdownToHtml = (markdown) => {
  if (!markdown) return '';
  const bold = (s) => s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  const lines = markdown.split('\n');
  const output = [];
  let inList = false;

  for (const line of lines) {
    if (line.startsWith('### ')) {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<h3>${bold(line.slice(4))}</h3>`);
    } else if (line.startsWith('## ')) {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<h2>${bold(line.slice(3))}</h2>`);
    } else if (line.startsWith('# ')) {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<h1>${bold(line.slice(2))}</h1>`);
    } else if (/^[\s]*[-*] /.test(line)) {
      const content = line.replace(/^[\s]*[-*] /, '');
      if (!inList) { output.push('<ul>'); inList = true; }
      output.push(`<li>${bold(content)}</li>`);
    } else if (line.trim()) {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<p>${bold(line)}</p>`);
    } else {
      if (inList) { output.push('</ul>'); inList = false; }
    }
  }
  if (inList) output.push('</ul>');

  return output.join('\n');
};

// ── Symbol switcher ───────────────────────────────────────────────────────────

const SymbolSwitcher = ({ currentSymbol }) => {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [instruments, setInstruments] = useState([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef(null);
  const containerRef = useRef(null);

  // Lazy-load on first open (instruments are stable; no need to re-fetch)
  useEffect(() => {
    if (open && instruments.length === 0) {
      portfolioAPI.getInstruments()
        .then((data) => setInstruments(data.instruments || []))
        .catch((err) => console.error('SymbolSwitcher: failed to load instruments', err));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Focus input when opened
  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handle = (e) => {
      if (!containerRef.current?.contains(e.target)) {
        setOpen(false);
        setQuery('');
      }
    };
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return instruments.slice(0, 8);
    return instruments
      .filter((i) => i.symbol.toLowerCase().includes(q) || i.name.toLowerCase().includes(q))
      .slice(0, 8);
  }, [query, instruments]);

  // Reset keyboard selection to top whenever the search query changes
  useEffect(() => { setActiveIdx(0); }, [query]);

  const handleSelect = (sym) => {
    setOpen(false);
    setQuery('');
    if (sym !== currentSymbol) navigate(`/stock/${encodeURIComponent(sym)}`);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') { setOpen(false); setQuery(''); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); setActiveIdx((i) => Math.min(i + 1, filtered.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActiveIdx((i) => Math.max(i - 1, 0)); }
    else if (e.key === 'Enter' && filtered[activeIdx]) { handleSelect(filtered[activeIdx].symbol); }
  };

  if (!open) {
    return (
      <button className="stock-symbol-btn" onClick={() => setOpen(true)} title="Switch stock">
        {currentSymbol} <span className="stock-symbol-caret">▾</span>
      </button>
    );
  }

  return (
    <div className="stock-switcher" ref={containerRef}>
      <input
        ref={inputRef}
        className="stock-switcher-input"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Search symbol or name…"
      />
      {filtered.length > 0 && (
        <div className="stock-switcher-dropdown">
          {filtered.map((inst, idx) => (
            <button
              key={inst.symbol}
              className={`stock-switcher-option${inst.symbol === currentSymbol ? ' current' : ''}${idx === activeIdx ? ' active' : ''}`}
              onClick={() => handleSelect(inst.symbol)}
              onMouseEnter={() => setActiveIdx(idx)}
            >
              <span className="stock-switcher-sym">{inst.symbol}</span>
              <span className="stock-switcher-name">{inst.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

SymbolSwitcher.propTypes = {
  currentSymbol: PropTypes.string.isRequired,
};

// ─────────────────────────────────────────────────────────────────────────────

// Collapsed by default: 489 rows for VUAG, 1,485 for R2SC.
const ETF_HOLDINGS_COLLAPSED = 25;

const Stock = () => {
  const { symbol } = useParams();
  const { hideAmounts } = useHideAmounts();
  const [data, setData] = useState(null);
  const [etfHoldingsOpen, setEtfHoldingsOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [activeSection, setActiveSection] = useState(null);
  const [showFullSummary, setShowFullSummary] = useState(false);
  const [selectedReport, setSelectedReport] = useState(null);
  const [reportHtml, setReportHtml] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);
  const [activeReportIdx, setActiveReportIdx] = useState(0);
  useEffect(() => { setActiveReportIdx(0); }, [symbol]);
  const [chartDays, setChartDays] = useState(() => {
    const stored = localStorage.getItem('stock_chart_days') || '365';
    return stored === 'ytd' ? 'ytd' : Number(stored);
  });
  const [priceMetric, setPriceMetric] = useState(() =>
    localStorage.getItem('stock_price_metric') || 'price'
  );
  const chartWindowDays = chartDays === 'ytd' ? ytdDays() : Number(chartDays);
  const eventRangeFloored = chartWindowDays < EVENT_SERIES_MIN_DAYS;

  // Event handlers
  const handlePriceMetricChange = (value) => {
    setPriceMetric(value);
    localStorage.setItem('stock_price_metric', value);
  };

  const handleChartDaysChange = (value) => {
    setChartDays(value);
    localStorage.setItem('stock_chart_days', String(value));
  };

  const toggleSummary = () => {
    setShowFullSummary(prev => !prev);
  };

  const openReportModal = async (symbol, reportDate) => {
    setSelectedReport({ symbol, reportDate });
    setLoadingReport(true);
    setReportHtml(null);

    try {
      const response = await portfolioAPI.getEarningsReportHtml(symbol, reportDate);
      setReportHtml(response);
    } catch (error) {
      console.error('Error loading report:', error);
      setReportHtml('<p>Error loading report. Please try again.</p>');
    } finally {
      setLoadingReport(false);
    }
  };

  const closeReportModal = () => {
    setSelectedReport(null);
    setReportHtml(null);
  };

  // Full payload on symbol change. Fetch 300 extra calendar days (~207 trading
  // days) so SMA 200 is warmed up from the very first displayed data point.
  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const daysParam = chartDays === 'ytd' ? ytdDays() : chartDays;
        const res = await portfolioAPI.getInstrument(symbol, Number(daysParam) + 300);
        setData(res);
        setError(null);
      } catch (e) {
        console.error(e);
        setError('Failed to load instrument');
      } finally {
        setLoading(false);
      }
    };
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  // Chart window only on range change — the full payload (13F, DCF, reviews…)
  // doesn't depend on the range, so re-fetching it per click was pure waste.
  useEffect(() => {
    if (!data) return; // initial load in flight; it already covers this range
    const load = async () => {
      try {
        setChartLoading(true);
        const daysParam = chartDays === 'ytd' ? ytdDays() : chartDays;
        const res = await portfolioAPI.getInstrumentPrices(symbol, Number(daysParam) + 300);
        setData(prev => (prev ? { ...prev, ...res } : prev));
      } catch (e) {
        console.error(e);
      } finally {
        setChartLoading(false);
      }
    };
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartDays]);

  // Scroll-spy: highlight the section nav pill for the panel nearest the top.
  // Re-runs per symbol (panel set can change); range merges don't retrigger it.
  useEffect(() => {
    if (loading || !data) return;
    const panels = Array.from(document.querySelectorAll('.stock-panels .panel[id]'));
    if (panels.length === 0) return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter(e => e.isIntersecting);
        if (visible.length > 0) {
          const top = visible.reduce((a, b) =>
            a.boundingClientRect.top < b.boundingClientRect.top ? a : b
          );
          setActiveSection(top.target.id);
        }
      },
      // Active band = below the sticky bars, upper 40% of the viewport
      { rootMargin: '-110px 0px -60% 0px', threshold: 0 }
    );
    panels.forEach(p => observer.observe(p));
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, loading]);

  const handleSectionClick = (e, id) => {
    e.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setActiveSection(id);
  };

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

  // Yahoo has the announcement date as soon as a company reports; our own report
  // row only lands once the SEC filing is ingested, days later.
  const pendingEarningsDate = useMemo(() => {
    const today = new Date().toLocaleDateString('en-CA');
    const announced = Object.keys(data?.earnings || {})
      .map((k) => k.slice(0, 10))
      .filter((d) => d <= today)
      .sort();
    const latest = announced[announced.length - 1];
    if (!latest) return null;
    const covered = (data?.earnings_reports || []).some((r) => (r.announcement_date || r.date) >= latest);
    return covered ? null : latest;
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
        originalDate: date,
        date: formatDate(date),
        strongBuy: v.strongBuy || 0,
        buy: v.buy || 0,
        hold: v.hold || 0,
        sell: v.sell || 0,
        strongSell: v.strongSell || 0,
        total: (v.strongBuy || 0) + (v.buy || 0) + (v.hold || 0) + (v.sell || 0) + (v.strongSell || 0)
      }))
      .sort((a, b) => a.originalDate.localeCompare(b.originalDate));
  }, [data]);

  const newsArticles = useMemo(() => {
    return data?.news || [];
  }, [data?.news]);

  // Process price data from the new API response
  const priceData = useMemo(() => {
    const prices = data?.prices || {};
    // Sort all entries chronologically
    const sorted = Object.entries(prices)
      .map(([date, value]) => ({ date, value }))
      .sort((a, b) => new Date(a.date) - new Date(b.date));

    // Clip by date so the displayed range matches exactly what the user selected.
    // The extra 300 calendar days fetched above are used only for SMA warm-up.
    const cutoff = getRangeCutoff(chartDays);
    const display = sorted.filter(p => new Date(p.date) >= cutoff);

    if (priceMetric === 'price_pct_change') {
      if (display.length === 0) return [];
      const base = display[0].value;
      return display.map(p => ({
        originalDate: p.date,
        date: formatDate(p.date),
        value: base > 0 ? ((p.value - base) / base) * 100 : 0
      }));
    } else {
      return display.map(p => ({
        originalDate: p.date,
        date: formatDate(p.date),
        value: p.value
      }));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data?.prices, priceMetric, chartDays]);

  // Benchmark %-change series for the overlay in % mode, each rebased to its
  // own first value inside the displayed window
  const benchSeries = useMemo(() => {
    if (priceMetric !== 'price_pct_change') return [];
    const benchmarks = data?.benchmarks || {};
    const cutoff = getRangeCutoff(chartDays);
    return Object.entries(benchmarks)
      .map(([sym, series]) => {
        const display = Object.entries(series)
          .map(([date, value]) => ({ date, value }))
          .sort((a, b) => new Date(a.date) - new Date(b.date))
          .filter(p => new Date(p.date) >= cutoff);
        if (display.length === 0) return null;
        const base = display[0].value;
        return {
          sym,
          byDate: new Map(display.map(p => [p.date, base > 0 ? ((p.value - base) / base) * 100 : 0])),
        };
      })
      .filter(Boolean);
  }, [data?.benchmarks, priceMetric, chartDays]);

  // Combine price data with orders, SMAs and benchmark overlays
  const chartData = useMemo(() => {
    const orders = data?.orders || {};

    // Calculate min/max order amounts for scaling
    const orderAmounts = Object.values(orders).map(o => o.total);
    const minAmount = Math.min(...orderAmounts, 0);
    const maxAmount = Math.max(...orderAmounts, 0);
    const amountRange = maxAmount - minAmount;

    // First order per calendar day (multiple same-day orders: keep the first,
    // matching the previous find() semantics)
    const ordersByDate = new Map();
    for (const [timestamp, order] of Object.entries(orders)) {
      const day = timestamp.split('T')[0];
      if (!ordersByDate.has(day)) ordersByDate.set(day, order);
    }

    // Prefix sums over the raw series (display window + SMA warm-up) so each
    // SMA value is O(1) instead of a per-point slice+reduce
    const sortedRaw = Object.entries(data?.prices || {}).sort(([a], [b]) => new Date(a) - new Date(b));
    const idxByDate = new Map(sortedRaw.map(([d], i) => [d, i]));
    const prefix = new Array(sortedRaw.length + 1);
    prefix[0] = 0;
    for (let i = 0; i < sortedRaw.length; i++) prefix[i + 1] = prefix[i] + sortedRaw[i][1];
    const smaAt = (idx, window) =>
      idx + 1 >= window ? (prefix[idx + 1] - prefix[idx + 1 - window]) / window : null;

    // % mode rebases SMAs to the displayed window's first price — the raw
    // series starts 300 warm-up days earlier and its first price is not the
    // base the price line itself uses
    const displayBaseIdx = priceData.length > 0 ? idxByDate.get(priceData[0].originalDate) : undefined;
    const basePrice = displayBaseIdx !== undefined ? sortedRaw[displayBaseIdx][1] : 1;
    const toDisplay = (raw) => {
      if (raw == null) return null;
      return priceMetric === 'price_pct_change' && basePrice > 0
        ? ((raw - basePrice) / basePrice) * 100
        : raw;
    };

    return thinSeries(priceData.map(pricePoint => {
      const point = { ...pricePoint };

      const order = ordersByDate.get(pricePoint.originalDate);
      if (order) {
        const category = getTransactionCategory(order.action);
        const normalizedAmount = amountRange > 0 ? (order.total - minAmount) / amountRange : 0;
        point.order = {
          action: order.action,
          total: order.total,
          category,
          color: TRANSACTION_COLORS[category],
          radius: Math.max(4, 4 + (normalizedAmount * 8)), // 4-12px range
          opacity: Math.max(0.6, 1 - (normalizedAmount * 0.3)), // 0.7-1.0 range
        };
      }

      const idx = idxByDate.get(pricePoint.originalDate);
      if (idx !== undefined) {
        point.sma50 = toDisplay(smaAt(idx, 50));
        point.sma200 = toDisplay(smaAt(idx, 200));
      }

      for (const bench of benchSeries) {
        const v = bench.byDate.get(pricePoint.originalDate);
        if (v !== undefined) point[bench.sym] = v;
      }

      return point;
    }));
  }, [priceData, data?.orders, data?.prices, priceMetric, benchSeries]);

  // The PE, EPS and analyst series ship whole in the payload, so the range is a
  // client-side slice — only the price chart refetches.
  const eventCutoff = useMemo(
    () => getRangeCutoff(eventRangeFloored ? EVENT_SERIES_MIN_DAYS : chartDays),
    [eventRangeFloored, chartDays]
  );

  // Process PE history data from the new API response. Forward estimates come
  // separately and render as a dashed continuation; the last historical point
  // is duplicated into the estimate series so the two lines join up.
  const peData = useMemo(() => {
    const peHistory = data?.pe_history || {};
    const peEstimates = data?.pe_estimates || {};
    // The payload carries 300 extra days for SMA warm-up; PE is quarterly, so
    // it follows the floored window rather than the raw range.
    const rows = withinCutoff(
      Object.entries(peHistory)
        .map(([date, pe]) => ({
          originalDate: date,
          date: formatDate(date),
          pe,
        }))
        .sort((a, b) => new Date(a.originalDate) - new Date(b.originalDate)),
      eventCutoff
    );
    const estRows = Object.entries(peEstimates)
      .map(([date, pe]) => ({
        originalDate: date,
        date: formatDate(date),
        peEstimate: pe,
        isEstimate: true,
      }))
      .sort((a, b) => new Date(a.originalDate) - new Date(b.originalDate));
    if (rows.length > 0 && estRows.length > 0) {
      rows[rows.length - 1] = { ...rows[rows.length - 1], peEstimate: rows[rows.length - 1].pe };
    }
    return rows.concat(estRows);
  }, [data?.pe_history, data?.pe_estimates, eventCutoff]);

  const orderMarkers = useMemo(() => chartData.filter(d => d.order), [chartData]);
  const epsInRange = useMemo(() => withinCutoff(epsSeries, eventCutoff), [epsSeries, eventCutoff]);
  const recsInRange = useMemo(() => withinCutoff(recommendationsSeries, eventCutoff), [recommendationsSeries, eventCutoff]);

  const priceSpan = formatSpan(priceData);
  const peSpan = formatSpan(peData);
  const epsSpan = formatSpan(epsInRange);
  const cashflowSpan = formatSpan(cashflowSeries);
  const recsSpan = formatSpan(recsInRange);

  // Calculate smart Y-axis domain for PE chart
  const peDomain = useMemo(() => {
    if (peData.length === 0) return [0, 100];

    const peValues = peData.map(d => d.pe ?? d.peEstimate).filter(v => v != null && !isNaN(v));
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

  const avgPeFromHistory = useMemo(() => {
    const values = peData.map(d => d.pe).filter(v => v != null && v > 0 && v < 500);
    return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
  }, [peData]);

  const qualityTrend = useMemo(() => {
    return (data?.quality_history || [])
      .filter(r => r.roic != null || r.operating_margin != null || r.profit_margin != null)
      .map(r => ({ ...r, date: formatDate(r.date) }));
  }, [data?.quality_history]);

  // Labelled SEC as-restated FY ROIC — separate from Yahoo quality trend / holdings scalar.
  const secRoicTrend = useMemo(() => {
    return (data?.roic_sec_history || [])
      .filter(r => r.roic != null)
      .map(r => ({ period_end: formatDate(r.period_end), roic_sec: r.roic }));
  }, [data?.roic_sec_history]);

  const yahooRoicMarker = data?.roic_yahoo_marker || null;

  if (loading) return (
    <div className="page-fixed stock-container">
      <div className="skeleton skeleton-breadcrumb" />
      <div className="skeleton skeleton-title" />
      <div className="skeleton-kpi-row">
        {Array.from({ length: 8 }, (_, idx) => <div key={idx} className="skeleton skeleton-kpi" />)}
      </div>
      <div className="skeleton skeleton-panel" />
      <div className="skeleton skeleton-panel" />
      <div className="skeleton skeleton-panel skeleton-panel-tall" />
    </div>
  );
  if (error) return <div className="page-fixed stock-container error">{error}</div>;
  if (!data) return null;

  const i = data.instrument;
  const f = data.fundamentals || {};

  const thesisSummary = i?.thesis?.summary ?? null;

  // Funds report neither cashflow nor market cap, so the overlay supplies the
  // yield already computed from constituents — same percent units.
  const fcfYield = f.fcfYield != null ? f.fcfYield
    : (f.freeCashflow && f.marketCap && f.marketCap > 0) ? (f.freeCashflow / f.marketCap) * 100
    : null;

  // The backend always sets both keys, so a missing figure is null, never
  // undefined — `!== undefined` passed and rendered a fabricated net debt of 0.
  const netDebt = (f.totalDebt != null && f.totalCash != null)
    ? f.totalDebt - f.totalCash : null;

  const sector = i.sector || '';
  const pbThresholds = getPbThresholds(sector);
  const psThresholds = getPsThresholds(sector);

  const dcfDiff = data.dcf_diff;
  const dcfPrice = data.dcf_price;
  const dcfLow = data.dcf_low;
  const dcfHigh = data.dcf_high;
  const dcfImplied = data.dcf_implied_growth;
  // Low confidence: reverse-DCF couldn't solve (market outside model's growth
  // band) OR sensitivity band spans >4x (inputs too fragile).
  const dcfLowConfidence = dcfDiff != null && (
    dcfImplied == null || (dcfLow && dcfHigh && dcfLow > 0 && dcfHigh / dcfLow > 4)
  );

  // --- Fair value vs price (Valuation panel) ---
  const fvCurrent = f.currentPrice ?? (priceData.length > 0 ? priceData[priceData.length - 1].value : null);
  const fvFmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
  const apt = data.analyst_price_targets || {};
  // Price at its 5y representative multiple; only when Yahoo's trailingPE is on
  // a comparable EPS basis to the scraped history (else the ratio is noise).
  const peFairPrice = (data.avg_pe_5y != null && data.pe_basis_matches && f.peRatio > 0 && fvCurrent != null)
    ? fvCurrent * data.avg_pe_5y / f.peRatio
    : null;
  const fairValueRows = [];
  if (apt.median != null || (apt.low != null && apt.high != null)) {
    fairValueRows.push({
      label: f.numberOfAnalystOpinions ? `Analysts (${f.numberOfAnalystOpinions})` : 'Analysts',
      low: apt.low ?? null,
      mid: apt.median ?? null,
      high: apt.high ?? null,
      tooltip: [
        '1-year analyst price targets',
        apt.low != null ? `Low: ${fvFmt(apt.low)}` : null,
        apt.median != null ? `Median: ${fvFmt(apt.median)}` : null,
        apt.high != null ? `High: ${fvFmt(apt.high)}` : null,
        'Targets average ~28% optimistic historically — read the band, not the level',
      ].filter(Boolean).join('\n'),
    });
  }
  if (dcfPrice != null) {
    fairValueRows.push({
      label: 'DCF model',
      low: dcfLow ?? null,
      mid: dcfPrice,
      high: dcfHigh ?? null,
      muted: dcfLowConfidence,
      tooltip: [
        `DCF fair value: ${fvFmt(dcfPrice)}`,
        dcfLow != null && dcfHigh != null ? `Sensitivity: ${fvFmt(dcfLow)} – ${fvFmt(dcfHigh)}` : null,
        'Growth capped at +30%, so high-growth names read as structurally overvalued',
        dcfLowConfidence ? 'Low confidence — model over-sensitive or disagrees with market' : null,
      ].filter(Boolean).join('\n'),
    });
  }
  if (peFairPrice != null) {
    fairValueRows.push({
      label: '5y avg multiple',
      mid: peFairPrice,
      tooltip: `Price at its 5-year harmonic average P/E (${Math.round(data.avg_pe_5y)} vs current ${Math.round(f.peRatio)})`,
    });
  }

  // Expectations check: growth the market pays for vs growth the company delivers.
  const impliedGrowthPct = dcfImplied != null ? dcfImplied * 100 : null;
  const deliveredGrowthPct = f.revenueGrowth != null ? f.revenueGrowth * 100 : null;
  let expectations = null;
  if (impliedGrowthPct != null) {
    const stretched = deliveredGrowthPct != null && impliedGrowthPct > deliveredGrowthPct + 10;
    const modest = deliveredGrowthPct != null && impliedGrowthPct <= deliveredGrowthPct;
    expectations = {
      implied: `${impliedGrowthPct.toFixed(1)}%`,
      className: stretched ? 'negative' : modest ? 'positive' : '',
    };
  } else if (data.dcf_implied_growth_status === 'above_band') {
    expectations = { implied: '>50%', className: 'negative' };
  } else if (data.dcf_implied_growth_status === 'below_band') {
    expectations = { implied: '< -10%', className: 'positive' };
  }

  // Composite score — same formula and inputs as the Holdings Score column.
  // Earnings signal comes from the newest report that has an LLM assessment.
  const assessmentReport = (data.earnings_reports || []).find(
    r => r.metrics?.investment_assessment?.recommendation
  );
  const compositeInput = {
    quote_type: i.quote_type,
    sector: i.sector,
    screener_score: data.screener_score,
    // Paired with the score above — both come from the same nightly snapshot.
    screener_score_max: data.screener_score_max,
    earnings_signal: assessmentReport?.metrics.investment_assessment.recommendation ?? null,
    earnings_conviction: assessmentReport?.metrics.investment_assessment.conviction ?? null,
    earnings_announcement_date: assessmentReport?.announcement_date ?? null,
    earnings_report_date: assessmentReport?.date ?? null,
    recommendation_mean: f.recommendationMean,
    form13f_score: data.form13f_score,
    // Funds: the screener pair and 13F above are aggregated from constituents,
    // and the flag is what admits them past the ETF guard.
    look_through: data.look_through,
    look_through_form13f: data.look_through_form13f,
    look_through_n: data.look_through_n,
    look_through_as_of: data.look_through_as_of,
  };
  const compositeScore = computeComposite(compositeInput);

  const passedScreeners = data.passed_screeners || [];
  // A fund passes no gates of its own: "No screeners passed" would read as a
  // verdict on the fund rather than as the absence of an equity screening.
  const screenerTooltip = data.look_through
    ? [
        `Screener: cap-weighted average of ${data.look_through_n} constituents' scores` +
          `${data.look_through_as_of ? ` (as of ${data.look_through_as_of})` : ''}`,
        'What the average pound in this fund is invested in — the fund passes no gates itself.',
      ].join('\n')
    : [
        `Screener score (nightly snapshot${data.screener_as_of ? ` · ${data.screener_as_of}` : ''})`,
        passedScreeners.length > 0
          ? `Passed:\n${passedScreeners.map(id => `✓ ${id.replace(/_/g, ' ')}`).join('\n')}`
          : 'No screeners passed',
      ].join('\n');

  const kpiTooltips = {
    'Market Cap': 'Total market value of all outstanding shares',
    'PE': avgPeFromHistory
      ? `Price-to-Earnings (trailing 12 months) · Hist avg: ${Math.round(avgPeFromHistory)}`
      : 'Price-to-Earnings ratio (trailing 12 months)',
    'Fwd PE': f.peRatio != null && f.forwardPE != null
      ? `Forward PE (next 12-month estimates) · Trailing: ${Math.round(f.peRatio)}`
      : 'Forward PE based on next 12-month earnings estimates',
    'PEG': 'PE divided by expected earnings growth rate · Green < 1.5, Red > 3',
    'Dividend': 'Annual dividend yield',
    'Beta': 'Stock volatility vs market (1 = market) · Green < 1, Red > 2',
    'Debt': 'Total debt',
    'Cash': 'Cash and equivalents',
    'FCF Yield': 'Free cash flow as % of market cap · Green > 6%, Red < 2%',
  };

  const kpis = [
    ...(compositeScore != null ? [{
      label: 'Score',
      value: compositeScore.toFixed(1),
      className: compositeScore >= 7 ? 'positive' : compositeScore < 3 ? 'negative' : '',
      // Same formula, but the screener input here is the nightly snapshot while
      // Holdings recomputes live — say so, or a day's lag reads as a bug. Funds
      // are the exception: both pages read the same stored look-through payload.
      tooltip: `${compositeTooltip(compositeScore, compositeInput)}\nSame formula as the Holdings Score column`
        + (data.look_through
            ? '\nBoth pages read the same stored look-through figures, so they cannot diverge'
            : `\nScreener from the ${data.screener_as_of || 'nightly'} snapshot, so this can lag Holdings by a day`),
    }] : []),
    ...(data.screener_score != null ? [{
      label: 'Screener',
      value: Math.round(data.screener_score),
      className: screenerRatio(data.screener_score, data.screener_score_max) >= 0.6 ? 'positive'
        : screenerRatio(data.screener_score, data.screener_score_max) < 0.2 ? 'negative' : '',
      tooltip: screenerTooltip,
    }] : []),
    { label: 'Market Cap', value: formatShort(f.marketCap), tooltip: kpiTooltips['Market Cap'] },
    {
      label: 'PE',
      value: f.peRatio != null ? Math.round(f.peRatio) : '-',
      className: f.peRatio != null
        ? (avgPeFromHistory != null
          ? (f.peRatio < avgPeFromHistory ? 'positive' : f.peRatio > avgPeFromHistory ? 'negative' : '')
          : (f.peRatio < 20 ? 'positive' : f.peRatio > 35 ? 'negative' : ''))
        : '',
      tooltip: kpiTooltips['PE'],
    },
    ...(f.forwardPE != null ? [{
      label: 'Fwd PE',
      value: Math.round(f.forwardPE),
      className: f.peRatio != null
        ? (f.forwardPE < f.peRatio ? 'positive' : f.forwardPE > f.peRatio ? 'negative' : '')
        : (f.forwardPE < 20 ? 'positive' : f.forwardPE > 35 ? 'negative' : ''),
      tooltip: kpiTooltips['Fwd PE'],
    }] : []),
    {
      label: 'PEG',
      value: f.pegRatio != null ? Math.round(f.pegRatio * 100) / 100 : '-',
      className: f.pegRatio != null
        ? (f.pegRatio < 1.5 ? 'positive' : f.pegRatio > 3 ? 'negative' : '')
        : '',
      tooltip: kpiTooltips['PEG'],
    },
    // 0% is a real yield (XNAS.L distributes nothing) — truthiness hid it as unknown.
    { label: 'Dividend', value: f.dividendYield != null ? `${f.dividendYield.toFixed(2)}%` : '-', tooltip: kpiTooltips['Dividend'] },
    {
      label: 'Beta',
      value: f.beta ?? '-',
      className: f.beta != null
        ? (f.beta < 1 ? 'positive' : f.beta > 2 ? 'negative' : '')
        : '',
      tooltip: kpiTooltips['Beta'],
    },
    { label: 'Debt', value: formatShort(f.totalDebt), tooltip: kpiTooltips['Debt'] },
    { label: 'Cash', value: formatShort(f.totalCash), tooltip: kpiTooltips['Cash'] },
    {
      label: 'FCF Yield',
      value: fcfYield != null ? `${fcfYield.toFixed(2)}%` : '-',
      className: fcfYield != null
        ? (fcfYield > 6 ? 'positive' : fcfYield < 2 ? 'negative' : '')
        : '',
      tooltip: kpiTooltips['FCF Yield'],
    },
    ...(() => {
      if (data.f_score == null) return [];
      const details = data.f_score_details || {};
      const breakdown = Object.entries(details)
        .map(([label, passed]) => `${passed ? '✓' : '✗'} ${label}`)
        .join('\n');
      const tooltip = `Piotroski F-Score: ${data.f_score}/9${breakdown ? `\n\n${breakdown}` : ''}`;
      return [{
        label: 'F-Score',
        value: `${data.f_score}/9`,
        className: data.f_score >= 8 ? 'positive' : data.f_score <= 3 ? 'negative' : '',
        tooltip,
      }];
    })(),
    ...(() => {
      const buys = data.insider_buy_count_90d;
      const sells = data.insider_sell_count_90d;
      const net = data.insider_net_value_90d;
      // Hide tile entirely when there's no coverage (NULL from backend) OR
      // no activity at all in the window — empty "0B / 0S" is just noise.
      if (buys == null && sells == null) return [];
      const b = buys ?? 0;
      const s = sells ?? 0;
      if (b === 0 && s === 0) return [];
      const n = net ?? 0;
      const formatK = (v) => {
        const abs = Math.abs(v);
        if (abs >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
        if (abs >= 1_000) return `${(v / 1_000).toFixed(0)}k`;
        return v.toFixed(0);
      };
      const sign = n >= 0 ? '+' : '-';
      const netLabel = `${sign}$${formatK(Math.abs(n))}`;
      return [{
        label: 'Insiders 90d',
        value: `${b}B / ${s}S`,
        className: b >= 2 && n > 0 ? 'positive' : (s >= 3 && n < 0 ? 'negative' : ''),
        tooltip: `Insider trades (last 90d): ${b} buyer(s), ${s} seller(s)\nNet value: ${netLabel}\nSource: yfinance / SEC Form 4 (US tickers only)`,
      }];
    })(),
    ...(() => {
      const apt = data.analyst_price_targets || {};
      const mean = apt.mean;
      const lo = apt.low;
      const hi = apt.high;
      const cur = f.currentPrice;
      if (mean == null) return [];
      const upside = cur ? (((mean - cur) / cur) * 100).toFixed(1) : null;
      const tooltip = [
        upside != null ? `Upside: ${Number(upside) > 0 ? '+' : ''}${upside}%` : null,
        lo != null ? `Low: ${Number(lo).toLocaleString(undefined, { maximumFractionDigits: 2 })}` : null,
        hi != null ? `High: ${Number(hi).toLocaleString(undefined, { maximumFractionDigits: 2 })}` : null,
        'Analyst consensus price target',
      ].filter(Boolean).join(' · ');
      const displayValue = upside != null
        ? `${Number(upside) >= 0 ? '+' : ''}${upside}%`
        : Number(mean).toLocaleString(undefined, { maximumFractionDigits: 2 });
      return [{
        label: 'Target',
        value: displayValue,
        className: cur ? (mean > cur ? 'positive' : mean < cur ? 'negative' : '') : '',
        tooltip,
      }];
    })(),
    ...(() => {
      const ts = f.nextEarningsDate;
      if (!ts) return [];
      const d = new Date(ts * 1000);
      const now = new Date();
      const diffDays = Math.round((d - now) / 86400000);
      if (diffDays < -30 || diffDays > 365) return [];
      const label = d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
      const tooltip = `Next earnings: ${d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })} · ${diffDays >= 0 ? `in ${diffDays} days` : `${-diffDays} days ago`}`;
      return [{
        label: 'Earnings',
        value: label,
        className: diffDays >= 0 && diffDays <= 14 ? 'negative' : '',
        tooltip,
      }];
    })(),
    ...(() => {
      if (dcfDiff == null) return [];
      const pct = (dcfDiff * 100).toFixed(1);
      const direction = dcfDiff > 0 ? 'undervalued' : 'overvalued';
      const fmt = (x) => Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 });
      const tooltipParts = ['2-stage DCF model'];
      if (dcfPrice != null) tooltipParts.push(`Fair value: ${fmt(dcfPrice)}`);
      if (dcfLow != null && dcfHigh != null) tooltipParts.push(`Range: ${fmt(dcfLow)} – ${fmt(dcfHigh)}`);
      tooltipParts.push(`${direction} by ${Math.abs(Number(pct))}% vs current price`);
      if (dcfImplied != null) tooltipParts.push(`Market-implied growth: ${(dcfImplied * 100).toFixed(1)}%`);
      else if (data.dcf_implied_growth_status === 'above_band') tooltipParts.push('Market prices in >50% initial growth');
      else if (data.dcf_implied_growth_status === 'below_band') tooltipParts.push('Market prices in < -10% initial growth');
      if (dcfLowConfidence) tooltipParts.push('Low confidence — model over-sensitive or disagrees with market');
      tooltipParts.push('Green > +10%, Red < -10%');
      return [{
        label: 'DCF',
        value: `${Number(pct) >= 0 ? '+' : ''}${pct}%`,
        className: dcfDiff > 0.10 ? 'positive' : dcfDiff < -0.10 ? 'negative' : '',
        itemClassName: dcfLowConfidence ? 'dcf-low-confidence' : '',
        tooltip: tooltipParts.join(' · '),
      }];
    })(),
  ];

  const sectorLabel = sector || 'general';
  const valuationTooltips = {
    'EV': 'Enterprise value (market cap + debt - cash)',
    'EV/EBITDA': 'Enterprise value to EBITDA · Green < 15, Red > 25',
    'EV/Sales': psThresholds
      ? `Enterprise value to revenue · Green < ${psThresholds[0]}, Red > ${psThresholds[1]} (${sectorLabel})`
      : `Enterprise value to revenue · Not color-coded for ${sectorLabel} sector`,
    'P/S (TTM)': psThresholds
      ? `Price-to-sales (TTM) · Green < ${psThresholds[0]}, Red > ${psThresholds[1]} (${sectorLabel})`
      : `Price-to-sales (TTM) · Not color-coded for ${sectorLabel} sector`,
    'P/B': pbThresholds
      ? `Price-to-book · Green < ${pbThresholds[0]}, Red > ${pbThresholds[1]} (${sectorLabel})`
      : `Price-to-book · Not color-coded for ${sectorLabel} (intangibles-heavy sector)`,
    'Net Debt': 'Total debt minus cash · Green = net cash position',
    'Net Debt/EBITDA': 'Net debt to EBITDA (leverage) · Green ≤ 1, Red > 3',
    'Price/FCF': 'Price to free cash flow · Green < 20, Red > 40',
  };

  const valuation = [
    { label: 'EV', value: formatShort(f.enterpriseValue), tooltip: valuationTooltips['EV'] },
    {
      label: 'EV/EBITDA',
      value: formatRatio(f.enterpriseToEbitda),
      className: f.enterpriseToEbitda != null
        ? (f.enterpriseToEbitda < 15 ? 'positive' : f.enterpriseToEbitda > 25 ? 'negative' : '')
        : '',
      tooltip: valuationTooltips['EV/EBITDA'],
    },
    {
      label: 'EV/Sales',
      value: formatRatio(f.enterpriseToRevenue),
      className: (() => {
        if (f.enterpriseToRevenue == null || psThresholds === null) return '';
        return f.enterpriseToRevenue < psThresholds[0] ? 'positive'
          : f.enterpriseToRevenue > psThresholds[1] ? 'negative' : '';
      })(),
      tooltip: valuationTooltips['EV/Sales'],
    },
    {
      label: 'P/S (TTM)',
      value: formatRatio(f.priceToSalesTtm),
      className: (() => {
        if (f.priceToSalesTtm == null || psThresholds === null) return '';
        return f.priceToSalesTtm < psThresholds[0] ? 'positive'
          : f.priceToSalesTtm > psThresholds[1] ? 'negative' : '';
      })(),
      tooltip: valuationTooltips['P/S (TTM)'],
    },
    {
      label: 'P/B',
      value: formatRatio(f.priceToBook),
      className: (() => {
        if (f.priceToBook == null || pbThresholds === null) return '';
        return f.priceToBook < pbThresholds[0] ? 'positive'
          : f.priceToBook > pbThresholds[1] ? 'negative' : '';
      })(),
      tooltip: valuationTooltips['P/B'],
    },
    {
      label: 'Net Debt',
      value: netDebt != null ? formatShort(netDebt) : '-',
      className: netDebt != null ? (netDebt < 0 ? 'positive' : '') : '',
      tooltip: valuationTooltips['Net Debt'],
    },
    {
      label: 'Net Debt/EBITDA',
      value: (f.ebitda && f.ebitda !== 0 && netDebt != null) ? formatRatio(netDebt / f.ebitda) : '-',
      className: (() => {
        if (netDebt == null || !f.ebitda || f.ebitda === 0) return '';
        const r = netDebt / f.ebitda;
        return r <= 1 ? 'positive' : r > 3 ? 'negative' : '';
      })(),
      tooltip: valuationTooltips['Net Debt/EBITDA'],
    },
    {
      label: 'Price/FCF',
      value: (f.marketCap && f.freeCashflow) ? formatRatio(f.marketCap / f.freeCashflow) : '-',
      className: (() => {
        if (!f.marketCap || !f.freeCashflow) return '';
        const r = f.marketCap / f.freeCashflow;
        return r < 0 ? 'negative' : r < 20 ? 'positive' : r > 40 ? 'negative' : '';
      })(),
      tooltip: valuationTooltips['Price/FCF'],
    },
  ];

  // Yahoo margins/growth are fractions; ROIC and Rule of 40 are already percent
  const fmtPct = (v, digits = 1) => (v != null ? `${(v * 100).toFixed(digits)}%` : '-');
  const ruleOf40 = (f.revenueGrowth != null && f.profitMargins != null)
    ? (f.revenueGrowth + f.profitMargins) * 100 : null;

  const quality = [
    {
      label: 'ROIC',
      value: f.roic != null ? `${f.roic.toFixed(1)}%` : '-',
      className: f.roic != null ? (f.roic > 20 ? 'positive' : f.roic < 10 ? 'negative' : '') : '',
      tooltip: 'Return on Invested Capital (NOPAT / invested capital) · Green > 20%, Red < 10% · Primary quality gate — leverage-neutral, unlike ROE',
    },
    {
      label: 'ROE',
      value: fmtPct(f.returnOnEquity),
      tooltip: 'Return on Equity · Not color-coded: distorted by debt and buybacks — prefer ROIC',
    },
    {
      label: 'Gross Margin',
      value: fmtPct(f.grossMargins),
      className: f.grossMargins != null
        ? (f.grossMargins > 0.60 ? 'positive' : f.grossMargins < 0.30 ? 'negative' : '')
        : '',
      tooltip: 'Gross profit / revenue · Green > 60%, Red < 30% · A proxy for pricing power / moat',
    },
    {
      label: 'Op Margin',
      value: fmtPct(f.operatingMargins),
      className: f.operatingMargins != null
        ? (f.operatingMargins > 0.25 ? 'positive' : f.operatingMargins < 0.05 ? 'negative' : '')
        : '',
      tooltip: 'Operating income / revenue · Green > 25%, Red < 5%',
    },
    {
      label: 'Net Margin',
      value: fmtPct(f.profitMargins),
      className: f.profitMargins != null
        ? (f.profitMargins > 0.30 ? 'positive' : f.profitMargins < 0.10 ? 'negative' : '')
        : '',
      tooltip: 'Net income / revenue · Green > 30%, Red < 10%',
    },
    {
      label: 'Rev Growth',
      value: fmtPct(f.revenueGrowth),
      className: f.revenueGrowth != null
        ? (f.revenueGrowth > 0.40 ? 'positive' : f.revenueGrowth < 0.15 ? 'negative' : '')
        : '',
      tooltip: 'Revenue growth (YoY) · Green > 40%, Red < 15%',
    },
    {
      label: 'Rule of 40',
      value: ruleOf40 != null ? Math.round(ruleOf40) : '-',
      className: ruleOf40 != null ? (ruleOf40 >= 40 ? 'positive' : ruleOf40 < 20 ? 'negative' : '') : '',
      tooltip: 'Revenue growth % + net margin % · Green ≥ 40, Red < 20 · Growth/profitability balance',
    },
  ];

  const sections = [
    { id: 'sec-valuation', label: 'Valuation' },
    { id: 'sec-quality', label: 'Quality' },
    ...(data.etf_holdings ? [{ id: 'sec-holdings', label: 'Holdings' }] : []),
    { id: 'sec-13f', label: '13F' },
    ...((thesisSummary || data.thesis_rule_eval) ? [{ id: 'sec-thesis', label: 'Thesis' }] : []),
    ...(data.agent_suggestion ? [{ id: 'sec-agent', label: 'Agent' }] : []),
    { id: 'sec-price', label: 'Price' },
    { id: 'sec-pe', label: 'PE' },
    { id: 'sec-eps', label: 'EPS' },
    { id: 'sec-cashflow', label: 'Cash Flow' },
    { id: 'sec-analysts', label: 'Analysts' },
    ...(data.earnings_reports?.length > 0 ? [{ id: 'sec-reports', label: 'Reports' }] : []),
    ...(data.position_review ? [{ id: 'sec-review', label: 'Review' }] : []),
    { id: 'sec-news', label: 'News' },
  ];

  return (
    <div className="page-fixed stock-container">
      <nav className="stock-breadcrumb">
        <Link to="/holdings">Holdings</Link>
        <span className="breadcrumb-sep">›</span>
        <SymbolSwitcher currentSymbol={symbol} />
      </nav>
      <div className="stock-header">
        <h2>{i.symbol} — {i.name}</h2>
        {data.my_position && (
          <div className="my-position-card">
            <h4>Your Position</h4>
            <div className="my-position-grid">
              <div className="my-position-item" title="% of your portfolio">
                <span className="my-position-label">%</span>
                <span className="my-position-value">{data.my_position.portfolio_pct.toFixed(2)}</span>
              </div>
              <div className="my-position-item" title="Current value in GBP">
                <span className="my-position-label">Value (£)</span>
                <span className="my-position-value">
                  {hideAmounts ? MASK : `£${data.my_position.market_value.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`}
                </span>
              </div>
              <div className="my-position-item" title="Profit/loss in GBP">
                <span className="my-position-label">Profit (£)</span>
                <span className={`my-position-value ${data.my_position.profit >= 0 ? 'positive' : 'negative'}`}>
                  {hideAmounts ? MASK : `£${data.my_position.profit >= 0 ? '+' : ''}${data.my_position.profit.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`}
                </span>
              </div>
              <div className="my-position-item" title="Return on cost">
                <span className="my-position-label">Return</span>
                <span className={`my-position-value ${data.my_position.return_pct >= 0 ? 'positive' : 'negative'}`}>
                  {data.my_position.return_pct >= 0 ? '+' : ''}{Math.round(data.my_position.return_pct)}%
                </span>
              </div>
              {data.my_position.quantity != null && (
                <div className="my-position-item" title="Shares held">
                  <span className="my-position-label">Shares</span>
                  <span className="my-position-value">
                    {hideAmounts ? MASK : data.my_position.quantity.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                  </span>
                </div>
              )}
              {data.my_position.avg_price != null && (
                <div className="my-position-item" title={`Average cost per share (${i.currency})`}>
                  <span className="my-position-label">Avg Cost</span>
                  <span className="my-position-value">
                    {data.my_position.avg_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </span>
                </div>
              )}
              {data.my_position.first_buy_date && (() => {
                const firstBuy = new Date(data.my_position.first_buy_date + 'T00:00:00');
                const years = (Date.now() - firstBuy.getTime()) / (365.25 * 86400000);
                const heldLabel = years >= 1 ? `${years.toFixed(1)}y` : `${Math.max(1, Math.round(years * 12))}mo`;
                return (
                  <div className="my-position-item" title={`First buy ${data.my_position.first_buy_date}`}>
                    <span className="my-position-label">Held</span>
                    <span className="my-position-value">{heldLabel}</span>
                  </div>
                );
              })()}
              {data.my_position.dividends_gbp > 0 && (
                <div className="my-position-item" title="Dividends received (all time)">
                  <span className="my-position-label">Dividends</span>
                  <span className="my-position-value">
                    {hideAmounts ? MASK : `£${data.my_position.dividends_gbp.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
        <div className="stock-kpis">
          {kpis.map(k => (
            <div key={k.label} className={`kpi${k.itemClassName ? ` ${k.itemClassName}` : ''}`} title={k.tooltip}>
              <div className="kpi-label">{k.label}</div>
              <div className={`kpi-value${k.className ? ` ${k.className}` : ''}`}>{k.value}</div>
            </div>
          ))}
        </div>
        <FiftyTwoWeekBar
          low={f.fiftyTwoWeekLow}
          high={f.fiftyTwoWeekHigh}
          current={f.currentPrice ?? (priceData.length > 0 ? priceData[priceData.length - 1].value : null)}
        />
        <div className="stock-meta">
          {i.sector && <span className="stock-meta-tag stock-meta-sector">{i.sector}</span>}
          {i.country && (
            <span className="stock-meta-tag stock-meta-country">
              {renderCountryWithFlag(i.country, { width: '1em', height: '0.75em', marginRight: '4px', fontSize: '0.85em' })}
            </span>
          )}
          {i.currency && <span className="stock-meta-tag stock-meta-currency">{i.currency}</span>}
        </div>
      </div>

      <nav className="stock-section-nav">
        {sections.map(s => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className={`stock-section-link${activeSection === s.id ? ' active' : ''}`}
            onClick={(e) => handleSectionClick(e, s.id)}
          >
            {s.label}
          </a>
        ))}
        <div
          className="stock-nav-range"
          style={{ opacity: chartLoading ? 0.6 : 1, pointerEvents: chartLoading ? 'none' : 'auto' }}
        >
          <div className="stock-segmented">
            {CHART_DAYS_OPTIONS.map(option => (
              <button
                key={option.value}
                className={`stock-seg-btn${chartDays === option.value ? ' active' : ''}`}
                onClick={() => handleChartDaysChange(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <select
            className="stock-range-select"
            value={chartDays}
            onChange={(e) => handleChartDaysChange(e.target.value === 'ytd' ? 'ytd' : Number(e.target.value))}
            aria-label="Chart range"
          >
            {CHART_DAYS_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </div>
      </nav>

      <div className="stock-panels">
        <div className="panel" id="sec-valuation">
          <h3>Valuation Multiples</h3>
          <div className="valuation-grid">
            {valuation.map(v => (
              <div key={v.label} className="kpi" title={v.tooltip}>
                <div className="kpi-label">{v.label}</div>
                <div className={`kpi-value${v.className ? ` ${v.className}` : ''}`}>{v.value}</div>
              </div>
            ))}
          </div>
          {(fairValueRows.length > 0 || expectations) && (
            <div className="fv-block">
              <h4>Fair Value vs Price</h4>
              {fairValueRows.length > 0 && <FairValueBand rows={fairValueRows} current={fvCurrent} />}
              {expectations && (
                <div
                  className="fv-expectations"
                  title={'Reverse DCF: the initial FCF growth rate that makes the model price equal the market price. Compare with what the business actually delivers.'}
                >
                  <span className="fv-expectations-label">Expectations</span>
                  <span>
                    market prices in <strong className={expectations.className}>{expectations.implied}</strong> initial growth
                    {deliveredGrowthPct != null && (
                      <> · delivered revenue growth <strong>{deliveredGrowthPct.toFixed(1)}%</strong> (last quarter, YoY)</>
                    )}
                  </span>
                </div>
              )}
              <p className="fv-caption">
                Independent estimates on one price axis — wide disagreement means fair value is contested. Context, not a signal.
              </p>
            </div>
          )}
        </div>
        <div className="panel" id="sec-quality">
          <h3>Quality</h3>
          <div className="valuation-grid">
            {quality.map(q => (
              <div key={q.label} className="kpi" title={q.tooltip}>
                <div className="kpi-label">{q.label}</div>
                <div className={`kpi-value${q.className ? ` ${q.className}` : ''}`}>{q.value}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Lights up automatically once FeaturesDaily has enough snapshots */}
        {qualityTrend.length >= 5 && (
          <div className="panel">
            <h3>Quality Trend</h3>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={qualityTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis tickFormatter={(v) => `${v}%`} />
                <Tooltip
                  content={<SharedTooltip
                    valueFormatter={(v) => (typeof v === 'number' ? `${v.toFixed(1)}%` : v)}
                    nameMap={{
                      roic: 'ROIC',
                      gross_margin: 'Gross Margin',
                      operating_margin: 'Op Margin',
                      profit_margin: 'Net Margin',
                      revenue_growth: 'Rev Growth',
                    }}
                  />}
                />
                <Legend />
                <Line type="monotone" dataKey="roic" name="ROIC" stroke="#16a34a" strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="gross_margin" name="Gross Margin" stroke="#94a3b8" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="operating_margin" name="Op Margin" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="profit_margin" name="Net Margin" stroke="#8b5cf6" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="revenue_growth" name="Rev Growth" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {secRoicTrend.length >= 3 && (
          <div className="panel">
            <h3>ROIC (SEC as-restated)</h3>
            <p className="text-muted" style={{ fontSize: '0.85rem', marginTop: 0 }}>
              Fiscal-year ROIC from SEC companyfacts. Holdings table still uses Yahoo; the marker is Yahoo&apos;s latest ROIC for comparison — not spliced into the series.
            </p>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={secRoicTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period_end" />
                <YAxis tickFormatter={(v) => `${v}%`} />
                <Tooltip
                  content={<SharedTooltip
                    valueFormatter={(v) => (typeof v === 'number' ? `${v.toFixed(1)}%` : v)}
                    nameMap={{ roic_sec: 'SEC ROIC' }}
                  />}
                />
                <Legend />
                <Line type="monotone" dataKey="roic_sec" name="SEC as-restated" stroke="#0f766e" strokeWidth={2.5} dot />
                {yahooRoicMarker && (
                  <ReferenceLine
                    y={yahooRoicMarker.roic}
                    stroke="#16a34a"
                    strokeDasharray="4 3"
                    label={{ value: `Yahoo ${yahooRoicMarker.roic.toFixed(1)}%`, position: 'insideTopRight', fill: '#16a34a', fontSize: 12 }}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        {data.etf_holdings && (() => {
          const h = data.etf_holdings;
          const list = etfHoldingsOpen ? h.constituents : h.constituents.slice(0, ETF_HOLDINGS_COLLAPSED);
          const hidden = h.count - list.length;
          const hiddenPct = h.total_pct - list.reduce((sum, c) => sum + c.weight_pct, 0);
          return (
            <div className="panel" id="sec-holdings">
              <h3>
                Holdings ({h.count})
                <span className="form13f-as-of"> — as of {formatDateShort(h.as_of)}</span>
              </h3>
              <p className="etf-holdings-summary">
                {h.held_pct > 0 && (
                  <><strong>{h.held_pct.toFixed(1)}%</strong> of this fund is in names you already hold directly.{' '}</>
                )}
                {h.resolved_pct < h.coverage_gate_pct
                  ? `Only ${h.resolved_pct.toFixed(1)}% is matched to tracked instruments — below the ${h.coverage_gate_pct}% needed to derive fund metrics, which is why they are blank.`
                  : data.look_through
                    ? `Derived metrics above are weighted over the ${h.resolved_pct.toFixed(1)}% matched to tracked instruments.`
                    : `${h.resolved_pct.toFixed(1)}% is matched to tracked instruments.`}
              </p>
              <div className={`form13f-list${etfHoldingsOpen ? ' etf-holdings-scroll' : ''}`}>
                {list.map((c) => (
                  <div className="form13f-row" key={c.key}>
                    <span className="etf-holdings-ticker">
                      {c.symbol
                        ? <Link to={`/stock/${encodeURIComponent(c.symbol)}`} className="form13f-link">{c.symbol}</Link>
                        : <span className="etf-holdings-untracked" title="Not a tracked instrument">—</span>}
                    </span>
                    <span className="form13f-name">{c.name}</span>
                    {c.held && <span className="etf-holdings-held" title="You hold this directly too">held</span>}
                    <span className="form13f-value">{c.weight_pct.toFixed(2)}%</span>
                  </div>
                ))}
              </div>
              {(hidden > 0 || etfHoldingsOpen) && (
                <button type="button" className="etf-holdings-toggle" onClick={() => setEtfHoldingsOpen(!etfHoldingsOpen)}>
                  {etfHoldingsOpen
                    ? `Show top ${ETF_HOLDINGS_COLLAPSED}`
                    : `Show all ${h.count.toLocaleString()} · ${hidden.toLocaleString()} more, ${hiddenPct.toFixed(1)}% of the fund`}
                </button>
              )}
            </div>
          );
        })()}

        <div className="panel" id="sec-13f">
          <h3>
            Institutional Holders (13F)
            {data.form13f_as_of && (
              <span className="form13f-as-of"> — as of {formatDateShort(data.form13f_as_of)}</span>
            )}
          </h3>
          {data.form13f_holdings && data.form13f_holdings.length > 0 ? (() => {
            const holdings = data.form13f_holdings;

            // Summary counts + net flow
            const buyCount  = holdings.filter(h => h.scored && (h.change === 'New' || (h.change !== 'Closed' && h.change !== '—' && parseFloat(h.change) > 0))).length;
            const sellCount = holdings.filter(h => h.scored && (h.change === 'Closed' || (h.change !== 'New' && h.change !== '—' && parseFloat(h.change) < 0))).length;
            const netFlow   = holdings.reduce((sum, h) => {
              if (h.shares == null || h.value == null || h.shares_prev == null) return sum;
              const price = h.shares > 0 ? h.value / h.shares : 0;
              return sum + (h.shares - h.shares_prev) * price;
            }, 0);
            const formatFlow = (n) => {
              const abs = Math.abs(n);
              const sign = n >= 0 ? '+' : '−';
              if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
              if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(0)}M`;
              if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(0)}K`;
              return `${sign}$${Math.round(abs)}`;
            };
            const getChangeClass = (change) => {
              if (change === 'New') return 'form13f-change--new';
              if (change === 'Closed') return 'form13f-change--closed';
              if (change === '—') return 'form13f-change--stable';
              const pct = parseFloat(change);
              if (pct >= 10)  return 'form13f-change--increase';
              if (pct <= -30) return 'form13f-change--trimmed';
              return 'form13f-change--stable';
            };

            return (
              <>
                <div className="form13f-summary">
                  {buyCount > 0  && <span className="form13f-summary-buy">{buyCount} buying</span>}
                  {sellCount > 0 && <span className="form13f-summary-sell">{sellCount} selling</span>}
                  <span className="form13f-summary-hold">{holdings.length - buyCount - sellCount} unchanged</span>
                  {netFlow !== 0 && (
                    <span className={`form13f-summary-flow ${netFlow > 0 ? 'positive' : 'negative'}`}>
                      {formatFlow(netFlow)} net flow
                    </span>
                  )}
                </div>
                <div className="form13f-list">
                  {holdings.map((h, idx) => {
                    const scored = h.scored !== false;
                    // Detect boundary: first noise row that follows at least one scored row
                    const isFirstNoise = !scored && idx > 0 && holdings[idx - 1]?.scored !== false;
                    const flow = (h.shares != null && h.value != null && h.shares_prev != null)
                      ? (h.shares - h.shares_prev) * (h.shares > 0 ? h.value / h.shares : 0)
                      : null;
                    const tooltipParts = [`Report date: ${h.report_date}`];
                    if (h.shares_prev != null)
                      tooltipParts.push(`Shares: ${h.shares_prev.toLocaleString()} → ${h.shares?.toLocaleString() ?? '-'}`);
                    else
                      tooltipParts.push(`Shares: ${h.shares?.toLocaleString() ?? '-'}`);
                    if (h.report_date_prev) tooltipParts.push(`Prev report: ${h.report_date_prev}`);
                    if (!scored && h.score_reason) tooltipParts.push(`Not scored: ${h.score_reason}`);
                    if (flow != null && flow !== 0) tooltipParts.push(`Flow: ${formatFlow(flow)}`);

                    const valueDisplay = h.pct_of_portfolio != null
                      ? `$${formatShort(h.value)} (${h.pct_of_portfolio}%)`
                      : `$${formatShort(h.value)}`;

                    // Link internally to manager page
                    const nameContent = h.manager_id ? (
                      <Link to={`/13f/${h.manager_id}`} className="form13f-link">{h.manager_name}</Link>
                    ) : h.manager_name;

                    const changeClass = getChangeClass(h.change);
                    const rowSignal = scored
                      ? (changeClass === 'form13f-change--new' || changeClass === 'form13f-change--increase'
                          ? ' form13f-row--buy'
                          : changeClass === 'form13f-change--closed' || changeClass === 'form13f-change--trimmed'
                            ? ' form13f-row--sell'
                            : '')
                      : ' form13f-row--noise';

                    return (
                      <React.Fragment key={idx}>
                        {isFirstNoise && (
                          <div className="form13f-noise-divider">Other holders</div>
                        )}
                        <div className={`form13f-row${rowSignal}`}
                             title={tooltipParts.join('\n')}>
                          <span className="form13f-name">{nameContent}</span>
                          <span className="form13f-value">{valueDisplay}</span>
                          {flow != null && flow !== 0 && (
                            <span className={`form13f-flow ${flow > 0 ? 'positive' : 'negative'}`}>
                              {formatFlow(flow)}
                            </span>
                          )}
                          <span className={`form13f-change ${changeClass}`}>{h.change}</span>
                        </div>
                      </React.Fragment>
                    );
                  })}
                </div>
              </>
            );
          })() : (
            <p className="form13f-empty">No institutional holders found for this stock.</p>
          )}
        </div>

        {i.business_summary && (
          <div className="panel">
            <h3>Business Summary</h3>
            <p className={!showFullSummary ? 'clamped' : ''}>{i.business_summary}</p>
            <button className="summary-toggle" onClick={toggleSummary}>
              {showFullSummary ? 'Show less' : 'Show more'}
            </button>
          </div>
        )}

        {(thesisSummary || data.thesis_rule_eval) && (
          <div className="panel" id="sec-thesis">
            <h3>Thesis</h3>
            {thesisSummary && <p style={{ whiteSpace: 'pre-line' }}>{thesisSummary}</p>}
            {data.thesis_rule_eval && (() => {
              const ev = data.thesis_rule_eval;
              const bandLabel = (ev.target_weight_min_pct != null && ev.target_weight_max_pct != null)
                ? `${ev.target_weight_min_pct}–${ev.target_weight_max_pct}%` : null;
              return (
                <div className="thesis-rules">
                  <div className="thesis-rules-pills">
                    {ev.buy_signal && <span className="thesis-pill buy">Buy rules met</span>}
                    {ev.sell_signal && (
                      <span
                        className="thesis-pill sell"
                        title={`Escalates to EXIT after ${data.sell_rule_persistence} consecutive daily snapshots`}
                      >
                        Sell firing — day {data.thesis_sell_streak} of {data.sell_rule_persistence}
                      </span>
                    )}
                    {!ev.buy_signal && !ev.sell_signal && <span className="thesis-pill none">No rules firing</span>}
                    {ev.allocation_status && (
                      <span className="thesis-pill alloc" title={ev.allocation_reason || undefined}>
                        {ev.allocation_status.replace(/_/g, ' ')}{bandLabel ? ` · target ${bandLabel}` : ''}
                      </span>
                    )}
                  </div>
                  {ev.buy_rules_met?.length > 0 && (
                    <ul className="thesis-rules-list buy">
                      {ev.buy_rules_met.map((r, idx) => <li key={idx}>✓ {r.description}</li>)}
                    </ul>
                  )}
                  {ev.sell_rules_met?.length > 0 && (
                    <ul className="thesis-rules-list sell">
                      {ev.sell_rules_met.map((r, idx) => <li key={idx}>✗ {r.description}</li>)}
                    </ul>
                  )}
                </div>
              );
            })()}
          </div>
        )}

        {data.agent_suggestion && (() => {
          const s = data.agent_suggestion;
          const ageDays = Math.floor((Date.now() - new Date(s.date + 'T00:00:00').getTime()) / 86400000);
          const isStale = ageDays > 7;
          return (
            <div className="panel" id="sec-agent">
              <div className="earnings-panel-header">
                <div>
                  <h3>Agent Suggestion</h3>
                  <p className="earnings-panel-dates">
                    {s.date}
                    {ageDays > 0 && <> · {ageDays}d ago</>}
                    {isStale && <> · stale</>}
                    <> · {s.strategy}</>
                  </p>
                </div>
                <span
                  className={`action-badge action-${s.action}`}
                  style={isStale ? { opacity: 0.5 } : undefined}
                >
                  {s.action.toUpperCase()}
                </span>
              </div>
              <div className="agent-sugg-line">
                {s.value_gbp > 0 ? (
                  <>
                    <strong>{hideAmounts ? MASK : `£${s.value_gbp.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}</strong>
                    {s.weight_before != null && s.weight_after != null && (
                      <span className="agent-sugg-weights">
                        {(s.weight_before * 100).toFixed(1)}% → {(s.weight_after * 100).toFixed(1)}%
                      </span>
                    )}
                  </>
                ) : (
                  <strong>Vetoed by constraints</strong>
                )}
                <span className={`status-badge status-${s.status}`}>{s.status}</span>
                <Link to="/agent" className="agent-sugg-link">All suggestions →</Link>
              </div>
              {s.rationale?.trigger && <p className="agent-sugg-trigger">{s.rationale.trigger}</p>}
              {s.constraint_adjustments.length > 0 && (
                <ul className="agent-sugg-adjustments">
                  {s.constraint_adjustments.map((a, idx) => <li key={idx}>⚠ {a}</li>)}
                </ul>
              )}
            </div>
          );
        })()}

        <div className="panel" id="sec-price">
          <h3>Price {priceMetric === 'price_pct_change' ? '(%)' : ''}</h3>
          <PanelSpan span={priceSpan} />
          <div className="inline-controls" style={{ opacity: chartLoading ? 0.6 : 1, pointerEvents: chartLoading ? 'none' : 'auto' }}>
            <div className="stock-segmented">
              {PRICE_METRICS.map(metric => (
                <button
                  key={metric.value}
                  className={`stock-seg-btn${priceMetric === metric.value ? ' active' : ''}`}
                  onClick={() => handlePriceMetricChange(metric.value)}
                >
                  {metric.label}
                </button>
              ))}
            </div>
          </div>
          {chartData.length > 0 ? (
            <div style={{ position: 'relative' }}>
              {chartLoading && (
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  padding: '8px 16px',
                  borderRadius: '4px',
                  fontSize: '0.9rem',
                  color: '#6c757d',
                  zIndex: 10,
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  Loading...
                </div>
              )}
              <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis
                  domain={
                    priceMetric === 'price_pct_change'
                      ? ['auto', 'auto']
                      : [
                          (dataMin) => Math.floor(dataMin * 0.95),
                          (dataMax) => Math.ceil(dataMax * 1.02),
                        ]
                  }
                  tickFormatter={(v) =>
                    priceMetric === 'price_pct_change'
                      ? `${v.toFixed?.(1)}%`
                      : (typeof v === 'number'
                          ? v.toLocaleString(undefined, { maximumFractionDigits: 2 })
                          : v)
                  }
                />
                <Tooltip content={<PriceTooltip priceMetric={priceMetric} benchSymbols={benchSeries.map(b => b.sym)} />} />
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
                {orderMarkers.map(d => (
                  <ReferenceDot
                    key={d.originalDate}
                    x={d.date}
                    y={d.value}
                    r={d.order.radius}
                    fill={d.order.color}
                    fillOpacity={d.order.opacity}
                    stroke="#fff"
                    strokeWidth={2}
                    strokeOpacity={d.order.opacity}
                  />
                ))}
                {/* Add invisible lines for order legends */}
                {['BUY', 'SELL', 'DIVIDEND', 'CASH', 'ADMIN'].map(category => {
                  const hasTransactions = chartData.some(d => d.order?.category === category);
                  if (!hasTransactions) return null;

                  const categoryNames = {
                    BUY: 'Buy Orders',
                    SELL: 'Sell Orders',
                    DIVIDEND: 'Dividends',
                    CASH: 'Cash Movements',
                    ADMIN: 'Administrative'
                  };

                  return (
                    <Line
                      key={`order-legend-${category}`}
                      type="monotone"
                      dataKey={() => null}
                      data={[]}
                      stroke="none"
                      name={categoryNames[category]}
                      dot={{
                        ...LEGEND_DOT_PROPS,
                        fill: TRANSACTION_COLORS[category]
                      }}
                      connectNulls={false}
                      activeDot={false}
                    />
                  );
                })}
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
                {chartWindowDays >= 30 && (
                  <Line
                    type="monotone"
                    dataKey="sma50"
                    name="SMA 50"
                    stroke={SMA_50_COLOR}
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    dot={false}
                    activeDot={false}
                    connectNulls={false}
                  />
                )}
                {chartWindowDays >= 30 && (
                  <Line
                    type="monotone"
                    dataKey="sma200"
                    name="SMA 200"
                    stroke={SMA_200_COLOR}
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    dot={false}
                    activeDot={false}
                    connectNulls={false}
                  />
                )}
                {benchSeries.map(b => (
                  <Line
                    key={`bench-${b.sym}`}
                    type="monotone"
                    dataKey={b.sym}
                    name={b.sym}
                    stroke={BENCH_COLORS[b.sym] || BENCH_FALLBACK_COLOR}
                    strokeWidth={1.5}
                    strokeDasharray="2 3"
                    dot={false}
                    activeDot={false}
                    connectNulls
                  />
                ))}
              </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <div className="empty">No price data</div>}
        </div>

        <div className="panel" id="sec-pe">
          <h3>PE Ratio</h3>
          <PanelSpan span={peSpan} note={eventRangeFloored ? '1Y minimum for quarterly data' : null} />
          {peData.length > 0 ? (
            <div style={{ position: 'relative' }}>
              {chartLoading && (
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  padding: '8px 16px',
                  borderRadius: '4px',
                  fontSize: '0.9rem',
                  color: '#6c757d',
                  zIndex: 10,
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  Loading...
                </div>
              )}
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
                      const v = data.pe ?? data.peEstimate;

                      return (
                        <div className="custom-tooltip">
                          <p className="tooltip-label">{label}</p>
                          <p className="tooltip-item" style={{ color: '#e74c3c' }}>
                            <span className="color-indicator" style={{ backgroundColor: '#e74c3c' }}></span>
                            {data.isEstimate ? 'PE (estimate)' : 'PE Ratio'}: {typeof v === 'number' ? v.toFixed(2) : v}
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
                <Line
                  type="monotone"
                  dataKey="peEstimate"
                  name="PE (estimate)"
                  stroke="#e74c3c"
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  dot={{ r: 3 }}
                  activeDot={false}
                  connectNulls
                />
                {avgPeFromHistory != null && (
                  <ReferenceLine
                    y={avgPeFromHistory}
                    stroke="#94a3b8"
                    strokeDasharray="4 3"
                    strokeWidth={1.5}
                    label={{
                      value: `Avg ${Math.round(avgPeFromHistory)}`,
                      position: 'insideTopRight',
                      style: { fontSize: 11, fill: '#94a3b8' }
                    }}
                  />
                )}
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
            </div>
          ) : <div className="empty">No PE data</div>}
        </div>

        <div className="panel" id="sec-eps">
          <h3>EPS: Estimate vs Reported (with Surprise%)</h3>
          <PanelSpan span={epsSpan} note={eventRangeFloored ? '1Y minimum for quarterly data' : null} />
          {epsInRange.length > 0 ? (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={epsInRange}>
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

        <div className="panel" id="sec-cashflow">
          <h3>Cash Flow (OCF, CapEx, FCF)</h3>
          <PanelSpan span={cashflowSpan} note="all available (annual)" />
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

        <div className="panel" id="sec-analysts">
          <h3>Analyst Recommendations</h3>
          <PanelSpan span={recsSpan} note={eventRangeFloored ? '1Y minimum for monthly data' : null} />
          {recsInRange.length > 0 ? (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={recsInRange}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis tickFormatter={(v) => `${v}`} />
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

        {/* Earnings Reports Section */}
        {data?.earnings_reports && data.earnings_reports.length > 0 && (() => {
          const report = data.earnings_reports[Math.min(activeReportIdx, data.earnings_reports.length - 1)];
          const metrics = report.metrics || {};
          // SEC filings rarely restate guidance/consensus; fall back to the pr_*
          // fields carried over from the same-day press release on supersede.
          const hasValues = (v) => {
            if (v == null) return false;
            if (Array.isArray(v)) return v.some(hasValues);
            if (typeof v === 'object') return Object.values(v).some(hasValues);
            return true;
          };
          const guidanceFromPR = !hasValues(metrics.guidance) && hasValues(metrics.pr_guidance);
          const guidance = (guidanceFromPR ? metrics.pr_guidance : metrics.guidance) || {};
          const consensusFromPR = !hasValues(metrics.consensus_comparison) && hasValues(metrics.pr_consensus_comparison);
          const consensus = (consensusFromPR ? metrics.pr_consensus_comparison : metrics.consensus_comparison) || {};
          const epsGuidance = guidance.eps_guidance || {};
          const revenueGuidance = guidance.revenue_guidance || {};
          const assessment = metrics.investment_assessment || {};

          const sinceEarningsPct = (() => {
            if (!report.price_at_announcement || !f.currentPrice) return null;
            return ((f.currentPrice - report.price_at_announcement) / report.price_at_announcement) * 100;
          })();

          const metricCards = [
            consensus.eps_beat_pct != null && {
              label: 'EPS',
              badge: consensus.eps_beat_pct >= 0
                ? `Beat +${consensus.eps_beat_pct.toFixed(0)}%`
                : `Miss ${consensus.eps_beat_pct.toFixed(0)}%`,
              type: consensus.eps_beat_pct >= 0 ? 'beat' : 'miss',
              sub: consensusFromPR ? 'from press release' : null,
            },
            consensus.revenue_beat_pct != null && {
              label: 'Revenue',
              badge: consensus.revenue_beat_pct >= 0
                ? `Beat +${consensus.revenue_beat_pct.toFixed(0)}%`
                : `Miss ${consensus.revenue_beat_pct.toFixed(0)}%`,
              type: consensus.revenue_beat_pct >= 0 ? 'beat' : 'miss',
              sub: consensusFromPR ? 'from press release' : null,
            },
            consensus.guidance_vs_consensus && {
              label: 'Guidance',
              badge: consensus.guidance_vs_consensus === 'above' ? '↑ Above' : consensus.guidance_vs_consensus === 'below' ? '↓ Below' : '→ In-Line',
              type: consensus.guidance_vs_consensus === 'above' ? 'beat' : consensus.guidance_vs_consensus === 'below' ? 'miss' : 'inline',
              sub: consensusFromPR ? 'from press release' : null,
            },
            assessment.recommendation && {
              label: 'Signal',
              badge: assessment.recommendation.charAt(0).toUpperCase() + assessment.recommendation.slice(1),
              type: `rec-${assessment.recommendation}`,
              sub: assessment.conviction ? `${assessment.conviction} conviction` : null,
            },
            sinceEarningsPct != null && {
              label: 'Since Earnings',
              badge: `${sinceEarningsPct >= 0 ? '+' : ''}${sinceEarningsPct.toFixed(1)}%`,
              type: sinceEarningsPct >= 0 ? 'beat' : 'miss',
              sub: report.announcement_date
                ? `from $${report.price_at_announcement.toFixed(2)} on ${new Date(report.announcement_date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`
                : `from $${report.price_at_announcement.toFixed(2)}`,
            },
          ].filter(Boolean);

          const periodFmt = new Date(report.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
          const announcedFmt = report.announcement_date
            ? new Date(report.announcement_date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
            : null;
          const pendingFmt = pendingEarningsDate
            ? new Date(pendingEarningsDate + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
            : null;

          return (
            <div className="panel" id="sec-reports">
              <div className="earnings-panel-header">
                <div>
                  <h3>Earnings Reports</h3>
                  <p className="earnings-panel-dates">
                    Period ends {periodFmt}
                    {announcedFmt && <> · Announced {announcedFmt}</>}
                  </p>
                </div>
                {report.metrics?.source_url ? (
                  <a
                    className="earnings-report-link-btn"
                    href={report.metrics.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="Open original filing/press release"
                  >
                    Full report ↗
                  </a>
                ) : (
                  <button
                    className="earnings-report-link-btn"
                    onClick={() => openReportModal(symbol, report.date)}
                    title="Open full report"
                  >
                    Full report ↗
                  </button>
                )}
              </div>

              {(data.earnings_reports.length > 1 || pendingEarningsDate) && (
                <div className="earnings-tabs">
                  {pendingEarningsDate && (
                    <button
                      className="earnings-tab earnings-tab-pending"
                      disabled
                      title={`Reported ${pendingFmt} · report not available yet`}
                    >
                      {new Date(pendingEarningsDate + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
                    </button>
                  )}
                  {data.earnings_reports.map((r, idx) => {
                    // Label by period-end month — calendar-quarter labels are wrong
                    // for companies with off-calendar fiscal years (e.g. FY ending Oct)
                    const d = new Date(r.date + 'T00:00:00');
                    const label = d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
                    return (
                      <button
                        key={r.id}
                        className={`earnings-tab${idx === activeReportIdx ? ' active' : ''}`}
                        onClick={() => setActiveReportIdx(idx)}
                        title={`Period ending ${r.date}${r.announcement_date ? ` · Announced ${r.announcement_date}` : ''}`}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
              )}

              {pendingEarningsDate && (
                <p className="earnings-pending-note">
                  Reported {pendingFmt} — summary pending, waiting on the filing.
                </p>
              )}

              {metricCards.length > 0 && (
                <div className="earnings-metric-cards">
                  {metricCards.map((card, idx) => (
                    <div key={idx} className={`earnings-metric-card emc-${card.type}`}>
                      <div className="emc-label">{card.label}</div>
                      <div className="emc-badge">{card.badge}</div>
                      {card.sub && <div className="emc-sub">{card.sub}</div>}
                    </div>
                  ))}
                </div>
              )}

              {assessment.rationale && (
                <p className="earnings-rationale">{assessment.rationale}</p>
              )}

              {(assessment.key_catalysts?.length > 0 || assessment.key_concerns?.length > 0) && (
                <div className="earnings-catalysts-concerns">
                  {assessment.key_catalysts?.length > 0 && (
                    <div className="earnings-catalysts">
                      <h5>Catalysts</h5>
                      <ul>
                        {assessment.key_catalysts.map((c, i) => <li key={i}>{c}</li>)}
                      </ul>
                    </div>
                  )}
                  {assessment.key_concerns?.length > 0 && (
                    <div className="earnings-concerns">
                      <h5>Concerns</h5>
                      <ul>
                        {assessment.key_concerns.map((c, i) => <li key={i}>{c}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {(epsGuidance.next_quarter || epsGuidance.next_year || revenueGuidance.next_quarter || revenueGuidance.next_year || guidance.operating_margin_guidance != null || guidance.outlook_commentary) && (
                <div className="earnings-guidance">
                  <h5>
                    Guidance
                    {guidanceFromPR && (
                      <span
                        className="guidance-pr-note"
                        title="From the same-day earnings press release — the SEC filing did not restate guidance"
                      >
                        press release
                      </span>
                    )}
                  </h5>
                  <div className="guidance-metrics">
                    {epsGuidance.next_year && (
                      <div className="guidance-metric">
                        <span className="guidance-label">EPS (Full Year):</span>
                        <span className="guidance-value">${epsGuidance.next_year.toFixed(2)}</span>
                        {epsGuidance.growth_pct != null && (
                          <span className={`guidance-growth ${epsGuidance.growth_pct >= 0 ? 'positive' : 'negative'}`}>
                            {epsGuidance.growth_pct >= 0 ? '+' : ''}{epsGuidance.growth_pct.toFixed(1)}%
                          </span>
                        )}
                      </div>
                    )}
                    {epsGuidance.next_quarter && (
                      <div className="guidance-metric">
                        <span className="guidance-label">EPS (Next Q):</span>
                        <span className="guidance-value">${epsGuidance.next_quarter.toFixed(2)}</span>
                      </div>
                    )}
                    {revenueGuidance.next_quarter && (
                      <div className="guidance-metric">
                        <span className="guidance-label">Revenue (Next Q):</span>
                        <span className="guidance-value">${(revenueGuidance.next_quarter / 1000).toFixed(2)}B</span>
                      </div>
                    )}
                    {revenueGuidance.next_year && (
                      <div className="guidance-metric">
                        <span className="guidance-label">Revenue (FY):</span>
                        <span className="guidance-value">${(revenueGuidance.next_year / 1000).toFixed(2)}B</span>
                        {revenueGuidance.growth_pct != null && (
                          <span className={`guidance-growth ${revenueGuidance.growth_pct >= 0 ? 'positive' : 'negative'}`}>
                            {revenueGuidance.growth_pct >= 0 ? '+' : ''}{revenueGuidance.growth_pct.toFixed(1)}%
                          </span>
                        )}
                      </div>
                    )}
                    {guidance.operating_margin_guidance != null && (
                      <div className="guidance-metric">
                        <span className="guidance-label">Op. Margin:</span>
                        <span className="guidance-value">{guidance.operating_margin_guidance.toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                  {guidance.outlook_commentary && (
                    <p className="guidance-outlook">{guidance.outlook_commentary}</p>
                  )}
                </div>
              )}

              {report.summary && (
                <div className="earnings-summary">
                  <div
                    className="earnings-summary-content"
                    dangerouslySetInnerHTML={{ __html: markdownToHtml(report.summary) }}
                  />
                </div>
              )}
            </div>
          );
        })()}

        {/* Position Review Section (LLM thesis review) */}
        {data?.position_review && (() => {
          const rv = data.position_review;
          const label = rv.position_action === 'no_action'
            ? 'No action'
            : rv.position_action.charAt(0).toUpperCase() + rv.position_action.slice(1);
          const ageDays = rv.reviewed_at
            ? Math.floor((Date.now() - new Date(rv.reviewed_at + 'T00:00:00').getTime()) / 86400000)
            : null;
          const isStale = ageDays != null && ageDays > 30;
          const qt = rv.quality_trajectory || {};
          const triggers = rv.trigger_evaluations || [];
          return (
            <div className="panel" id="sec-review">
              <div className="earnings-panel-header">
                <div>
                  <h3>Position Review</h3>
                  <p className="earnings-panel-dates">
                    Reviewed {rv.reviewed_at}
                    {ageDays != null && <> · {ageDays}d ago</>}
                    {isStale && <> · stale</>}
                    {rv.model && <> · {rv.model}</>}
                  </p>
                </div>
                <span
                  className={`review-badge rv-${rv.position_action}`}
                  style={isStale ? { opacity: 0.5 } : undefined}
                >
                  {label}
                </span>
              </div>

              {rv.headline && <p className="review-headline">{rv.headline}</p>}

              <div className="earnings-metric-cards">
                <div className={`earnings-metric-card emc-${rv.thesis_status === 'intact' ? 'beat' : rv.thesis_status === 'broken' ? 'miss' : 'inline'}`}>
                  <div className="emc-label">Thesis</div>
                  <div className="emc-badge">{rv.thesis_status.replace(/_/g, ' ')}</div>
                </div>
                <div className="earnings-metric-card emc-inline">
                  <div className="emc-label">Strength</div>
                  <div className="emc-badge">{rv.signal_strength}</div>
                </div>
                <div className="earnings-metric-card emc-inline">
                  <div className="emc-label">Confidence</div>
                  <div className="emc-badge">{Math.round(rv.confidence * 100)}%</div>
                </div>
                <div className="earnings-metric-card emc-inline">
                  <div className="emc-label">Quality</div>
                  <div className="emc-badge">ROIC {qt.roic_trend} · margins {qt.margin_trend}</div>
                </div>
              </div>

              {triggers.length > 0 && (
                <div className="review-triggers">
                  <h5>Trigger evaluations</h5>
                  <ul>
                    {triggers.map((t, i) => (
                      <li key={i} className="trigger-item">
                        <span className={`trigger-status ts-${t.status} tt-${t.trigger_type}`}>
                          {t.status.replace(/_/g, ' ')}
                        </span>
                        <span className={`trigger-type tt-${t.trigger_type}`}>{t.trigger_type}</span>
                        <span className="trigger-text" title={t.evidence}>{t.trigger_text}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {(rv.catalysts?.length > 0 || rv.risk_flags?.length > 0) && (
                <div className="earnings-catalysts-concerns">
                  {rv.catalysts?.length > 0 && (
                    <div className="earnings-catalysts">
                      <h5>Catalysts</h5>
                      <ul>
                        {rv.catalysts.map((c, i) => <li key={i}>{c}</li>)}
                      </ul>
                    </div>
                  )}
                  {rv.risk_flags?.length > 0 && (
                    <div className="earnings-concerns">
                      <h5>Risk flags</h5>
                      <ul>
                        {rv.risk_flags.map((c, i) => <li key={i}>{c}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {rv.rationale && (
                <div className="earnings-summary">
                  <div
                    className="earnings-summary-content"
                    dangerouslySetInnerHTML={{ __html: markdownToHtml(rv.rationale) }}
                  />
                </div>
              )}
            </div>
          );
        })()}

        {/* News Section */}
        <div className="panel" id="sec-news">
          <h3>Latest News</h3>
          {newsArticles && newsArticles.length > 0 ? (
            <div className="news-list">
              {newsArticles.map((article, index) => {
                const content = article.content;
                const thumbnail = content?.thumbnail;
                const thumbnailUrl = thumbnail?.originalUrl;
                const pubDate = new Date(content?.pubDate);
                const timeAgo = Math.max(0, Math.floor((Date.now() - pubDate) / (1000 * 60))); // minutes ago
                const displayTime = timeAgo < 60 ? `${timeAgo}m ago` :
                                  timeAgo < 1440 ? `${Math.floor(timeAgo / 60)}h ago` :
                                  `${Math.floor(timeAgo / 1440)}d ago`;

                return (
                  <a
                    key={article.id || index}
                    href={content?.clickThroughUrl?.url || content?.canonicalUrl?.url || '#'}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="news-item"
                  >
                    {thumbnailUrl ? (
                      <div className="news-thumbnail">
                        <img src={thumbnailUrl} alt={content?.title} loading="lazy" decoding="async" />
                      </div>
                    ) : (
                      <div className="news-thumbnail news-thumbnail-placeholder">
                        {(content?.provider?.displayName || '·').charAt(0)}
                      </div>
                    )}
                    <div className="news-content">
                      <div className="news-title">{content?.title}</div>
                      {content?.summary && (
                        <div className="news-summary">{content.summary}</div>
                      )}
                      <div className="news-meta">
                        <span className="news-provider">{content?.provider?.displayName || 'Unknown'}</span>
                        <span className="news-time">{displayTime}</span>
                        {content?.metadata?.editorsPick && (
                          <span className="news-badge">Editor&apos;s Pick</span>
                        )}
                      </div>
                    </div>
                  </a>
                );
              })}
            </div>
          ) : (
            <div className="empty">No news available</div>
          )}
          {newsArticles && newsArticles.length > 0 && (
            <div className="news-footer">
              <a
                href={`https://finance.yahoo.com/quote/${symbol}/news/`}
                target="_blank"
                rel="noopener noreferrer"
                className="news-more-link"
              >
                More news →
              </a>
            </div>
          )}
        </div>
      </div>

      {/* Earnings Report Modal */}
      {selectedReport && (
        <div className="modal-overlay" onClick={closeReportModal}>
          <div className="modal-content earnings-report-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Earnings Report - {selectedReport.symbol} - {new Date(selectedReport.reportDate).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</h2>
              <button className="modal-close" onClick={closeReportModal}>&times;</button>
            </div>
            <div className="modal-body">
              {loadingReport ? (
                <div className="loading">Loading report...</div>
              ) : reportHtml ? (
                <iframe
                  srcDoc={reportHtml}
                  className="earnings-report-iframe"
                  title="Earnings Report"
                  sandbox="allow-same-origin"
                />
              ) : (
                <div className="error">Failed to load report</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Stock;


