import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { portfolioAPI } from '../services/api';
import { useHideAmounts, MASK } from '../context/HideAmountsContext';
import './Transactions.css';

// ── Constants ─────────────────────────────────────────────────────────────────

const PAGE_SIZE = 50;

const ACTION_META = {
  'Market buy':                              { label: 'Buy',       cls: 'action-buy' },
  'Limit buy':                               { label: 'Buy',       cls: 'action-buy' },
  'Market sell':                             { label: 'Sell',      cls: 'action-sell' },
  'Limit sell':                              { label: 'Sell',      cls: 'action-sell' },
  'Dividend (Dividend)':                     { label: 'Dividend',  cls: 'action-dividend' },
  'Dividend (Property income distribution)': { label: 'Dividend',  cls: 'action-dividend' },
  'Dividend (Tax exempted)':                 { label: 'Dividend',  cls: 'action-dividend' },
  'Deposit':                                 { label: 'Deposit',   cls: 'action-deposit' },
  'Withdrawal':                              { label: 'Withdrawal',cls: 'action-withdraw' },
  'Interest on cash':                        { label: 'Interest',  cls: 'action-interest' },
  'Stock split open':                        { label: 'Split',     cls: 'action-admin' },
  'Stock split close':                       { label: 'Split',     cls: 'action-admin' },
  'Result adjustment':                       { label: 'Adjust',    cls: 'action-admin' },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

const fmtMoney = (v, hide, { sign = false } = {}) => {
  if (hide) return MASK;
  if (v == null) return '—';
  const abs = Math.abs(v).toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const prefix = sign && v > 0 ? '+£' : sign && v < 0 ? '-£' : '£';
  return `${prefix}${abs}`;
};

// Plain number formatting (no £ symbol) — used when the column header already says (£)
const fmtAmt = (v, hide, { sign = false } = {}) => {
  if (hide) return MASK;
  if (v == null) return '—';
  const abs = Math.abs(v).toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (sign) return `${v > 0 ? '+' : v < 0 ? '−' : ''}${abs}`;
  return abs;
};

const CURRENCY_SYMBOLS = { GBP: '£', USD: '$', EUR: '€', GBX: 'p', CHF: 'Fr', JPY: '¥', CAD: 'CA$', AUD: 'A$', SEK: 'kr', NOK: 'kr', DKK: 'kr', HKD: 'HK$', SGD: 'S$' };
const currencySymbol = (code) => CURRENCY_SYMBOLS[code] || (code ? `${code} ` : '');

const fmtDate = (iso) =>
  new Date(iso).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });

const fmtMonthLabel = (ym) => {
  const [y, m] = ym.split('-');
  return new Date(Number(y), Number(m) - 1).toLocaleDateString('en-GB', { month: 'short', year: '2-digit' });
};

// ── Sub-components ────────────────────────────────────────────────────────────

const SummaryCard = ({ label, value, positive, negative, sub }) => (
  <div className="txn-summary-card">
    <div className="txn-summary-label">{label}</div>
    <div className={`txn-summary-value${positive ? ' positive' : negative ? ' negative' : ''}`}>{value}</div>
    {sub && <div className="txn-summary-sub">{sub}</div>}
  </div>
);

const ISACard = ({ used, total, remaining, taxYear, hide }) => {
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  const isNearLimit = pct >= 80;
  const limitLabel = `£${(total / 1000).toFixed(0)}k`;
  return (
    <div className="txn-summary-card">
      <div className="txn-summary-label">ISA Allowance {taxYear}</div>
      <div className={`txn-summary-value${isNearLimit ? ' negative' : ''}`}>
        {hide ? '****' : `£${remaining.toLocaleString('en-GB', { maximumFractionDigits: 0 })}`}
        <span className="txn-summary-isa-sub"> left of {limitLabel}</span>
      </div>
      <div className="txn-summary-sub">
        {hide
          ? '**** of **** used'
          : `£${used.toLocaleString('en-GB', { maximumFractionDigits: 0 })} of £${total.toLocaleString('en-GB', { maximumFractionDigits: 0 })} used`}
      </div>
      <div className="txn-isa-bar-wrap">
        <div className={`txn-isa-bar-fill${isNearLimit ? ' near-limit' : ''}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
};

const DividendTooltip = ({ active, payload, label, hide }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="txn-tooltip">
      <div className="txn-tooltip-label">{fmtMonthLabel(label)}</div>
      <div className="txn-tooltip-value positive">{fmtMoney(payload[0]?.value, hide)}</div>
    </div>
  );
};

// ── Main component ────────────────────────────────────────────────────────────

const Transactions = () => {
  const { hideAmounts } = useHideAmounts();
  const [data, setData]         = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [yearFilter, setYear]   = useState('');
  const [typeFilter, setType]   = useState('all');
  const [search, setSearch]     = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo]     = useState('');
  const [page, setPage]         = useState(1);

  useEffect(() => {
    portfolioAPI.getTransactions()
      .then(setData)
      .catch(() => setError('Failed to load transactions'))
      .finally(() => setLoading(false));
  }, []);

  // ── Filtered transaction list (client-side) ────────────────────────────────
  const filtered = useMemo(() => {
    if (!data) return [];
    let txns = data.transactions;

    if (yearFilter)
      txns = txns.filter(t => t.date.startsWith(yearFilter));

    // Date range: t.date is ISO string "YYYY-MM-DDTHH:mm:ss", inputs are "YYYY-MM-DD"
    if (dateFrom)
      txns = txns.filter(t => t.date.slice(0, 10) >= dateFrom);
    if (dateTo)
      txns = txns.filter(t => t.date.slice(0, 10) <= dateTo);

    if (typeFilter === 'dividends')
      txns = txns.filter(t => t.action.startsWith('Dividend'));
    else if (typeFilter === 'trades')
      txns = txns.filter(t => t.action.includes('buy') || t.action.includes('sell'));
    else if (typeFilter === 'cash')
      txns = txns.filter(t => ['Deposit', 'Withdrawal', 'Interest on cash'].includes(t.action));

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      txns = txns.filter(t =>
        (t.ticker || '').toLowerCase().includes(q) ||
        (t.name  || '').toLowerCase().includes(q)
      );
    }
    return txns;
  }, [data, yearFilter, typeFilter, search, dateFrom, dateTo]);

  const paginated  = useMemo(() => filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE), [filtered, page]);
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);

  // ── Summary recomputed whenever any date filter changes ───────────────────
  const summary = useMemo(() => {
    if (!data) return null;
    // No filters at all → use pre-computed backend summary
    if (!yearFilter && !dateFrom && !dateTo) return data.summary;
    let txns = data.transactions;
    if (yearFilter) txns = txns.filter(t => t.date.startsWith(yearFilter));
    if (dateFrom)   txns = txns.filter(t => t.date.slice(0, 10) >= dateFrom);
    if (dateTo)     txns = txns.filter(t => t.date.slice(0, 10) <= dateTo);
    let divs = 0, gains = 0, fees = 0, interest = 0, deposited = 0;
    for (const t of txns) {
      fees += t.fees || 0;
      if (t.action.startsWith('Dividend'))               divs      += t.total;
      else if (t.action === 'Deposit')                   deposited += t.total;
      else if (t.action === 'Interest on cash')          interest  += t.total;
      if (t.result != null && t.action.includes('sell')) gains     += t.result;
    }
    return {
      total_dividends: divs, total_realized_gains: gains,
      total_fees: fees, total_interest: interest, total_deposited: deposited,
    };
  }, [data, yearFilter, dateFrom, dateTo]);

  // ── ISA allowance for the relevant tax year ───────────────────────────────
  // ISA tax year = 6 Apr Y → 5 Apr Y+1.
  // Default view (no year selected) uses backend-computed current tax year values.
  // When a calendar year is selected, we approximate by summing deposits inside that tax year window.
  const isaData = useMemo(() => {
    if (!data) return null;

    // Default view: rely on backend-computed current tax year values.
    if (!yearFilter) {
      const total = data.summary.isa_allowance_total ?? 20000;
      const used = data.summary.isa_allowance_used ?? 0;
      const remaining = data.summary.isa_allowance_remaining ?? Math.max(0, total - used);
      const taxYear = data.summary.isa_tax_year ?? '—';
      return { used, total, remaining, taxYear };
    }

    const ISA_LIMIT = data.summary.isa_allowance_total ?? 20000;
    const isaYear = parseInt(yearFilter, 10);
    if (!Number.isFinite(isaYear)) return null;
    const startStr = `${isaYear}-04-06`;
    const endStr   = `${isaYear + 1}-04-05`;
    const taxYear  = `${isaYear}/${String(isaYear + 1).slice(2)}`;
    let used = 0;
    for (const t of data.transactions) {
      if (t.action === 'Deposit') {
        const d = t.date.slice(0, 10);
        if (d >= startStr && d <= endStr) used += t.total;
      }
    }
    used = Math.round(used * 100) / 100;
    return { used, total: ISA_LIMIT, remaining: Math.max(0, Math.round((ISA_LIMIT - used) * 100) / 100), taxYear };
  }, [data, yearFilter]);

  // ── Dividend chart clipped to last 24 months (or selected year) ───────────
  const chartData = useMemo(() => {
    if (!data) return [];
    if (yearFilter) return data.dividends_chart.filter(d => d.month.startsWith(yearFilter));
    return data.dividends_chart.slice(-24);
  }, [data, yearFilter]);

  // ── Average monthly dividend for reference line ───────────────────────────
  const avgMonthlyDividend = useMemo(() =>
    chartData.length > 0 ? chartData.reduce((s, d) => s + d.amount, 0) / chartData.length : 0,
  [chartData]);

  // ── Filtered totals shown in toolbar ──────────────────────────────────────
  const filteredTotal = useMemo(() => {
    let amount = 0, fees = 0, pnl = 0, hasPnl = false;
    for (const t of filtered) {
      if (t.total   != null) amount += t.total;
      if (t.fees     > 0)    fees   += t.fees;
      if (t.result  != null) { pnl  += t.result; hasPnl = true; }
    }
    return { amount, fees, pnl, hasPnl };
  }, [filtered]);

  // ── CSV export of the current filtered view ───────────────────────────────
  const exportCSV = () => {
    const headers = ['Date', 'Ticker', 'Name', 'Type', 'Shares', 'Price (local)', 'Total (£)', 'Realized P&L (£)', 'Fees (£)'];
    const rows = filtered.map(txn => {
      const meta = ACTION_META[txn.action] || { label: txn.action };
      return [
        fmtDate(txn.date),
        txn.ticker || '',
        txn.name || txn.notes || '',
        meta.label,
        txn.quantity != null ? Math.abs(txn.quantity)  : '',
        txn.price    != null ? (() => {
          const isGBX = txn.currency === 'GBX';
          return `${isGBX ? '£' : currencySymbol(txn.currency)}${(isGBX ? txn.price / 100 : txn.price).toFixed(4)}`;
        })() : '',
        txn.total    != null ? txn.total.toFixed(2)    : '',
        txn.result   != null ? txn.result.toFixed(2)   : '',
        txn.fees > 0         ? (-txn.fees).toFixed(2)  : '',
      ];
    });
    const csv = [headers, ...rows]
      .map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(','))
      .join('\n');
    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `transactions${yearFilter ? '_' + yearFilter : '_all'}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Reset page when filters change ────────────────────────────────────────
  const setFilter = (setter) => (val) => { setter(val); setPage(1); };

  // ── Render ─────────────────────────────────────────────────────────────────
  if (loading) return <div className="page-fixed txn-container"><div className="txn-state">Loading…</div></div>;
  if (error)   return <div className="page-fixed txn-container"><div className="txn-state txn-error">{error}</div></div>;
  if (!data)   return null;

  const s = summary;
  const totalTxns = data.transactions.length;

  return (
    <div className="page-fixed txn-container">

      {/* Header */}
      <div className="txn-header">
        <div>
          <h2>Transaction History</h2>
          <p className="txn-subtitle">{totalTxns.toLocaleString()} transactions total</p>
        </div>
        <select
          className="txn-year-select"
          value={yearFilter}
          onChange={e => setFilter(setYear)(e.target.value)}
        >
          <option value="">All time</option>
          {data.years.map(y => <option key={y} value={String(y)}>{y}</option>)}
        </select>
      </div>

      {/* Summary cards */}
      <div className="txn-summary-row">
        <SummaryCard
          label="Dividends Received"
          value={fmtMoney(s.total_dividends, hideAmounts)}
          positive={s.total_dividends > 0}
          sub={`${(data.top_dividend_payers || []).length} paying stocks`}
        />
        <SummaryCard
          label="Realized Gains"
          value={fmtMoney(s.total_realized_gains, hideAmounts, { sign: true })}
          positive={s.total_realized_gains > 0}
          negative={s.total_realized_gains < 0}
        />
        <SummaryCard
          label="Interest Earned"
          value={fmtMoney(s.total_interest, hideAmounts)}
          positive={s.total_interest > 0}
        />
        <SummaryCard
          label="Fees Paid"
          value={fmtMoney(s.total_fees, hideAmounts)}
          negative={s.total_fees > 0}
        />
        <SummaryCard
          label="Total Deposited"
          value={fmtMoney(s.total_deposited, hideAmounts)}
        />
        {isaData && (
          <ISACard
            used={isaData.used}
            total={isaData.total}
            remaining={isaData.remaining}
            taxYear={isaData.taxYear}
            hide={hideAmounts}
          />
        )}
      </div>

      {/* Charts row */}
      <div className="txn-charts-row">

        {/* Dividend income chart */}
        <div className="txn-panel txn-chart-panel">
          <h3>Dividend Income — {yearFilter || 'Last 24 Months'}</h3>
          {chartData.length > 0 ? (
            <div className="txn-chart-grow">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                  <XAxis
                    dataKey="month"
                    tickFormatter={fmtMonthLabel}
                    tick={{ fontSize: 11 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tickFormatter={v => (hideAmounts ? MASK : `£${v}`)}
                    tick={{ fontSize: 11 }}
                    width={55}
                  />
                  <Tooltip content={<DividendTooltip hide={hideAmounts} />} />
                  {!hideAmounts && avgMonthlyDividend > 0 && (
                    <ReferenceLine
                      y={avgMonthlyDividend}
                      stroke="#adb5bd"
                      strokeDasharray="4 4"
                      label={{ value: `avg £${avgMonthlyDividend.toFixed(0)}`, position: 'insideTopRight', fontSize: 10, fill: '#6c757d' }}
                    />
                  )}
                  <Bar dataKey="amount" fill="#28a745" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="txn-empty">No dividends in this period</div>
          )}
        </div>

        {/* Top dividend payers */}
        <div className="txn-panel">
          <h3>Top {Math.min(data.top_dividend_payers.length, 10)} Dividend Payers</h3>
          <div className="txn-payers-list">
            {data.top_dividend_payers.slice(0, 10).map(p => (
              <div key={p.ticker} className="txn-payer-row">
                <div className="txn-payer-left">
                  <Link to={`/stock/${p.ticker}`} className="txn-payer-ticker">{p.ticker}</Link>
                  <span className="txn-payer-name">{p.name}</span>
                </div>
                <div className="txn-payer-right">
                  <span className="positive" style={{ fontWeight: 600, fontSize: '0.85rem' }}>
                    {fmtMoney(p.total, hideAmounts)}
                  </span>
                  <span className="txn-payer-count">{p.count}×</span>
                </div>
              </div>
            ))}
            {data.top_dividend_payers.length === 0 && (
              <div className="txn-empty">No dividends recorded</div>
            )}
          </div>
        </div>
      </div>

      {/* Transaction table */}
      <div className="txn-table-panel">
        <div className="txn-table-toolbar">
          <div className="txn-type-pills">
            {[
              { id: 'all',      label: 'All' },
              { id: 'trades',   label: 'Trades' },
              { id: 'dividends',label: 'Dividends' },
              { id: 'cash',     label: 'Cash' },
            ].map(({ id, label }) => (
              <button
                key={id}
                className={`txn-pill${typeFilter === id ? ' active' : ''}`}
                onClick={() => setFilter(setType)(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <input
            className="txn-search"
            placeholder="Search ticker or name…"
            value={search}
            onChange={e => setFilter(setSearch)(e.target.value)}
          />
          <div className="txn-date-range">
            <input
              type="date"
              className="txn-date-input"
              value={dateFrom}
              onChange={e => setFilter(setDateFrom)(e.target.value)}
              title="From date"
            />
            <span className="txn-date-sep">–</span>
            <input
              type="date"
              className="txn-date-input"
              value={dateTo}
              onChange={e => setFilter(setDateTo)(e.target.value)}
              title="To date"
            />
            {(dateFrom || dateTo) && (
              <button
                className="txn-date-clear"
                onClick={() => { setDateFrom(''); setDateTo(''); setPage(1); }}
                title="Clear date range"
              >✕</button>
            )}
          </div>
          <span className="txn-result-count">
            {filtered.length.toLocaleString()} result{filtered.length !== 1 ? 's' : ''}
            {!hideAmounts && typeFilter === 'dividends' && filteredTotal.amount !== 0 && (
              <> · Total: <strong className="positive">{fmtMoney(filteredTotal.amount, false)}</strong></>
            )}
            {!hideAmounts && typeFilter === 'cash' && filteredTotal.amount !== 0 && (
              <> · Net: <strong>{fmtMoney(filteredTotal.amount, false)}</strong></>
            )}
            {!hideAmounts && typeFilter === 'trades' && filteredTotal.hasPnl && (
              <> · Realized P&amp;L: <strong className={filteredTotal.pnl >= 0 ? 'positive' : 'negative'}>
                {fmtMoney(filteredTotal.pnl, false, { sign: true })}
              </strong></>
            )}
          </span>
          <button className="txn-export-btn" onClick={exportCSV} title="Export filtered transactions to CSV">
            ↓ Export CSV
          </button>
        </div>

        <div className="txn-table-wrap">
          <table className="txn-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Stock</th>
                <th>Type</th>
                <th className="r">Shares</th>
                <th className="r">Price</th>
                <th className="r">Total (£)</th>
                <th className="r">P&amp;L (£)</th>
                <th className="r">Fees (£)</th>
              </tr>
            </thead>
            <tbody>
              {paginated.map(txn => {
                const meta = ACTION_META[txn.action] || { label: txn.action, cls: 'action-admin' };
                const isBuy  = txn.action.includes('buy');
                const isSell = txn.action.includes('sell');
                const isDividend = txn.action.startsWith('Dividend');
                return (
                  <tr key={txn.id}>
                    <td className="txn-date">{fmtDate(txn.date)}</td>
                    <td className="txn-stock-cell">
                      {txn.ticker ? (
                        <>
                          <Link to={`/stock/${txn.ticker}`} className="txn-ticker">{txn.ticker}</Link>
                          {txn.name && <span className="txn-stock-name">{txn.name}</span>}
                        </>
                      ) : (
                        <span className="txn-stock-name">{txn.notes || '—'}</span>
                      )}
                    </td>
                    <td><span className={`txn-badge ${meta.cls}`}>{meta.label}</span></td>
                    <td className="r txn-mono">
                      {hideAmounts ? MASK : txn.quantity
                        ? Math.abs(txn.quantity).toLocaleString('en-GB', { maximumFractionDigits: 4 })
                        : '—'}
                    </td>
                    <td className="r txn-mono">
                      {txn.price != null ? (() => {
                        // GBX (pence) → convert to £ for consistency with Total column
                        const isGBX = txn.currency === 'GBX';
                        const price = isGBX ? txn.price / 100 : txn.price;
                        const sym   = isGBX ? '£' : currencySymbol(txn.currency);
                        return (
                          <span title={txn.currency || ''}>
                            {sym}{price.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 4 })}
                          </span>
                        );
                      })() : '—'}
                    </td>
                    <td className={`r txn-mono${isBuy ? ' negative' : isSell || isDividend ? ' positive' : ''}`}>
                      {fmtAmt(txn.total, hideAmounts)}
                    </td>
                    <td className="r txn-mono">
                      {txn.result != null ? (
                        <span className={txn.result >= 0 ? 'positive' : 'negative'}>
                          {fmtAmt(txn.result, hideAmounts, { sign: true })}
                        </span>
                      ) : '—'}
                    </td>
                    <td className="r txn-mono">
                      {txn.fees > 0
                        ? <span className="negative">{hideAmounts ? MASK : `−${txn.fees.toFixed(2)}`}</span>
                        : '—'}
                    </td>
                  </tr>
                );
              })}
              {paginated.length === 0 && (
                <tr>
                  <td colSpan={8} className="txn-empty-row">No transactions match filters</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {totalPages > 1 && (
          <div className="txn-pagination">
            <button disabled={page === 1}          onClick={() => setPage(1)}>«</button>
            <button disabled={page === 1}          onClick={() => setPage(p => p - 1)}>‹</button>
            <span>Page {page} of {totalPages}</span>
            <button disabled={page === totalPages} onClick={() => setPage(p => p + 1)}>›</button>
            <button disabled={page === totalPages} onClick={() => setPage(totalPages)}>»</button>
          </div>
        )}
      </div>

    </div>
  );
};

export default Transactions;
