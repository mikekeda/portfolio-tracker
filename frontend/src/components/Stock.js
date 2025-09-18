import React, {useEffect, useMemo, useState} from 'react';
import {useParams} from 'react-router-dom';
import {portfolioAPI} from '../services/api';
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

const Stock = () => {
  const { symbol } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showFullSummary, setShowFullSummary] = useState(false);
  const [priceDays, setPriceDays] = useState(() => Number(localStorage.getItem('stock_price_days') || 365));
  const [priceMetric, setPriceMetric] = useState(() => localStorage.getItem('stock_price_metric') || 'price'); // 'price' | 'price_pct_change'
  const [priceData, setPriceData] = useState([]);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const res = await portfolioAPI.getInstrument(symbol);
        setData(res);
      } catch (e) {
        console.error(e);
        setError('Failed to load instrument');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [symbol]);

  // Load price chart data for this symbol
  useEffect(() => {
    const loadPrices = async () => {
      try {
        if (!symbol) return;
        const res = await portfolioAPI.getChartPrices(symbol, priceDays);
        const apiData = res.data || {};
        const series = apiData[symbol] || [];
        if (priceMetric === 'price_pct_change') {
          const sorted = [...series].sort((a,b)=> new Date(a.date)-new Date(b.date));
          if (sorted.length === 0) { setPriceData([]); return; }
          const base = sorted[0].value;
          setPriceData(sorted.map(p=>({ date: p.date, value: base>0 ? ((p.value-base)/base)*100 : 0 })));
        } else {
          setPriceData(series.map(p=>({ date: p.date, value: p.value })));
        }
      } catch (e) {
        console.error('Failed to load price data', e);
        setPriceData([]);
      }
    };
    loadPrices();
  }, [symbol, priceDays, priceMetric]);

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

  if (loading) return <div className="stock-container">Loading...</div>;
  if (error) return <div className="stock-container error">{error}</div>;
  if (!data) return null;

  const i = data.instrument;
  const f = data.fundamentals || {};
  const fmtShort = (n) => {
    if (n === null || n === undefined || isNaN(n)) return '-';
    const abs = Math.abs(n);
    if (abs >= 1e12) return (n / 1e12).toFixed(2) + 'T';
    if (abs >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (abs >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (abs >= 1e3) return (n / 1e3).toFixed(2) + 'K';
    return String(n);
  };
  const fmtRatio = (n) => (n === null || n === undefined || isNaN(n) ? '-' : Number(n).toFixed(2));
  const kpis = [
    { label: 'Market Cap', value: fmtShort(f.marketCap) },
    { label: 'PE', value: f.peRatio ?? '-' },
    { label: 'PEG', value: f.pegRatio ?? '-' },
    { label: 'Dividend', value: f.dividendYield ? `${(f.dividendYield*100).toFixed(2)}%` : '-' },
    { label: 'Beta', value: f.beta ?? '-' },
    { label: 'Debt', value: fmtShort(f.totalDebt) },
    { label: 'Cash', value: fmtShort(f.totalCash) },
    { label: 'FCF Yield', value: (f.freeCashflow && f.marketCap && f.marketCap>0) ? `${((f.freeCashflow/f.marketCap)*100).toFixed(2)}%` : '-' },
  ];

  const valuation = [
    { label: 'EV', value: fmtShort(f.enterpriseValue) },
    { label: 'EV/EBITDA', value: fmtRatio(f.enterpriseToEbitda) },
    { label: 'EV/Sales', value: fmtRatio(f.enterpriseToRevenue) },
    { label: 'P/S (TTM)', value: fmtRatio(f.priceToSalesTtm) },
    { label: 'P/B', value: fmtRatio(f.priceToBook) },
    { label: 'Net Debt', value: (f.totalDebt !== undefined && f.totalCash !== undefined) ? fmtShort((f.totalDebt || 0) - (f.totalCash || 0)) : '-' },
    { label: 'Net Debt/EBITDA', value: (f.ebitda && f.ebitda !== 0 && (f.totalDebt !== undefined && f.totalCash !== undefined)) ? fmtRatio((((f.totalDebt || 0) - (f.totalCash || 0)) / f.ebitda)) : '-' },
    { label: 'Price/FCF', value: (f.marketCap && f.freeCashflow) ? fmtRatio(f.marketCap / f.freeCashflow) : '-' },
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
          <button className="summary-toggle" onClick={()=>setShowFullSummary(s=>!s)}>{showFullSummary ? 'Show less' : 'Show more'}</button>
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
            <select aria-label="Price metric" value={priceMetric} onChange={(e)=>{ const v=e.target.value; setPriceMetric(v); localStorage.setItem('stock_price_metric', v);} }>
              <option value="price">Price</option>
              <option value="price_pct_change">Price % Change</option>
            </select>
            <label style={{marginLeft:8}}>Range: </label>
            <select aria-label="Price range" value={priceDays} onChange={(e)=>{ const v=Number(e.target.value); setPriceDays(v); localStorage.setItem('stock_price_days', String(v)); }}>
              <option value={30}>30d</option>
              <option value={90}>90d</option>
              <option value={180}>6m</option>
              <option value={365}>1y</option>
              <option value={1827}>5y</option>
              <option value={3652}>10y</option>
            </select>
          </div>
          {priceData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={priceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis tickFormatter={(v)=> priceMetric==='price_pct_change' ? `${v.toFixed?.(1)}%` : (typeof v==='number' ? v.toLocaleString(undefined,{maximumFractionDigits:2}) : v)} />
                <Tooltip content={<SharedTooltip valueFormatter={(v)=> priceMetric==='price_pct_change' ? `${v.toFixed?.(2)}%` : (typeof v==='number' ? v.toLocaleString(undefined,{maximumFractionDigits:2}) : v)} nameMap={{value: priceMetric==='price_pct_change' ? 'Price %' : 'Price'}} />} />
                <Legend />
                <Line type="monotone" dataKey="value" name={priceMetric==='price_pct_change' ? 'Price %' : 'Price'} stroke="#6f42c1" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : <div className="empty">No price data</div>}
        </div>

        <div className="panel">
          <h3>EPS: Estimate vs Reported (with Surprise%)</h3>
          {epsSeries.length > 0 ? (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={epsSeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" tickFormatter={(v)=>`${v?.toFixed?.(0)}%`} />
                <Tooltip content={<SharedTooltip valueFormatter={(v)=> (typeof v==='number'? v.toFixed(2) : v)} nameMap={{eps_est:'EPS Estimate', eps_rep:'Reported EPS', surprise:'Surprise %'}} />} />
                <Legend />
                <Bar yAxisId="left" dataKey="eps_est" name="EPS Estimate" fill="#adb5bd" />
                <Bar yAxisId="left" dataKey="eps_rep" name="Reported EPS" fill="#2a9d8f" />
                <Line yAxisId="right" type="monotone" dataKey="surprise" name="Surprise %" stroke="#f4a261" strokeWidth={3} dot={{ r: 3 }} />
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
                <YAxis tickFormatter={(v)=> (typeof v==='number'? (v/1e9).toFixed(1)+'B' : v)} />
                <Tooltip content={<SharedTooltip valueFormatter={(v)=> typeof v==='number'? (v/1e9).toFixed(2)+'B' : v} />} />
                <Legend />
                <ReferenceLine y={0} stroke="#ccc" />
                <Bar dataKey="ocf" name="Operating CF" fill="#0d6efd" />
                <Bar dataKey="capex" name="CapEx" fill="#ff7f0e" />
                <Line type="monotone" dataKey="fcf" name="Free CF" stroke="#28a745" strokeWidth={3} dot={false} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="empty">No cash flow data</div>}
        </div>
      </div>
    </div>
  );
};

export default Stock;


