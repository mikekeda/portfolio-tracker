import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import apiClient from '../services/api';
import './Form13F.css';

// ─── Formatting helpers ───────────────────────────────────────────────────────

function formatAUM(value) {
  if (value == null) return '—';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(0)}M`;
  return `$${value.toLocaleString()}`;
}

function formatValueChange(change) {
  if (change == null) return null;
  const abs = Math.abs(change);
  const sign = change >= 0 ? '+' : '-';
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(0)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(0)}K`;
  return `${sign}$${abs}`;
}

function formatQuarter(isoDate) {
  if (!isoDate) return '—';
  const d = new Date(isoDate + 'T00:00:00');
  const month = d.getMonth() + 1;
  const year = d.getFullYear();
  const q = Math.ceil(month / 3);
  return `Q${q} ${year}`;
}

function formatShares(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString();
}

function getChangeClass(change) {
  if (!change || change === '—') return 'change-stable';
  if (change === 'New') return 'change-new';
  if (change === 'Closed') return 'change-closed';
  const match = change.match(/^([+-]?\d+(?:\.\d+)?)%$/);
  if (match) {
    const pct = parseFloat(match[1]);
    if (pct >= 10) return 'change-increase';
    if (pct <= -30) return 'change-trimmed';
    return 'change-stable';
  }
  return 'change-stable';
}

// ─── Highlights section ───────────────────────────────────────────────────────

const HighlightItem = ({ item, mode }) => {
  const value = mode === 'sell'
    ? item.total_value_removed
    : mode === 'held'
      ? item.total_value
      : item.total_value_added;

  const buyMgrs = (item.buy_managers || []).map(m => typeof m === 'string' ? m : m.name);
  const sellMgrs = (item.sell_managers || []).map(m => typeof m === 'string' ? m : m.name);
  const heldMgrs = item.managers || [];

  let tooltip = '';
  if (mode === 'held') {
    tooltip = heldMgrs.join(', ');
  } else if (mode === 'mixed') {
    tooltip = [
      buyMgrs.length ? `Buying: ${buyMgrs.join(', ')}` : '',
      sellMgrs.length ? `Selling: ${sellMgrs.join(', ')}` : '',
    ].filter(Boolean).join('\n');
  } else {
    const primary = mode === 'buy' ? buyMgrs : sellMgrs;
    const other   = mode === 'buy' ? sellMgrs : buyMgrs;
    const otherLabel = mode === 'buy' ? 'Also selling' : 'Also buying';
    tooltip = [
      primary.length ? primary.join(', ') : '',
      other.length   ? `${otherLabel}: ${other.join(', ')}` : '',
    ].filter(Boolean).join('\n');
  }

  // Held: derive trend from buy_count / sell_count if available
  const heldTrend = mode === 'held' && item.buy_count != null
    ? item.buy_count > item.sell_count ? 'accumulating'
    : item.sell_count > item.buy_count ? 'distributing'
    : 'stable'
    : null;

  return (
    <div className="hi-item" title={tooltip}>
      <div className="hi-item-left">
        {item.yahoo_symbol ? (
          <Link className="hi-ticker" to={`/stock/${encodeURIComponent(item.yahoo_symbol)}`}>
            {item.yahoo_symbol}
          </Link>
        ) : (
          <span className="hi-ticker hi-ticker-none">—</span>
        )}
        <span className="hi-name" title={item.name || item.issuer}>{item.name || item.issuer}</span>
      </div>
      <div className="hi-item-right">
        {mode === 'held' ? (
          <>
            <span className="hi-count-badge hi-badge-held">{item.count} mgrs</span>
            {heldTrend && heldTrend !== 'stable' && (
              <span className={`hi-trend hi-trend-${heldTrend}`}>
                {heldTrend === 'accumulating' ? '▲' : '▼'}
              </span>
            )}
          </>
        ) : mode === 'mixed' ? (
          <span className="hi-count-badge hi-badge-mixed">
            {item.buy_count}↑ {item.sell_count}↓
          </span>
        ) : (
          <span className={`hi-count-badge hi-badge-${mode}`}>
            {mode === 'buy' ? `+${item.net_managers}` : item.net_managers}
            <span className="hi-badge-detail">
              {item.buy_count}↑{item.sell_count > 0 ? ` ${item.sell_count}↓` : ''}
            </span>
          </span>
        )}
        {value > 0 && <span className="hi-value">{formatAUM(value)}</span>}
      </div>
    </div>
  );
};

const HighlightsSection = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    apiClient.get('/api/13f/highlights')
      .then(res => setData(res.data))
      .catch(() => {}); // non-critical — fail silently
  }, []);

  if (!data) return null;

  const buys     = data.most_bought || [];
  const sells    = data.most_sold   || [];
  const held     = data.most_held   || [];
  const mixed    = data.disputed    || [];
  const hasMixed = mixed.length > 0;

  return (
    <div className="highlights-section">
      <div className="highlights-header">
        <h3 className="highlights-title">Consensus Signals</h3>
        <span className="highlights-subtitle">
          Net conviction across all tracked managers · ranked by net buyers − sellers
        </span>
      </div>
      <div className={`highlights-grid ${hasMixed ? 'highlights-grid-4' : ''}`}>

        <div className="highlight-panel hl-buys">
          <div className="hi-panel-title">
            Consensus Buys
            <span className="hi-panel-count">{buys.length}</span>
          </div>
          {buys.length === 0 ? <div className="hi-empty">No data</div>
            : buys.map(item => <HighlightItem key={item.cusip} item={item} mode="buy" />)}
        </div>

        <div className="highlight-panel hl-sells">
          <div className="hi-panel-title">
            Consensus Sells
            <span className="hi-panel-count">{sells.length}</span>
          </div>
          {sells.length === 0 ? <div className="hi-empty">No data</div>
            : sells.map(item => <HighlightItem key={item.cusip} item={item} mode="sell" />)}
        </div>

        <div className="highlight-panel hl-held">
          <div className="hi-panel-title">
            Most Widely Held
            <span className="hi-panel-count">{held.length}</span>
          </div>
          {held.length === 0 ? <div className="hi-empty">No data</div>
            : held.map(item => <HighlightItem key={item.cusip} item={item} mode="held" />)}
        </div>

        {hasMixed && (
          <div className="highlight-panel hl-disputed">
            <div className="hi-panel-title">
              Mixed Signals
              <span className="hi-panel-count">{mixed.length}</span>
            </div>
            <div className="hi-panel-subtitle">Smart money divided equally</div>
            {mixed.map(item => <HighlightItem key={item.cusip} item={item} mode="mixed" />)}
          </div>
        )}

      </div>
    </div>
  );
};

// ─── Overview page ────────────────────────────────────────────────────────────

const ManagerCard = ({ manager }) => {
  const navigate = useNavigate();
  const act = manager.activity;

  return (
    <div className="manager-card" onClick={() => navigate(`/13f/${manager.id}`)}>
      <div className="manager-card-header">
        <h3 className="manager-card-name">{manager.name}</h3>
        <span className="manager-card-quarter">{formatQuarter(manager.latest_report_date)}</span>
      </div>

      <div className="manager-card-stats">
        <div className="manager-stat">
          <span className="manager-stat-value">{formatAUM(manager.total_value)}</span>
          <span className="manager-stat-label">AUM</span>
        </div>
        <div className="manager-stat">
          <span className="manager-stat-value">{manager.num_positions}</span>
          <span className="manager-stat-label">Positions</span>
        </div>
        <div className="manager-stat">
          <span className="manager-stat-value">{manager.filing_count}</span>
          <span className="manager-stat-label">Quarters</span>
        </div>
      </div>

      {act ? (
        <div className="manager-card-activity">
          {act.new > 0 && (
            <span className="activity-pill activity-new">+{act.new} New</span>
          )}
          {act.increased > 0 && (
            <span className="activity-pill activity-increase">↑{act.increased} Inc</span>
          )}
          {act.trimmed > 0 && (
            <span className="activity-pill activity-trimmed">↓{act.trimmed} Trim</span>
          )}
          {act.closed > 0 && (
            <span className="activity-pill activity-closed">✕{act.closed} Closed</span>
          )}
          {act.stable > 0 && (
            <span className="activity-pill activity-stable">={act.stable} Stable</span>
          )}
        </div>
      ) : (
        <div className="manager-card-activity">
          <span className="activity-pill activity-stable">First filing</span>
        </div>
      )}

      <div className="manager-card-footer">
        <span className="manager-card-link">View Portfolio →</span>
      </div>
    </div>
  );
};

const MANAGER_SORTS = [
  { key: 'name', label: 'Name' },
  { key: 'aum', label: 'AUM' },
  { key: 'activity', label: 'Most Active' },
  { key: 'positions', label: 'Positions' },
];

function sortManagers(managers, sortBy) {
  return [...managers].sort((a, b) => {
    if (sortBy === 'name') return a.name.localeCompare(b.name);
    if (sortBy === 'aum') return (b.total_value || 0) - (a.total_value || 0);
    if (sortBy === 'positions') return (b.num_positions || 0) - (a.num_positions || 0);
    if (sortBy === 'activity') {
      const act = m => m.activity
        ? (m.activity.new + m.activity.closed + m.activity.increased + m.activity.trimmed)
        : 0;
      return act(b) - act(a);
    }
    return 0;
  });
}

const Form13FOverview = () => {
  const [managers, setManagers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState('aum');

  useEffect(() => {
    const fetch = async () => {
      try {
        setLoading(true);
        const res = await apiClient.get('/api/13f/managers');
        setManagers(res.data);
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to load managers');
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  const sorted = useMemo(() => sortManagers(managers, sortBy), [managers, sortBy]);

  if (loading) return <div className="f13f-loading">Loading institutional holders…</div>;
  if (error) return <div className="f13f-error">{error}</div>;

  return (
    <div className="f13f-container">
      <div className="f13f-page-header">
        <h2>13F Institutional Holdings</h2>
        <p className="f13f-page-subtitle">
          {managers.length} tracked managers · SEC 13F filings are quarterly (45-day lag)
        </p>
      </div>

      <HighlightsSection />

      {managers.length === 0 ? (
        <div className="f13f-empty">
          No 13F data yet. Run <code>python -m scripts.scrape_13f</code> to populate.
        </div>
      ) : (
        <>
          <div className="manager-sort-bar">
            <span className="sort-label">Sort by:</span>
            {MANAGER_SORTS.map(s => (
              <button
                key={s.key}
                className={`sort-btn ${sortBy === s.key ? 'active' : ''}`}
                onClick={() => setSortBy(s.key)}
              >
                {s.label}
              </button>
            ))}
          </div>
          <div className="manager-grid">
            {sorted.map(m => <ManagerCard key={m.id} manager={m} />)}
          </div>
        </>
      )}
    </div>
  );
};

// ─── Moves section ────────────────────────────────────────────────────────────

const PositionCard = ({ position, showPrev = false }) => {
  const isLinked = !!position.yahoo_symbol;
  const nameEl = isLinked ? (
    <Link className="pos-ticker-link" to={`/stock/${encodeURIComponent(position.yahoo_symbol)}`}>
      {position.yahoo_symbol}
    </Link>
  ) : (
    <span className="pos-ticker-nolink">{position.yahoo_symbol || '—'}</span>
  );

  return (
    <div className={`position-card ${getChangeClass(position.change)}`}>
      <div className="position-card-top">
        {nameEl}
        <span className={`pos-change-badge ${getChangeClass(position.change)}`}>
          {position.change}
        </span>
      </div>
      <div className="position-card-name" title={position.issuer}>{position.name || position.issuer}</div>
      <div className="position-card-bottom">
        <span className="pos-value">{formatAUM(position.value || position.value_prev)}</span>
        {position.pct_of_portfolio != null && (
          <span className="pos-pct">{position.pct_of_portfolio}% of fund</span>
        )}
        {position.value_change != null && (
          <span className={`pos-value-change ${position.value_change >= 0 ? 'positive' : 'negative'}`}>
            {formatValueChange(position.value_change)}
          </span>
        )}
      </div>
    </div>
  );
};

const MovesSection = ({ moves }) => {
  const hasAny =
    moves.new_positions.length + moves.closed_positions.length +
    moves.top_buys.length + moves.top_sells.length > 0;

  if (!hasAny) {
    return (
      <div className="f13f-empty">
        No prior quarter available — moves require at least 2 quarters of data.
      </div>
    );
  }

  return (
    <div className="moves-grid">
      {moves.new_positions.length > 0 && (
        <div className="moves-section">
          <h4 className="moves-section-title moves-title-new">
            New Positions <span className="moves-count">{moves.new_positions.length}</span>
          </h4>
          <div className="moves-cards">
            {moves.new_positions.map(p => <PositionCard key={p.cusip} position={p} />)}
          </div>
        </div>
      )}

      {moves.top_buys.length > 0 && (
        <div className="moves-section">
          <h4 className="moves-section-title moves-title-buys">
            Top Buys
            <span className="moves-count">{moves.top_buys.length}</span>
            {moves.increased_count > moves.top_buys.length && (
              <span className="moves-count-total"> of {moves.increased_count}</span>
            )}
          </h4>
          <div className="moves-cards">
            {moves.top_buys.map(p => <PositionCard key={p.cusip} position={p} />)}
          </div>
        </div>
      )}

      {moves.top_sells.length > 0 && (
        <div className="moves-section">
          <h4 className="moves-section-title moves-title-sells">
            Top Sells
            <span className="moves-count">{moves.top_sells.length}</span>
            {moves.trimmed_count > moves.top_sells.length && (
              <span className="moves-count-total"> of {moves.trimmed_count}</span>
            )}
          </h4>
          <div className="moves-cards">
            {moves.top_sells.map(p => <PositionCard key={p.cusip} position={p} />)}
          </div>
        </div>
      )}

      {moves.closed_positions.length > 0 && (
        <div className="moves-section">
          <h4 className="moves-section-title moves-title-closed">
            Closed Positions <span className="moves-count">{moves.closed_positions.length}</span>
          </h4>
          <div className="moves-cards">
            {moves.closed_positions.map(p => <PositionCard key={p.cusip} position={p} />)}
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Portfolio table ──────────────────────────────────────────────────────────

const PortfolioTable = ({ portfolio }) => {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState('rank');
  const [sortDir, setSortDir] = useState('asc');

  const handleSort = useCallback((key) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir(key === 'rank' ? 'asc' : 'desc');
    }
  }, [sortKey]);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    let rows = q
      ? portfolio.filter(p =>
          (p.yahoo_symbol || '').toLowerCase().includes(q) ||
          (p.name || '').toLowerCase().includes(q) ||
          (p.issuer || '').toLowerCase().includes(q)
        )
      : portfolio;

    return [...rows].sort((a, b) => {
      let va = a[sortKey], vb = b[sortKey];
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') {
        return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return sortDir === 'asc' ? va - vb : vb - va;
    });
  }, [portfolio, search, sortKey, sortDir]);

  const SortHeader = ({ label, field }) => (
    <th className="sortable-th" onClick={() => handleSort(field)}>
      {label}
      <span className="sort-arrow">
        {sortKey === field ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ' ↕'}
      </span>
    </th>
  );


  return (
    <div className="portfolio-table-wrapper">
      <div className="portfolio-search-bar">
        <input
          type="text"
          placeholder="Filter by ticker, name…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="portfolio-search-input"
        />
        <span className="portfolio-count">{filtered.length} of {portfolio.length} positions</span>
      </div>

      <div className="portfolio-table-scroll">
        <table className="portfolio-table">
          <thead>
            <tr>
              <SortHeader label="#" field="rank" />
              <SortHeader label="Ticker" field="yahoo_symbol" />
              <SortHeader label="Company" field="name" />
              <SortHeader label="Value ($)" field="value" />
              <SortHeader label="% Fund" field="pct_of_portfolio" />
              <SortHeader label="Shares" field="shares" />
              <SortHeader label="QoQ Change" field="change_sort" />
              <SortHeader label="Value Δ" field="value_change" />
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.cusip}>
                <td className="pt-rank">{p.rank}</td>
                <td className="pt-ticker">
                  {p.yahoo_symbol ? (
                    <Link to={`/stock/${encodeURIComponent(p.yahoo_symbol)}`} className="pt-ticker-link">
                      {p.yahoo_symbol}
                    </Link>
                  ) : <span className="pt-ticker-none">—</span>}
                </td>
                <td className="pt-name" title={p.issuer}>
                  {p.yahoo_symbol ? (
                    <Link to={`/stock/${encodeURIComponent(p.yahoo_symbol)}`} className="pt-name-link">
                      {p.name || p.issuer}
                    </Link>
                  ) : (
                    <span>{p.name || p.issuer}</span>
                  )}
                </td>
                <td className="pt-value">{formatAUM(p.value)}</td>
                <td className="pt-pct">{p.pct_of_portfolio}%</td>
                <td className="pt-shares">{formatShares(p.shares)}</td>
                <td className="pt-change">
                  <span className={`pt-change-badge ${getChangeClass(p.change)}`}>
                    {p.change}
                  </span>
                </td>
                <td className={`pt-delta ${p.value_change > 0 ? 'positive' : p.value_change < 0 ? 'negative' : ''}`}>
                  {p.value_change != null ? formatValueChange(p.value_change) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ─── Detail page ──────────────────────────────────────────────────────────────

const Form13FDetail = () => {
  const { managerId } = useParams();
  const [data, setData] = useState(null);
  const [selectedQuarter, setSelectedQuarter] = useState(null);
  const [activeTab, setActiveTab] = useState('moves');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async (quarter) => {
    try {
      setLoading(true);
      setError(null);
      const params = quarter ? { report_date: quarter } : {};
      const res = await apiClient.get(`/api/13f/managers/${managerId}`, { params });
      setData(res.data);
      setSelectedQuarter(res.data.report_date);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load manager data');
    } finally {
      setLoading(false);
    }
  }, [managerId]);

  useEffect(() => { fetchData(null); }, [fetchData]);

  const handleQuarterChange = (e) => {
    const q = e.target.value;
    setSelectedQuarter(q);
    fetchData(q);
  };

  if (loading) return <div className="f13f-loading">Loading portfolio…</div>;
  if (error) return (
    <div className="f13f-container">
      <Link to="/13f" className="f13f-back">← Back to all managers</Link>
      <div className="f13f-error">{error}</div>
    </div>
  );
  if (!data) return null;

  const { manager, available_quarters, report_date, prev_report_date, total_value, num_positions, portfolio, moves } = data;

  const prevTotal = prev_report_date ? data.portfolio.reduce((s, p) => s + (p.value_prev || 0), 0) : null;
  const aumChange = prevTotal != null ? total_value - prevTotal : null;

  return (
    <div className="f13f-container">
      <Link to="/13f" className="f13f-back">← All managers</Link>

      <div className="f13f-detail-header">
        <div className="f13f-manager-title">
          <h2>{manager.name}</h2>
          <div className="f13f-manager-meta">
            <span className="f13f-cik">CIK: {manager.cik}</span>
            <a href={manager.sec_url} target="_blank" rel="noreferrer" className="f13f-sec-link">
              SEC EDGAR ↗
            </a>
          </div>
        </div>

        <div className="f13f-header-stats">
          <div className="f13f-header-stat">
            <span className="f13f-header-stat-value">{formatAUM(total_value)}</span>
            <span className="f13f-header-stat-label">Portfolio AUM</span>
            {aumChange != null && (
              <span className={`f13f-header-stat-delta ${aumChange >= 0 ? 'positive' : 'negative'}`}>
                {formatValueChange(aumChange)} vs prev Q
              </span>
            )}
          </div>
          <div className="f13f-header-stat-divider" />
          <div className="f13f-header-stat">
            <span className="f13f-header-stat-value">{num_positions}</span>
            <span className="f13f-header-stat-label">Positions</span>
          </div>
        </div>

        <div className="f13f-quarter-selector">
          <label>Quarter:</label>
          <select value={selectedQuarter || ''} onChange={handleQuarterChange}>
            {available_quarters.map(q => (
              <option key={q} value={q}>{formatQuarter(q)}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Tabs */}
      <div className="f13f-tabs">
        <button
          className={`f13f-tab ${activeTab === 'moves' ? 'active' : ''}`}
          onClick={() => setActiveTab('moves')}
        >
          Latest Moves
        </button>
        <button
          className={`f13f-tab ${activeTab === 'portfolio' ? 'active' : ''}`}
          onClick={() => setActiveTab('portfolio')}
        >
          Full Portfolio ({num_positions})
        </button>
      </div>

      <div className="f13f-tab-content">
        {activeTab === 'moves' && <MovesSection moves={moves} />}
        {activeTab === 'portfolio' && <PortfolioTable portfolio={portfolio} />}
      </div>
    </div>
  );
};

// ─── Route dispatcher ─────────────────────────────────────────────────────────

const Form13F = () => {
  const { managerId } = useParams();
  return managerId ? <Form13FDetail /> : <Form13FOverview />;
};

export default Form13F;
