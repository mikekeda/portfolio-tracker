import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
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
  if (n == null) return '—';
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

// ── Tooltip helpers ──────────────────────────────────────────────────────────

// Parse a manager's change string ("+87.5%", "-45%", "New") into a sortable number.
// "New" is +Infinity — highest conviction action.
function parseMgrChange(m) {
  const c = m?.change;
  if (!c || c === '—') return 0;
  if (c === 'New') return Infinity;
  const n = parseFloat(c.replace('+', '').replace('%', ''));
  return isNaN(n) ? 0 : n;
}

// Primary sort key: dollar value_change (capital deployed/extracted).
// Falls back to % change when value unavailable.
function mgrSortKey(m, direction) {
  if (m.value_change != null) return m.value_change;
  const pct = parseMgrChange(m);
  return direction === 'buy' ? pct : -pct;
}

function buildMgrSection(label, mgrs, direction) {
  if (!mgrs.length) return null;
  const sorted = [...mgrs].sort((a, b) =>
    direction === 'buy'
      ? mgrSortKey(b, 'buy')  - mgrSortKey(a, 'buy')
      : mgrSortKey(a, 'sell') - mgrSortKey(b, 'sell')
  );
  const isGreen   = direction === 'buy';
  const hasValChg = sorted.some(m => m.value_change != null && m.value_change !== 0);
  const total     = hasValChg ? sorted.reduce((s, m) => s + (m.value_change || 0), 0) : 0;
  return (
    <div className="hi-tip-section">
      <div className={`hi-tip-section-label ${isGreen ? 'buying' : 'selling'}`}>{label}</div>
      <table className="hi-tip-table">
        <thead>
          <tr>
            <th>Manager</th>
            <th>Change</th>
            {hasValChg && <th>Value Δ</th>}
            {hasValChg && <th>% Fund</th>}
          </tr>
        </thead>
        <tbody>
          {sorted.map((m, i) => {
            const pct    = parseMgrChange(m);
            const chgCls = !m.change || m.change === '—' ? ''
              : pct > 0 || m.change === 'New' ? 'positive' : 'negative';
            const valCls = !m.value_change ? '' : m.value_change > 0 ? 'positive' : 'negative';
            return (
              <tr key={i}>
                <td className="hi-tip-name">{m.name}</td>
                <td className={`hi-tip-change ${chgCls}`}>{m.change || '—'}</td>
                {hasValChg && (
                  <>
                    <td className={`hi-tip-change ${valCls}`}>
                      {m.value_change != null && m.value_change !== 0 ? formatValueChange(m.value_change) : '—'}
                    </td>
                    <td className={`hi-tip-change ${valCls}`}>
                      {m.pct_impact ? `${m.pct_impact > 0 ? '+' : ''}${m.pct_impact.toFixed(2)}%` : '—'}
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
        {hasValChg && total !== 0 && sorted.length > 1 && (
          <tfoot>
            <tr className="hi-tip-total-row">
              <td colSpan={2}>Total</td>
              <td className={`hi-tip-change ${total > 0 ? 'positive' : 'negative'}`}>
                {formatValueChange(total)}
              </td>
              <td></td>
            </tr>
          </tfoot>
        )}
      </table>
    </div>
  );
}

// Holding section: managers who didn't change position.
// Shows current position value (not delta) sorted biggest first, with a total.
function buildHoldingSection(mgrs) {
  if (!mgrs.length) return null;
  const sorted = [...mgrs].sort((a, b) => {
    if (a.current_value !== b.current_value) return (b.current_value || 0) - (a.current_value || 0);
    return a.name.localeCompare(b.name);
  });
  const hasValue = sorted.some(m => m.current_value);
  const total    = hasValue ? sorted.reduce((s, m) => s + (m.current_value || 0), 0) : 0;
  return (
    <div className="hi-tip-section">
      <div className="hi-tip-section-label holding">= Holding ({sorted.length})</div>
      <table className="hi-tip-table">
        <thead>
          <tr>
            <th>Manager</th>
            {hasValue && <th>Position</th>}
          </tr>
        </thead>
        <tbody>
          {sorted.map((m, i) => (
            <tr key={i}>
              <td className="hi-tip-name">{m.name}</td>
              {hasValue && <td className="hi-tip-change">{m.current_value ? formatAUM(m.current_value) : '—'}</td>}
            </tr>
          ))}
        </tbody>
        {hasValue && total > 0 && sorted.length > 1 && (
          <tfoot>
            <tr className="hi-tip-total-row">
              <td>Total</td>
              <td className="hi-tip-change">{formatAUM(total)}</td>
            </tr>
          </tfoot>
        )}
      </table>
    </div>
  );
}

// Build rich JSX tooltip for a highlight panel item.
function buildHighlightContent(item, mode) {
  const buyMgrs  = item.buy_managers  || [];
  const sellMgrs = item.sell_managers || [];
  // Stable = all holders that are neither buying nor selling this quarter
  const buyNames  = new Set(buyMgrs.map(m => m.name));
  const sellNames = new Set(sellMgrs.map(m => m.name));
  const heldMgrs  = (item.managers || []).filter(m => !buyNames.has(m.name) && !sellNames.has(m.name));

  let sections;
  if (mode === 'held') {
    sections = [
      buyMgrs.length  ? buildMgrSection(`▲ Buying (${buyMgrs.length})`,   buyMgrs,  'buy')  : null,
      heldMgrs.length ? buildHoldingSection(heldMgrs)                                        : null,
      sellMgrs.length ? buildMgrSection(`▼ Trimming (${sellMgrs.length})`, sellMgrs, 'sell') : null,
    ];
  } else if (mode === 'mixed') {
    sections = [
      buildMgrSection(`▲ Buying (${buyMgrs.length})`,   buyMgrs,  'buy'),
      heldMgrs.length ? buildHoldingSection(heldMgrs)              : null,
      buildMgrSection(`▼ Selling (${sellMgrs.length})`, sellMgrs, 'sell'),
    ];
  } else if (mode === 'buy') {
    sections = [
      buildMgrSection(`▲ Buying (${buyMgrs.length})`, buyMgrs, 'buy'),
      sellMgrs.length ? buildMgrSection(`▼ Also selling (${sellMgrs.length})`, sellMgrs, 'sell') : null,
    ];
  } else {
    sections = [
      buildMgrSection(`▼ Selling (${sellMgrs.length})`, sellMgrs, 'sell'),
      buyMgrs.length ? buildMgrSection(`▲ Also buying (${buyMgrs.length})`, buyMgrs, 'buy') : null,
    ];
  }

  const filtered = sections.filter(Boolean);
  if (!filtered.length) return null;
  return <div className="hi-tip-wrap">{filtered}</div>;
}

// Tooltip rendered via portal into document.body — bypasses CSS transform
// containing-blocks on ancestor elements (e.g. the card hover-lift).
// Always floats below the trigger element (rich table popover).
const HiTooltip = ({ visible, content, x, y }) => {
  if (!visible || !content) return null;
  return createPortal(
    <div className="hi-tooltip-popup" style={{ top: y, left: x }}>
      {content}
    </div>,
    document.body
  );
};

function useHiTooltip() {
  const [tip, setTip] = useState({ visible: false, content: null, x: 0, y: 0 });
  // showRich: JSX content, appears below the trigger element
  const showRich = useCallback((e, content) => {
    if (!content) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const tooltipMaxH = 340;
    const below = rect.bottom + 6 + tooltipMaxH;
    const y = below > window.innerHeight
      ? Math.max(4, rect.top - tooltipMaxH - 6)
      : rect.bottom + 6;
    setTip({
      visible: true, content,
      x: Math.min(rect.left, window.innerWidth - 300),
      y,
    });
  }, []);
  const hide = useCallback(() => setTip(t => ({ ...t, visible: false })), []);
  return { tip, showRich, hide };
}

// ── Highlight item ────────────────────────────────────────────────────────────

const HighlightItem = React.memo(function HighlightItem({ item, mode, onTipRich, onTipHide }) {
  const netValue = (item.total_value_added || 0) - (item.total_value_removed || 0);
  const value = mode === 'held' ? item.total_value : null;

  const heldTrend = mode === 'held' && item.buy_count != null
    ? item.buy_count > item.sell_count ? 'accumulating'
    : item.sell_count > item.buy_count ? 'distributing'
    : 'stable'
    : null;

  // Tooltip only on the badge — avoids triggering on ticker/name hover.
  const content = buildHighlightContent(item, mode);
  const badgeTip = onTipRich && content
    ? { onMouseEnter: e => onTipRich(e, content), onMouseLeave: onTipHide }
    : {};

  return (
    <div className="hi-item">
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
            <span className="hi-count-badge hi-badge-held" {...badgeTip}>{item.count} mgrs</span>
            {heldTrend && heldTrend !== 'stable' && (
              <span className={`hi-trend hi-trend-${heldTrend}`}>
                {heldTrend === 'accumulating' ? '▲' : '▼'}
              </span>
            )}
          </>
        ) : mode === 'mixed' ? (
          <span className="hi-count-badge hi-badge-mixed" {...badgeTip}>
            {item.buy_count}↑ {item.sell_count}↓
          </span>
        ) : (
          <span className={`hi-count-badge hi-badge-${mode}`} {...badgeTip}>
            {mode === 'buy' ? `+${item.net_managers}` : item.net_managers}
            <span className="hi-badge-detail">
              {item.buy_count}↑{item.sell_count > 0 ? ` ${item.sell_count}↓` : ''}
            </span>
          </span>
        )}
        {(mode === 'buy' || mode === 'sell' || mode === 'mixed') ? (
          netValue !== 0 && <span className={`hi-value ${netValue > 0 ? 'positive' : 'negative'}`}>{formatValueChange(netValue)}</span>
        ) : (
          value > 0 && <span className="hi-value">{formatAUM(value)}</span>
        )}
      </div>
    </div>
  );
});

// ── Isolated panel component — React.memo ensures it only re-renders when its
// own items change (not when sibling panels' tooltips update).

const PANEL_SORTS = [
  { key: 'conviction', label: 'Conviction' },
  { key: 'capital',    label: 'Capital'    },
];

const HighlightPanel = React.memo(function HighlightPanel({ className, title, count, subtitle, items, mode }) {
  const [panelSort, setPanelSort] = useState('conviction');
  const { tip, showRich, hide } = useHiTooltip();

  const sortedItems = useMemo(() => {
    if (mode !== 'buy' && mode !== 'sell' && mode !== 'mixed') return items;

    const r = [...items];
    r.sort((a, b) => {
      if (mode === 'buy') {
        if (panelSort === 'conviction') {
          return (b.net_managers - a.net_managers) ||
                 (b.new_count - a.new_count) ||
                 ((b.total_conviction_added - b.total_conviction_removed) - (a.total_conviction_added - a.total_conviction_removed));
        } else {
          return ((b.total_value_added - b.total_value_removed) - (a.total_value_added - a.total_value_removed));
        }
      } else if (mode === 'sell') {
        if (panelSort === 'conviction') {
          return (a.net_managers - b.net_managers) ||
                 (b.closed_count - a.closed_count) ||
                 ((b.total_conviction_removed - b.total_conviction_added) - (a.total_conviction_removed - a.total_conviction_added));
        } else {
          return ((b.total_value_removed - b.total_value_added) - (a.total_value_removed - a.total_value_added));
        }
      } else if (mode === 'mixed') {
        if (panelSort === 'conviction') {
          return ((b.buy_count + b.sell_count) - (a.buy_count + a.sell_count)) ||
                 ((b.total_conviction_added + b.total_conviction_removed) - (a.total_conviction_added + a.total_conviction_removed));
        } else {
          const aNet = (a.total_value_added || 0) - (a.total_value_removed || 0);
          const bNet = (b.total_value_added || 0) - (b.total_value_removed || 0);
          return bNet - aNet;
        }
      }
      return 0;
    });
    return r.slice(0, 20);
  }, [items, panelSort, mode]);

  return (
    <div className={`highlight-panel ${className}`}>
      <div className="hi-panel-title">
        {title}
        <span className="hi-panel-count">{count}</span>
        {(mode === 'buy' || mode === 'sell' || mode === 'mixed') && (
          <div className="hi-panel-sort">
            {PANEL_SORTS.map(s => (
              <button
                key={s.key}
                className={`hi-sort-pill ${panelSort === s.key ? 'active' : ''}`}
                onClick={e => { e.stopPropagation(); setPanelSort(s.key); }}
              >
                {s.label}
              </button>
            ))}
          </div>
        )}
      </div>
      {subtitle && <div className="hi-panel-subtitle">{subtitle}</div>}
      {sortedItems.length === 0 ? (
        <div className="hi-empty">No data</div>
      ) : (
        <div className="hi-items-scroll">
          {sortedItems.map((item) => (
            <HighlightItem key={item.cusip} item={item} mode={mode} onTipRich={showRich} onTipHide={hide} />
          ))}
        </div>
      )}
      <HiTooltip {...tip} />
    </div>
  );
});

// ── Radar panel (isolated component so sort/hover don't re-render other panels)

const RADAR_SORTS = [
  { key: 'mgrs',   label: 'Mgrs'   },
  { key: 'value',  label: 'Value'  },
  { key: 'buying', label: 'Buying' },
];

function buildRadarContent(item) {
  const buyMgrs  = item.buying_managers  || [];
  const sellMgrs = item.selling_managers || [];
  const buyNames  = new Set(buyMgrs.map(m => m.name));
  const sellNames = new Set(sellMgrs.map(m => m.name));
  const stableMgrs = (item.managers || []).filter(m => !buyNames.has(m.name) && !sellNames.has(m.name));

  const sections = [
    buyMgrs.length    ? buildMgrSection(`▲ Buying (${buyMgrs.length})`,  buyMgrs,  'buy')  : null,
    stableMgrs.length ? buildHoldingSection(stableMgrs)                                     : null,
    sellMgrs.length   ? buildMgrSection(`▼ Selling (${sellMgrs.length})`, sellMgrs, 'sell') : null,
  ].filter(Boolean);

  if (!sections.length) return null;
  return <div className="hi-tip-wrap">{sections}</div>;
}

const RadarPanel = React.memo(function RadarPanel({ radar }) {
  const [radarSort, setRadarSort] = useState('mgrs');
  const { tip, showRich, hide } = useHiTooltip();

  const sorted = useMemo(() => {
    const r = [...radar];
    if (radarSort === 'value')       r.sort((a, b) => b.total_value - a.total_value);
    else if (radarSort === 'buying') r.sort((a, b) => b.buy_count - a.buy_count || b.manager_count - a.manager_count);
    else                             r.sort((a, b) => b.manager_count - a.manager_count || b.total_value - a.total_value);
    return r;
  }, [radar, radarSort]);

  return (
    <div className="highlight-panel hl-radar">
      <div className="hi-panel-title">
        On Their Radar
        <span className="hi-panel-count">{sorted.length}</span>
        <div className="hi-panel-sort">
          {RADAR_SORTS.map(s => (
            <button
              key={s.key}
              className={`hi-sort-pill ${radarSort === s.key ? 'active' : ''}`}
              onClick={e => { e.stopPropagation(); setRadarSort(s.key); }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
      <div className="hi-panel-subtitle">Not in your portfolio · held by 2+ managers · buying activity shown</div>
      <div className="hi-items-scroll">
        {sorted.map(item => (
          <div key={item.yahoo_symbol || item.name} className="hi-item">
            <div className="hi-item-left">
              {item.yahoo_symbol ? (
                <Link className="hi-ticker" to={`/stock/${encodeURIComponent(item.yahoo_symbol)}`}>
                  {item.yahoo_symbol}
                </Link>
              ) : (
                <span className="hi-ticker hi-ticker-none">—</span>
              )}
              <span className="hi-name" title={item.name}>{item.name}</span>
            </div>
            <div className="hi-item-right">
              <span className={`hi-trend ${item.buy_count > 0 ? 'hi-trend-accumulating' : 'hi-trend-stable'}`}>
                ▲{item.buy_count}
              </span>
              <span className={`hi-trend ${item.sell_count > 0 ? 'hi-trend-distributing' : 'hi-trend-stable'}`}>
                ▼{item.sell_count}
              </span>
              <span
                className="hi-count-badge hi-badge-held"
                onMouseEnter={e => showRich(e, buildRadarContent(item))}
                onMouseLeave={hide}
              >
                {item.manager_count} mgrs
              </span>
              <span className="hi-value">{formatAUM(item.total_value)}</span>
            </div>
          </div>
        ))}
      </div>
      <HiTooltip {...tip} />
    </div>
  );
});

// ── Highlights section ────────────────────────────────────────────────────────

const HighlightsSection = () => {
  const [data, setData]   = useState(null);
  const [radar, setRadar] = useState([]);

  useEffect(() => {
    // Independent fetches so a failure in one doesn't blank out the other
    apiClient.get('/api/13f/highlights')
      .then(r => setData(r.data))
      .catch(() => setData({ most_bought: [], most_sold: [], most_held: [], disputed: [] }));

    apiClient.get('/api/13f/not-in-portfolio')
      .then(r => setRadar(r.data || []))
      .catch(() => setRadar([]));
  }, []);

  if (!data) return (
    <div className="highlights-section">
      <div className="highlights-header">
        <h3 className="highlights-title">Consensus Signals</h3>
      </div>
      <div className="highlights-grid">
        {[0, 1, 2].map(i => (
          <div key={i} className="highlight-panel hi-skeleton">
            <div className="hi-skel-title" />
            {[0, 1, 2, 3, 4].map(j => <div key={j} className="hi-skel-row" />)}
          </div>
        ))}
      </div>
    </div>
  );

  const buys     = data.most_bought || [];
  const sells    = data.most_sold   || [];
  const held     = data.most_held   || [];
  const mixed    = data.disputed    || [];
  const hasMixed = mixed.length > 0;
  const hasRadar = radar.length > 0;
  // 5 cols when both Mixed + Radar; 4 cols when only one; 3 cols default.
  // highlight-panel has min-width:0 so columns shrink gracefully on narrow screens.
  let gridClass = '';
  if (hasRadar && hasMixed) gridClass = 'highlights-grid-5';
  else if (hasRadar || hasMixed) gridClass = 'highlights-grid-4';

  return (
    <div className="highlights-section">
      <div className="highlights-header">
        <h3 className="highlights-title">Consensus Signals</h3>
        <span className="highlights-subtitle">
          Net conviction across all tracked managers · ranked by net buyers − sellers
        </span>
      </div>
      <div className={`highlights-grid ${gridClass}`}>
        <HighlightPanel className="hl-buys"  title="Consensus Buys"  count={buys.length}  items={buys}  mode="buy"  />
        <HighlightPanel className="hl-sells" title="Consensus Sells" count={sells.length} items={sells} mode="sell" />
        <HighlightPanel className="hl-held"  title="Most Widely Held" count={held.length} items={held}  mode="held" />
        {hasMixed && (
          <HighlightPanel
            className="hl-disputed"
            title="Mixed Signals"
            count={mixed.length}
            subtitle="Smart money divided equally"
            items={mixed}
            mode="mixed"
          />
        )}
        {hasRadar && <RadarPanel radar={radar} />}
      </div>
    </div>
  );
};

// ─── Overview page ────────────────────────────────────────────────────────────

// Build JSX table content for an activity pill popover (mirrors Allocations style).
// totalCount is the real count from activity (may exceed the top-5 names shown).
function buildPillContent(names, title, totalCount) {
  if (!names?.length) return null;
  const hasChange = names.some(n => typeof n === 'object' && n.change);
  const hiddenCount = (totalCount || 0) - names.length;
  return (
    <div className="pill-tip-wrap">
      <div className="pill-tip-title">{title}</div>
      <table className="pill-tip-table">
        <thead>
          <tr>
            <th>Stock</th>
            <th>Portfolio</th>
            {hasChange && <th>Change</th>}
          </tr>
        </thead>
        <tbody>
          {names.map((item, i) => {
            const name   = typeof item === 'string' ? item : item.name;
            const pct    = typeof item === 'object' ? item.pct    : null;
            const change = typeof item === 'object' ? item.change : null;
            const pos    = change && !change.startsWith('-');
            return (
              <tr key={i}>
                <td className="pill-tip-name">{name}</td>
                <td className="pill-tip-pct">{pct != null ? `${pct}%` : '—'}</td>
                {hasChange && (
                  <td className={`pill-tip-change ${pos ? 'positive' : 'negative'}`}>
                    {change || '—'}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {hiddenCount > 0 && (
        <div className="pill-tip-more">…and {hiddenCount} more</div>
      )}
    </div>
  );
}

// Build pill hover props using the rich table popover (showRich positions below the pill).
function pillRichProps(names, title, totalCount, showRich, hide) {
  if (!names?.length) return {};
  const content = buildPillContent(names, title, totalCount);
  if (!content) return {};
  return {
    onMouseEnter: e => { e.stopPropagation(); showRich(e, content); },
    onMouseLeave: hide,
    onClick:      e => e.stopPropagation(),
  };
}

const ManagerCard = React.memo(function ManagerCard({ manager }) {
  const navigate = useNavigate();
  const { tip, showRich, hide } = useHiTooltip();
  const act   = manager.activity;
  const names = manager.activity_names || {};

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
            <span className="activity-pill activity-new" {...pillRichProps(names.new, 'New Positions', act.new, showRich, hide)}>
              +{act.new} New
            </span>
          )}
          {act.increased > 0 && (
            <span className="activity-pill activity-increase" {...pillRichProps(names.increased, 'Increased', act.increased, showRich, hide)}>
              ↑{act.increased} Inc
            </span>
          )}
          {act.trimmed > 0 && (
            <span className="activity-pill activity-trimmed" {...pillRichProps(names.trimmed, 'Trimmed', act.trimmed, showRich, hide)}>
              ↓{act.trimmed} Trim
            </span>
          )}
          {act.closed > 0 && (
            <span className="activity-pill activity-closed" {...pillRichProps(names.closed, 'Closed Positions', act.closed, showRich, hide)}>
              ✕{act.closed} Closed
            </span>
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

      <HiTooltip {...tip} />
    </div>
  );
});

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

  const latestFilingDate = useMemo(() => {
    if (!managers.length) return null;
    return managers.reduce((max, m) =>
      !max || m.latest_report_date > max ? m.latest_report_date : max, null);
  }, [managers]);

  const stalenessMonths = useMemo(() => {
    if (!latestFilingDate) return null;
    const d = new Date(latestFilingDate + 'T00:00:00');
    const now = new Date();
    return (now.getFullYear() - d.getFullYear()) * 12 + (now.getMonth() - d.getMonth());
  }, [latestFilingDate]);

  if (loading) return <div className="f13f-loading">Loading institutional holders…</div>;
  if (error) return <div className="f13f-error">{error}</div>;

  return (
    <div className="f13f-container">
      <div className="f13f-page-header">
        <h2>13F Institutional Holdings</h2>
        <p className="f13f-page-subtitle">
          {managers.length} tracked managers · SEC 13F filings report with 45-day lag
          {latestFilingDate && ` · Latest data: ${formatQuarter(latestFilingDate)}`}
        </p>
        {stalenessMonths > 4 && (
          <div className="f13f-stale-warning">
            Data is {stalenessMonths} months old — positions may have changed significantly.
            Run <code>scripts/scrape_13f.py</code> to update.
          </div>
        )}
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

const PositionCard = ({ position }) => {
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

// Defined outside PortfolioTable so React sees a stable component type across renders.
const SortHeader = ({ label, field, sortKey, sortDir, onSort }) => (
  <th className="sortable-th" onClick={() => onSort(field)}>
    {label}
    <span className="sort-arrow">
      {sortKey === field ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ' ↕'}
    </span>
  </th>
);

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
              <SortHeader label="#"          field="rank"            sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
              <SortHeader label="Ticker"     field="yahoo_symbol"    sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
              <SortHeader label="Company"    field="name"            sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
              <SortHeader label="Value"      field="value"           sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
              <SortHeader label="% Fund"     field="pct_of_portfolio" sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
              <SortHeader label="Shares"     field="shares"          sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
              <SortHeader label="QoQ Change" field="change_sort"     sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
              <SortHeader label="Value Δ"    field="value_change"    sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
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

  // Must be computed before any early returns (Rules of Hooks — no conditional hook calls).
  // prevTotal must include closed positions (value_prev > 0, not in portfolio array).
  // Without them, aumChange is overstated by the full value of any closed positions.
  const prevTotal = useMemo(() => {
    if (!data?.prev_report_date) return null;
    const openPrev   = (data.portfolio || []).reduce((s, p) => s + (p.value_prev || 0), 0);
    const closedPrev = (data.moves?.closed_positions || []).reduce((s, p) => s + (p.value_prev || 0), 0);
    return openPrev + closedPrev;
  }, [data]);

  const handleQuarterChange = (e) => {
    const q = e.target.value;
    setSelectedQuarter(q);
    fetchData(q);
  };

  // On initial load (no data yet) show full-page loader.
  // On quarter change data is still set to the previous quarter — keep the header
  // visible and only replace the tab content with a loading indicator.
  if (loading && !data) return <div className="f13f-loading">Loading portfolio…</div>;
  if (error) return (
    <div className="f13f-container">
      <Link to="/13f" className="f13f-back">← Back to all managers</Link>
      <div className="f13f-error">{error}</div>
    </div>
  );
  if (!data) return null;

  const { manager, available_quarters, report_date, prev_report_date, total_value, num_positions, portfolio, moves } = data;
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
        {loading
          ? <div className="f13f-loading">Loading…</div>
          : activeTab === 'moves'
            ? <MovesSection moves={moves} />
            : <PortfolioTable portfolio={portfolio} />
        }
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
