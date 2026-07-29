import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { portfolioAPI } from '../services/api';
import { useHideAmounts, MASK } from '../context/HideAmountsContext';
import './Agent.css';

const formatGbp = (value) => {
  if (value == null || !Number.isFinite(value)) return '—';
  return `£${Math.round(value).toLocaleString()}`;
};

const formatPct = (value, digits = 1) => {
  if (value == null || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
};

const ACTION_ORDER = { exit: 0, trim: 1, add: 2, buy: 3 };

const SCORE_TOOLTIP =
  'Composite conviction score vs the whole monitored universe (~1,000 symbols): ' +
  'weighted z-scores of momentum, trend, relative strength, screener score and quality. ' +
  '0 = universe average, positive = stronger, negative = weaker (roughly ±3 at the extremes). ' +
  'Exits triggered by thesis rules can have a positive score — the rule overrides the ranking.';

const Agent = () => {
  const { hideAmounts } = useHideAmounts();
  const [data, setData] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

  const load = useCallback(async () => {
    try {
      const [today, hist] = await Promise.all([
        portfolioAPI.getAgentSuggestions(),
        portfolioAPI.getAgentSuggestionsHistory(30),
      ]);
      setData(today);
      setHistory(hist.suggestions.filter((s) => s.date !== today.date));
    } catch (e) {
      setError('Failed to load trade suggestions');
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (error) {
    return (
      <div className="page-fixed agent-container">
        <div className="agent-error">{error}</div>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="page-fixed agent-container">
        <div className="agent-loading">Loading suggestions…</div>
      </div>
    );
  }

  const maskable = (v) => (hideAmounts ? MASK : formatGbp(v));
  const suggestions = [...data.suggestions].sort(
    (a, b) => (ACTION_ORDER[a.action] ?? 9) - (ACTION_ORDER[b.action] ?? 9) || b.value_gbp - a.value_gbp
  );
  const vetoed = suggestions.filter((s) => s.value_gbp === 0);
  const actionable = suggestions.filter((s) => s.value_gbp > 0);

  const row = (s, showDate = false) => (
    <React.Fragment key={s.id}>
      <tr className={`agent-row ${expandedId === s.id ? 'expanded' : ''}`} onClick={() => setExpandedId(expandedId === s.id ? null : s.id)}>
        {showDate && <td>{s.date}</td>}
        <td>
          <span className={`action-badge action-${s.action}`}>{s.action.toUpperCase()}</span>
        </td>
        <td>
          <Link to={`/stock/${s.symbol}`} onClick={(e) => e.stopPropagation()}>
            {s.symbol}
          </Link>
          <span className="agent-name">{s.name}</span>
        </td>
        <td className="num">{s.value_gbp > 0 ? maskable(s.value_gbp) : 'vetoed'}</td>
        <td className="num">
          {formatPct(s.weight_before)} → {formatPct(s.weight_after)}
        </td>
        <td className="num">
          {s.rationale?.composite != null
            ? s.rationale.composite.toFixed(2)
            : s.score != null
              ? s.score.toFixed(2)
              : '—'}
        </td>
        <td className="agent-reason">
          {s.rationale?.trigger || '—'}
          {s.constraint_adjustments.length > 0 && <span className="agent-reason-flag"> ⚠</span>}
        </td>
        <td>
          <span className={`status-badge status-${s.status}`}>{s.status}</span>
        </td>
      </tr>
      {expandedId === s.id && (
        <tr className="agent-detail-row">
          <td colSpan={showDate ? 8 : 7}>
            {s.rationale?.composite != null && (
              <div className="agent-detail-line">
                Composite score {s.rationale.composite}
                {s.rationale.percentile != null && ` (percentile ${Math.round(s.rationale.percentile * 100)} of universe)`}
                {s.score != null && ` — budget priority ${s.score.toFixed(2)}`}
              </div>
            )}
            {s.constraint_adjustments.length > 0 && (
              <div className="agent-detail-line agent-adjustments">
                {s.constraint_adjustments.map((a, i) => (
                  <div key={i}>⚠ {a}</div>
                ))}
              </div>
            )}
            {s.fee_gbp > 0 && <div className="agent-detail-line">FX fee: £{s.fee_gbp.toFixed(2)}</div>}
          </td>
        </tr>
      )}
    </React.Fragment>
  );

  const headers = (showDate = false) => (
    <tr>
      {showDate && <th>Date</th>}
      <th>Action</th>
      <th>Instrument</th>
      <th className="num">Value</th>
      <th className="num">Weight</th>
      <th className="num">
        <span className="agent-tooltip" title={SCORE_TOOLTIP}>
          Score
        </span>
      </th>
      <th>Reason</th>
      <th>Status</th>
    </tr>
  );

  return (
    <div className="page-fixed agent-container">
      <div className="agent-header">
        <h1>Agent</h1>
        <p className="agent-subtitle">
          Daily trade suggestions from the rules strategy — momentum + screener composite with HRP risk
          tilts, passed through the safety caps (5% daily budget per side — sells never crowd out buys —
          plus position and cluster limits). Suggest-only: nothing is executed. Click a row for details.
        </p>
      </div>

      {data.date == null ? (
        <div className="agent-empty">No suggestions yet — run scripts/run_trade_agent.py on the server.</div>
      ) : (
        <>
          <h2 className="agent-section-title">Suggestions for {data.date}</h2>
          {actionable.length === 0 && <div className="agent-empty">Nothing to do today — the book is where the strategy wants it.</div>}
          {actionable.length > 0 && (
            <table className="agent-table">
              <thead>{headers()}</thead>
              <tbody>{actionable.map((s) => row(s))}</tbody>
            </table>
          )}
          {vetoed.length > 0 && (
            <>
              <h2 className="agent-section-title muted">Vetoed by safety caps</h2>
              <table className="agent-table muted">
                <thead>{headers()}</thead>
                <tbody>{vetoed.map((s) => row(s))}</tbody>
              </table>
            </>
          )}
        </>
      )}

      {history.length > 0 && (
        <>
          <h2 className="agent-section-title">Last 30 days</h2>
          <table className="agent-table">
            <thead>{headers(true)}</thead>
            <tbody>{history.map((s) => row(s, true))}</tbody>
          </table>
        </>
      )}
    </div>
  );
};

export default Agent;
