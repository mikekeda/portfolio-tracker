import React, { useCallback, useEffect, useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { portfolioAPI } from '../services/api';
import './EarningsCalendar.css';

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];

// Returns 0-based Mon–Sun index (Mon=0 … Sun=6)
const isoDay = (d) => {
  const day = d.getDay();
  return day === 0 ? 6 : day - 1;
};

// Build a flat array of ISO date strings (or null for padding) for a 5-day grid.
// Skips Sat/Sun entirely. Leading/trailing nulls fill incomplete first/last rows.
const buildMonthGrid = (year, month) => {
  const firstDay = new Date(year, month, 1);
  const lastDay = new Date(year, month + 1, 0);
  const firstDow = isoDay(firstDay); // 0=Mon..6=Sun
  const leadingPad = firstDow < 5 ? firstDow : 0;

  const cells = Array(leadingPad).fill(null);
  for (let d = new Date(firstDay); d <= lastDay; d.setDate(d.getDate() + 1)) {
    const dow = d.getDay(); // 0=Sun..6=Sat
    if (dow === 0 || dow === 6) continue;
    cells.push(toLocalIso(new Date(d)));
  }
  while (cells.length % 5 !== 0) cells.push(null);
  return cells;
};

const toLocalIso = (d) => {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${dd}`;
};

const todayIso = toLocalIso(new Date());

// ── Shared helpers ────────────────────────────────────────────────────────────

const SurpriseBadge = ({ pct }) => {
  if (pct == null) return null;
  const positive = pct >= 0;
  return (
    <span className={`ec-surprise ${positive ? 'beat' : 'miss'}`}>
      {positive ? '+' : ''}{pct.toFixed(1)}%
    </span>
  );
};

SurpriseBadge.propTypes = { pct: PropTypes.number };

const SIGNAL_CLASSES = { buy: 'sig-buy', consider: 'sig-consider', hold: 'sig-hold', avoid: 'sig-avoid' };

const SignalPill = ({ signal, conviction }) => {
  if (!signal) return null;
  const label = signal.charAt(0).toUpperCase() + signal.slice(1);
  return (
    <span className={`ec-signal-pill ${SIGNAL_CLASSES[signal] || ''}`} title={conviction ? `${label} · ${conviction} conviction` : label}>
      {label}
      {conviction && <span className="ec-signal-conv"> · {conviction}</span>}
    </span>
  );
};

SignalPill.propTypes = {
  signal: PropTypes.string,
  conviction: PropTypes.string,
};

const PriceDelta = ({ priceAtDate, currentPrice, priceChangePct }) => {
  if (priceAtDate == null || currentPrice == null || priceChangePct == null) return null;
  const positive = priceChangePct >= 0;
  return (
    <div className="ec-price-delta">
      <span className="ec-price-delta-label">At earnings</span>
      <span className="ec-price-delta-at">${priceAtDate.toFixed(2)}</span>
      <span className="ec-price-delta-sep">→</span>
      <span className="ec-price-delta-now">${currentPrice.toFixed(2)}</span>
      <span className={`ec-price-delta-pct ${positive ? 'positive' : 'negative'}`}>
        {positive ? '+' : ''}{priceChangePct.toFixed(1)}%
      </span>
    </div>
  );
};

PriceDelta.propTypes = {
  priceAtDate: PropTypes.number,
  currentPrice: PropTypes.number,
  priceChangePct: PropTypes.number,
};

// ── Event chip (tiny, shown in calendar cells) ───────────────────────────────

const eventChipClass = (event) => {
  if (event.type === 'upcoming') return 'ec-chip-upcoming';
  if (event.surprise_pct != null) return event.surprise_pct >= 0 ? 'ec-chip-beat' : 'ec-chip-miss';
  return 'ec-chip-past';
};

const EventChip = ({ event, onDayClick }) => {
  const surpriseTip = event.surprise_pct != null
    ? ` · ${event.surprise_pct >= 0 ? '+' : ''}${event.surprise_pct.toFixed(1)}% surprise`
    : '';
  const signalTip = event.signal ? ` · ${event.signal}` : '';
  return (
    <button
      className={`ec-chip ${eventChipClass(event)}${event.signal ? ` ec-chip-sig-${event.signal}` : ''}`}
      onClick={(e) => { e.stopPropagation(); onDayClick(event.date); }}
      title={`${event.name}${surpriseTip}${signalTip}`}
    >
      <span className="ec-chip-symbol">{event.symbol}</span>
      {event.surprise_pct != null && (
        <span className={`ec-chip-dot ${event.surprise_pct >= 0 ? 'beat' : 'miss'}`} />
      )}
    </button>
  );
};

EventChip.propTypes = {
  event: PropTypes.shape({
    date: PropTypes.string.isRequired,
    symbol: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    type: PropTypes.string.isRequired,
    surprise_pct: PropTypes.number,
    signal: PropTypes.string,
  }).isRequired,
  onDayClick: PropTypes.func.isRequired,
};

// ── Detail panel for all events on a selected day ────────────────────────────

const fmtEps = (v) =>
  v != null ? (v >= 0 ? `$${v.toFixed(2)}` : `-$${Math.abs(v).toFixed(2)}`) : '—';

// Format a period-end date as a quarter label, e.g. "Q4 2025 · Dec 31"
const fmtPeriod = (isoDate) => {
  if (!isoDate) return null;
  const d = new Date(isoDate + 'T00:00:00');
  const m = d.getMonth();
  const q = m <= 2 ? 'Q1' : m <= 5 ? 'Q2' : m <= 8 ? 'Q3' : 'Q4';
  const day = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  return `${q} ${d.getFullYear()} · ends ${day}`;
};

const DayDetail = ({ date, events }) => {
  if (!date) {
    return <p className="ec-detail-empty">Click a day to see all earnings details.</p>;
  }

  const dateLabel = new Date(date + 'T00:00:00').toLocaleDateString('en-GB', {
    weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
  });

  return (
    <div className="ec-day-detail">
      <p className="ec-day-detail-date">{dateLabel}</p>
      <div className="ec-day-detail-list">
        {events.map((ev) => {
          const hasEps = ev.eps_estimate != null || ev.eps_actual != null;
          const periodLabel = fmtPeriod(ev.report_period_date);
          return (
            <div key={ev.symbol} className={`ec-day-event ec-day-event-${ev.type}`}>
              <div className="ec-day-event-header">
                <Link to={`/stock/${ev.symbol}`} className="ec-detail-symbol">{ev.symbol}</Link>
                <span className={`ec-detail-badge ec-detail-badge-${ev.type}`}>
                  {ev.type === 'upcoming' ? 'Upcoming' : 'Past'}
                </span>
                {ev.signal && <SignalPill signal={ev.signal} conviction={ev.conviction} />}
              </div>
              {periodLabel && (
                <p className="ec-report-period" title="SEC filing period this report covers">
                  Report · {periodLabel}
                </p>
              )}
              <p className="ec-day-event-name">{ev.name}</p>

              {hasEps && (
                <div className="ec-detail-eps">
                  <div className="ec-detail-eps-row">
                    <span className="ec-detail-eps-label">EPS est.</span>
                    <span className="ec-detail-eps-value">{fmtEps(ev.eps_estimate)}</span>
                  </div>
                  <div className="ec-detail-eps-row">
                    <span className="ec-detail-eps-label">EPS actual</span>
                    <span className="ec-detail-eps-value">{fmtEps(ev.eps_actual)}</span>
                  </div>
                  {ev.surprise_pct != null && (
                    <div className="ec-detail-eps-row">
                      <span className="ec-detail-eps-label">Surprise</span>
                      <SurpriseBadge pct={ev.surprise_pct} />
                    </div>
                  )}
                </div>
              )}

              <PriceDelta
                priceAtDate={ev.price_at_date}
                currentPrice={ev.current_price}
                priceChangePct={ev.price_change_pct}
              />

              {ev.rationale_snippet && (
                <p className="ec-rationale-snippet">&quot;{ev.rationale_snippet}{ev.rationale_snippet.length >= 180 ? '…' : ''}&quot;</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

DayDetail.propTypes = {
  date: PropTypes.string,
  events: PropTypes.arrayOf(PropTypes.object).isRequired,
};

// ── Month view (5-day Mon–Fri grid) ──────────────────────────────────────────

const MonthView = ({ year, month, eventsByDate, onDayClick, selectedDate }) => {
  const cells = buildMonthGrid(year, month);

  return (
    <div className="ec-month-grid">
      {WEEKDAYS.map((d) => (
        <div key={d} className="ec-month-header-cell">{d}</div>
      ))}
      {cells.map((iso, i) => {
        if (!iso) return <div key={i} className="ec-month-cell ec-month-cell-empty" />;
        const events = eventsByDate[iso] || [];
        const isToday = iso === todayIso;
        const isSelected = selectedDate === iso;
        return (
          <div
            key={iso}
            className={`ec-month-cell${isToday ? ' today' : ''}${isSelected ? ' selected' : ''}${events.length ? ' has-events' : ''}`}
            onClick={() => events.length && onDayClick(iso)}
          >
            <span className="ec-month-day-num">{new Date(iso + 'T00:00:00').getDate()}</span>
            <div className="ec-month-chips">
              {events.slice(0, 4).map((ev) => (
                <EventChip key={ev.symbol} event={ev} onDayClick={onDayClick} />
              ))}
              {events.length > 4 && (
                <button
                  className="ec-chip ec-chip-overflow"
                  onClick={(e) => { e.stopPropagation(); onDayClick(iso); }}
                >
                  +{events.length - 4} more
                </button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

MonthView.propTypes = {
  year: PropTypes.number.isRequired,
  month: PropTypes.number.isRequired,
  eventsByDate: PropTypes.object.isRequired,
  onDayClick: PropTypes.func.isRequired,
  selectedDate: PropTypes.string,
};

// ── List view ─────────────────────────────────────────────────────────────────

const ListView = ({ year, month, eventsByDate, onDayClick, selectedDate }) => {
  const firstDay = new Date(year, month, 1);
  const lastDay = new Date(year, month + 1, 0);

  // Group days into ISO weeks
  const weeks = [];
  let currentWeek = [];
  for (let d = new Date(firstDay); d <= lastDay; d.setDate(d.getDate() + 1)) {
    const iso = toLocalIso(new Date(d));
    const events = eventsByDate[iso] || [];
    if (events.length) currentWeek.push({ iso, events });
    if (isoDay(new Date(d)) === 6 && currentWeek.length) {
      weeks.push(currentWeek);
      currentWeek = [];
    }
  }
  if (currentWeek.length) weeks.push(currentWeek);

  if (!weeks.length) {
    return <p className="ec-list-empty">No earnings events this month.</p>;
  }

  return (
    <div className="ec-list">
      {weeks.map((week, wi) => (
        <div key={wi} className="ec-list-week">
          {week.map(({ iso, events }) => {
            const d = new Date(iso + 'T00:00:00');
            const isToday = iso === todayIso;
            const isSelected = selectedDate === iso;
            return (
              <div
                key={iso}
                className={`ec-list-day${isToday ? ' today' : ''}${isSelected ? ' selected' : ''}`}
                onClick={() => onDayClick(iso)}
              >
                <div className="ec-list-day-label">
                  <span className="ec-list-weekday">
                    {d.toLocaleDateString('en-GB', { weekday: 'short' })}
                  </span>
                  <span className={`ec-list-date-num${isToday ? ' today' : ''}`}>
                    {d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}
                  </span>
                </div>
                <div className="ec-list-events">
                  {events.map((ev) => (
                    <div key={ev.symbol} className={`ec-list-event ec-list-event-${ev.type}`}>
                      <span className="ec-list-event-symbol">{ev.symbol}</span>
                      <span className="ec-list-event-name">{ev.name}</span>
                      {ev.surprise_pct != null && <SurpriseBadge pct={ev.surprise_pct} />}
                      {ev.signal && <SignalPill signal={ev.signal} conviction={null} />}
                      {ev.price_change_pct != null && (
                        <span className={`ec-list-price-chg ${ev.price_change_pct >= 0 ? 'positive' : 'negative'}`}>
                          {ev.price_change_pct >= 0 ? '+' : ''}{ev.price_change_pct.toFixed(1)}%
                        </span>
                      )}
                      {ev.type === 'upcoming' && (
                        <span className="ec-list-event-est">est.</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

ListView.propTypes = {
  year: PropTypes.number.isRequired,
  month: PropTypes.number.isRequired,
  eventsByDate: PropTypes.object.isRequired,
  onDayClick: PropTypes.func.isRequired,
  selectedDate: PropTypes.string,
};

// ── Month stats (shown in sidebar when no day is selected) ───────────────────

const MonthStats = ({ monthEvents }) => {
  const past = monthEvents.filter((e) => e.type === 'past');
  const upcoming = monthEvents.filter((e) => e.type === 'upcoming');
  const withSurprise = past.filter((e) => e.surprise_pct != null);
  const beats = withSurprise.filter((e) => e.surprise_pct >= 0).length;
  const misses = withSurprise.filter((e) => e.surprise_pct < 0).length;
  const beatRate = withSurprise.length > 0 ? Math.round((beats / withSurprise.length) * 100) : null;

  if (past.length === 0 && upcoming.length === 0) {
    return <p className="ec-detail-empty">No events this month.</p>;
  }

  return (
    <div className="ec-month-stats">
      <p className="ec-month-stats-title">This month</p>
      <div className="ec-month-stats-grid">
        {past.length > 0 && (
          <div className="ec-stats-item">
            <span className="ec-stats-num">{past.length}</span>
            <span className="ec-stats-label">Reported</span>
          </div>
        )}
        {beats > 0 && (
          <div className="ec-stats-item beat">
            <span className="ec-stats-num">{beats}{beatRate != null ? ` (${beatRate}%)` : ''}</span>
            <span className="ec-stats-label">Beat</span>
          </div>
        )}
        {misses > 0 && (
          <div className="ec-stats-item miss">
            <span className="ec-stats-num">{misses}</span>
            <span className="ec-stats-label">Missed</span>
          </div>
        )}
        {upcoming.length > 0 && (
          <div className="ec-stats-item upcoming">
            <span className="ec-stats-num">{upcoming.length}</span>
            <span className="ec-stats-label">Upcoming</span>
          </div>
        )}
      </div>
      <p className="ec-month-stats-hint">Click a day to see details</p>
    </div>
  );
};

MonthStats.propTypes = {
  monthEvents: PropTypes.arrayOf(PropTypes.object).isRequired,
};

// ── Next upcoming banner (shown when no upcoming events in current month) ─────

const NextUpcoming = ({ allEvents, onJump }) => {
  const next = allEvents
    .filter((e) => e.type === 'upcoming' && e.date > todayIso)
    .sort((a, b) => a.date.localeCompare(b.date));

  if (!next.length) return null;

  const firstDate = next[0].date;
  const sameDay = next.filter((e) => e.date === firstDate);
  const dateLabel = new Date(firstDate + 'T00:00:00').toLocaleDateString('en-GB', {
    day: 'numeric', month: 'short', year: 'numeric',
  });
  const symbols = sameDay
    .slice(0, 4)
    .map((e) => e.symbol)
    .join(', ');
  const extra = sameDay.length > 4 ? ` +${sameDay.length - 4}` : '';

  return (
    <div className="ec-next-upcoming">
      <div className="ec-next-upcoming-body">
        <span className="ec-next-upcoming-label">Next earnings</span>
        <span className="ec-next-upcoming-date">{dateLabel}</span>
        <span className="ec-next-upcoming-symbols">{symbols}{extra}</span>
      </div>
      <button className="ec-next-upcoming-btn" onClick={() => onJump(firstDate)}>
        View →
      </button>
    </div>
  );
};

NextUpcoming.propTypes = {
  allEvents: PropTypes.arrayOf(PropTypes.object).isRequired,
  onJump: PropTypes.func.isRequired,
};

// ── Main component ────────────────────────────────────────────────────────────

const EarningsCalendar = () => {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [portfolioOnly, setPortfolioOnly] = useState(true);
  const [view, setView] = useState('month'); // 'month' | 'list'
  const [selectedDate, setSelectedDate] = useState(null);

  const now = new Date();
  const [navYear, setNavYear] = useState(now.getFullYear());
  const [navMonth, setNavMonth] = useState(now.getMonth()); // 0-based

  const fetchCalendar = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await portfolioAPI.getEarningsCalendar(portfolioOnly);
      setEvents(data.events || []);
    } catch (e) {
      setError('Failed to load earnings calendar.');
    } finally {
      setLoading(false);
    }
  }, [portfolioOnly]);

  useEffect(() => {
    fetchCalendar();
  }, [fetchCalendar]);

  // Build date → events map for the current view month
  const eventsByDate = useMemo(() => {
    const map = {};
    for (const ev of events) {
      const d = new Date(ev.date + 'T00:00:00');
      if (d.getFullYear() === navYear && d.getMonth() === navMonth) {
        (map[ev.date] = map[ev.date] || []).push(ev);
      }
    }
    // Sort events within each day: upcoming first, then by symbol
    for (const key of Object.keys(map)) {
      map[key].sort((a, b) => {
        if (a.type !== b.type) return a.type === 'upcoming' ? -1 : 1;
        return a.symbol.localeCompare(b.symbol);
      });
    }
    return map;
  }, [events, navYear, navMonth]);

  const monthLabel = new Date(navYear, navMonth, 1).toLocaleDateString('en-GB', {
    month: 'long', year: 'numeric',
  });

  const prevMonth = () => {
    if (navMonth === 0) { setNavYear((y) => y - 1); setNavMonth(11); }
    else setNavMonth((m) => m - 1);
    setSelectedDate(null);
  };

  const nextMonth = () => {
    if (navMonth === 11) { setNavYear((y) => y + 1); setNavMonth(0); }
    else setNavMonth((m) => m + 1);
    setSelectedDate(null);
  };

  const goToToday = () => {
    setNavYear(now.getFullYear());
    setNavMonth(now.getMonth());
    setSelectedDate(null);
  };

  const handleDayClick = (iso) => {
    setSelectedDate((prev) => (prev === iso ? null : iso));
  };

  const handleJumpToDate = (iso) => {
    const d = new Date(iso + 'T00:00:00');
    setNavYear(d.getFullYear());
    setNavMonth(d.getMonth());
    setSelectedDate(iso);
  };

  // Summary counts for current month
  const monthEvents = Object.values(eventsByDate).flat();
  const upcomingCount = monthEvents.filter((e) => e.type === 'upcoming').length;
  const pastCount = monthEvents.filter((e) => e.type === 'past').length;
  const withSurprise = monthEvents.filter((e) => e.type === 'past' && e.surprise_pct != null);
  const beatCount = withSurprise.filter((e) => e.surprise_pct >= 0).length;
  const missCount = withSurprise.filter((e) => e.surprise_pct < 0).length;

  return (
    <div className="ec-container">
      <div className="ec-page-header">
        <h2>Earnings Calendar</h2>
        <p className="ec-page-subtitle">
          Past and upcoming earnings reports
          {portfolioOnly ? ' for your portfolio' : ' across all instruments'}
        </p>
      </div>

      {/* Controls */}
      <div className="ec-controls">
        <div className="ec-controls-left">
          <button className="ec-btn-today" onClick={goToToday}>Today</button>
          <div className="ec-nav">
            <button className="ec-nav-btn" onClick={prevMonth} aria-label="Previous month">‹</button>
            <span className="ec-nav-label">{monthLabel}</span>
            <button className="ec-nav-btn" onClick={nextMonth} aria-label="Next month">›</button>
          </div>
          {(upcomingCount > 0 || pastCount > 0) && (
            <div className="ec-month-summary">
              {pastCount > 0 && (
                <span className="ec-summary-pill past">{pastCount} reported</span>
              )}
              {beatCount > 0 && (
                <span className="ec-summary-pill beat">{beatCount} beat</span>
              )}
              {missCount > 0 && (
                <span className="ec-summary-pill miss">{missCount} miss</span>
              )}
              {upcomingCount > 0 && (
                <span className="ec-summary-pill upcoming">{upcomingCount} upcoming</span>
              )}
            </div>
          )}
        </div>

        <div className="ec-controls-right">
          {/* Portfolio toggle */}
          <div className="ec-toggle-group">
            <button
              className={`ec-toggle-btn${portfolioOnly ? ' active' : ''}`}
              onClick={() => setPortfolioOnly(true)}
            >
              Portfolio
            </button>
            <button
              className={`ec-toggle-btn${!portfolioOnly ? ' active' : ''}`}
              onClick={() => setPortfolioOnly(false)}
            >
              All
            </button>
          </div>

          {/* View toggle */}
          <div className="ec-toggle-group">
            <button
              className={`ec-toggle-btn${view === 'month' ? ' active' : ''}`}
              onClick={() => setView('month')}
              title="Month view"
            >
              ▦ Month
            </button>
            <button
              className={`ec-toggle-btn${view === 'list' ? ' active' : ''}`}
              onClick={() => setView('list')}
              title="List view"
            >
              ☰ List
            </button>
          </div>
        </div>
      </div>

      {loading && <div className="ec-loading">Loading earnings data…</div>}
      {error && <div className="ec-error">{error}</div>}

      {!loading && !error && upcomingCount === 0 && (
        <NextUpcoming allEvents={events} onJump={handleJumpToDate} />
      )}

      {!loading && !error && (
        <div className="ec-body">
          <div className="ec-main">
            {view === 'month' ? (
              <MonthView
                year={navYear}
                month={navMonth}
                eventsByDate={eventsByDate}
                onDayClick={handleDayClick}
                selectedDate={selectedDate}
              />
            ) : (
              <ListView
                year={navYear}
                month={navMonth}
                eventsByDate={eventsByDate}
                onDayClick={handleDayClick}
                selectedDate={selectedDate}
              />
            )}
          </div>

          <aside className="ec-sidebar">
            <h4 className="ec-sidebar-title">
              {selectedDate ? 'Details' : 'Summary'}
            </h4>
            {selectedDate ? (
              <DayDetail
                date={selectedDate}
                events={eventsByDate[selectedDate] || []}
              />
            ) : (
              <MonthStats monthEvents={monthEvents} />
            )}

            <div className="ec-legend">
              <h5 className="ec-legend-title">Legend</h5>
              <div className="ec-legend-row">
                <span className="ec-chip ec-chip-upcoming" style={{ pointerEvents: 'none' }}>
                  <span className="ec-chip-symbol">AAPL</span>
                </span>
                <span>Upcoming</span>
              </div>
              <div className="ec-legend-row">
                <span className="ec-chip ec-chip-beat" style={{ pointerEvents: 'none' }}>
                  <span className="ec-chip-symbol">NVDA</span>
                  <span className="ec-chip-dot beat" />
                </span>
                <span>Beat consensus</span>
              </div>
              <div className="ec-legend-row">
                <span className="ec-chip ec-chip-miss" style={{ pointerEvents: 'none' }}>
                  <span className="ec-chip-symbol">META</span>
                  <span className="ec-chip-dot miss" />
                </span>
                <span>Missed consensus</span>
              </div>
              <div className="ec-legend-row">
                <span className="ec-chip ec-chip-past" style={{ pointerEvents: 'none' }}>
                  <span className="ec-chip-symbol">MSFT</span>
                </span>
                <span>No EPS data</span>
              </div>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
};

export default EarningsCalendar;
