// Composite score helpers shared by Holdings (table column) and Stock (KPI tile).

// Fallback when the backend sends no screener_score_max (it normally does, scaled
// down for sectors excluded from several screeners). Mirrors SCORE_NORMALIZER.
export const SCREENER_NORMALIZER = 60;

/**
 * Screener score as a fraction of what its sector can reach.
 *
 * Colour thresholds must use this rather than the raw score: the sector-scaled
 * max runs from 60 down to ~23, so a fixed cutoff is unreachable for financials.
 *
 * @returns {number|null} ratio, or null when there is no score
 */
export function screenerRatio(score, max) {
  if (score == null) return null;
  return score / (max ?? SCREENER_NORMALIZER);
}

const MS_PER_AVG_MONTH = 1000 * 60 * 60 * 24 * 30.44;

/**
 * Date the earnings signal should be aged from.
 *
 * Falls back to the report's own date when no announcement date could be matched.
 * For SEC filings that date is period-end, up to ~90 days early, so decay fires
 * slightly sooner than it should — conservative, and better than no decay at all.
 */
export function effectiveSignalDate(h) {
  return h.earnings_announcement_date ?? h.earnings_report_date ?? null;
}

/** @returns {number|null} months since announcement, or null if missing/invalid */
export function earningsReportAgeMonths(announcementDateStr) {
  if (!announcementDateStr) return null;
  const t = new Date(announcementDateStr).getTime();
  if (!Number.isFinite(t)) return null;
  return (Date.now() - t) / MS_PER_AVG_MONTH;
}

/**
 * Composite earnings weight by age. Tooltip age copy. Invalid/missing age → full weight.
 */
export function earningsAgeDecay(monthsOld) {
  if (monthsOld == null || !Number.isFinite(monthsOld)) {
    return { freshness: 1, ageLabel: null };
  }
  if (monthsOld >= 24) {
    return {
      freshness: 0,
      ageLabel: `${Math.round(monthsOld)} months ago — excluded from score`,
    };
  }
  if (monthsOld >= 12) {
    return {
      freshness: 0.6,
      ageLabel: `${Math.round(monthsOld)} months ago — reduced weight in score`,
    };
  }
  return {
    freshness: 1,
    ageLabel: `${Math.round(monthsOld)} months ago`,
  };
}

/**
 * 13F score feeding the composite.
 *
 * A fund's aggregate arrives on its own key: writing it to `form13f_score` would
 * fill the 13F column, asserting managers hold the *fund* when what was measured
 * is that they hold its constituents.
 *
 * @returns {number|null}
 */
export function form13fScore(h) {
  return h.look_through ? (h.look_through_form13f ?? null) : (h.form13f_score ?? null);
}

/**
 * Composite score (–10 … 10):
 *   Screener quality score    50%  — screener_score / screener_score_max, which
 *                                    the backend scales down for sectors excluded
 *                                    from several screeners. Negative values are
 *                                    preserved so red-flag stocks stay negative.
 *   Earnings signal strength  25%  — conviction-adjusted
 *   Analyst recommendation    10%  — recommendation_mean (1=strong buy … 5=strong sell)
 *   Institutional 13F score   15%
 *
 * Funds return null unless the backend attached look-through metrics, in which
 * case the screener and 13F legs come from their constituents (see
 * backend/views/_etf_overlay.py). Earnings is never aggregated, so a fund scores
 * on three legs at most.
 * Missing components are re-weighted proportionally so the formula stays fair
 * (e.g. non-US stocks without 13F data, stocks without an analysed report).
 * Output is capped at 10; there is no lower cap — losers can go negative.
 */
export function computeComposite(h) {
  // An absent key means the caller forgot a field; a null value means the
  // instrument genuinely has no sector. Only the first is a bug, so say so.
  if (h.sector === undefined) {
    console.error('computeComposite: input is missing `sector` — score suppressed', h);
    return null;
  }
  // Funds have no equity screener data or earnings signal — show blank. The
  // backend decides which (is_fund); a missing sector also marks a degraded profile.
  if (!h.look_through && h.is_fund) return null;

  // Only the upper bound is clamped: scores above the sector max are possible but
  // shouldn't outweigh their 50%, while red-flag stocks must stay negative.
  const max = h.screener_score_max ?? SCREENER_NORMALIZER;
  const screenerRaw = h.screener_score != null ? Math.min(1, h.screener_score / max) : null;

  // buy and consider sit close together because forward returns don't separate
  // them; avoid is 0 because it does predict underperformance.
  const SIGNAL_VALUES = { buy: 0.95, consider: 0.90, hold: 0.40, avoid: 0.0 };
  const CONV_MULT    = { high: 1.05, medium: 1.0, low: 0.95 };
  let signalRaw = h.earnings_signal ? (SIGNAL_VALUES[h.earnings_signal] ?? null) : null;
  if (signalRaw != null && h.earnings_conviction) {
    signalRaw = Math.min(1, signalRaw * (CONV_MULT[h.earnings_conviction] ?? 1));
  }
  // Decay signal by age: full weight < 12 months, 60% for 12–24 months, excluded after 24 months.
  // Excluded means null (remaining components reweight) — a zero-value signal
  // would score a stale report worse than a fresh "avoid".
  const signalDate = effectiveSignalDate(h);
  if (signalRaw != null && signalDate) {
    const monthsOld = earningsReportAgeMonths(signalDate);
    const { freshness } = earningsAgeDecay(monthsOld);
    signalRaw = freshness === 0 ? null : signalRaw * freshness;
  }

  // recommendation_mean: 1=strong buy … 5=strong sell → normalise to [0, 1]
  // Validate range before using: some tickers return values outside [1, 5]
  const recRaw = (h.recommendation_mean != null &&
                  h.recommendation_mean >= 1 &&
                  h.recommendation_mean <= 5)
    ? (5 - h.recommendation_mean) / 4
    : null;

  // form13f_score is roughly –2 … +2; normalise to [0, 1]
  const f13f = form13fScore(h);
  const f13fRaw = f13f != null
    ? Math.max(0, Math.min(1, (f13f + 2) / 4))
    : null;

  const components = [
    { val: screenerRaw, weight: 0.50 },
    { val: signalRaw,   weight: 0.25 },
    { val: recRaw,      weight: 0.10 },
    { val: f13fRaw,     weight: 0.15 },
  ].filter(c => c.val != null);

  if (components.length === 0) return null;

  const totalWeight = components.reduce((s, c) => s + c.weight, 0);
  const weightedSum = components.reduce((s, c) => s + c.val * c.weight, 0);
  const raw = (weightedSum / totalWeight) * 10;
  // Cap at 10; NO lower cap — negative scores identify the portfolio's weakest links
  return Math.round(Math.min(10, raw) * 10) / 10;
}

/**
 * Tooltip text describing which components fed a composite score and at what
 * effective weights. Mirrors the Holdings score-cell tooltip.
 */
export function compositeTooltip(score, h) {
  const hasScreener = h.screener_score != null;
  const signalAge   = earningsAgeDecay(earningsReportAgeMonths(effectiveSignalDate(h)));
  const hasSignal   = !!h.earnings_signal && signalAge.freshness > 0;
  const hasRec      = h.recommendation_mean != null &&
                      h.recommendation_mean >= 1 &&
                      h.recommendation_mean <= 5;
  const f13f        = form13fScore(h);
  const hasF13f     = f13f != null;
  const presentTotal =
    (hasScreener ? 50 : 0) +
    (hasSignal   ? 25 : 0) +
    (hasRec      ? 10 : 0) +
    (hasF13f     ? 15 : 0);
  const eff = (base) =>
    presentTotal > 0 ? Math.round(base / presentTotal * 100) : 0;

  // A fund's points are its constituents' average, not gates it passed — same
  // scale as a stock's line, but say which it is.
  const screenerMax = Math.round(h.screener_score_max ?? SCREENER_NORMALIZER);
  const screenerLine = !hasScreener
    ? 'Screener: no data'
    : h.look_through
      ? `Screener: ${Math.round(h.screener_score)} / ${screenerMax} pts, constituent avg  (eff. ${eff(50)}%)`
      : `Screener: ${h.screener_score} / ${screenerMax} pts  (eff. ${eff(50)}%)`;
  // Partially-decayed signals (12–24 months) still count, so they take the first
  // branch — surface the age there or the tooltip implies undecayed full weight.
  const signalLine = hasSignal
    ? `Signal: ${h.earnings_signal}${h.earnings_conviction ? ` (${h.earnings_conviction})` : ''}` +
      `${signalAge.freshness < 1 ? ` — ${signalAge.ageLabel}` : ''}  (eff. ${eff(25)}%)`
    : h.earnings_signal
      ? `Signal: ${h.earnings_signal} — ${signalAge.ageLabel}`
      : 'Signal: no earnings report analysed yet';
  const recLine = hasRec
    ? `Analyst rec: ${h.recommendation_mean?.toFixed(1)}/5  (eff. ${eff(10)}%)`
    : 'Analyst rec: no data';
  const f13fLine = hasF13f
    ? `13F: score ${f13f.toFixed(1)}  (eff. ${eff(15)}%)`
    : '13F: no data (non-US or not filed)';

  const lines = [screenerLine, signalLine, recLine, f13fLine];
  if (h.look_through) {
    // Earnings is never aggregated, so the reweighting is large and invisible.
    lines.push(
      `Look-through: weighted average of ${h.look_through_n} constituents` +
      `${h.look_through_as_of ? `, ${h.look_through_as_of}` : ''} — no earnings signal for funds`
    );
  }
  return `Score: ${score.toFixed(1)} / 10\n${lines.join('\n')}`;
}
