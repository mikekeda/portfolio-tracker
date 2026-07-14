// Composite score helpers shared by Holdings (table column) and Stock (KPI tile).

// Screener score normalisation baseline (see computeComposite).
// 50 ≈ sum of the 5 highest screener weights (9+9+9+8+8 = 43) plus a typical
// cross-category combination bonus (~6 pts).  A round number, easy to reason
// about, and stable regardless of which stocks are loaded.
export const SCREENER_NORMALIZER = 50;

const MS_PER_AVG_MONTH = 1000 * 60 * 60 * 24 * 30.44;

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
 * Composite score (–10 … 10):
 *   Screener quality score    50%  — screener_score / SCREENER_NORMALIZER (50).
 *                                    Stable across all views; negative values
 *                                    preserved so red-flag stocks stay negative.
 *   Earnings signal strength  25%  — conviction-adjusted
 *   Analyst recommendation    10%  — recommendation_mean (1=strong buy … 5=strong sell)
 *   Institutional 13F score   15%
 *
 * ETFs always return null (they can't pass equity screeners, have no signal).
 * Missing components are re-weighted proportionally so the formula stays fair
 * (e.g. non-US stocks without 13F data, stocks without an analysed report).
 * Output is capped at 10; there is no lower cap — losers can go negative.
 */
export function computeComposite(h) {
  // ETFs have no equity screener data or earnings signal — show blank
  if (h.quote_type === 'ETF') return null;

  const max = SCREENER_NORMALIZER;
  const screenerRaw = h.screener_score != null ? h.screener_score / max : null;

  const SIGNAL_VALUES = { buy: 1.0, consider: 0.75, hold: 0.5, avoid: 0.1 };
  const CONV_MULT    = { high: 1.1, medium: 1.0, low: 0.9 };
  let signalRaw = h.earnings_signal ? (SIGNAL_VALUES[h.earnings_signal] ?? null) : null;
  if (signalRaw != null && h.earnings_conviction) {
    signalRaw = Math.min(1, signalRaw * (CONV_MULT[h.earnings_conviction] ?? 1));
  }
  // Decay signal weight by age: full weight < 12 months, 60% for 12–24 months, zero after 24 months
  if (signalRaw != null && h.earnings_announcement_date) {
    const monthsOld = earningsReportAgeMonths(h.earnings_announcement_date);
    const { freshness } = earningsAgeDecay(monthsOld);
    signalRaw = signalRaw * freshness;
  }

  // recommendation_mean: 1=strong buy … 5=strong sell → normalise to [0, 1]
  // Validate range before using: some tickers return values outside [1, 5]
  const recRaw = (h.recommendation_mean != null &&
                  h.recommendation_mean >= 1 &&
                  h.recommendation_mean <= 5)
    ? (5 - h.recommendation_mean) / 4
    : null;

  // form13f_score is roughly –2 … +2; normalise to [0, 1]
  const f13fRaw = h.form13f_score != null
    ? Math.max(0, Math.min(1, (h.form13f_score + 2) / 4))
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
  const hasSignal   = !!h.earnings_signal;
  const hasRec      = h.recommendation_mean != null &&
                      h.recommendation_mean >= 1 &&
                      h.recommendation_mean <= 5;
  const hasF13f     = h.form13f_score != null;
  const presentTotal =
    (hasScreener ? 50 : 0) +
    (hasSignal   ? 25 : 0) +
    (hasRec      ? 10 : 0) +
    (hasF13f     ? 15 : 0);
  const eff = (base) =>
    presentTotal > 0 ? Math.round(base / presentTotal * 100) : 0;

  const screenerLine = hasScreener
    ? `Screener: ${h.screener_score} pts  (eff. ${eff(50)}%)`
    : 'Screener: no data';
  const signalLine = hasSignal
    ? `Signal: ${h.earnings_signal}${h.earnings_conviction ? ` (${h.earnings_conviction})` : ''}  (eff. ${eff(25)}%)`
    : 'Signal: no earnings report analysed yet';
  const recLine = hasRec
    ? `Analyst rec: ${h.recommendation_mean?.toFixed(1)}/5  (eff. ${eff(10)}%)`
    : 'Analyst rec: no data';
  const f13fLine = hasF13f
    ? `13F: score ${h.form13f_score?.toFixed(1)}  (eff. ${eff(15)}%)`
    : '13F: no data (non-US or not filed)';

  return `Score: ${score.toFixed(1)} / 10\n${screenerLine}\n${signalLine}\n${recLine}\n${f13fLine}`;
}
