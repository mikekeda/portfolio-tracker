// Sector-aware P/B thresholds [greenMax, redMin]
// null = don't color-code (metric unreliable for the sector)
export const PB_THRESHOLDS = {
  'Financial Services':     [1.2,  2.5],  // banks: book value is meaningful
  'Real Estate':            [1.0,  2.0],  // REITs trade near NAV
  'Utilities':              [1.5,  3.0],
  'Industrials':            [2.0,  5.0],
  'Consumer Defensive':     [2.0,  5.0],
  'Consumer Cyclical':      [2.0,  6.0],
  'Energy':                 [1.5,  4.0],
  'Basic Materials':        [1.5,  4.0],
  'Technology':             null,          // intangibles dominate — book value is misleading
  'Healthcare':             null,
  'Communication Services': null,
};

// Sector-aware P/S thresholds [greenMax, redMin]
// null = don't color-code (revenue definition varies or ratio is not standard)
export const PS_THRESHOLDS = {
  'Technology':             [5.0, 15.0],  // SaaS/software: higher multiples normal
  'Communication Services': [2.0,  8.0],
  'Healthcare':             [2.0,  8.0],  // biotech can trade at high P/S pre-profit
  'Consumer Cyclical':      [1.0,  3.0],
  'Consumer Defensive':     [0.8,  2.0],  // low-margin → low P/S expected
  'Industrials':            [1.0,  3.0],
  'Basic Materials':        [1.0,  2.5],
  'Energy':                 [0.8,  2.0],
  'Utilities':              [1.5,  3.0],
  'Financial Services':     null,          // revenue = interest income, not comparable
  'Real Estate':            null,          // use FFO multiples instead
};

/**
 * Returns the P/B threshold pair for a sector, or a generic fallback.
 * Returns null if the metric should not be color-coded for that sector.
 */
export const getPbThresholds = (sector) =>
  Object.prototype.hasOwnProperty.call(PB_THRESHOLDS, sector)
    ? PB_THRESHOLDS[sector]
    : [3.0, 8.0];

/**
 * Returns the P/S threshold pair for a sector, or a generic fallback.
 * Returns null if the metric should not be color-coded for that sector.
 */
export const getPsThresholds = (sector) =>
  Object.prototype.hasOwnProperty.call(PS_THRESHOLDS, sector)
    ? PS_THRESHOLDS[sector]
    : [1.5, 4.0];
