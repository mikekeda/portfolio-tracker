import {
  SCREENER_NORMALIZER,
  compositeTooltip,
  computeComposite,
  form13fScore,
  earningsAgeDecay,
  earningsReportAgeMonths,
  effectiveSignalDate,
  screenerRatio,
} from './compositeScore';

// NVDA's real figures the day the composite was reworked, so a future rescale
// fails loudly rather than drifting.
const NVDA = {
  quote_type: 'EQUITY',
  sector: 'Technology',
  screener_score: 65,
  screener_score_max: 60,
  earnings_signal: 'buy',
  earnings_conviction: 'high',
  earnings_announcement_date: new Date().toISOString(),
  recommendation_mean: 1.3,
  form13f_score: 1.2,
};

// VUAG.L as the writer stores it: the screener pair is a transport encoding of
// the constituent-weighted ratio 0.303, not points the fund earned.
const VUAG = {
  quote_type: 'ETF',
  sector: null,
  is_fund: true,
  look_through: true,
  look_through_n: 489,
  look_through_as_of: '2026-08-15',
  screener_score: 0.303 * SCREENER_NORMALIZER,
  screener_score_max: SCREENER_NORMALIZER,
  recommendation_mean: 2.1,
  look_through_form13f: 0.8,
};

const monthsAgo = (n) => new Date(Date.now() - n * 30.44 * 864e5).toISOString();

describe('screenerRatio', () => {
  test('divides by the sector max', () => {
    expect(screenerRatio(30, 60)).toBe(0.5);
    expect(screenerRatio(10, 23.2)).toBeCloseTo(0.431, 3);
  });

  test('falls back to the shared normalizer, which mirrors the backend', () => {
    expect(SCREENER_NORMALIZER).toBe(60);
    expect(screenerRatio(30, null)).toBe(0.5);
  });

  test('returns null when there is no score', () => {
    expect(screenerRatio(null, 60)).toBeNull();
  });
});

describe('screener component', () => {
  test('is clamped above 1 so it cannot outweigh its 50%', () => {
    // 65/60 = 1.08; without the clamp this would exceed a perfect screener.
    const capped = computeComposite({ ...NVDA, screener_score: 65 });
    const perfect = computeComposite({ ...NVDA, screener_score: 60 });
    expect(capped).toBe(perfect);
  });

  test('keeps the negative tail so red-flag stocks stay negative', () => {
    const score = computeComposite({
      sector: 'Technology',
      screener_score: -9,
      screener_score_max: 60,
    });
    expect(score).toBeLessThan(0);
  });

  test('a null score reweights out instead of scoring zero', () => {
    const withScore = computeComposite({ ...NVDA, screener_score: 0 });
    const withoutScore = computeComposite({ ...NVDA, screener_score: null });
    expect(withoutScore).toBeGreaterThan(withScore);
  });
});

describe('earnings signal', () => {
  test('buy and consider sit close together — forward returns do not separate them', () => {
    const buy = computeComposite({ ...NVDA, earnings_signal: 'buy', earnings_conviction: 'medium' });
    const consider = computeComposite({ ...NVDA, earnings_signal: 'consider', earnings_conviction: 'medium' });
    expect(buy - consider).toBeCloseTo(0.1, 1);
  });

  test('avoid is a real alarm, not a mild penalty', () => {
    const buy = computeComposite({ ...NVDA, earnings_signal: 'buy', earnings_conviction: 'medium' });
    const avoid = computeComposite({ ...NVDA, earnings_signal: 'avoid', earnings_conviction: 'medium' });
    expect(buy - avoid).toBeGreaterThan(2);
  });

  test('conviction moves the score in BOTH directions, including for buy', () => {
    const high = computeComposite({ ...NVDA, earnings_conviction: 'high' });
    const medium = computeComposite({ ...NVDA, earnings_conviction: 'medium' });
    const low = computeComposite({ ...NVDA, earnings_conviction: 'low' });
    expect(high).toBeGreaterThan(medium);
    expect(medium).toBeGreaterThan(low);
  });

  test('high-conviction consider outranks low-conviction buy (accepted inversion)', () => {
    const considerHigh = computeComposite({ ...NVDA, earnings_signal: 'consider', earnings_conviction: 'high' });
    const buyLow = computeComposite({ ...NVDA, earnings_signal: 'buy', earnings_conviction: 'low' });
    expect(considerHigh).toBeGreaterThan(buyLow);
  });

  test('an unrecognised signal reweights out rather than scoring zero', () => {
    expect(computeComposite({ ...NVDA, earnings_signal: 'nonsense' }))
      .toBe(computeComposite({ ...NVDA, earnings_signal: null }));
  });
});

describe('signal age', () => {
  test('decays from the report date when no announcement date was matched', () => {
    const base = { ...NVDA, earnings_announcement_date: null };
    const fresh = computeComposite({ ...base, earnings_report_date: monthsAgo(1) });
    const stale = computeComposite({ ...base, earnings_report_date: monthsAgo(18) });
    expect(stale).toBeLessThan(fresh);
  });

  test('drops the component entirely past 24 months', () => {
    const base = { ...NVDA, earnings_announcement_date: null };
    const ancient = computeComposite({ ...base, earnings_report_date: monthsAgo(30) });
    const noSignal = computeComposite({ ...base, earnings_signal: null });
    expect(ancient).toBe(noSignal);
  });

  test('the announcement date wins when both are present', () => {
    expect(effectiveSignalDate({ earnings_announcement_date: 'A', earnings_report_date: 'B' })).toBe('A');
    expect(effectiveSignalDate({ earnings_announcement_date: null, earnings_report_date: 'B' })).toBe('B');
    expect(effectiveSignalDate({})).toBeNull();
  });

  test('tooltip ages from the same date the score does', () => {
    const h = { ...NVDA, earnings_announcement_date: null, earnings_report_date: monthsAgo(18) };
    // Score decays, so the tooltip must not claim full weight.
    expect(compositeTooltip(computeComposite(h), h)).toContain('reduced weight');
  });

  test('invalid or missing dates do not decay', () => {
    expect(earningsReportAgeMonths(null)).toBeNull();
    expect(earningsReportAgeMonths('not-a-date')).toBeNull();
    expect(earningsAgeDecay(null).freshness).toBe(1);
  });
});

describe('reweighting', () => {
  test('missing components redistribute rather than counting as zero', () => {
    const full = computeComposite(NVDA);
    const noF13f = computeComposite({ ...NVDA, form13f_score: null });
    // 13F at 1.2 is a weaker component than the rest, so dropping it lifts the score.
    expect(noF13f).toBeGreaterThan(full);
  });

  test('an out-of-range analyst mean is ignored, not clamped', () => {
    expect(computeComposite({ ...NVDA, recommendation_mean: 9 }))
      .toBe(computeComposite({ ...NVDA, recommendation_mean: null }));
  });

  test('returns null when nothing is scoreable', () => {
    expect(computeComposite({ sector: 'Technology' })).toBeNull();
  });
});

describe('exclusions', () => {
  test('funds are not scored, on the backend verdict rather than a missing sector', () => {
    expect(computeComposite({ ...NVDA, quote_type: 'ETF', is_fund: true })).toBeNull();
    // SGLN.L is quoteType EQUITY with no sector; the flag is what catches it.
    expect(computeComposite({ ...NVDA, sector: null, is_fund: true })).toBeNull();
  });

  test('a degraded Yahoo profile still scores', () => {
    // FISV and Leonardo's German listing lose sector without being funds; the
    // old sector-null guard hid their score.
    expect(computeComposite({ ...NVDA, sector: null, is_fund: false })).not.toBeNull();
  });

  test('a caller that omits sector entirely is reported, not silently blanked', () => {
    // The Stock page lost its Score tile this way; a null sector is legitimate,
    // an absent key is a bug, and the two must not look alike.
    const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const missingSector = { ...NVDA };
    delete missingSector.sector;
    expect(computeComposite(missingSector)).toBeNull();
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});

describe('ETF look-through', () => {
  test('a fund with look-through metrics scores', () => {
    expect(computeComposite(VUAG)).not.toBeNull();
  });

  test('the same fund without the flag is still blank', () => {
    // The flag, not the presence of a score, is what admits a fund — gold ETCs
    // and funds below the coverage gate never get one.
    const { look_through, ...noFlag } = VUAG;
    expect(computeComposite(noFlag)).toBeNull();
  });

  test('gold ETCs stay null even though they are quoteType EQUITY', () => {
    expect(computeComposite({ quote_type: 'EQUITY', sector: null, is_fund: true, screener_score: null })).toBeNull();
    expect(computeComposite({ ...VUAG, look_through: undefined, quote_type: 'EQUITY' })).toBeNull();
  });

  test('the 13F leg comes from look_through_form13f, never form13f_score', () => {
    expect(form13fScore(VUAG)).toBe(0.8);
    // A stray form13f_score on a fund row must not be picked up: it would mean
    // "managers hold this fund" rather than "managers hold its constituents".
    expect(form13fScore({ ...VUAG, form13f_score: -1.5 })).toBe(0.8);
    expect(form13fScore({ quote_type: 'EQUITY', sector: 'Technology', form13f_score: 1.2 })).toBe(1.2);
  });

  test('the 13F leg counts toward the reweighting', () => {
    // Without it the fund silently reweights to 83% screener / 17% analyst.
    const { look_through_form13f, ...noF13f } = VUAG;
    expect(compositeTooltip(5, VUAG)).toContain('13F: score 0.8');
    expect(compositeTooltip(5, VUAG)).toContain('(eff. 20%)');
    expect(compositeTooltip(5, noF13f)).toContain('13F: no data');
    expect(computeComposite(VUAG)).not.toBe(computeComposite(noF13f));
  });

  test('the tooltip shows points on the same scale as a stock, labelled as an average', () => {
    // Same units as every other averaged column, but it must never read as
    // gates the fund passed — the Screeners badge column stays empty for funds.
    const tip = compositeTooltip(4.4, VUAG);
    expect(tip).toContain('Screener: 18 / 60 pts, constituent avg');
    expect(tip).toContain('489 constituents');
  });

  test('funds score on three legs — earnings is never aggregated', () => {
    const tip = compositeTooltip(4.4, VUAG);
    expect(tip).toContain('Signal: no earnings report analysed yet');
    // 50/10/15 of a 75 present total.
    expect(tip).toContain('constituent avg  (eff. 67%)');
  });
});

describe('Stock and Holdings agree', () => {
  // The helper takes an untyped bag of fields, so a page that forgets one fails
  // silently — this is what let the Stock page lose its Score tile entirely.
  test('the same instrument scores identically from either page shape', () => {
    const holdingsRow = {
      quote_type: 'EQUITY',
      sector: 'Technology',
      screener_score: 65,
      screener_score_max: 60,
      earnings_signal: 'buy',
      earnings_conviction: 'high',
      earnings_announcement_date: NVDA.earnings_announcement_date,
      earnings_report_date: NVDA.earnings_announcement_date,
      recommendation_mean: 1.3,
      form13f_score: 1.2,
      // Holdings rows carry many extra keys the helper must ignore.
      ppl: 1234,
      market_value: 5678,
    };
    const stockInput = {
      quote_type: 'EQUITY',
      sector: 'Technology',
      screener_score: 65,
      screener_score_max: 60,
      earnings_signal: 'buy',
      earnings_conviction: 'high',
      earnings_announcement_date: NVDA.earnings_announcement_date,
      earnings_report_date: NVDA.earnings_announcement_date,
      recommendation_mean: 1.3,
      form13f_score: 1.2,
    };
    expect(computeComposite(stockInput)).toBe(computeComposite(holdingsRow));
    expect(computeComposite(stockInput)).not.toBeNull();
  });

  test('a sector-scaled max flows through for excluded sectors', () => {
    // LLOY.L: 5 points against a Financial Services max of 23.2 is not the same
    // as 5 against 60 — the ratio is what makes the two comparable.
    const bank = { ...NVDA, sector: 'Financial Services', screener_score: 5, screener_score_max: 23.2 };
    const tech = { ...NVDA, screener_score: 5, screener_score_max: 60 };
    expect(computeComposite(bank)).toBeGreaterThan(computeComposite(tech));
  });
});

describe('tooltip', () => {
  test('reports no data rather than printing a null score', () => {
    const tip = compositeTooltip(5, { ...NVDA, screener_score: null });
    expect(tip).toContain('Screener: no data');
    expect(tip).not.toContain('null');
  });

  test('shows the score against its sector max', () => {
    expect(compositeTooltip(9.1, NVDA)).toContain('Screener: 65 / 60 pts');
  });
});

describe('known values', () => {
  test('NVDA scores as expected end to end', () => {
    // screener 1.0×0.50 | signal 0.9975×0.25 | rec 0.925×0.10 | 13F 0.8×0.15
    expect(computeComposite(NVDA)).toBe(9.6);
  });
});
