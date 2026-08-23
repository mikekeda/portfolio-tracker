import {
  convertRateForMode,
  fractionAtOrAbove,
  futureValue,
  parsePctInput,
  runSimulation,
  solveRequiredCagr,
  solveRequiredContribution,
  toNominal,
  toReal,
} from './projectionMath';

const SEED = 0x5eed;

describe('futureValue', () => {
  test('compounds the starting pot with no contributions', () => {
    expect(futureValue(100000, 0, 0.065, 0, 10)).toBeCloseTo(100000 * 1.065 ** 10, 6);
  });

  test('pays contributions at the start of the year', () => {
    // One year, one £20k lump: both the pot and the lump compound for the full year.
    expect(futureValue(100000, 20000, 0.10, 0, 1)).toBeCloseTo(132000, 6);
  });

  test('handles r === g without dividing by zero', () => {
    // q = 1 makes the closed-form annuity singular; the branch falls back to S terms.
    const fv = futureValue(0, 1000, 0.04, 0.04, 5);
    expect(Number.isFinite(fv)).toBe(true);
    expect(fv).toBeCloseTo(1.04 ** 5 * 5000, 6);
  });

  test('stops contributing after S years', () => {
    const capped = futureValue(0, 10000, 0.05, 0, 20, 10);
    const uncapped = futureValue(0, 10000, 0.05, 0, 20, 20);
    expect(capped).toBeLessThan(uncapped);
    // The first 10 lumps still compound for the full 20 years.
    expect(capped).toBeCloseTo(1.05 ** 10 * futureValue(0, 10000, 0.05, 0, 10), 6);
  });

  test('a frozen contribution decays in real terms', () => {
    const frozen = futureValue(0, 20000, 0.065, -0.036, 21);
    const uprated = futureValue(0, 20000, 0.065, 0, 21);
    expect(frozen).toBeLessThan(uprated);
  });
});

describe('solveRequiredCagr', () => {
  test('inverts futureValue', () => {
    const target = futureValue(100000, 20000, 0.0723, 0, 15);
    expect(solveRequiredCagr(100000, 20000, 0, 15, target)).toBeCloseTo(0.0723, 6);
  });

  test('returns null when the target is out of the bisection range', () => {
    expect(solveRequiredCagr(100000, 20000, 0, 15, 1e15)).toBeNull();
    expect(solveRequiredCagr(100000, 20000, 0, 15, -1)).toBeNull();
  });
});

describe('solveRequiredContribution', () => {
  test('inverts futureValue on the contribution', () => {
    const target = futureValue(100000, 17500, 0.065, 0, 20);
    expect(solveRequiredContribution(100000, 0.065, 0, 20, target)).toBeCloseTo(17500, 4);
  });

  test('is zero when the starting pot already gets there', () => {
    expect(solveRequiredContribution(1000000, 0.065, 0, 20, 500000)).toBe(0);
  });
});

describe('fractionAtOrAbove', () => {
  test('counts the tail at or above the target', () => {
    const sorted = Float64Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(fractionAtOrAbove(sorted, 0)).toBeCloseTo(1.0, 9);
    expect(fractionAtOrAbove(sorted, 5)).toBeCloseTo(0.5, 9);
    expect(fractionAtOrAbove(sorted, 10)).toBeCloseTo(0.0, 9);
  });

  test('is inclusive at the boundary, so ruin at exactly zero is not solvent', () => {
    const sorted = Float64Array.from([0, 0, 0, 100, 200]);
    expect(fractionAtOrAbove(sorted, 1)).toBeCloseTo(0.4, 9);
  });
});

describe('Fisher conversion', () => {
  test('round-trips', () => {
    expect(toReal(toNominal(0.065, 0.036), 0.036)).toBeCloseTo(0.065, 12);
  });

  test('is multiplicative, not additive', () => {
    // The bug this guards: 8.5% nominal at 3.6% CPI is 4.73% real, not 4.9%.
    expect(toReal(0.085, 0.036)).toBeCloseTo(0.0473, 4);
    expect(toNominal(0.065, 0.036)).toBeCloseTo(0.10334, 5);
  });
});

describe('convertRateForMode', () => {
  test('restates a custom rate so the toggle preserves the forecast', () => {
    // 6.5% real is 10.3% nominal at 3.6% CPI — the same forecast, not 6.5% nominal.
    expect(convertRateForMode('6.5', 'nominal', 0.036)).toBe('10.3');
    expect(convertRateForMode('10.3', 'real', 0.036)).toBe('6.5');
  });

  test('round-trips through both modes', () => {
    const there = convertRateForMode('7.0', 'nominal', 0.036);
    expect(convertRateForMode(there, 'real', 0.036)).toBe('7.0');
  });

  test('leaves a half-typed box alone', () => {
    expect(convertRateForMode('', 'nominal', 0.036)).toBeNull();
    expect(convertRateForMode('  ', 'nominal', 0.036)).toBeNull();
    expect(convertRateForMode('-', 'nominal', 0.036)).toBeNull();
  });

  test('what the engine sees in nominal mode matches the benchmark treatment', () => {
    const real = 0.065;
    const custom = Number(convertRateForMode('6.5', 'nominal', 0.036)) / 100;
    expect(custom).toBeCloseTo(toNominal(real, 0.036), 3);
  });
});

describe('runSimulation', () => {
  const base = {
    startingValue: 100000,
    contribution: 0,
    contributionGrowth: 0,
    expectedReturn: 0.065,
    volatility: 0.16,
    horizonYears: 20,
    paths: 20000,
    seed: SEED,
  };

  test('the median path compounds at the input CAGR', () => {
    // Log-drift is ln(1+r) with no -σ²/2, so the median is the deterministic path.
    // This is the load-bearing calibration choice: a geometric input stays geometric.
    const sim = runSimulation(base);
    for (const t of [1, 5, 10, 20]) {
      expect(sim.percentiles.p50[t] / (100000 * 1.065 ** t)).toBeCloseTo(1.0, 1);
    }
  });

  test('percentiles are ordered at every step', () => {
    // Guards the numeric default of Float64Array#sort: Array#sort would order
    // these lexicographically and silently scramble every band.
    const sim = runSimulation(base);
    for (let t = 0; t <= 20; t += 1) {
      const { p10, p25, p50, p75, p90 } = sim.percentiles;
      expect(p10[t]).toBeLessThanOrEqual(p25[t]);
      expect(p25[t]).toBeLessThanOrEqual(p50[t]);
      expect(p50[t]).toBeLessThanOrEqual(p75[t]);
      expect(p75[t]).toBeLessThanOrEqual(p90[t]);
    }
  });

  test('the deterministic series is exactly the zero-volatility path', () => {
    const sim = runSimulation(base);
    expect(sim.deterministic[20]).toBeCloseTo(100000 * 1.065 ** 20, 6);
  });

  test('is reproducible for a given seed and diverges for another', () => {
    expect(runSimulation(base).percentiles.p90[20])
      .toBe(runSimulation(base).percentiles.p90[20]);
    expect(runSimulation({ ...base, seed: 12345 }).percentiles.p90[20])
      .not.toBe(runSimulation(base).percentiles.p90[20]);
  });

  test('zero volatility collapses the fan onto the CAGR path', () => {
    const sim = runSimulation({ ...base, volatility: 0 });
    expect(sim.percentiles.p10[20]).toBeCloseTo(sim.percentiles.p90[20], 6);
    expect(sim.percentiles.p50[20]).toBeCloseTo(100000 * 1.065 ** 20, 6);
  });

  test('tracks invested capital without returns', () => {
    const sim = runSimulation({ ...base, contribution: 20000, horizonYears: 5 });
    expect(sim.invested[5]).toBeCloseTo(100000 + 5 * 20000, 6);
  });

  test('stops contributing at contributionYears', () => {
    const sim = runSimulation({
      ...base, contribution: 20000, contributionYears: 3, horizonYears: 10,
    });
    expect(sim.invested[10]).toBeCloseTo(100000 + 3 * 20000, 6);
  });

  test('ruin is absorbing — a depleted pot never recovers', () => {
    // Withdrawing more than the pot holds guarantees ruin in year 1.
    const sim = runSimulation({
      ...base,
      withdrawalAnnual: 500000,
      withdrawalStartYear: 0,
      horizonYears: 5,
    });
    expect(sim.ruin[0]).toBe(0);
    expect(sim.ruin[1]).toBe(1);
    expect(sim.ruin[5]).toBe(1);
    expect(sim.percentiles.p90[5]).toBe(0);
  });

  test('ruin rises monotonically through the drawdown', () => {
    const sim = runSimulation({
      ...base,
      expectedReturn: 0.065,
      withdrawalAnnual: 12000,
      withdrawalStartYear: 5,
      horizonYears: 40,
    });
    for (let t = 1; t <= 40; t += 1) expect(sim.ruin[t]).toBeGreaterThanOrEqual(sim.ruin[t - 1]);
    expect(sim.ruin[5]).toBe(0);
  });

  test('the drawdown phase compounds at drawdownReturn, not expectedReturn', () => {
    const shared = { ...base, withdrawalStartYear: 10, horizonYears: 20, volatility: 0 };
    const flat = runSimulation({ ...shared, drawdownReturn: 0.065 });
    const derisked = runSimulation({ ...shared, drawdownReturn: 0.02 });
    expect(derisked.percentiles.p50[10]).toBeCloseTo(flat.percentiles.p50[10], 6);
    expect(derisked.percentiles.p50[20]).toBeLessThan(flat.percentiles.p50[20]);
  });

  test('withdrawals start the year after withdrawalStartYear', () => {
    const sim = runSimulation({
      ...base, volatility: 0, withdrawalAnnual: 10000, withdrawalStartYear: 3, horizonYears: 4,
    });
    // Year 3 is untouched; year 4 takes the first withdrawal before compounding.
    expect(sim.percentiles.p50[3]).toBeCloseTo(100000 * 1.065 ** 3, 4);
    expect(sim.percentiles.p50[4]).toBeCloseTo((100000 * 1.065 ** 3 - 10000) * 1.065, 4);
  });
});

describe('parsePctInput', () => {
  test('converts percent to decimal', () => {
    expect(parsePctInput('6.5', 0.04)).toBeCloseTo(0.065, 12);
  });

  test('falls back on empty or unparseable input mid-typing', () => {
    expect(parsePctInput('', 0.04)).toBe(0.04);
    expect(parsePctInput('   ', 0.04)).toBe(0.04);
    expect(parsePctInput('abc', 0.04)).toBe(0.04);
  });
});
