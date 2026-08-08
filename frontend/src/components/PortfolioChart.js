import React, { useState, useEffect, useCallback, useMemo } from 'react';
import PropTypes from 'prop-types';
import { useNavigate } from 'react-router-dom';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  Label,
  ReferenceLine,
} from 'recharts';
import { portfolioAPI } from '../services/api';
import { ytdDays } from '../utils/dates';
import { useHideAmounts, MASK } from '../context/HideAmountsContext';
import './PortfolioChart.css';
import SharedTooltip from './SharedTooltip';
import UnderwaterChart from './UnderwaterChart';
import AllocationTreemap from './AllocationTreemap';

// Shared date format used by the history data and any overlays (deposit
// markers) so <ReferenceLine x=...> snaps to the same XAxis category.
const formatChartDate = (iso) =>
  new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: '2-digit',
  });

// Deposit markers want compact labels. Sub-£1k shows exact pounds (so small
// top-ups don't collapse to "£0k"); above that, round to the nearest £1k.
const formatDepositAmount = (amount) => {
  if (amount < 1000) return `£${Math.round(amount)}`;
  return `£${Math.round(amount / 1000)}k`;
};

// Custom tooltip for combined Total Value + Profit/Loss chart
const CombinedTooltip = ({ active, payload, label, hideAmounts }) => {
  if (active && payload && payload.length) {
    const getVal = (key) => {
      const entry = payload.find((p) => p.dataKey === key);
      const v = entry ? Number(entry.value) : 0;
      return isNaN(v) ? 0 : v;
    };
    const invested = getVal('invested');
    const profit = getVal('profitPos') + getVal('lossNeg'); // lossNeg is negative

    const fmt = (n) => (hideAmounts ? MASK : `£${Number(n).toLocaleString()}`);

    return (
      <div className="custom-tooltip">
        <p className="tooltip-label">{label}</p>
        <p className="tooltip-item">Total: {fmt(invested + profit)}</p>
        <p className={`tooltip-item ${profit >= 0 ? 'positive' : 'negative'}`}>
          Profit: {fmt(profit)}
        </p>
        <p className="tooltip-item tooltip-item-base">Base Cost: {fmt(invested)}</p>
      </div>
    );
  }
  return null;
};

CombinedTooltip.propTypes = {
  active: PropTypes.bool,
  payload: PropTypes.arrayOf(
    PropTypes.shape({
      dataKey: PropTypes.string,
      value: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
    })
  ),
  label: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  hideAmounts: PropTypes.bool,
};

CombinedTooltip.defaultProps = {
  active: false,
  payload: [],
  label: '',
  hideAmounts: false,
};

// Finviz colour scheme: 3 cutoffs per side -> 4 zones per side, scaled per
// period so the shading stays meaningful at every horizon. A "strong" 1d
// move (>= 3%) is roughly comparable in significance to a "strong" 90d move
// (>= 20%); the band widths grow with sqrt-of-time.
const BAND_THRESHOLDS = {
  '1d':  { low: 1,   mid: 2,    strong: 3 },
  '1w':  { low: 2,   mid: 4,    strong: 6 },
  '1m':  { low: 3.3, mid: 6.7,  strong: 10 },
  '90d': { low: 6.7, mid: 13.3, strong: 20 },
  // Anchored to a mid-year YTD (~180 days); early January runs narrow.
  'ytd': { low: 10,  mid: 20,   strong: 30 },
};

const PERIOD_LABELS = {
  '1d': '1 day',
  '1w': '1 week',
  '1m': '1 month',
  '90d': '90 days',
  'ytd': 'year to date',
};

// Holdings smaller than this share of their bucket collapse into a single
// "Other" tile so the long tail doesn't fill the treemap with unreadable boxes.
const OTHER_THRESHOLD_PCT = 0.5;

const PortfolioChart = ({ selectedPeriod, movers, moversError }) => {
  const navigate = useNavigate();
  const { hideAmounts } = useHideAmounts();
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('30'); // days
  const [benchmarkNames, setBenchmarkNames] = useState(['S&P 500', 'NASDAQ']);
  const [deposits, setDeposits] = useState([]);
  const [groupBy, setGroupBy] = useState('flat'); // 'flat' | 'sector' | 'region'
  const [selectedGroup, setSelectedGroup] = useState(null); // active group filter
  const [expandOther, setExpandOther] = useState(false);

  const timeRanges = [
    { label: '1 Month', value: '30' },
    { label: '3 Months', value: '90' },
    { label: '6 Months', value: '180' },
    { label: 'YTD', value: 'ytd' },
    { label: '1 Year', value: '365' },
    { label: 'All', value: 'all' }
  ];


  const fetchChartData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const days = timeRange === 'all' ? 365 : timeRange === 'ytd' ? ytdDays() : parseInt(timeRange);
      const [historyData, depositsData] = await Promise.all([
        portfolioAPI.getHistory(days),
        portfolioAPI.getDeposits(days),
      ]);

      let newBenchmarkNames = ['S&P 500', 'NASDAQ'];
      if (historyData.benchmark && Array.isArray(historyData.benchmark)) {
        const nameMap = {
          'VUAG.L': 'S&P 500',
          'XNAS.L': 'NASDAQ'
        };
        newBenchmarkNames = historyData.benchmark.map(symbol => nameMap[symbol] || symbol);
      }

      let newChartData = null;
      if (historyData.history && historyData.history.length > 0) {
        const processedData = historyData.history.map(item => {
          const processedItem = {
            date: formatChartDate(item.date),
            fullDate: item.date,
            totalValue: item.total_value || 0,
            totalProfit: item.total_profit || 0,
            totalReturn: item.total_return_pct || 0,
            spyReturn: item.benchmark_return_pct[0],
            nasdaqReturn: item.benchmark_return_pct[1],
          };

          const invested = (processedItem.totalValue - processedItem.totalProfit) || 0;
          const profitPos = Math.max(processedItem.totalProfit, 0);
          const lossNeg = Math.min(processedItem.totalProfit, 0);

          processedItem.invested = invested;
          processedItem.profitPos = profitPos;
          processedItem.lossNeg = lossNeg;

          return processedItem;
        });

        newChartData = processedData.sort((a, b) => new Date(a.fullDate) - new Date(b.fullDate));
      }

      // Deposit markers — date labels must match the XAxis category strings
      // produced above so <ReferenceLine x=...> lines up on a tick.
      const newDeposits = Array.isArray(depositsData?.deposits)
        ? depositsData.deposits.map((d) => ({
            date: d.date,
            dateLabel: formatChartDate(d.date),
            amount: Number(d.amount) || 0,
          }))
        : [];

      setBenchmarkNames(newBenchmarkNames);
      setDeposits(newDeposits);
      if (newChartData) {
        setChartData(newChartData);
      }
    } catch (err) {
      setError('Failed to fetch chart data');
      console.error('Error fetching chart data:', err);
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    fetchChartData();
  }, [fetchChartData]);

  const allocationLoading = !movers && !moversError;

  // Allocation shares the movers payload Dashboard fetches for the Top Movers
  // table — the same rows, weighted by value instead of by change.
  const assetAllocation = useMemo(() => {
    if (!movers) return null;
    const totalValue = movers.reduce((sum, asset) => sum + (asset.value || 0), 0);
    return movers
      .filter(asset => asset.value > 0)
      .map(asset => ({
        symbol: asset.symbol || 'N/A',
        name: asset.name || '',
        value: asset.value || 0,
        allocation_pct: totalValue > 0 ? ((asset.value || 0) / totalValue) * 100 : 0,
        change_pct: asset.change_pct || 0,
        sector: asset.sector || 'Other',
        country: asset.country || 'Other',
      }))
      .sort((a, b) => b.value - a.value);
  }, [movers]);

  // Drop any drill-down filter / Other-expansion when the user switches
  // grouping or period, so the view doesn't reference stale group names.
  useEffect(() => {
    setSelectedGroup(null);
    setExpandOther(false);
  }, [groupBy, selectedPeriod]);

  // One-line "32 up / 11 down / weighted +0.47%" header. Computed off the raw
  // (unfiltered) allocation list so it always describes the whole portfolio,
  // not just whatever group is currently filtered into the treemap.
  const summary = useMemo(() => {
    if (!assetAllocation || assetAllocation.length === 0) return null;
    let up = 0;
    let down = 0;
    let totalValue = 0;
    let weightedSum = 0;
    assetAllocation.forEach((a) => {
      if (a.change_pct > 0) up += 1;
      else if (a.change_pct < 0) down += 1;
      totalValue += a.value;
      weightedSum += a.change_pct * a.value;
    });
    const weighted = totalValue > 0 ? weightedSum / totalValue : 0;
    return { up, down, weighted };
  }, [assetAllocation]);

  // Per-group totals for the chip row (sector / region view). Each chip shows
  // the group's share of the portfolio and its value-weighted change so the
  // user can spot where the day's move is coming from before drilling in.
  const groupTotals = useMemo(() => {
    if (!assetAllocation || groupBy === 'flat') return [];
    const key = groupBy === 'sector' ? 'sector' : 'country';
    const buckets = new Map();
    assetAllocation.forEach((a) => {
      const name = a[key] || 'Other';
      const existing = buckets.get(name) || { name, value: 0, weightedSum: 0, count: 0 };
      existing.value += a.value;
      existing.weightedSum += a.change_pct * a.value;
      existing.count += 1;
      buckets.set(name, existing);
    });
    const totalValue = assetAllocation.reduce((sum, a) => sum + a.value, 0);
    return Array.from(buckets.values())
      .map((b) => ({
        name: b.name,
        value: b.value,
        count: b.count,
        allocation_pct: totalValue > 0 ? (b.value / totalValue) * 100 : 0,
        change_pct: b.value > 0 ? b.weightedSum / b.value : 0,
      }))
      .sort((a, b) => b.value - a.value);
  }, [assetAllocation, groupBy]);

  const bands = BAND_THRESHOLDS[selectedPeriod] ?? BAND_THRESHOLDS['1d'];
  const periodLabel = PERIOD_LABELS[selectedPeriod] ?? selectedPeriod;

  return (
    <div className="portfolio-chart-container">
      <div className="chart-header">
        <h2>Portfolio Summary</h2>
        <div className="time-range-selector">
          {timeRanges.map(range => (
            <button
              key={range.value}
              className={`time-range-btn ${timeRange === range.value ? 'active' : ''}`}
              onClick={() => setTimeRange(range.value)}
              disabled={loading}
            >
              {range.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* Performance Metrics */}
      <div className={`performance-charts ${loading ? 'loading' : ''}`}>
        {loading && (
          <div className="loading-overlay">
            <div className="loading">Loading chart data...</div>
          </div>
        )}
        <div className="chart-panel">
          <h3>Total Value (£) + Profit/Loss</h3>
          <ResponsiveContainer width="100%" height={220} key={`area-${timeRange}`}>
            <AreaChart data={chartData || []} isAnimationActive={false}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis tickFormatter={(value) => (hideAmounts ? MASK : `${(value / 1000).toFixed(0)}k`)} domain={['auto', 'auto']} />
              <Tooltip content={<CombinedTooltip hideAmounts={hideAmounts} />} />
              {/* Base invested cost */}
              <Area type="monotone" dataKey="invested" name="Base Cost" stackId="1" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} isAnimationActive={false} />
              {/* Profit overlay (positive values) */}
              <Area type="monotone" dataKey="profitPos" name="Profit" stackId="1" stroke="#28a745" fill="#28a745" fillOpacity={0.4} isAnimationActive={false} />
              {/* Loss overlay (negative values) - rendered in red */}
              <Area type="monotone" dataKey="lossNeg" name="Loss" stackId="1" stroke="#dc3545" fill="#dc3545" fillOpacity={0.35} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel">
          <h3>Total Return (%)</h3>
          <ResponsiveContainer width="100%" height={220} key={`line-${timeRange}`}>
            <LineChart data={chartData || []} isAnimationActive={false}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis tickFormatter={(v) => `${v}%`} domain={['auto', 'auto']} />
              <Tooltip
                content={<SharedTooltip
                  valueFormatter={(v) => `${Number(v).toFixed(2)}%`}
                  nameMap={{
                    totalReturn: 'Portfolio',
                    spyReturn: benchmarkNames[0],
                    nasdaqReturn: benchmarkNames[1]
                  }}
                  sortOrder={['totalReturn', 'spyReturn', 'nasdaqReturn']}
                />}
              />
              <ReferenceLine y={0} stroke="#6c757d" strokeDasharray="3 3" />
              {deposits.map((d) => (
                <ReferenceLine
                  key={`dep-${d.date}`}
                  x={d.dateLabel}
                  stroke="#16a34a"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                  opacity={0.65}
                  ifOverflow="hidden"
                >
                  <Label
                    value={hideAmounts ? MASK : formatDepositAmount(d.amount)}
                    position="insideBottomLeft"
                    fontSize={10}
                    fill="#16a34a"
                  />
                </ReferenceLine>
              ))}
              <Line
                type="monotone"
                dataKey="totalReturn"
                name="Portfolio"
                stroke="#ffc658"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="spyReturn"
                name={benchmarkNames[0]}
                stroke="#2563eb"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="nasdaqReturn"
                name={benchmarkNames[1]}
                stroke="#dc2626"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <UnderwaterChart timeRange={timeRange} />

      {/* Asset Allocation */}
      <div className={`asset-allocation-section ${allocationLoading ? 'loading' : ''}`}>
        <div className="asset-allocation-header">
          <h3>Asset Allocation</h3>
          <p className="asset-allocation-caption">
            Change over {periodLabel}
            {summary && (
              <>
                {' \u00b7 '}
                <span className="positive">{summary.up} up</span>
                {' \u00b7 '}
                <span className="negative">{summary.down} down</span>
                {' \u00b7 '}
                <span className={summary.weighted >= 0 ? 'positive' : 'negative'}>
                  weighted {summary.weighted >= 0 ? '+' : ''}{summary.weighted.toFixed(2)}%
                </span>
              </>
            )}
          </p>
        </div>

        <div className="asset-allocation-controls">
          <div className="group-by-selector">
            <span className="group-by-label">Group by:</span>
            {[
              { id: 'flat',   label: 'Flat' },
              { id: 'sector', label: 'Sector' },
              { id: 'region', label: 'Region' },
            ].map((opt) => (
              <button
                key={opt.id}
                type="button"
                className={`group-by-btn ${groupBy === opt.id ? 'active' : ''}`}
                onClick={() => setGroupBy(opt.id)}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {groupBy !== 'flat' && groupTotals.length > 0 && (
          <div className="group-chips">
            <button
              type="button"
              className={`group-chip ${selectedGroup === null ? 'active' : ''}`}
              onClick={() => setSelectedGroup(null)}
            >
              All
            </button>
            {groupTotals.map((g) => {
              const isPositive = g.change_pct >= 0;
              return (
                <button
                  key={g.name}
                  type="button"
                  className={`group-chip ${selectedGroup === g.name ? 'active' : ''}`}
                  onClick={() => setSelectedGroup(selectedGroup === g.name ? null : g.name)}
                  title={`${g.count} holdings · ${g.allocation_pct.toFixed(1)}% of portfolio`}
                >
                  <span className="group-chip-name">{g.name}</span>
                  <span className="group-chip-alloc">{g.allocation_pct.toFixed(1)}%</span>
                  <span className={`group-chip-change ${isPositive ? 'positive' : 'negative'}`}>
                    {isPositive ? '+' : ''}{g.change_pct.toFixed(2)}%
                  </span>
                </button>
              );
            })}
          </div>
        )}

        {expandOther && (
          <div className="other-expanded-banner">
            Showing all small holdings &middot;{' '}
            <button type="button" className="link-button" onClick={() => setExpandOther(false)}>
              collapse Other
            </button>
          </div>
        )}

        {allocationLoading && (
          <div className="loading-overlay loading-overlay-tall">
            <div className="loading">Loading asset allocation...</div>
          </div>
        )}
        {assetAllocation && assetAllocation.length > 0 ? (
          <AllocationTreemap
            data={assetAllocation.map((a) => ({
              symbol: a.symbol,
              name: a.symbol,
              fullName: a.name,
              value: a.value,
              change_pct: a.change_pct,
              sector: a.sector,
              country: a.country,
            }))}
            groupBy={groupBy}
            selectedGroup={selectedGroup}
            onSelectGroup={(name) => setSelectedGroup(selectedGroup === name ? null : name)}
            expandOther={expandOther}
            onOtherClick={() => setExpandOther(true)}
            bands={bands}
            otherThreshold={groupBy === 'flat' ? 0 : OTHER_THRESHOLD_PCT}
            height={600}
            navigate={navigate}
          />
        ) : (
          <div className="error">
            {moversError || 'No asset allocation data available'}
          </div>
        )}

        <div className="treemap-legend-container">
          <div className="treemap-legend">
            <div className="treemap-legend-swatch legend-color-brightest-loss">-{bands.strong}%</div>
            <div className="treemap-legend-swatch legend-color-strong-loss">-{bands.mid}%</div>
            <div className="treemap-legend-swatch legend-color-medium-loss">-{bands.low}%</div>
            <div className="treemap-legend-swatch legend-color-neutral">0%</div>
            <div className="treemap-legend-swatch legend-color-medium-gain">+{bands.low}%</div>
            <div className="treemap-legend-swatch legend-color-strong-gain">+{bands.mid}%</div>
            <div className="treemap-legend-swatch legend-color-brightest-gain">+{bands.strong}%</div>
          </div>
        </div>
      </div>

    </div>
  );
};

PortfolioChart.propTypes = {
  selectedPeriod: PropTypes.string,
  movers: PropTypes.arrayOf(PropTypes.object),
  moversError: PropTypes.string,
};

PortfolioChart.defaultProps = {
  selectedPeriod: '1d',
  movers: null,
  moversError: null,
};

export default PortfolioChart;
