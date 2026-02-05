import React, { useState, useEffect, useCallback } from 'react';
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
  Treemap
} from 'recharts';
import { portfolioAPI } from '../services/api';
import { useHideAmounts, MASK } from '../context/HideAmountsContext';
import './PortfolioChart.css';
import SharedTooltip from './SharedTooltip';

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

// Treemap content renderer for Asset Allocation
const TreemapContent = (props) => {
  const { x, y, width, height, payload, navigate } = props;

  // Defensive checks for rendering
  if (
    typeof x !== 'number' ||
    typeof y !== 'number' ||
    typeof width !== 'number' ||
    typeof height !== 'number' ||
    width <= 0 ||
    height <= 0
  ) {
    return null;
  }

  // Fallback for data access
  const itemData = payload || props;
  const name = itemData.name || itemData.root?.name || 'N/A';

  // Only render leaf nodes (items with no children) or items with explicit values
  if (!itemData.change_pct && itemData.change_pct !== 0 && !itemData.value) {
    return null;
  }

  const changePct = itemData.change_pct || 0;
  const isPositive = changePct >= 0;

  let bgColor;
  // Trading212-like colors: Bright for significant moves, Dark/Muted for small moves
  if (changePct >= 2.0) {
    bgColor = '#00A846'; // Bright Green (> 2%)
  } else if (changePct >= 0) {
    bgColor = '#0D3D22'; // Dark Green (0 - 2%)
  } else if (changePct >= -2.0) {
    bgColor = '#5a1a1a'; // Lighter Dark Red (-2% - 0)
  } else {
    bgColor = '#CC2929'; // Bright Red (< -2%)
  }

  const handleTileClick = () => {
    const symbol = itemData.name || itemData.root?.name;
    if (symbol && symbol !== 'N/A' && navigate) {
      navigate(`/stock/${encodeURIComponent(symbol)}`);
    }
  };

  return (
    <g className="treemap-item">
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill={bgColor}
        stroke="#fff"
        strokeWidth={2}
        rx={6}
        ry={6}
        onClick={handleTileClick}
        style={{ cursor: 'pointer' }}
      />
      {width > 30 && height > 30 && (
        <>
          <text
            x={x + width / 2}
            y={y + height / 2 - 2}
            textAnchor="middle"
            fill="white"
            fontSize={Math.max(9, Math.min(14, width / 4))}
            fontWeight="700"
            style={{ textShadow: '0 1px 2px rgba(0,0,0,0.3)', pointerEvents: 'none' }}
          >
            {name}
          </text>
          <text
            x={x + width / 2}
            y={y + height / 2 + (height < 50 ? 10 : 15)}
            textAnchor="middle"
            fill="white"
            fontSize={Math.max(8, Math.min(11, width / 5))}
            fontWeight="500"
            style={{ textShadow: '0 1px 2px rgba(0,0,0,0.3)', pointerEvents: 'none' }}
          >
            {isPositive ? '+' : ''}{changePct.toFixed(2)}%
          </text>
        </>
      )}
    </g>
  );
};

TreemapContent.propTypes = {
  x: PropTypes.number,
  y: PropTypes.number,
  width: PropTypes.number,
  height: PropTypes.number,
  payload: PropTypes.oneOfType([
    PropTypes.object,
    PropTypes.shape({
      name: PropTypes.string,
      fullName: PropTypes.string,
      change_pct: PropTypes.number,
      value: PropTypes.number,
      root: PropTypes.shape({
        name: PropTypes.string,
      }),
    }),
  ]),
  navigate: PropTypes.func,
};

TreemapContent.defaultProps = {
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  payload: null,
  navigate: null,
};

// Treemap tooltip content renderer
const TreemapTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload || {};
    const changePct = data.change_pct || 0;
    const isPositive = changePct >= 0;
    const allocText = typeof data.allocation_pct === 'number' ? data.allocation_pct.toFixed(2) : '0.00';
    const changeText = Math.abs(changePct).toFixed(2);

    return (
      <div className="custom-tooltip treemap-tooltip">
        <p className="tooltip-title">{data.fullName || data.name || 'N/A'}</p>
        <p className="tooltip-text">allocation: {allocText}%</p>
        <p className={`tooltip-text ${isPositive ? 'positive' : 'negative'}`}>
          {isPositive ? 'up' : 'down'}: {changeText}%
        </p>
      </div>
    );
  }
  return null;
};

TreemapTooltip.propTypes = {
  active: PropTypes.bool,
  payload: PropTypes.arrayOf(
    PropTypes.shape({
      payload: PropTypes.shape({
        name: PropTypes.string,
        fullName: PropTypes.string,
        change_pct: PropTypes.number,
        allocation_pct: PropTypes.number,
      }),
    })
  ),
};

TreemapTooltip.defaultProps = {
  active: false,
  payload: [],
};

const PortfolioChart = ({ selectedPeriod }) => {
  const navigate = useNavigate();
  const { hideAmounts } = useHideAmounts();
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('30'); // days
  const [benchmarkNames, setBenchmarkNames] = useState(['S&P 500', 'NASDAQ']);
  const [assetAllocation, setAssetAllocation] = useState(null);
  const [allocationLoading, setAllocationLoading] = useState(true);

  const timeRanges = [
    { label: '1 Month', value: '30' },
    { label: '3 Months', value: '90' },
    { label: '6 Months', value: '180' },
    { label: '1 Year', value: '365' },
    { label: 'All', value: 'all' }
  ];


  const fetchChartData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const days = timeRange === 'all' ? 365 : parseInt(timeRange);
      const historyData = await portfolioAPI.getHistory(days);

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
            date: new Date(item.date).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
              year: '2-digit',
            }),
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

      // Update all state at once to minimize re-renders
      setBenchmarkNames(newBenchmarkNames);
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

  useEffect(() => {
    const fetchAssetAllocation = async () => {
      try {
        setAllocationLoading(true);
        const movers = await portfolioAPI.getTopMovers(selectedPeriod);

        if (movers && Array.isArray(movers) && movers.length > 0) {
          // Calculate total portfolio value to determine allocation percentages
          const totalValue = movers.reduce((sum, asset) => sum + (asset.value || 0), 0);

          const assetsWithAllocation = movers
            .filter(asset => asset.value > 0)
            .map(asset => ({
              symbol: asset.symbol || 'N/A',
              name: asset.name || '',
              value: asset.value || 0,
              allocation_pct: totalValue > 0 ? ((asset.value || 0) / totalValue) * 100 : 0,
              change_pct: asset.change_pct || 0
            }))
            .sort((a, b) => b.value - a.value);

          setAssetAllocation(assetsWithAllocation);
        } else {
          console.warn('No movers data found');
          setAssetAllocation([]);
        }
      } catch (err) {
        console.error('Error fetching asset allocation:', err);
        setAssetAllocation([]);
      } finally {
        setAllocationLoading(false);
      }
    };

    fetchAssetAllocation();
  }, [selectedPeriod]);

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
              <YAxis domain={['auto', 'auto']} />
              <Tooltip
                content={<SharedTooltip
                  valueFormatter={(v) => `${Number(v).toFixed(2)}%`}
                  nameMap={{
                    totalReturn: 'Portfolio',
                    spyReturn: benchmarkNames[0],
                    nasdaqReturn: benchmarkNames[1]
                  }}
                />}
              />
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

      {/* Asset Allocation */}
      <div className={`asset-allocation-section ${allocationLoading ? 'loading' : ''}`}>
        <h3>Asset Allocation</h3>
        {allocationLoading && (
          <div className="loading-overlay loading-overlay-tall">
            <div className="loading">Loading asset allocation...</div>
          </div>
        )}
        {assetAllocation && assetAllocation.length > 0 ? (
          <ResponsiveContainer width="100%" height={600} key={`treemap-${selectedPeriod}`}>
            <Treemap
              data={assetAllocation.map(asset => ({
                name: asset.symbol || 'N/A',
                fullName: asset.name || '',
                size: asset.value || 0,
                value: asset.value || 0,
                change_pct: asset.change_pct || 0,
                allocation_pct: asset.allocation_pct || 0
              }))}
              dataKey="size"
              ratio={4 / 3}
              stroke="#fff"
              fill="#fff"
              isAnimationActive={false}
              content={<TreemapContent navigate={navigate} />}
            >
              <Tooltip content={<TreemapTooltip />} />
            </Treemap>
          </ResponsiveContainer>
        ) : (
          <div className="error">
            No asset allocation data available
          </div>
        )}

        <div className="treemap-legend-container">
          <div className="treemap-legend">
            <div className="legend-item">
              <span className="legend-color legend-color-strong-gain"></span>
              <span>Strong Gain (&gt; +2%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color legend-color-gain"></span>
              <span>Gain (0% to +2%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color legend-color-loss"></span>
              <span>Loss (-2% to 0%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color legend-color-strong-loss"></span>
              <span>Strong Loss (&lt; -2%)</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

PortfolioChart.propTypes = {
  selectedPeriod: PropTypes.string,
};

PortfolioChart.defaultProps = {
  selectedPeriod: '1d',
};

export default PortfolioChart;
