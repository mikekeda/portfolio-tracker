import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';
import { portfolioAPI } from '../services/api';
import './PortfolioChart.css';
import SharedTooltip from './SharedTooltip';

const PortfolioChart = () => {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('30'); // days
  const [benchmarkNames, setBenchmarkNames] = useState(['S&P 500', 'NASDAQ']);

  const timeRanges = [
    { label: '1 Month', value: '30' },
    { label: '3 Months', value: '90' },
    { label: '6 Months', value: '180' },
    { label: '1 Year', value: '365' },
    { label: 'All', value: 'all' }
  ];

  useEffect(() => {
    fetchChartData();
  }, [timeRange]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchChartData = async () => {
    try {
      setLoading(true);
      const days = timeRange === 'all' ? 365 : parseInt(timeRange);
      const historyData = await portfolioAPI.getHistory(days);

      // Store benchmark names from API response
      if (historyData.benchmark && Array.isArray(historyData.benchmark)) {
        // Map benchmark symbols to readable names
        const nameMap = {
          'VUAG.L': 'S&P 500',
          'XNAS.L': 'NASDAQ'
        };
        const names = historyData.benchmark.map(symbol => nameMap[symbol] || symbol);
        setBenchmarkNames(names);
      }

      if (historyData.history && historyData.history.length > 0) {
        // Process the data for charts
            const processedData = historyData.history.map(item => {
          const processedItem = {
            date: new Date(item.date).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric'
            }),
            fullDate: item.date,
            totalValue: item.total_value || 0,
            totalProfit: item.total_profit || 0,
            totalReturn: item.total_return_pct || 0,
            benchmarkReturn: item.benchmark_return_pct ?? null,
            // Handle multiple benchmarks
            spyReturn: Array.isArray(item.benchmark_return_pct) ? item.benchmark_return_pct[0] : item.benchmark_return_pct,
            nasdaqReturn: Array.isArray(item.benchmark_return_pct) ? item.benchmark_return_pct[1] : null,
          };

              // Derived fields for combined chart (invested base, profit/loss bands)
              const invested = (processedItem.totalValue - processedItem.totalProfit) || 0;
              const profitPos = Math.max(processedItem.totalProfit, 0);
              const lossNeg = Math.min(processedItem.totalProfit, 0); // negative or 0
              processedItem.invested = invested;
              processedItem.profitPos = profitPos;
              processedItem.lossNeg = lossNeg; // negative values

          return processedItem;
        });

        // Sort data by date to ensure chronological order (oldest first, left to right)
        const sortedData = processedData.sort((a, b) => new Date(a.fullDate) - new Date(b.fullDate));
        setChartData(sortedData);
      }
      setError(null);
    } catch (err) {
      setError('Failed to fetch chart data');
      console.error('Error fetching chart data:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="portfolio-chart-container">
        <div className="loading">Loading chart data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="portfolio-chart-container">
        <div className="error">{error}</div>
      </div>
    );
  }

  if (!chartData || chartData.length === 0) {
    return (
      <div className="portfolio-chart-container">
        <div className="error">No chart data available</div>
      </div>
    );
  }

  // Custom tooltip for combined Total Value + Profit/Loss chart
  const CombinedTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const getVal = (key) => {
        const entry = payload.find((p) => p.dataKey === key);
        const v = entry ? Number(entry.value) : 0;
        return isNaN(v) ? 0 : v;
      };
      const invested = getVal('invested');
      const profit = getVal('profitPos') + getVal('lossNeg'); // lossNeg is negative
      const profitColor = profit >= 0 ? '#28a745' : '#dc3545';

      const fmt = (n) => `£${Number(n).toLocaleString()}`;

      return (
        <div className="custom-tooltip">
          <p className="tooltip-label">{label}</p>
          <p className="tooltip-item">Total: {fmt(invested + profit)}</p>
          <p className="tooltip-item" style={{ color: profitColor }}>Profit: {fmt(profit)}</p>
          <p className="tooltip-item" style={{ color: '#8884d8' }}>Base Cost: {fmt(invested)}</p>
        </div>
      );
    }
    return null;
  };

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
            >
              {range.label}
            </button>
          ))}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="performance-charts">
        <div className="chart-panel">
          <h3>Total Value (£) + Profit/Loss</h3>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip content={<CombinedTooltip />} />
              {/* Base invested cost */}
              <Area type="monotone" dataKey="invested" name="Base Cost" stackId="1" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} />
              {/* Profit overlay (positive values) */}
              <Area type="monotone" dataKey="profitPos" name="Profit" stackId="1" stroke="#28a745" fill="#28a745" fillOpacity={0.4} />
              {/* Loss overlay (negative values) - rendered in red */}
              <Area type="monotone" dataKey="lossNeg" name="Loss" stackId="1" stroke="#dc3545" fill="#dc3545" fillOpacity={0.35} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel">
          <h3>Total Return (%)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
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
              />
              <Line
                type="monotone"
                dataKey="spyReturn"
                name={benchmarkNames[0]}
                stroke="#2563eb"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="nasdaqReturn"
                name={benchmarkNames[1]}
                stroke="#dc2626"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
};

export default PortfolioChart;
