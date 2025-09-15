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
          };


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
          <h3>Total Value (£)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip
                formatter={(value) => [`£${value.toLocaleString()}`, 'Total Value']}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Area
                type="monotone"
                dataKey="totalValue"
                stroke="#8884d8"
                fill="#8884d8"
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel">
          <h3>Total Profit/Loss (£)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip
                formatter={(value) => [`£${value.toLocaleString()}`, 'Total Profit/Loss']}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Line
                type="monotone"
                dataKey="totalProfit"
                stroke="#82ca9d"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel">
          <h3>Total Return (%)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip
                content={<SharedTooltip valueFormatter={(v) => `${Number(v).toFixed(2)}%`} />}
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
                dataKey="benchmarkReturn"
                name="Benchmark"
                stroke="#2563eb"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
};

export default PortfolioChart;
