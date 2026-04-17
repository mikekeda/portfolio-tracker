import React, { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { portfolioAPI } from '../services/api';
import SharedTooltip from './SharedTooltip';

// Resolve benchmark symbols to friendly names (mirrors PortfolioChart).
const BENCH_NAME_MAP = {
  'VUAG.L': 'S&P 500',
  'XNAS.L': 'NASDAQ',
};

const UnderwaterChart = ({ timeRange }) => {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [benchmarkNames, setBenchmarkNames] = useState(['S&P 500', 'NASDAQ']);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const days = timeRange === 'all' ? null : parseInt(timeRange);
      const resp = await portfolioAPI.getUnderwater(days);

      const names = (resp.benchmark || []).map((s) => BENCH_NAME_MAP[s] || s);
      if (names.length >= 2) {
        setBenchmarkNames(names);
      }

      if (resp.history && resp.history.length > 0) {
        const processed = resp.history.map((item) => ({
          date: new Date(item.date).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: '2-digit',
          }),
          fullDate: item.date,
          portfolio: item.portfolio_dd_pct,
          vuag: item.benchmark_dd_pct ? item.benchmark_dd_pct[0] : null,
          xnas: item.benchmark_dd_pct ? item.benchmark_dd_pct[1] : null,
        }));
        setChartData(processed);
      } else {
        setChartData([]);
      }
    } catch (err) {
      setError('Failed to fetch underwater data');
      console.error('Error fetching underwater data:', err);
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return (
    <div className={`chart-panel underwater-panel ${loading ? 'loading' : ''}`}>
      <div className="underwater-header">
        <h3>Underwater (drawdown from peak)</h3>
        <div className="underwater-legend">
          <span className="uw-legend-item">
            <span className="uw-legend-swatch uw-swatch-portfolio" />
            Portfolio
          </span>
          <span className="uw-legend-item">
            <span className="uw-legend-swatch uw-swatch-vuag" />
            {benchmarkNames[0]}
          </span>
          <span className="uw-legend-item">
            <span className="uw-legend-swatch uw-swatch-xnas" />
            {benchmarkNames[1]}
          </span>
        </div>
      </div>
      {loading && (
        <div className="loading-overlay">
          <div className="loading">Loading underwater data...</div>
        </div>
      )}
      {error && <div className="error-message">{error}</div>}
      <ResponsiveContainer width="100%" height={160} key={`uw-${timeRange}`}>
        <ComposedChart data={chartData || []} isAnimationActive={false}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" minTickGap={40} />
          <YAxis
            tickFormatter={(v) => `${Math.round(v)}%`}
            domain={['auto', 0]}
          />
          <Tooltip
            content={(
              <SharedTooltip
                valueFormatter={(v) => `${Number(v).toFixed(2)}%`}
                nameMap={{
                  portfolio: 'Portfolio',
                  vuag: benchmarkNames[0],
                  xnas: benchmarkNames[1],
                }}
                sortOrder={['portfolio', 'vuag', 'xnas']}
              />
            )}
          />
          <ReferenceLine y={0} stroke="#6c757d" />
          <Area
            type="monotone"
            dataKey="portfolio"
            name="Portfolio"
            stroke="#ffc658"
            strokeWidth={2}
            fill="#ffc658"
            fillOpacity={0.35}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="vuag"
            name={benchmarkNames[0]}
            stroke="#2563eb"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="xnas"
            name={benchmarkNames[1]}
            stroke="#dc2626"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

UnderwaterChart.propTypes = {
  timeRange: PropTypes.string.isRequired,
};

export default UnderwaterChart;
