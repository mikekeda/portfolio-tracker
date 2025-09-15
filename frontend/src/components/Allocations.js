import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { portfolioAPI } from '../services/api';
import { renderCountryWithFlag } from '../utils/countryUtils';
import SharedTooltip from './SharedTooltip';
import './Allocations.css';

const Allocations = () => {
  const [chartData, setChartData] = useState(null);
  const [allocations, setAllocations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Time ranges and their corresponding day windows
  const timeRanges = [
    { value: '1m', label: '1 Month', days: 30 },
    { value: '3m', label: '3 Months', days: 90 },
    { value: '6m', label: '6 Months', days: 180 },
    { value: '1y', label: '1 Year', days: 365 },
    { value: 'all', label: 'All', days: 365 },
  ];
  const [timeRange, setTimeRange] = useState('1m');

  const fetchData = async () => {
    try {
      setLoading(true);
      const selected = timeRanges.find(r => r.value === timeRange) || timeRanges[0];
      const days = selected.days;
      const [historyData, allocationsData] = await Promise.all([
        portfolioAPI.getHistory(days),
        portfolioAPI.getAllocations()
      ]);

      if (historyData.history && historyData.history.length > 0) {
        // Process the data for charts
        const processedData = historyData.history.map(item => {
          const processedItem = {
            date: new Date(item.date).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric'
            }),
            fullDate: item.date,
          };

          // Flatten allocation data for charts
          if (item.country_allocation) {
            Object.entries(item.country_allocation).forEach(([key, value]) => {
              processedItem[`country_${key}`] = value;
            });
          }

          if (item.sector_allocation) {
            Object.entries(item.sector_allocation).forEach(([key, value]) => {
              processedItem[`sector_${key}`] = value;
            });
          }

          if (item.currency_allocation) {
            Object.entries(item.currency_allocation).forEach(([key, value]) => {
              processedItem[`currency_${key}`] = value;
            });
          }

          if (item.etf_equity_split) {
            Object.entries(item.etf_equity_split).forEach(([key, value]) => {
              processedItem[`etf_${key}`] = value;
            });
          }

          return processedItem;
        });

        // Sort data by date to ensure chronological order (oldest first, left to right)
        const sortedData = processedData.sort((a, b) => new Date(a.fullDate) - new Date(b.fullDate));
        setChartData(sortedData);
      }

      setAllocations(allocationsData);
      setError(null);
    } catch (err) {
      setError('Failed to fetch chart data');
      console.error('Error fetching chart data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [timeRange]); // eslint-disable-line react-hooks/exhaustive-deps

  const getTopAllocations = (data, type, limit = 8) => {
    if (!data || data.length === 0) return [];

    const latestData = data[data.length - 1];

    // Extract allocation data based on type prefix
    let prefix = '';
    switch (type) {
      case 'country_allocation':
        prefix = 'country_';
        break;
      case 'sector_allocation':
        prefix = 'sector_';
        break;
      case 'currency_allocation':
        prefix = 'currency_';
        break;
      case 'etf_equity_split':
        prefix = 'etf_';
        break;
      default:
        return [];
    }

    // Filter data by prefix and extract allocations
    const allocations = {};
    Object.entries(latestData).forEach(([key, value]) => {
      if (key.startsWith(prefix)) {
        const cleanKey = key.replace(prefix, '');
        allocations[cleanKey] = value;
      }
    });


    // Filter out invalid values and sort
    return Object.entries(allocations)
      .filter(([, value]) => typeof value === 'number' && !isNaN(value))
      .sort(([,a], [,b]) => b - a)
      .slice(0, limit)
      .map(([key, value]) => ({ key, value }));
  };

  const getChartColors = (index) => {
    const colors = [
      '#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00ff00',
      '#ff00ff', '#00ffff', '#ffff00', '#ff0000', '#0000ff',
      '#800080', '#008000', '#ffa500', '#ff69b4', '#40e0d0'
    ];
    return colors[index % colors.length];
  };

  if (loading) {
    return (
      <div className="allocations-container">
        <div className="loading">Loading allocation data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="allocations-container">
        <div className="error">{error}</div>
      </div>
    );
  }

  if (!chartData || chartData.length === 0) {
    return (
      <div className="allocations-container">
        <div className="error">No allocation data available</div>
      </div>
    );
  }

  const topCountries = getTopAllocations(chartData, 'country_allocation');
  const topSectors = getTopAllocations(chartData, 'sector_allocation');
  const topCurrencies = getTopAllocations(chartData, 'currency_allocation');
  const etfEquityData = getTopAllocations(chartData, 'etf_equity_split');

  return (
    <div className="allocations-container">
      <div className="chart-header">
        <h2>Portfolio Allocations</h2>
      </div>

      {/* Allocation Tables first */}
      <div id="tables" className="allocation-tables">
        {allocations && (
          <>
            {/* ETF/Equity Split and Currency Allocation */}
            <div className="allocations-section">
              <div className="allocation-table">
                <h3>ETF/Equity Split</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Type</th>
                      <th>Allocation (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allocations.etf_equity_split && Object.entries(allocations.etf_equity_split).length > 0 ? (
                      Object.entries(allocations.etf_equity_split)
                        .sort(([,a], [,b]) => b - a)
                        .map(([type, percentage]) => (
                          <tr key={type}>
                            <td>{type}</td>
                            <td style={{ '--bar-width': `${Math.min(percentage, 100)}%` }}>
                              <span>{percentage.toFixed(2)}%</span>
                            </td>
                          </tr>
                        ))
                    ) : (
                      <tr>
                        <td colSpan="2" style={{ textAlign: 'center', color: '#6c757d' }}>
                          No ETF/Equity data available
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="allocation-table">
                <h3>Currency Allocation</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Currency</th>
                      <th>Allocation (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allocations.currency_allocation && Object.entries(allocations.currency_allocation).length > 0 ? (
                      Object.entries(allocations.currency_allocation)
                        .sort(([,a], [,b]) => b - a)
                        .map(([currency, percentage]) => (
                          <tr key={currency}>
                            <td>
                              <span className="currency-code">{currency}</span>
                              {currency === 'USD' && ' 🇺🇸'}
                              {currency === 'GBP' && ' 🇬🇧'}
                              {currency === 'EUR' && ' 🇪🇺'}
                              {currency === 'CAD' && ' 🇨🇦'}
                              {currency === 'JPY' && ' 🇯🇵'}
                            </td>
                            <td style={{ '--bar-width': `${Math.min(percentage, 100)}%` }}>
                              <span>{percentage.toFixed(2)}%</span>
                            </td>
                          </tr>
                        ))
                    ) : (
                      <tr>
                        <td colSpan="2" style={{ textAlign: 'center', color: '#6c757d' }}>
                          No currency data available
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Sector and Country Allocations */}
            <div className="allocations-section">
              <div className="allocation-table">
                <h3>Sector Allocation</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Sector</th>
                      <th>Allocation (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(allocations.sector_allocation)
                      .sort(([,a], [,b]) => b - a)
                      .map(([sector, percentage]) => (
                        <tr key={sector}>
                          <td>{sector}</td>
                          <td style={{ '--bar-width': `${Math.min(percentage, 100)}%` }}>
                            <span>{percentage.toFixed(2)}%</span>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              <div className="allocation-table">
                <h3>Country Allocation</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Country</th>
                      <th>Allocation (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(allocations.country_allocation)
                      .sort(([,a], [,b]) => b - a)
                      .map(([country, percentage]) => (
                        <tr key={country}>
                          <td>
                            {renderCountryWithFlag(country)}
                          </td>
                          <td style={{ '--bar-width': `${Math.min(percentage, 100)}%` }}>
                            <span>{percentage.toFixed(2)}%</span>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Charts after tables with selector */}
      <div id="charts" className="allocation-charts">
        <div style={{ gridColumn: '1 / -1', display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
          {timeRanges.map((range) => (
            <button
              key={range.value}
              className={`time-range-btn ${timeRange === range.value ? 'active' : ''}`}
              onClick={() => setTimeRange(range.value)}
              style={{ marginLeft: 8 }}
            >
              {range.label}
            </button>
          ))}
        </div>

        <div className="chart-panel allocation-panel">
          <h3>Country Allocation Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 100]} />
              <Tooltip content={<SharedTooltip prefix="country_" valueFormatter={(value) => `${value.toFixed(2)}%`} />} />
              <Legend />
              {topCountries.map((country, index) => (
                <Line
                  key={country.key}
                  type="monotone"
                  dataKey={`country_${country.key}`}
                  name={country.key}
                  stroke={getChartColors(index)}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel allocation-panel">
          <h3>Sector Allocation Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 100]} />
              <Tooltip content={<SharedTooltip prefix="sector_" valueFormatter={(value) => `${value.toFixed(2)}%`} />} />
              <Legend />
              {topSectors.map((sector, index) => (
                <Line
                  key={sector.key}
                  type="monotone"
                  dataKey={`sector_${sector.key}`}
                  name={sector.key}
                  stroke={getChartColors(index)}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel allocation-panel">
          <h3>Currency Allocation Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 100]} />
              <Tooltip content={<SharedTooltip prefix="currency_" valueFormatter={(value) => `${value.toFixed(2)}%`} />} />
              <Legend />
              {topCurrencies.map((currency, index) => (
                <Line
                  key={currency.key}
                  type="monotone"
                  dataKey={`currency_${currency.key}`}
                  name={currency.key}
                  stroke={getChartColors(index)}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel allocation-panel">
          <h3>ETF/Equity Split Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 100]} />
              <Tooltip content={<SharedTooltip prefix="etf_" valueFormatter={(value) => `${value.toFixed(2)}%`} />} />
              <Legend />
              {etfEquityData.map((item, index) => (
                <Line
                  key={item.key}
                  type="monotone"
                  dataKey={`etf_${item.key}`}
                  name={item.key}
                  stroke={getChartColors(index)}
                  strokeWidth={3}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Allocations;
