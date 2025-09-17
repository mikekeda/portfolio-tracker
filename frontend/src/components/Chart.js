import React, { useState, useEffect, useMemo, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { portfolioAPI } from '../services/api';
import SharedTooltip from './SharedTooltip';
import './Chart.css';

const Chart = () => {
  const [instruments, setInstruments] = useState([]);
  const [selectedSymbols, setSelectedSymbols] = useState(['VUAG.L']);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingInstruments, setLoadingInstruments] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [days, setDays] = useState(30);
  const [selectedMetric, setSelectedMetric] = useState('price');
  const autocompleteRef = useRef(null);
  const inputRef = useRef(null);

  // Load instruments for autocomplete
  useEffect(() => {
    const loadInstruments = async () => {
      try {
        setLoadingInstruments(true);
        const response = await portfolioAPI.getInstruments();
        setInstruments(response.instruments || []);
      } catch (err) {
        console.error('Error loading instruments:', err);
        setError('Failed to load instruments');
      } finally {
        setLoadingInstruments(false);
      }
    };
    loadInstruments();
  }, []);

  // Handle click outside autocomplete
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (autocompleteRef.current && !autocompleteRef.current.contains(event.target)) {
        setShowAutocomplete(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Load chart data when symbols, days, or metric change
  useEffect(() => {
    const loadChartData = async () => {
      if (selectedSymbols.length === 0) {
        setChartData([]);
        return;
      }

      try {
        setLoading(true);
        setError(null);

        let response;
        if (selectedMetric === 'price' || selectedMetric === 'price_pct_change') {
          response = await portfolioAPI.getChartPrices(selectedSymbols, days);
        } else {
          response = await portfolioAPI.getChartMetrics(selectedSymbols, days, selectedMetric);
        }

        if (response.error) {
          setError(response.error);
          return;
        }

        if (!response.data || Object.keys(response.data).length === 0) {
          setError(`No ${selectedMetric} data available for the selected symbols`);
          return;
        }

        // Transform data for Recharts
        const transformedData = transformChartData(response.data, selectedMetric);
        setChartData(transformedData);
      } catch (err) {
        setError(`Failed to load ${selectedMetric} data`);
        console.error('Error loading chart data:', err);
      } finally {
        setLoading(false);
      }
    };

    loadChartData();
  }, [selectedSymbols, days, selectedMetric]);

  // Transform API data to Recharts format
  const transformChartData = (apiData, metric = 'price') => {
    if (!apiData || Object.keys(apiData).length === 0) return [];

    // Filter out symbols with no data
    const validSymbols = Object.keys(apiData).filter(symbol =>
      apiData[symbol] && apiData[symbol].length > 0
    );

    if (validSymbols.length === 0) return [];

    // Get all unique dates
    const allDates = new Set();
    validSymbols.forEach(symbol => {
      apiData[symbol].forEach(point => {
        if (point.date && point.value !== undefined) {
          allDates.add(point.date);
        }
      });
    });

    // Sort dates
    const sortedDates = Array.from(allDates).sort();

    // If metric is price_pct_change, calculate percentage changes
    if (metric === 'price_pct_change') {
      // Calculate percentage change for each symbol
      const percentageData = {};
      validSymbols.forEach(symbol => {
        const symbolData = apiData[symbol].sort((a, b) => new Date(a.date) - new Date(b.date));
        if (symbolData.length > 0) {
          const firstPrice = symbolData[0].value;
          percentageData[symbol] = symbolData.map(point => ({
            date: point.date,
            value: firstPrice > 0 ? ((point.value - firstPrice) / firstPrice) * 100 : 0
          }));
        }
      });

      // Create data points for each date with percentage values
      return sortedDates.map(date => {
        const dataPoint = { date };
        validSymbols.forEach(symbol => {
          const symbolPoint = percentageData[symbol]?.find(point => point.date === date);
          if (symbolPoint && symbolPoint.value !== undefined) {
            dataPoint[symbol] = symbolPoint.value;
          }
        });
        return dataPoint;
      });
    }

    // Default behavior for other metrics
    return sortedDates.map(date => {
      const dataPoint = { date };
      validSymbols.forEach(symbol => {
        const symbolPoint = apiData[symbol].find(point => point.date === date);
        if (symbolPoint && symbolPoint.value !== undefined) {
          dataPoint[symbol] = symbolPoint.value;
        }
      });
      return dataPoint;
    });
  };

  // Filter instruments based on search term
  const filteredInstruments = useMemo(() => {
    if (!searchTerm) return instruments.slice(0, 10);

    return instruments
      .filter(instrument =>
        instrument.symbol?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        instrument.name?.toLowerCase().includes(searchTerm.toLowerCase())
      )
      .slice(0, 10);
  }, [instruments, searchTerm]);

  // Add symbol to chart
  const addSymbol = (symbol) => {
    if (!selectedSymbols.includes(symbol)) {
      setSelectedSymbols([...selectedSymbols, symbol]);
    }
    setSearchTerm('');
    setShowAutocomplete(false);
  };

  // Remove symbol from chart
  const removeSymbol = (symbol) => {
    setSelectedSymbols(selectedSymbols.filter(s => s !== symbol));
  };

  // Get display name for metric
  const getMetricDisplayName = (metric) => {
    const metricNames = {
      'price': 'Price',
      'pe_ratio': 'P/E Ratio',
      'institutional': 'Institutional Ownership',
      'profit': 'Profit',
      'profit_pct': 'Profit %'
    };
    return metricNames[metric] || 'Price';
  };

  // Generate colors for chart lines
  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];

  // Value formatter based on selected metric
  const getValueFormatter = (metric) => {
    switch (metric) {
      case 'price':
        return (value) => value.toFixed(2);
      case 'price_pct_change':
        return (value) => `${value.toFixed(2)}%`;
      case 'pe_ratio':
        return (value) => value.toFixed(1);
      case 'institutional':
        return (value) => `${value.toFixed(1)}%`;
      case 'profit':
        return (value) => `£${value.toFixed(2)}`;
      case 'profit_pct':
        return (value) => `${value.toFixed(1)}%`;
      default:
        return (value) => value.toFixed(2);
    }
  };

  return (
    <div className="chart-container">
      <h1>Stock {getMetricDisplayName(selectedMetric)} Chart</h1>

      {/* Controls */}
      <div className="chart-controls">
        <div className="symbol-input-container" ref={autocompleteRef}>
          <input
            ref={inputRef}
            type="text"
            placeholder={loadingInstruments ? "Loading instruments..." : "Search for stocks..."}
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setShowAutocomplete(true);
            }}
            onFocus={() => setShowAutocomplete(true)}
            className="symbol-input"
            disabled={loadingInstruments}
          />

          {showAutocomplete && (
            <div className="autocomplete-dropdown">
              {filteredInstruments.map((instrument) => (
                <div
                  key={instrument.id}
                  className="autocomplete-item"
                  onClick={() => addSymbol(instrument.symbol)}
                >
                  <span className="symbol">{instrument.symbol}</span>
                  <span className="name">{instrument.name}</span>
                </div>
              ))}
              {filteredInstruments.length === 0 && (
                <div className="autocomplete-item no-results">
                  No instruments found
                </div>
              )}
            </div>
          )}
        </div>

        <div className="metric-selector">
          <label>Metric:</label>
          <select value={selectedMetric} onChange={(e) => setSelectedMetric(e.target.value)}>
            <option value="price">Price</option>
            <option value="price_pct_change">Price % Change</option>
            <option value="pe_ratio">P/E Ratio</option>
            <option value="institutional">Institutional Ownership (%)</option>
            <option value="profit">Profit (£)</option>
            <option value="profit_pct">Profit (%)</option>
          </select>
        </div>

        <div className="days-selector">
          <label>Days:</label>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
            <option value={180}>180 days</option>
            <option value={365}>1 year</option>
            <option value={1827}>5 year</option>
          </select>
        </div>
      </div>

      {/* Selected Symbols */}
      {selectedSymbols.length > 0 && (
        <div className="selected-symbols">
          <h3>Selected Stocks:</h3>
          <div className="symbol-tags">
            {selectedSymbols.map((symbol, index) => (
              <span
                key={symbol}
                className="symbol-tag"
                style={{ backgroundColor: colors[index % colors.length] }}
              >
                {symbol}
                <button
                  onClick={() => removeSymbol(symbol)}
                  className="remove-symbol"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="chart-wrapper">
        {loading && <div className="loading">Loading chart data...</div>}
        {error && <div className="error">{error}</div>}

        {!loading && !error && chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={500}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(date) => new Date(date).toLocaleDateString()}
              />
              <YAxis
                tickFormatter={(value) => {
                  if (selectedMetric === 'price_pct_change') {
                    return `${value.toFixed(1)}%`;
                  }
                  return value;
                }}
              />
              <Tooltip content={<SharedTooltip valueFormatter={getValueFormatter(selectedMetric)} />} />
              <Legend />
              {selectedSymbols.map((symbol, index) => (
                <Line
                  key={symbol}
                  type="monotone"
                  dataKey={symbol}
                  stroke={colors[index % colors.length]}
                  strokeWidth={2}
                  dot={false}
                  name={symbol}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}

        {!loading && !error && chartData.length === 0 && selectedSymbols.length > 0 && (
          <div className="no-data">No price data available for selected stocks</div>
        )}

        {!loading && !error && selectedSymbols.length === 0 && (
          <div className="no-symbols">Add stocks to see the chart</div>
        )}
      </div>
    </div>
  );
};

export default Chart;
