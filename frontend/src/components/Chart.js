import React, { useState, useEffect, useMemo, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { portfolioAPI } from '../services/api';
import './Chart.css';

const Chart = () => {
  const [instruments, setInstruments] = useState([]);
  const [selectedSymbols, setSelectedSymbols] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingInstruments, setLoadingInstruments] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [days, setDays] = useState(30);
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

  // Load chart data when symbols or days change
  useEffect(() => {
    const loadChartData = async () => {
      if (selectedSymbols.length === 0) {
        setChartData([]);
        return;
      }

      try {
        setLoading(true);
        setError(null);
        const response = await portfolioAPI.getChartPrices(selectedSymbols, days);

        if (response.error) {
          setError(response.error);
          return;
        }

        if (!response.data || Object.keys(response.data).length === 0) {
          setError('No price data available for the selected symbols');
          return;
        }

        // Transform data for Recharts
        const transformedData = transformChartData(response.data);
        setChartData(transformedData);
      } catch (err) {
        setError('Failed to load chart data');
        console.error('Error loading chart data:', err);
      } finally {
        setLoading(false);
      }
    };

    loadChartData();
  }, [selectedSymbols, days]);

  // Transform API data to Recharts format
  const transformChartData = (apiData) => {
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
        if (point.date && point.price) {
          allDates.add(point.date);
        }
      });
    });

    // Sort dates
    const sortedDates = Array.from(allDates).sort();

    // Create data points for each date
    return sortedDates.map(date => {
      const dataPoint = { date };
      validSymbols.forEach(symbol => {
        const symbolPoint = apiData[symbol].find(point => point.date === date);
        if (symbolPoint && symbolPoint.price) {
          dataPoint[symbol] = symbolPoint.price;
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

  // Generate colors for chart lines
  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];

  return (
    <div className="chart-container">
      <h1>Stock Price Chart</h1>

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

        <div className="days-selector">
          <label>Days:</label>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
            <option value={180}>180 days</option>
            <option value={365}>1 year</option>
          </select>
        </div>
      </div>

      {/* Selected Symbols */}
      {selectedSymbols.length > 0 && (
        <div className="selected-symbols">
          <h3>Selected Stocks:</h3>
          <div className="symbol-tags">
            {selectedSymbols.map((symbol, index) => (
              <span key={symbol} className="symbol-tag">
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
              <YAxis />
              <Tooltip
                labelFormatter={(date) => new Date(date).toLocaleDateString()}
                formatter={(value, name) => [value.toFixed(2), name]}
              />
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
