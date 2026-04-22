import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { portfolioAPI } from '../services/api';
import SharedTooltip from './SharedTooltip';
import './Chart.css';

// Days selector configuration
const DAYS_OPTIONS = [
  { label: '5D', value: 5 },
  { label: '1M', value: 30 },
  { label: '3M', value: 90 },
  { label: '6M', value: 180 },
  { label: 'YTD', value: 'ytd' },
  { label: '1Y', value: 365 },
  { label: '2Y', value: 730 },
  { label: '5Y', value: 1825 },
  { label: '10Y', value: 3650 },
];

// Helper function to calculate YTD days
const calculateYTDDays = () => {
  const now = new Date();
  const startOfYear = new Date(now.getFullYear(), 0, 1);
  const diffTime = Math.abs(now - startOfYear);
  return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
};

const Chart = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  // Initialize selectedSymbols from URL or default. Always upper-cased to match
  // backend canonicalization.
  const getSymbolsFromURL = useCallback(() => {
    const normalize = (list) => list.map((s) => s.trim().toUpperCase()).filter(Boolean);

    const symbolsParam = searchParams.get('symbols');
    if (symbolsParam) {
      if (symbolsParam.includes(' ')) return normalize(symbolsParam.split(' '));
      if (symbolsParam.includes(',')) return normalize(symbolsParam.split(','));
      return normalize([symbolsParam]);
    }

    // Fallback: check for multiple 'symbols' params (legacy from previous step)
    const multipleParams = searchParams.getAll('symbols');
    if (multipleParams.length > 0) return normalize(multipleParams);

    return ["VUAG.L", "XNAS.L"];
  }, [searchParams]);

  const [instruments, setInstruments] = useState([]);
  const [selectedSymbols, setSelectedSymbols] = useState(getSymbolsFromURL);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingInstruments, setLoadingInstruments] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);

  // Days selector state - get from URL or default to 30 (1M)
  const getDaysFromURL = useCallback(() => {
    const daysParam = searchParams.get('days');
    if (daysParam) {
      const parsed = daysParam === 'ytd' ? 'ytd' : Number(daysParam);
      return parsed || 30;
    }
    return 30; // Default to 1M
  }, [searchParams]);

  // Metric state - get from URL or default to 'price_pct_change' (gains)
  const getMetricFromURL = useCallback(() => {
    const metricParam = searchParams.get('metric');
    return metricParam || 'price_pct_change';
  }, [searchParams]);

  const [days, setDays] = useState(getDaysFromURL);
  const [selectedMetric, setSelectedMetric] = useState(getMetricFromURL);
  const [rawSeries, setRawSeries] = useState({});
  const [fundamentals, setFundamentals] = useState({});
  // null sortBy = preserve selection order; otherwise the column key
  const [sortBy, setSortBy] = useState(null);
  const [sortDir, setSortDir] = useState('desc');
  const autocompleteRef = useRef(null);
  const inputRef = useRef(null);
  const listRef = useRef(null);

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

  // Update URL when selectedSymbols or days change (skip on initial mount to avoid replacing URL)
  const isInitialMount = useRef(true);
  useEffect(() => {
    // Skip URL update on initial mount since we're reading from URL
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    const params = new URLSearchParams();
    if (selectedSymbols.length > 0) {
      // Join with space, which URLSearchParams encodes as +
      params.set('symbols', selectedSymbols.join(' '));
    }
    const daysValue = days === 'ytd' ? 'ytd' : String(days);
    params.set('days', daysValue);
    params.set('metric', selectedMetric);
    // Use replace: true to avoid creating history entries for each change
    setSearchParams(params, { replace: true });
  }, [selectedSymbols, days, selectedMetric, setSearchParams]);

  // Load chart data when symbols, days, or metric change.
  // `cancelled` guards against stale responses overwriting newer ones when the
  // user changes inputs faster than the network resolves.
  useEffect(() => {
    let cancelled = false;
    const loadChartData = async () => {
      if (selectedSymbols.length === 0) {
        setChartData([]);
        return;
      }

      try {
        setLoading(true);
        setError(null);

        const daysParam = days === 'ytd' ? calculateYTDDays() : days;

        let response;
        if (selectedMetric === 'price' || selectedMetric === 'price_pct_change') {
          response = await portfolioAPI.getChartPrices(selectedSymbols, daysParam);
        } else {
          response = await portfolioAPI.getChartMetrics(selectedSymbols, daysParam, selectedMetric);
        }

        if (cancelled) return;

        if (response.error) {
          setError(response.error);
          return;
        }

        if (!response.data || Object.keys(response.data).length === 0) {
          setError(`No ${selectedMetric} data available for the selected symbols`);
          setRawSeries({});
          return;
        }

        const transformedData = transformChartData(response.data, selectedMetric);
        setChartData(transformedData);
        setRawSeries(response.data);
      } catch (err) {
        if (cancelled) return;
        setError(`Failed to load ${selectedMetric} data`);
        console.error('Error loading chart data:', err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    loadChartData();
    return () => { cancelled = true; };
  }, [selectedSymbols, days, selectedMetric]);

  // Load fundamentals for the summary row. Only depends on the selected symbols.
  useEffect(() => {
    let cancelled = false;
    const loadFundamentals = async () => {
      if (selectedSymbols.length === 0) {
        setFundamentals({});
        return;
      }
      try {
        const response = await portfolioAPI.getChartFundamentals(selectedSymbols);
        if (!cancelled) setFundamentals(response.data || {});
      } catch (err) {
        if (cancelled) return;
        console.error('Error loading fundamentals:', err);
        setFundamentals({});
      }
    };
    loadFundamentals();
    return () => { cancelled = true; };
  }, [selectedSymbols]);

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

  // Filter instruments based on search term - remove duplicates by symbol
  const filteredInstruments = useMemo(() => {
    if (!searchTerm.trim()) return instruments.slice(0, 10);

    const searchLower = searchTerm.toLowerCase().trim();
    const filtered = instruments.filter(instrument => {
      const symbol = instrument.symbol?.toLowerCase() || '';
      const name = instrument.name?.toLowerCase() || '';
      return symbol.includes(searchLower) || name.includes(searchLower);
    });

    // Remove duplicates by symbol (keep first occurrence)
    const seen = new Set();
    const unique = filtered.filter(instrument => {
      if (!instrument.symbol) return false;
      const symbol = instrument.symbol.trim();
      if (!symbol || seen.has(symbol)) return false;
      seen.add(symbol);
      return true;
    });

    return unique.slice(0, 10);
  }, [instruments, searchTerm]);

  // Add symbol to chart. Normalize to upper-case so it matches the backend's
  // canonicalization and prevents duplicate entries like "aapl" + "AAPL".
  const addSymbol = (symbol) => {
    if (!symbol || !symbol.trim()) return;
    const normalized = symbol.trim().toUpperCase();
    if (!selectedSymbols.includes(normalized)) {
      setSelectedSymbols([...selectedSymbols, normalized]);
    }
    setSearchTerm('');
    setShowAutocomplete(false);
    setHighlightedIndex(-1);
  };

  // Remove symbol from chart
  const removeSymbol = (symbol) => {
    setSelectedSymbols(selectedSymbols.filter(s => s !== symbol));
  };

  // Get display name for metric
  const getMetricDisplayName = (metric) => {
    const metricNames = {
      'price': 'Price',
      'price_pct_change': 'Gains',
      'pe_ratio': 'P/E Ratio',
      'institutional': 'Institutional Ownership',
      'profit': 'Profit',
      'profit_pct': 'Profit %'
    };
    return metricNames[metric] || 'Price';
  };

  // Generate colors for chart lines
  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];

  // Return for the selected metric over the visible range.
  // rawSeries holds the backend response as-is: raw prices for price /
  // price_pct_change, and absolute values (PE, profit, etc.) for other metrics.
  // - price-based metrics: % change from first to last point
  // - other metrics: absolute delta (last - first)
  const returns = useMemo(() => {
    const out = {};
    const isPriceBased = selectedMetric === 'price' || selectedMetric === 'price_pct_change';
    selectedSymbols.forEach((symbol) => {
      const series = rawSeries[symbol];
      if (!series || series.length === 0) {
        out[symbol] = null;
        return;
      }
      const values = [...series]
        .sort((a, b) => new Date(a.date) - new Date(b.date))
        .map((p) => p.value)
        .filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
      if (values.length === 0) {
        out[symbol] = null;
        return;
      }
      const first = values[0];
      const last = values[values.length - 1];
      if (isPriceBased) {
        out[symbol] = first !== 0 ? ((last - first) / first) * 100 : null;
      } else {
        out[symbol] = last - first;
      }
    });
    return out;
  }, [selectedSymbols, rawSeries, selectedMetric]);

  // Formatters for the fundamentals cells.
  const formatMarketCap = (val) => {
    if (val === null || val === undefined) return '—';
    const abs = Math.abs(val);
    if (abs >= 1e12) return `${(val / 1e12).toFixed(2)}T`;
    if (abs >= 1e9) return `${(val / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `${(val / 1e6).toFixed(2)}M`;
    return val.toFixed(0);
  };
  const formatRatio = (val) => (val === null || val === undefined ? '—' : val.toFixed(1));
  const formatPercent = (val, signed = false) => {
    if (val === null || val === undefined) return '—';
    const sign = signed && val > 0 ? '+' : '';
    return `${sign}${val.toFixed(1)}%`;
  };

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

  const renderSummary = () => {
    const formatter = getValueFormatter(selectedMetric);
    const formatReturn = (val) => {
      if (val === null || val === undefined || Number.isNaN(val)) return '—';
      if (selectedMetric === 'price' || selectedMetric === 'price_pct_change' || selectedMetric === 'profit_pct') {
        return `${val >= 0 ? '+' : ''}${val.toFixed(2)}%`;
      }
      return formatter(val);
    };

    // Column definitions: key, label, tooltip, value accessor, formatter.
    // `value` returns a numeric/string sort key (null = missing, sorts last).
    const cols = [
      {
        key: 'symbol',
        label: 'Symbol',
        tooltip: 'Ticker symbol (click to sort alphabetically)',
        value: (sym) => sym,
        render: (sym, _f, color) => (
          <>
            <span className="chart-summary-swatch" style={{ backgroundColor: color }} />
            <span className="chart-summary-ticker">{sym}</span>
          </>
        ),
        className: 'chart-summary-symbol',
      },
      {
        key: 'return',
        label: 'Return',
        tooltip: 'Change in the selected metric over the chosen range',
        value: (sym) => returns[sym],
        render: (sym) => {
          const ret = returns[sym];
          const cls = ret === null || ret === undefined ? '' : ret >= 0 ? 'positive' : 'negative';
          return <span className={`chart-summary-return ${cls}`}>{formatReturn(ret)}</span>;
        },
      },
      {
        key: 'market_cap',
        label: 'Mkt Cap',
        tooltip: 'Market capitalization (shares outstanding × price)',
        value: (sym) => fundamentals[sym]?.market_cap ?? null,
        render: (_s, f) => formatMarketCap(f.market_cap),
      },
      {
        key: 'pe',
        label: 'PE',
        tooltip: 'Trailing P/E: price divided by last 12 months of earnings per share. Lower = cheaper on current earnings',
        value: (sym) => fundamentals[sym]?.pe ?? null,
        render: (_s, f) => formatRatio(f.pe),
      },
      {
        key: 'fwd_pe',
        label: 'Fwd PE',
        tooltip: 'Forward P/E: price divided by next-12-month analyst EPS estimate. Lower than trailing PE means analysts expect earnings growth',
        value: (sym) => fundamentals[sym]?.fwd_pe ?? null,
        render: (_s, f) => formatRatio(f.fwd_pe),
      },
      {
        key: 'peg',
        label: 'PEG',
        tooltip: 'PEG ratio: forward PE divided by expected earnings growth rate. Below 1 = potentially undervalued for the growth, above 2 = expensive even for a growth story',
        value: (sym) => fundamentals[sym]?.peg ?? null,
        render: (_s, f) => formatRatio(f.peg),
      },
      {
        key: 'roic',
        label: 'ROIC',
        tooltip: 'Return on Invested Capital: NOPAT / (equity + debt - cash). Measures how efficiently the business turns capital into profit. Primary quality gate for compounders',
        value: (sym) => fundamentals[sym]?.roic ?? null,
        render: (_s, f) => formatPercent(f.roic),
      },
      {
        key: 'revenue_growth',
        label: 'Rev Growth',
        tooltip: 'Year-over-year revenue growth (most recent quarter vs same quarter last year)',
        value: (sym) => fundamentals[sym]?.revenue_growth ?? null,
        render: (_s, f) => formatPercent(f.revenue_growth, true),
      },
      {
        key: 'profit_margins',
        label: 'Margins',
        tooltip: 'Net profit margin: net income as a percent of revenue. Higher = more profit kept per dollar of sales (quality signal)',
        value: (sym) => fundamentals[sym]?.profit_margins ?? null,
        render: (_s, f) => formatPercent(f.profit_margins),
      },
      {
        key: 'fifty_two_week_high_distance',
        label: '52W H',
        tooltip: 'Distance from the 52-week high as a percent. 0% = at the high, -20% = 20% below it',
        value: (sym) => fundamentals[sym]?.fifty_two_week_high_distance ?? null,
        render: (_s, f) => formatPercent(f.fifty_two_week_high_distance, true),
      },
    ];

    const rows = selectedSymbols.map((symbol, index) => ({ symbol, index, color: colors[index % colors.length] }));

    if (sortBy) {
      const col = cols.find((c) => c.key === sortBy);
      if (col) {
        const dirMul = sortDir === 'asc' ? 1 : -1;
        rows.sort((a, b) => {
          const va = col.value(a.symbol);
          const vb = col.value(b.symbol);
          const aMissing = va === null || va === undefined || Number.isNaN(va);
          const bMissing = vb === null || vb === undefined || Number.isNaN(vb);
          if (aMissing && bMissing) return a.index - b.index;
          if (aMissing) return 1;
          if (bMissing) return -1;
          if (typeof va === 'string' || typeof vb === 'string') {
            return String(va).localeCompare(String(vb)) * dirMul;
          }
          return (va - vb) * dirMul;
        });
      }
    }

    const onHeaderClick = (key) => {
      if (sortBy === key) {
        setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
      } else {
        setSortBy(key);
        setSortDir(key === 'symbol' ? 'asc' : 'desc');
      }
    };

    const sortIndicator = (key) => {
      if (sortBy !== key) return '';
      return sortDir === 'asc' ? ' ▲' : ' ▼';
    };

    return (
      <div className="chart-summary">
        <div className="chart-summary-header">
          {cols.map((col) => (
            <div
              key={col.key}
              className={`chart-summary-cell chart-summary-th ${col.className || ''} ${sortBy === col.key ? 'sorted' : ''}`}
              title={col.tooltip}
              onClick={() => onHeaderClick(col.key)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onHeaderClick(col.key);
                }
              }}
            >
              {col.label}{sortIndicator(col.key)}
            </div>
          ))}
        </div>
        {rows.map(({ symbol, color }) => {
          const f = fundamentals[symbol] || {};
          return (
            <div key={symbol} className="chart-summary-row">
              {cols.map((col) => (
                <div key={col.key} className={`chart-summary-cell ${col.className || ''}`}>
                  {col.render(symbol, f, color)}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="page-fixed chart-container">
      {/* Header */}
      <div className="chart-header">
        <h1>Stock {getMetricDisplayName(selectedMetric)} Chart</h1>
      </div>

      {/* Controls Row 1: Search and Selected Stocks */}
      <div className="chart-controls-row">
        <div className="symbol-input-container" ref={autocompleteRef}>
          <input
            ref={inputRef}
            type="text"
            placeholder={loadingInstruments ? "Loading instruments..." : "Search for stocks..."}
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setShowAutocomplete(true);
              setHighlightedIndex(-1);
            }}
            onFocus={() => setShowAutocomplete(true)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                setShowAutocomplete(false);
                setHighlightedIndex(-1);
                inputRef.current?.blur();
              } else if (e.key === 'ArrowDown' && showAutocomplete && filteredInstruments.length > 0) {
                e.preventDefault();
                setHighlightedIndex(prev =>
                  prev < filteredInstruments.length - 1 ? prev + 1 : prev
                );
              } else if (e.key === 'ArrowUp' && showAutocomplete) {
                e.preventDefault();
                setHighlightedIndex(prev => prev > 0 ? prev - 1 : -1);
              } else if (e.key === 'Enter' && highlightedIndex >= 0 && filteredInstruments.length > 0) {
                e.preventDefault();
                const selectedInstrument = filteredInstruments[highlightedIndex];
                if (selectedInstrument?.symbol) {
                  addSymbol(selectedInstrument.symbol);
                  setHighlightedIndex(-1);
                }
              }
            }}
            className="symbol-input"
            disabled={loadingInstruments}
          />

          {showAutocomplete && searchTerm.trim() && (
            <div className="autocomplete-dropdown" ref={listRef}>
              {filteredInstruments.length > 0 ? (
                filteredInstruments.map((instrument, index) => {
                  // Use symbol as key, fallback to index if symbol is missing
                  const uniqueKey = instrument.symbol || `instrument-${index}`;
                  const isHighlighted = index === highlightedIndex;
                  return (
                    <div
                      key={uniqueKey}
                      className={`autocomplete-item ${isHighlighted ? 'highlighted' : ''}`}
                      onClick={() => {
                        addSymbol(instrument.symbol);
                        setHighlightedIndex(-1);
                      }}
                      onMouseEnter={() => setHighlightedIndex(index)}
                    >
                      <span className="symbol">{instrument.symbol || 'N/A'}</span>
                      <span className="name">{instrument.name || instrument.symbol || 'Unknown'}</span>
                    </div>
                  );
                })
              ) : (
                <div className="autocomplete-item no-results">
                  <span>No instruments found for &quot;{searchTerm}&quot;</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Selected Symbols - Integrated into controls */}
        {selectedSymbols.length > 0 && (
          <div className="selected-symbols-inline">
            <span className="selected-label">Selected:</span>
            <div className="symbol-tags">
              {selectedSymbols.map((symbol, index) => {
                // Find instrument name for better display
                const instrument = instruments.find(i => i.symbol === symbol);
                const displayName = instrument?.name || symbol;
                return (
                  <span
                    key={symbol}
                    className="symbol-tag"
                    style={{ backgroundColor: colors[index % colors.length] }}
                    title={displayName}
                  >
                    {symbol}
                    <button
                      onClick={() => removeSymbol(symbol)}
                      className="remove-symbol"
                      aria-label={`Remove ${symbol}`}
                      type="button"
                    >
                      ×
                    </button>
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Controls Row 2: Metric and Range */}
      <div className="chart-controls-row">
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

        <div className="range-selector">
          <label>Range:</label>
          <div className="range-buttons">
            {DAYS_OPTIONS.map((option) => {
              const isSelected = days === option.value;
              return (
                <button
                  key={option.label}
                  type="button"
                  className={`range-btn ${isSelected ? 'active' : ''}`}
                  onClick={() => setDays(option.value)}
                  aria-label={`Select ${option.label} range`}
                >
                  {option.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className={`chart-wrapper ${loading ? 'loading' : ''}`}>
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
              <YAxis tickFormatter={getValueFormatter(selectedMetric)} />
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
          <div className="no-data">No {getMetricDisplayName(selectedMetric).toLowerCase()} data available for selected stocks</div>
        )}

        {!loading && !error && selectedSymbols.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">📈</div>
            <h3>No stocks selected</h3>
            <p>Search and add stocks above to view their chart</p>
          </div>
        )}
      </div>

      {/* Per-symbol summary row */}
      {!loading && !error && chartData.length > 0 && selectedSymbols.length > 0 && renderSummary()}
    </div>
  );
};

export default Chart;
