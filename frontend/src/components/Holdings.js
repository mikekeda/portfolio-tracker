import React, { useState, useEffect, useMemo } from 'react';
import { portfolioAPI } from '../services/api';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  createColumnHelper,
} from '@tanstack/react-table';
import { renderCountryWithFlag } from '../utils/countryUtils';
import { calculateBarWidth, getBarColorScheme, calculateMinMax, getBarStyle, shouldBeNegativeBar } from '../utils/barUtils';
import { getAvailableScreeners } from '../services/screeners';
import './Holdings.css';

const Holdings = () => {
  const [holdings, setHoldings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [globalFilter, setGlobalFilter] = useState('');
  const [sorting, setSorting] = useState([]);
  const [selectedScreeners, setSelectedScreeners] = useState([]);
  const [availableScreeners, setAvailableScreeners] = useState([]);
  const [screenersLoading, setScreenersLoading] = useState(true);

  const columnHelper = createColumnHelper();

  // Calculate min/max values for bar columns
  const barRanges = useMemo(() => {
    if (!holdings.length) return {};

    return {
      portfolioPct: calculateMinMax(holdings, 'portfolio_pct'),
      marketValue: calculateMinMax(holdings, 'market_value'),
      institutional: calculateMinMax(holdings, 'institutional_ownership'),
      short: calculateMinMax(holdings, 'short_percent_of_float'),
      weekHighChange: calculateMinMax(holdings, 'fifty_two_week_high_distance')
    };
  }, [holdings]);

  // Use screener results from backend (no client-side calculation needed)
  const holdingsWithScreeners = useMemo(() => {
    return holdings; // Backend already includes passedScreeners field
  }, [holdings]);

  const columns = useMemo(
    () => [
      columnHelper.accessor('yahoo_symbol', {
        header: 'Symbol',
        cell: (info) => (
          <span className="symbol">{info.getValue() || info.row.original.t212_code}</span>
        ),
        enableSorting: true,
        enableGlobalFilter: true,
        size: 80,
      }),
      columnHelper.accessor('name', {
        header: 'Name',
        cell: (info) => {
          const row = info.row.original || {};
          const businessSummary = row.business_summary;
          const tooltip = businessSummary ? businessSummary : info.getValue();
          return (
            <span className="name" title={tooltip}>
              {info.getValue()}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: true,
        size: 200,
      }),
      columnHelper.accessor('portfolio_pct', {
        header: '%',
        cell: (info) => {
          const value = info.getValue();
          const barWidth = calculateBarWidth(value, barRanges.portfolioPct?.min || 0, barRanges.portfolioPct?.max || 100);
          const colorScheme = getBarColorScheme('percentage', value);
          const barStyle = getBarStyle(barWidth, colorScheme);

          return (
            <span className="bar-column" style={barStyle}>
              <span>{value?.toFixed(2) || ''}</span>
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 40,
      }),
      columnHelper.accessor('market_value', {
        header: 'Value (£)',
        cell: (info) => {
          const value = info.getValue();
          const barWidth = calculateBarWidth(value, barRanges.marketValue?.min || 0, barRanges.marketValue?.max || 100000);
          const colorScheme = getBarColorScheme('value', value);
          const barStyle = getBarStyle(barWidth, colorScheme);

          return (
            <span className="bar-column" style={barStyle}>
              <span>
                {value.toLocaleString(undefined, {
                  minimumFractionDigits: 0,
                  maximumFractionDigits: 0
                })}
              </span>
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 100,
      }),
      columnHelper.accessor('profit', {
        header: 'Profit (£)',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="pnl"></span>;
          return (
            <span className={`pnl ${value >= 0 ? 'positive' : 'negative'}`}>
              {value.toLocaleString(undefined, {
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
              })}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      columnHelper.accessor('return_pct', {
        header: 'Return',
        cell: (info) => {
          const value = info.getValue();
          return (
            <span className={`return ${value >= 0 ? 'positive' : 'negative'}`}>
              {value >= 0 ? '+' : ''}{value.toFixed(1)}%
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      // Dividend yield column (from backend field dividend_yield)
      columnHelper.accessor('dividend_yield', {
        header: 'Dividend',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="dividend"></span>;
          return (
            <span className="dividend">{value.toFixed(2)}%</span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 70,
      }),
      columnHelper.accessor('prediction', {
        header: 'Prediction',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="prediction"></span>;
          const isPositive = value > 20;
          const isNegative = value < 0;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`prediction ${className}`}>{value >= 0 ? '+' : ''}{Math.round(value * 100) / 100}%</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      columnHelper.accessor('institutional_ownership', {
        header: 'Instit',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="institutional"></span>;

          const barWidth = calculateBarWidth(value, barRanges.institutional?.min || 0, barRanges.institutional?.max || 100);
          const colorScheme = getBarColorScheme('institutional', value);
          const isNegative = shouldBeNegativeBar('institutional', value);
          const barStyle = getBarStyle(barWidth, colorScheme);

          // Apply original text color logic
          const isPositive = value > 80;
          const isNegativeText = value < 40;
          const textClassName = isPositive ? 'positive' : isNegativeText ? 'negative' : '';

          return (
            <span className={`bar-column ${isNegative ? 'negative' : ''}`} style={barStyle}>
              <span className={`institutional ${textClassName}`}>{Math.round(value * 100) / 100}%</span>
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 40,
      }),
      columnHelper.accessor('market_cap', {
        header: 'Market Cap',
        cell: (info) => {
          const value = info.getValue();
          if (!value) return <span className="market-cap"></span>;
          const billions = value / 1000000000;
          return <span className="market-cap">{billions.toFixed(1)}</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      columnHelper.accessor('peg_ratio', {
        header: 'PEG',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="peg"></span>;
          const isPositive = value < 1.5;
          const isNegative = value > 3.0;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`peg ${className}`}>{value}</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('pe_ratio', {
        header: 'PE',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="pe"></span>;
          const isPositive = value < 30;
          const isNegative = value > 100;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`pe ${className}`}>{Math.round(value)}</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 50,
      }),
      columnHelper.accessor('beta', {
        header: 'Beta',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="beta"></span>;
          const isPositive = value < 1;
          const isNegative = value > 2;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`beta ${className}`}>{value}</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('profit_margins', {
        header: 'Margins',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="margins"></span>;
          const isPositive = value > 30;
          const isNegative = value < 10;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`margins ${className}`}>{value >= 0 ? '+' : ''}{Math.round(value * 100) / 100}%</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      columnHelper.accessor('revenue_growth', {
        header: 'Growth',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="growth"></span>;
          const isPositive = value > 40;
          const isNegative = value < 15;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`growth ${className}`}>{value >= 0 ? '+' : ''}{Math.round(value * 100) / 100}%</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      columnHelper.accessor('return_on_assets', {
        header: 'ROA',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="roa"></span>;
          const isPositive = value > 10;
          const isNegative = value < 2;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`roa ${className}`}>{value >= 0 ? '+' : ''}{Math.round(value * 100) / 100}%</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('free_cashflow_yield', {
        header: 'FCF Yield',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="fcf-yield"></span>;
          const isPositive = value > 6;
          const isNegative = value < 2;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`fcf-yield ${className}`}>{value >= 0 ? '+' : ''}{Math.round(value * 100) / 100}%</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      columnHelper.accessor('recommendation_mean', {
        header: 'Rec',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="recommendation"></span>;

          // Read extra fields from the row for tooltip
          const row = info.row.original || {};
          const key = row.recommendation_key; // e.g., buy/hold/sell
          const opinions = row.number_of_analyst_opinions; // count

          const isPositive = value < 1.5;
          const isNegative = value > 2.5;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';

          const tooltip = [
            key ? `Recommendation: ${key}` : null,
            opinions !== undefined && opinions !== null ? `Analyst opinions: ${opinions}` : null,
          ].filter(Boolean).join('\n');

          return (
            <span className={`recommendation ${className}`} title={tooltip}>
              {value}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('fifty_two_week_high_distance', {
        header: '52WH Change',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="week-high-change"></span>;

          const barWidth = calculateBarWidth(Math.abs(value), 0, Math.max(Math.abs(barRanges.weekHighChange?.min || 0), Math.abs(barRanges.weekHighChange?.max || 100)));
          const colorScheme = getBarColorScheme('weekHighChange', value);
          const isNegative = shouldBeNegativeBar('weekHighChange', value);
          const barStyle = getBarStyle(barWidth, colorScheme);

          // Apply original text color logic
          const isPositive = value > 0;
          const isNegativeText = value < -20;
          const textClassName = isPositive ? 'positive' : isNegativeText ? 'negative' : '';

          return (
            <span className={`bar-column ${isNegative ? 'negative' : ''}`} style={barStyle}>
              <span className={`week-high-change ${textClassName}`}>
                {value >= 0 ? '+' : ''}{value.toFixed(0)}%
              </span>
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      columnHelper.accessor('short_percent_of_float', {
        header: 'Short',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="short"></span>;

          const barWidth = calculateBarWidth(value, barRanges.short?.min || 0, barRanges.short?.max || 100);
          const colorScheme = getBarColorScheme('short', value);
          const isNegative = shouldBeNegativeBar('short', value);
          const barStyle = getBarStyle(barWidth, colorScheme);

          // Apply original text color logic
          const isPositive = value < 0;
          const isNegativeText = value > 20;
          const textClassName = isPositive ? 'positive' : isNegativeText ? 'negative' : '';

          return (
            <span className={`bar-column ${isNegative ? 'negative' : ''}`} style={barStyle}>
              <span className={`short ${textClassName}`}>{Math.round(value * 100) / 100}%</span>
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 40,
      }),
      columnHelper.accessor('rsi', {
        header: 'RSI',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="rsi"></span>;

          // Apply text color logic for RSI
          const isOverbought = value > 70;
          const isOversold = value < 30;
          const textClassName = isOverbought ? 'negative' : isOversold ? 'positive' : '';

          return (
            <span className={`rsi ${textClassName}`}>{value}</span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 50,
      }),
      columnHelper.accessor('screener_score', {
        header: 'Score',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="screener-score"></span>;

          // Color coding based on score ranges
          let className = '';
          if (value >= 8) className = 'excellent';
          else if (value >= 6) className = 'good';
          else if (value >= 4) className = 'average';
          else if (value >= 2) className = 'poor';
          else className = 'very-poor';

          return (
            <span className={`screener-score ${className}`}>
              {value.toFixed(1)}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('passedScreeners', {
        header: 'Screeners',
        cell: (info) => {
          const passedScreeners = info.getValue();
          if (!passedScreeners || passedScreeners.length === 0) {
            return <span className="no-screeners"></span>;
          }

          return (
            <div className="screener-badges">
              {passedScreeners.map((screenerId) => {
                const screener = availableScreeners.find(s => s.id === screenerId);
                if (!screener) return null;

                const isActive = selectedScreeners.includes(screenerId);
                const criteriaText = screener.criteria.map(c => {
                  const value = c.value && c.value.fieldRef ? c.value.fieldRef : c.value;
                  return `${c.field.replace(/_/g, ' ')} ${c.operator} ${value}`;
                }).join(' & ');

                // Create combine with text
                const combineWithText = screener.combine_with && screener.combine_with.length > 0
                  ? `\n\nRecommended to combine with: ${screener.combine_with.map(id => {
                      const combinedScreener = availableScreeners.find(s => s.id === id);
                      return combinedScreener ? combinedScreener.name : id;
                    }).join(', ')}`
                  : '';

                return (
                  <button
                    key={screenerId}
                    className={`screener-badge clickable category-${screener.category} ${isActive ? 'active' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleScreenerChange(screenerId);
                    }}
                    title={`${screener.description}\n\nCriteria: ${criteriaText}\n\nCategory: ${screener.category}\nWeight: ${screener.weight || 5}/10${combineWithText}`}
                  >
                    {screener.name}
                  </button>
                );
              })}
            </div>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 120,
        sortingFn: (rowA, rowB) => {
          const screenersA = rowA.original.passedScreeners || [];
          const screenersB = rowB.original.passedScreeners || [];
          return screenersA.length - screenersB.length;
        },
      }),
      columnHelper.accessor('country', {
        header: 'Country',
        cell: (info) => {
          const countryName = info.getValue();

          return (
            <span className="country" title={countryName || ''}>
              {renderCountryWithFlag(countryName)}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: true,
        size: 120,
      }),
      columnHelper.accessor('sector', {
        header: 'Sector',
        cell: (info) => (
          <span className="sector" title={info.getValue() || ''}>
            {info.getValue() || ''}
          </span>
        ),
        enableSorting: true,
        enableGlobalFilter: true,
        size: 140,
      }),
      columnHelper.accessor('quote_type', {
        header: 'Type',
        cell: (info) => (
          <span className="quote-type">{info.getValue() || ''}</span>
        ),
        enableSorting: true,
        enableGlobalFilter: true,
        size: 80,
      }),
      columnHelper.accessor('currency', {
        header: 'Currency',
        cell: (info) => (
          <span>{info.getValue()}</span>
        ),
        enableSorting: true,
        enableGlobalFilter: true,
        size: 70,
      }),
      columnHelper.accessor('current_price', {
        header: 'Price (£)',
        cell: (info) => {
          const row = info.row.original || {};
          const tooltip = [
            row.sma_20 !== undefined && row.sma_20 !== null ? `SMA20: ${Number(row.sma_20).toFixed(2)}` : null,
            row.sma_50 !== undefined && row.sma_50 !== null ? `SMA50: ${Number(row.sma_50).toFixed(2)}` : null,
            row.sma_200 !== undefined && row.sma_200 !== null ? `SMA200: ${Number(row.sma_200).toFixed(2)}` : null,
          ].filter(Boolean).join('\n');
          return (
            <span title={tooltip || undefined}>{info.getValue().toFixed(2)}</span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
    ],
    [columnHelper, barRanges]
  );

  useEffect(() => {
    const fetchHoldings = async () => {
      try {
        setLoading(true);
        const data = await portfolioAPI.getCurrentHoldings();
        setHoldings(data.holdings || []);
        setError(null);
      } catch (err) {
        setError('Failed to fetch holdings data');
        console.error('Error fetching holdings:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchHoldings();
  }, []);

  // Fetch available screeners
  useEffect(() => {
    const fetchScreeners = async () => {
      try {
        setScreenersLoading(true);
        const screeners = await getAvailableScreeners();
        setAvailableScreeners(screeners);
      } catch (err) {
        console.error('Error fetching screeners:', err);
        // Set empty array as fallback
        setAvailableScreeners([]);
      } finally {
        setScreenersLoading(false);
      }
    };

    fetchScreeners();
  }, []);

  // Handle screener filter change (multi-select functionality)
  const handleScreenerChange = (screenerId) => {
    setSelectedScreeners(prev => {
      if (screenerId === '') {
        return []; // Clear all screeners
      }
      if (prev.includes(screenerId)) {
        return prev.filter(id => id !== screenerId); // Remove screener
      } else {
        return [...prev, screenerId]; // Add screener
      }
    });
  };

  // Calculate screener counts considering active screeners
  const screenerCounts = useMemo(() => {
    if (!holdingsWithScreeners.length || !availableScreeners.length) {
      return {};
    }

    // Calculate counts for each screener considering AND logic with active screeners
    const counts = {};
    availableScreeners.forEach(screener => {
      if (selectedScreeners.length === 0) {
        // No active screeners: count all holdings that pass this screener
        counts[screener.id] = holdingsWithScreeners.filter(holding =>
          holding.passedScreeners && holding.passedScreeners.includes(screener.id)
        ).length;
      } else {
        // Active screeners: count holdings that pass this screener AND all active screeners
        counts[screener.id] = holdingsWithScreeners.filter(holding =>
          holding.passedScreeners &&
          holding.passedScreeners.includes(screener.id) &&
          selectedScreeners.every(activeScreenerId =>
            holding.passedScreeners.includes(activeScreenerId)
          )
        ).length;
      }
    });
    return counts;
  }, [holdingsWithScreeners, availableScreeners, selectedScreeners]);

  // Get filtered holdings based on screener selection
  const filteredHoldings = useMemo(() => {
    // If no screeners are selected, show all holdings
    if (selectedScreeners.length === 0) {
      return holdingsWithScreeners;
    }

    // If screeners are selected, filter holdings locally using passedScreeners (AND logic)
    return holdingsWithScreeners.filter(holding =>
      holding.passedScreeners &&
      selectedScreeners.every(activeScreenerId =>
        holding.passedScreeners.includes(activeScreenerId)
      )
    );
  }, [holdingsWithScreeners, selectedScreeners]);

  const table = useReactTable({
    data: filteredHoldings,
    columns,
    state: {
      sorting,
      globalFilter,
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: 'includesString',
    initialState: {
      sorting: [{ id: 'market_value', desc: true }], // Sort by market value descending by default
    },
    columnResizeMode: 'onChange',
  });

  if (loading) {
    return (
      <div className="holdings-container">
        <div className="loading">Loading holdings...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="holdings-container">
        <div className="error">{error}</div>
      </div>
    );
  }

  return (
    <div className="holdings-container">
      <h2>Current Holdings</h2>

      {/* Search and Filter Controls */}
      <div className="table-controls">
        <div className="search-box">
          <input
            type="text"
            placeholder="Search by symbol, name, sector, country, or currency..."
            value={globalFilter ?? ''}
            onChange={(e) => setGlobalFilter(e.target.value)}
            className="search-input"
          />
        </div>

        <div className="screener-badges-section">
          <label>Available Screeners:</label>
          <div className="screener-badges-container">
            {screenersLoading ? (
              <span className="loading-indicator">Loading screeners...</span>
            ) : (
              <>
                <button
                  className={`screener-badge all-holdings ${selectedScreeners.length === 0 ? 'active' : ''}`}
                  onClick={() => handleScreenerChange('')}
                  title="Show all holdings"
                >
                  <span className="screener-name">All Holdings</span>
                  <span className="screener-count">({holdingsWithScreeners.length})</span>
                </button>
                {availableScreeners
                  .sort((a, b) => (b.weight || 0) - (a.weight || 0)) // Sort by weight (higher first)
                  .map((screener) => {
                  const count = screenerCounts[screener.id] || 0;
                  const isActive = selectedScreeners.includes(screener.id);
                  const criteriaText = screener.criteria.map(c => {
                    const value = c.value && c.value.fieldRef ? c.value.fieldRef : c.value;
                    return `${c.field.replace(/_/g, ' ')} ${c.operator} ${value}`;
                  }).join(' & ');

                  // Create combine with text
                  const combineWithText = screener.combine_with && screener.combine_with.length > 0
                    ? `\n\nRecommended to combine with: ${screener.combine_with.map(id => {
                        const combinedScreener = availableScreeners.find(s => s.id === id);
                        return combinedScreener ? combinedScreener.name : id;
                      }).join(', ')}`
                    : '';

                  return (
                    <button
                      key={screener.id}
                      className={`screener-badge category-${screener.category} ${isActive ? 'active' : ''}`}
                      onClick={() => handleScreenerChange(screener.id)}
                      title={`${screener.description}\n\nCriteria: ${criteriaText}\n\nCategory: ${screener.category}\nWeight: ${screener.weight || 5}/10${combineWithText}`}
                    >
                      <span className="screener-name">{screener.name}</span>
                      <span className="screener-count">({count})</span>
                    </button>
                  );
                })}
              </>
            )}
          </div>
        </div>

        <div className="table-info">
          <span>
            Showing {table.getFilteredRowModel().rows.length} of {holdings.length} holdings
            {selectedScreeners.length > 0 && (
              <span className="filter-status">
                {' '}(filtered by {selectedScreeners.length === 1
                  ? availableScreeners.find(s => s.id === selectedScreeners[0])?.name
                  : `${selectedScreeners.length} screeners`})
              </span>
            )}
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="holdings-table-container">
        <table className="holdings-table">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className={header.column.getCanSort() ? 'sortable' : ''}
                  >
                    <div className="header-content">
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                      {header.column.getCanSort() && (
                        <span className="sort-indicator">
                          {{
                            asc: ' 🔼',
                            desc: ' 🔽',
                          }[header.column.getIsSorted()] ?? ' ↕️'}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <tr key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={columns.length} className="no-results">
                  {selectedScreeners.length > 0 ? (
                    <div className="no-screener-results">
                      <p>No holdings match the selected screener{selectedScreeners.length > 1 ? 's' : ''} criteria.</p>
                      <p>Try selecting different screener{selectedScreeners.length > 1 ? 's' : ''} or clear the filter to see all holdings.</p>
                    </div>
                  ) : (
                    <div className="no-holdings">
                      <p>No holdings found.</p>
                    </div>
                  )}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      <div className="holdings-summary">
        <p>Total Holdings: {holdings.length}</p>
        <p>Filtered Holdings: {table.getFilteredRowModel().rows.length}</p>
        <p>Last Updated: {holdings.length > 0 ? new Date(holdings[0].date).toLocaleDateString() : ''}</p>
      </div>
    </div>
  );
};

export default Holdings;
