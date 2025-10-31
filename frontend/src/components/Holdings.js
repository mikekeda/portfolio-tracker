import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Link } from 'react-router-dom';
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

// This component will only re-render if its own props change
const HoldingRow = React.memo(({ row, isSelected }) => {
  return (
    <tr className={isSelected ? 'selected-row' : ''}>
      {row.getVisibleCells().map((cell) => (
        <td key={cell.id}>
          {flexRender(cell.column.columnDef.cell, cell.getContext())}
        </td>
      ))}
    </tr>
  );
});

const Holdings = () => {
  const [holdings, setHoldings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [globalFilter, setGlobalFilter] = useState('');
  const [sorting, setSorting] = useState([]);
  const [selectedScreeners, setSelectedScreeners] = useState([]);
  const [availableScreeners, setAvailableScreeners] = useState([]);
  const [screenersLoading, setScreenersLoading] = useState(true);
  const [quickRatioThresholds, setQuickRatioThresholds] = useState({});
  const [selectedStocks, setSelectedStocks] = useState(new Set());
  const [showOnlySelected, setShowOnlySelected] = useState(false);

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

  // Selection handlers (wrapped in useCallback for performance)
  const toggleStockSelection = useCallback((symbol) => {
    setSelectedStocks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(symbol)) {
        newSet.delete(symbol);
      } else {
        newSet.add(symbol);
      }
      return newSet;
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedStocks(new Set());
    setShowOnlySelected(false);
  }, []);

  const toggleShowSelected = useCallback(() => {
    setShowOnlySelected(prev => !prev);
  }, []);

  // Get filtered holdings based on screener selection and stock selection
  const filteredHoldings = useMemo(() => {
    let result = holdingsWithScreeners;

    // Apply screener filter
    if (selectedScreeners.length > 0) {
      result = result.filter(holding =>
        holding.passedScreeners &&
        selectedScreeners.every(activeScreenerId =>
          holding.passedScreeners.includes(activeScreenerId)
        )
      );
    }

    // Apply selection filter - only when actually showing selected stocks
    if (showOnlySelected && selectedStocks.size > 0) {
      result = result.filter(h => selectedStocks.has(h.yahoo_symbol || h.t212_code));
    }

    return result;
  }, [holdingsWithScreeners, selectedScreeners, showOnlySelected, selectedStocks]);

  // Toggle select all based on currently filtered/visible rows
  const toggleSelectAll = useCallback(() => {
    // Get symbols from filteredHoldings at call time (not dependency time)
    const visibleSymbols = filteredHoldings.map(h => h.yahoo_symbol || h.t212_code);

    setSelectedStocks(prev => {
      const allSelected = visibleSymbols.length > 0 &&
                          visibleSymbols.every(symbol => prev.has(symbol));
      const newSet = new Set(prev);

      if (allSelected) {
        // Deselect all visible rows
        visibleSymbols.forEach(symbol => newSet.delete(symbol));
      } else {
        // Select all visible rows
        visibleSymbols.forEach(symbol => newSet.add(symbol));
      }

      return newSet;
    });
  }, [filteredHoldings]);

  const columns = useMemo(
    () => [
      columnHelper.display({
        id: 'select',
        header: ({ table }) => {
          const visibleRows = table.getRowModel().rows;
          const visibleSymbols = visibleRows.map(row =>
            row.original.yahoo_symbol || row.original.t212_code
          );
          const allSelected = visibleSymbols.length > 0 &&
                              visibleSymbols.every(symbol => selectedStocks.has(symbol));

          return (
            <input
              type="checkbox"
              checked={allSelected}
              onChange={toggleSelectAll}
              title="Select/Deselect all visible rows"
            />
          );
        },
        cell: (info) => {
          const symbol = info.row.original.yahoo_symbol || info.row.original.t212_code;
          const isChecked = selectedStocks.has(symbol);
          return (
            <input
              type="checkbox"
              checked={isChecked}
              onChange={() => toggleStockSelection(symbol)}
            />
          );
        },
        size: 30,
      }),
      columnHelper.accessor('yahoo_symbol', {
        header: 'Symbol',
        cell: (info) => {
          const symbol = info.getValue() || info.row.original.t212_code;
          return (
            <Link className="symbol" to={`/stock/${encodeURIComponent(symbol)}`}>{symbol}</Link>
          );
        },
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
              {value >= 0 ? '+' : ''}{Math.round(value)}%
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
          return <span className={`prediction ${className}`}>{value >= 0 ? '+' : ''}{Math.round(value)}%</span>;
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
              <span className={`institutional ${textClassName}`}>{Math.round(value)}%</span>
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
          return <span className={`peg ${className}`}>{Math.round(value * 100) / 100}</span>;
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
          const avgPe = info.row?.original?.avg_pe;
          let className = '';
          if (avgPe !== null && avgPe !== undefined) {
            if (value < avgPe) className = 'positive';
            else if (value > avgPe) className = 'negative';
          }
          const title = avgPe !== null && avgPe !== undefined ? `Avg PE: ${Math.round(avgPe)}` : undefined;
          return <span className={`pe ${className}`} title={title}>{Math.round(value)}</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 50,
      }),
      columnHelper.accessor('ps_ratio', {
        header: 'PS',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="ps"></span>;

          let className = '';
          if (value < 1.5) {
            className = 'positive'; // Green for PS < 1.5
          } else if (value > 3) {
            className = 'negative'; // Red for PS > 3
          }

          return (
            <span className={`ps ${className}`} title={`Price-to-Sales Ratio: ${value.toFixed(2)}`}>
              {value.toFixed(1)}
            </span>
          );
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
          return <span className={`margins ${className}`}>{value >= 0 ? '+' : ''}{Math.round(value)}%</span>;
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
          return <span className={`growth ${className}`}>{value >= 0 ? '+' : ''}{Math.round(value)}%</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      columnHelper.accessor('roic', {
        header: 'ROIC',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="roic"></span>;

          let className = '';
          if (value > 20) {
            className = 'positive'; // Green for ROIC > 20%
          } else if (value < 10) {
            className = 'negative'; // Red for ROIC < 10%
          }

          return (
            <span className={`roic ${className}`} title={`Return on Invested Capital: ${value.toFixed(2)}%`}>
              {value >= 0 ? '+' : ''}{Math.round(value)}%
            </span>
          );
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
      columnHelper.accessor('quickRatio', {
        header: 'Quick',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="quick-ratio"></span>;

          // Get sector-specific thresholds from the row data
          const row = info.row.original;
          const sector = row.sector || 'Other';
          const thresholds = quickRatioThresholds[sector] || quickRatioThresholds['Other'];

          let className = '';
          if (value >= thresholds[1]) {
            className = 'positive'; // Green for Quick Ratio >= green threshold
          } else if (value < thresholds[0]) {
            className = 'negative'; // Red for Quick Ratio < red threshold
          }

          return (
            <span className={`quick-ratio ${className}`} title={`Quick Ratio: ${value.toFixed(2)} (${sector} sector)\nThresholds: Red < ${thresholds[0]}, Green ≥ ${thresholds[1]}`}>
              {value.toFixed(1)}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('debtToEquity', {
        header: 'D/E',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="debt-equity"></span>;

          let className = '';
          if (value <= 50) {
            className = 'positive'; // Green for D/E <= 30% (low debt)
          } else if (value > 100) {
            className = 'negative'; // Red for D/E > 100% (high debt)
          }

          return (
            <span className={`debt-equity ${className}`} title={`Debt-to-Equity: ${value.toFixed(2)}\nGreen: ≤ 50% (Low debt), Red: > 100% (High debt)`}>
              {Math.round(value)}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
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
      columnHelper.accessor('recommendation_trend', {
        header: 'Rec Trend',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="recommendation-trend"></span>;

          // Determine color and direction based on trend
          let className = '';
          let interpretation = '';

          if (value > 0.1) {
            className = 'positive';
            interpretation = 'Improving';
          } else if (value < -0.1) {
            className = 'negative';
            interpretation = 'Declining';
          } else {
            className = '';
            interpretation = 'Stable';
          }

          const tooltip = `Trend: ${interpretation}\nValue: ${value.toFixed(3)}\nRange: -1.0 (declining) to +1.0 (improving)`;

          return (
            <span className={`recommendation-trend ${className}`} title={tooltip}>
              {value.toFixed(2)}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
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
              <span className={`short ${textClassName}`}>{Math.round(value)}%</span>
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
            <span className={`rsi ${textClassName}`}>{Math.round(value)}</span>
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
              {value}
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
      columnHelper.accessor('dcf_diff', {
        header: 'DCF Diff',
        cell: (info) => {
          const row = info.row.original || {};
          const dcfDiff = row.dcf_diff;
          const dcfPrice = row.dcf_price;

          if (dcfDiff === null || dcfDiff === undefined) {
            return <span className="dcf-diff"></span>;
          }

          // Convert from decimal (1 = 100%) to percentage
          const potentialProfitPct = dcfDiff * 100;
          const isProfit = potentialProfitPct > 0; // Positive = profit potential
          const isLoss = potentialProfitPct < 0; // Negative = loss potential

          const className = isProfit ? 'positive' : isLoss ? 'negative' : '';

          return (
            <span className={`dcf-diff ${className}`} title={dcfPrice ? `DCF: ${dcfPrice.toFixed(2)}` : undefined}>
              {potentialProfitPct >= 0 ? '+' : ''}{Math.round(potentialProfitPct)}%
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
      columnHelper.accessor('current_price', {
        header: 'Price',
        cell: (info) => {
          const row = info.row.original || {};
          const targets = row.analyst_price_targets || {};
          const currentPrice = info.getValue();

          const tooltip = [
            // Price targets (if available)
            targets.high !== undefined && targets.high !== null ? `High: ${Number(targets.high).toFixed(2)}` : null,
            targets.median !== undefined && targets.median !== null ? `Median: ${Number(targets.median).toFixed(2)}` : null,
            targets.mean !== undefined && targets.mean !== null ? `Mean: ${Number(targets.mean).toFixed(2)}` : null,
            targets.low !== undefined && targets.low !== null ? `Low: ${Number(targets.low).toFixed(2)}` : null,
            row.number_of_analyst_opinions !== undefined && row.number_of_analyst_opinions !== null ? `Analysts: ${Number(row.number_of_analyst_opinions)}` : null,
          ].filter(Boolean).join('\n');

          // Simple text coloring based on price targets
          let textColor = '';
          if (Object.keys(targets).length > 0) {
            const { low, high } = targets;
            if (low !== undefined && low !== null && currentPrice < low) {
              textColor = '#28a745'; // Green: below low
            } else if (high !== undefined && high !== null && currentPrice > high) {
              textColor = '#dc3545'; // Red: above high
            }
            // Default color for everything else
          }

          return (
            <span style={{ color: textColor }} title={tooltip || undefined}>
              {currentPrice.toFixed(2)}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
    ],
    [columnHelper, barRanges, quickRatioThresholds, selectedStocks, toggleSelectAll, toggleStockSelection, availableScreeners, selectedScreeners]
  );

  useEffect(() => {
    const fetchHoldings = async () => {
      try {
        setLoading(true);
        const data = await portfolioAPI.getCurrentHoldings();
        setHoldings(data.holdings || []);
        setQuickRatioThresholds(data.quick_ratio_thresholds || {});
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
        <div className="search-screeners-row">
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

        {/* Comparison Controls */}
        {selectedStocks.size > 0 && (
          <div className="comparison-controls">
            <div className="comparison-buttons">
              <button
                className={`btn-compare ${showOnlySelected ? 'active' : ''}`}
                onClick={toggleShowSelected}
                title={showOnlySelected ? 'Show all holdings' : 'Show only selected stocks'}
              >
                {showOnlySelected ? '📋 Show All' : '🔍 Show Only Selected'}
              </button>
              <button
                className="btn-clear"
                onClick={clearSelection}
                title="Clear selection"
              >
                ✕ Clear Selection
              </button>
            </div>
            <div className="selection-info">
              <span className="selection-count">
                {selectedStocks.size} stock{selectedStocks.size !== 1 ? 's' : ''} selected
              </span>
            </div>
          </div>
        )}
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
              table.getRowModel().rows.map((row) => {
                const symbol = row.original.yahoo_symbol || row.original.t212_code;
                const isSelected = selectedStocks.has(symbol);
                return (
                  <HoldingRow
                    key={row.id}
                    row={row}
                    isSelected={isSelected}
                  />
                );
              })
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
