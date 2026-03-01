import React, { useState, useEffect, useMemo, useCallback } from 'react';
import PropTypes from 'prop-types';
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
import { getPsThresholds } from '../utils/valuationUtils';
import { calculateBarWidth, getBarColorScheme, calculateMinMax, getBarStyle, shouldBeNegativeBar } from '../utils/barUtils';
import { getAvailableScreeners } from '../services/screeners';
import './Holdings.css';

const POSITION_ONLY_COLUMN_IDS = ['portfolio_pct', 'market_value', 'profit', 'return_pct'];


const columnHelper = createColumnHelper();

// --- 13F helper functions (module-level, no closure dependencies) ---

const HOLDER_NAME_SUFFIXES = [
  ' Fund Management', ' Family Office', ' Management', ' Holdings', ' Capital', ' Group',
];

function trimHolderName(name) {
  let trimmed = name;
  for (const s of HOLDER_NAME_SUFFIXES) {
    if (trimmed.endsWith(s)) {
      trimmed = trimmed.slice(0, -s.length);
      break;
    }
  }
  if (trimmed.length > 14) {
    const first = trimmed.split(/[\s,]/)[0];
    return first.length > 14 ? first.slice(0, 12) + '…' : first;
  }
  return trimmed;
}

function formatHolderValue(val) {
  if (val == null) return null;
  const n = Number(val);
  if (n >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${Math.round(n)}`;
}

function formatHolderShares(val) {
  if (val == null) return null;
  const n = Number(val);
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString();
}

function buildHolderTooltip(h) {
  const lines = [h.name, `Change: ${h.change}`];
  if (h.report_date) lines.push(`Report date: ${h.report_date}`);
  if (h.shares != null && h.shares_prev != null) {
    lines.push(`Shares: ${formatHolderShares(h.shares_prev)} → ${formatHolderShares(h.shares)}`);
  } else if (h.shares != null) {
    lines.push(`Shares: ${formatHolderShares(h.shares)}`);
  }
  if (h.value != null) {
    const valueStr = formatHolderValue(h.value);
    const reason = h.scored === false && h.score_reason ? ` — not scored (${h.score_reason})` : '';
    lines.push(`Value: ${valueStr}${reason}`);
  }
  return lines.join('\n');
}

/**
 * Maps a change string to a sort score matching the backend signal bands.
 * New=+2, Increase=+1, Stable=0, Trim=-1, Closed=-2.
 */
function scoreFromChange(change) {
  if (change === 'New') return 2;
  if (change === 'Closed') return -2;
  if (change === '—') return 0;
  const match = change.match(/^([+-]?\d+(?:\.\d+)?)%$/);
  if (!match) return 0;
  const pct = parseFloat(match[1]);
  if (pct >= 1000) return 2;   // effective new
  if (pct >= 10) return 1;     // increase
  if (pct >= -30) return 0;    // stable
  if (pct >= -90) return -1;   // trim
  return -2;                    // effective liquidation
}

/**
 * Composite sort score for earnings signal + conviction.
 * Ascending = most actionable positive signals first.
 * For positive signals (buy/consider) high conviction ranks higher (lower score).
 * For negative signals (avoid) high conviction ranks lower (higher score) — more certain to avoid.
 */
const SIGNAL_CONVICTION_SCORE = {
  'buy:high': 0,  'buy:medium': 1,  'buy:low': 2,  'buy:': 3,
  'consider:high': 4,  'consider:medium': 5,  'consider:low': 6,  'consider:': 7,
  'hold:high': 8,  'hold:medium': 9,  'hold:low': 10,  'hold:': 11,
  'avoid:low': 12,  'avoid:medium': 13,  'avoid:high': 14,  'avoid:': 13,
};

function signalScore(row) {
  const key = `${row.earnings_signal || ''}:${row.earnings_conviction || ''}`;
  return SIGNAL_CONVICTION_SCORE[key] ?? 99;
}

// Net institutional momentum: sum of change scores across all 13F holders.
function netHolderScore(holders) {
  return (holders || []).reduce((sum, h) => sum + scoreFromChange(h.change), 0);
}

function getHolderCategory(change) {
  if (change === 'New') return 'form13f-new';
  if (change === 'Closed') return 'form13f-closed';
  if (change === '—') return 'form13f-stable';
  const match = change.match(/^([+-]?\d+(?:\.\d+)?)%$/);
  if (match) {
    const pct = parseFloat(match[1]);
    if (pct >= 10) return 'form13f-increase';
    if (pct <= -30) return 'form13f-trimmed';
    return 'form13f-stable';
  }
  return 'form13f-stable';
}

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

HoldingRow.displayName = 'HoldingRow';
HoldingRow.propTypes = {
  row: PropTypes.shape({
    getVisibleCells: PropTypes.func.isRequired,
  }).isRequired,
  isSelected: PropTypes.bool.isRequired,
};

const Holdings = () => {
  const [showAll, setShowAll] = useState(false);
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
    () => {
      const cols = [
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
          if (value === null || value === undefined) return <span className="return">—</span>;
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

          const sector = info.row.original?.sector || '';
          const thresholds = getPsThresholds(sector);

          let className = '';
          if (thresholds !== null) {
            if (value < thresholds[0]) className = 'positive';
            else if (value > thresholds[1]) className = 'negative';
          }

          const tooltipSuffix = thresholds
            ? ` · Green < ${thresholds[0]}, Red > ${thresholds[1]}${sector ? ` (${sector})` : ''}`
            : ` · Not color-coded for ${sector || 'this'} sector`;

          return (
            <span className={`ps ${className}`} title={`Price-to-Sales: ${value.toFixed(2)}${tooltipSuffix}`}>
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
      columnHelper.accessor('form13f_score', {
        header: '13F Score',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="form13f-score"></span>;

          let className = '';
          if (value >= 1.5) className = 'excellent';
          else if (value >= 0.5) className = 'good';
          else if (value >= -0.5) className = 'average';
          else if (value >= -1.5) className = 'poor';
          else className = 'very-poor';

          return (
            <span className={`form13f-score ${className}`}>
              {value}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('form13f_holders', {
        header: '13F Holders',
        cell: (info) => {
          const holders = info.getValue();
          if (!holders || holders.length === 0) {
            return <span className="no-form13f-holders"></span>;
          }

          const sortedHolders = [...holders].sort((a, b) => scoreFromChange(b.change) - scoreFromChange(a.change));

          return (
            <div className="form13f-holder-badges">
              {sortedHolders.map((h, idx) => {
                const content = <>{h.change} {trimHolderName(h.name)}</>;
                const scored = h.scored !== false; // default true for legacy data
                const cls = [
                  'form13f-holder-badge',
                  getHolderCategory(h.change),
                  !scored ? 'form13f-holder-badge--noise' : '',
                ].filter(Boolean).join(' ');
                const tip = buildHolderTooltip(h);
                return h.manager_id ? (
                  <Link
                    key={`${h.name}-${idx}`}
                    to={`/13f/${h.manager_id}`}
                    className={`${cls} form13f-holder-badge--link`}
                    title={tip}
                  >
                    {content}
                  </Link>
                ) : (
                  <span
                    key={`${h.name}-${idx}`}
                    className={cls}
                    title={tip}
                  >
                    {content}
                  </span>
                );
              })}
            </div>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 180,
        sortingFn: (rowA, rowB) =>
          netHolderScore(rowA.original.form13f_holders) - netHolderScore(rowB.original.form13f_holders),
      }),
      columnHelper.accessor('earnings_signal', {
        header: 'Signal',
        cell: (info) => {
          const signal = info.getValue();
          const conviction = info.row.original.earnings_conviction;
          const annDate = info.row.original.earnings_announcement_date;
          if (!signal) return <span className="earnings-signal-cell" />;
          const label = signal.charAt(0).toUpperCase() + signal.slice(1);
          const tip = [
            label,
            conviction ? `${conviction} conviction` : null,
            annDate ? `Announced ${annDate}` : null,
          ].filter(Boolean).join(' · ');
          return (
            <span className={`earnings-signal-badge es-${signal}`} title={tip}>
              {label}
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
        sortingFn: (rowA, rowB) => {
          return signalScore(rowA.original) - signalScore(rowB.original);
        },
      }),
      columnHelper.accessor('since_earnings_pct', {
        header: 'Since Earn.',
        cell: (info) => {
          const pct = info.getValue();
          const annDate = info.row.original.earnings_announcement_date;
          if (pct == null) return <span className="since-earnings-cell" />;
          const positive = pct >= 0;
          const tip = annDate ? `Price change since earnings announced ${annDate}` : 'Price change since earnings announcement';
          return (
            <span
              className={`since-earnings ${positive ? 'positive' : 'negative'}`}
              title={tip}
            >
              {positive ? '+' : ''}{pct.toFixed(1)}%
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
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
    ];
      return showAll ? cols.filter(col => !POSITION_ONLY_COLUMN_IDS.includes(col.id)) : cols;
    },
    [columnHelper, barRanges, quickRatioThresholds, selectedStocks, toggleSelectAll, toggleStockSelection, availableScreeners, selectedScreeners, showAll]
  );

  useEffect(() => {
    const fetchHoldings = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await portfolioAPI.getCurrentHoldings(showAll);
        setHoldings(data.holdings || []);
        setQuickRatioThresholds(data.quick_ratio_thresholds || {});
      } catch (err) {
        const msg = err.response?.data?.detail;
        setError(
          typeof msg === 'string' ? msg : Array.isArray(msg) ? msg.map((e) => e.msg || JSON.stringify(e)).join('; ') : 'Failed to fetch holdings data'
        );
        console.error('Error fetching holdings:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchHoldings();
  }, [showAll]);

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
      sorting: showAll ? [{ id: 'name', desc: false }] : [{ id: 'market_value', desc: true }],
    },
    columnResizeMode: 'onChange',
  });

  // CSV Export Function - Must be defined before any conditional returns
  const exportToCSV = useCallback(() => {
    if (!table) return;
    // Get the filtered/visible rows
    const rowsToExport = table.getFilteredRowModel().rows;

    if (rowsToExport.length === 0) {
      alert('No data to export');
      return;
    }

    // Define CSV headers - map all available fields to readable names
    const headerMap = {
      'yahoo_symbol': 'Symbol',
      'name': 'Name',
      'portfolio_pct': '%',
      'market_value': 'Value (£)',
      'profit': 'Profit (£)',
      'return_pct': 'Return',
      'dividend_yield': 'Dividend',
      'prediction': 'Prediction',
      'institutional_ownership': 'Instit',
      'market_cap': 'Market Cap',
      'peg_ratio': 'PEG',
      'pe_ratio': 'PE',
      'ps_ratio': 'PS',
      'beta': 'Beta',
      'profit_margins': 'Margins',
      'revenue_growth': 'Growth',
      'roic': 'ROIC',
      'free_cashflow_yield': 'FCF Yield',
      'quickRatio': 'Quick',
      'debtToEquity': 'D/E',
      'recommendation_mean': 'Rec',
      'recommendation_trend': 'Rec Trend',
      'fifty_two_week_high_distance': '52WH Change',
      'short_percent_of_float': 'Short',
      'rsi': 'RSI',
      'screener_score': 'Score',
      'passedScreeners': 'Screeners',
      'form13f_score': '13F Score',
      'form13f_holders': '13F Holders',
      'earnings_signal': 'Signal',
      'since_earnings_pct': 'Since Earn.',
      'country': 'Country',
      'sector': 'Sector',
      'quote_type': 'Type',
      'currency': 'Currency',
      'dcf_diff': 'DCF Diff',
      'current_price': 'Price',
    };

    // Get all available keys from the data (for columns not shown in UI)
    const allAvailableKeys = new Set();
    rowsToExport.forEach(row => {
      Object.keys(row.original).forEach(key => {
        if (!key.startsWith('_') && key !== 'analyst_price_targets') {
          allAvailableKeys.add(key);
        }
      });
    });

    // Get visible column order from table (matching the page display order)
    // Exclude 'select' column (checkbox) and 'date' column
    const visibleColumns = table.getAllColumns().filter(col => {
      const colDef = col.columnDef;
      // Skip select column (checkbox)
      if (colDef.id === 'select') return false;
      // Skip date column (it's in the filename)
      if (colDef.accessorKey === 'date') return false;
      // Only include columns with accessorKey (data columns)
      return colDef.accessorKey !== undefined;
    });

    // Extract visible columns first (in page order)
    const keysToExport = [];
    const headers = [];
    const visibleKeys = new Set();

    visibleColumns.forEach(col => {
      const colDef = col.columnDef;
      const accessorKey = colDef.accessorKey;

      if (!accessorKey) return;

      // Get header from column definition (all headers are strings)
      const headerText = typeof colDef.header === 'string'
        ? colDef.header
        : headerMap[accessorKey] || accessorKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

      keysToExport.push(accessorKey);
      headers.push(headerText);
      visibleKeys.add(accessorKey);
    });

    // Add remaining columns (not shown in UI) after visible columns, alphabetically
    const remainingKeys = Array.from(allAvailableKeys).filter(key =>
      !visibleKeys.has(key) &&
      key !== 'date' // Skip date column
    ).sort();

    remainingKeys.forEach(key => {
      keysToExport.push(key);
      headers.push(headerMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
    });

    // Helper function to format CSV value
    const formatCSVValue = (value, key) => {
      if (value === null || value === undefined) {
        return '';
      }

      // Handle arrays (like passedScreeners, form13f_holders)
      if (Array.isArray(value)) {
        if (key === 'form13f_holders') {
          return value.map(h => `${h.name}: ${h.change}`).join('; ');
        }
        return value.join('; ');
      }

      // Handle objects - skip complex objects for now
      if (typeof value === 'object') {
        return '';
      }

      // Handle numbers - format appropriately
      if (typeof value === 'number') {
        // For percentages and ratios, format with 2 decimals
        const percentKeys = ['portfolio_pct', 'return_pct', 'dividend_yield', 'prediction',
                            'institutional_ownership', 'profit_margins', 'revenue_growth',
                            'roic', 'free_cashflow_yield', 'fifty_two_week_high_distance',
                            'short_percent_of_float', 'dcf_diff'];
        if (percentKeys.includes(key)) {
          return value.toFixed(2);
        }
        return value.toString();
      }

      // Handle strings - escape commas and quotes
      let stringValue = String(value);
      if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
        stringValue = `"${stringValue.replace(/"/g, '""')}"`;
      }
      return stringValue;
    };

    // Extract data for each row
    const csvRows = rowsToExport.map(row => {
      return keysToExport.map(key => formatCSVValue(row.original[key], key));
    });

    // Combine headers and rows
    const csvContent = [
      headers.join(','),
      ...csvRows.map(row => row.join(','))
    ].join('\n');

    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    // Generate filename with timestamp
    const timestamp = new Date().toISOString().split('T')[0];
    const filename = `holdings_export_${timestamp}.csv`;

    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [table]);

  return (
    <div className="holdings-container">
      <h2>{showAll ? 'All instruments' : 'Current Holdings'}</h2>

      {error && (
        <div className="holdings-error" role="alert">
          {error}
        </div>
      )}

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
        <div className="holdings-actions">
          <label className="holdings-show-all">
            <span className="holdings-show-all-label">Show all</span>
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} aria-label="Show all monitored instruments" />
            <span className="holdings-show-all-slider" />
          </label>
          <button
            onClick={exportToCSV}
            className="btn-export"
            title="Export filtered instruments to CSV"
            aria-label="Export filtered instruments to CSV"
            disabled={table.getFilteredRowModel().rows.length === 0}
          >
            <svg className="export-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <path d="M8 11L8 3M8 11L5.5 8.5M8 11L10.5 8.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2.5 12.5H13.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
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
                  title="Clear screener filters"
                >
                  <span className="screener-name">All</span>
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
            Showing {table.getFilteredRowModel().rows.length} of {holdings.length} instruments
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
        {loading ? (
          <div className="loading">Loading instruments...</div>
        ) : (
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
                      <p>No instruments match the selected screener{selectedScreeners.length > 1 ? 's' : ''} criteria.</p>
                      <p>Try selecting different screener{selectedScreeners.length > 1 ? 's' : ''} or clear the filter to see all instruments.</p>
                    </div>
                  ) : (
                    <div className="no-holdings">
                      <p>No instruments found.</p>
                    </div>
                  )}
                </td>
              </tr>
            )}
          </tbody>
        </table>
        )}
      </div>

      {/* Summary */}
      <div className="holdings-summary">
        <p>Total instruments: {holdings.length}</p>
        <p>Filtered instruments: {table.getFilteredRowModel().rows.length}</p>
        <p>Last Updated: {holdings.length > 0 ? new Date(holdings[0].date).toLocaleDateString() : ''}</p>
      </div>
    </div>
  );
};

export default Holdings;
