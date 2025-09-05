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
import './Holdings.css';

const Holdings = () => {
  const [holdings, setHoldings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [globalFilter, setGlobalFilter] = useState('');
  const [sorting, setSorting] = useState([]);

  const columnHelper = createColumnHelper();

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
        cell: (info) => (
          <span className="name" title={info.getValue()}>
            {info.getValue()}
          </span>
        ),
        enableSorting: true,
        enableGlobalFilter: true,
        size: 200,
      }),
      columnHelper.accessor('portfolio_pct', {
        header: '%',
        cell: (info) => (
          <span className="portfolio-pct">{info.getValue()?.toFixed(2) || ''}</span>
        ),
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('market_value', {
        header: 'Value (£)',
        cell: (info) => (
          <span>
            {info.getValue().toLocaleString(undefined, {
              minimumFractionDigits: 0,
              maximumFractionDigits: 0
            })}
          </span>
        ),
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
      columnHelper.accessor('prediction', {
        header: 'Prediction',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="prediction"></span>;
          const isPositive = value > 20;
          const isNegative = value < 0;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`prediction ${className}`}>{value >= 0 ? '+' : ''}{value}%</span>;
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
          const isPositive = value > 80;
          const isNegative = value < 40;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`institutional ${className}`}>{value}%</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
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
          return <span className={`margins ${className}`}>{value >= 0 ? '+' : ''}{value}%</span>;
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
          return <span className={`growth ${className}`}>{value >= 0 ? '+' : ''}{value}%</span>;
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
          return <span className={`roa ${className}`}>{value >= 0 ? '+' : ''}{value}%</span>;
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
          return <span className={`fcf-yield ${className}`}>{value >= 0 ? '+' : ''}{value}%</span>;
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
          const isPositive = value < 1.5;
          const isNegative = value > 2.5;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`recommendation ${className}`}>{value}</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('fifty_two_week_change', {
        header: '52W Change',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="week-change"></span>;
          const isPositive = value > 0;
          const isNegative = value < -20;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return (
            <span className={`week-change ${className}`}>
              {value >= 0 ? '+' : ''}{value.toFixed(0)}%
            </span>
          );
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 100,
      }),
      columnHelper.accessor('short_percent_of_float', {
        header: 'Short',
        cell: (info) => {
          const value = info.getValue();
          if (value === null || value === undefined) return <span className="short"></span>;
          const isPositive = value < 0;
          const isNegative = value > 20;
          const className = isPositive ? 'positive' : isNegative ? 'negative' : '';
          return <span className={`short ${className}`}>{value}%</span>;
        },
        enableSorting: true,
        enableGlobalFilter: false,
        size: 60,
      }),
      columnHelper.accessor('rsi', {
        header: 'RSI',
        cell: (info) => (
          <span className="rsi">{info.getValue() || ''}</span>
        ),
        enableSorting: true,
        enableGlobalFilter: false,
        size: 50,
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
        cell: (info) => (
          <span>{info.getValue().toFixed(2)}</span>
        ),
        enableSorting: true,
        enableGlobalFilter: false,
        size: 80,
      }),
    ],
    [columnHelper]
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

  const table = useReactTable({
    data: holdings,
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

        <div className="table-info">
          <span>
            Showing {table.getFilteredRowModel().rows.length} of {holdings.length} holdings
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
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
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
