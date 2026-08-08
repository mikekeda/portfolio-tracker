import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { nativePrice } from '../utils/currency';
import './TopMovers.css';

const TopMovers = ({ selectedPeriod, setSelectedPeriod, movers, error }) => {
  const periods = [
    { value: '1d', label: '1 Day' },
    { value: '1w', label: '1 Week' },
    { value: '1m', label: '1 Month' },
    { value: '90d', label: '90 Days' },
    { value: 'ytd', label: 'YTD' }
  ];

  const loading = !movers && !error;
  const data = useMemo(() => {
    if (!movers) return null;
    const limit = 10;
    return {
      movers,
      gainers: movers.slice(0, limit),
      losers: movers.slice(-limit).reverse(), // Reverse to show biggest losers first
      total_symbols: movers.length,
    };
  }, [movers]);

  const formatChange = (change) => {
    const isPositive = change >= 0;
    return (
      <span className={`change ${isPositive ? 'positive' : 'negative'}`}>
        {isPositive ? '+' : ''}{change.toFixed(2)}%
      </span>
    );
  };

  const formatGain = (gain) => {
    if (gain === null || gain === undefined) return '-';
    const isPositive = gain >= 0;
    return (
      <span className={`change ${isPositive ? 'positive' : 'negative'}`}>
        {isPositive ? '+' : ''}{gain.toFixed(2)}%
      </span>
    );
  };

  const formatPrice = (price, currency) => {
    const { symbol, value } = nativePrice(price, currency);
    return `${symbol}${value.toFixed(2)}`;
  };

  if (loading) {
    return (
      <div className="top-movers-container">
        <div className="loading">Loading top movers...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="top-movers-container">
        <div className="error">{error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="top-movers-container">
        <div className="error">No data available</div>
      </div>
    );
  }

  return (
    <div className="top-movers-container">
      <div className="top-movers-header">
        <h3>Top Movers</h3>
        <div className="period-selector">
          {periods.map(period => (
            <button
              key={period.value}
              className={`period-btn ${selectedPeriod === period.value ? 'active' : ''}`}
              onClick={() => setSelectedPeriod && setSelectedPeriod(period.value)}
            >
              {period.label}
            </button>
          ))}
        </div>
      </div>

      <div className="movers-tables">
        {/* Top Gainers */}
        <div className="movers-table">
          <h4 className="gainers-title">Top Gainers</h4>
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Name</th>
                <th>Gain</th>
                <th>Price</th>
                <th>Change</th>
              </tr>
            </thead>
            <tbody>
              {data.gainers && data.gainers.length > 0 ? (
                data.gainers.map((stock) => (
                  <tr key={stock.symbol}>
                    <td className="symbol">
                      <Link className="symbol" to={`/stock/${encodeURIComponent(stock.symbol)}`}>{stock.symbol}</Link>
                    </td>
                    <td className="name" title={stock.name}>
                      {stock.name.length > 25 ? `${stock.name.substring(0, 25)}...` : stock.name}
                    </td>
                    <td
                      className={`change-cell ${stock.gain_pct < 0 ? 'neg' : 'pos'}`}
                      style={{ '--bar-width': `${Math.min(Math.abs(stock.gain_pct || 0), 100)}%` }}
                    >
                      <span>{formatGain(stock.gain_pct)}</span>
                    </td>
                    <td className="price">{formatPrice(stock.current_price, stock.currency)}</td>
                    <td
                      className={`change-cell ${stock.change_pct < 0 ? 'neg' : 'pos'}`}
                      style={{ '--bar-width': `${Math.min(Math.abs(stock.change_pct), 100)}%` }}
                    >
                      <span>{formatChange(stock.change_pct)}</span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" className="no-data">No gainers data available</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Top Losers */}
        <div className="movers-table">
          <h4 className="losers-title">Top Losers</h4>
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Name</th>
                <th>Gain</th>
                <th>Price</th>
                <th>Change</th>
              </tr>
            </thead>
            <tbody>
              {data.losers && data.losers.length > 0 ? (
                data.losers.map((stock) => (
                  <tr key={stock.symbol}>
                    <td className="symbol">
                      <Link className="symbol" to={`/stock/${encodeURIComponent(stock.symbol)}`}>{stock.symbol}</Link>
                    </td>
                    <td className="name" title={stock.name}>
                      {stock.name.length > 25 ? `${stock.name.substring(0, 25)}...` : stock.name}
                    </td>
                    <td
                      className={`change-cell ${stock.gain_pct < 0 ? 'neg' : 'pos'}`}
                      style={{ '--bar-width': `${Math.min(Math.abs(stock.gain_pct || 0), 100)}%` }}
                    >
                      <span>{formatGain(stock.gain_pct)}</span>
                    </td>
                    <td className="price">{formatPrice(stock.current_price, stock.currency)}</td>
                    <td
                      className={`change-cell ${stock.change_pct < 0 ? 'neg' : 'pos'}`}
                      style={{ '--bar-width': `${Math.min(Math.abs(stock.change_pct), 100)}%` }}
                    >
                      <span>{formatChange(stock.change_pct)}</span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" className="no-data">No losers data available</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {(data.total_symbols && data.total_symbols > 0) ? (
        <div className="movers-footer">
          <small>Based on {data.total_symbols} symbols</small>
        </div>
      ) : null}
    </div>
  );
};

TopMovers.propTypes = {
  selectedPeriod: PropTypes.string,
  setSelectedPeriod: PropTypes.func,
  movers: PropTypes.arrayOf(PropTypes.object),
  error: PropTypes.string,
};

TopMovers.defaultProps = {
  selectedPeriod: '1d',
  setSelectedPeriod: undefined,
  movers: null,
  error: null,
};

export default TopMovers;
