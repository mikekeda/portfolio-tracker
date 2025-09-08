import React, { useState, useEffect } from 'react';
import { portfolioAPI } from '../services/api';
import { renderCountryWithFlag } from '../utils/countryUtils';
import TopMovers from './TopMovers';
import './Dashboard.css';

const Dashboard = () => {
  const [summary, setSummary] = useState(null);
  const [allocations, setAllocations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [summaryData, allocationsData] = await Promise.all([
          portfolioAPI.getSummary(),
          portfolioAPI.getAllocations()
        ]);

        setSummary(summaryData);
        setAllocations(allocationsData);
        setError(null);
      } catch (err) {
        setError('Failed to fetch portfolio data');
        console.error('Error fetching data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading">Loading portfolio data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-container">
        <div className="error">{error}</div>
      </div>
    );
  }

  if (!summary || !allocations) {
    return (
      <div className="dashboard-container">
        <div className="error">No portfolio data available</div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <h1>Portfolio Dashboard</h1>

      {/* Portfolio Summary Cards */}
      <div className="summary-cards">
        <div className="card">
          <h3>Total Value</h3>
          <p className="value">£{summary.total_value.toLocaleString()}</p>
        </div>
        <div className="card">
          <h3>Total Profit/Loss</h3>
          <p className={`value ${summary.total_profit >= 0 ? 'positive' : 'negative'}`}>
            £{summary.total_profit.toLocaleString()}
          </p>
        </div>
        <div className="card">
          <h3>Total Return</h3>
          <p className={`value ${summary.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
            {summary.total_return_pct >= 0 ? '+' : ''}{summary.total_return_pct.toFixed(2)}%
          </p>
        </div>
        <div className="card">
          <h3>Total Holdings</h3>
          <p className="value">{summary.total_holdings}</p>
        </div>
        <div className="card">
          <h3>Profitable Positions</h3>
          <p className="value positive">{summary.profitable_holdings}</p>
        </div>
        <div className="card">
          <h3>Losing Positions</h3>
          <p className="value negative">{summary.losing_holdings}</p>
        </div>
        <div className="card">
          <h3>Win Rate</h3>
          <p className="value">{summary.win_rate.toFixed(1)}%</p>
        </div>
        <div className="card">
          <h3>Last Updated</h3>
          <p className="value">{summary.last_updated ? new Date(summary.last_updated).toLocaleDateString() : 'N/A'}</p>
        </div>
      </div>

      {/* Top Movers Section */}
      <div className="top-movers-section">
        <TopMovers />
      </div>

      {/* Allocations */}
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
    </div>
  );
};

export default Dashboard;
