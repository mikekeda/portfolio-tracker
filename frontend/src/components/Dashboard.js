import React, { useState, useEffect } from 'react';
import { portfolioAPI } from '../services/api';
import TopMovers from './TopMovers';
import PortfolioChart from './PortfolioChart';
import './Dashboard.css';

const Dashboard = () => {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const summaryData = await portfolioAPI.getSummary();

        setSummary(summaryData);
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

  if (!summary) {
    return (
      <div className="dashboard-container">
        <div className="error">No portfolio data available</div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>Portfolio Summary</h1>
        {summary.last_updated && (
          <div className="last-updated">Last updated: {new Date(summary.last_updated).toLocaleDateString()}</div>
        )}
      </div>

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
          <h3>Positions</h3>
          <p className="value">
            {summary.total_holdings}
            {typeof summary.profitable_holdings === 'number' && typeof summary.losing_holdings === 'number' && (
              <span className="positions-breakdown"> (
                <span className="pos">{summary.profitable_holdings}</span> / <span className="neg">{summary.losing_holdings}</span>
              )</span>
            )}
          </p>
        </div>
      </div>

      {/* Portfolio Analytics Charts */}
      <div className="portfolio-charts-section">
        <PortfolioChart />
      </div>

      {/* Top Movers Section */}
      <div className="top-movers-section">
        <TopMovers />
      </div>

    </div>
  );
};

export default Dashboard;
