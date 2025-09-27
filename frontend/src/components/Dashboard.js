import React, { useState, useEffect } from 'react';
import { portfolioAPI } from '../services/api';
import TopMovers from './TopMovers';
import PortfolioChart from './PortfolioChart';
import './Dashboard.css';

// Helper function to get Fear & Greed color from backend label
const getFearGreedColor = (label) => {
  switch (label.toLowerCase()) {
    case 'extreme fear': return 'extreme-fear';
    case 'fear': return 'fear';
    case 'neutral': return 'neutral';
    case 'greed': return 'greed';
    case 'extreme greed': return 'extreme-greed';
    default: return 'neutral';
  }
};

// Helper function to generate VIX tooltip
const getVixTooltip = (vix) => {
  let recommendation = '';
  let level = '';
  
  if (vix < 15) {
    level = 'Low Volatility';
    recommendation = 'Market complacency - consider hedging or reducing risk';
  } else if (vix < 25) {
    level = 'Normal Volatility';
    recommendation = 'Normal market conditions - standard risk management';
  } else if (vix < 35) {
    level = 'Elevated Volatility';
    recommendation = 'Increased market stress - be cautious with new positions';
  } else {
    level = 'High Volatility';
    recommendation = 'Market panic - potential buying opportunity for contrarians';
  }
  
  return `VIX: ${vix.toFixed(2)} (${level})\n\n${recommendation}\n\nScale: 0-15 (Low), 15-25 (Normal), 25-35 (Elevated), 35+ (High)`;
};

// Helper function to generate Fear & Greed tooltip
const getFearGreedTooltip = (fearGreed) => {
  const { value, label } = fearGreed;
  let recommendation = '';
  
  switch (label.toLowerCase()) {
    case 'extreme fear':
      recommendation = '🎯 CONTRARIAN BUY SIGNAL - Market oversold, potential buying opportunity';
      break;
    case 'fear':
      recommendation = '⚠️ CAUTION - Market stress, consider defensive positions';
      break;
    case 'neutral':
      recommendation = '📊 NEUTRAL - Standard market conditions, normal risk management';
      break;
    case 'greed':
      recommendation = '⚠️ CAUTION - Market optimism, consider taking some profits';
      break;
    case 'extreme greed':
      recommendation = '🚨 SELL SIGNAL - Market euphoria, time to be very careful';
      break;
    default:
      recommendation = '📊 Market sentiment indicator';
  }
  
  return `Fear & Greed: ${value.toFixed(1)} (${label})\n\n${recommendation}\n\nScale: 0-25 (Extreme Fear), 25-40 (Fear), 40-60 (Neutral), 60-75 (Greed), 75-100 (Extreme Greed)`;
};

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
        {summary.vix && (
          <div className="card" title={getVixTooltip(summary.vix)}>
            <h3>VIX</h3>
            <p className="value">{summary.vix.toFixed(2)}</p>
          </div>
        )}
        {summary.fear_greed_index && (
          <div className="card" title={getFearGreedTooltip(summary.fear_greed_index)}>
            <h3>Fear & Greed</h3>
            <p className={`value ${getFearGreedColor(summary.fear_greed_index.label)}`}>
              {summary.fear_greed_index.value.toFixed(1)}
            </p>
          </div>
        )}
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
