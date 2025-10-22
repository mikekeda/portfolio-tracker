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

// Helper function to get Sortino Ratio color
const getSortinoColor = (sortino) => {
  if (sortino < 1.0) return 'negative';
  if (sortino > 2.0) return 'positive';
  return '';
};

// Helper function to get Beta color
const getBetaColor = (beta) => {
  if (beta < 0.9) return 'negative';
  if (beta > 1.3) return 'positive';
  return '';
};

// Helper function to generate Sortino Ratio tooltip
const getSortinoTooltip = (sortino) => {
  let recommendation = '';
  let level = '';

  if (sortino < 1.0) {
    level = 'Poor';
    recommendation = 'Review your holdings: Your stock selections may be underperforming or exhibiting too much unrewarded downside risk. Use your screener and ROIC analysis to identify the weakest companies in your portfolio—those with declining fundamentals or poor screener scores.\n\nAction: Consider trimming or selling the laggards and reallocating the capital to your high-conviction, high-ROIC "compounder" stocks. The goal is to increase your portfolio\'s overall quality and return potential.';
  } else if (sortino <= 2.0) {
    level = 'Acceptable';
    recommendation = 'Your portfolio shows acceptable downside protection. Continue monitoring and consider optimizing your holdings for better risk-adjusted returns.';
  } else {
    level = 'Excellent';
    recommendation = 'Excellent downside protection! Your portfolio is well-positioned to handle market volatility while maintaining strong returns.';
  }

  return `Sortino Ratio: ${sortino.toFixed(2)} (${level})\n\n${recommendation}\n\nScale: < 1.0 (Poor), 1.0-2.0 (Acceptable), > 2.0 (Excellent)`;
};

// Helper function to generate Beta tooltip
const getBetaTooltip = (beta) => {
  let recommendation = '';
  let level = '';

  if (beta < 0.9) {
    level = 'Low';
    recommendation = 'Review your allocations: A low Beta suggests your portfolio is not positioned aggressively enough to meet your growth objectives. You may be overly diversified or have too much invested in lower-volatility sectors.\n\nAction: Check your sector and country allocations. If you are underweight in growth sectors like Technology or have a high allocation to a broad-market ETF like VUSA, consider increasing your exposure to individual growth stocks to increase your portfolio\'s market sensitivity.';
  } else if (beta <= 1.3) {
    level = 'Acceptable';
    recommendation = 'Your portfolio shows acceptable market sensitivity. Continue monitoring and consider optimizing your allocations for better growth potential.';
  } else {
    level = 'High';
    recommendation = 'High market sensitivity detected. Consider diversifying your portfolio to reduce volatility and improve risk-adjusted returns.';
  }

  return `Beta: ${beta.toFixed(2)} (${level})\n\n${recommendation}\n\nScale: < 0.9 (Low), 0.9-1.3 (Acceptable), > 1.3 (High)`;
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

  return `Fear & Greed: ${value.toFixed(1)} (${label})\n\n${recommendation}\n\nScale: 0-25 (Extreme Fear), 25-45 (Fear), 45-55 (Neutral), 55-75 (Greed), 75-100 (Extreme Greed)`;
};

// Helper function to get MWRR color
const getMwrrColor = (mwrr) => {
  if (mwrr < 0) return 'negative';
  if (mwrr > 10) return 'positive';
  return '';
};

// Helper function to get TWRR color
const getTwrrColor = (twrr) => {
  if (twrr < 0) return 'negative';
  if (twrr > 10) return 'positive';
  return '';
};

// Helper function to generate MWRR tooltip
const getMwrrTooltip = (mwrr) => {
  let recommendation = '';
  let level = '';

  if (mwrr < 0) {
    level = 'Negative';
    recommendation = 'Your portfolio is losing money overall. Review your investment strategy and consider rebalancing.';
  } else if (mwrr < 5) {
    level = 'Low';
    recommendation = 'Low returns detected. Consider reviewing your stock selection and diversification strategy.';
  } else if (mwrr < 15) {
    level = 'Moderate';
    recommendation = 'Decent returns. Continue monitoring your portfolio and consider optimizing your holdings.';
  } else {
    level = 'Strong';
    recommendation = 'Excellent returns! Your investment strategy is performing well.';
  }

  return `Money-Weighted Return: ${mwrr.toFixed(2)}% (${level})\n\n${recommendation}\n\nThis measures the return you've actually earned on your money, considering when you invested it.`;
};

// Helper function to generate TWRR tooltip
const getTwrrTooltip = (twrr) => {
  let recommendation = '';
  let level = '';

  if (twrr < 0) {
    level = 'Negative';
    recommendation = 'Your investment strategy is underperforming. Review your stock selection and consider using better screeners.';
  } else if (twrr < 5) {
    level = 'Low';
    recommendation = 'Low strategy returns. Consider improving your stock selection process and using quality screeners.';
  } else if (twrr < 15) {
    level = 'Moderate';
    recommendation = 'Decent strategy performance. Continue monitoring and consider optimizing your stock picks.';
  } else {
    level = 'Strong';
    recommendation = 'Excellent strategy performance! Your stock selection is working well.';
  }

  return `Time-Weighted Return: ${twrr.toFixed(2)}% (${level})\n\n${recommendation}\n\nThis measures the pure performance of your investment strategy, ignoring when you added/withdrew money.`;
};

// Helper function to get Yield Spread color
const getYieldSpreadColor = (yieldSpread) => {
  if (yieldSpread < 0) return 'negative'; // Inverted yield curve
  if (yieldSpread > 1.5) return 'positive'; // Healthy spread
  return '';
};

// Helper function to generate Yield Spread tooltip
const getYieldSpreadTooltip = (yieldSpread) => {
  let recommendation = '';
  let level = '';

  if (yieldSpread < 0) {
    level = 'Inverted';
    recommendation = 'Yield curve is inverted - recession warning signal. Consider defensive positioning and quality stocks.';
  } else if (yieldSpread < 0.5) {
    level = 'Flat';
    recommendation = 'Yield curve is flat - economic uncertainty. Focus on high-quality, defensive stocks.';
  } else if (yieldSpread < 1.5) {
    level = 'Normal';
    recommendation = 'Normal yield curve - economic growth expected. Balanced portfolio approach is appropriate.';
  } else {
    level = 'Steep';
    recommendation = 'Steep yield curve - strong economic growth expected. Consider growth-oriented investments.';
  }

  return `10Y-2Y Yield Spread: ${yieldSpread.toFixed(2)}% (${level})\n\n${recommendation}\n\nThis measures the difference between 10-year and 2-year Treasury yields, indicating economic outlook.`;
};

// Helper function to get Buffett Indicator color
const getBuffettIndicatorColor = (buffettIndicator) => {
  if (buffettIndicator > 150) return 'negative'; // Overvalued
  if (buffettIndicator < 75) return 'positive'; // Undervalued
  return '';
};

// Helper function to generate Buffett Indicator tooltip
const getBuffettIndicatorTooltip = (buffettIndicator) => {
  let recommendation = '';
  let level = '';

  if (buffettIndicator > 150) {
    level = 'Overvalued';
    recommendation = 'Market appears overvalued relative to GDP. Consider defensive positioning and value stocks.';
  } else if (buffettIndicator > 120) {
    level = 'Expensive';
    recommendation = 'Market is expensive. Focus on quality stocks and consider reducing risk exposure.';
  } else if (buffettIndicator < 75) {
    level = 'Undervalued';
    recommendation = 'Market appears undervalued. Good time for long-term investments and growth stocks.';
  } else {
    level = 'Fair Value';
    recommendation = 'Market is fairly valued. Balanced approach with quality stock selection is appropriate.';
  }

  return `Buffett Indicator: ${buffettIndicator.toFixed(1)}% (${level})\n\n${recommendation}\n\nThis measures total market cap as % of GDP, indicating overall market valuation.`;
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
          <h3>Total Profit</h3>
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
        {summary.mwrr && (
          <div className="card" title={getMwrrTooltip(summary.mwrr)}>
            <h3>Money-Weighted RR</h3>
            <p className={`value ${getMwrrColor(summary.mwrr)}`}>
              {summary.mwrr.toFixed(2)}%
            </p>
          </div>
        )}
        {summary.twrr && (
          <div className="card" title={getTwrrTooltip(summary.twrr)}>
            <h3>Time-Weighted RR</h3>
            <p className={`value ${getTwrrColor(summary.twrr)}`}>
              {summary.twrr.toFixed(2)}%
            </p>
          </div>
        )}
        {summary.sortino_ratio && (
          <div className="card" title={getSortinoTooltip(summary.sortino_ratio)}>
            <h3>Sortino</h3>
            <p className={`value ${getSortinoColor(summary.sortino_ratio)}`}>
              {summary.sortino_ratio.toFixed(2)}
            </p>
          </div>
        )}
        {summary.beta && (
          <div className="card" title={getBetaTooltip(summary.beta)}>
            <h3>Beta</h3>
            <p className={`value ${getBetaColor(summary.beta)}`}>
              {summary.beta.toFixed(2)}
            </p>
          </div>
        )}
        {summary.buffett_indicator !== null && summary.buffett_indicator !== undefined && (
          <div className="card" title={getBuffettIndicatorTooltip(summary.buffett_indicator)}>
            <h3>Buffett Indicator</h3>
            <a
              href="https://currentmarketvaluation.com/models/buffett-indicator.php"
              target="_blank"
              rel="noopener noreferrer"
              className="value-link"
            >
              <p className={`value ${getBuffettIndicatorColor(summary.buffett_indicator)}`}>
                {summary.buffett_indicator.toFixed(1)}%
              </p>
            </a>
          </div>
        )}
        {summary.yield_spread !== null && summary.yield_spread !== undefined && (
          <div className="card" title={getYieldSpreadTooltip(summary.yield_spread)}>
            <h3>Yield Spread</h3>
            <a
              href="https://fred.stlouisfed.org/series/T10Y2Y"
              target="_blank"
              rel="noopener noreferrer"
              className="value-link"
            >
              <p className={`value ${getYieldSpreadColor(summary.yield_spread)}`}>
                {summary.yield_spread.toFixed(2)}%
              </p>
            </a>
          </div>
        )}
        {summary.fear_greed_index && (
          <div className="card" title={getFearGreedTooltip(summary.fear_greed_index)}>
            <h3>Fear & Greed</h3>
            <a
              href="https://edition.cnn.com/markets/fear-and-greed"
              target="_blank"
              rel="noopener noreferrer"
              className="value-link"
            >
              <p className={`value ${getFearGreedColor(summary.fear_greed_index.label)}`}>
                {summary.fear_greed_index.value.toFixed(1)}
              </p>
            </a>
          </div>
        )}
        {summary.vix && (
          <div className="card" title={getVixTooltip(summary.vix)}>
            <h3>VIX</h3>
            <a
              href="https://markets.businessinsider.com/index/vix"
              target="_blank"
              rel="noopener noreferrer"
              className="value-link"
            >
              <p className="value">{summary.vix.toFixed(2)}</p>
            </a>
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
