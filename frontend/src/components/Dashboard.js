import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { portfolioAPI } from '../services/api';
import { useHideAmounts, MASK } from '../context/HideAmountsContext';
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

// Helper function to get VIX color
const getVIXColor = (vix) => {
  if (vix > 25) return 'positive';
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

// Helper function to get Alpha color
const getAlphaColor = (alpha) => {
  if (alpha < 0) return 'negative';
  if (alpha > 0) return 'positive';
  return '';
};

// Helper function to generate Alpha tooltip
const getAlphaTooltip = (alpha) => {
  let recommendation = '';
  let level = '';

  if (alpha < -5) {
    level = 'Significant Underperformance';
    recommendation = 'Your active stock picking is significantly dragging down returns compared to a passive S&P 500 ETF. Consider reevaluating your strategy or moving to index funds.';
  } else if (alpha < 0) {
    level = 'Underperformance';
    recommendation = 'Your portfolio is slightly underperforming the market on a risk-adjusted basis. Review your holdings.';
  } else if (alpha < 5) {
    level = 'Outperformance';
    recommendation = 'Good job! You are generating positive excess returns above the market benchmark.';
  } else {
    level = 'Significant Outperformance';
    recommendation = 'Excellent! Your stock picking strategy is generating massive excess returns compared to a passive index fund.';
  }

  return `Jensen's Alpha: ${alpha > 0 ? '+' : ''}${alpha.toFixed(2)}% (${level})\n\n${recommendation}\n\nAlpha measures your portfolio's risk-adjusted outperformance (or underperformance) relative to the S&P 500 benchmark.`;
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

// Helper function to get Max Drawdown color.
// Thresholds match typical equity-portfolio expectations for a growth-tilted,
// 10y-horizon investor: up to -15% is normal noise, -15% to -25% is a
// meaningful correction, and deeper than -25% warrants review.
const getMaxDdColor = (pct) => {
  if (pct <= -25) return 'negative';
  if (pct <= -15) return 'warn';
  return '';
};

// Helper function to generate Max Drawdown tooltip
const getMaxDdTooltip = (pct, days) => {
  let level;
  let recommendation;
  if (pct >= -10) {
    level = 'Mild';
    recommendation = 'Portfolio has held up well. Normal market noise — no action needed.';
  } else if (pct >= -20) {
    level = 'Moderate';
    recommendation = 'Typical correction for a growth-tilted portfolio. Consider whether the drawdown came from market-wide risk-off or from specific holdings — if a single name dragged the portfolio disproportionately, check the thesis.';
  } else if (pct >= -30) {
    level = 'Deep';
    recommendation = 'Significant drawdown. Compare against VUAG.L / XNAS.L in the underwater chart: if your drawdown is worse than NASDAQ, review whether individual holdings have broken their thesis (deteriorating ROIC, moat erosion) rather than just trading on sentiment.';
  } else {
    level = 'Severe';
    recommendation = 'Severe drawdown. Worth a full thesis review on the largest losers — distinguish short-term noise (market-wide sell-off, weak guidance) from long-term damage (ROIC decline, moat erosion, bad capital allocation).';
  }
  const duration = days ? ` over ${days} days` : '';
  return `Max Drawdown: ${pct.toFixed(1)}%${duration} (${level})\n\n${recommendation}\n\nThis is the worst peak-to-trough decline since your account started. Compare against benchmarks in the underwater chart below the Portfolio Summary.`;
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

// Helper function to get Market Breadth Indicator color (7-day rolling mean)
const getMarketBreadthColor = (breadth) => {
  if (breadth < -0.15) return 'negative';
  if (breadth > 0.15) return 'positive';
  return '';
};

// Helper function to generate Market Breadth Indicator tooltip
const getMarketBreadthTooltip = (breadth) => {
  let recommendation = '';
  let level = '';
  const pct = breadth * 100;

  if (breadth < -0.25) {
    level = 'Weak';
    recommendation = 'On average over the last week, more S&P 500 names fell than rose day-to-day. Participation is narrow or defensive.';
  } else if (breadth < -0.1) {
    level = 'Slightly weak';
    recommendation = 'Mild negative breadth. Underlying participation is soft but not extreme.';
  } else if (breadth <= 0.1) {
    level = 'Neutral';
    recommendation = 'Participation is mixed. Advancers and decliners are roughly balanced over the rolling week.';
  } else if (breadth <= 0.25) {
    level = 'Healthy';
    recommendation = 'More names are up than down on average over the rolling week. Broad participation is supportive.';
  } else {
    level = 'Strong';
    recommendation = 'Very broad advance on average. Many names contributing to the move, not just a handful of leaders.';
  }

  return `7-Day Breadth: ${pct >= 0 ? '+' : ''}${pct.toFixed(1)}% (${level})\n\n${recommendation}\n\n7-day average of daily (Advancers − Decliners) / 503 — how many more S&P names rose vs fell each session, smoothed over the last week.`;
};

const getRealYield10yColor = (value) => {
  if (value > 2) return 'negative';
  if (value < 0.5) return 'positive';
  return '';
};

const getRealYield10yTooltip = (value) => {
  let level = '';
  let recommendation = '';

  if (value > 2.5) {
    level = 'Restrictive';
    recommendation = 'Real rates are high — a headwind for long-duration growth multiples. Favour quality and avoid overpaying for hype.';
  } else if (value > 1.5) {
    level = 'Elevated';
    recommendation = 'Above-average real yields. Discount rates are less supportive for high-multiple growth.';
  } else if (value > 0.5) {
    level = 'Normal';
    recommendation = 'Real yields in a normal range. No extreme macro tailwind or headwind from rates.';
  } else {
    level = 'Low';
    recommendation = 'Low or negative real yields support higher equity multiples. Favourable backdrop for growth.';
  }

  return `10Y Real Yield (TIPS): ${value.toFixed(2)}% (${level})\n\n${recommendation}\n\nFRED DFII10. Real return on 10-year inflation-protected Treasuries — a proxy for the discount rate on long-duration cash flows.`;
};

const getHyOasColor = (value) => {
  if (value > 5.5) return 'negative';
  return '';
};

const getHyOasTooltip = (value) => {
  let level = '';
  let recommendation = '';

  if (value > 8) {
    level = 'Stressed';
    recommendation = 'Credit spreads are wide — risk-off conditions. Historically a contrarian backdrop for long-term equity adds if fundamentals hold.';
  } else if (value > 5.5) {
    level = 'Elevated';
    recommendation = 'HY spreads are elevated. Credit markets are pricing stress — confirm with VIX and Fear & Greed before adding.';
  } else if (value > 4) {
    level = 'Normal';
    recommendation = 'Spreads in a typical range. No acute credit stress signal.';
  } else {
    level = 'Tight';
    recommendation = 'HY spreads are compressed — complacent credit conditions. Less margin of safety if risk-off arrives.';
  }

  return `HY OAS: ${value.toFixed(2)}% (${level})\n\n${recommendation}\n\nFRED BAMLH0A0HYM2. ICE BofA US High Yield option-adjusted spread — credit risk premium over Treasuries.`;
};

// Helper function to get SMA200 color
const getSma200Color = (value) => {
  if (value < 40) return 'negative';
  if (value > 60) return 'positive';
  return '';
};

// Helper function to generate SMA200 tooltip
const getSma200Tooltip = (value) => {
  let recommendation = '';
  let level = '';

  if (value < 30) {
    level = 'Oversold';
    recommendation = 'Major market weakness. Most stocks are in a downtrend. Potential long-term buying opportunity if capitulation occurs.';
  } else if (value < 50) {
    level = 'Bearish';
    recommendation = 'Market is weak. More than half of stocks are in a downtrend. Caution advised.';
  } else if (value < 70) {
    level = 'Bullish';
    recommendation = 'Market is healthy. Majority of stocks are in an uptrend. Good environment for trend following.';
  } else {
    level = 'Overbought';
    recommendation = 'Strong broad market participation. Watch for potential exhaustion if it stays extremely high for too long.';
  }

  return `% > SMA200: ${value.toFixed(1)}% (${level})\n\n${recommendation}\n\nPercentage of S&P 500 stocks trading above their 200-day Simple Moving Average. A long-term trend indicator.`;
};

// Helper function to get Consumer Sentiment color (contrarian: low = potential opportunity)
const getConsumerSentimentColor = (value) => {
  if (value < 70) return 'positive';  // Depressed sentiment, contrarian buy
  if (value > 90) return 'negative';  // Euphoria, caution
  return '';
};

// Helper function to generate Consumer Sentiment tooltip
const getConsumerSentimentTooltip = (value) => {
  let level = '';
  let recommendation = '';

  if (value < 60) {
    level = 'Very Low';
    recommendation = 'Contrarian signal: Consumer sentiment is deeply depressed. For long-term buy & hold, historically this has often coincided with better entry points.';
  } else if (value < 75) {
    level = 'Low';
    recommendation = 'Below-average sentiment. Can indicate fear or recession concerns. Long-term investors often find value when others are fearful.';
  } else if (value < 90) {
    level = 'Neutral to Optimistic';
    recommendation = 'Sentiment in a normal range. No strong contrarian signal either way.';
  } else {
    level = 'High / Euphoria';
    recommendation = 'Consumers are very optimistic. Historically, extreme optimism has sometimes preceded pullbacks. Stay disciplined with your plan.';
  }

  return `Consumer Sentiment: ${value.toFixed(1)} (${level})\n\n${recommendation}\n\nUniversity of Michigan index (base 1966 = 100). Monthly survey of consumer expectations; low readings often align with fear, high with optimism.`;
};

// ── Macro / Portfolio indicator history chart ──────────────────────────────────

const INDICATOR_CLUSTERS = {
  tactical: {
    label: 'Tactical',
    className: 'cluster-tactical',
    legendClass: 'legend-tactical',
    tooltip: 'Short-term sentiment and volatility. Mood context — not a long-term thesis signal.',
  },
  structure: {
    label: 'Structure',
    className: 'cluster-structure',
    legendClass: 'legend-structure',
    tooltip: 'Market breadth and participation — whether moves are broad or driven by a narrow set of names.',
  },
  valuation: {
    label: 'Valuation',
    className: 'cluster-valuation',
    legendClass: 'legend-valuation',
    tooltip: 'How expensive the market is vs fundamentals. Slow-moving backdrop, not a timing tool.',
  },
  macro: {
    label: 'Macro',
    className: 'cluster-macro',
    legendClass: 'legend-macro',
    tooltip: 'Rates, credit, and economic conditions — discount rates and risk appetite for growth assets.',
  },
};

// Market indicators (stored in MarketMetricsDaily)
const MARKET_INDICATOR_CONFIGS = {
  fear_greed_index: {
    label: 'Fear & Greed Index',
    yFormatter: v => v?.toFixed(1),
    color: '#9b59b6',
    referenceLines: [
      { y: 25, label: 'Extreme Fear',  stroke: '#28a745' },
      { y: 50, label: 'Neutral',       stroke: '#6c757d' },
      { y: 75, label: 'Extreme Greed', stroke: '#dc3545' },
    ],
  },
  vix: {
    label: 'VIX',
    yFormatter: v => v?.toFixed(1),
    color: '#e67e22',
    referenceLines: [
      { y: 15, label: 'Low',      stroke: '#28a745' },
      { y: 25, label: 'Elevated', stroke: '#ffc107' },
      { y: 35, label: 'High',     stroke: '#dc3545' },
    ],
  },
  sp500_above_sma200: {
    label: '% S&P 500 > SMA200',
    yFormatter: v => `${v?.toFixed(1)}%`,
    color: '#1abc9c',
    referenceLines: [
      { y: 40, label: 'Bear Zone', stroke: '#dc3545' },
      { y: 60, label: 'Bull Zone', stroke: '#28a745' },
    ],
  },
  market_breadth_indicator: {
    label: '7-Day Market Breadth',
    yFormatter: v => `${v >= 0 ? '+' : ''}${v?.toFixed(1)}%`,
    color: '#2ecc71',
    referenceLines: [
      { y: 0, label: 'Neutral', stroke: '#6c757d' },
    ],
  },
  buffett_indicator: {
    label: 'Buffett Indicator',
    yFormatter: v => `${v?.toFixed(1)}%`,
    color: '#e74c3c',
    referenceLines: [
      { y: 75,  label: 'Undervalued', stroke: '#28a745' },
      { y: 100, label: '100%',        stroke: '#ffc107' },
      { y: 150, label: 'Overvalued',  stroke: '#dc3545' },
    ],
  },
  real_yield_10y: {
    label: '10Y Real Yield (TIPS)',
    yFormatter: v => `${v?.toFixed(2)}%`,
    color: '#34495e',
    referenceLines: [
      { y: 0.5, label: 'Low',         stroke: '#28a745' },
      { y: 2.0, label: 'Restrictive', stroke: '#dc3545' },
    ],
  },
  hy_oas: {
    label: 'HY OAS',
    yFormatter: v => `${v?.toFixed(2)}%`,
    color: '#c0392b',
    referenceLines: [
      { y: 3.5, label: 'Tight',    stroke: '#ffc107' },
      { y: 5.5, label: 'Elevated', stroke: '#dc3545' },
    ],
  },
  yield_spread: {
    label: 'Yield Spread (10Y–2Y)',
    yFormatter: v => `${v?.toFixed(2)}%`,
    color: '#3498db',
    referenceLines: [
      { y: 0, label: 'Inverted', stroke: '#dc3545' },
    ],
  },
};

// Consumer Sentiment (monthly FRED data, fetched via dedicated endpoint)
const CS_CONFIG = {
  label: 'Consumer Sentiment',
  yFormatter: v => v?.toFixed(1),
  color: '#f39c12',
  referenceLines: [
    { y: 60, label: 'Very Low', stroke: '#28a745' },
    { y: 90, label: 'High',     stroke: '#dc3545' },
  ],
};

// Portfolio risk/return metrics (stored in PortfolioDaily)
const PORTFOLIO_INDICATOR_CONFIGS = {
  value: {
    label: 'Total Value',
    yFormatter: v => `£${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
    color: '#3498db',
    area: true,
    referenceLines: [],
  },
  profit: {
    label: 'Total Profit',
    yFormatter: v => `£${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
    color: '#28a745',
    area: true,
    referenceLines: [
      { y: 0, label: 'Breakeven', stroke: '#6c757d' },
    ],
  },
  return_pct: {
    label: 'Total Return',
    yFormatter: v => `${v?.toFixed(2)}%`,
    color: '#9b59b6',
    area: true,
    referenceLines: [
      { y: 0, label: 'Breakeven', stroke: '#6c757d' },
    ],
  },
  sortino_ratio: {
    label: 'Sortino Ratio',
    yFormatter: v => v?.toFixed(2),
    color: '#3498db',
    referenceLines: [
      { y: 1.0, label: 'Min. Acceptable', stroke: '#ffc107' },
      { y: 2.0, label: 'Excellent',        stroke: '#28a745' },
    ],
  },
  beta: {
    label: 'Beta',
    yFormatter: v => v?.toFixed(2),
    color: '#e67e22',
    referenceLines: [
      { y: 0.9, label: 'Low Beta',  stroke: '#ffc107' },
      { y: 1.3, label: 'High Beta', stroke: '#dc3545' },
    ],
  },
  alpha: {
    label: 'Jensen\'s Alpha',
    yFormatter: v => `${v?.toFixed(1)}%`,
    color: '#20c997',
    area: true,
    referenceLines: [
      { y: 0.0, label: 'Market Match', stroke: '#6c757d' },
    ],
  },
  mwrr: {
    label: 'Money-Weighted Return',
    yFormatter: v => `${v?.toFixed(1)}%`,
    color: '#9b59b6',
    referenceLines: [
      { y: 0,  label: 'Breakeven', stroke: '#6c757d' },
      { y: 10, label: '10%',       stroke: '#28a745' },
    ],
  },
  twrr: {
    label: 'Time-Weighted Return',
    yFormatter: v => `${v?.toFixed(1)}%`,
    color: '#2ecc71',
    referenceLines: [
      { y: 0,  label: 'Breakeven', stroke: '#6c757d' },
      { y: 10, label: '10%',       stroke: '#28a745' },
    ],
  },
  positions: {
    label: 'Portfolio Positions',
    yFormatter: v => v?.toFixed(0),
    area: true,
    stacked: true,
    winRate: true,
    series: [
      { dataKey: 'winning', name: 'Winning', color: '#28a745' },
      { dataKey: 'losing', name: 'Losing', color: '#dc3545' },
    ],
  },
};

const RANGE_OPTIONS = [
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
  { label: 'All', days: 0 },
];

const ChartIcon = () => (
  <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
    <polyline
      points="1,10 4,6 6.5,8 9.5,3 12,5"
      stroke="currentColor" strokeWidth="1.6"
      strokeLinecap="round" strokeLinejoin="round"
    />
  </svg>
);

const formatTickDate = (dateStr) => {
  const d = new Date(dateStr + 'T00:00:00');
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: '2-digit' });
};

const MacroTooltip = ({ active, payload, label, config }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="macro-tooltip">
      <div className="macro-tooltip-date">{formatTickDate(label)}</div>
      {config.series ? (
        <div className="macro-tooltip-series">
          {payload.map((entry, idx) => (
            <div key={idx} className="macro-tooltip-value" style={{ color: entry.color }}>
              {entry.name}: {config.yFormatter(entry.value)}
            </div>
          ))}
          {config.winRate && (() => {
            const total = payload.reduce((s, e) => s + (e.value || 0), 0);
            const winning = payload.find(e => e.dataKey === 'winning')?.value || 0;
            return total > 0 ? (
              <div className="macro-tooltip-value" style={{ color: '#aaa' }}>
                Win rate: {(winning / total * 100).toFixed(1)}%
              </div>
            ) : null;
          })()}
        </div>
      ) : (
        <div className="macro-tooltip-value" style={{ color: config.color }}>
          {config.yFormatter(payload[0].value)}
        </div>
      )}
    </div>
  );
};

/**
 * Generic chart modal for any indicator.
 * config     – { label, yFormatter, color, referenceLines }
 * currentValue – the current scalar to show in the header
 * fetchHistory – async (days: number) => { date: string, value: number }[]
 *                days=0 means "all available"
 */
const MacroChartModal = ({ config, currentValue, fetchHistory, clusterKey, onClose }) => {
  const [range, setRange] = useState(365);
  const [history, setHistory] = useState([]);
  const [loadingChart, setLoadingChart] = useState(true);

  // Keep a stable ref so useEffect doesn't need fetchHistory in its deps
  const fetchRef = useRef(fetchHistory);
  fetchRef.current = fetchHistory;

  // Close on ESC
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  // Fetch history when range changes.
  // isCurrent prevents a slower earlier request from overwriting a faster later one
  // when the user switches range tabs quickly.
  useEffect(() => {
    let isCurrent = true;
    setLoadingChart(true);
    fetchRef.current(range)
      .then(data  => { if (isCurrent) setHistory(data); })
      .catch(err  => { if (isCurrent) console.error('Failed to load indicator history', err); })
      .finally(() => { if (isCurrent) setLoadingChart(false); });
    return () => { isCurrent = false; };
  }, [range]);

  const ChartEl = config.area ? AreaChart : LineChart;
  const SeriesEl = config.area ? Area : Line;
  const cluster = clusterKey ? INDICATOR_CLUSTERS[clusterKey] : null;

  return (
    <div className="macro-modal-overlay" onClick={onClose}>
      <div className="macro-modal" onClick={e => e.stopPropagation()}>
        <div className="macro-modal-header">
          <div>
            {cluster && (
              <span className={`macro-modal-cluster ${cluster.legendClass}`}>{cluster.label}</span>
            )}
            <h3 className="macro-modal-title">{config.label}</h3>
            {currentValue != null && (
              <span className="macro-modal-current">
                Current: {config.yFormatter(currentValue)}
              </span>
            )}
          </div>
          <div className="macro-modal-controls">
            <div className="macro-range-buttons">
              {RANGE_OPTIONS.map(opt => (
                <button
                  key={opt.label}
                  className={`macro-range-btn${range === opt.days ? ' active' : ''}`}
                  onClick={() => setRange(opt.days)}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <button className="macro-modal-close" onClick={onClose} aria-label="Close">×</button>
          </div>
        </div>
        <div className="macro-modal-body">
          {loadingChart ? (
            <div className="macro-chart-placeholder">Loading…</div>
          ) : history.length === 0 ? (
            <div className="macro-chart-placeholder">No historical data available</div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <ChartEl data={history} margin={{ top: 8, right: 20, bottom: 4, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="date"
                  tickFormatter={formatTickDate}
                  interval="preserveStartEnd"
                  tick={{ fontSize: 11 }}
                  minTickGap={60}
                />
                <YAxis
                  tickFormatter={config.yFormatter}
                  tick={{ fontSize: 11 }}
                  width={60}
                />
                <Tooltip content={(props) => <MacroTooltip {...props} config={config} />} />
                {config.referenceLines && config.referenceLines.map(ref => (
                  <ReferenceLine
                    key={ref.y}
                    y={ref.y}
                    stroke={ref.stroke}
                    strokeDasharray="4 3"
                    strokeWidth={1}
                    label={{
                      value: ref.label,
                      position: 'insideTopRight',
                      fontSize: 10,
                      fill: ref.stroke,
                    }}
                  />
                ))}
                {config.series ? config.series.map(s => (
                  <SeriesEl
                    key={s.dataKey}
                    type="monotone"
                    dataKey={s.dataKey}
                    name={s.name}
                    stroke={s.color}
                    fill={s.color}
                    stackId={config.stacked ? "1" : undefined}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                    fillOpacity={0.12}
                  />
                )) : (
                  <SeriesEl
                    type="monotone"
                    dataKey="value"
                    stroke={config.color}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                    {...(config.area ? { fill: config.color, fillOpacity: 0.12 } : {})}
                  />
                )}
              </ChartEl>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────

const Dashboard = () => {
  const { hideAmounts } = useHideAmounts();
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPeriod, setSelectedPeriod] = useState('1d');
  // { config, currentValue, fetchHistory, clusterKey } | null
  const [macroModal, setMacroModal] = useState(null);
  const openMacroModal = useCallback((config, currentValue, fetchHistory, clusterKey = null) => {
    setMacroModal({ config, currentValue, fetchHistory, clusterKey });
  }, []);
  const closeMacroModal = useCallback(() => setMacroModal(null), []);

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
      <div className="page-fixed dashboard-container">
        <div className="loading">Loading portfolio data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-fixed dashboard-container">
        <div className="error">{error}</div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="page-fixed dashboard-container">
        <div className="error">No portfolio data available</div>
      </div>
    );
  }

  return (
    <div className="page-fixed dashboard-container">
      <div className="dashboard-header">
        <h1>Portfolio Summary</h1>
        {summary.last_updated && (
          <div className="last-updated">Last updated: {new Date(summary.last_updated).toLocaleString()}</div>
        )}
      </div>

      {/* Portfolio Indicators Section */}
      <div className="indicators-section">
        <h2 className="section-heading">Portfolio Indicators</h2>
        <div className="summary-cards">
          <div className="card macro-card">
            {!hideAmounts && (
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                PORTFOLIO_INDICATOR_CONFIGS.value,
                summary.total_value,
                async (days) => {
                  const d = await portfolioAPI.getPortfolioIndicatorsHistory(days);
                  return d.filter(r => r.value != null).map(r => ({ date: r.date, value: r.value }));
                }
              )}>
                <ChartIcon />
              </button>
            )}
            <h3>Total Value</h3>
            <p className="value">{hideAmounts ? MASK : `£${summary.total_value.toLocaleString()}`}</p>
          </div>
          <div className="card macro-card">
            {!hideAmounts && (
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                PORTFOLIO_INDICATOR_CONFIGS.profit,
                summary.total_profit,
                async (days) => {
                  const d = await portfolioAPI.getPortfolioIndicatorsHistory(days);
                  return d.filter(r => r.profit != null).map(r => ({ date: r.date, value: r.profit }));
                }
              )}>
                <ChartIcon />
              </button>
            )}
            <h3>Total Profit</h3>
            <p className={`value ${summary.total_profit >= 0 ? 'positive' : 'negative'}`}>
              {hideAmounts ? MASK : `£${summary.total_profit.toLocaleString()}`}
            </p>
          </div>
          <div className="card macro-card">
            <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
              PORTFOLIO_INDICATOR_CONFIGS.return_pct,
              summary.total_return_pct,
              async (days) => {
                const d = await portfolioAPI.getPortfolioIndicatorsHistory(days);
                return d.filter(r => r.return_pct != null).map(r => ({ date: r.date, value: r.return_pct }));
              }
            )}>
              <ChartIcon />
            </button>
            <h3>Total Return</h3>
            <p className={`value ${summary.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
              {summary.total_return_pct >= 0 ? '+' : ''}{summary.total_return_pct.toFixed(2)}%
            </p>
          </div>
          <div className="card macro-card">
            <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
              PORTFOLIO_INDICATOR_CONFIGS.positions,
              summary.total_holdings,
              async (days) => {
                const d = await portfolioAPI.getPortfolioIndicatorsHistory(days);
                return d.filter(r => r.positions_total > 0)
                        .map(r => ({ date: r.date, winning: r.positions_winning, losing: r.positions_total - r.positions_winning }));
              }
            )}>
              <ChartIcon />
            </button>
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
          {summary.mwrr != null && (
            <div className="card macro-card" title={getMwrrTooltip(summary.mwrr)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                PORTFOLIO_INDICATOR_CONFIGS.mwrr,
                summary.mwrr,
                async (days) => {
                  const d = await portfolioAPI.getPortfolioIndicatorsHistory(days);
                  return d.filter(r => r.mwrr != null).map(r => ({ date: r.date, value: r.mwrr }));
                }
              )}>
                <ChartIcon />
              </button>
              <h3>Money-Weighted RR</h3>
              <p className={`value ${getMwrrColor(summary.mwrr)}`}>
                {summary.mwrr.toFixed(2)}%
              </p>
            </div>
          )}
          {summary.twrr != null && (
            <div className="card macro-card" title={getTwrrTooltip(summary.twrr)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                PORTFOLIO_INDICATOR_CONFIGS.twrr,
                summary.twrr,
                async (days) => {
                  const d = await portfolioAPI.getPortfolioIndicatorsHistory(days);
                  return d.filter(r => r.twrr != null).map(r => ({ date: r.date, value: r.twrr }));
                }
              )}>
                <ChartIcon />
              </button>
              <h3>Time-Weighted RR</h3>
              <p className={`value ${getTwrrColor(summary.twrr)}`}>
                {summary.twrr.toFixed(2)}%
              </p>
            </div>
          )}
          {summary.sortino_ratio && (
            <div className="card macro-card" title={getSortinoTooltip(summary.sortino_ratio)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                PORTFOLIO_INDICATOR_CONFIGS.sortino_ratio,
                summary.sortino_ratio,
                async (days) => {
                  const d = await portfolioAPI.getPortfolioIndicatorsHistory(days);
                  return d.filter(r => r.sortino_ratio != null).map(r => ({ date: r.date, value: r.sortino_ratio }));
                }
              )}>
                <ChartIcon />
              </button>
              <h3>Sortino</h3>
              <p className={`value ${getSortinoColor(summary.sortino_ratio)}`}>
                {summary.sortino_ratio.toFixed(2)}
              </p>
            </div>
          )}
          {summary.alpha != null && (
            <div className="card macro-card" title={getAlphaTooltip(summary.alpha)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                PORTFOLIO_INDICATOR_CONFIGS.alpha,
                summary.alpha,
                async (days) => {
                  const d = await portfolioAPI.getPortfolioIndicatorsHistory(days);
                  return d.filter(r => r.jensens_alpha != null).map(r => ({ date: r.date, value: r.jensens_alpha }));
                }
              )}>
                <ChartIcon />
              </button>
              <h3>Jensen&apos;s Alpha</h3>
              <p className={`value ${getAlphaColor(summary.alpha)}`}>
                {summary.alpha > 0 ? '+' : ''}{summary.alpha.toFixed(2)}%
              </p>
            </div>
          )}
          {summary.beta && (
            <div className="card macro-card" title={getBetaTooltip(summary.beta)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                PORTFOLIO_INDICATOR_CONFIGS.beta,
                summary.beta,
                async (days) => {
                  const d = await portfolioAPI.getPortfolioIndicatorsHistory(days);
                  return d.filter(r => r.beta != null).map(r => ({ date: r.date, value: r.beta }));
                }
              )}>
                <ChartIcon />
              </button>
              <h3>Beta</h3>
              <p className={`value ${getBetaColor(summary.beta)}`}>
                {summary.beta.toFixed(2)}
              </p>
            </div>
          )}
          {summary.max_drawdown_pct != null && (
            <div className="card" title={getMaxDdTooltip(summary.max_drawdown_pct, summary.drawdown_duration_days)}>
              <h3>Max DD</h3>
              <p className={`value ${getMaxDdColor(summary.max_drawdown_pct)}`}>
                {summary.max_drawdown_pct.toFixed(1)}%
                {summary.drawdown_duration_days > 0 && (
                  <span className="dd-duration">({summary.drawdown_duration_days}d)</span>
                )}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Market Indicators Section */}
      <div className="indicators-section">
        <h2 className="section-heading">Market Indicators</h2>
        <p className="indicators-cluster-legend">
          <span className={`${INDICATOR_CLUSTERS.tactical.legendClass} legend-item`} title={INDICATOR_CLUSTERS.tactical.tooltip}>Tactical</span>
          <span className="legend-sep"> · </span>
          <span className={`${INDICATOR_CLUSTERS.structure.legendClass} legend-item`} title={INDICATOR_CLUSTERS.structure.tooltip}>Structure</span>
          <span className="legend-sep"> · </span>
          <span className={`${INDICATOR_CLUSTERS.valuation.legendClass} legend-item`} title={INDICATOR_CLUSTERS.valuation.tooltip}>Valuation</span>
          <span className="legend-sep"> · </span>
          <span className={`${INDICATOR_CLUSTERS.macro.legendClass} legend-item`} title={INDICATOR_CLUSTERS.macro.tooltip}>Macro</span>
        </p>
        <div className="summary-cards">
          {summary.fear_greed_index && (
            <div className={`card macro-card ${INDICATOR_CLUSTERS.tactical.className}`} title={getFearGreedTooltip(summary.fear_greed_index)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                MARKET_INDICATOR_CONFIGS.fear_greed_index,
                summary.fear_greed_index.value,
                async (days) => {
                  const d = await portfolioAPI.getMarketIndicatorsHistory(days);
                  return d.filter(r => r.fear_greed_index != null).map(r => ({ date: r.date, value: r.fear_greed_index }));
                },
                'tactical'
              )}>
                <ChartIcon />
              </button>
              <h3>Fear & Greed</h3>
              <a href="https://edition.cnn.com/markets/fear-and-greed" target="_blank" rel="noopener noreferrer" className="value-link">
                <p className={`value ${getFearGreedColor(summary.fear_greed_index.label)}`}>
                  {summary.fear_greed_index.value.toFixed(1)}
                </p>
              </a>
            </div>
          )}
          {summary.vix != null && (
            <div className={`card macro-card ${INDICATOR_CLUSTERS.tactical.className}`} title={getVixTooltip(summary.vix)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                MARKET_INDICATOR_CONFIGS.vix,
                summary.vix,
                async (days) => {
                  const d = await portfolioAPI.getMarketIndicatorsHistory(days);
                  return d.filter(r => r.vix != null).map(r => ({ date: r.date, value: r.vix }));
                },
                'tactical'
              )}>
                <ChartIcon />
              </button>
              <h3>VIX</h3>
              <p className={`value ${getVIXColor(summary.vix)}`}>{summary.vix.toFixed(2)}</p>
            </div>
          )}
          {summary.sp500_above_sma200 != null && (
            <div className={`card macro-card ${INDICATOR_CLUSTERS.structure.className}`} title={getSma200Tooltip(summary.sp500_above_sma200)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                MARKET_INDICATOR_CONFIGS.sp500_above_sma200,
                summary.sp500_above_sma200,
                async (days) => {
                  const d = await portfolioAPI.getMarketIndicatorsHistory(days);
                  return d.filter(r => r.sp500_above_sma200 != null).map(r => ({ date: r.date, value: r.sp500_above_sma200 }));
                },
                'structure'
              )}>
                <ChartIcon />
              </button>
              <h3>% &gt; SMA200</h3>
              <a href="https://stockcharts.com/h-sc/ui?s=$SPXA200R" target="_blank" rel="noopener noreferrer" className="value-link">
                <p className={`value ${getSma200Color(summary.sp500_above_sma200)}`}>
                  {summary.sp500_above_sma200.toFixed(1)}%
                </p>
              </a>
            </div>
          )}
          {summary.market_breadth_indicator != null && (
            <div className={`card macro-card ${INDICATOR_CLUSTERS.structure.className}`} title={getMarketBreadthTooltip(summary.market_breadth_indicator)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                MARKET_INDICATOR_CONFIGS.market_breadth_indicator,
                summary.market_breadth_indicator * 100,
                async (days) => {
                  const d = await portfolioAPI.getMarketIndicatorsHistory(days);
                  return d.filter(r => r.market_breadth_indicator != null).map(r => ({ date: r.date, value: r.market_breadth_indicator }));
                },
                'structure'
              )}>
                <ChartIcon />
              </button>
              <h3>7-Day Breadth</h3>
              <p className={`value ${getMarketBreadthColor(summary.market_breadth_indicator)}`}>
                {(summary.market_breadth_indicator * 100 >= 0 ? '+' : '')}
                {(summary.market_breadth_indicator * 100).toFixed(1)}%
              </p>
            </div>
          )}
          {summary.buffett_indicator != null && (
            <div className={`card macro-card ${INDICATOR_CLUSTERS.valuation.className}`} title={getBuffettIndicatorTooltip(summary.buffett_indicator)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                MARKET_INDICATOR_CONFIGS.buffett_indicator,
                summary.buffett_indicator,
                async (days) => {
                  const d = await portfolioAPI.getMarketIndicatorsHistory(days);
                  return d.filter(r => r.buffett_indicator != null).map(r => ({ date: r.date, value: r.buffett_indicator }));
                },
                'valuation'
              )}>
                <ChartIcon />
              </button>
              <h3>Buffett Indicator</h3>
              <a href="https://currentmarketvaluation.com/models/buffett-indicator.php" target="_blank" rel="noopener noreferrer" className="value-link">
                <p className={`value ${getBuffettIndicatorColor(summary.buffett_indicator)}`}>
                  {summary.buffett_indicator.toFixed(1)}%
                </p>
              </a>
            </div>
          )}
          {summary.real_yield_10y != null && (
            <div className={`card macro-card ${INDICATOR_CLUSTERS.macro.className}`} title={getRealYield10yTooltip(summary.real_yield_10y)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                MARKET_INDICATOR_CONFIGS.real_yield_10y,
                summary.real_yield_10y,
                async (days) => {
                  const d = await portfolioAPI.getMarketIndicatorsHistory(days);
                  return d.filter(r => r.real_yield_10y != null).map(r => ({ date: r.date, value: r.real_yield_10y }));
                },
                'macro'
              )}>
                <ChartIcon />
              </button>
              <h3>10Y Real Yield</h3>
              <a href="https://fred.stlouisfed.org/series/DFII10" target="_blank" rel="noopener noreferrer" className="value-link">
                <p className={`value ${getRealYield10yColor(summary.real_yield_10y)}`}>
                  {summary.real_yield_10y.toFixed(2)}%
                </p>
              </a>
            </div>
          )}
          {summary.hy_oas != null && (
            <div className={`card macro-card ${INDICATOR_CLUSTERS.macro.className}`} title={getHyOasTooltip(summary.hy_oas)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                MARKET_INDICATOR_CONFIGS.hy_oas,
                summary.hy_oas,
                async (days) => {
                  const d = await portfolioAPI.getMarketIndicatorsHistory(days);
                  return d.filter(r => r.hy_oas != null).map(r => ({ date: r.date, value: r.hy_oas }));
                },
                'macro'
              )}>
                <ChartIcon />
              </button>
              <h3>HY OAS</h3>
              <a href="https://fred.stlouisfed.org/series/BAMLH0A0HYM2" target="_blank" rel="noopener noreferrer" className="value-link">
                <p className={`value ${getHyOasColor(summary.hy_oas)}`}>
                  {summary.hy_oas.toFixed(2)}%
                </p>
              </a>
            </div>
          )}
          {summary.yield_spread != null && (
            <div className={`card macro-card ${INDICATOR_CLUSTERS.macro.className}`} title={getYieldSpreadTooltip(summary.yield_spread)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                MARKET_INDICATOR_CONFIGS.yield_spread,
                summary.yield_spread,
                async (days) => {
                  const d = await portfolioAPI.getMarketIndicatorsHistory(days);
                  return d.filter(r => r.yield_spread != null).map(r => ({ date: r.date, value: r.yield_spread }));
                },
                'macro'
              )}>
                <ChartIcon />
              </button>
              <h3>Yield Spread</h3>
              <a href="https://fred.stlouisfed.org/series/T10Y2Y" target="_blank" rel="noopener noreferrer" className="value-link">
                <p className={`value ${getYieldSpreadColor(summary.yield_spread)}`}>
                  {summary.yield_spread.toFixed(2)}%
                </p>
              </a>
            </div>
          )}
          {summary.consumer_sentiment != null && (
            <div className={`card macro-card ${INDICATOR_CLUSTERS.macro.className}`} title={getConsumerSentimentTooltip(summary.consumer_sentiment)}>
              <button className="macro-chart-btn" title="View history" onClick={() => openMacroModal(
                CS_CONFIG,
                summary.consumer_sentiment,
                async (days) => {
                  // Prefer DB history (stored daily alongside other metrics).
                  // Fall back to live FRED fetch if DB has no CS data yet
                  // (e.g. before the first run after the migration).
                  const d = await portfolioAPI.getMarketIndicatorsHistory(days);
                  const dbData = d.filter(r => r.consumer_sentiment != null)
                                  .map(r => ({ date: r.date, value: r.consumer_sentiment }));
                  if (dbData.length > 0) return dbData;
                  const months = days === 0 ? 0 : Math.max(3, Math.ceil(days / 30));
                  return portfolioAPI.getConsumerSentimentHistory(months);
                },
                'macro'
              )}>
                <ChartIcon />
              </button>
              <h3>Consumer Sentiment</h3>
              <a href="https://fred.stlouisfed.org/series/UMCSENT" target="_blank" rel="noopener noreferrer" className="value-link">
                <p className={`value ${getConsumerSentimentColor(summary.consumer_sentiment)}`}>
                  {summary.consumer_sentiment.toFixed(1)}
                </p>
              </a>
            </div>
          )}
        </div>
      </div>

      {/* Portfolio Analytics Charts */}
      <div className="portfolio-charts-section">
        <PortfolioChart selectedPeriod={selectedPeriod} />
      </div>

      {/* Top Movers Section */}
      <div className="top-movers-section">
        <TopMovers selectedPeriod={selectedPeriod} setSelectedPeriod={setSelectedPeriod} />
      </div>

      {/* Indicator History Modal */}
      {macroModal && (
        <MacroChartModal
          config={macroModal.config}
          currentValue={macroModal.currentValue}
          fetchHistory={macroModal.fetchHistory}
          clusterKey={macroModal.clusterKey}
          onClose={closeMacroModal}
        />
      )}
    </div>
  );
};

export default Dashboard;
