import axios from 'axios';
import config from '../config';

// API configuration
const API_BASE_URL = config.API_BASE_URL;

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: config.API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add token to all requests
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('api_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Check if this is a 401 authentication error
    if (error.response && error.response.status === 401) {
      // Don't redirect if we're already on the login page
      if (window.location.pathname !== '/login') {
        // Store the current path to redirect back after login
        const currentPath = window.location.pathname + window.location.search;
        sessionStorage.setItem('redirectAfterLogin', currentPath);
        // Redirect to login page
        window.location.href = '/login';
      }
    } else if (error.response) {
      // Server responded with other error status
      console.error('API Error:', error.response.status, error.response.data);
    } else if (error.request) {
      // Request was made but no response received
      console.error('No response received from server');
    } else {
      // Something else happened
      console.error('Request setup error:', error.message);
    }
    return Promise.reject(error);
  }
);

// Portfolio API methods
export const portfolioAPI = {
  // Get portfolio holdings. showAll=false (default): current holdings only. showAll=true: all monitored instruments.
  getCurrentHoldings: async (showAll = false) => {
    const response = await apiClient.get('/api/portfolio/current', { params: { show_all: showAll } });
    return response.data;
  },

  // Get portfolio summary statistics
  getSummary: async () => {
    const response = await apiClient.get('/api/portfolio/summary');
    return response.data;
  },

  // Get portfolio allocations
  getAllocations: async () => {
    const response = await apiClient.get('/api/portfolio/allocations');
    return response.data;
  },

  // Get portfolio history
  getHistory: async (days = null) => {
    let url = "/api/portfolio/history"
    if (days !== null) {
      url += `?days=${days}`;
    }
    const response = await apiClient.get(url);
    return response.data;
  },

  // Get all instruments for autocomplete
  getInstruments: async () => {
    const response = await apiClient.get('/api/tickers');
    return response.data;
  },

  // Get chart price data
  getChartPrices: async (symbols, days = 30) => {
    const symbolsParam = Array.isArray(symbols) ? symbols.join(',') : symbols;
    const response = await apiClient.get(`/api/chart/prices?symbols=${symbolsParam}&days=${days}`);
    return response.data;
  },

  // Get chart data for different metrics
  getChartMetrics: async (symbols, days = 30, metric = 'price') => {
    const symbolsParam = Array.isArray(symbols) ? symbols.join(',') : symbols;
    const response = await apiClient.get(`/api/chart/metrics?symbols=${symbolsParam}&days=${days}&metric=${metric}`);
    return response.data;
  },

  // Get compact fundamentals (price, mkt cap, PE, ROIC, etc.) for chart summary row
  getChartFundamentals: async (symbols) => {
    const symbolsParam = Array.isArray(symbols) ? symbols.join(',') : symbols;
    const response = await apiClient.get(`/api/chart/fundamentals?symbols=${symbolsParam}`);
    return response.data;
  },

  // Get all portfolio movers (holdings with price changes) for a given period
  getTopMovers: async (period = '1d') => {
    const response = await apiClient.get(`/api/market/movers?period=${period}`);
    return response.data;
  },

  // Get detailed instrument data
  getInstrument: async (symbol, days = 30) => {
    const response = await apiClient.get(`/api/instrument/${encodeURIComponent(symbol)}?days=${days}`);
    return response.data;
  },

  // Get chart-window data only (prices, orders, splits, PE) — for range changes
  getInstrumentPrices: async (symbol, days = 365) => {
    const response = await apiClient.get(`/api/instrument/${encodeURIComponent(symbol)}/prices?days=${days}`);
    return response.data;
  },

  // Get all pies
  getPies: async () => {
    const response = await apiClient.get('/api/pies');
    return response.data;
  },

  // Get earnings calendar events
  getEarningsCalendar: async (portfolioOnly = false) => {
    const response = await apiClient.get('/api/earnings/calendar', { params: { portfolio_only: portfolioOnly } });
    return response.data;
  },

  // Get earnings report HTML
  getEarningsReportHtml: async (symbol, reportDate) => {
    const response = await apiClient.get(`/api/earnings-report/${encodeURIComponent(symbol)}/${reportDate}`, {
      responseType: 'text',  // Get as text/HTML, not JSON
    });
    return response.data;
  },

  // Get full transaction history + income summary
  getTransactions: async () => {
    const response = await apiClient.get('/api/transactions');
    return response.data;
  },

  // Get historical market indicator data for charting (days=0 = all history)
  getMarketIndicatorsHistory: async (days = 365) => {
    const response = await apiClient.get(`/api/market/indicators/history?days=${days}`);
    return response.data;
  },

  // Get historical consumer sentiment from FRED (months=0 = ~10 years)
  getConsumerSentimentHistory: async (months = 36) => {
    const response = await apiClient.get(`/api/market/indicators/consumer-sentiment/history?months=${months}`);
    return response.data;
  },

  // Get historical portfolio risk/return metrics (days=0 = all history)
  getPortfolioIndicatorsHistory: async (days = 365) => {
    const response = await apiClient.get(`/api/portfolio/indicators/history?days=${days}`);
    return response.data;
  },

  // Get underwater (drawdown-from-peak) series for portfolio + benchmarks
  getUnderwater: async (days = null) => {
    let url = '/api/portfolio/underwater';
    if (days !== null) {
      url += `?days=${days}`;
    }
    const response = await apiClient.get(url);
    return response.data;
  },

  // Get deposit events (for overlaying as markers on charts)
  getDeposits: async (days = null) => {
    let url = '/api/portfolio/deposits';
    if (days !== null) {
      url += `?days=${days}`;
    }
    const response = await apiClient.get(url);
    return response.data;
  },

  // Get inputs for the portfolio projection page (TWRR, vol, UK CPI, benchmarks)
  getProjectionInputs: async () => {
    const response = await apiClient.get('/api/projection/inputs');
    return response.data;
  },

  // Get HRP reference weights + risk-contribution diagnostics for the risk page
  getRisk: async () => {
    const response = await apiClient.get('/api/risk');
    return response.data;
  },

  // Get trade-agent suggestions for one day (default: latest available)
  getAgentSuggestions: async (forDate = null) => {
    let url = '/api/agent/suggestions';
    if (forDate) {
      url += `?for_date=${forDate}`;
    }
    const response = await apiClient.get(url);
    return response.data;
  },

  // Get trade-agent suggestion history over the trailing N days
  getAgentSuggestionsHistory: async (days = 30) => {
    const response = await apiClient.get(`/api/agent/suggestions/history?days=${days}`);
    return response.data;
  },

  // Record accept/dismiss decision on a trade suggestion
  setAgentSuggestionStatus: async (id, status) => {
    const response = await apiClient.post(`/api/agent/suggestions/${id}/status`, { status });
    return response.data;
  },
};

// Export the apiClient for use in other modules
export default apiClient;
