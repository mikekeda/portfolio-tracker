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

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
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
    console.error('API Response Error:', error);
    if (error.response) {
      // Server responded with error status
      console.error('Error Status:', error.response.status);
      console.error('Error Data:', error.response.data);
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
  // Get current portfolio holdings
  getCurrentHoldings: async () => {
    const response = await apiClient.get('/api/portfolio/current');
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
  getHistory: async (days = 30) => {
    const response = await apiClient.get(`/api/portfolio/history?days=${days}`);
    return response.data;
  },

  // Get all instruments for autocomplete
  getInstruments: async () => {
    const response = await apiClient.get('/api/instruments');
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

  // Get top movers (gainers and losers)
  getTopMovers: async (period = '1d', limit = 10) => {
    const response = await apiClient.get(`/api/market/top-movers?period=${period}&limit=${limit}`);
    return response.data;
  },

  // Get detailed instrument data
  getInstrument: async (symbol, days = 30) => {
    const response = await apiClient.get(`/api/instrument/${encodeURIComponent(symbol)}?days=${days}`);
    return response.data;
  },

  // Get all pies
  getPies: async () => {
    const response = await apiClient.get('/api/pies');
    return response.data;
  },
};
