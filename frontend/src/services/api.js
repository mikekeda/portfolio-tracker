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
    // Check if this is a 401 authentication error FIRST
    if (error.response && error.response.status === 401) {
      const errorMessage = error.response.data?.detail || error.response.data?.message || 'Authentication failed';
      console.error("Set authentication token with: localStorage.setItem('api_token', 'your-token-here')");

      // Show authentication-specific error message
      alert(`Authentication Error (401 Unauthorized)\n\n${errorMessage}\n\nPlease check your API token.`);
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

// Export the apiClient for use in other modules
export default apiClient;
