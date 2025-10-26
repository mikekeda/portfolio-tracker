/**
 * Screeners Service
 * ================
 * Contains screener logic and API calls for the stock screening functionality.
 */

import apiClient from './api';

// Cache for screeners data
let screenersCache = null;
let screenersCacheTimestamp = null;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Fetch screeners from API
export const fetchScreeners = async () => {
  try {
    const response = await apiClient.get('/api/screeners');
    return response.data.screeners || [];
  } catch (error) {
    console.error('Failed to fetch screeners:', error);
    throw error;
  }
};

// Get screeners with caching
export const getScreeners = async (useCache = true) => {
  const now = Date.now();

  // Return cached data if still valid
  if (useCache && screenersCache && screenersCacheTimestamp &&
      (now - screenersCacheTimestamp) < CACHE_DURATION) {
    return screenersCache;
  }

  try {
    const screeners = await fetchScreeners();
    screenersCache = screeners;
    screenersCacheTimestamp = now;
    return screeners;
  } catch (error) {
    // Return cached data if available, even if expired
    if (screenersCache) {
      console.warn('Using expired screener cache due to API error:', error);
      return screenersCache;
    }
    throw error;
  }
};

// Get available screeners only
export const getAvailableScreeners = async () => {
  const screeners = await getScreeners();
  return screeners.filter(screener => screener.available);
};
