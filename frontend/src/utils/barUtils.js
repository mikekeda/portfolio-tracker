/**
 * Utility functions for creating horizontal background bars
 */

/**
 * Calculate bar width percentage based on value and range
 * @param {number} value - The current value
 * @param {number} min - Minimum value in the range
 * @param {number} max - Maximum value in the range
 * @param {number} maxBarWidth - Maximum bar width percentage (default 100)
 * @returns {number} - Bar width percentage
 */
export const calculateBarWidth = (value, min, max, maxBarWidth = 100) => {
  if (value === null || value === undefined || min === max) return 0;
  const percentage = ((value - min) / (max - min)) * maxBarWidth;
  return Math.max(0, Math.min(percentage, maxBarWidth));
};

/**
 * Get color scheme for different column types
 * @param {string} columnType - Type of column (percentage, value, institutional, short, weekChange, weekHighChange)
 * @param {number} value - The value to determine color
 * @returns {object} - Color scheme object
 */
export const getBarColorScheme = (columnType, value) => {
  // Simplified color schemes - only green and red
  const positiveScheme = {
    background: 'linear-gradient(90deg, #e8f5e8 0%, #c8e6c9 100%)',
    hoverBackground: 'linear-gradient(90deg, #c8e6c9 0%, #a5d6a7 100%)'
  };

  const negativeScheme = {
    background: 'linear-gradient(90deg, #ffebee 0%, #ffcdd2 100%)',
    hoverBackground: 'linear-gradient(90deg, #ffcdd2 0%, #ef9a9a 100%)'
  };

  // Determine if value should be considered "negative" (red, right-to-left)
  const isNegative = shouldBeNegativeBar(columnType, value);

  return isNegative ? negativeScheme : positiveScheme;
};

/**
 * Calculate min/max values for a dataset
 * @param {Array} data - Array of values
 * @param {string} accessor - Property name to access (for objects)
 * @returns {object} - Object with min and max values
 */
export const calculateMinMax = (data, accessor = null) => {
  const values = accessor ? data.map(item => item[accessor]) : data;
  const validValues = values.filter(val => val !== null && val !== undefined && !isNaN(val));

  if (validValues.length === 0) return { min: 0, max: 100 };

  return {
    min: Math.min(...validValues),
    max: Math.max(...validValues)
  };
};

/**
 * Determine if a bar should be negative (right-to-left, red)
 * @param {string} columnType - Type of column
 * @param {number} value - The value
 * @returns {boolean} - Whether the bar should be negative
 */
export const shouldBeNegativeBar = (columnType, value) => {
  switch (columnType) {
    case 'weekChange':
    case 'weekHighChange':
      return value < 0; // Only truly negative values get red bars
    case 'short':
      return true; // Short interest is always a negative effect, so always red bars
    case 'rsi':
      return value > 70; // Overbought RSI (>70) is considered "negative" for investors
    case 'institutional':
    case 'percentage':
    case 'value':
    default:
      // For all other columns, values are always positive percentages/amounts
      // Bar direction is based on mathematical sign, not performance interpretation
      return false;
  }
};

/**
 * Get CSS custom properties for bar styling
 * @param {number} barWidth - Bar width percentage
 * @param {object} colorScheme - Color scheme object
 * @returns {object} - CSS custom properties
 */
export const getBarStyle = (barWidth, colorScheme) => ({
  '--bar-width': `${barWidth}%`,
  '--bar-background': colorScheme.background,
  '--bar-hover-background': colorScheme.hoverBackground
  // Don't set text color - keep original text colors
});
