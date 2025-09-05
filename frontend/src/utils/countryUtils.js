import React from 'react';
import ReactCountryFlag from 'react-country-flag';

// Country name to ISO country code mapping
const COUNTRY_MAP = {
  'United States': 'US',
  'United Kingdom': 'GB',
  'Canada': 'CA',
  'Germany': 'DE',
  'France': 'FR',
  'Japan': 'JP',
  'China': 'CN',
  'India': 'IN',
  'Australia': 'AU',
  'Brazil': 'BR',
  'South Korea': 'KR',
  'Netherlands': 'NL',
  'Switzerland': 'CH',
  'Italy': 'IT',
  'Spain': 'ES',
  'Sweden': 'SE',
  'Norway': 'NO',
  'Denmark': 'DK',
  'Finland': 'FI',
  'Belgium': 'BE',
  'Ireland': 'IE',
  'Austria': 'AT',
  'Poland': 'PL',
  'Czech Republic': 'CZ',
  'Hungary': 'HU',
  'Portugal': 'PT',
  'Greece': 'GR',
  'Turkey': 'TR',
  'Russia': 'RU',
  'South Africa': 'ZA',
  'Mexico': 'MX',
  'Argentina': 'AR',
  'Chile': 'CL',
  'Colombia': 'CO',
  'Peru': 'PE',
  'Israel': 'IL',
  'Saudi Arabia': 'SA',
  'United Arab Emirates': 'AE',
  'Singapore': 'SG',
  'Hong Kong': 'HK',
  'Taiwan': 'TW',
  'Thailand': 'TH',
  'Malaysia': 'MY',
  'Indonesia': 'ID',
  'Philippines': 'PH',
  'Vietnam': 'VN',
  'New Zealand': 'NZ',
  'Egypt': 'EG',
  'Nigeria': 'NG',
  'Kenya': 'KE',
  'Morocco': 'MA',
  'Uruguay': 'UY',
  'Bermuda': 'BM',
  'Cayman Islands': 'KY',
  'Ukraine': 'UA',
  'Other': 'GLOBE'
};

/**
 * Get ISO country code for a given country name
 * @param {string} countryName - The country name
 * @returns {string|null} - ISO country code or null if not found
 */
export const getCountryCode = (countryName) => {
  if (!countryName) return null;
  return COUNTRY_MAP[countryName] || null;
};

/**
 * Render country flag or globe icon with country name
 * @param {string} countryName - The country name
 * @param {object} style - Additional styles to apply
 * @returns {JSX.Element} - React element with flag/globe and country name
 */
export const renderCountryWithFlag = (countryName, style = {}) => {
  const countryCode = getCountryCode(countryName);
  const defaultStyle = {
    width: '1.2em',
    height: '0.8em',
    marginRight: '6px',
    fontSize: '0.8em'
  };

  const combinedStyle = { ...defaultStyle, ...style };

  return (
    <>
      {countryCode && countryCode !== 'GLOBE' && (
        <ReactCountryFlag
          countryCode={countryCode}
          svg
          style={{
            ...combinedStyle,
            verticalAlign: 'middle'
          }}
          title={countryName}
        />
      )}
      {countryCode === 'GLOBE' && (
        <span
          style={combinedStyle}
          title={countryName}
        >
          🌍
        </span>
      )}
      {countryName || ''}
    </>
  );
};

/**
 * Get all available country names
 * @returns {string[]} - Array of country names
 */
export const getAvailableCountries = () => {
  return Object.keys(COUNTRY_MAP).filter(country => country !== 'Other');
};
