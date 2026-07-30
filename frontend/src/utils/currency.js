const CURRENCY_SYMBOLS = {
  GBP: '£',
  USD: '$',
  EUR: '€',
  GBX: 'p',
  CHF: 'Fr',
  JPY: '¥',
  CAD: 'CA$',
  AUD: 'A$',
  SEK: 'kr',
  NOK: 'kr',
  DKK: 'kr',
  HKD: 'HK$',
  SGD: 'S$',
};

export const currencySymbol = (code) => CURRENCY_SYMBOLS[code] || (code ? `${code} ` : '');

/** Split a natively-quoted price into its display symbol and value. */
export const nativePrice = (price, currency) =>
  // GBX is quoted in pence; show pounds so it compares with every other price.
  currency === 'GBX' ? { symbol: '£', value: price / 100 } : { symbol: currencySymbol(currency), value: price };
