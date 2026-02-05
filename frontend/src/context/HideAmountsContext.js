import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import PropTypes from 'prop-types';

const STORAGE_KEY = 't212_hide_amounts';

const HideAmountsContext = createContext(null);

export function HideAmountsProvider({ children }) {
  const [hideAmounts, setHideAmountsState] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'false');
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(hideAmounts));
    } catch (e) {
      console.warn('Could not persist hide amounts preference', e);
    }
  }, [hideAmounts]);

  const setHideAmounts = useCallback((value) => {
    setHideAmountsState((prev) => (typeof value === 'function' ? value(prev) : value));
  }, []);

  return (
    <HideAmountsContext.Provider value={{ hideAmounts, setHideAmounts }}>
      {children}
    </HideAmountsContext.Provider>
  );
}

HideAmountsProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export function useHideAmounts() {
  const ctx = useContext(HideAmountsContext);
  if (!ctx) {
    throw new Error('useHideAmounts must be used within HideAmountsProvider');
  }
  return ctx;
}

/** Placeholder shown when amounts are hidden */
export const MASK = '•••';
