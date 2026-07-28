import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useHideAmounts } from '../context/HideAmountsContext';
import './Navigation.css';

// Must match the responsive breakpoint in Navigation.css.
const MOBILE_BREAKPOINT = 1200;

const EyeIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const EyeOffIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
    <line x1="1" y1="1" x2="23" y2="23" />
  </svg>
);

const Navigation = () => {
  const location = useLocation();
  const { hideAmounts, setHideAmounts } = useHideAmounts();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Close mobile menu when route changes
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [location.pathname]);

  // Close mobile menu when window resizes to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > MOBILE_BREAKPOINT && isMobileMenuOpen) {
        setIsMobileMenuOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isMobileMenuOpen]);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen((prev) => !prev);
  };

  const handleLinkClick = () => {
    setIsMobileMenuOpen(false);
  };

  const linkClass = (path) =>
    location.pathname === path || (path !== '/' && location.pathname.startsWith(path)) ? 'active' : '';

  return (
    <nav className="navigation">
      <div className="nav-container">
        <div className="nav-brand">
          <Link to="/" title="Trading212 Portfolio">
            T212
          </Link>
        </div>

        {/* Navigation links */}
        <div className={`nav-links ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
          <Link to="/holdings" className={linkClass('/holdings')} onClick={handleLinkClick}>
            Holdings
          </Link>
          <Link to="/allocations" className={linkClass('/allocations')} onClick={handleLinkClick}>
            Allocations
          </Link>
          <Link to="/calendar" className={linkClass('/calendar')} onClick={handleLinkClick}>
            Calendar
          </Link>
          <Link to="/chart" className={linkClass('/chart')} onClick={handleLinkClick}>
            Chart
          </Link>
          <Link to="/pies" className={linkClass('/pies')} onClick={handleLinkClick}>
            Pies
          </Link>
          <Link to="/13f" className={linkClass('/13f')} onClick={handleLinkClick}>
            13F
          </Link>
          <Link to="/transactions" className={linkClass('/transactions')} onClick={handleLinkClick}>
            Transactions
          </Link>
          <Link to="/projection" className={linkClass('/projection')} onClick={handleLinkClick}>
            Projection
          </Link>
          <Link to="/risk" className={linkClass('/risk')} onClick={handleLinkClick}>
            Risk
          </Link>
          <Link to="/agent" className={linkClass('/agent')} onClick={handleLinkClick}>
            Agent
          </Link>
        </div>

        {/* Right-hand controls — outside .nav-links so the privacy toggle stays
            reachable on mobile without opening the menu. */}
        <div className="nav-actions">
          <button
            className="nav-hide-amounts"
            type="button"
            onClick={() => setHideAmounts(!hideAmounts)}
            aria-pressed={hideAmounts}
            aria-label={hideAmounts ? 'Show amounts' : 'Hide amounts'}
            title={hideAmounts ? 'Show amounts' : 'Hide amounts'}
          >
            {hideAmounts ? <EyeOffIcon /> : <EyeIcon />}
          </button>

          {/* Hamburger button for mobile */}
          <button
            className="mobile-menu-toggle"
            aria-label="Toggle navigation menu"
            aria-expanded={isMobileMenuOpen}
            onClick={toggleMobileMenu}
          >
            <span className={`hamburger ${isMobileMenuOpen ? 'open' : ''}`}>
              <span></span>
              <span></span>
              <span></span>
            </span>
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
