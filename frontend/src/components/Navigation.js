import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navigation.css';

const Navigation = () => {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Close mobile menu when route changes
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [location.pathname]);

  // Close mobile menu when window resizes to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 768 && isMobileMenuOpen) {
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

  const linkClass = (path) => (location.pathname === path ? 'active' : '');

  return (
    <nav className="navigation">
      <div className="nav-container">
        <div className="nav-brand">
          <Link to="/">Trading212 Portfolio</Link>
        </div>

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

        {/* Navigation links */}
        <div className={`nav-links ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
          <Link to="/" className={linkClass('/')} onClick={handleLinkClick}>
            Summary
          </Link>
          <Link to="/allocations" className={linkClass('/allocations')} onClick={handleLinkClick}>
            Allocations
          </Link>
          <Link to="/holdings" className={linkClass('/holdings')} onClick={handleLinkClick}>
            Holdings
          </Link>
          <Link to="/pies" className={linkClass('/pies')} onClick={handleLinkClick}>
            Pies
          </Link>
          <Link to="/chart" className={linkClass('/chart')} onClick={handleLinkClick}>
            Chart
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
