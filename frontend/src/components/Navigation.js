import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navigation.css';

const Navigation = () => {
  const location = useLocation();

  return (
    <nav className="navigation">
      <div className="nav-container">
        <div className="nav-brand">
          <Link to="/">Trading212 Portfolio</Link>
        </div>
        <div className="nav-links">
          <Link
            to="/"
            className={location.pathname === '/' ? 'active' : ''}
          >
            Summary
          </Link>
          <Link
            to="/allocations"
            className={location.pathname === '/allocations' ? 'active' : ''}
          >
            Allocations
          </Link>
          <Link
            to="/holdings"
            className={location.pathname === '/holdings' ? 'active' : ''}
          >
            Holdings
          </Link>
          <Link
            to="/chart"
            className={location.pathname === '/chart' ? 'active' : ''}
          >
            Chart
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
