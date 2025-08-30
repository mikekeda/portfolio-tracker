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
            Dashboard
          </Link>
          <Link
            to="/holdings"
            className={location.pathname === '/holdings' ? 'active' : ''}
          >
            Holdings
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
