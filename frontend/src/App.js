import React, { useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { HideAmountsProvider } from './context/HideAmountsContext';
import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import Holdings from './components/Holdings';
import Pies from './components/Pies';
import Chart from './components/Chart';
import Stock from './components/Stock';
import Allocations from './components/Allocations';
import Login from './components/Login';
import Form13F from './components/Form13F';
import EarningsCalendar from './components/EarningsCalendar';
import Transactions from './components/Transactions';
import './App.css';

const RELOAD_THRESHOLD_MS = 15 * 60 * 1000; // 15 minutes — reload stale tabs on visibility

function AppContent() {
  const location = useLocation();
  const isLoginPage = location.pathname === '/login';
  const pageLoadTime = useRef(Date.now());

  useEffect(() => {
    const handleVisibilityChange = () => {
      // Only reload when tab becomes visible (user switches back to it)
      if (document.visibilityState === 'visible') {
        const timeSinceLoad = Date.now() - pageLoadTime.current;
        if (timeSinceLoad >= RELOAD_THRESHOLD_MS) {
          window.location.reload();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  return (
    <div className="App">
      {!isLoginPage && <Navigation />}
      <main className={isLoginPage ? 'main-content-full' : 'main-content'}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<Dashboard />} />
          <Route path="/allocations" element={<Allocations />} />
          <Route path="/holdings" element={<Holdings />} />
          <Route path="/pies" element={<Pies />} />
          <Route path="/chart" element={<Chart />} />
          <Route path="/stock/:symbol" element={<Stock />} />
          <Route path="/13f" element={<Form13F />} />
          <Route path="/13f/:managerId" element={<Form13F />} />
          <Route path="/calendar" element={<EarningsCalendar />} />
          <Route path="/transactions" element={<Transactions />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <Router>
      <HideAmountsProvider>
        <AppContent />
      </HideAmountsProvider>
    </Router>
  );
}

export default App;
