import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import Holdings from './components/Holdings';
import Chart from './components/Chart';
import Stock from './components/Stock';
import Allocations from './components/Allocations';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/allocations" element={<Allocations />} />
            <Route path="/holdings" element={<Holdings />} />
            <Route path="/chart" element={<Chart />} />
            <Route path="/stock/:symbol" element={<Stock />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
