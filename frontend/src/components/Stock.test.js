import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Stock from './Stock';
import { portfolioAPI } from '../services/api';

jest.mock('react-router-dom', () => ({
  Link: 'a', useParams: () => ({ symbol: 'AVB' }), useNavigate: () => jest.fn(),
}), { virtual: true });
jest.mock('../context/HideAmountsContext', () => ({ useHideAmounts: () => ({ hideAmounts: false }) }));
jest.mock('../services/api', () => ({ portfolioAPI: { getInstrument: jest.fn() } }));
jest.mock('recharts', () => ({
  ResponsiveContainer: () => null, Bar: () => null, BarChart: () => null,
  CartesianGrid: () => null, Legend: () => null, Line: () => null, LineChart: () => null,
  ReferenceDot: () => null, ReferenceLine: () => null, Tooltip: () => null,
  XAxis: () => null, YAxis: () => null,
}));

beforeEach(() => {
  window.IntersectionObserver = jest.fn(() => ({ observe: jest.fn(), disconnect: jest.fn() }));
});

const payload = (active) => ({
  instrument: { symbol: 'AVB', name: 'AvalonBay', currency: 'USD', sector: 'Real Estate', is_active: active },
  fundamentals: { currentPrice: 184.06, fiftyTwoWeekHigh: 185, fiftyTwoWeekLow: 57, peRatio: 25 },
  prices: [], dcf_diff: -.68, dcf_price: 58, analyst_price_targets: { median: 201 },
  earnings_reports: [{ date: '2026-06-30', summary: 'Historical filing retained', metrics: {} }],
});

test('inactive page hides current valuation signals but retains reports and price chart', async () => {
  portfolioAPI.getInstrument.mockResolvedValue(payload(false));
  render(<Stock />);
  expect(await screen.findByRole('status')).toHaveTextContent('Inactive price feed');
  expect(screen.queryByRole('heading', { name: 'Fair Value vs Price' })).not.toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: 'Valuation Multiples' })).not.toBeInTheDocument();
  expect(screen.queryByText('-68.0%')).not.toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Price', exact: true })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Historical Earnings Reports' })).toBeInTheDocument();
  expect(screen.getByText('Historical filing retained')).toBeInTheDocument();
});

test('active page retains its current valuation panels', async () => {
  portfolioAPI.getInstrument.mockResolvedValue(payload(true));
  render(<Stock />);
  expect(await screen.findByRole('heading', { name: 'Valuation Multiples' })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Fair Value vs Price' })).toBeInTheDocument();
  expect(screen.queryByText(/Inactive price feed/)).not.toBeInTheDocument();
});
