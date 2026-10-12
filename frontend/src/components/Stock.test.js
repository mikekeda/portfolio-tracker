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

test('valuation currency and withheld DCF are explained without raw P/B', async () => {
  const data = payload(true);
  data.fundamentals.valuationCurrency = 'USD';
  data.fundamentals.priceToBook = 1342;
  data.dcf_price = null;
  data.dcf_diff = null;
  data.dcf_unavailable_reason = 'Statement and summary currencies differ.';
  portfolioAPI.getInstrument.mockResolvedValue(data);
  render(<Stock />);
  expect(await screen.findByText(/Amounts in USD/)).toBeInTheDocument();
  expect(screen.getByText(/DCF unavailable: Statement and summary currencies differ/)).toBeInTheDocument();
  expect(screen.queryByText('P/B', { exact: true })).not.toBeInTheDocument();
  expect(screen.queryByText('1342')).not.toBeInTheDocument();
});

test('zero FCF yield is measured but Price/FCF is undefined', async () => {
  const data = payload(true);
  data.fundamentals.fcfYield = 0;
  data.fundamentals.valuationIssues = {};
  portfolioAPI.getInstrument.mockResolvedValue(data);
  render(<Stock />);
  const yieldTile = (await screen.findByText('FCF Yield', { exact: true })).closest('.kpi');
  const multipleTile = screen.getByText('Price/FCF', { exact: true }).closest('.kpi');
  expect(yieldTile.querySelector('.kpi-value')).toHaveTextContent('0.00%');
  expect(multipleTile.querySelector('.kpi-value')).toHaveTextContent('N/M');
  expect(multipleTile).toHaveAttribute('title', 'Price/FCF is undefined because free cash flow is zero.');
});

test('missing debt, cash and FCF explain their unavailable inputs', async () => {
  const data = payload(true);
  data.fundamentals = {
    ...data.fundamentals, totalDebt: null, totalCash: null, fcfYield: null,
    valuationIssues: {
      totalDebt: 'Unavailable: debt is missing.',
      totalCash: 'Unavailable: currency or exchange rate is missing.',
      fcfYield: 'Unavailable: cash flow is missing.',
    },
  };
  portfolioAPI.getInstrument.mockResolvedValue(data);
  render(<Stock />);
  await screen.findByText('FCF Yield', { exact: true });
  for (const [label, key] of [['Debt', 'totalDebt'], ['Cash', 'totalCash'], ['FCF Yield', 'fcfYield'], ['Price/FCF', 'fcfYield']]) {
    const tile = screen.getByText(label, { exact: true }).closest('.kpi');
    expect(tile.querySelector('.kpi-value')).toHaveTextContent('-');
    expect(tile).toHaveAttribute('title', data.fundamentals.valuationIssues[key]);
  }
});

test.each([10, -10])('nonzero FCF yield %s retains a finite signed multiple', async (yieldValue) => {
  const data = payload(true);
  data.fundamentals.fcfYield = yieldValue;
  portfolioAPI.getInstrument.mockResolvedValue(data);
  render(<Stock />);
  const tile = (await screen.findByText('Price/FCF', { exact: true })).closest('.kpi');
  const value = tile.querySelector('.kpi-value');
  expect(parseFloat(value.textContent)).toBe(100 / yieldValue);
  expect(value).toHaveClass(yieldValue < 0 ? 'negative' : 'positive');
});
