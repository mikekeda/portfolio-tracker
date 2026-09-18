import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import Transactions from './Transactions';
import { portfolioAPI } from '../services/api';

jest.mock('react-router-dom', () => ({ Link: 'a' }), { virtual: true });
jest.mock('../context/HideAmountsContext', () => ({ useHideAmounts: () => ({ hideAmounts: false }) }));
jest.mock('../services/api', () => ({ portfolioAPI: { getTransactions: jest.fn() } }));
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => <div>{children}</div>,
  BarChart: ({ data }) => <div data-testid="dividend-chart">{JSON.stringify(data)}</div>,
  Bar: () => null, CartesianGrid: () => null, ReferenceLine: () => null,
  Tooltip: () => null, XAxis: () => null, YAxis: () => null,
}));

const dividend = (date, ticker, total) => ({ id: `${date}-${ticker}`, date, ticker, name: ticker, total,
  action: 'Dividend (Dividend)', fees: 0, quantity: 1 });

beforeEach(() => {
  portfolioAPI.getTransactions.mockResolvedValue({
    years: [2026, 2025], transactions: [
      dividend('2026-06-01', 'NEW', 20), dividend('2026-01-01', 'NEW', 10),
      dividend('2025-12-01', 'OLD', 100),
    ],
    summary: { total_dividends: 130, total_realized_gains: 0, total_interest: 0,
      total_fees: 0, total_deposited: 0 },
    top_dividend_payers: [{ ticker: 'OLD', total: 100 }, { ticker: 'NEW', total: 30 }],
    dividends_chart: [],
  });
});

test('year selection changes payer count, top payers and chart together', async () => {
  render(<Transactions />);
  expect(await screen.findByText('2 paying stocks')).toBeInTheDocument();
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '2026' } });
  expect(screen.getByText('1 paying stocks')).toBeInTheDocument();
  const payers = screen.getByRole('heading', { name: 'Top 1 Dividend Payers' }).parentElement;
  expect(within(payers).getByText('£30.00')).toBeInTheDocument();
  expect(within(payers).queryByText('OLD')).not.toBeInTheDocument();
  expect(screen.getByTestId('dividend-chart')).not.toHaveTextContent('2025-12');
});

test('date range applies to payer totals and chart; empty range clears both', async () => {
  render(<Transactions />);
  await screen.findByText('2 paying stocks');
  fireEvent.change(screen.getByTitle('From date'), { target: { value: '2026-02-01' } });
  expect(screen.getByRole('heading', { name: 'Dividend Income — Selected Period' })).toBeInTheDocument();
  expect(screen.getByTestId('dividend-chart')).toHaveTextContent('"amount":20');
  expect(screen.getByTestId('dividend-chart')).not.toHaveTextContent('2026-01');
  const payers = screen.getByRole('heading', { name: 'Top 1 Dividend Payers' }).parentElement;
  expect(within(payers).getByText('£20.00')).toBeInTheDocument();
  fireEvent.change(screen.getByTitle('From date'), { target: { value: '2027-01-01' } });
  expect(screen.getByText('0 paying stocks')).toBeInTheDocument();
  expect(screen.getByText('No dividends recorded')).toBeInTheDocument();
  expect(screen.getByText('No dividends in this period')).toBeInTheDocument();
});
