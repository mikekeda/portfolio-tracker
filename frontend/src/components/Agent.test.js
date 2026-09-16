import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Agent from './Agent';
import { portfolioAPI } from '../services/api';

// CRA's Jest resolver predates React Router 7 package exports.
jest.mock('react-router-dom', () => ({ Link: 'a' }), { virtual: true });
jest.mock('../context/HideAmountsContext', () => ({ useHideAmounts: () => ({ hideAmounts: false }) }));
jest.mock('../services/api', () => ({ portfolioAPI: {
  getAgentSuggestions: jest.fn(), getAgentSuggestionsHistory: jest.fn(),
} }));

test('empty proposals do not claim a successful run or a correctly positioned portfolio', async () => {
  portfolioAPI.getAgentSuggestions.mockResolvedValue({ date: '2026-09-16', suggestions: [] });
  portfolioAPI.getAgentSuggestionsHistory.mockResolvedValue({ suggestions: [] });
  render(<Agent />);
  expect(await screen.findByText(/No proposals recorded for this date/)).toHaveTextContent('does not confirm that the agent ran successfully');
  expect(screen.queryByText(/book is where/)).not.toBeInTheDocument();
});

test('successful zero-action run is explicitly distinguished from missing proposals', async () => {
  const run = { status: 'success', ran_at: '2026-09-16T07:00:00Z', strategy: 'rules', intent_count: 0, order_count: 0, executable_count: 0 };
  portfolioAPI.getAgentSuggestions.mockResolvedValue({ date: '2026-09-15', suggestions: [], run, latest_run: run });
  portfolioAPI.getAgentSuggestionsHistory.mockResolvedValue({ suggestions: [] });
  render(<Agent />);
  expect(await screen.findByText('Agent completed successfully with no proposals.')).toBeInTheDocument();
  expect(screen.getByText(/Suggestions for 2026-09-15/)).toBeInTheDocument();
  expect(screen.getByText(/0 intents; 0 orders, 0 executable/)).toBeInTheDocument();
});

test('failed latest attempt stays visible while showing the last successful run', async () => {
  const run = { status: 'success', ran_at: '2026-09-15T07:00:00Z', strategy: 'rules', intent_count: 0, order_count: 0, executable_count: 0 };
  portfolioAPI.getAgentSuggestions.mockResolvedValue({ date: '2026-09-14', suggestions: [], run,
    latest_run: { status: 'failed', ran_at: '2026-09-16T07:00:00Z', reason: 'Agent run failed; see worker logs' } });
  portfolioAPI.getAgentSuggestionsHistory.mockResolvedValue({ suggestions: [] });
  render(<Agent />);
  expect(await screen.findByRole('status')).toHaveTextContent('Latest attempt failed');
  expect(screen.getByRole('status')).toHaveTextContent('Showing the last successful evaluation below');
});
