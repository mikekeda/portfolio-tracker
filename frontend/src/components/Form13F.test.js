import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { PositionCard } from './Form13F';

jest.mock('react-router-dom', () => ({ Link: 'a', useParams: jest.fn(), useNavigate: jest.fn() }), { virtual: true });
jest.mock('../services/api', () => ({ get: jest.fn() }));

test.each([
  ['+20.0%', 2000000, -3000000, 'positive', 'Est. flow +$2M'],
  ['-20.0%', -2000000, 3000000, 'negative', 'Est. flow -$2M'],
])('card %s follows share flow even when market value moves the other way', (change, estimated_flow, value_change, className, label) => {
  render(<PositionCard position={{ name: 'Example', value: 10000000, change, estimated_flow, value_change }} />);
  expect(screen.getByText(label)).toHaveClass(className);
  expect(screen.queryByText(/\$3M/)).not.toBeInTheDocument();
});

test('closed position does not pretend previous market value is actual sale proceeds', () => {
  render(<PositionCard position={{ name: 'Example', value: 0, value_prev: 10000000, change: 'Closed', value_change: -10000000 }} />);
  expect(screen.queryByText(/Est. flow/)).not.toBeInTheDocument();
});
