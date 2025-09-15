import React from 'react';

// Reusable tooltip component that sorts entries by value
const SharedTooltip = ({ active, payload, label, prefix = '', valueFormatter = (value) => value.toFixed(2) }) => {
  if (active && payload && payload.length) {
    // Sort payload by value (descending)
    const sortedPayload = [...payload].sort((a, b) => {
      const aValue = typeof a.value === 'number' ? a.value : parseFloat(a.value) || 0;
      const bValue = typeof b.value === 'number' ? b.value : parseFloat(b.value) || 0;
      return bValue - aValue;
    });

    return (
      <div className="custom-tooltip">
        <p className="tooltip-label">{label}</p>
        {sortedPayload.map((entry, index) => {
          const name = entry.dataKey ? entry.dataKey.replace(prefix, '') : entry.name;
          const value = entry.value;
          const color = entry.color;
          const formattedValue = valueFormatter(value);

          return (
            <p key={index} className="tooltip-item" style={{ color }}>
              {name}: {formattedValue}
            </p>
          );
        })}
      </div>
    );
  }
  return null;
};

export default SharedTooltip;
