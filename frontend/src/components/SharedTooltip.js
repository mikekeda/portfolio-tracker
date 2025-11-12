import React from 'react';
import PropTypes from 'prop-types';

// Reusable tooltip component that sorts entries by value or custom order
const SharedTooltip = ({
  active,
  payload,
  label,
  prefix = '',
  valueFormatter = (value) => value.toFixed(2),
  nameMap = {},
  sortOrder = null,
}) => {
  if (active && payload && payload.length) {
    let sortedPayload;

    if (sortOrder) {
      // Sort by custom order
      sortedPayload = [...payload].sort((a, b) => {
        const aKey = a.dataKey || a.name;
        const bKey = b.dataKey || b.name;
        const aIndex = sortOrder.indexOf(aKey);
        const bIndex = sortOrder.indexOf(bKey);
        return aIndex - bIndex;
      });
    } else {
      // Sort payload by value (descending)
      sortedPayload = [...payload].sort((a, b) => {
        const aValue = typeof a.value === 'number' ? a.value : parseFloat(a.value) || 0;
        const bValue = typeof b.value === 'number' ? b.value : parseFloat(b.value) || 0;
        return bValue - aValue;
      });
    }

    return (
      <div className="custom-tooltip">
        <p className="tooltip-label">{label}</p>
        {sortedPayload.map((entry, index) => {
          const rawName = entry.dataKey || entry.name;
          const name = nameMap[rawName] || rawName.replace(prefix, '');
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

SharedTooltip.propTypes = {
  active: PropTypes.bool,
  payload: PropTypes.arrayOf(
    PropTypes.shape({
      dataKey: PropTypes.string,
      name: PropTypes.string,
      value: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
      color: PropTypes.string,
    })
  ),
  label: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  prefix: PropTypes.string,
  valueFormatter: PropTypes.func,
  nameMap: PropTypes.objectOf(PropTypes.string),
  sortOrder: PropTypes.arrayOf(PropTypes.string),
};

export default SharedTooltip;
