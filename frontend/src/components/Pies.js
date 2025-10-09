import React, { useState, useEffect, useMemo, useCallback } from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { portfolioAPI } from '../services/api';
import './Pies.css';

// Constants - Expanded color palette for pies with many instruments
const PIE_COLORS = [
  '#ff6b6b', // Red
  '#4ecdc4', // Teal
  '#45b7d1', // Blue
  '#96ceb4', // Light Green
  '#feca57', // Yellow
  '#ff9ff3', // Pink
  '#54a0ff', // Light Blue
  '#5f27cd', // Purple
  '#00d2d3', // Cyan
  '#ff9f43', // Orange
  '#10ac84', // Emerald
  '#ee5a24', // Red Orange
  '#0984e3', // Royal Blue
  '#6c5ce7', // Lavender
  '#fd79a8', // Rose
  '#fdcb6e'  // Peach
];
const DEFAULT_PIE_COLOR = '#ff6b6b';
const PIE_CHART_SIZE = 180;
const PIE_CHART_STROKE_WIDTH = 12;
const ROUND_BAR_SIZE = 50;
const ROUND_BAR_STROKE_WIDTH = 6;

// Utility Functions
const formatCurrency = (amount) => {
  if (amount === null || amount === undefined || isNaN(amount)) {
    return '£0.00';
  }
  return new Intl.NumberFormat('en-GB', {
    style: 'currency',
    currency: 'GBP'
  }).format(amount);
};

const formatPercentage = (percentage, decimals = 2) => {
  if (percentage === null || percentage === undefined || isNaN(percentage)) {
    return '0.00%';
  }
  return `${percentage.toFixed(decimals)}%`;
};

const formatAllocation = (allocation) => {
  if (allocation === null || allocation === undefined || isNaN(allocation)) {
    return '0%';
  }
  return `${(allocation * 100).toFixed(0)}%`;
};

const generateInstrumentColor = (index) => {
  if (index < 0 || !Number.isInteger(index)) {
    return DEFAULT_PIE_COLOR;
  }
  return PIE_COLORS[index % PIE_COLORS.length];
};

const getInstrumentColor = (t212Code, originalInstruments) => {
  if (!t212Code || !Array.isArray(originalInstruments)) {
    return DEFAULT_PIE_COLOR;
  }
  const index = originalInstruments.findIndex(
    instrument => instrument.t212_code === t212Code
  );
  return index >= 0 ? generateInstrumentColor(index) : DEFAULT_PIE_COLOR;
};

const createPieChartData = (instruments) => {
  if (!Array.isArray(instruments)) {
    return [];
  }
  return instruments.map((instrument, index) => ({
    name: instrument.instrument_name || instrument.t212_code,
    current_share: instrument.current_share || 0,
    expected_share: instrument.expected_share || 0,
    color: generateInstrumentColor(index),
    t212_code: instrument.t212_code
  }));
};

// PieChart Component
const PieChart = ({
  data,
  centerValue,
  centerProfit,
  size = 200,
  strokeWidth = 20,
  showProfit = true,
  hoveredInstrument = null,
  onSliceHover = null,
  onSliceLeave = null
}) => {
  const center = size / 2;
  const radius = center - strokeWidth;

  // Helper function to convert polar coordinates to cartesian
  const polarToCartesian = (centerX, centerY, radius, angleInDegrees) => {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
    return {
      x: centerX + (radius * Math.cos(angleInRadians)),
      y: centerY + (radius * Math.sin(angleInRadians))
    };
  };

  // Generate SVG path for a pie segment (donut style)
  const generatePath = (startAngle, endAngle, innerRadius, outerRadius) => {
    const startInner = polarToCartesian(center, center, innerRadius, endAngle);
    const endInner = polarToCartesian(center, center, innerRadius, startAngle);
    const startOuter = polarToCartesian(center, center, outerRadius, startAngle);
    const endOuter = polarToCartesian(center, center, outerRadius, endAngle);

    const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";

    return [
      "M", startInner.x, startInner.y,
      "A", innerRadius, innerRadius, 0, largeArcFlag, 0, endInner.x, endInner.y,
      "L", startOuter.x, startOuter.y,
      "A", outerRadius, outerRadius, 0, largeArcFlag, 1, endOuter.x, endOuter.y,
      "L", startInner.x, startInner.y,
      "Z"
    ].join(" ");
  };

  // Calculate total for normalization
  const totalCurrent = data.reduce((sum, item) => sum + item.current_share, 0);

  // Calculate segments based on current allocation
  let currentAngle = 0;
  const innerRadius = radius * 0.6; // Inner hole radius
  const referenceRadius = radius * 0.85; // Reference circle radius

  const segments = data.map((instrument, index) => {
    // Calculate angle span based on current allocation
    const angleSpan = (instrument.current_share / totalCurrent) * 360;
    const startAngle = currentAngle;
    const endAngle = currentAngle + angleSpan;

    currentAngle += angleSpan;

    // Calculate outer radius based on expected vs current allocation
    const allocationRatio = instrument.current_share / instrument.expected_share;

    // Determine outer radius based on allocation ratio
    let outerRadius = referenceRadius; // Default to reference circle

    if (allocationRatio > 1.1) {
      // Significantly over-allocated - extend beyond reference
      outerRadius = radius * Math.min(1.0, 0.85 + (allocationRatio - 1.1) * 0.3);
    } else if (allocationRatio < 0.9) {
      // Under-allocated - stay closer to inner radius
      outerRadius = innerRadius + (referenceRadius - innerRadius) * allocationRatio;
    }

    return {
      ...instrument,
      startAngle,
      endAngle,
      path: generatePath(startAngle, endAngle, innerRadius, outerRadius),
      color: instrument.color,
      id: `segment-${index}`,
      allocationRatio,
      outerRadius
    };
  });

  return (
    <div className="pie-chart-container" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="pie-chart-svg">
        {/* Data segments */}
        {segments.map((segment) => {
          const isHovered = hoveredInstrument === segment.t212_code;
          return (
            <path
              key={segment.id}
              d={segment.path}
              fill={segment.color}
              className={`pie-segment ${isHovered ? 'pie-segment-hovered' : ''}`}
              stroke="none"
              onMouseEnter={() => onSliceHover && onSliceHover(segment)}
              onMouseLeave={() => onSliceLeave && onSliceLeave()}
              style={{
                opacity: hoveredInstrument && !isHovered ? 0.5 : 1
              }}
            />
          );
        })}
        {/* Reference circle for desired allocations */}
        <circle
          cx={center}
          cy={center}
          r={referenceRadius}
          fill="none"
          stroke="#6c757d"
          strokeWidth="1"
          opacity="0.5"
        />
      </svg>

      {/* Center content */}
      <div className="pie-chart-center">
        <div className="center-value">{centerValue}</div>
        {showProfit && centerProfit && (
          <div className="center-profit">{centerProfit}</div>
        )}
      </div>
    </div>
  );
};

// RoundBar Component
const RoundBar = ({
  percentage,
  color,
  size = 60,
  strokeWidth = 6,
  showPercentage = false
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const center = size / 2;

  // Calculate stroke dash array for the progress
  const strokeDasharray = `${(percentage / 100) * circumference} ${circumference}`;

  return (
    <div className="round-bar-container">
      <svg width={size} height={size} className="round-bar-svg">
        {/* Background circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="#e9ecef"
          strokeWidth={strokeWidth}
        />

        {/* Progress circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={strokeDasharray}
          strokeDashoffset="0"
          strokeLinecap="round"
          className="round-bar-progress"
        />
      </svg>

      {/* Center percentage text */}
      {showPercentage && (
        <div className="round-bar-text">
          {percentage.toFixed(0)}%
        </div>
      )}
    </div>
  );
};

// PieCard Component
const PieCard = React.memo(({ pie, isExpanded, onToggle }) => {
  // State for interactive highlighting
  const [hoveredInstrument, setHoveredInstrument] = useState(null);

  // Memoize pie chart data to prevent color changes
  const pieChartData = useMemo(() => {
    return createPieChartData(pie.instruments);
  }, [pie.instruments]);

  // Memoize toggle handler to prevent unnecessary re-renders
  const handleToggle = useCallback(() => {
    onToggle(pie.id);
  }, [onToggle, pie.id]);

  // Handlers for interactive highlighting (only when pie is expanded)
  const handlePieSliceHover = useCallback((instrument) => {
    if (isExpanded) {
      setHoveredInstrument(instrument.t212_code);
    }
  }, [isExpanded]);

  const handlePieSliceLeave = useCallback(() => {
    if (isExpanded) {
      setHoveredInstrument(null);
    }
  }, [isExpanded]);

  const handleInstrumentHover = useCallback((t212Code) => {
    setHoveredInstrument(t212Code);
  }, []);

  const handleInstrumentLeave = useCallback(() => {
    setHoveredInstrument(null);
  }, []);

  // Clear hovered state when pie is collapsed
  useEffect(() => {
    if (!isExpanded) {
      setHoveredInstrument(null);
    }
  }, [isExpanded]);

  return (
    <div className="pie-card">
      <div
        className="pie-header"
        onClick={handleToggle}
        role="button"
        tabIndex={0}
        aria-expanded={isExpanded}
        aria-label={`${isExpanded ? 'Collapse' : 'Expand'} pie ${pie.name || `Pie ${pie.id}`}`}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleToggle();
          }
        }}
      >
        <div className="pie-info">
          <div className="pie-summary">
            <div className="pie-chart-section">
              <PieChart
                data={pieChartData}
                centerValue=""
                centerProfit=""
                size={PIE_CHART_SIZE}
                strokeWidth={PIE_CHART_STROKE_WIDTH}
                showProfit={false}
                hoveredInstrument={isExpanded ? hoveredInstrument : null}
                onSliceHover={handlePieSliceHover}
                onSliceLeave={handlePieSliceLeave}
              />
            </div>
            <div className="pie-name">
              <h3>{pie.name || `Pie ${pie.id}`}</h3>
              {pie.goal && (
                <span className="pie-goal">Goal: {formatCurrency(pie.goal)}</span>
              )}
            </div>
            <div className="pie-value-section">
              <span className="value">{formatCurrency(pie.result.priceAvgValue)}</span>
              <span className="profit">
                {formatCurrency(pie.result.priceAvgResult)} ({formatPercentage(pie.result.priceAvgResultCoef * 100)})
              </span>
            </div>
          </div>
        </div>
        <div className="expand-icon" aria-hidden="true">
          {isExpanded ? '−' : '+'}
        </div>
      </div>

      {isExpanded && (
        <div className="pie-instruments" role="region" aria-label="Pie instruments details">
          <div className="instruments-header">
            <h4>Instruments ({pie.instruments.length})</h4>
            <span className="sort-info">Sorted by current allocation</span>
          </div>
          <div className="instruments-list" role="list">
            {pie.instruments.map((instrument) => {
              const isHovered = hoveredInstrument === instrument.t212_code;
              return (
              <div
                key={`${pie.id}-${instrument.t212_code}`}
                className={`instrument-item ${isHovered ? 'instrument-item-hovered' : ''}`}
                role="listitem"
                onMouseEnter={() => handleInstrumentHover(instrument.t212_code)}
                onMouseLeave={handleInstrumentLeave}
                style={{
                  backgroundColor: isHovered ? '#f8f9fa' : 'white',
                  borderColor: isHovered ? getInstrumentColor(instrument.t212_code, pie.instruments) : 'transparent'
                }}
              >
                <div className="instrument-icon">
                  <RoundBar
                    percentage={instrument.current_share * 100}
                    color={getInstrumentColor(instrument.t212_code, pie.instruments)}
                    size={ROUND_BAR_SIZE}
                    strokeWidth={ROUND_BAR_STROKE_WIDTH}
                    showPercentage={false}
                  />
                </div>
                <div className="instrument-details">
                  <div className="instrument-name">
                    {instrument.yahoo_symbol ? (
                      <Link className="symbol" to={`/stock/${encodeURIComponent(instrument.yahoo_symbol)}`}>
                        {instrument.instrument_name}
                      </Link>
                    ) : (
                      <span>{instrument.instrument_name}</span>
                    )}
                  </div>
                  <div className="instrument-code">{instrument.yahoo_symbol || instrument.t212_code}</div>
                </div>
                <div className="instrument-allocation">
                  <div className="allocation-percentages">
                    <span className="current-allocation-text">
                      {formatAllocation(instrument.current_share)}
                    </span>
                    <span className="separator">/</span>
                    <span className="target-allocation-text">
                      {formatAllocation(instrument.expected_share)}
                    </span>
                  </div>
                </div>

                <div className="instrument-value">
                  <span className="value">{formatCurrency(instrument.result.priceAvgValue)}</span>
                  <span className="profit">
                    {formatCurrency(instrument.result.priceAvgResult)} ({formatPercentage(instrument.result.priceAvgResultCoef * 100)})
                  </span>
                </div>
              </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
});

// PropTypes for PieCard
PieCard.propTypes = {
  pie: PropTypes.shape({
    id: PropTypes.number.isRequired,
    name: PropTypes.string,
    goal: PropTypes.number,
    instruments: PropTypes.arrayOf(PropTypes.shape({
      t212_code: PropTypes.string.isRequired,
      instrument_name: PropTypes.string,
      yahoo_symbol: PropTypes.string,
      current_share: PropTypes.number.isRequired,
      expected_share: PropTypes.number.isRequired,
      result: PropTypes.shape({
        priceAvgValue: PropTypes.number.isRequired,
        priceAvgResult: PropTypes.number.isRequired,
        priceAvgResultCoef: PropTypes.number.isRequired
      }).isRequired
    })).isRequired,
    result: PropTypes.shape({
      priceAvgValue: PropTypes.number.isRequired,
      priceAvgResult: PropTypes.number.isRequired,
      priceAvgResultCoef: PropTypes.number.isRequired
    }).isRequired
  }).isRequired,
  isExpanded: PropTypes.bool.isRequired,
  onToggle: PropTypes.func.isRequired
};

// PropTypes for PieChart
PieChart.propTypes = {
  data: PropTypes.arrayOf(PropTypes.shape({
    name: PropTypes.string.isRequired,
    current_share: PropTypes.number.isRequired,
    expected_share: PropTypes.number.isRequired,
    color: PropTypes.string,
    t212_code: PropTypes.string
  })).isRequired,
  centerValue: PropTypes.string,
  centerProfit: PropTypes.string,
  size: PropTypes.number,
  strokeWidth: PropTypes.number,
  showProfit: PropTypes.bool,
  hoveredInstrument: PropTypes.string,
  onSliceHover: PropTypes.func,
  onSliceLeave: PropTypes.func
};

PieChart.defaultProps = {
  centerValue: '',
  centerProfit: '',
  size: 200,
  strokeWidth: 20,
  showProfit: true,
  hoveredInstrument: null,
  onSliceHover: null,
  onSliceLeave: null
};

// PropTypes for RoundBar
RoundBar.propTypes = {
  percentage: PropTypes.number.isRequired,
  color: PropTypes.string.isRequired,
  size: PropTypes.number,
  strokeWidth: PropTypes.number,
  showPercentage: PropTypes.bool
};

RoundBar.defaultProps = {
  size: 60,
  strokeWidth: 6,
  showPercentage: false
};

// Main Pies Component
const Pies = () => {
  const [pies, setPies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedPies, setExpandedPies] = useState(new Set());

  useEffect(() => {
    const fetchPies = async () => {
      try {
        setLoading(true);
        const data = await portfolioAPI.getPies();
        setPies(data);
      } catch (err) {
        console.error('Error fetching pies:', err);
        setError('Failed to load pies');
      } finally {
        setLoading(false);
      }
    };

    fetchPies();
  }, []);

  const togglePieExpansion = (pieId) => {
    setExpandedPies(prev => {
      const newSet = new Set(prev);
      if (newSet.has(pieId)) {
        newSet.delete(pieId);
      } else {
        newSet.add(pieId);
      }
      return newSet;
    });
  };

  if (loading) {
    return (
      <div className="pies-container">
        <div className="loading">Loading pies...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="pies-container">
        <div className="error">{error}</div>
      </div>
    );
  }

  return (
    <div className="pies-container">
      <h2>Investment Pies</h2>

      {pies.length === 0 ? (
        <div className="no-pies">
          <p>No pies found.</p>
        </div>
      ) : (
        <div className="pies-list">
          {pies.map((pie) => (
            <PieCard
              key={pie.id}
              pie={pie}
              isExpanded={expandedPies.has(pie.id)}
              onToggle={togglePieExpansion}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default Pies;