import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import { squarify } from '../utils/squarify';

const HEADER_STRIP = 22;
const GROUP_PADDING = 1;
const TOOLTIP_OFFSET = 12;
const TOOLTIP_W = 220;
const TOOLTIP_H = 100;

// Finviz-style 7 zones: 3 green shades, neutral grey (covers small moves
// inside +/-low), 3 red shades. `bands` scales per period.
const tileFill = (changePct, bands, isOther) => {
  if (isOther) return '#3a3a3a';
  if (changePct >= bands.strong)  return '#00C853';
  if (changePct >= bands.mid)     return '#00A846';
  if (changePct >= bands.low)     return '#1c7a3f';
  if (changePct >  -bands.low)    return '#2c2c2c';
  if (changePct >  -bands.mid)    return '#7a1f1f';
  if (changePct >  -bands.strong) return '#CC2929';
  return '#E63838';
};

const formatPct = (n) => `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;

// Roll holdings smaller than `threshold` (% of the bucket's value) into a
// synthetic "Other" tile. Applied per-group so a sector's long tail collapses
// independently of the rest of the portfolio. We never mutate the input so
// returning the array as-is in early-exit branches is safe.
const collapseOther = (items, threshold, expand) => {
  if (expand || items.length <= 1 || threshold <= 0) return items;
  const total = items.reduce((sum, it) => sum + (it.value || 0), 0);
  if (total <= 0) return items;

  const big = [];
  const small = [];
  for (const it of items) {
    const share = (it.value / total) * 100;
    if (share >= threshold) big.push(it);
    else small.push(it);
  }

  if (small.length <= 1) return items;

  const otherValue = small.reduce((sum, it) => sum + it.value, 0);
  const otherWeighted = small.reduce((sum, it) => sum + it.change_pct * it.value, 0);
  const otherChange = otherValue > 0 ? otherWeighted / otherValue : 0;

  return [
    ...big,
    {
      symbol: '__other__',
      name: 'Other',
      value: otherValue,
      change_pct: otherChange,
      isOther: true,
      count: small.length,
    },
  ];
};

const Tile = React.memo(({ rect, bands, navigate, onOtherClick, onHover, onLeave, totalValue }) => {
  const { x, y, w, h } = rect;
  if (w <= 0 || h <= 0) return null;

  const isOther = rect.isOther === true;
  const changePct = rect.change_pct || 0;
  const isPositive = changePct >= 0;
  const fill = tileFill(changePct, bands, isOther);

  const allocation = totalValue > 0 ? (rect.value / totalValue) * 100 : 0;

  const handleClick = () => {
    if (isOther) {
      if (onOtherClick) onOtherClick();
      return;
    }
    const symbol = rect.symbol || rect.name;
    if (symbol && navigate) navigate(`/stock/${encodeURIComponent(symbol)}`);
  };

  const handleMouseMove = (e) => {
    if (!onHover) return;
    // Clamp tooltip into the viewport so big tiles near the right/bottom edges
    // don't push it offscreen. Width/height are upper bounds; actual content
    // wraps shorter.
    const tx = Math.max(0, Math.min(e.clientX + TOOLTIP_OFFSET, window.innerWidth - TOOLTIP_W));
    const ty = Math.max(0, Math.min(e.clientY + TOOLTIP_OFFSET, window.innerHeight - TOOLTIP_H));
    onHover({
      name: isOther ? `Other (${rect.count} holdings)` : (rect.fullName || rect.name || rect.symbol),
      allocation,
      changePct,
      isOther,
      x: tx,
      y: ty,
    });
  };

  // Match the previous treemap font sizing. Tiny tiles get only the rect.
  const showLabel = w > 30 && h > 30;
  const labelSize = Math.max(9, Math.min(14, w / 4));
  const subSize = Math.max(8, Math.min(11, w / 5));

  return (
    <g
      className="alloc-tile"
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      onMouseLeave={onLeave}
      style={{ cursor: 'pointer' }}
    >
      <rect
        x={x + 0.5}
        y={y + 0.5}
        width={Math.max(0, w - 1)}
        height={Math.max(0, h - 1)}
        fill={fill}
        stroke="#fff"
        strokeWidth={1}
      />
      {showLabel && (
        <>
          <text
            x={x + w / 2}
            y={y + h / 2 - 2}
            textAnchor="middle"
            fill="white"
            fontSize={labelSize}
            fontWeight={700}
            style={{ textShadow: '0 1px 2px rgba(0,0,0,0.3)', pointerEvents: 'none' }}
          >
            {isOther ? `Other (${rect.count || 0})` : (rect.symbol || rect.name)}
          </text>
          <text
            x={x + w / 2}
            y={y + h / 2 + (h < 50 ? 10 : 15)}
            textAnchor="middle"
            fill="white"
            fontSize={subSize}
            fontWeight={500}
            style={{ textShadow: '0 1px 2px rgba(0,0,0,0.3)', pointerEvents: 'none' }}
          >
            {isPositive ? '+' : ''}{changePct.toFixed(2)}%
          </text>
        </>
      )}
    </g>
  );
});

Tile.displayName = 'Tile';

Tile.propTypes = {
  rect: PropTypes.object.isRequired,
  bands: PropTypes.shape({
    low: PropTypes.number.isRequired,
    mid: PropTypes.number.isRequired,
    strong: PropTypes.number.isRequired,
  }).isRequired,
  navigate: PropTypes.func,
  onOtherClick: PropTypes.func,
  onHover: PropTypes.func,
  onLeave: PropTypes.func,
  totalValue: PropTypes.number,
};

Tile.defaultProps = {
  navigate: null,
  onOtherClick: null,
  onHover: null,
  onLeave: null,
  totalValue: 0,
};

const GroupHeader = React.memo(({ rect, name, changePct, count, onSelect, isFiltered }) => {
  if (rect.w <= 0 || rect.h <= 0) return null;
  const isPositive = changePct >= 0;
  const stripHeight = Math.min(HEADER_STRIP, rect.h);
  const handleClick = () => onSelect && onSelect(name);
  return (
    <g
      className="alloc-group-header"
      onClick={handleClick}
      style={{ cursor: 'pointer' }}
    >
      <rect
        x={rect.x}
        y={rect.y}
        width={rect.w}
        height={stripHeight}
        fill={isFiltered ? '#0d6efd' : '#1a1a1a'}
        stroke="#fff"
        strokeWidth={1}
      />
      {rect.w > 60 && (
        <>
          <text
            x={rect.x + 6}
            y={rect.y + 14}
            fill="#fff"
            fontSize={11}
            fontWeight={700}
            style={{ textTransform: 'uppercase', pointerEvents: 'none' }}
          >
            {name} <tspan fill="#aaa" fontWeight={500}>· {count}</tspan>
          </text>
          {rect.w > 130 && (
            <text
              x={rect.x + rect.w - 6}
              y={rect.y + 14}
              fill={isPositive ? '#4ade80' : '#f87171'}
              fontSize={10}
              fontWeight={600}
              textAnchor="end"
              style={{ pointerEvents: 'none' }}
            >
              {formatPct(changePct)}
            </text>
          )}
        </>
      )}
    </g>
  );
});

GroupHeader.displayName = 'GroupHeader';

GroupHeader.propTypes = {
  rect: PropTypes.object.isRequired,
  name: PropTypes.string.isRequired,
  changePct: PropTypes.number.isRequired,
  count: PropTypes.number.isRequired,
  onSelect: PropTypes.func,
  isFiltered: PropTypes.bool,
};

GroupHeader.defaultProps = {
  onSelect: null,
  isFiltered: false,
};

const AllocationTreemap = ({
  data,
  groupBy,
  selectedGroup,
  onSelectGroup,
  expandOther,
  onOtherClick,
  bands,
  otherThreshold,
  height,
  navigate,
}) => {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(Math.floor(entry.contentRect.width));
      }
    });
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const groupKey = groupBy === 'sector' ? 'sector' : groupBy === 'region' ? 'country' : null;

  const layout = useMemo(() => {
    if (width <= 0 || !data || data.length === 0) return null;

    // Filtered view (single group selected) renders flat - no inner header
    // strip needed because the chip + caption already say which group it is.
    const filtered = groupKey && selectedGroup
      ? data.filter((d) => (d[groupKey] || 'Other') === selectedGroup)
      : data;
    const totalValue = filtered.reduce((sum, d) => sum + (d.value || 0), 0);

    if (!groupKey || selectedGroup) {
      const items = collapseOther(filtered, otherThreshold, expandOther);
      const rects = squarify(items, { x: 0, y: 0, w: width, h: height });
      return {
        totalValue,
        groups: [{ rect: { x: 0, y: 0, w: width, h: height }, header: null, children: rects }],
      };
    }

    // Two-level layout: outer squarify of groups, then squarify of each
    // group's children inside its rect minus a header strip.
    const buckets = new Map();
    for (const d of filtered) {
      const name = d[groupKey] || 'Other';
      if (!buckets.has(name)) {
        buckets.set(name, { name, value: 0, weighted: 0, count: 0, children: [] });
      }
      const b = buckets.get(name);
      b.value += d.value;
      b.weighted += d.change_pct * d.value;
      b.count += 1;
      b.children.push(d);
    }
    const groups = Array.from(buckets.values()).map((b) => ({
      ...b,
      change_pct: b.value > 0 ? b.weighted / b.value : 0,
    }));

    const groupRects = squarify(groups, { x: 0, y: 0, w: width, h: height });
    const built = groupRects.map((gr) => {
      const innerRect = {
        x: gr.x + GROUP_PADDING,
        y: gr.y + HEADER_STRIP,
        w: Math.max(0, gr.w - 2 * GROUP_PADDING),
        h: Math.max(0, gr.h - HEADER_STRIP - GROUP_PADDING),
      };
      const childItems = collapseOther(gr.children, otherThreshold, expandOther);
      const childRects = squarify(childItems, innerRect);
      return {
        rect: { x: gr.x, y: gr.y, w: gr.w, h: gr.h },
        header: { name: gr.name, change_pct: gr.change_pct, count: gr.count },
        children: childRects,
      };
    });

    return { totalValue, groups: built };
  }, [data, groupKey, selectedGroup, expandOther, otherThreshold, width, height]);

  // Stable refs so React.memo on Tile/GroupHeader can skip re-renders when
  // only the local hover state changes.
  const handleHover = useCallback((info) => setHover(info), []);
  const handleLeave = useCallback(() => setHover(null), []);

  return (
    <div ref={ref} className="alloc-treemap-container" style={{ width: '100%', height, position: 'relative' }}>
      {width > 0 && layout && (
        <svg width={width} height={height}>
          {layout.groups.map((g, gi) => (
            <g key={gi}>
              {g.header && (
                <GroupHeader
                  rect={g.rect}
                  name={g.header.name}
                  changePct={g.header.change_pct}
                  count={g.header.count}
                  isFiltered={selectedGroup === g.header.name}
                  onSelect={onSelectGroup}
                />
              )}
              {g.children.map((c, ci) => (
                <Tile
                  key={`${gi}-${ci}-${c.symbol || c.name}`}
                  rect={c}
                  bands={bands}
                  navigate={navigate}
                  onOtherClick={onOtherClick}
                  onHover={handleHover}
                  onLeave={handleLeave}
                  totalValue={layout.totalValue}
                />
              ))}
            </g>
          ))}
        </svg>
      )}
      {hover && (
        <div
          className="alloc-tooltip"
          style={{
            position: 'fixed',
            left: hover.x,
            top: hover.y,
            pointerEvents: 'none',
          }}
        >
          <div className="alloc-tooltip-title">{hover.name}</div>
          <div className="alloc-tooltip-row">allocation: {hover.allocation.toFixed(2)}%</div>
          <div className={`alloc-tooltip-row ${hover.changePct >= 0 ? 'positive' : 'negative'}`}>
            {hover.changePct >= 0 ? 'up' : 'down'}: {Math.abs(hover.changePct).toFixed(2)}%{hover.isOther ? ' (weighted)' : ''}
          </div>
          {hover.isOther && (
            <div className="alloc-tooltip-row" style={{ opacity: 0.7, fontSize: 11 }}>
              click to expand
            </div>
          )}
        </div>
      )}
    </div>
  );
};

AllocationTreemap.propTypes = {
  data: PropTypes.array.isRequired,
  groupBy: PropTypes.oneOf(['flat', 'sector', 'region']).isRequired,
  selectedGroup: PropTypes.string,
  onSelectGroup: PropTypes.func,
  expandOther: PropTypes.bool,
  onOtherClick: PropTypes.func,
  bands: PropTypes.shape({
    low: PropTypes.number.isRequired,
    mid: PropTypes.number.isRequired,
    strong: PropTypes.number.isRequired,
  }).isRequired,
  otherThreshold: PropTypes.number,
  height: PropTypes.number,
  navigate: PropTypes.func,
};

AllocationTreemap.defaultProps = {
  selectedGroup: null,
  onSelectGroup: null,
  expandOther: false,
  onOtherClick: null,
  otherThreshold: 0.5,
  height: 600,
  navigate: null,
};

export default AllocationTreemap;
