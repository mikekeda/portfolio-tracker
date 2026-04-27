// Squarified treemap layout (Bruls, Huijsen, van Wijk, 2000).
//
// Given items with numeric `value` fields and a target rectangle, returns the
// items annotated with `x`, `y`, `w`, `h` such that each rectangle has area
// proportional to its value and the overall layout favours near-square tiles.
//
// We use this instead of recharts' built-in nesting because we need to reserve
// a header strip inside parent group rects (finviz-style "TECHNOLOGY ..."
// labels), which recharts squarify can't do - children always tile the full
// parent bbox and cover any header drawn there.

const worstAspectRatio = (row, shortSide) => {
  if (row.length === 0) return Infinity;
  const total = row.reduce((sum, item) => sum + item._area, 0);
  if (total <= 0 || shortSide <= 0) return Infinity;
  const longSide = total / shortSide;
  let worst = 0;
  for (const item of row) {
    const itemSide = item._area / longSide;
    if (itemSide <= 0) return Infinity;
    const ratio = Math.max(longSide / itemSide, itemSide / longSide);
    if (ratio > worst) worst = ratio;
  }
  return worst;
};

export function squarify(items, rect) {
  if (!Array.isArray(items) || items.length === 0) return [];
  if (!rect || rect.w <= 0 || rect.h <= 0) return [];

  const positiveItems = items.filter((it) => (it.value || 0) > 0);
  if (positiveItems.length === 0) return [];

  const sorted = [...positiveItems].sort((a, b) => b.value - a.value);
  const total = sorted.reduce((sum, it) => sum + it.value, 0);
  const scale = (rect.w * rect.h) / total;
  const remaining = sorted.map((it) => ({ ...it, _area: it.value * scale }));

  const out = [];
  let { x, y, w, h } = rect;

  while (remaining.length > 0 && w > 0 && h > 0) {
    const shortSide = Math.min(w, h);

    // Greedily grow the row while the worst aspect ratio doesn't get worse.
    let row = [remaining[0]];
    let i = 1;
    while (i < remaining.length) {
      const candidate = [...row, remaining[i]];
      if (worstAspectRatio(candidate, shortSide) <= worstAspectRatio(row, shortSide)) {
        row = candidate;
        i += 1;
      } else {
        break;
      }
    }

    const rowSum = row.reduce((sum, it) => sum + it._area, 0);

    if (w <= h) {
      // Lay row across the top.
      const rowHeight = rowSum / w;
      let cursor = x;
      for (const item of row) {
        const itemWidth = item._area / rowHeight;
        // eslint-disable-next-line no-unused-vars
        const { _area, ...rest } = item;
        out.push({ ...rest, x: cursor, y, w: itemWidth, h: rowHeight });
        cursor += itemWidth;
      }
      y += rowHeight;
      h -= rowHeight;
    } else {
      // Lay row down the left.
      const rowWidth = rowSum / h;
      let cursor = y;
      for (const item of row) {
        const itemHeight = item._area / rowWidth;
        // eslint-disable-next-line no-unused-vars
        const { _area, ...rest } = item;
        out.push({ ...rest, x, y: cursor, w: rowWidth, h: itemHeight });
        cursor += itemHeight;
      }
      x += rowWidth;
      w -= rowWidth;
    }

    remaining.splice(0, row.length);
  }

  return out;
}
