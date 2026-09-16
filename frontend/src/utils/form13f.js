/** Quarter-end-price estimate, not actual cash spent or received. */
export function holderFlow(holder) {
  // Explicit null from the new API means the comparison is unverified. Only
  // fall back for legacy responses that do not include the adjusted field.
  const previous = Object.prototype.hasOwnProperty.call(holder, 'shares_prev_adjusted')
    ? holder.shares_prev_adjusted : holder.shares_prev;
  if (![holder.shares, holder.value, previous].every(Number.isFinite) || holder.shares <= 0) return null;
  return (holder.shares - previous) * (holder.value / holder.shares);
}
