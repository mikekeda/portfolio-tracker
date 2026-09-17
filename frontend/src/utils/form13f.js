/** Current-quarter price proxy; exits use prior value. Not actual cash flows. */
export function holderFlow(holder) {
  if (holder.shares === 0 && Number.isFinite(holder.shares_prev) && holder.shares_prev > 0) {
    // prior shares × prior price = prior value. Do not mix a split-adjusted
    // quantity with the old unadjusted price (which would multiply exits).
    return Number.isFinite(holder.value_prev) && holder.value_prev >= 0 ? -holder.value_prev : null;
  }
  // Explicit null from the new API means the comparison is unverified. Only
  // fall back for legacy responses that do not include the adjusted field.
  const previous = Object.prototype.hasOwnProperty.call(holder, 'shares_prev_adjusted')
    ? holder.shares_prev_adjusted : holder.shares_prev;
  if (![holder.shares, holder.value, previous].every(Number.isFinite) || holder.shares <= 0) return null;
  return (holder.shares - previous) * (holder.value / holder.shares);
}
