/**
 * Calendar days from Jan 1 through today, for the "YTD" range selectors.
 * Compared as UTC midnights so a DST boundary can't shift the count.
 */
export const ytdDays = () => {
  const now = new Date();
  const elapsed = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) - Date.UTC(now.getFullYear(), 0, 1);
  return Math.max(1, elapsed / 86400000);
};
