-- Uppercase form13f_holdings.cusip rows stored as filed.
-- _save_to_db wrote the raw string but matched the instrument map on .upper(), and the
-- consensus dicts in backend/views/form13f.py key on the raw CUSIP. Akre files KKR as
-- 48251w104 while the other holders use 48251W104, so KKR split into two consensus rows
-- in every stored quarter. 6 rows affected as of 2026-08-18, all Akre/KKR. Write-side fix
-- is normalize_cusip() in scripts/scrape_13f.py. Idempotent.

BEGIN;

UPDATE form13f_holdings
SET cusip = upper(cusip)
WHERE cusip <> upper(cusip);

-- Expect 0 rows.
SELECT count(*) AS still_lowercase
FROM form13f_holdings
WHERE cusip <> upper(cusip);

COMMIT;
