from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/api/earnings-report/{symbol}/{report_date}", response_class=HTMLResponse)
async def get_earnings_report_html(symbol: str, report_date: str) -> HTMLResponse:
    """
    Get the full HTML earnings report file for a given symbol and date.
    For local dev server only - production should use nginx to serve static files directly.
    """
    project_root = Path(__file__).parent.parent.parent
    filings_dir = project_root / "data" / "filings" / symbol

    if not filings_dir.exists():
        raise HTTPException(status_code=404, detail=f"Report directory not found for {symbol}")

    matching_files = sorted(filings_dir.glob(f"{report_date}_*.html"))

    if not matching_files:
        raise HTTPException(status_code=404, detail=f"Report file not found for {symbol} on {report_date}")

    html_content = matching_files[0].read_text(encoding="utf-8")

    return HTMLResponse(content=html_content)
