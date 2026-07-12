"""Performance metrics for backtest equity curves."""

import numpy as np
import pandas as pd

from backend.utils.drawdown import compute_max_drawdown
from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def summarize(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """CAGR, Sharpe/Sortino, max drawdown, turnover and cost stats."""
    returns = equity.pct_change().dropna()
    n_days = len(returns)
    if n_days < 2 or equity.iloc[0] <= 0:
        return {"error": "not enough data"}

    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    excess = returns.mean() * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    dd = compute_max_drawdown(list(equity.values))

    traded = float(trades["value_gbp"].sum()) if not trades.empty else 0.0
    fees = float(trades["fee_gbp"].sum()) if not trades.empty else 0.0
    return {
        "start": equity.index[0].isoformat() if hasattr(equity.index[0], "isoformat") else str(equity.index[0]),
        "end": equity.index[-1].isoformat() if hasattr(equity.index[-1], "isoformat") else str(equity.index[-1]),
        "final_value": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "annual_volatility_pct": round(float(ann_vol) * 100, 2),
        "sharpe": round(float(excess / ann_vol), 2) if ann_vol > 0 else None,
        "sortino": round(float(excess / downside), 2) if downside and downside > 0 else None,
        "max_drawdown_pct": round(dd["max_drawdown_pct"], 2),
        "annual_turnover_pct": round(traded / float(equity.mean()) / years * 100, 1),
        "total_fees_gbp": round(fees, 2),
        "n_trades": int(len(trades)),
    }
