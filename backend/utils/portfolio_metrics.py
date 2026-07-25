"""
Portfolio Performance Metrics Calculator

Calculates Tier 1 portfolio metrics from Trading212 transaction history:
1. Total & Annualized Returns
2. Maximum Drawdown
3. Sharpe/Sortino Ratios
4. Fee Impact Analysis
5. Additional Risk Metrics (Calmar, Omega, Tail Ratios)

This module provides comprehensive portfolio performance analysis using
complete transaction history data with robust manual implementations of
industry-standard risk-adjusted performance metrics.
"""

from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import DAYS_PER_YEAR, RISK_FREE_RATE
from models import TransactionHistory


def calculate_total_return(
    total_invested: float, current_portfolio_value: float, total_proceeds: float = 0.0
) -> dict[str, float]:
    """
    Calculate total return metrics.

    Args:
        total_invested: Actual cash deposited (deposits - withdrawals)
        current_portfolio_value: Current portfolio value
        total_proceeds: Total proceeds from sales (optional, not used in return calculation)

    Returns:
        Dictionary with return metrics
    """
    # Note: total_proceeds is not subtracted because the current_portfolio_value
    # already reflects the cash from sales (it's part of your portfolio).
    # We only care about: how much did you deposit vs. what's it worth now?
    net_invested = total_invested

    # Total return calculation
    if net_invested <= 0:
        total_return_pct = 0.0
        total_return_abs = current_portfolio_value
    else:
        total_return_abs = current_portfolio_value - net_invested
        total_return_pct = (total_return_abs / net_invested) * 100

    return {
        "total_return_abs": total_return_abs,
        "total_return_pct": total_return_pct,
        "net_invested": net_invested,
        "current_value": current_portfolio_value,
        "total_invested": total_invested,
        "total_proceeds": total_proceeds,
    }


def calculate_annualized_return(total_return_pct: float, start_date: date, end_date: date = None) -> dict[str, float]:
    """
    Calculate annualized return.

    Args:
        total_return_pct: Total return percentage
        start_date: Investment start date
        end_date: End date (defaults to today)

    Returns:
        Dictionary with annualized return metrics
    """
    if end_date is None:
        end_date = date.today()

    # Calculate days between dates
    days_held = (end_date - start_date).days

    if days_held <= 0:
        return {"annualized_return_pct": 0.0, "days_held": 0}

    # Convert percentage to decimal
    total_return_decimal = total_return_pct / 100.0

    # Calculate annualized return
    years_held = days_held / DAYS_PER_YEAR  # Account for leap years

    if years_held <= 0:
        annualized_return_pct = 0.0
    else:
        annualized_return_pct = ((1 + total_return_decimal) ** (1 / years_held) - 1) * 100

    return {
        "annualized_return_pct": annualized_return_pct,
        "days_held": days_held,
        "years_held": years_held,
        "total_return_pct": total_return_pct,
    }


async def calculate_daily_portfolio_values_with_interpolation(
    session: AsyncSession,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_portfolio_value: float = 0.0,
) -> dict[date, float]:
    """
    Calculate daily portfolio values from transaction history with interpolation.

    This creates a complete daily portfolio value history by:
    1. Starting with current portfolio value
    2. Working backwards through transactions to get key portfolio values
    3. Interpolating between transaction dates for missing days
    4. Creating a complete daily timeline

    Args:
        session: Async database session
        start_date: Start date for analysis (optional)
        end_date: End date for analysis (optional)
        current_portfolio_value: Current portfolio value (starting point)

    Returns:
        Dictionary mapping dates to portfolio values
    """
    # Get all transactions in date range
    query = select(TransactionHistory).order_by(TransactionHistory.timestamp.desc())

    if start_date:
        query = query.filter(TransactionHistory.timestamp >= start_date)
    if end_date:
        query = query.filter(TransactionHistory.timestamp <= end_date)

    result = await session.execute(query)
    transactions = result.scalars().all()

    if not transactions:
        return {}

    # Group transactions by date and calculate cumulative effect
    daily_transactions = defaultdict(list)
    for transaction in transactions:
        transaction_date = transaction.timestamp.date()
        daily_transactions[transaction_date].append(transaction)

    # Get all unique transaction dates and sort them
    transaction_dates = sorted(daily_transactions.keys(), reverse=True)

    # Calculate portfolio values at transaction dates (working backwards)
    transaction_portfolio_values = {}
    current_value = current_portfolio_value

    # Process each transaction date from most recent to oldest
    for transaction_date in transaction_dates:
        day_transactions = daily_transactions[transaction_date]

        # Process transactions for this day (most recent first)
        for transaction in day_transactions:
            # Reverse the transaction effect to get previous day's value
            if transaction.action in {"Market buy", "Limit buy"}:
                # If we bought something, subtract the cost to get previous value
                current_value -= transaction.total
            elif transaction.action in {"Market sell", "Limit sell"}:
                # If we sold something, add the proceeds to get previous value
                current_value += transaction.total
            elif transaction.action == "Deposit":
                # If we deposited, subtract the deposit to get previous value
                current_value -= transaction.total
            elif transaction.action == "Withdrawal":
                # If we withdrew, add the withdrawal to get previous value
                current_value += transaction.total

        # Store the portfolio value for this transaction date
        transaction_portfolio_values[transaction_date] = current_value

    # Add the current portfolio value for today (if not already there)
    today = date.today()
    if today not in transaction_portfolio_values:
        transaction_portfolio_values[today] = current_portfolio_value

    # Add the start value (before first transaction)
    if transaction_dates:
        first_transaction_date = min(transaction_dates)
        # Estimate portfolio value before first transaction (should be close to 0)
        transaction_portfolio_values[first_transaction_date - timedelta(days=1)] = max(0, current_value - 1000)

    # Create complete daily timeline with interpolation
    all_dates = sorted(transaction_portfolio_values.keys())

    if len(all_dates) < 2:
        return transaction_portfolio_values

    # Fill in missing dates with linear interpolation
    complete_portfolio_values = {}
    start_date_final = all_dates[0]
    end_date_final = all_dates[-1]

    current_date = start_date_final
    while current_date <= end_date_final:
        if current_date in transaction_portfolio_values:
            # Use actual transaction value
            complete_portfolio_values[current_date] = transaction_portfolio_values[current_date]
        else:
            # Interpolate between surrounding transaction dates
            prev_date = None
            next_date = None

            # Find the previous and next transaction dates
            for i, tx_date in enumerate(all_dates):
                if tx_date < current_date:
                    prev_date = tx_date
                elif tx_date > current_date and next_date is None:
                    next_date = tx_date
                    break

            if prev_date and next_date:
                # Linear interpolation
                prev_value = transaction_portfolio_values[prev_date]
                next_value = transaction_portfolio_values[next_date]

                days_between = (next_date - prev_date).days
                days_to_current = (current_date - prev_date).days

                if days_between > 0:
                    interpolation_factor = days_to_current / days_between
                    interpolated_value = prev_value + (next_value - prev_value) * interpolation_factor
                    complete_portfolio_values[current_date] = interpolated_value
                else:
                    complete_portfolio_values[current_date] = prev_value
            elif prev_date:
                # Use previous value if no next date
                complete_portfolio_values[current_date] = transaction_portfolio_values[prev_date]
            elif next_date:
                # Use next value if no previous date
                complete_portfolio_values[current_date] = transaction_portfolio_values[next_date]

        current_date += timedelta(days=1)

    return complete_portfolio_values


async def calculate_maximum_drawdown(
    session: AsyncSession,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_portfolio_value: float = 0.0,
) -> dict[str, float]:
    """
    Calculate maximum drawdown from transaction history with interpolation.

    Args:
        session: Async database session
        start_date: Start date for analysis (optional)
        end_date: End date for analysis (optional)
        current_portfolio_value: Current portfolio value

    Returns:
        Dictionary with drawdown metrics
    """
    # Get daily portfolio values from transaction history with interpolation
    portfolio_values = await calculate_daily_portfolio_values_with_interpolation(
        session, start_date, end_date, current_portfolio_value
    )

    if not portfolio_values:
        return {
            "max_drawdown_pct": 0.0,
            "max_drawdown_abs": 0.0,
            "drawdown_duration_days": 0,
            "peak_date": None,
            "trough_date": None,
            "current_peak_value": 0.0,
        }

    # Convert to sorted list of (date, value) tuples
    portfolio_data = sorted(portfolio_values.items())

    if not portfolio_data:
        return {
            "max_drawdown_pct": 0.0,
            "max_drawdown_abs": 0.0,
            "drawdown_duration_days": 0,
            "peak_date": None,
            "trough_date": None,
            "current_peak_value": 0.0,
        }

    # Calculate running peak and drawdown
    peak_value = 0.0
    max_drawdown_pct = 0.0
    max_drawdown_abs = 0.0
    peak_date = None
    trough_date = None
    current_drawdown_days = 0
    max_drawdown_days = 0

    for transaction_date, current_value in portfolio_data:
        # Update peak if we hit a new high
        if current_value > peak_value:
            peak_value = current_value
            peak_date = transaction_date
            current_drawdown_days = 0
        else:
            # Calculate drawdown from peak
            drawdown_abs = peak_value - current_value
            drawdown_pct = (drawdown_abs / peak_value) * 100 if peak_value > 0 else 0

            current_drawdown_days += 1

            # Update maximum drawdown if this is worse
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
                max_drawdown_abs = drawdown_abs
                trough_date = transaction_date
                max_drawdown_days = current_drawdown_days

    return {
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_abs": max_drawdown_abs,
        "drawdown_duration_days": max_drawdown_days,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "current_peak_value": peak_value,
    }


async def calculate_risk_adjusted_returns(
    session: AsyncSession,
    risk_free_rate: float = RISK_FREE_RATE * 100,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_portfolio_value: float = 0.0,
    known_annual_return: Optional[float] = None,
) -> dict[str, float]:
    """
    Calculate risk-adjusted returns with robust manual implementation.

    Provides industry-standard metrics including:
    - Sharpe Ratio
    - Sortino Ratio
    - Calmar Ratio
    - Omega Ratio
    - Tail Ratio
    - Volatility (annualized)

    Args:
        session: Async database session
        risk_free_rate: Annual risk-free rate (default 2%)
        start_date: Start date for analysis (optional)
        end_date: End date for analysis (optional)
        current_portfolio_value: Current portfolio value
        known_annual_return: Known annual return (from return metrics)

    Returns:
        Dictionary with risk-adjusted return metrics
    """
    # Get daily portfolio values from transaction history with interpolation
    portfolio_values = await calculate_daily_portfolio_values_with_interpolation(
        session, start_date, end_date, current_portfolio_value
    )

    if not portfolio_values:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "omega_ratio": 0.0,
            "tail_ratio": 0.0,
            "volatility": 0.0,
            "downside_deviation": 0.0,
            "avg_daily_return": 0.0,
            "avg_annual_return": 0.0,
            "risk_free_rate": risk_free_rate,
        }

    # Convert to sorted list of (date, value) tuples
    portfolio_data = sorted(portfolio_values.items())

    if len(portfolio_data) < 2:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "omega_ratio": 0.0,
            "tail_ratio": 0.0,
            "volatility": 0.0,
            "downside_deviation": 0.0,
            "avg_daily_return": 0.0,
            "avg_annual_return": 0.0,
            "risk_free_rate": risk_free_rate,
        }

    # Calculate daily returns
    returns = []
    for i in range(1, len(portfolio_data)):
        prev_date, prev_value = portfolio_data[i - 1]
        curr_date, curr_value = portfolio_data[i]

        if prev_value > 0:
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)

    if not returns:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "omega_ratio": 0.0,
            "tail_ratio": 0.0,
            "volatility": 0.0,
            "downside_deviation": 0.0,
            "avg_daily_return": 0.0,
            "avg_annual_return": 0.0,
            "risk_free_rate": risk_free_rate,
        }

    returns_array = np.array(returns)

    # Calculate basic metrics
    avg_daily_return = np.mean(returns_array)
    volatility = np.std(returns_array, ddof=1)  # Sample standard deviation

    # Annualize returns and volatility
    avg_annual_return_geometric = avg_daily_return * DAYS_PER_YEAR
    annual_volatility = volatility * np.sqrt(DAYS_PER_YEAR)
    annual_risk_free_rate = risk_free_rate / 100.0

    # Use the known annual return if provided
    if known_annual_return is not None:
        actual_annual_return = known_annual_return / 100.0
    else:
        actual_annual_return = avg_annual_return_geometric

    # Sharpe Ratio
    if annual_volatility > 0:
        sharpe_ratio = (actual_annual_return - annual_risk_free_rate) / (annual_volatility / 100.0)
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (downside deviation)
    negative_returns = returns_array[returns_array < 0]
    if len(negative_returns) > 0:
        downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(DAYS_PER_YEAR)
        if downside_deviation > 0:
            sortino_ratio = (actual_annual_return - annual_risk_free_rate) / (downside_deviation / 100.0)
        else:
            sortino_ratio = 0.0
    else:
        downside_deviation = 0.0
        sortino_ratio = 0.0

    # Calmar Ratio: Annual Return / Max Drawdown
    drawdown_metrics = await calculate_maximum_drawdown(session, start_date, end_date, current_portfolio_value)
    max_dd = drawdown_metrics["max_drawdown_pct"]
    if max_dd > 0:
        calmar_ratio = (actual_annual_return * 100) / max_dd
    else:
        calmar_ratio = 0.0

    # Omega Ratio: probability weighted gains / losses
    daily_risk_free = annual_risk_free_rate / DAYS_PER_YEAR
    excess_returns = returns_array - daily_risk_free
    gains = excess_returns[excess_returns > 0].sum()
    losses = abs(excess_returns[excess_returns < 0].sum())
    if losses > 0:
        omega_ratio = gains / losses
    else:
        omega_ratio = float("inf") if gains > 0 else 0.0

    # Tail Ratio: 95th percentile / 5th percentile (absolute value)
    # Measures extreme upside vs extreme downside
    if len(returns_array) >= 20:  # Need enough data for percentiles
        percentile_95 = np.percentile(returns_array, 95)
        percentile_5 = np.percentile(returns_array, 5)

        # Use absolute values to compare magnitudes
        abs_p95 = abs(percentile_95)
        abs_p5 = abs(percentile_5)

        if abs_p5 > 1e-10:  # Avoid division by zero
            tail_ratio = abs_p95 / abs_p5
        else:
            # If 5th percentile is near zero, use max gain vs max loss instead
            max_loss = abs(returns_array.min())
            if max_loss > 1e-10:
                tail_ratio = abs(returns_array.max()) / max_loss
            else:
                tail_ratio = 0.0
    else:
        tail_ratio = 0.0

    # Sanitize output - handle inf and nan
    def sanitize(value):
        if value is None or np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)

    return {
        "sharpe_ratio": sanitize(sharpe_ratio),
        "sortino_ratio": sanitize(sortino_ratio),
        "calmar_ratio": sanitize(calmar_ratio),
        "omega_ratio": sanitize(omega_ratio),
        "tail_ratio": sanitize(tail_ratio),
        "volatility": sanitize(annual_volatility),
        "downside_deviation": sanitize(downside_deviation),
        "avg_daily_return": sanitize(avg_daily_return),
        "avg_annual_return": sanitize(actual_annual_return),
        "avg_annual_return_geometric": sanitize(avg_annual_return_geometric),
        "risk_free_rate": risk_free_rate,
        "data_points": len(returns),
    }


async def calculate_fee_impact(
    session: AsyncSession,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict[str, float]:
    """
    Calculate fee impact analysis.

    Args:
        session: Async database session
        start_date: Start date for analysis (optional)
        end_date: End date for analysis (optional)

    Returns:
        Dictionary with fee impact metrics
    """
    # Build query for transaction history
    query = select(TransactionHistory)

    if start_date:
        query = query.filter(TransactionHistory.timestamp >= start_date)
    if end_date:
        query = query.filter(TransactionHistory.timestamp <= end_date)

    result = await session.execute(query)
    transactions = result.scalars().all()

    if not transactions:
        return {
            "total_fees": 0.0,
            "avg_fee_per_transaction": 0.0,
            "fee_drag_pct": 0.0,
            "total_transactions": 0,
            "fee_breakdown": {},
        }

    # Calculate fee metrics
    total_fees = 0.0
    total_transactions = 0
    fee_breakdown = {}

    for transaction in transactions:
        total_transactions += 1

        # Get fees from transaction
        if transaction.fees:
            transaction_fees = 0.0
            for fee in transaction.fees:
                if isinstance(fee, dict) and "quantity" in fee:
                    fee_amount = abs(fee["quantity"])  # Fees are negative
                    transaction_fees += fee_amount
                    fee_name = fee.get("name", "Unknown")
                    fee_breakdown[fee_name] = fee_breakdown.get(fee_name, 0.0) + fee_amount

            total_fees += transaction_fees

    # Calculate total invested for fee drag calculation (same logic as main function)
    total_deposited = 0.0
    total_withdrawn = 0.0

    for transaction in transactions:
        if transaction.action == "Deposit":
            total_deposited += transaction.total
        elif transaction.action == "Withdrawal":
            total_withdrawn += abs(transaction.total)

    total_invested = total_deposited - total_withdrawn

    # Calculate metrics
    avg_fee_per_transaction = total_fees / total_transactions if total_transactions > 0 else 0.0
    fee_drag_pct = (total_fees / total_invested) * 100 if total_invested > 0 else 0.0

    return {
        "total_fees": total_fees,
        "avg_fee_per_transaction": avg_fee_per_transaction,
        "fee_drag_pct": fee_drag_pct,
        "total_transactions": total_transactions,
        "total_invested": total_invested,
        "fee_breakdown": fee_breakdown,
    }


async def get_tier1_metrics(
    session: AsyncSession,
    current_portfolio_value: float,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    risk_free_rate: float = RISK_FREE_RATE * 100,
) -> dict[str, any]:
    """
    Calculate all Tier 1 portfolio metrics.

    Args:
        session: Async database session
        current_portfolio_value: Current portfolio value
        start_date: Start date for analysis (optional)
        end_date: End date for analysis (optional)
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Dictionary with all Tier 1 metrics
    """
    # Get transaction summary
    query = select(TransactionHistory)

    if start_date:
        query = query.filter(TransactionHistory.timestamp >= start_date)
    if end_date:
        query = query.filter(TransactionHistory.timestamp <= end_date)

    result = await session.execute(query)
    transactions = result.scalars().all()

    # Calculate actual invested amount (only deposits/withdrawals, not reinvestments)

    total_deposited = 0.0
    total_withdrawn = 0.0
    total_proceeds = 0.0

    for transaction in transactions:
        # Only count actual cash deposits/withdrawals as "invested"
        if transaction.action == "Deposit":
            total_deposited += transaction.total  # Use total (already positive)
        elif transaction.action == "Withdrawal":
            total_withdrawn += abs(transaction.total)  # Withdrawals reduce investment
        # Track sales proceeds separately (cash returned from selling positions)
        elif transaction.action in {"Market sell", "Limit sell"}:
            total_proceeds += abs(transaction.total)  # Proceeds from sales

    # Net invested = deposits - withdrawals (the actual cash YOU put in)
    total_invested = total_deposited - total_withdrawn

    # Calculate return metrics
    return_metrics = calculate_total_return(total_invested, current_portfolio_value, total_proceeds)

    # Calculate annualized return
    if transactions:
        investment_start_date = min(t.timestamp.date() for t in transactions)
        annualized_metrics = calculate_annualized_return(
            return_metrics["total_return_pct"], investment_start_date, end_date
        )
    else:
        annualized_metrics = {
            "annualized_return_pct": 0.0,
            "days_held": 0,
            "years_held": 0,
        }

    # Calculate risk metrics using transaction history
    drawdown_metrics = await calculate_maximum_drawdown(session, start_date, end_date, current_portfolio_value)
    risk_metrics = await calculate_risk_adjusted_returns(
        session,
        risk_free_rate,
        start_date,
        end_date,
        current_portfolio_value,
        known_annual_return=annualized_metrics.get("annualized_return_pct"),
    )
    fee_metrics = await calculate_fee_impact(session, start_date, end_date)

    # Calculate money-weighted return (IRR) considering actual investment timing
    cash_flows = await get_transaction_cash_flows(session, start_date, end_date)
    money_weighted_metrics = calculate_money_weighted_return(cash_flows, current_portfolio_value, start_date, end_date)

    # Combine all metrics
    return {
        "return_metrics": return_metrics,
        "annualized_metrics": annualized_metrics,
        "drawdown_metrics": drawdown_metrics,
        "risk_metrics": risk_metrics,
        "fee_metrics": fee_metrics,
        "money_weighted_metrics": money_weighted_metrics,
        "analysis_period": {
            "start_date": start_date,
            "end_date": end_date,
            "risk_free_rate": risk_free_rate,
        },
    }


def format_metrics_for_display(metrics: dict[str, any]) -> dict[str, str]:
    """
    Format metrics for human-readable display.

    Args:
        metrics: Raw metrics dictionary

    Returns:
        Dictionary with formatted strings for display
    """
    formatted = {}

    # Return metrics
    return_metrics = metrics["return_metrics"]
    formatted["total_return"] = (
        f"£{return_metrics['total_return_abs']:,.2f} ({return_metrics['total_return_pct']:+.2f}%)"
    )
    formatted["net_invested"] = f"£{return_metrics['net_invested']:,.2f}"
    formatted["current_value"] = f"£{return_metrics['current_value']:,.2f}"

    # Annualized metrics
    annualized = metrics["annualized_metrics"]
    formatted["annualized_return"] = f"{annualized['annualized_return_pct']:+.2f}%"
    formatted["investment_period"] = f"{annualized['days_held']} days ({annualized['years_held']:.1f} years)"

    # Drawdown metrics
    drawdown = metrics["drawdown_metrics"]
    formatted["max_drawdown"] = f"{drawdown['max_drawdown_pct']:.2f}% (£{drawdown['max_drawdown_abs']:,.2f})"

    # Risk metrics (including new empyrical metrics)
    risk = metrics["risk_metrics"]
    formatted["sharpe_ratio"] = f"{risk['sharpe_ratio']:.2f}"
    formatted["sortino_ratio"] = f"{risk['sortino_ratio']:.2f}"
    formatted["calmar_ratio"] = f"{risk.get('calmar_ratio', 0.0):.2f}"
    formatted["omega_ratio"] = f"{risk.get('omega_ratio', 0.0):.2f}"
    formatted["tail_ratio"] = f"{risk.get('tail_ratio', 0.0):.2f}"
    formatted["volatility"] = f"{risk['volatility']:.2f}%"
    formatted["downside_deviation"] = f"{risk.get('downside_deviation', 0.0):.2f}%"

    # Fee metrics
    fees = metrics["fee_metrics"]
    formatted["total_fees"] = f"£{fees['total_fees']:,.2f}"
    formatted["fee_drag"] = f"{fees['fee_drag_pct']:.2f}%"
    formatted["avg_fee_per_trade"] = f"£{fees['avg_fee_per_transaction']:.2f}"

    # Money-weighted return metrics
    if "money_weighted_metrics" in metrics:
        mwr = metrics["money_weighted_metrics"]
        formatted["money_weighted_return"] = f"{mwr['money_weighted_return_annual']:.2f}%"
        formatted["total_cash_flows"] = f"£{mwr['total_cash_flows']:,.2f}"
        formatted["investment_period_days"] = f"{mwr['period_days']} days"

    return formatted


def calculate_money_weighted_return(
    transactions: list[dict],
    current_portfolio_value: float,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict[str, float]:
    """
    Calculate Money-Weighted Return (IRR) considering actual investment timing.

    This gives the true rate of return on your actual invested capital,
    considering when you made each investment and withdrawal.

    Args:
        transactions: List of transaction dictionaries with date, amount, type
        current_portfolio_value: Current portfolio value
        start_date: Start date for analysis (optional)
        end_date: End date for analysis (optional)

    Returns:
        Dictionary with money-weighted return metrics
    """
    try:
        from scipy.optimize import fsolve
    except ImportError:
        # Fallback without scipy
        return {
            "money_weighted_return_pct": 0.0,
            "money_weighted_return_annual": 0.0,
            "total_cash_flows": 0.0,
            "period_days": 0,
            "error": "scipy not available",
        }

    # Filter transactions by date range
    if start_date or end_date:
        filtered_transactions = []
        for tx in transactions:
            tx_date = tx["date"]
            if start_date and tx_date < start_date:
                continue
            if end_date and tx_date > end_date:
                continue
            filtered_transactions.append(tx)
        transactions = filtered_transactions

    if not transactions:
        return {
            "money_weighted_return_pct": 0.0,
            "money_weighted_return_annual": 0.0,
            "total_cash_flows": 0.0,
            "period_days": 0,
        }

    # Prepare cash flows for IRR calculation
    # Positive = money in (deposits), Negative = money out (withdrawals)
    cash_flows = []
    dates = []

    for tx in transactions:
        if tx["type"] == "deposit":
            cash_flows.append(tx["amount"])  # Positive (money in)
        elif tx["type"] == "withdrawal":
            cash_flows.append(-tx["amount"])  # Negative (money out)
        dates.append(tx["date"])

    # Add final cash flow (current portfolio value as negative - money "out")
    cash_flows.append(-current_portfolio_value)
    dates.append(date.today())

    # Calculate time periods in years from first transaction
    if not dates:
        return {
            "money_weighted_return_pct": 0.0,
            "money_weighted_return_annual": 0.0,
            "total_cash_flows": 0.0,
            "period_days": 0,
        }

    first_date = min(dates)
    last_date = max(dates)
    period_days = (last_date - first_date).days
    period_years = period_days / DAYS_PER_YEAR

    # Convert dates to years from first date
    time_periods = []
    for d in dates:
        days_from_start = (d - first_date).days
        years_from_start = days_from_start / DAYS_PER_YEAR
        time_periods.append(years_from_start)

    # IRR function: NPV = 0
    def npv(rate):
        return sum(cf / (1 + rate) ** t for cf, t in zip(cash_flows, time_periods))

    # Solve for IRR using numerical methods
    try:
        # Initial guess for IRR
        irr = fsolve(npv, 0.1)[0]  # Start with 10% guess

        # Convert to percentage and annualize
        money_weighted_return_pct = irr * 100

        # Annual return is the IRR itself (already annualized)
        money_weighted_return_annual = money_weighted_return_pct

        total_cash_flows = sum(abs(cf) for cf in cash_flows[:-1])  # Exclude final value

    except Exception:
        # Fallback if IRR calculation fails
        money_weighted_return_pct = 0.0
        money_weighted_return_annual = 0.0
        total_cash_flows = sum(abs(cf) for cf in cash_flows[:-1])

    return {
        "money_weighted_return_pct": money_weighted_return_pct,
        "money_weighted_return_annual": money_weighted_return_annual,
        "total_cash_flows": total_cash_flows,
        "period_days": period_days,
        "period_years": period_years,
        "first_transaction_date": first_date,
        "last_transaction_date": last_date,
    }


async def get_transaction_cash_flows(
    session: AsyncSession,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[dict[str, str | date | float]]:
    """
    Get cash flows from transaction history for money-weighted return calculation.

    Args:
        session: Async database session
        start_date: Start date for analysis (optional)
        end_date: End date for analysis (optional)

    Returns:
        List of transaction dictionaries with date, amount, type
    """
    # Get all transactions in date range
    query = select(TransactionHistory)

    if start_date:
        query = query.filter(TransactionHistory.timestamp >= start_date)
    if end_date:
        query = query.filter(TransactionHistory.timestamp <= end_date)

    result = await session.execute(query)
    transactions = result.scalars().all()

    cash_flows: list[dict[str, str | date | float]] = []
    for transaction in transactions:
        if transaction.action == "Deposit":
            cash_flows.append(
                {
                    "date": transaction.timestamp.date(),
                    "amount": transaction.total,
                    "type": "deposit",
                }
            )
        elif transaction.action == "Withdrawal":
            cash_flows.append(
                {
                    "date": transaction.timestamp.date(),
                    "amount": transaction.total,
                    "type": "withdrawal",
                }
            )

    return cash_flows
