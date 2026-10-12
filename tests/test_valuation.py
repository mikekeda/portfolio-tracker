"""Independent arithmetic for quote/statement currencies and ADR/pence units."""

from types import SimpleNamespace

import pytest

from backend.utils.dcf import _compute_dcf_full
from backend.utils.fcf import TrailingFcf
from backend.utils.valuation import valuation_metrics


def info(**changes):
    return {"currency": "USD", "financialCurrency": "USD", "marketCap": 1000,
            "totalDebt": 200, "totalCash": 50, "ebitda": 100, "totalRevenue": 500, **changes}


def test_asml_ev_converts_eur_balances_not_raw_yahoo_multiples():
    source = info(financialCurrency="EUR", marketCap=600e9, totalDebt=4e9, totalCash=10e9,
                  ebitda=15e9, totalRevenue=30e9, enterpriseValue=34e12,
                  enterpriseToEbitda=2519, enterpriseToRevenue=961, priceToBook=1342)
    result = valuation_metrics("ASML", source, {"USD": .75, "EUR": .9}, TrailingFcf(10e9, 30e9))
    assert result["enterpriseValue"] == pytest.approx(592.8e9)
    assert result["enterpriseToEbitda"] == pytest.approx(592.8 / 18)
    assert result["enterpriseToRevenue"] == pytest.approx(592.8 / 36)
    assert result["priceToSalesTtm"] == pytest.approx(600 / 36)
    assert result["fcfYield"] == pytest.approx(2)
    assert result["totalDebt"] == pytest.approx(4.8e9)
    assert result["priceToBook"] is None
    assert source["totalDebt"] == 4e9


@pytest.mark.parametrize("symbol", ["PBR", "VALE"])
def test_usd_statements_do_not_relabel_brl_info_fields(symbol):
    source = info(financialCurrency="BRL", totalDebt=1000, totalCash=250,
                  ebitda=500, totalRevenue=2500, sharesOutstanding=20)
    result = valuation_metrics(symbol, source, {"USD": .75, "BRL": .15}, TrailingFcf(100, 500))
    assert result["statementCurrency"] == "USD"
    assert result["fcfYield"] == 10
    assert result["totalDebt"] == pytest.approx(200)
    assert result["enterpriseValue"] == pytest.approx(1150)
    assert result["priceToSalesTtm"] == pytest.approx(2)
    # One ADS represents two ordinary shares; aggregate ratios must not double FCF.
    source["sharesOutstanding"] *= 2
    assert valuation_metrics(symbol, source, {"USD": .75, "BRL": .15}, TrailingFcf(100, 500)) == result


@pytest.mark.parametrize("currency", ["GBp", "GBX"])
def test_pence_quotes_do_not_multiply_market_cap_or_ev_by_100(currency):
    result = valuation_metrics("AZN.L", info(currency=currency), {"USD": .8}, TrailingFcf(100, 500))
    assert result["valuationCurrency"] == "GBP"
    assert result["enterpriseValue"] == 1120
    assert result["priceToSalesTtm"] == 2.5
    assert result["fcfYield"] == 8


def test_zero_balances_and_negative_cashflow_are_preserved():
    result = valuation_metrics("TEST", info(totalDebt=0, totalCash=0), {}, TrailingFcf(0, 500))
    assert result["enterpriseValue"] == 1000
    assert result["netDebtToEbitda"] == 0
    assert result["fcfYield"] == 0
    assert valuation_metrics("TEST", info(), {}, TrailingFcf(-100, 500))["fcfYield"] == -10


@pytest.mark.parametrize("missing", ["totalDebt", "totalCash"])
def test_missing_balance_is_not_zero(missing):
    result = valuation_metrics("TEST", info(**{missing: None}), {}, TrailingFcf(100, 500))
    assert result["enterpriseValue"] is None
    assert result["netDebtToEbitda"] is None
    assert result["priceToSalesTtm"] == 2
    assert result["valuationIssues"]["enterpriseValue"]
    assert result["valuationIssues"][missing]


@pytest.mark.parametrize("rates", [{}, {"USD": .75}, {"USD": .75, "EUR": 0},
                                   {"USD": .75, "EUR": float("nan")}, {"USD": .75, "EUR": -1}])
def test_missing_or_invalid_fx_withholds_cross_currency_metrics(rates):
    result = valuation_metrics("ASML", info(financialCurrency="EUR"), rates, TrailingFcf(100, 500))
    for key in ("totalDebt", "totalCash", "enterpriseValue", "enterpriseToRevenue", "priceToSalesTtm", "fcfYield"):
        assert result[key] is None
        assert "exchange rate" in result["valuationIssues"][key]
    assert result["netDebtToEbitda"] == 1.5
    assert result["netCash"] is False


@pytest.mark.parametrize("missing", ["totalDebt", "totalCash", "ebitda"])
def test_leverage_missing_amount_reason_does_not_blame_fx(missing):
    result = valuation_metrics("ASML", info(financialCurrency="EUR", **{missing: None}), {}, None)
    assert result["netDebtToEbitda"] is None
    assert result["valuationIssues"]["netDebtToEbitda"] == "Unavailable: debt, cash or positive EBITDA is missing."


def test_leverage_missing_currency_reason_is_distinct_from_missing_fx():
    result = valuation_metrics("TEST", info(financialCurrency=None), {}, None)
    assert result["valuationIssues"]["netDebtToEbitda"] == "Unavailable: summary currency is missing."


def test_valid_metrics_do_not_repeat_static_pb_message():
    result = valuation_metrics("TEST", info(), {}, TrailingFcf(100, 500))
    assert result["priceToBook"] is None
    assert result["valuationIssues"] == {}


@pytest.mark.parametrize("field", ["currency", "financialCurrency"])
def test_missing_currency_is_not_assumed_usd(field):
    result = valuation_metrics("TEST", info(**{field: None}), {}, TrailingFcf(100, 500))
    assert result["fcfYield"] is None
    assert result["enterpriseValue"] is None


@pytest.mark.parametrize("denominator", [0, -1, float("nan"), float("inf"), True])
def test_invalid_earnings_denominators_do_not_create_multiples(denominator):
    result = valuation_metrics("TEST", info(ebitda=denominator, totalRevenue=denominator), {}, None)
    assert result["enterpriseToEbitda"] is None
    assert result["enterpriseToRevenue"] is None
    assert result["priceToSalesTtm"] is None


def instrument(symbol, **changes):
    return SimpleNamespace(yahoo_symbol=symbol, yahoo=SimpleNamespace(
        info=info(currentPrice=10, sharesOutstanding=100, **changes),
        cashflow={"2025-12-31": {"Free Cash Flow": 100}},
    ))


@pytest.mark.parametrize("symbol", ["PBR", "VALE"])
def test_dcf_withholds_known_mixed_currency_modules(symbol):
    result = _compute_dcf_full(instrument(symbol, financialCurrency="BRL"), {}, {"USD": .75, "BRL": .15})
    assert result["price"] is None and result["implied_growth"] is None
    assert "currencies differ" in result["unavailable_reason"]


def test_dcf_same_currency_still_computes_and_unknown_currency_is_explained():
    assert _compute_dcf_full(instrument("TEST"), {}, {})["price"] > 0
    result = _compute_dcf_full(instrument("TEST", financialCurrency=None), {}, {})
    assert result["price"] is None
    assert "currency" in result["unavailable_reason"]
