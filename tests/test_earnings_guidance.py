def test_guidance_schema_keeps_currency_and_minor_units_without_inference():
    from scripts._earnings_common import Guidance

    guidance = Guidance.model_validate(
        {
            "eps_guidance": {"unit": "GBp", "next_year": 178.5},
            "revenue_guidance": {"currency": "EUR", "next_year": 44000},
        }
    ).model_dump()
    assert guidance["eps_guidance"]["unit"] == "GBp"
    assert guidance["revenue_guidance"]["currency"] == "EUR"
    legacy = Guidance.model_validate({"eps_guidance": {"next_year": 178.5}}).model_dump()
    assert legacy["eps_guidance"]["unit"] is None


def test_unit_only_guidance_is_not_a_numeric_outlook_but_zero_is():
    from scripts._earnings_common import Guidance
    units_only = Guidance.model_validate({'eps_guidance': {'unit': 'USD'},
                                         'revenue_guidance': {'currency': 'EUR'}})
    assert units_only.eps_guidance is None and units_only.revenue_guidance is None
    zero = Guidance.model_validate({'eps_guidance': {'unit': 'GBp', 'next_year': 0}})
    assert zero.eps_guidance.next_year == 0 and zero.eps_guidance.unit == 'GBp'
