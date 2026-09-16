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
