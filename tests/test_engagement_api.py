"""
API tests for engagement endpoints (no DB/MedGemma; uses overridden dependency).
"""
import pytest


BASE = "/api/v1/engagement"


@pytest.mark.asyncio
async def test_fertility_readiness_ok(client):
    """Fertility readiness returns 200 and expected shape."""
    payload = {
        "age": 32,
        "medical_history": [],
        "lifestyle_smoking": False,
        "menstrual_pattern": "regular",
        "previous_pregnancies": 0,
        "use_ai_insight": False,
    }
    response = await client.post(f"{BASE}/fertility-readiness", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_level" in data
    assert data["risk_level"] in ("low", "moderate", "high")
    assert 0 <= data["risk_score"] <= 100
    assert "next_steps" in data
    assert isinstance(data["next_steps"], list)
    assert "guidance_text" in data


@pytest.mark.asyncio
async def test_fertility_readiness_validation(client):
    """Fertility readiness rejects invalid age (minimum female age 21)."""
    response = await client.post(
        f"{BASE}/fertility-readiness",
        json={"age": 15, "menstrual_pattern": "regular"},
    )
    assert response.status_code in (400, 422)
    response20 = await client.post(
        f"{BASE}/fertility-readiness",
        json={"age": 20, "menstrual_pattern": "regular"},
    )
    assert response20.status_code in (400, 422)


@pytest.mark.asyncio
async def test_hormonal_predictor_ok(client):
    """Hormonal predictor returns 200 and expected shape."""
    payload = {
        "age": 34,
        "sex": "female",
        "irregular_cycles": True,
        "years_trying": 2,
        "use_ai_insight": False,
    }
    response = await client.post(f"{BASE}/hormonal-predictor", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "suggest_amh" in data
    assert "suggest_semen_analysis" in data
    assert "suggest_specialist" in data
    assert "reasoning" in data
    assert "when_to_test" in data


@pytest.mark.asyncio
async def test_visual_health_ok(client):
    """Visual health returns 200 and disclaimer."""
    payload = {
        "self_reported_sleep_hours": 6,
        "self_reported_stress_level": "moderate",
        "use_ai_insight": False,
    }
    response = await client.post(f"{BASE}/visual-health", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "disclaimer" in data
    assert "wellness_indicators" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


@pytest.mark.asyncio
async def test_treatment_pathway_ok(client):
    """Treatment pathway returns 200 and suggested pathways."""
    payload = {
        "age": 36,
        "sex": "female",
        "years_trying": 2,
        "known_diagnosis": [],
        "use_ai_insight": False,
    }
    response = await client.post(f"{BASE}/treatment-pathway", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "suggested_pathways" in data
    assert "primary_recommendation" in data
    assert "reasoning" in data
    assert isinstance(data["suggested_pathways"], list)


@pytest.mark.asyncio
async def test_treatment_pathway_rejects_random_data(client):
    """Treatment pathway returns 422 when diagnosis/treatment contain random text."""
    payload = {
        "age": 36,
        "sex": "female",
        "known_diagnosis": ["hjbhscbdscjdcdjcdcjdscjdcjdcjdcnj"],
        "previous_treatments": ["hbchbckjsbcjkccjcjncjnjcnscns"],
        "use_ai_insight": False,
    }
    response = await client.post(f"{BASE}/treatment-pathway", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_home_ivf_eligibility_ok(client):
    """Home IVF eligibility returns 200 and eligible/reasons."""
    payload = {
        "female_age": 34,
        "male_age": 36,
        "medical_contraindications": [],
        "has_consulted_specialist": False,
        "use_ai_insight": False,
    }
    response = await client.post(f"{BASE}/home-ivf-eligibility", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "eligible" in data
    assert "reasons" in data
    assert "missing_criteria" in data
    assert "booking_message" in data
    assert "prompt_consultation" in data
