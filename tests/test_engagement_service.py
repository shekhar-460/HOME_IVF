"""
Unit tests for EngagementService (no FastAPI, no DB, no MedGemma).
"""
import pytest

from app.services.engagement_service import EngagementService
from app.models.engagement_schemas import (
    FertilityReadinessRequest,
    HormonalPredictorRequest,
    VisualHealthRequest,
    TreatmentPathwayRequest,
    HomeIVFEligibilityRequest,
    MenstrualPattern,
)


@pytest.fixture
def service():
    return EngagementService(knowledge_engine=None)


class TestFertilityReadiness:
    def test_low_risk_young_healthy(self, service):
        req = FertilityReadinessRequest(
            age=25,
            medical_history=[],
            menstrual_pattern=MenstrualPattern.regular,
            previous_pregnancies=0,
            use_ai_insight=False,
        )
        res = service.fertility_readiness(req)
        assert res.risk_level == "low"
        assert res.risk_score < 30
        assert len(res.next_steps) >= 1

    def test_high_risk_age_plus_factors(self, service):
        req = FertilityReadinessRequest(
            age=42,
            medical_history=["PCOS", "endometriosis"],
            lifestyle_smoking=True,
            menstrual_pattern=MenstrualPattern.irregular,
            previous_pregnancies=0,
            years_trying=3,
            use_ai_insight=False,
        )
        res = service.fertility_readiness(req)
        assert res.risk_level == "high"
        assert res.risk_score >= 60
        assert any("specialist" in s.lower() for s in res.next_steps)


class TestHormonalPredictor:
    def test_suggests_amh_for_irregular_cycles(self, service):
        req = HormonalPredictorRequest(
            age=33,
            sex="female",
            irregular_cycles=True,
            years_trying=1,
            use_ai_insight=False,
        )
        res = service.hormonal_predictor(req)
        assert res.suggest_amh is True
        assert len(res.reasoning) >= 1

    def test_suggests_specialist_after_two_years(self, service):
        req = HormonalPredictorRequest(
            age=35,
            sex="couple",
            years_trying=2.5,
            use_ai_insight=False,
        )
        res = service.hormonal_predictor(req)
        assert res.suggest_specialist is True


class TestVisualHealth:
    def test_recommendations_from_sleep_stress(self, service):
        req = VisualHealthRequest(
            self_reported_sleep_hours=5,
            self_reported_stress_level="high",
            use_ai_insight=False,
        )
        res = service.visual_health(req)
        assert "disclaimer" in res.disclaimer.lower() or "awareness" in res.disclaimer.lower()
        assert len(res.recommendations) >= 1
        assert res.wellness_indicators.get("sleep_hours") == 5
        assert res.wellness_indicators.get("stress_level") == "high"


class TestTreatmentPathway:
    def test_natural_conception_when_recently_trying(self, service):
        req = TreatmentPathwayRequest(
            age=28,
            sex="couple",
            years_trying=0.5,
            use_ai_insight=False,
        )
        res = service.treatment_pathway(req)
        assert "natural_conception_support" in res.suggested_pathways
        assert res.primary_recommendation == "natural_conception_support"

    def test_ivf_suggested_with_tubal_diagnosis(self, service):
        req = TreatmentPathwayRequest(
            age=34,
            sex="couple",
            years_trying=2,
            known_diagnosis=["tubal factor"],
            use_ai_insight=False,
        )
        res = service.treatment_pathway(req)
        assert "ivf" in res.suggested_pathways
        assert res.primary_recommendation == "ivf"


class TestHomeIVFEligibility:
    def test_eligible_basic_couple(self, service):
        req = HomeIVFEligibilityRequest(
            female_age=32,
            male_age=34,
            medical_contraindications=[],
            use_ai_insight=False,
        )
        res = service.home_ivf_eligibility(req)
        assert res.eligible is True
        assert res.prompt_consultation is True
        assert "book" in res.booking_message.lower() or "consultation" in res.booking_message.lower()

    def test_not_eligible_female_age_over_45(self, service):
        req = HomeIVFEligibilityRequest(
            female_age=46,
            medical_contraindications=[],
            use_ai_insight=False,
        )
        res = service.home_ivf_eligibility(req)
        assert res.eligible is False
        assert any("age" in r.lower() for r in res.reasons)
