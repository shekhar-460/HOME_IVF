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
        assert "PCOS" in (res.medical_history_recognized or [])
        assert "Endometriosis" in (res.medical_history_recognized or [])

    def test_medical_history_variations_normalized(self, service):
        """Variations like pcod, PCO, thyroid are recognized and used; score is consistent."""
        req = FertilityReadinessRequest(
            age=30,
            medical_history=["pcod", "PCO", "thyroid"],
            menstrual_pattern=MenstrualPattern.regular,
            previous_pregnancies=0,
            use_ai_insight=False,
        )
        res = service.fertility_readiness(req)
        assert "PCOS" in (res.medical_history_recognized or [])
        assert "Thyroid disorder" in (res.medical_history_recognized or [])
        assert res.medical_history_unrecognized is None or len(res.medical_history_unrecognized) == 0

    def test_medical_history_unrecognized_not_used_in_score(self, service):
        """Invalid or typo entries are not used in scoring and are reported to the user."""
        req = FertilityReadinessRequest(
            age=30,
            medical_history=["pcos", "xyz typo", "unknown_condition"],
            menstrual_pattern=MenstrualPattern.regular,
            previous_pregnancies=0,
            use_ai_insight=False,
        )
        res = service.fertility_readiness(req)
        assert res.medical_history_recognized == ["PCOS"]
        assert "xyz typo" in (res.medical_history_unrecognized or [])
        assert "unknown_condition" in (res.medical_history_unrecognized or [])


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
            sex="female",
            years_trying=2.5,
            use_ai_insight=False,
        )
        res = service.hormonal_predictor(req)
        assert res.suggest_specialist is True

    def test_does_not_suggest_amh_when_already_had_amh_test(self, service):
        """If patient already had AMH test, do not suggest AMH again."""
        req = HormonalPredictorRequest(
            age=36,
            sex="female",
            irregular_cycles=True,
            years_trying=10,
            previous_tests_amh=True,
            use_ai_insight=False,
        )
        res = service.hormonal_predictor(req)
        assert res.suggest_amh is False
        assert res.suggest_specialist is True  # still suggest specialist for 10 years trying


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
            sex="female",
            years_trying=0.5,
            use_ai_insight=False,
        )
        res = service.treatment_pathway(req)
        assert "natural_conception_support" in res.suggested_pathways
        assert res.primary_recommendation == "natural_conception_support"

    def test_ivf_suggested_with_tubal_diagnosis(self, service):
        req = TreatmentPathwayRequest(
            age=34,
            sex="female",
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
