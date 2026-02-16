"""
Tests for Pydantic schema validation (engagement and chat schemas).
"""
import pytest
from pydantic import ValidationError

from app.models.engagement_schemas import (
    FertilityReadinessRequest,
    FertilityReadinessResponse,
    HormonalPredictorRequest,
    VisualHealthRequest,
    TreatmentPathwayRequest,
    HomeIVFEligibilityRequest,
    MenstrualPattern,
)


class TestFertilityReadinessSchemas:
    def test_request_valid_minimal(self):
        r = FertilityReadinessRequest(age=30, menstrual_pattern="regular")
        assert r.age == 30
        assert r.menstrual_pattern == MenstrualPattern.regular
        assert r.use_ai_insight is True

    def test_request_age_bounds(self):
        """Female minimum age is 21 for fertility readiness."""
        with pytest.raises(ValidationError):
            FertilityReadinessRequest(age=20, menstrual_pattern="regular")
        with pytest.raises(ValidationError):
            FertilityReadinessRequest(age=56, menstrual_pattern="regular")

    def test_response_risk_levels(self):
        for level in ("low", "moderate", "high"):
            FertilityReadinessResponse(risk_score=50, risk_level=level, next_steps=[], guidance_text="Ok")


class TestHormonalPredictorSchemas:
    def test_request_sex_values(self):
        for sex in ("female", "male"):
            r = HormonalPredictorRequest(age=30, sex=sex)
            assert r.sex == sex

    def test_request_requires_age_and_sex(self):
        with pytest.raises(ValidationError):
            HormonalPredictorRequest(age=30)  # missing sex
        with pytest.raises(ValidationError):
            HormonalPredictorRequest(sex="female")  # missing age

    def test_request_min_age_21(self):
        """Patient age (female and male) must be at least 21."""
        with pytest.raises(ValidationError):
            HormonalPredictorRequest(age=20, sex="female")
        with pytest.raises(ValidationError):
            HormonalPredictorRequest(age=20, sex="male")
        HormonalPredictorRequest(age=21, sex="female")
        HormonalPredictorRequest(age=21, sex="male")


class TestVisualHealthSchemas:
    def test_request_optional_fields(self):
        r = VisualHealthRequest()
        assert r.self_reported_sleep_hours is None
        assert r.self_reported_stress_level is None
        r2 = VisualHealthRequest(self_reported_sleep_hours=7, self_reported_bmi=22)
        assert r2.self_reported_sleep_hours == 7
        assert r2.self_reported_bmi == 22


class TestTreatmentPathwaySchemas:
    def test_request_valid(self):
        r = TreatmentPathwayRequest(age=35, sex="female", years_trying=1, known_diagnosis=["PCOS"])
        assert r.known_diagnosis == ["PCOS"]

    def test_request_rejects_random_diagnosis(self):
        with pytest.raises(ValidationError):
            TreatmentPathwayRequest(
                age=35, sex="female",
                known_diagnosis=["hjbhscbdscjdcdjcdcjdscjdcjdcjdcnj"],
            )

    def test_request_rejects_random_treatment(self):
        with pytest.raises(ValidationError):
            TreatmentPathwayRequest(
                age=35, sex="female",
                previous_treatments=["hbchbckjsbcjkccjcjncjnjcnscns"],
            )

    def test_request_accepts_valid_custom_phrase(self):
        r = TreatmentPathwayRequest(
            age=35, sex="female",
            known_diagnosis=["mild male factor"],
        )
        assert r.known_diagnosis == ["mild male factor"]

    def test_request_min_age_21(self):
        """Patient age (female and male) must be at least 21."""
        with pytest.raises(ValidationError):
            TreatmentPathwayRequest(age=20, sex="female", known_diagnosis=[])
        with pytest.raises(ValidationError):
            TreatmentPathwayRequest(age=20, sex="male", known_diagnosis=[])
        TreatmentPathwayRequest(age=21, sex="female", known_diagnosis=[])
        TreatmentPathwayRequest(age=21, sex="male", known_diagnosis=[])


class TestHomeIVFEligibilitySchemas:
    def test_request_female_age_required(self):
        with pytest.raises(ValidationError):
            HomeIVFEligibilityRequest()
        r = HomeIVFEligibilityRequest(female_age=34)
        assert r.female_age == 34
        assert r.male_age is None

    def test_request_female_min_age_21(self):
        """Female age minimum is 21."""
        with pytest.raises(ValidationError):
            HomeIVFEligibilityRequest(female_age=20)
        r = HomeIVFEligibilityRequest(female_age=21)
        assert r.female_age == 21

    def test_request_male_min_age_21(self):
        """Male age minimum is 21 when provided."""
        with pytest.raises(ValidationError):
            HomeIVFEligibilityRequest(female_age=25, male_age=20)
        r = HomeIVFEligibilityRequest(female_age=25, male_age=21)
        assert r.male_age == 21
