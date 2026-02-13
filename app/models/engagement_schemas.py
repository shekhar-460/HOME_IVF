"""
Pydantic schemas for AI-driven fertility engagement tools.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# --- Fertility Readiness Calculator ---

class MenstrualPattern(str, Enum):
    regular = "regular"
    irregular = "irregular"
    absent = "absent"
    unknown = "unknown"


class FertilityReadinessRequest(BaseModel):
    """Input for fertility readiness score."""
    age: int = Field(..., ge=18, le=55, description="Age in years")
    medical_history: Optional[List[str]] = Field(
        default=[],
        description="e.g. PCOS, endometriosis, thyroid, diabetes, prior surgery"
    )
    lifestyle_smoking: bool = Field(default=False, description="Current smoking")
    lifestyle_alcohol: str = Field(default="none", description="none / occasional / regular")
    lifestyle_exercise: str = Field(default="moderate", description="sedentary / moderate / active")
    bmi: Optional[float] = Field(default=None, ge=15, le=50, description="Body mass index")
    menstrual_pattern: MenstrualPattern = Field(default=MenstrualPattern.regular)
    cycle_length_days: Optional[int] = Field(default=None, ge=21, le=45)
    previous_pregnancies: int = Field(default=0, ge=0, le=20)
    live_births: int = Field(default=0, ge=0, le=20)
    miscarriages: int = Field(default=0, ge=0, le=20)
    years_trying: Optional[float] = Field(default=None, ge=0, le=20)
    language: Optional[str] = Field(default="en", description="en or hi")
    use_ai_insight: bool = Field(default=True, description="Include MedGemma explanation when enabled")


class FertilityReadinessResponse(BaseModel):
    """Fertility readiness result."""
    risk_score: float = Field(..., ge=0, le=100, description="0=lower risk, 100=higher risk")
    risk_level: str = Field(..., description="low / moderate / high")
    next_steps: List[str] = Field(default_factory=list)
    guidance_text: str = Field(default="")
    ai_insight: Optional[str] = None
    medical_history_recognized: Optional[List[str]] = Field(
        default=None,
        description="Canonical conditions used in scoring (e.g. PCOS, thyroid)"
    )
    medical_history_unrecognized: Optional[List[str]] = Field(
        default=None,
        description="Raw entries that could not be mapped; not used in scoring"
    )


# --- Hormonal & Ovarian Health Predictor ---

class HormonalPredictorRequest(BaseModel):
    """Questionnaire + data for when to test AMH, semen analysis, or specialist."""
    age: int = Field(..., ge=18, le=55)
    sex: str = Field(..., description="female / male")
    irregular_cycles: bool = Field(default=False)
    cycle_length_days: Optional[int] = Field(default=None, ge=21, le=45)
    symptoms_acne: bool = Field(default=False)
    symptoms_hirsutism: bool = Field(default=False)
    symptoms_heavy_bleeding: bool = Field(default=False)
    symptoms_pain: bool = Field(default=False)
    years_trying: Optional[float] = Field(default=None, ge=0, le=20)
    previous_tests_amh: bool = Field(default=False)
    previous_tests_semen: bool = Field(default=False)
    language: Optional[str] = Field(default="en")
    use_ai_insight: bool = Field(default=True)


class HormonalPredictorResponse(BaseModel):
    """Suggestions for AMH, semen analysis, or specialist."""
    suggest_amh: bool = Field(default=False)
    suggest_semen_analysis: bool = Field(default=False)
    suggest_specialist: bool = Field(default=False)
    when_to_test: str = Field(default="")
    reasoning: List[str] = Field(default_factory=list)
    ai_insight: Optional[str] = None


# --- Face / Visual Health Indicator (Exploratory, non-diagnostic) ---

class VisualHealthRequest(BaseModel):
    """Optional image + self-reported wellness for awareness-focused recommendations."""
    image_base64: Optional[str] = Field(default=None, description="Optional face/wellness image (exploratory)")
    self_reported_sleep_hours: Optional[float] = Field(default=None, ge=0, le=24)
    self_reported_stress_level: Optional[str] = Field(default=None, description="low / moderate / high")
    self_reported_bmi: Optional[float] = Field(default=None, ge=15, le=50)
    skin_concerns: Optional[List[str]] = Field(default=[], description="e.g. dryness, acne, dullness")
    language: Optional[str] = Field(default="en")
    use_ai_insight: bool = Field(default=True)


class VisualHealthResponse(BaseModel):
    """Non-diagnostic wellness and reproductive health awareness."""
    summary: Optional[str] = Field(
        default=None,
        description="Short, patient-friendly takeaway (1–2 sentences) for quick understanding."
    )
    disclaimer: str = Field(
        default="This is for general wellness awareness only and is not a medical diagnosis."
    )
    wellness_summary: Optional[str] = Field(
        default=None,
        description="Plain-language summary of what the patient shared (sleep, stress, BMI, image)."
    )
    wellness_indicators: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    image_analysis: Optional[str] = Field(
        default=None,
        description="When an image was uploaded and analyzed, short non-diagnostic wellness observation."
    )
    ai_insight: Optional[str] = None


# --- Personalized Treatment Pathway Recommender ---

class TreatmentPathwayRequest(BaseModel):
    """Input for recommending natural conception, IUI, IVF, or fertility preservation."""
    age: int = Field(..., ge=18, le=55)
    sex: str = Field(..., description="female / male")
    years_trying: Optional[float] = Field(default=None, ge=0, le=20)
    known_diagnosis: Optional[List[str]] = Field(
        default=[],
        description="e.g. tubal factor, male factor, PCOS, unexplained"
    )
    previous_treatments: Optional[List[str]] = Field(default=[], description="e.g. IUI, ovulation induction")
    preserving_fertility: bool = Field(default=False, description="Interest in egg/sperm freezing")
    language: Optional[str] = Field(default="en")
    use_ai_insight: bool = Field(default=True)


class TreatmentPathwayResponse(BaseModel):
    """Suggested pathway(s) and primary recommendation."""
    suggested_pathways: List[str] = Field(
        default_factory=list,
        description="e.g. natural_conception_support, iui, ivf, fertility_preservation"
    )
    primary_recommendation: str = Field(default="")
    reasoning: List[str] = Field(default_factory=list)
    ai_insight: Optional[str] = None


# --- Home IVF Eligibility Checker ---

class HomeIVFEligibilityRequest(BaseModel):
    """Couple/patient info for Home IVF suitability."""
    female_age: int = Field(..., ge=18, le=50)
    male_age: Optional[int] = Field(default=None, ge=18, le=60)
    medical_contraindications: Optional[List[str]] = Field(
        default=[],
        description="e.g. severe OHSS history, uncontrolled diabetes"
    )
    has_consulted_specialist: bool = Field(default=False)
    ovarian_reserve_known: bool = Field(default=False)
    semen_analysis_known: bool = Field(default=False)
    stable_relationship_or_single_with_donor: bool = Field(default=True)
    language: Optional[str] = Field(default="en")
    use_ai_insight: bool = Field(default=True)


class HomeIVFEligibilityResponse(BaseModel):
    """Eligibility result and consultation prompt."""
    eligible: bool = Field(default=False)
    reasons: List[str] = Field(default_factory=list)
    missing_criteria: List[str] = Field(default_factory=list)
    prompt_consultation: bool = Field(default=True)
    booking_message: str = Field(default="")
    ai_insight: Optional[str] = None
