"""
AI-driven engagement tools: fertility readiness, hormonal predictor,
visual wellness (exploratory), treatment pathway, and Home IVF eligibility.
Uses rule-based logic + optional MedGemma for explanations.
"""
from typing import Optional, List, TYPE_CHECKING
import logging

from app.models.engagement_schemas import (
    FertilityReadinessRequest,
    FertilityReadinessResponse,
    HormonalPredictorRequest,
    HormonalPredictorResponse,
    VisualHealthRequest,
    VisualHealthResponse,
    TreatmentPathwayRequest,
    TreatmentPathwayResponse,
    HomeIVFEligibilityRequest,
    HomeIVFEligibilityResponse,
)

if TYPE_CHECKING:
    from app.services.knowledge_engine import KnowledgeEngine

logger = logging.getLogger(__name__)


class EngagementService:
    """Implements the five engagement tools with rule-based logic and optional MedGemma."""

    def __init__(self, knowledge_engine: Optional["KnowledgeEngine"] = None):
        self.knowledge_engine = knowledge_engine

    def _get_ai_insight(
        self,
        query: str,
        language: str = "en",
        context: Optional[str] = None,
        image: Optional[str] = None,
    ) -> Optional[str]:
        """Optional short MedGemma insight (text-only or multimodal with image)."""
        if not self.knowledge_engine:
            return None
        try:
            return self.knowledge_engine.get_answer_from_medgemma(
                query,
                language=language,
                context=context,
                use_cache=(image is None),
                image=image,
            )
        except Exception as e:
            logger.warning(f"MedGemma insight failed: {e}")
            return None

    # --- 1. Fertility Readiness Calculator ---

    def fertility_readiness(self, req: FertilityReadinessRequest) -> FertilityReadinessResponse:
        """Preliminary fertility risk score and next-step guidance (non-diagnostic)."""
        score = 0.0
        reasons: List[str] = []

        # Age factor (higher age -> higher risk)
        if req.age >= 40:
            score += 35
            reasons.append("Age 40+ is associated with lower ovarian reserve and higher risk.")
        elif req.age >= 35:
            score += 20
            reasons.append("Age 35+ suggests considering earlier screening.")
        elif req.age >= 30:
            score += 10

        # Medical history
        high_impact = {"pcos", "endometriosis", "thyroid", "diabetes", "prior surgery", "tubal"}
        for h in (req.medical_history or []):
            if any(k in h.lower() for k in high_impact):
                score += 15
                reasons.append(f"Medical history ({h}) may affect fertility; specialist review recommended.")

        # Lifestyle
        if req.lifestyle_smoking:
            score += 15
            reasons.append("Smoking can reduce fertility; quitting is recommended.")
        if req.lifestyle_alcohol == "regular":
            score += 10
            reasons.append("Regular alcohol may affect fertility.")
        if req.lifestyle_exercise == "sedentary":
            score += 5

        # BMI
        if req.bmi is not None:
            if req.bmi < 18.5 or req.bmi > 30:
                score += 15
                reasons.append("BMI outside 18.5–30 may affect cycle regularity and outcomes.")

        # Menstrual
        if req.menstrual_pattern.value == "irregular":
            score += 15
            reasons.append("Irregular cycles suggest checking ovulation and hormonal health.")
        elif req.menstrual_pattern.value == "absent":
            score += 25
            reasons.append("Absent periods require medical evaluation.")

        # Pregnancy history
        if req.miscarriages > 0:
            score += 10
            reasons.append("Recurrent loss evaluation may be helpful.")
        if req.years_trying is not None and req.years_trying >= 1:
            score += 10
            reasons.append("Trying for 1+ year suggests considering basic fertility workup.")

        risk_score = min(100.0, score)
        if risk_score < 30:
            risk_level = "low"
            next_steps = [
                "Maintain healthy lifestyle and track cycles.",
                "Consider preconception checkup if planning pregnancy.",
            ]
        elif risk_score < 60:
            risk_level = "moderate"
            next_steps = [
                "Discuss with your doctor or a fertility specialist.",
                "Consider basic screening (AMH, semen analysis) if trying to conceive.",
            ]
        else:
            risk_level = "high"
            next_steps = [
                "Book a consultation with a fertility specialist.",
                "Consider AMH and/or semen analysis as advised.",
            ]

        guidance_text = (
            f"Based on the inputs provided, your preliminary fertility risk is {risk_level}. "
            "This is not a diagnosis. A doctor can give you personalized advice."
        )

        ai_insight = None
        if req.use_ai_insight and self.knowledge_engine:
            q = (
                f"Brief 2–3 sentence fertility awareness advice for a {req.age}-year-old with "
                f"{risk_level} preliminary risk, menstrual pattern {req.menstrual_pattern.value}. "
                "Focus on next steps and reassurance. IVF/fertility context only."
            )
            ai_insight = self._get_ai_insight(q, req.language or "en")

        return FertilityReadinessResponse(
            risk_score=risk_score,
            risk_level=risk_level,
            next_steps=next_steps,
            guidance_text=guidance_text,
            ai_insight=ai_insight,
        )

    # --- 2. Hormonal & Ovarian Health Predictor ---

    def hormonal_predictor(self, req: HormonalPredictorRequest) -> HormonalPredictorResponse:
        """Suggest when to test AMH, semen analysis, or consult a specialist."""
        suggest_amh = False
        suggest_semen = False
        suggest_specialist = False
        reasoning: List[str] = []
        when_to_test = ""

        if req.sex in ("female", "couple"):
            if req.age >= 35 or req.irregular_cycles or req.symptoms_acne or req.symptoms_hirsutism:
                suggest_amh = True
                reasoning.append("AMH can help assess ovarian reserve; useful with age 35+, irregular cycles, or PCOS-related symptoms.")
            if req.years_trying is not None and req.years_trying >= 1 and not req.previous_tests_amh:
                suggest_amh = True
                reasoning.append("Trying for a year or more without prior AMH suggests testing.")

        if req.sex in ("male", "couple"):
            if req.years_trying is not None and req.years_trying >= 1 and not req.previous_tests_semen:
                suggest_semen = True
                reasoning.append("Semen analysis is often recommended when trying for 1+ year.")

        if req.years_trying is not None and req.years_trying >= 2:
            suggest_specialist = True
            reasoning.append("Trying for 2+ years warrants a specialist consultation.")
        if req.age >= 38 and req.sex in ("female", "couple"):
            suggest_specialist = True
            reasoning.append("Age 38+ suggests earlier specialist review.")
        if req.irregular_cycles and (req.symptoms_heavy_bleeding or req.symptoms_pain):
            suggest_specialist = True
            reasoning.append("Irregular cycles with heavy bleeding or pain should be evaluated by a doctor.")

        if suggest_amh or suggest_semen:
            when_to_test = "Testing is typically done early in the menstrual cycle (day 2–5) for AMH; semen analysis can be scheduled as per lab advice. Consult your doctor for timing."
        if suggest_specialist:
            when_to_test = (when_to_test + " " if when_to_test else "") + "Booking a fertility consultation is recommended to interpret results and plan next steps."

        ai_insight = None
        if req.use_ai_insight and self.knowledge_engine:
            q = (
                f"Very brief: when should someone consider AMH test or semen analysis for fertility? "
                f"Age {req.age}, sex {req.sex}, irregular cycles {req.irregular_cycles}, years trying {req.years_trying}. "
                "2–3 sentences, IVF/fertility context only."
            )
            ai_insight = self._get_ai_insight(q, req.language or "en")

        return HormonalPredictorResponse(
            suggest_amh=suggest_amh,
            suggest_semen_analysis=suggest_semen,
            suggest_specialist=suggest_specialist,
            when_to_test=when_to_test.strip(),
            reasoning=reasoning,
            ai_insight=ai_insight,
        )

    # --- 3. Face / Visual Health Indicator (Exploratory, non-diagnostic) ---

    def visual_health(self, req: VisualHealthRequest) -> VisualHealthResponse:
        """Wellness and reproductive health awareness from self-reported inputs; image optional/future."""
        disclaimer = (
            "This is for general wellness awareness only and is not a medical diagnosis. "
            "Always consult a healthcare provider for medical advice."
        )
        indicators: dict = {}
        recommendations: List[str] = []

        if req.self_reported_sleep_hours is not None:
            indicators["sleep_hours"] = req.self_reported_sleep_hours
            if req.self_reported_sleep_hours < 6:
                recommendations.append("Improving sleep (7–8 hours) can support overall and reproductive wellness.")
            elif req.self_reported_sleep_hours >= 7:
                indicators["sleep_ok"] = True

        if req.self_reported_stress_level:
            indicators["stress_level"] = req.self_reported_stress_level
            if req.self_reported_stress_level == "high":
                recommendations.append("Managing stress (e.g. relaxation, exercise) may benefit general and reproductive health.")

        if req.self_reported_bmi is not None:
            indicators["bmi"] = req.self_reported_bmi
            if req.self_reported_bmi < 18.5 or req.self_reported_bmi > 30:
                recommendations.append("BMI in a healthy range (18.5–30) is often recommended for fertility; discuss with your doctor.")

        if req.image_base64:
            indicators["image_provided"] = True
            recommendations.append("Image-based analysis is exploratory; AI may provide general wellness observations (non-diagnostic).")

        if not recommendations:
            recommendations.append("Your inputs suggest general wellness focus; maintain a balanced lifestyle and see a doctor for any concerns.")

        ai_insight = None
        if req.use_ai_insight and self.knowledge_engine:
            if req.image_base64:
                q = (
                    "Based on this image, give 2 brief sentences of general wellness and lifestyle awareness that may support reproductive health. "
                    "Strictly non-diagnostic, for awareness only. Do not diagnose; suggest general wellness only. IVF/fertility context."
                )
                ai_insight = self._get_ai_insight(q, req.language or "en", image=req.image_base64)
            else:
                q = (
                    "Brief 2 sentences: general wellness and lifestyle tips that may support reproductive health. "
                    "Non-diagnostic, awareness only. IVF/fertility context."
                )
                ai_insight = self._get_ai_insight(q, req.language or "en")

        return VisualHealthResponse(
            disclaimer=disclaimer,
            wellness_indicators=indicators,
            recommendations=recommendations,
            ai_insight=ai_insight,
        )

    # --- 4. Treatment Pathway Recommender ---

    def treatment_pathway(self, req: TreatmentPathwayRequest) -> TreatmentPathwayResponse:
        """Recommend natural conception support, IUI, IVF, or fertility preservation."""
        pathways: List[str] = []
        reasoning: List[str] = []
        primary = "consultation_recommended"

        if req.preserving_fertility:
            pathways.append("fertility_preservation")
            reasoning.append("You indicated interest in fertility preservation; egg or sperm freezing may be discussed with a specialist.")

        if req.years_trying is None or req.years_trying < 1:
            pathways.append("natural_conception_support")
            reasoning.append("Optimizing timing and lifestyle can support natural conception in the first year.")
            primary = "natural_conception_support"

        if req.known_diagnosis:
            d = " ".join(req.known_diagnosis).lower()
            if "tubal" in d or "male factor" in d or "severe" in d:
                pathways.append("ivf")
                primary = "ivf"
                reasoning.append("Some diagnoses (e.g. tubal factor, significant male factor) often lead to IVF discussion.")
            elif "ovulation" in d or "pcos" in d or "mild" in d:
                pathways.append("iui")
                if primary == "consultation_recommended":
                    primary = "iui"
                reasoning.append("Ovulation issues or mild male factor may be addressed with IUI or ovulation induction; specialist can advise.")

        if req.years_trying is not None and req.years_trying >= 2 and "ivf" not in pathways:
            pathways.append("iui")
            pathways.append("ivf")
            reasoning.append("After 2+ years of trying, IUI or IVF may be considered depending on workup.")
            if primary == "consultation_recommended":
                primary = "ivf"

        if req.age >= 38 and "fertility_preservation" not in pathways:
            pathways.append("fertility_preservation")
            reasoning.append("Age 38+ may warrant discussion of fertility preservation with a specialist.")

        pathways = list(dict.fromkeys(pathways))
        if not pathways:
            pathways = ["natural_conception_support", "consultation_recommended"]
            primary = "consultation_recommended"

        ai_insight = None
        if req.use_ai_insight and self.knowledge_engine:
            q = (
                f"Very brief: what treatment pathway might be considered for age {req.age}, "
                f"years trying {req.years_trying}, diagnosis {req.known_diagnosis}? "
                "Natural conception, IUI, or IVF. 2–3 sentences only. IVF context."
            )
            ai_insight = self._get_ai_insight(q, req.language or "en")

        return TreatmentPathwayResponse(
            suggested_pathways=pathways,
            primary_recommendation=primary,
            reasoning=reasoning,
            ai_insight=ai_insight,
        )

    # --- 5. Home IVF Eligibility Checker ---

    def home_ivf_eligibility(self, req: HomeIVFEligibilityRequest) -> HomeIVFEligibilityResponse:
        """Check whether the couple may be suitable for Home IVF and prompt consultation."""
        eligible = True
        reasons: List[str] = []
        missing: List[str] = []
        prompt_consultation = True
        booking_message = "Book a consultation to confirm eligibility and discuss your Home IVF plan."

        if req.female_age > 45:
            eligible = False
            reasons.append("Female age over 45 is often outside typical Home IVF criteria.")
        elif req.female_age >= 40:
            reasons.append("Age 40+ may require specialist assessment for Home IVF suitability.")

        if req.medical_contraindications:
            for c in req.medical_contraindications:
                if any(k in c.lower() for k in ("ohss", "severe", "uncontrolled", "cancer", "heart")):
                    eligible = False
                    reasons.append(f"Medical contraindication: {c}")

        if not req.has_consulted_specialist:
            missing.append("Consultation with a fertility specialist to confirm suitability.")
        if not req.ovarian_reserve_known and eligible:
            missing.append("Ovarian reserve assessment (e.g. AMH) helps tailor Home IVF protocol.")
        if not req.semen_analysis_known and req.male_age is not None and eligible:
            missing.append("Semen analysis helps confirm male factor suitability for Home IVF.")

        if not req.stable_relationship_or_single_with_donor:
            missing.append("Home IVF typically requires a clear plan (partner or donor).")

        if not reasons:
            reasons.append("Based on the information provided, you may be a candidate for Home IVF; a consultation will confirm.")

        ai_insight = None
        if req.use_ai_insight and self.knowledge_engine:
            q = (
                "One sentence: who might be suitable for Home IVF and what they should do next. "
                "Encourage consultation. IVF context only."
            )
            ai_insight = self._get_ai_insight(q, req.language or "en")

        return HomeIVFEligibilityResponse(
            eligible=eligible,
            reasons=reasons,
            missing_criteria=missing,
            prompt_consultation=prompt_consultation,
            booking_message=booking_message,
            ai_insight=ai_insight,
        )
