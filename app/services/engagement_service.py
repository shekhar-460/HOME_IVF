"""
AI-driven engagement tools: fertility readiness, hormonal predictor,
visual wellness (exploratory), treatment pathway, and Home IVF eligibility.
Uses rule-based logic + optional MedGemma for explanations.
"""
from typing import Optional, List, Set, Tuple, TYPE_CHECKING
import logging
import re

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

# Canonical fertility-relevant conditions and accepted variations (lowercase, no extra spaces).
# Only recognized conditions are used in scoring; unrecognized entries are reported and not scored.
MEDICAL_HISTORY_CANONICAL: List[Tuple[str, Set[str]]] = [
    ("PCOS", {"pcos", "pcod", "pco", "pocs", "polycystic", "polycystic ovary", "polycystic ovarian", "polycystic ovarian syndrome"}),
    ("Endometriosis", {"endometriosis", "endo"}),
    ("Thyroid disorder", {"thyroid", "hypothyroid", "hyperthyroid", "hypothyroidism", "hyperthyroidism"}),
    ("Diabetes", {"diabetes", "diabetic", "type 1", "type 2", "prediabetes"}),
    ("Prior pelvic/fertility surgery", {"prior surgery", "surgery", "pelvic surgery", "fertility surgery", "laparoscopy", "laparoscopic"}),
    ("Tubal factor", {"tubal", "tubal factor", "blocked tubes", "tubal blockage", "fallopian"}),
]


def _normalize_medical_history(raw_list: Optional[List[str]]) -> Tuple[Set[str], List[str]]:
    """
    Map free-text medical history entries to canonical conditions.
    Returns (set of canonical names used in scoring, list of unrecognized raw entries).
    Unrecognized entries are not used in scoring to avoid invalid data affecting results.
    """
    if not raw_list:
        return set(), []
    recognized: Set[str] = set()
    unrecognized: List[str] = []
    # Normalize input: strip, collapse spaces, lowercase for matching
    for raw in raw_list:
        s = re.sub(r"\s+", " ", (raw or "").strip()).lower()
        if not s:
            continue
        matched = False
        for canonical, aliases in MEDICAL_HISTORY_CANONICAL:
            if s in aliases:
                recognized.add(canonical)
                matched = True
                break
            # Allow "condition" or "history of condition" (alias appears as word in user input)
            if any(alias in s for alias in aliases if len(alias) >= 3):
                recognized.add(canonical)
                matched = True
                break
        if not matched:
            unrecognized.append(raw.strip())
    return recognized, unrecognized


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

        # Medical history: only recognized conditions affect the score; unrecognized entries are ignored
        recognized_conditions, unrecognized_entries = _normalize_medical_history(req.medical_history)
        for canonical in recognized_conditions:
            score += 15
            reasons.append(f"Medical history ({canonical}) may affect fertility; specialist review recommended.")

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
                reasons.append("BMI below 18.5 or above 30 may affect cycle regularity and outcomes.")

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
            medical_history_recognized=sorted(recognized_conditions) if recognized_conditions else None,
            medical_history_unrecognized=unrecognized_entries if unrecognized_entries else None,
        )

    # --- 2. Hormonal & Ovarian Health Predictor ---

    def hormonal_predictor(self, req: HormonalPredictorRequest) -> HormonalPredictorResponse:
        """Suggest when to test AMH, semen analysis, or consult a specialist."""
        suggest_amh = False
        suggest_semen = False
        suggest_specialist = False
        reasoning: List[str] = []
        when_to_test = ""

        if req.sex == "female":
            if (req.age >= 35 or req.irregular_cycles or req.symptoms_acne or req.symptoms_hirsutism) and not req.previous_tests_amh:
                suggest_amh = True
                reasoning.append("AMH can help assess ovarian reserve; useful with age 35+, irregular cycles, or PCOS-related symptoms.")
            if req.years_trying is not None and req.years_trying >= 1 and not req.previous_tests_amh:
                suggest_amh = True
                reasoning.append("Trying for a year or more without prior AMH suggests testing.")

        if req.sex == "male":
            if req.years_trying is not None and req.years_trying >= 1 and not req.previous_tests_semen:
                suggest_semen = True
                reasoning.append("Semen analysis is often recommended when trying for 1+ year.")

        if req.years_trying is not None and req.years_trying >= 2:
            suggest_specialist = True
            reasoning.append("Trying for 2+ years warrants a specialist consultation.")
        if req.age >= 38 and req.sex == "female":
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
            try:
                if req.sex == "male":
                    q = (
                        f"Very brief: when should a male consider semen analysis for fertility? "
                        f"Age {req.age}, years trying {req.years_trying or 'not specified'}. "
                        "Do NOT mention menstrual cycles, AMH, or ovarian reserve. 2–3 sentences, IVF/fertility context only."
                    )
                else:
                    q = (
                        f"Very brief: when should someone consider AMH test or semen analysis for fertility? "
                        f"Age {req.age}, sex female, irregular cycles {req.irregular_cycles}, years trying {req.years_trying or 'not specified'}. "
                        "2–3 sentences, IVF/fertility context only."
                    )
                ai_insight = self._get_ai_insight(q, req.language or "en")
            except Exception as e:
                logger.warning(f"AI insight for hormonal predictor failed: {e}", exc_info=True)
                ai_insight = None

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
            "This is general wellness awareness only, not a medical diagnosis. "
            "For any health or fertility concerns, please speak with your doctor or a healthcare provider."
        )
        indicators: dict = {}
        recommendations: List[str] = []
        wellness_parts: List[str] = []

        if req.self_reported_sleep_hours is not None:
            indicators["sleep_hours"] = req.self_reported_sleep_hours
            wellness_parts.append(f"{req.self_reported_sleep_hours} hours of sleep")
            if req.self_reported_sleep_hours < 6:
                recommendations.append("Aim for 7–8 hours of sleep when you can—it can help your overall health and wellbeing.")
            elif req.self_reported_sleep_hours >= 7:
                indicators["sleep_ok"] = True

        if req.self_reported_stress_level:
            indicators["stress_level"] = req.self_reported_stress_level
            wellness_parts.append(f"{req.self_reported_stress_level} stress")
            if req.self_reported_stress_level == "high":
                recommendations.append("Finding time to relax—e.g. gentle exercise, meditation, or time outdoors—can help you feel better and support your health.")

        if req.self_reported_bmi is not None:
            indicators["bmi"] = req.self_reported_bmi
            wellness_parts.append(f"BMI {req.self_reported_bmi}")
            if req.self_reported_bmi < 18.5 or req.self_reported_bmi > 30:
                recommendations.append("A healthy weight range often supports fertility; your doctor can help you with a plan that’s right for you.")

        image_analysis = None
        if req.image_base64:
            indicators["image_provided"] = True
            wellness_parts.append("a photo")

        wellness_summary = None
        if wellness_parts:
            wellness_summary = "You shared: " + ", ".join(wellness_parts) + "."

        if not recommendations:
            recommendations.append("Keeping a balanced lifestyle and talking to your doctor about any concerns is a good approach.")

        ai_insight = None
        if req.use_ai_insight and self.knowledge_engine:
            if req.image_base64:
                q = (
                    "Based on this image, give 2 brief sentences of general wellness and lifestyle awareness that may support reproductive health. "
                    "Strictly non-diagnostic, for awareness only. Do not diagnose; suggest general wellness only. IVF/fertility context."
                )
                image_analysis = self._get_ai_insight(q, req.language or "en", image=req.image_base64)
                if image_analysis:
                    recommendations.insert(0, "The note below is based on your photo and is for general awareness only—it is not a diagnosis.")
            else:
                q = (
                    "Brief 2 sentences: general wellness and lifestyle tips that may support reproductive health. "
                    "Non-diagnostic, awareness only. IVF/fertility context."
                )
                ai_insight = self._get_ai_insight(q, req.language or "en")

        if image_analysis:
            summary = "We’ve looked at what you shared. Below are general wellness ideas that may support your wellbeing. This is not a medical diagnosis."
        elif wellness_parts:
            summary = "Here are some simple wellness suggestions based on what you told us. For personalised advice, please consult your doctor."
        else:
            summary = "Here are some general wellness tips. For any health or fertility questions, your doctor is the best person to talk to."

        return VisualHealthResponse(
            summary=summary,
            disclaimer=disclaimer,
            wellness_summary=wellness_summary,
            wellness_indicators=indicators,
            recommendations=recommendations,
            image_analysis=image_analysis,
            ai_insight=ai_insight,
        )

    # --- 4. Treatment Pathway Recommender ---

    def treatment_pathway(self, req: TreatmentPathwayRequest) -> TreatmentPathwayResponse:
        """Recommend natural conception support, IUI, IVF, or fertility preservation based on age, diagnosis, previous treatments, and duration trying."""
        pathways: List[str] = []
        reasoning: List[str] = []
        diagnosis_text = " ".join(req.known_diagnosis or []).lower()
        previous_text = " ".join(req.previous_treatments or []).lower()
        years = req.years_trying

        # 1) Fertility preservation: explicit interest or age 38+
        if req.preserving_fertility:
            pathways.append("fertility_preservation")
            reasoning.append("You indicated interest in fertility preservation; egg or sperm freezing may be discussed with a specialist.")
        if req.age >= 38 and "fertility_preservation" not in pathways:
            pathways.append("fertility_preservation")
            reasoning.append("Age 38+ may warrant discussion of fertility preservation with a specialist.")

        # 2) Diagnosis-driven: strong indicators for IVF vs IUI
        if req.known_diagnosis:
            if any(k in diagnosis_text for k in ("tubal", "male factor", "severe", "azoospermia", "blocked tube")):
                pathways.append("ivf")
                reasoning.append("Diagnoses such as tubal factor, significant male factor, or azoospermia often lead to IVF discussion with a specialist.")
            elif any(k in diagnosis_text for k in ("ovulation", "pcos", "pcod", "mild", "unexplained")):
                pathways.append("iui")
                reasoning.append("Ovulation issues, PCOS, or mild male factor may be addressed with IUI or ovulation induction; a specialist can advise.")

        # 3) Previous treatments: failed or repeated IUI → consider IVF; already did IVF → keep IVF in pathway
        if req.previous_treatments:
            if "ivf" in previous_text:
                pathways.append("ivf")
                reasoning.append("You have already had or considered IVF; a specialist can advise on next steps.")
            elif "iui" in previous_text and (years is None or years >= 1):
                pathways.append("iui")
                pathways.append("ivf")
                reasoning.append("After previous IUI, both repeat IUI and IVF may be options depending on your workup and preferences.")

        # 4) Duration trying: first year → natural support; 2+ years → IUI/IVF discussion
        if years is None or years < 1:
            pathways.append("natural_conception_support")
            reasoning.append("In the first year, optimizing timing and lifestyle can support natural conception; a specialist can help if needed.")
        elif years >= 2:
            if "ivf" not in pathways:
                pathways.append("ivf")
            if "iui" not in pathways:
                pathways.append("iui")
            reasoning.append("After 2+ years of trying, IUI or IVF may be considered depending on workup and diagnosis.")

        # 5) Always suggest consultation when multiple options or when pathway is not only natural
        if "natural_conception_support" in pathways and len(pathways) > 1:
            pathways.append("consultation_recommended")
            reasoning.append("A fertility specialist can help choose the right option for your situation.")
        elif "ivf" in pathways or "iui" in pathways or "fertility_preservation" in pathways:
            pathways.append("consultation_recommended")

        pathways = list(dict.fromkeys(pathways))

        # Primary recommendation: priority IVF > IUI > fertility_preservation > natural > consultation
        if "ivf" in pathways:
            primary = "ivf"
        elif "iui" in pathways:
            primary = "iui"
        elif req.preserving_fertility and "fertility_preservation" in pathways:
            primary = "fertility_preservation"
        elif "natural_conception_support" in pathways and (years is None or years < 1) and "ivf" not in pathways and "iui" not in pathways:
            primary = "natural_conception_support"
        elif "fertility_preservation" in pathways:
            primary = "fertility_preservation"
        else:
            primary = "consultation_recommended"

        if not pathways:
            pathways = ["natural_conception_support", "consultation_recommended"]
            primary = "consultation_recommended"

        def _pathway_display(s: str) -> str:
            if s == "ivf":
                return "IVF"
            if s == "iui":
                return "IUI"
            return s

        suggested_display = [_pathway_display(p) for p in pathways]
        primary_display = _pathway_display(primary)

        ai_insight = None
        if req.use_ai_insight and self.knowledge_engine:
            q = (
                f"Very brief: what treatment pathway might be considered for age {req.age}, "
                f"years trying {req.years_trying}, diagnosis {req.known_diagnosis}? "
                "Natural conception, IUI, or IVF. 2–3 sentences only. IVF context."
            )
            if req.other_information:
                q += " Other info: " + (req.other_information or "")
            ai_insight = self._get_ai_insight(q, req.language or "en")

        return TreatmentPathwayResponse(
            suggested_pathways=suggested_display,
            primary_recommendation=primary_display,
            reasoning=reasoning,
            ai_insight=ai_insight,
        )

    # --- 5. Home IVF Eligibility Checker ---

    def _normalize_contraindications(
        self, raw: Optional[List[str]]
    ) -> List[str]:
        """Accept list or comma-separated string; return trimmed non-empty list."""
        if raw is None:
            return []
        out: List[str] = []
        for item in raw:
            if isinstance(item, str):
                for part in item.split(","):
                    s = part.strip()
                    if s:
                        out.append(s)
            elif item is not None and str(item).strip():
                out.append(str(item).strip())
        return out

    def home_ivf_eligibility(self, req: HomeIVFEligibilityRequest) -> HomeIVFEligibilityResponse:
        """Check whether the couple may be suitable for Home IVF and prompt consultation. Uses female/male age, diagnosis, contraindications, and workup status."""
        eligible = True
        reasons: List[str] = []
        missing: List[str] = []
        prompt_consultation = True
        contraindications = self._normalize_contraindications(req.medical_contraindications)
        diagnosis_text = " ".join(req.known_diagnosis or []).lower()
        previous_text = " ".join(req.previous_treatments or []).lower()

        # Hard ineligibility: female age > 45
        if req.female_age > 45:
            eligible = False
            reasons.append("Female age over 45 is often outside typical Home IVF criteria; a specialist can discuss options.")
        elif req.female_age >= 40:
            reasons.append("Female age 40+ may require specialist assessment for Home IVF suitability.")

        # Male age: soft note if 50+
        if req.male_age is not None and req.male_age >= 50:
            reasons.append("Male age 50+ may warrant discussion of semen quality and suitability with a specialist.")

        # Diagnosis: some require specialist review (do not automatically set ineligible)
        if req.known_diagnosis:
            if any(k in diagnosis_text for k in ("tubal", "severe", "azoospermia", "blocked tube")):
                reasons.append("Some diagnoses (e.g. tubal factor, severe male factor) may require specialist review before Home IVF.")
            if "pcos" in diagnosis_text or "pcod" in diagnosis_text:
                reasons.append("PCOS can be managed in Home IVF; ovarian reserve and protocol should be confirmed with a specialist.")

        # Previous treatments: OHSS or multiple IVF may need mention
        if req.previous_treatments:
            if "ohss" in previous_text or "hyperstimulation" in previous_text:
                eligible = False
                reasons.append("Previous OHSS or hyperstimulation history is a consideration for Home IVF; specialist review is important.")
            elif "ivf" in previous_text:
                reasons.append("Previous IVF experience is useful; a specialist can confirm if Home IVF is appropriate for you.")

        # Medical contraindications: serious keywords → ineligible
        for c in contraindications:
            cl = c.lower()
            if any(k in cl for k in ("ohss", "severe ohss", "uncontrolled", "cancer", "heart disease", "severe hypertension", "thrombosis")):
                eligible = False
                reasons.append(f"Medical contraindication noted: {c}. A specialist can advise on suitability.")

        if not req.has_consulted_specialist:
            missing.append("Consultation with a fertility specialist to confirm suitability.")
        if not req.ovarian_reserve_known and eligible:
            missing.append("Ovarian reserve assessment (e.g. AMH) helps tailor the Home IVF protocol.")
        if not req.semen_analysis_known and req.male_age is not None and eligible:
            missing.append("Semen analysis helps confirm male factor suitability for Home IVF.")
        if not req.stable_relationship_or_single_with_donor:
            missing.append("Home IVF typically requires a clear plan (partner or donor).")

        if not reasons:
            reasons.append("Based on the information provided, you may be a candidate for Home IVF; a consultation will confirm.")

        if eligible:
            booking_message = "Book a consultation to confirm eligibility and discuss your Home IVF plan."
        else:
            booking_message = "Book a consultation to discuss your situation and explore options, including whether Home IVF or clinic-based care is more suitable."

        ai_insight = None
        if req.use_ai_insight and self.knowledge_engine:
            q = (
                "One sentence: who might be suitable for Home IVF and what they should do next. "
                "Encourage consultation. IVF context only."
            )
            if req.known_diagnosis or req.previous_treatments or req.other_information:
                q += " Diagnosis: " + str(req.known_diagnosis or []) + ". Previous treatments: " + str(req.previous_treatments or [])
                if req.other_information:
                    q += " Other info: " + (req.other_information or "")
            ai_insight = self._get_ai_insight(q, req.language or "en")

        return HomeIVFEligibilityResponse(
            eligible=eligible,
            reasons=reasons,
            missing_criteria=missing,
            prompt_consultation=prompt_consultation,
            booking_message=booking_message,
            ai_insight=ai_insight,
        )
