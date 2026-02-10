"""
AI-driven engagement tools: fertility readiness, hormonal predictor,
visual wellness, treatment pathway, and Home IVF eligibility.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging

from app.database.connection import get_db
from app.services.knowledge_engine import KnowledgeEngine
from app.services.engagement_service import EngagementService
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

logger = logging.getLogger(__name__)

router = APIRouter()


def get_engagement_service(db: Session = Depends(get_db)) -> EngagementService:
    """Engagement service with optional MedGemma via KnowledgeEngine."""
    knowledge_engine = KnowledgeEngine(db)
    return EngagementService(knowledge_engine=knowledge_engine)


@router.post(
    "/fertility-readiness",
    response_model=FertilityReadinessResponse,
    summary="Fertility Readiness Calculator",
    description="""
    **AI Fertility Readiness Calculator** – Based on age, medical history, lifestyle,
    menstrual patterns, and past pregnancy history to give a **preliminary fertility risk score**
    and next-step guidance. Non-diagnostic; encourages consultation for personalized advice.
    Optionally uses MedGemma for a short AI insight.
    """,
)
async def fertility_readiness(
    request: FertilityReadinessRequest,
    service: EngagementService = Depends(get_engagement_service),
) -> FertilityReadinessResponse:
    return service.fertility_readiness(request)


@router.post(
    "/hormonal-predictor",
    response_model=HormonalPredictorResponse,
    summary="Hormonal & Ovarian Health Predictor",
    description="""
    **Hormonal & Ovarian Health Predictor** – AI-led questionnaire + data logic suggesting
    **when to test AMH**, **semen analysis**, or **consult a specialist**. Provides reasoning
    and optional MedGemma insight.
    """,
)
async def hormonal_predictor(
    request: HormonalPredictorRequest,
    service: EngagementService = Depends(get_engagement_service),
) -> HormonalPredictorResponse:
    return service.hormonal_predictor(request)


@router.post(
    "/visual-health",
    response_model=VisualHealthResponse,
    summary="Face / Visual Health Indicator (Exploratory)",
    description="""
    **Face or Visual Health Indicator (Exploratory)** – Uses self-reported sleep, stress, BMI,
    and optional image for **wellness and reproductive health awareness**. Strictly non-diagnostic;
    for general awareness only. Optional MedGemma insight.
    """,
)
async def visual_health(
    request: VisualHealthRequest,
    service: EngagementService = Depends(get_engagement_service),
) -> VisualHealthResponse:
    return service.visual_health(request)


@router.post(
    "/treatment-pathway",
    response_model=TreatmentPathwayResponse,
    summary="Personalized Treatment Pathway Recommender",
    description="""
    **Personalized Treatment Pathway Recommender** – Guides users toward **natural conception support**,
    **IUI**, **IVF**, or **fertility preservation** based on age, duration trying, diagnosis,
    and previous treatments. Optional MedGemma insight.
    """,
)
async def treatment_pathway(
    request: TreatmentPathwayRequest,
    service: EngagementService = Depends(get_engagement_service),
) -> TreatmentPathwayResponse:
    return service.treatment_pathway(request)


@router.post(
    "/home-ivf-eligibility",
    response_model=HomeIVFEligibilityResponse,
    summary="Home IVF Eligibility Checker",
    description="""
    **Home IVF Eligibility Checker** – Instantly indicates whether a couple may be **suitable for Home IVF**
    based on age, medical contraindications, and prior workup. Prompts **consultation booking**
    for confirmation. Optional MedGemma insight.
    """,
)
async def home_ivf_eligibility(
    request: HomeIVFEligibilityRequest,
    service: EngagementService = Depends(get_engagement_service),
) -> HomeIVFEligibilityResponse:
    return service.home_ivf_eligibility(request)
