"""
Page translation API using googletrans.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from app.utils.translator import translation_service, PAGE_LANGUAGES

router = APIRouter()


class TranslateRequest(BaseModel):
    """Request body for batch translate"""
    texts: List[str] = Field(..., description="List of text strings to translate")
    dest: str = Field(..., description="Target language code (e.g. en, hi, es, fr)")
    src: Optional[str] = Field(default=None, description="Source language code (auto-detect if omitted)")


class TranslateResponse(BaseModel):
    """Response with translated strings"""
    translations: List[str]
    dest: str


@router.get(
    "/translate/languages",
    response_model=dict,
    summary="List page translation languages",
    description="Returns language codes and names for the page translation dropdown.",
)
def list_languages():
    """Return supported page translation languages (googletrans codes)."""
    return {"languages": PAGE_LANGUAGES}


@router.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate texts",
    description="Translate a list of strings to the target language using googletrans.",
)
def translate_texts(request: TranslateRequest):
    """Batch translate strings for page UI."""
    if not request.texts:
        return TranslateResponse(translations=[], dest=request.dest)
    # Limit batch size to avoid timeouts
    max_batch = 100
    if len(request.texts) > max_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Too many texts (max {max_batch}). Send in smaller batches.",
        )
    try:
        translations = translation_service.batch_translate(
            request.texts,
            target_lang=request.dest,
            source_lang=request.src,
        )
        return TranslateResponse(translations=translations, dest=request.dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
