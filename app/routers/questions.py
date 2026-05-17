from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from prompts.templates import QUESTION_GENERATION_SYSTEM, QUESTION_GENERATION_USER
from app.gemini_client import call_llm

router = APIRouter()


# ── 요청/응답 스키마 ──────────────────────────────────────────
class GenerateQuestionsRequest(BaseModel):
    submission_id: str
    code: str


class AnalysisResult(BaseModel):
    purpose: str
    key_concepts: list[str]
    complexity: str  # simple | medium | complex


class QuestionsResult(BaseModel):
    q1: str
    q2: str
    q3: str


class GenerateQuestionsResponse(BaseModel):
    submission_id: str
    analysis: AnalysisResult
    questions: QuestionsResult


# ── 라우터 ────────────────────────────────────────────────────
@router.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions(req: GenerateQuestionsRequest):
    """
    코드를 받아 ai로 검증 질문 3개 생성.
    프롬프트 1 사용. 코드 분석 결과 + 질문 3개를 반환.
    """
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="코드가 비어있습니다.")

    user_prompt = QUESTION_GENERATION_USER.format(code=req.code)

    try:
        result = await call_llm(
            user_prompt=user_prompt,
            system_prompt=QUESTION_GENERATION_SYSTEM,
            response_type="questions",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {str(e)}")

    return GenerateQuestionsResponse(
        submission_id=req.submission_id,
        analysis=result["analysis"],
        questions=result["questions"],
    )