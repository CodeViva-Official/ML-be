from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from prompts.templates import QUESTION_GENERATION_PROMPT
from app.gemini_client import call_gemini

router = APIRouter()


# ── 요청/응답 스키마 ──────────────────────────────────────────
class GenerateQuestionsRequest(BaseModel):
    submission_id: str   # 백엔드에서 과제 식별용으로 넘겨주는 ID
    code: str            # 학생이 제출한 코드 전문


class QuestionItem(BaseModel):
    id: int
    question: str
    intent: str


class GenerateQuestionsResponse(BaseModel):
    submission_id: str
    questions: list[QuestionItem]


# ── 라우터 ────────────────────────────────────────────────────
@router.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions(req: GenerateQuestionsRequest):
    """
    코드를 받아 Gemini로 검증 질문 3개 생성.
    백엔드 서버가 이 엔드포인트를 호출하고, 응답을 DB에 저장 후 프론트에 전달.
    """
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="코드가 비어있습니다.")

    prompt = QUESTION_GENERATION_PROMPT.format(code=req.code)

    try:
        result = await call_gemini(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {str(e)}")

    return GenerateQuestionsResponse(
        submission_id=req.submission_id,
        questions=result["questions"],
    )
