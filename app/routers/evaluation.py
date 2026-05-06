from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from prompts.templates import EVALUATION_SYSTEM, EVALUATION_USER
from app.gemini_client import call_gemini
from app.stt_service import transcribe_audio

router = APIRouter()


# ── 개별 평가 응답 스키마 ─────────────────────────────────────
class SingleEvalResponse(BaseModel):
    submission_id: str
    question_type: int          # 1, 2, 3
    transcribed_text: str       # STT 결과
    level: str                  # excellent | good | bad | miss
    accuracy: str
    depth: str
    summary: str


# ── 개별 질문 평가 (3번 호출됨) ───────────────────────────────
@router.post("/evaluate-answer", response_model=SingleEvalResponse)
async def evaluate_answer(
    submission_id: str = Form(...),
    question_type: int = Form(...),        # 1, 2, 3
    code: str = Form(...),
    question: str = Form(...),             # 해당 질문 텍스트 (q1, q2, q3 중 하나)
    audio_file: UploadFile = File(...),
):
    """
    질문 1개 + 음성 답변 1개를 받아서:
    1. Whisper로 음성 → 텍스트
    2. Gemini 프롬프트 2로 이해도 채점
    3. 개별 결과 반환

    프론트에서 Q1 답변 → 이 API 호출 (type=1)
              Q2 답변 → 이 API 호출 (type=2)
              Q3 답변 → 이 API 호출 (type=3)
    """
    # ── 1. 음성 → 텍스트 (Whisper) ───────────────────────────
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="음성 파일이 비어있습니다.")

    try:
        transcribed_text = transcribe_audio(audio_bytes, audio_file.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 변환 실패: {str(e)}")

    # ── 2. Gemini 채점 ────────────────────────────────────────
    user_prompt = EVALUATION_USER.format(
        code=code,
        question=question,
        answer=transcribed_text,
    )

    try:
        result = await call_gemini(
            user_prompt=user_prompt,
            system_prompt=EVALUATION_SYSTEM,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {str(e)}")

    return SingleEvalResponse(
        submission_id=submission_id,
        question_type=question_type,
        transcribed_text=transcribed_text,
        level=result.get("level", "miss"),
        accuracy=result.get("accuracy", ""),
        depth=result.get("depth", ""),
        summary=result.get("summary", ""),
    )


# ── 종합 등급 산출 (코드 로직, LLM 미사용) ────────────────────
# ⚠️ 아래 기준은 임시입니다. ML팀 확정 후 이 함수만 수정하면 됩니다.

GRADE_ORDER = ["miss", "bad", "good", "excellent"]


def _grade_to_num(grade: str) -> int:
    """등급을 숫자로 변환. miss=0, bad=1, good=2, excellent=3"""
    try:
        return GRADE_ORDER.index(grade.lower())
    except ValueError:
        return 0


def _calculate_final_grade(q1: str, q2: str, q3: str) -> str:
    """
    ⚠️ 임시 종합 등급 기준 — 나중에 여기만 수정하면 됨
    
    현재 기준:
    - excellent: 3개 모두 excellent 또는 2개 excellent + 1개 good
    - good:      과반(2개 이상)이 good 이상
    - bad:       과반(2개 이상)이 bad 이하
    - miss:      과반(2개 이상)이 miss
    """
    levels = [q1.lower(), q2.lower(), q3.lower()]
    nums = [_grade_to_num(l) for l in levels]

    excellent_count = levels.count("excellent")
    good_count = levels.count("good")
    bad_count = levels.count("bad")
    miss_count = levels.count("miss")

    if excellent_count >= 3 or (excellent_count == 2 and good_count >= 1):
        return "excellent"
    elif (excellent_count + good_count) >= 2:
        return "good"
    elif (bad_count + miss_count) >= 2:
        if miss_count >= 2:
            return "miss"
        return "bad"
    else:
        return "good"


class FinalGradeRequest(BaseModel):
    submission_id: str
    q1_level: str  # excellent | good | bad | miss
    q2_level: str
    q3_level: str


class FinalGradeResponse(BaseModel):
    submission_id: str
    final_grade: str  # excellent | good | bad | miss
    q1_level: str
    q2_level: str
    q3_level: str


@router.post("/evaluate-final", response_model=FinalGradeResponse)
async def evaluate_final(req: FinalGradeRequest):
    """
    3개의 개별 평가 등급을 받아 종합 등급 산출.
    LLM 호출 없이 코드 로직으로 처리.
    기준 변경 시 _calculate_final_grade() 함수만 수정.
    """
    final = _calculate_final_grade(req.q1_level, req.q2_level, req.q3_level)

    return FinalGradeResponse(
        submission_id=req.submission_id,
        final_grade=final,
        q1_level=req.q1_level,
        q2_level=req.q2_level,
        q3_level=req.q3_level,
    )