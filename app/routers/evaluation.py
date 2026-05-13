from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from prompts.templates import EVALUATION_SYSTEM, EVALUATION_USER
from app.gemini_client import call_gemini
from app.stt_service import process_audio

router = APIRouter()


# ── 개별 평가 응답 스키마 ─────────────────────────────────────
class SingleEvalResponse(BaseModel):
    submission_id: str
    question_type: int          # 1, 2, 3
    raw_stt: str                # Whisper 원본 텍스트
    refined_text: str           # Qwen 보간 후 텍스트 (이걸로 채점)
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
    1. faster-whisper로 음성 → 텍스트
    2. Qwen으로 STT 텍스트 보간 (코드 용어 정제)
    3. Gemini 프롬프트 2로 이해도 채점
    4. 개별 결과 반환
    """
    # ── 1. 음성 → STT → 보간 ─────────────────────────────────
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="음성 파일이 비어있습니다.")

    try:
        stt_result = process_audio(
            audio_bytes=audio_bytes,
            filename=audio_file.filename or "audio.wav",
            submitted_code=code,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT/보간 처리 실패: {str(e)}")

    # ── 2. Gemini 채점 (보간된 텍스트 사용) ───────────────────
    user_prompt = EVALUATION_USER.format(
        code=code,
        question=question,
        answer=stt_result["refined"],
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
        raw_stt=stt_result["raw_stt"],
        refined_text=stt_result["refined"],
        level=result.get("level", "miss"),
        accuracy=result.get("accuracy", ""),
        depth=result.get("depth", ""),
        summary=result.get("summary", ""),
    )