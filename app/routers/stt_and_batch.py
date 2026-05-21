import re
import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from prompts.templates import EVALUATION_SYSTEM, EVALUATION_USER
from app.gemini_client import call_llm
from app.stt_service import transcribe_audio

router = APIRouter()


# ── 불필요 토큰 제거 ──────────────────────────────────────────
FILLER_PATTERNS = [
    r'\b(음+|어+|아+|에+|으+)\b',       # 음, 어, 아, 에, 으
    r'\b(그+니까|그러니까)\b',            # 그니까
    r'\b(뭐지|뭐더라|뭐였지)\b',          # 뭐지, 뭐더라
    r'\b(약간|좀|이제|그냥)\b',           # 불필요 부사
    r'\b(그래서+|그리고+)\s*\1',          # 반복 접속사
    r'\s{2,}',                           # 다중 공백
]


def remove_filler_tokens(text: str) -> str:
    """
    STT 결과에서 음, 아 같은 불필요 토큰 제거.
    하드코딩 패턴 + 다중 공백 정리.
    """
    cleaned = text
    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ── API 1: STT 전용 ──────────────────────────────────────────
class SttResponse(BaseModel):
    raw_stt: str          # Whisper 원본
    cleaned_text: str     # 불필요 토큰 제거 후 (최종)


@router.post("/stt", response_model=SttResponse)
async def stt_only(
    audio_file: UploadFile = File(...),
):
    """
    음성 → STT → 불필요 토큰 제거 → 텍스트 반환.
    채점은 하지 않음. 프론트에서 텍스트를 보여주는 용도.
    """
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="음성 파일이 비어있습니다.")

    try:
        raw_stt = transcribe_audio(audio_bytes, audio_file.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 실패: {str(e)}")

    cleaned = remove_filler_tokens(raw_stt)

    return SttResponse(
        raw_stt=raw_stt,
        cleaned_text=cleaned,
    )


# ── API 2: 배치 채점 ─────────────────────────────────────────
class BatchQuestionItem(BaseModel):
    submission_id: str
    question_type: int       # 1, 2, 3
    code: str
    question: str
    answer_text: str         # STT → 정제 → 클리닝 완료된 텍스트


class BatchEvaluateRequest(BaseModel):
    questions: list[BatchQuestionItem]   # 모든 학생의 답변 (제한 없음)


class SingleResult(BaseModel):
    submission_id: str
    question_type: int
    answer_text: str
    level: str               # excellent | good | bad | miss
    accuracy: str
    depth: str
    summary: str


class BatchEvaluateResponse(BaseModel):
    results: list[SingleResult]


@router.post("/batch-evaluate", response_model=BatchEvaluateResponse)
async def batch_evaluate(req: BatchEvaluateRequest):
    """
    모든 학생의 답변을 한번에 받아서 프롬프트 2로 채점.
    전부 동시에 LLM 호출 (비동기 병렬).
    프론트에서 "결과 보기" 버튼 누를 때 호출.
    """
    if len(req.questions) == 0:
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    async def evaluate_single(q: BatchQuestionItem) -> SingleResult:
        user_prompt = EVALUATION_USER.format(
            code=q.code,
            question=q.question,
            answer=q.answer_text,
        )

        try:
            result = await call_llm(
                user_prompt=user_prompt,
                system_prompt=EVALUATION_SYSTEM,
                response_type="evaluation",
            )
        except Exception as e:
            return SingleResult(
                submission_id=q.submission_id,
                question_type=q.question_type,
                answer_text=q.answer_text,
                level="miss",
                accuracy=f"채점 실패: {str(e)}",
                depth="",
                summary="LLM 호출 실패로 miss 처리",
            )

        return SingleResult(
            submission_id=q.submission_id,
            question_type=q.question_type,
            answer_text=q.answer_text,
            level=result.get("level", "miss"),
            accuracy=result.get("accuracy", ""),
            depth=result.get("depth", ""),
            summary=result.get("summary", ""),
        )

    # 전부 동시 호출 (비동기 병렬)
    results = await asyncio.gather(
        *[evaluate_single(q) for q in req.questions]
    )

    return BatchEvaluateResponse(
        results=list(results),
    )