import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from prompts.templates import EVALUATION_PROMPT
from app.gemini_client import call_gemini
from app.stt_service import transcribe_audio

router = APIRouter()


# ── 응답 스키마 ───────────────────────────────────────────────
class PerQuestionResult(BaseModel):
    id: int
    score: int
    feedback: str


class EvaluateAnswerResponse(BaseModel):
    submission_id: str
    transcribed_text: str        # STT 결과 텍스트 (백엔드가 저장용으로 씀)
    total_score: int
    grade: str
    per_question: list[PerQuestionResult]
    overall_feedback: str
    suspicion_flags: list[str]   # 의심 근거 목록. 비어있으면 정상


# ── 라우터 ────────────────────────────────────────────────────
@router.post("/evaluate-answer", response_model=EvaluateAnswerResponse)
async def evaluate_answer(
    submission_id: str = Form(...),
    code: str = Form(...),
    questions_json: str = Form(...),   # JSON 직렬화된 질문 리스트
    audio_file: UploadFile = File(...),
):
    """
    음성 파일 + 코드 + 질문들을 받아서:
    1. Whisper로 음성 → 텍스트
    2. Gemini로 이해도 채점
    3. 결과 반환 (백엔드가 DB 저장 + S3 URL 연결)

    multipart/form-data로 받는 이유:
    음성 파일(바이너리)과 텍스트 데이터를 동시에 전송할 때는
    JSON body를 쓸 수 없고 form-data를 써야 함.
    """
    # ── 1. 음성 → 텍스트 (Whisper) ───────────────────────────
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="음성 파일이 비어있습니다.")

    try:
        transcribed_text = transcribe_audio(audio_bytes, audio_file.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 변환 실패: {str(e)}")

    # ── 2. 질문 파싱 ──────────────────────────────────────────
    try:
        questions = json.loads(questions_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="questions_json 형식이 잘못됐습니다.")

    # 프롬프트에 넣을 Q&A 포맷으로 조립
    # 질문이 3개, 답변이 하나의 텍스트인 경우 — 전체 텍스트를 Q1 답변에 넣고 나머지는 "(동일 답변 참조)"
    # 실제 서비스에서는 질문별로 음성을 따로 받는 게 더 정확하지만, MVP에서는 한 번에 받음
    qa_pairs = "\n".join([
        f"Q{q['id']}: {q['question']}\nA{q['id']}: {transcribed_text if i == 0 else '(위 답변 참조)'}"
        for i, q in enumerate(questions)
    ])

    # ── 3. Gemini 채점 ────────────────────────────────────────
    prompt = EVALUATION_PROMPT.format(code=code, qa_pairs=qa_pairs)

    try:
        result = await call_gemini(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {str(e)}")

    return EvaluateAnswerResponse(
        submission_id=submission_id,
        transcribed_text=transcribed_text,
        **result,
    )
