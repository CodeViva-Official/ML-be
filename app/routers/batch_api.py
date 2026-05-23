"""
OpenAI Batch API 채점 라우터 (신규)

기존 /stt, /batch-evaluate 는 건드리지 않음.
이 파일은 비용 50% 절감용 Batch API 경로를 별도로 추가한다.

흐름:
  1) POST /batch-grade/submit   → 전체 학생 답변을 .jsonl로 만들어 OpenAI에 제출, batch_id 반환
  2) (OpenAI가 백그라운드 처리, 수 분~수십 분, SLA 24h)
  3) GET  /batch-grade/result/{batch_id} → 상태 확인 + 완료 시 채점 결과 반환

핵심: custom_id = "{submission_id}__q{question_type}" 로 유일하게 부여.
      출력 순서가 입력과 달라도 custom_id로 매칭하므로 학생 id 순서 무관.

answer_text: OpenAI 결과 파일에는 answer_text가 돌아오지 않으므로
      submit 시점에 {custom_id: answer_text} 를 보관했다가 GET 때 합친다.
      → 기존 /batch-evaluate 와 결과 필드를 일치시키기 위함.
"""

import io
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from openai import OpenAI
from app.config import settings
from prompts.templates import EVALUATION_SYSTEM, EVALUATION_USER

router = APIRouter()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

OPENAI_MODEL = settings.OPENAI_MODEL
VALID_LEVELS = {"excellent", "good", "bad", "miss"}

# ──────────────────────────────────────────────────────────────
# answer_text 보관소 (batch_id → {custom_id: answer_text})
#   [주의] 모듈 레벨 메모리 저장. 서버 재시작 시 사라진다.
#   영속성이 필요하면 DB/Redis 로 교체할 것.
#   백엔드 DB에 답변 원문이 이미 있다면, 이 보관소 대신
#   결과의 submission_id+question_type 으로 백엔드에서 붙이는 게 더 견고하다.
# ──────────────────────────────────────────────────────────────
_answer_store: dict[str, dict[str, str]] = {}


def make_custom_id(submission_id: str, question_type: int) -> str:
    return f"{submission_id}__q{question_type}"


def parse_custom_id(custom_id: str) -> tuple[str, int]:
    # "sub_001__q2" → ("sub_001", 2)
    sid, _, qpart = custom_id.rpartition("__q")
    try:
        return sid, int(qpart)
    except ValueError:
        return custom_id, 0


# ──────────────────────────────────────────────────────────────
# 요청/응답 모델
# ──────────────────────────────────────────────────────────────
class BatchQuestionItem(BaseModel):
    submission_id: str
    question_type: int        # 1, 2, 3
    code: str
    question: str
    answer_text: str          # STT → 정제 → 클리닝 완료된 텍스트


class BatchSubmitRequest(BaseModel):
    questions: list[BatchQuestionItem]    # 전체 학생 답변 (수십~수백 개)


class BatchSubmitResponse(BaseModel):
    batch_id: str
    status: str
    total: int                # 제출한 요청 수


class SingleResult(BaseModel):
    submission_id: str
    question_type: int
    answer_text: str          # 기존 /batch-evaluate 와 필드 일치
    level: str                # excellent | good | bad | miss
    accuracy: str
    depth: str
    summary: str


class BatchResultResponse(BaseModel):
    batch_id: str
    status: str               # validating | in_progress | finalizing | completed | failed | expired | cancelled
    completed: int            # 완료 건수
    failed: int               # 실패 건수
    total: int                # 전체 건수
    results: list[SingleResult] | None    # completed 일 때만 채워짐, 그 외 None


# ──────────────────────────────────────────────────────────────
# 1단계: 제출
# ──────────────────────────────────────────────────────────────
@router.post("/batch-grade/submit", response_model=BatchSubmitResponse)
def batch_grade_submit(req: BatchSubmitRequest):
    if len(req.questions) == 0:
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    if len(req.questions) > 50000:
        raise HTTPException(
            status_code=400,
            detail="한 배치는 최대 50,000건까지 가능합니다. 나눠서 제출하세요.",
        )

    seen = set()
    lines = []
    answer_map: dict[str, str] = {}   # custom_id → answer_text 보관용

    for q in req.questions:
        cid = make_custom_id(q.submission_id, q.question_type)
        if cid in seen:
            raise HTTPException(
                status_code=400,
                detail=f"custom_id 중복: {cid} (submission_id + question_type 조합은 유일해야 합니다)",
            )
        seen.add(cid)
        answer_map[cid] = q.answer_text   # GET 때 합치기 위해 보관

        user_prompt = EVALUATION_USER.format(
            code=q.code,
            question=q.question,
            answer=q.answer_text,
        )

        lines.append(json.dumps({
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": EVALUATION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 500,
            },
        }, ensure_ascii=False))

    jsonl_bytes = ("\n".join(lines)).encode("utf-8")

    try:
        upload = client.files.create(
            file=io.BytesIO(jsonl_bytes),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "codeviva batch grading"},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"배치 제출 실패: {str(e)}")

    # 제출 성공 후 batch_id 기준으로 answer_text 보관
    _answer_store[batch.id] = answer_map

    return BatchSubmitResponse(
        batch_id=batch.id,
        status=batch.status,
        total=len(req.questions),
    )


# ──────────────────────────────────────────────────────────────
# 2단계: 결과 조회 (폴링 or 버튼 클릭 시 호출)
# ──────────────────────────────────────────────────────────────
@router.get("/batch-grade/result/{batch_id}", response_model=BatchResultResponse)
def batch_grade_result(batch_id: str):
    try:
        batch = client.batches.retrieve(batch_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"배치 조회 실패: {str(e)}")

    counts = batch.request_counts
    completed = getattr(counts, "completed", 0) if counts else 0
    failed = getattr(counts, "failed", 0) if counts else 0
    total = getattr(counts, "total", 0) if counts else 0

    # 보관해둔 answer_text 맵 (없으면 빈 dict → answer_text 는 "")
    answer_map = _answer_store.get(batch_id, {})

    def get_answer(cid: str) -> str:
        return answer_map.get(cid, "")

    if batch.status != "completed":
        return BatchResultResponse(
            batch_id=batch_id,
            status=batch.status,
            completed=completed,
            failed=failed,
            total=total,
            results=None,
        )

    results: list[SingleResult] = []

    # ── 정상 결과 파일 파싱 ────────────────────────────────
    if batch.output_file_id:
        try:
            out_text = client.files.content(batch.output_file_id).text
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"결과 파일 다운로드 실패: {str(e)}")

        for line in out_text.strip().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            cid = row.get("custom_id", "")
            sid, qtype = parse_custom_id(cid)

            try:
                content = row["response"]["body"]["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                level = parsed.get("level", "miss")
                if level not in VALID_LEVELS:
                    level = "miss"
                results.append(SingleResult(
                    submission_id=sid,
                    question_type=qtype,
                    answer_text=get_answer(cid),
                    level=level,
                    accuracy=parsed.get("accuracy", ""),
                    depth=parsed.get("depth", ""),
                    summary=parsed.get("summary", ""),
                ))
            except Exception:
                results.append(SingleResult(
                    submission_id=sid,
                    question_type=qtype,
                    answer_text=get_answer(cid),
                    level="miss",
                    accuracy="응답 형식 오류",
                    depth="",
                    summary="LLM 응답 파싱 실패로 miss 처리",
                ))

    # ── 에러 파일 파싱 (실패/만료 건) ──────────────────────
    if batch.error_file_id:
        try:
            err_text = client.files.content(batch.error_file_id).text
            for line in err_text.strip().split("\n"):
                if not line:
                    continue
                row = json.loads(line)
                cid = row.get("custom_id", "")
                sid, qtype = parse_custom_id(cid)
                err = row.get("error") or {}
                results.append(SingleResult(
                    submission_id=sid,
                    question_type=qtype,
                    answer_text=get_answer(cid),
                    level="miss",
                    accuracy=f"채점 실패: {err.get('code', 'unknown')}",
                    depth="",
                    summary=err.get("message", "배치 처리 실패로 miss"),
                ))
        except Exception:
            pass

    return BatchResultResponse(
        batch_id=batch_id,
        status="completed",
        completed=completed,
        failed=failed,
        total=total,
        results=results,
    )