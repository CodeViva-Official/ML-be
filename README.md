# CodeViva ML Server

FastAPI 기반 ML 백엔드 서버. 질문 생성, STT, 이해도 채점을 담당합니다.

---

## 폴더 구조

```
ML-be/
├── app/
│   ├── main.py              # FastAPI 앱 진입점 (+ GET /health)
│   ├── config.py            # 환경변수 관리 (pydantic-settings)
│   ├── exaone_client.py     # 프롬프트 1: EXAONE 3.5 + LoRA 로컬 추론
│   ├── gemini_client.py     # 프롬프트 2: OpenAI/Gemini API 호출 (LLM_PROVIDER로 전환)
│   ├── stt_service.py       # faster-whisper STT + Qwen 보간
│   └── routers/
│       ├── questions.py     # POST /ml/generate-questions (EXAONE)
│       ├── evaluation.py    # POST /ml/evaluate-answer (기존)
│       ├── stt_and_batch.py # POST /ml/stt, /ml/batch-evaluate
│       └── batch_api.py     # POST /ml/batch-grade/submit, GET /ml/batch-grade/result/{id}
├── prompts/
│   └── templates.py         # LLM 프롬프트 (ML팀 확정본)
├── scripts/
│   └── prompt1_schema.py    # 프롬프트 1 출력 Pydantic 스키마 + JSON 파서/재요청 메시지
├── requirements.txt         # 최소 의존성 (버전 미고정)
├── requirements_working.txt # GPU 서버에서 검증된 고정 버전 (pip freeze)
└── README.md
```

---

## 모델 구성

| 구분 | 모델 | 실행 방식 | 용도 |
|------|------|----------|------|
| 질문 생성 | EXAONE-3.5-7.8B-Instruct + LoRA(stage3) | 로컬 GPU, transformers in-process | 프롬프트 1 |
| 채점 | `OPENAI_MODEL` (기본 `gpt-5.5`) | OpenAI API (실시간 / Batch API) | 프롬프트 2 |
| STT | faster-whisper `medium` (fp16) | 로컬 GPU | 음성 → 텍스트 |
| 보간 | Qwen2.5-7B-Instruct (4bit, bitsandbytes) | 로컬 GPU | STT 코드 용어 정제 (`/ml/evaluate-answer`에서만 사용) |

- 채점 LLM은 `.env`의 `LLM_PROVIDER`로 OpenAI ↔ Gemini 전환 가능 (`GEMINI_MODEL` 기본값 `gemini-2.5-flash`)
- 로컬 모델은 **첫 요청 시 1회 로딩**되고, 이후 프로세스 메모리에 상주합니다. 그래서 서버를 재시작한 뒤 첫 요청은 느립니다.

---

## 모델 서빙 (프롬프트 1 · EXAONE)

ML팀이 학습한 LoRA 어댑터를 EXAONE 베이스에 병합하여 FastAPI 프로세스 안에서 직접 추론합니다. 별도 추론 서버(vLLM 등)는 쓰지 않습니다.

### 모델 경로

| 구분 | 경로 |
|------|------|
| 베이스 | `/mnt/data/sungwonnoh/model/exaone-7b-base` (EXAONE-3.5-7.8B-Instruct) |
| 어댑터 | `/mnt/data/sungwonnoh/model/exaone-7b-stage3` |

모델 파일은 `.gitignore` 대상이라 레포에 포함되지 않습니다.

### 로딩 과정 (`app/exaone_client.py::_load`)

1. 베이스를 **bf16**으로 로딩 (`device_map="auto"`, `trust_remote_code=True`)
2. `get_input_embeddings` / `set_input_embeddings`를 `.wte`에 주입
   - EXAONE 구현체에 이 메서드가 없어서 PEFT 어댑터를 붙일 때 필요합니다.
3. `PeftModel.from_pretrained`로 어댑터를 결합한 뒤 `merge_and_unload()`로 병합 → `eval()`

### 추론 과정

1. `apply_chat_template(tokenize=False)`로 **문자열만** 생성한 뒤 `tokenizer(text)`로 따로 토크나이즈
   - 템플릿이 텐서와 마스크를 직접 만들게 하면 EXAONE modeling 코드와 transformers 5.x의 `create_causal_mask`가 충돌합니다 (ML팀 추론 방식과 동일하게 유지).
2. `token_type_ids` 제거 후 `generate` (`max_new_tokens=2048`)
3. 블로킹 추론은 `asyncio.to_thread`로 실행해서 이벤트 루프를 막지 않습니다.

### 출력 검증 및 재시도

| 단계 | 처리 |
|------|------|
| 1차 시도 | greedy decoding |
| 2·3차 시도 | sampling (`temperature=0.3`, `top_p=0.9`) |
| 이중 인코딩 | 출력이 `"{...}"` 형태의 JSON 문자열이면 한 번 벗겨냄 (`_unwrap_json_string`) |
| 스키마 검증 | `scripts/prompt1_schema.py`의 Pydantic 스키마로 검증 (`validate_or_error`) |
| 재요청 | 검증 실패 시 위반 내용을 프롬프트 앞에 붙여 원본 코드와 함께 다시 요청 |
| 3회 실패 | `/ml/generate-questions`가 502 반환 |

> lm-format-enforcer(grammar 강제)는 transformers 5.x와 호환되지 않아 설치하지 않습니다. 코드에는 해당 분기가 남아 있지만, 패키지가 없으면 자유 디코딩으로 동작합니다.

### 실행 환경

| 항목 | 값 |
|------|------|
| GPU | NVIDIA RTX 4000 Ada (20GB), 드라이버 CUDA 12.7 |
| torch | `2.6.0+cu124` — **cu130 빌드 사용 불가** (드라이버가 CUDA 12.7까지 지원) |
| transformers / peft | `5.9.0` / `0.19.1` |
| accelerate / bitsandbytes | `1.13.0` / `0.49.2` |

전체 고정 버전은 `requirements_working.txt`를 참고하세요.

---

## 실행 방법

```bash
# 1. 의존성 설치 (torch는 반드시 cu124 — PyPI에 없으므로 PyTorch 인덱스 추가)
pip install -r requirements_working.txt --extra-index-url https://download.pytorch.org/whl/cu124

# 2. 환경변수 (.env)
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5
GEMINI_API_KEY=...
WHISPER_MODEL=medium

# 3. 서버 실행 (venv 절대경로 사용)
nohup /mnt/data/sungwonnoh/ML-be/venv/bin/python -m uvicorn app.main:app \
  --host 0.0.0.0 --port 8001 > server.log 2>&1 &

# 4. 헬스체크
curl localhost:8001/health
```

- 외부 접근: AWS EC2(퍼블릭 게이트웨이) ↔ Tailscale(`--tun=userspace-networking`) ↔ GPU 서버 8001 포트
- 파드 재시작 시 `/mnt/data` 밖은 초기화되므로 venv, 모델, Tailscale state는 모두 `/mnt/data`에 둡니다.

---

## 엔드포인트

### 0. GET /health

서버 상태 확인. `{"status": "ok"}` 반환.

---

### 1. POST /ml/generate-questions

코드를 받아 구술 평가용 질문 3개를 생성합니다. (프롬프트 1, **로컬 EXAONE**)

**요청 (JSON)**
```json
{
  "submission_id": "sub_001",
  "code": "def bubble_sort(arr): ..."
}
```

**응답**
```json
{
  "submission_id": "sub_001",
  "analysis": {
    "purpose": "버블 정렬 알고리즘으로 배열을 오름차순 정렬",
    "key_concepts": ["중첩 반복문", "인접 요소 비교", "스왑 연산"],
    "complexity": "simple"
  },
  "questions": {
    "q1": "이 함수는 전체적으로 무엇을 하나요?",
    "q2": "arr = [5, 3, 1]일 때 첫 번째 패스에서 어떤 일이 일어나나요?",
    "q3": "이 정렬을 다른 방식으로 구현한다면 어떻게 하시겠어요?"
  }
}
```

`complexity` 값: `simple` | `medium` | `complex`

---

### 2. POST /ml/stt

음성 → STT → 불필요 토큰 제거 → 텍스트 반환. 채점 없음.

**요청 (multipart/form-data)**
- `audio_file`: 음성 파일 (wav/mp4/webm)

**응답**
```json
{
  "raw_stt": "위스퍼 원본 텍스트",
  "cleaned_text": "불필요 토큰 제거 후 최종 텍스트"
}
```

---

### 3. POST /ml/batch-evaluate

텍스트 답변 3개를 한번에 받아서 프롬프트 2로 동시 채점 (비동기 병렬). 프론트에서 "결과 보기" 클릭 시 호출.

**요청 (JSON)**
```json
{
  "questions": [
    {
      "submission_id": "sub_001",
      "question_type": 1,
      "code": "코드 전문",
      "question": "Q1 질문 텍스트",
      "answer_text": "정제된 답변 텍스트"
    },
    {
      "submission_id": "sub_001",
      "question_type": 2,
      "code": "코드 전문",
      "question": "Q2 질문 텍스트",
      "answer_text": "정제된 답변 텍스트"
    },
    {
      "submission_id": "sub_001",
      "question_type": 3,
      "code": "코드 전문",
      "question": "Q3 질문 텍스트",
      "answer_text": "정제된 답변 텍스트"
    }
  ]
}
```

**응답**
```json
{
  "results": [
    {
      "submission_id": "sub_001",
      "question_type": 1,
      "answer_text": "답변 텍스트",
      "level": "good",
      "accuracy": "정확성 근거",
      "depth": "깊이 분석",
      "summary": "판정 사유"
    }
  ]
}
```

level 값: `excellent` | `good` | `bad` | `miss`

---

### 4. POST /ml/evaluate-answer (기존 유지)

음성 + 코드 + 질문 → STT → Qwen 보간 → 채점을 한번에 처리.

**요청 (multipart/form-data)**
- `submission_id`: 과제 ID (text)
- `question_type`: 1, 2, 3 중 하나 (integer)
- `code`: 제출 코드 (text)
- `question`: 해당 질문 텍스트 (text)
- `audio_file`: 음성 파일 (file)

**응답**
```json
{
  "submission_id": "sub_001",
  "question_type": 1,
  "raw_stt": "위스퍼 원본 텍스트",
  "refined_text": "Qwen 정제 후 텍스트",
  "level": "good",
  "accuracy": "핵심 개념은 맞으나 종료 조건 설명 누락",
  "depth": "WQ1, WQ2 충족. WQ3 미충족 (코드 토큰 미인용)",
  "summary": "what 정확 + why 부분 충족으로 good 판정"
}
```

---

### 5. POST /ml/batch-grade/submit · GET /ml/batch-grade/result/{batch_id}

OpenAI **Batch API**로 채점합니다 (토큰 비용 50% 절감, 완료까지 최대 24시간). 기존 `/batch-evaluate`와 결과 필드가 같습니다.

**제출 요청**: `/ml/batch-evaluate`와 같은 `questions` 배열 (최대 50,000건)
- `submission_id + question_type` 조합은 유일해야 합니다 (`custom_id = "{submission_id}__q{question_type}"`).

**제출 응답**
```json
{ "batch_id": "batch_abc123", "status": "validating", "total": 120 }
```

**결과 조회 응답**
```json
{
  "batch_id": "batch_abc123",
  "status": "completed",
  "completed": 118,
  "failed": 2,
  "total": 120,
  "results": [
    {
      "submission_id": "sub_001",
      "question_type": 1,
      "answer_text": "답변 텍스트",
      "level": "good",
      "accuracy": "정확성 근거",
      "depth": "깊이 분석",
      "summary": "판정 사유"
    }
  ]
}
```

- `status` 값: `validating` | `in_progress` | `finalizing` | `completed` | `failed` | `expired` | `cancelled`
- `status`가 `completed`가 아니면 `results`는 `null`입니다.
- 실패하거나 파싱 오류가 난 건은 `level: "miss"`로 반환됩니다.
- ⚠️ `answer_text`는 서버 메모리에 보관하므로, 제출 후 서버가 재시작되면 빈 문자열로 반환됩니다.

---

## 서비스 전체 흐름

```
1. 코드 제출
   POST /ml/generate-questions → (EXAONE) q1, q2, q3 반환

2. Q1 음성 답변
   POST /ml/stt (음성) → cleaned_text 반환
   프론트에서 텍스트 표시

3. Q2 음성 답변
   POST /ml/stt → cleaned_text 반환

4. Q3 음성 답변
   POST /ml/stt → cleaned_text 반환

5. "결과 보기" 클릭
   POST /ml/batch-evaluate (질문 3개 + 텍스트 답변 3개)
   → 3개를 비동기 병렬로 동시 채점
   → 채점 결과 3개 한번에 반환

(대량 채점 시) 5-1. POST /ml/batch-grade/submit → batch_id
                5-2. GET  /ml/batch-grade/result/{batch_id} 폴링 → completed 시 결과 반환
```

---

## 예외 처리

| 상황 | 처리 |
|------|------|
| EXAONE 출력 검증 실패 | 오류 내용을 붙여 즉시 재요청, 최대 3회 |
| OpenAI/Gemini 호출·파싱 실패 | 최대 3회 시도 (사이 대기 1초 → 2초) |
| level 값 이상 | excellent/good/bad/miss 외 → miss로 교정 |
| 배치 중 1개 실패 | 나머지 정상 반환, 실패 건 miss 처리 |
| 빈 코드 / 음성 파일 없음 | 400 에러 |
| STT/보간 실패 | 500 에러 |
| LLM 3회 실패 | 502 에러 (`/batch-evaluate`는 502 대신 해당 건 miss 처리) |
