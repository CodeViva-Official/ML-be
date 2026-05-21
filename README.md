# CodeViva ML Server

FastAPI 기반 ML 백엔드 서버. 질문 생성, STT, 이해도 채점을 담당합니다.

---

## 폴더 구조

```
ML-be/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 진입점
│   ├── config.py            # 환경변수 관리
│   ├── gemini_client.py     # LLM 호출 (OpenAI/Gemini 전환 가능)
│   ├── stt_service.py       # faster-whisper STT + Qwen 보간
│   └── routers/
│       ├── __init__.py
│       ├── questions.py     # POST /ml/generate-questions
│       ├── evaluation.py    # POST /ml/evaluate-answer (기존)
│       └── stt_and_batch.py # POST /ml/stt, /ml/batch-evaluate (신규)
├── prompts/
│   └── templates.py         # LLM 프롬프트 (ML팀 확정본)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 모델 구성

| 구분 | 모델 | 실행 방식 | 용도 |
|------|------|----------|------|
| LLM | gpt-4o-mini | OpenAI API | 질문 생성 (프롬프트 1) + 이해도 채점 (프롬프트 2) |
| STT | faster-whisper medium | 로컬 GPU | 음성 → 텍스트 변환 |
| 보간 | Qwen-7B (4bit, bitsandbytes) | 로컬 GPU | STT 텍스트에서 코드 용어 정제 |

- LLM은 `.env`의 `LLM_PROVIDER` 값으로 OpenAI ↔ Gemini 전환 가능
- 모델은 서버 시작 시 1회만 로딩, 이후 메모리에 상주

---

## 엔드포인트

### 1. POST /ml/generate-questions

코드를 받아 구술 평가용 질문 3개를 생성합니다. (프롬프트 1 사용)

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

## 서비스 전체 흐름

```
1. 코드 제출
   POST /ml/generate-questions → q1, q2, q3 반환

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
```

---

## 예외 처리

| 상황 | 처리 |
|------|------|
| LLM API 호출 실패 | 최대 3번 재시도 (1초 → 2초 → 4초) |
| JSON 파싱 실패 | 재시도 |
| level 값 이상 | excellent/good/bad/miss 외 → miss로 교정 |
| 배치 중 1개 실패 | 나머지 정상 반환, 실패 건 miss 처리 |
| 음성 파일 없음 | 400 에러 |
| STT/보간 실패 | 500 에러 |
| LLM 3번 실패 | 502 에러 |