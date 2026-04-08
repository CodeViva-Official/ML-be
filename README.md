# CodeViva ML Server

FastAPI 기반 ML 백엔드 서버. 질문 생성과 음성 답변 평가 두 가지만 담당합니다.

---

## 폴더 구조

```
codeviva-ml/
├── app/
│   ├── main.py            # FastAPI 앱 진입점, 라우터 등록
│   ├── config.py          # 환경변수 관리 (API 키 등)
│   ├── gemini_client.py   # Gemini API 호출 래퍼
│   ├── stt_service.py     # Whisper STT 처리
│   └── routers/
│       ├── questions.py   # POST /ml/generate-questions
│       └── evaluation.py  # POST /ml/evaluate-answer
├── prompts/
│   └── templates.py       # Gemini 프롬프트 문자열
├── .env.example           # 환경변수 템플릿
└── requirements.txt
```

---

## 설치 및 실행

```bash
# 1. 가상환경 만들기 (처음 한 번만)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경변수 설정
cp .env.example .env
# .env 파일 열어서 GEMINI_API_KEY 값 채우기

# 4. 서버 실행
uvicorn app.main:app --reload --port 8001
```

서버 실행 후 http://localhost:8001/docs 에서 Swagger UI로 바로 테스트 가능합니다.

---

## 엔드포인트 설명

### 1. POST /ml/generate-questions
코드를 받아 검증 질문 3개를 생성합니다.

**요청 (JSON body)**
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
  "questions": [
    {"id": 1, "question": "이 함수의 전체 목적은...", "intent": "설계 의도 이해"},
    {"id": 2, "question": "내부 루프가 왜 두 개인가요?", "intent": "핵심 로직 이해"},
    {"id": 3, "question": "오름차순을 내림차순으로 바꾸려면?", "intent": "변형 대응"}
  ]
}
```

---

### 2. POST /ml/evaluate-answer
음성 파일을 STT로 변환하고 채점합니다.

**요청 (multipart/form-data)**
- `submission_id`: 과제 ID (text)
- `code`: 제출 코드 (text)
- `questions_json`: 질문 리스트 JSON 직렬화 문자열 (text)
- `audio_file`: 음성 파일 (file, wav/mp4/webm 모두 가능)

**왜 JSON이 아니라 form-data?**
파일(바이너리)과 텍스트를 동시에 보낼 때 JSON body는 안 됩니다.
Content-Type이 multipart/form-data여야 파일 전송이 가능합니다.

**응답**
```json
{
  "submission_id": "sub_001",
  "transcribed_text": "버블 정렬은 인접한 두 원소를 비교해서...",
  "total_score": 72,
  "grade": "B",
  "per_question": [
    {"id": 1, "score": 80, "feedback": "전체 흐름은 이해하고 있으나..."},
    {"id": 2, "score": 65, "feedback": "루프 중첩 이유 설명이 불명확"},
    {"id": 3, "score": 70, "feedback": "수정 위치는 알지만 이유 설명 부족"}
  ],
  "overall_feedback": "기본 동작 원리는 이해하고 있으나 최적화 이유에 대한 설명이 부족합니다.",
  "suspicion_flags": []
}
```

---

## 백엔드 서버와 연동 방법

백엔드(Spring/Express/Django 등)에서 이 ML 서버를 HTTP로 호출하면 됩니다.

```
[프론트] ──코드 제출──▶ [백엔드] ──POST /ml/generate-questions──▶ [ML서버] ──▶ Gemini
                                  ◀──────질문 3개──────────────────
         ◀──질문 전달──

[프론트] ──음성 업로드──▶ [백엔드] ──S3 저장──▶ S3
                                   ──POST /ml/evaluate-answer──▶ [ML서버] ──Whisper──▶ 텍스트
                                                                           ──Gemini──▶ 채점 결과
                          ◀──────────결과 저장 + 응답──────────────
         ◀──결과 전달──
```

---

## Whisper 모델 크기 선택 가이드

| 모델 | 파일 크기 | 속도 | 정확도 | 권장 환경 |
|------|----------|------|--------|----------|
| tiny | 39MB | 가장 빠름 | 낮음 | 테스트용 |
| base | 74MB | 빠름 | 보통 | MVP 개발 |
| small | 244MB | 보통 | 좋음 | 스테이징 |
| medium | 769MB | 느림 | 높음 | 프로덕션 권장 |

.env에서 `WHISPER_MODEL=base`로 시작하고, 정확도 부족하면 올리면 됩니다.
