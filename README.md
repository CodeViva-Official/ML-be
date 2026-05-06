# CodeViva ML Server

FastAPI 기반 ML 백엔드 서버. 질문 생성과 음성 답변 평가를 담당합니다.

---

## 폴더 구조

```
codeviva-ml/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI 앱 진입점
│   ├── config.py          # 환경변수 관리
│   ├── gemini_client.py   # Gemini API 호출 (system prompt 지원)
│   ├── stt_service.py     # Whisper STT 처리
│   └── routers/
│       ├── __init__.py
│       ├── questions.py   # POST /ml/generate-questions
│       └── evaluation.py  # POST /ml/evaluate-answer, /ml/evaluate-final
├── prompts/
│   └── templates.py       # Gemini 프롬프트 (ML팀 확정본)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 설치 및 실행

```bash
# 1. 가상환경
python3 -m venv venv
source venv/bin/activate

# 2. 패키지 설치
pip3 install "fastapi[standard]"
pip3 install openai-whisper google-genai python-dotenv aiofiles pydantic-settings

# 3. ffmpeg 설치 (Whisper가 음성 변환에 사용)
# Mac
brew install ffmpeg
# Ubuntu
sudo apt install ffmpeg

# 4. 환경변수 설정
cp .env.example .env
# .env 파일 열어서 GEMINI_API_KEY 값 채우기

# 5. 서버 실행
fastapi dev app/main.py
```

서버 실행 후 http://localhost:8000/docs 에서 Swagger UI로 테스트 가능합니다.

---

## 엔드포인트

### 1. POST /ml/generate-questions

코드를 받아 검증 질문 3개를 생성합니다. (프롬프트 1 사용)

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

### 2. POST /ml/evaluate-answer (3번 호출)

질문 1개 + 음성 답변 1개를 받아 개별 채점합니다. (프롬프트 2 사용)

Q1, Q2, Q3 각각에 대해 한 번씩 총 3번 호출합니다.

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
  "transcribed_text": "버블 정렬은 인접한 두 원소를 비교해서...",
  "level": "good",
  "accuracy": "핵심 개념은 맞으나 종료 조건 설명 누락",
  "depth": "WQ1, WQ2 충족. WQ3 미충족 (코드 토큰 미인용)",
  "summary": "what 정확 + why 부분 충족으로 good 판정"
}
```

level 값: `excellent` | `good` | `bad` | `miss`

---

### 3. POST /ml/evaluate-final

3개의 개별 평가 등급을 받아 종합 등급을 산출합니다.
LLM 호출 없이 코드 로직으로 처리합니다.

⚠️ 종합 기준은 임시입니다. 확정 후 `_calculate_final_grade()` 함수만 수정하면 됩니다.

**요청 (JSON body)**
```json
{
  "submission_id": "sub_001",
  "q1_level": "excellent",
  "q2_level": "good",
  "q3_level": "good"
}
```

**응답**
```json
{
  "submission_id": "sub_001",
  "final_grade": "good",
  "q1_level": "excellent",
  "q2_level": "good",
  "q3_level": "good"
}
```

---

## 서비스 전체 흐름

```
1. 백엔드 → POST /ml/generate-questions (코드 전달)
   ← q1, q2, q3 + 코드 분석 결과 반환

2. 프론트에서 q1 보여줌 → 학생 음성 답변
   백엔드 → POST /ml/evaluate-answer (type=1, q1, 음성)
   ← level: excellent/good/bad/miss

3. 프론트에서 q2 보여줌 → 학생 음성 답변
   백엔드 → POST /ml/evaluate-answer (type=2, q2, 음성)
   ← level

4. 프론트에서 q3 보여줌 → 학생 음성 답변
   백엔드 → POST /ml/evaluate-answer (type=3, q3, 음성)
   ← level

5. 백엔드 → POST /ml/evaluate-final (3개 level 전달)
   ← final_grade
```

---

## Whisper 모델 선택

.env 파일에서 `WHISPER_MODEL` 값을 변경합니다.

| 모델 | 크기 | 정확도 | 속도 | 권장 |
|------|------|--------|------|------|
| tiny | 39MB | 낮음 | 가장 빠름 | 테스트용 |
| base | 74MB | 보통 | 빠름 | - |
| small | 244MB | 좋음 | 보통 | - |
| medium | 769MB | 높음 | 느림 | - |
| turbo | 809MB | 최고급 | 빠름 | ✅ 프로덕션 권장 |
| large-v3 | 1.5GB | 최고 | 가장 느림 | 최고 정확도 필요 시 |

개발 시에는 `base`, 실제 서비스에서는 `turbo` 권장합니다.

```
WHISPER_MODEL=turbo
```

---

## 미구현 사항

- STT 보간 처리 (Whisper 결과를 오픈소스 LLM으로 정제)
- 종합 등급 기준 확정 (`_calculate_final_grade()` 함수 수정 필요)