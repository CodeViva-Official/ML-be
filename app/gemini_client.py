import json
import re
import asyncio
from app.config import settings

# 최대 재시도 횟수
MAX_RETRIES = 3

# 유효한 level 값
VALID_LEVELS = {"excellent", "good", "bad", "miss"}


# ── 클라이언트 모듈 레벨 1회 생성 (lazy) ──────────────────────
# 매 호출마다 새로 만들지 않도록 캐싱한다.
# import 시점이 아니라 첫 호출 시점에 만들어서, 모듈 로드 때
# OPENAI_API_KEY 가 없어도 죽지 않게 한다(batch_api 에서 겪었던 문제 회피).
_openai_client = None
_gemini_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def _parse_json_response(text: str) -> dict:
    """
    LLM이 가끔 ```json ... ``` 마크다운으로 감싸서 응답함.
    그걸 벗겨내고 파싱하는 함수.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)


def _validate_evaluation_response(result: dict) -> dict:
    """
    프롬프트 2 응답의 level 필드 검증.
    이상한 값이 오면 기본값으로 교정.
    """
    if "level" in result:
        level = result["level"].lower().strip()
        if level not in VALID_LEVELS:
            result["level"] = "miss"
            result["summary"] = result.get("summary", "") + f" (원래 응답: {level}, miss로 교정됨)"
        else:
            result["level"] = level

    # 필수 필드가 없으면 기본값
    result.setdefault("level", "miss")
    result.setdefault("accuracy", "응답 형식 오류")
    result.setdefault("depth", "응답 형식 오류")
    result.setdefault("summary", "응답 형식 오류로 miss 처리")

    return result


def _validate_questions_response(result: dict) -> dict:
    """
    프롬프트 1 응답의 필수 필드 검증.
    """
    if "questions" not in result:
        raise ValueError("응답에 questions 필드가 없습니다")
    if "analysis" not in result:
        raise ValueError("응답에 analysis 필드가 없습니다")

    q = result["questions"]
    for key in ["q1", "q2", "q3"]:
        if key not in q or not q[key].strip():
            raise ValueError(f"응답에 {key} 질문이 비어있습니다")

    return result


# ── OpenAI ────────────────────────────────────────────────────
async def _call_openai(user_prompt: str, system_prompt: str = None) -> str:
    client = _get_openai_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # [핵심 수정] 동기 blocking 호출을 별도 스레드로 빼서
    # 이벤트 루프를 막지 않게 한다. 이래야 asyncio.gather 의
    # 동시 호출이 실제로 동시에 돌아간다(기존엔 순차 처리됨).
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=settings.OPENAI_MODEL,
        messages=messages,
    )

    return response.choices[0].message.content


# ── Gemini ────────────────────────────────────────────────────
async def _call_gemini(user_prompt: str, system_prompt: str = None) -> str:
    client = _get_gemini_client()

    config = {}
    if system_prompt:
        config["system_instruction"] = system_prompt

    # [핵심 수정] Gemini SDK 의 generate_content 도 동기 blocking 이므로
    # 동일하게 스레드로 빼서 이벤트 루프를 막지 않게 한다.
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=settings.GEMINI_MODEL,
        contents=user_prompt,
        config=config if config else None,
    )

    return response.text


# ── 통합 함수 (재시도 + 검증 포함) ─────────────────────────────
async def call_llm(
    user_prompt: str,
    system_prompt: str = None,
    response_type: str = "evaluation",  # "evaluation" 또는 "questions"
) -> dict:
    """
    LLM 호출 + JSON 파싱 + 검증.
    실패 시 최대 3번 재시도.

    response_type:
      "evaluation" → level 필드 검증
      "questions"  → questions 필드 검증
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 1. LLM 호출
            if settings.LLM_PROVIDER == "openai":
                raw_text = await _call_openai(user_prompt, system_prompt)
            elif settings.LLM_PROVIDER == "gemini":
                raw_text = await _call_gemini(user_prompt, system_prompt)
            else:
                raise ValueError(f"지원하지 않는 LLM_PROVIDER: {settings.LLM_PROVIDER}")

            # 2. JSON 파싱
            result = _parse_json_response(raw_text)

            # 3. 응답 검증
            if response_type == "evaluation":
                result = _validate_evaluation_response(result)
            elif response_type == "questions":
                result = _validate_questions_response(result)

            return result

        except json.JSONDecodeError as e:
            last_error = f"JSON 파싱 실패 (시도 {attempt}/{MAX_RETRIES}): {str(e)}"
            print(f"[LLM] {last_error}")
        except ValueError as e:
            last_error = f"응답 검증 실패 (시도 {attempt}/{MAX_RETRIES}): {str(e)}"
            print(f"[LLM] {last_error}")
        except Exception as e:
            last_error = f"API 호출 실패 (시도 {attempt}/{MAX_RETRIES}): {str(e)}"
            print(f"[LLM] {last_error}")

        # 재시도 전 대기 (1초, 2초, 4초)
        if attempt < MAX_RETRIES:
            await asyncio.sleep(2 ** (attempt - 1))

    raise Exception(f"LLM 호출 {MAX_RETRIES}번 실패: {last_error}")