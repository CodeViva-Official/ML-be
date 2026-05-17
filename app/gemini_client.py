import json
import re
from app.config import settings


def _parse_json_response(text: str) -> dict:
    """
    LLM이 가끔 ```json ... ``` 마크다운으로 감싸서 응답함.
    그걸 벗겨내고 파싱하는 함수.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)


# ── OpenAI ────────────────────────────────────────────────────
async def _call_openai(user_prompt: str, system_prompt: str = None) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=0.1,
    )

    return _parse_json_response(response.choices[0].message.content)


# ── Gemini ────────────────────────────────────────────────────
async def _call_gemini(user_prompt: str, system_prompt: str = None) -> dict:
    from google import genai

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    config = {}
    if system_prompt:
        config["system_instruction"] = system_prompt

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=user_prompt,
        config=config if config else None,
    )

    return _parse_json_response(response.text)


# ── 통합 함수 (이것만 호출하면 됨) ────────────────────────────
async def call_llm(user_prompt: str, system_prompt: str = None) -> dict:
    """
    .env의 LLM_PROVIDER 값에 따라 OpenAI 또는 Gemini를 호출.
    코드 다른 곳에서는 이 함수만 쓰면 됨.
    """
    if settings.LLM_PROVIDER == "openai":
        return await _call_openai(user_prompt, system_prompt)
    elif settings.LLM_PROVIDER == "gemini":
        return await _call_gemini(user_prompt, system_prompt)
    else:
        raise ValueError(f"지원하지 않는 LLM_PROVIDER: {settings.LLM_PROVIDER}")