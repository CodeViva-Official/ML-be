import json
import re
from google import genai
from app.config import settings

_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def _parse_json_response(text: str) -> dict:
    """
    Gemini가 가끔 ```json ... ``` 마크다운으로 감싸서 응답함.
    그걸 벗겨내고 파싱하는 함수.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)


async def call_gemini(user_prompt: str, system_prompt: str = None) -> dict:
    """
    Gemini에 프롬프트 보내고 JSON 응답 반환.
    system_prompt가 있으면 system instruction으로 전달.
    """
    config = {}
    if system_prompt:
        config["system_instruction"] = system_prompt

    response = _client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_prompt,
        config=config if config else None,
    )
    return _parse_json_response(response.text)