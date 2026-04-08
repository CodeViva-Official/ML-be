import json
import re
import google.genai as genai
from app.config import settings

_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def _parse_json_response(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)


async def call_gemini(prompt: str) -> dict:
    response = _client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )
    return _parse_json_response(response.text)