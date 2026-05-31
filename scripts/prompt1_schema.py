#!/usr/bin/env python3
"""
prompt1 출력 Pydantic 스키마 + JSON 안전 파서.

설계 목적:
  학습된 SLM의 출력이 JSON 형식이 깨지거나 스키마를 위반할 때,
  ① 정확한 위반 위치를 식별하고
  ② 재요청 가능한 형태의 오류 메시지를 만들어내며
  ③ vLLM/transformers 양쪽에서 동일하게 사용 가능.

사용:
  from prompt1_schema import Prompt1Output, validate_or_error

  data, err = validate_or_error(raw_response)
  if err:
      # err.detail 을 다음 시도의 프롬프트에 첨부하여 재요청
      ...
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Optional

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError as e:
    raise SystemExit("ERROR: pydantic 미설치. pip install 'pydantic>=2.0'") from e


# ─── 스키마 정의 ───────────────────────────────────────────────────────────

COMPLEXITY_VALUES = ("simple", "medium", "complex")


class Analysis(BaseModel):
    """코드 분석 결과."""
    purpose: str = Field(min_length=20, max_length=2000,
                         description="코드의 전체 동작·목적을 자연어로 서술 (20자 이상)")
    key_concepts: list[str] = Field(min_length=3, max_length=3,
                                    description="핵심 개념 정확히 3개")
    complexity: Literal["simple", "medium", "complex"]

    @field_validator("key_concepts")
    @classmethod
    def _each_concept_nonempty(cls, v: list[str]) -> list[str]:
        for i, c in enumerate(v):
            if not isinstance(c, str):
                raise ValueError(f"key_concepts[{i}]: 문자열이 아님")
            if len(c.strip()) < 5:
                raise ValueError(f"key_concepts[{i}]: 5자 이상 필요 (현재 {len(c.strip())}자)")
        if len(set(v)) < 3:
            raise ValueError("key_concepts: 3개 모두 서로 달라야 함")
        return v

    @field_validator("purpose")
    @classmethod
    def _purpose_not_template(cls, v: str) -> str:
        # 모델이 placeholder를 그대로 돌려주는 패턴 차단
        forbidden = ["<purpose>", "TODO", "여기에", "purpose:"]
        low = v.lower()
        if any(f.lower() in low for f in forbidden):
            raise ValueError("purpose: 템플릿 placeholder 가 포함됨")
        return v.strip()


class Questions(BaseModel):
    """구술 평가 질문 3개 (개요/트레이싱/심화)."""
    q1: str = Field(min_length=20, max_length=1000)
    q2: str = Field(min_length=20, max_length=1500)
    q3: str = Field(min_length=20, max_length=1500)

    @field_validator("q1", "q2", "q3")
    @classmethod
    def _ends_clean(cls, v: str) -> str:
        v = v.strip()
        if v.endswith((",", "(", "[", "{")):
            raise ValueError("문장이 미완성으로 잘림 (구두점 미종결)")
        return v


class Prompt1Output(BaseModel):
    """prompt1 의 최종 출력 스키마."""
    analysis: Analysis
    questions: Questions

    @classmethod
    def from_raw(cls, raw: str) -> "Prompt1Output":
        """raw 문자열 → Pydantic 인스턴스. 마크다운 코드펜스 자동 제거."""
        cleaned = _strip_markdown_fence(raw)
        data = json.loads(cleaned)
        return cls.model_validate(data)


# ─── 파싱 보조 ─────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _strip_markdown_fence(text: str) -> str:
    """모델이 ```json ... ``` 코드펜스로 감쌌을 때 제거."""
    text = text.strip()
    # 가장 바깥 한 쌍의 펜스만 제거
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    """텍스트 안에 섞인 첫 번째 { ... } 균형 잡힌 JSON 오브젝트를 추출."""
    text = _strip_markdown_fence(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ─── 검증 결과 객체 ────────────────────────────────────────────────────────

@dataclass
class ValidationFailure:
    """재요청 프롬프트에 그대로 첨부 가능한 오류 정보."""
    kind: Literal["json_decode", "schema_violation", "empty", "no_json_found"]
    detail: str          # 다음 시도 프롬프트에 그대로 붙일 한국어 오류 메시지
    raw: str             # 원본 응답

    def to_retry_message(self) -> str:
        """재요청 시 모델에 줄 user 메시지."""
        return (
            "직전 응답이 JSON 스키마를 위반했습니다. 동일한 코드에 대해 "
            "오직 유효한 JSON 객체만 출력하세요.\n\n"
            f"오류:\n{self.detail}\n\n"
            "요구 스키마:\n"
            '{\n'
            '  "analysis": {\n'
            '    "purpose": "<20자 이상 문자열>",\n'
            '    "key_concepts": ["<5자 이상>", "<5자 이상>", "<5자 이상>"],\n'
            '    "complexity": "simple | medium | complex"\n'
            '  },\n'
            '  "questions": {\n'
            '    "q1": "<20자 이상>",\n'
            '    "q2": "<20자 이상>",\n'
            '    "q3": "<20자 이상>"\n'
            '  }\n'
            "}"
        )


# ─── 통합 검증 함수 ────────────────────────────────────────────────────────

def validate_or_error(raw: str) -> tuple[Optional[Prompt1Output], Optional[ValidationFailure]]:
    """raw 응답을 검증하여 (성공값, 오류) 튜플 반환. 한쪽이 None.

    검증 순서:
      1) 비어있음 체크
      2) 마크다운 코드펜스 제거
      3) 텍스트 안에 섞인 JSON 오브젝트 추출 (모델이 앞뒤로 설명 붙였을 때)
      4) json.loads
      5) Pydantic 검증
    """
    if not raw or not raw.strip():
        return None, ValidationFailure(
            kind="empty", detail="응답이 비어 있습니다.", raw=raw,
        )

    json_text = _extract_first_json_object(raw)
    if json_text is None:
        return None, ValidationFailure(
            kind="no_json_found",
            detail="응답에서 `{` 로 시작하는 JSON 오브젝트를 찾을 수 없습니다.",
            raw=raw,
        )

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return None, ValidationFailure(
            kind="json_decode",
            detail=f"JSON 파싱 실패: {e.msg} (line {e.lineno}, col {e.colno})",
            raw=raw,
        )

    try:
        obj = Prompt1Output.model_validate(data)
    except ValidationError as e:
        # Pydantic 의 errors() 를 사람이 읽기 좋은 한국어로 정제
        msgs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            msgs.append(f"  - {loc}: {err['msg']}")
        return None, ValidationFailure(
            kind="schema_violation",
            detail="스키마 위반:\n" + "\n".join(msgs),
            raw=raw,
        )

    return obj, None


# ─── vLLM guided_json 용 JSON Schema 익스포트 ──────────────────────────────

def export_json_schema() -> dict:
    """vLLM `guided_json=...` 또는 OpenAI `response_format=json_schema` 에 사용."""
    return Prompt1Output.model_json_schema()


if __name__ == "__main__":
    # 빠른 자가 검증
    import sys

    if len(sys.argv) > 1:
        raw = open(sys.argv[1], encoding="utf-8").read()
    else:
        # 데모 케이스
        raw = json.dumps({
            "analysis": {
                "purpose": "주어진 정수 배열의 합계를 구해 반환하는 함수입니다.",
                "key_concepts": [
                    "for 반복문을 이용한 누적 합산",
                    "초기값 0을 가진 누산자 변수의 활용",
                    "return 문을 통한 결과 반환"
                ],
                "complexity": "simple",
            },
            "questions": {
                "q1": "이 코드는 무엇을 계산하는지 한 문장으로 설명해주세요.",
                "q2": "입력이 [1, 2, 3, 4] 일 때 각 반복마다 누적값이 어떻게 변하는지 단계별로 말해주세요.",
                "q3": "만약 빈 배열이 입력으로 들어온다면 결과는 어떻게 되며 그 이유는 무엇인가요?",
            },
        })

    out, err = validate_or_error(raw)
    if err:
        print(f"❌ {err.kind}: {err.detail}")
    else:
        print(f"✅ valid (complexity={out.analysis.complexity}, q1_len={len(out.questions.q1)})")
