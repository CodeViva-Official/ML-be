"""
app/exaone_client.py
--------------------
prompt 1(질문 생성)을 로컬 EXAONE 3.5(LoRA) transformers in-process 로 돌리는 클라이언트.

ML팀 scripts/opensource_infer.py 의 load_model / generate 방식을 그대로 따름.
  - apply_chat_template(tokenize=False) 로 '문자열'만 만든 뒤 tokenizer(text) 로 토크나이즈
    (apply_chat_template 이 직접 텐서/마스크를 만들면 EXAONE 옛 modeling 의
     create_causal_mask 와 5.9.0 이 충돌 -> 이 방식으로 우회)
  - 어댑터 주입 전 get/set_input_embeddings 주입
  - 4bit 로딩 후 merge_and_unload

환경: transformers 5.9.0 / peft 0.19.1 / torch 2.11.0+cu130
  베이스: /mnt/data/sungwonnoh/model/exaone-7b-base
  어댑터: /mnt/data/sungwonnoh/model/exaone-7b-stage3

gemini_client.call_llm 과 동일 시그니처/반환형 -> 라우터는 import만 교체.
"""

import asyncio
import json
import re
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from app.gemini_client import (
    _validate_questions_response,
    _validate_evaluation_response,
)

BASE_MODEL = "/mnt/data/sungwonnoh/model/exaone-7b-base"
ADAPTER_PATH = "/mnt/data/sungwonnoh/model/exaone-7b-stage3"

MAX_NEW_TOKENS = 2048
MAX_RETRIES = 3

_tokenizer = None
_model = None


def _inject_embedding_accessors(model):
    """EXAONE 등 get_input_embeddings 미구현 모델 대응 (PEFT 주입 시 필요).
    어댑터 로드 '전에' 호출. (ML팀 load_model 과 동일)"""
    for sub in (getattr(model, "transformer", None), getattr(model, "model", None)):
        if sub is None:
            continue
        emb_attr = "wte" if hasattr(sub, "wte") else (
            "embed_tokens" if hasattr(sub, "embed_tokens") else None)
        if emb_attr is None:
            continue
        cls = type(sub)
        if "get_input_embeddings" not in cls.__dict__:
            cls.get_input_embeddings = (lambda a: lambda self: getattr(self, a))(emb_attr)
            cls.set_input_embeddings = (lambda a: lambda self, v: setattr(self, a, v))(emb_attr)


def _load():
    """베이스(bf16) -> 임베딩 주입 -> 어댑터 결합 -> 병합. 최초 1회만. (ML팀 방식 동일)"""
    global _tokenizer, _model
    if _model is None:
        print(f"[EXAONE] 베이스 로딩 중(bf16): {BASE_MODEL}")
        # GPU(20GB)가 비어있으므로 bitsandbytes 4bit 없이 bf16 로 올림(~16GB).
        # -> bitsandbytes CPU 폴백 문제 원천 차단, GPU 추론 정상 속도.
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        _inject_embedding_accessors(base)

        print(f"[EXAONE] 어댑터 얹는 중: {ADAPTER_PATH}")
        base = PeftModel.from_pretrained(base, ADAPTER_PATH)
        base = base.merge_and_unload()
        base.eval()
        _model = base
        print("[EXAONE] 로딩 완료")
    return _tokenizer, _model


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 앞뒤 군더더기 대비: 첫 { ~ 마지막 }
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def _generate_sync(system_prompt: str, user_prompt: str, sample: bool) -> str:
    tokenizer, model = _load()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # ▼ ML팀 방식: 문자열로 먼저 만든 뒤 tokenizer 로 토크나이즈
    #   (apply_chat_template 이 직접 텐서/마스크 생성하면 EXAONE 옛 modeling 과 충돌)
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)   # 일부 토크나이저가 반환하나 generate 가 거부

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    if sample:
        gen_kwargs.update(do_sample=True, temperature=0.3, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return decoded.strip()


async def call_llm(
    user_prompt: str,
    system_prompt: str = None,
    response_type: str = "questions",
) -> dict:
    """gemini_client.call_llm 과 동일 시그니처. 호출 + 파싱 + 검증 + 재시도."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = await asyncio.to_thread(
                _generate_sync, system_prompt, user_prompt, attempt > 1
            )
            result = _extract_json(raw)

            if response_type == "questions":
                result = _validate_questions_response(result)
            elif response_type == "evaluation":
                result = _validate_evaluation_response(result)

            return result

        except json.JSONDecodeError as e:
            last_error = f"JSON 파싱 실패 (시도 {attempt}/{MAX_RETRIES}): {e}"
            print(f"[EXAONE] {last_error}")
        except ValueError as e:
            last_error = f"응답 검증 실패 (시도 {attempt}/{MAX_RETRIES}): {e}"
            print(f"[EXAONE] {last_error}")
        except Exception as e:
            traceback.print_exc()
            last_error = f"추론 실패 (시도 {attempt}/{MAX_RETRIES}): {repr(e)}"
            print(f"[EXAONE] {last_error}")

    raise Exception(f"EXAONE 호출 {MAX_RETRIES}번 실패: {last_error}")


def warmup():
    _load()