import tempfile
import os
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from app.config import settings

# ── faster-whisper 모델 (서버 시작 시 1회 로딩) ────────────────
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"[Whisper] faster-whisper 모델 로딩 중: {settings.WHISPER_MODEL}")
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device="cuda",
            compute_type="float16",
        )
        print("[Whisper] 모델 로딩 완료")
    return _whisper_model


# ── Qwen 보간 모델 (서버 시작 시 1회 로딩) ─────────────────────
_qwen_tokenizer = None
_qwen_model = None
QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 7B + bitsandbytes 4bit, 약 5GB VRAM


def get_qwen_model():
    global _qwen_tokenizer, _qwen_model
    if _qwen_model is None:
        print(f"[Qwen] 보간 모델 로딩 중: {QWEN_MODEL_NAME}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME,
            trust_remote_code=True,
        )
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("[Qwen] 모델 로딩 완료")
    return _qwen_tokenizer, _qwen_model


# ── STT: 음성 → 텍스트 ────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """
    faster-whisper로 음성을 텍스트로 변환.
    """
    model = get_whisper_model()

    suffix = os.path.splitext(filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, language="ko", beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        return text
    finally:
        os.unlink(tmp_path)


# ── 보간: STT 텍스트 정제 ──────────────────────────────────────
def refine_stt_text(raw_stt_text: str, submitted_code: str) -> str:
    """
    Whisper STT 결과에서 코드 용어가 깨진 부분을 Qwen으로 정제.
    """
    tokenizer, model = get_qwen_model()

    prompt = f"""당신은 프로그래밍 교육용 STT 후처리 전문가입니다.

아래는 학생이 코드에 대해 구술 설명한 것을 STT로 변환한 텍스트입니다.
STT 특성상 코드 용어(변수명, 함수명, 연산자 등)가 한국어 발음으로 잘못 변환됐을 수 있습니다.

## 제출된 코드
```
{submitted_code}
```

## STT 원본 텍스트
{raw_stt_text}

## 규칙
1. 코드에 실제 등장하는 변수명, 함수명, 연산자를 참고하여 잘못 변환된 용어만 교정하세요.
2. 학생이 말한 내용의 의미나 문장 구조는 절대 바꾸지 마세요.
3. 코드에 없는 용어는 건드리지 마세요.
4. 정제된 텍스트만 출력하세요. 설명이나 마크다운은 금지합니다.

## 정제된 텍스트"""

    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    refined_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return refined_text


# ── 통합 함수: STT + 보간 ──────────────────────────────────────
def process_audio(audio_bytes: bytes, filename: str, submitted_code: str) -> dict:
    """
    음성 → STT → 보간까지 한번에 처리.
    
    반환값:
    {
        "raw_stt": "위스퍼가 변환한 원본 텍스트",
        "refined": "Qwen이 정제한 텍스트"
    }
    """
    raw_stt = transcribe_audio(audio_bytes, filename)
    refined = refine_stt_text(raw_stt, submitted_code)

    return {
        "raw_stt": raw_stt,
        "refined": refined,
    }