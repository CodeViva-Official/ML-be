import tempfile
import os
import whisper
from app.config import settings

# 모델 로딩은 서버 시작 시 한 번만. 매 요청마다 로드하면 너무 느림.
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"[Whisper] 모델 로딩 중: {settings.WHISPER_MODEL}")
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL)
        print("[Whisper] 모델 로딩 완료")
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """
    음성 바이트를 받아서 텍스트로 변환.
    Whisper는 파일 경로를 받아야 해서 임시 파일 생성 후 삭제.
    """
    model = get_whisper_model()

    # 확장자 보존 (wav, mp4, webm 등 Whisper가 ffmpeg으로 다 처리함)
    suffix = os.path.splitext(filename)[-1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="ko")
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)  # 임시 파일 반드시 삭제
