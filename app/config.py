from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM 설정 — LLM_PROVIDER로 전환 가능
    LLM_PROVIDER: str = "openai"  # "openai" 또는 "gemini"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.5"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # STT
    WHISPER_MODEL: str = "medium"  # faster-whisper 모델: tiny/base/small/medium/large-v3

    class Config:
        env_file = ".env"


settings = Settings()