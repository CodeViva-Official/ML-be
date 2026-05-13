from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    WHISPER_MODEL: str = "medium"  # faster-whisper 모델: tiny/base/small/medium/large-v3

    class Config:
        env_file = ".env"


settings = Settings()