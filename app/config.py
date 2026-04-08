from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    WHISPER_MODEL: str = "base"

    class Config:
        env_file = ".env"


settings = Settings()