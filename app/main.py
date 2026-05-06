from fastapi import FastAPI
from app.routers import questions, evaluation

app = FastAPI(
    title="CodeViva ML Server",
    description="질문 생성 및 음성 답변 평가 ML 백엔드",
    version="0.2.0",
)

# 라우터 등록
app.include_router(questions.router, prefix="/ml", tags=["질문 생성"])
app.include_router(evaluation.router, prefix="/ml", tags=["답변 평가"])


@app.get("/health")
def health_check():
    return {"status": "ok"}