from fastapi import FastAPI
from app.routers import questions, evaluation

app = FastAPI(
    title="CodeViva ML Server",
    description="질문 생성 및 음성 답변 평가 ML 백엔드",
    version="0.1.0",
)

# 라우터 등록 — 모든 엔드포인트에 /ml prefix 붙임
app.include_router(questions.router, prefix="/ml", tags=["질문 생성"])
app.include_router(evaluation.router, prefix="/ml", tags=["답변 평가"])


@app.get("/health")
def health_check():
    """서버 살아있는지 확인용. 백엔드에서 주기적으로 호출."""
    return {"status": "ok"}
