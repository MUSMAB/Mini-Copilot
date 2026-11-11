"""FastAPI backend the editor talks to. Run: uvicorn mini_copilot.api.app:app --reload"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mini Copilot")


class SuggestRequest(BaseModel):
    code: str
    cursor: int
    model: str = "hybrid"


class AnalyzeRequest(BaseModel):
    code: str


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/suggest")
def suggest(req: SuggestRequest) -> dict:
    """Top 5 suggestions for the word at the cursor."""
    return {"model": req.model, "suggestions": []}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    """Misconception findings and suggested fixes for the student's code."""
    return {"findings": []}
