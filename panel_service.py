"""HTTP front for panel_critic.py so Alice's pipeline can call the panel as its Critic.

Run:  .venv1/bin/uvicorn panel_service:app --port 8765
POST /critique  {"variants": [{"content": "..."}], "channel": "linkedin", "format": "linkedin_post", "profile": {...}|null}
     -> {"run_id", "model", "critiques": [{"scores": {...}, "notes": {...}, "panel": {...}}], "tokens": 0}
"""
from fastapi import FastAPI
from pydantic import BaseModel

import panel_critic

app = FastAPI(title="judgement-panel critic")


class VariantIn(BaseModel):
    content: str


class CritiqueIn(BaseModel):
    variants: list[VariantIn]
    channel: str
    format: str
    profile: dict | None = None


@app.get("/health")
async def health():
    return {"ok": True, "model": panel_critic.MODEL, "exchange": panel_critic.EXCHANGE}


@app.post("/critique")
async def critique(body: CritiqueIn):
    out = await panel_critic.critique([v.content for v in body.variants], body.channel, body.format, body.profile)
    out["tokens"] = 0  # local model, no metered cost; Alice sums this into tokenCost
    return out
