"""
server.py
---------
FastAPI + WebSocket backend for the BioType Web Dashboard.

Architecture:
  - HTTP POST /enroll       → enroll the user from a batch of keystroke events
  - WebSocket  /ws/verify   → real-time stream of keystroke events → trust scores
  - HTTP GET  /             → serve the index.html dashboard
  - HTTP GET  /status       → JSON snapshot of current session state

Run:
  .\.venv\Scripts\python.exe -m uvicorn src.api.server:app --reload --port 8000
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.authenticator import BioTypeAuthenticator
from src.decision.trust_manager import TrustManager, Decision
from src.preprocessing.feature_extraction import FeatureExtractor
from src.preprocessing.interpolation import LinearInterpolator
from src.preprocessing.gafmat import GAFMATTransformer
from src.utils.event_schema import KeyEvent

# ---------------------------------------------------------------------------
# Lifespan Event Handler
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global authenticator
    weights_path = Path(__file__).resolve().parents[1] / "model" / "biotype_trained_weights.weights.h5"
    authenticator = BioTypeAuthenticator(weights_path=str(weights_path))
    print(f"[BioType API] Server ready. Model loaded.")
    yield

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(title="BioType Authentication Server", version="1.0.0", lifespan=lifespan)

# Static files (web dashboard)
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# ---------------------------------------------------------------------------
# Shared Pipeline Components (loaded once at startup)
# ---------------------------------------------------------------------------
authenticator: BioTypeAuthenticator = None
extractor    = FeatureExtractor()
interpolator = LinearInterpolator(target_length=10)
gafmat       = GAFMATTransformer(image_size=10)

WINDOW_SIZE = 20
STEP_SIZE   = 10


# ---------------------------------------------------------------------------
# Session State (per server instance — single-user demo)
# ---------------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.trust_manager = TrustManager(
            high_threshold = 0.75,
            low_threshold  = 0.45,
            ema_alpha      = 0.4,
            max_distance   = 0.80,
        )
        self.enrolled = False
        self.windows_processed = 0
        self.last_distance = 0.0

    def reset(self):
        self.__init__()

session = SessionState()


# ---------------------------------------------------------------------------
# HTTP Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/status")
async def get_status():
    return {
        "enrolled": session.enrolled,
        "trust": session.trust_manager.trust_score if session.enrolled else None,
        "decision": session.trust_manager.current_decision.value if session.enrolled else None,
        "windows_processed": session.windows_processed,
    }


class EnrollRequest(BaseModel):
    events: list  # List of {key, event_type, ts}


@app.post("/enroll")
async def enroll(req: EnrollRequest):
    """
    Accepts a batch of raw keystroke events, runs the full pipeline,
    and builds the user's biometric prototype.
    """
    session.reset()

    raw_events = [
        KeyEvent(key=e["key"], event_type=e["event_type"], ts=e["ts"])
        for e in req.events
    ]

    if len(raw_events) < WINDOW_SIZE:
        return {"success": False, "message": f"Not enough events ({len(raw_events)}). Keep typing."}

    images = _events_to_images(raw_events)
    if not images:
        return {"success": False, "message": "Could not extract GAFMAT windows. Type more."}

    ok = authenticator.enroll(images)
    session.enrolled = ok

    return {
        "success": ok,
        "message": f"Enrolled successfully using {len(images)} windows.",
        "windows": len(images),
    }


# ---------------------------------------------------------------------------
# WebSocket — Real-Time Verification Stream
# ---------------------------------------------------------------------------
@app.websocket("/ws/verify")
async def verify_stream(ws: WebSocket):
    """
    Receives keystroke event batches from the browser over WebSocket.
    Runs the GAFMAT pipeline and streams back trust scores in real time.

    Message format (browser → server):
      { "events": [{key, event_type, ts}, ...] }

    Response format (server → browser):
      { "trust": 0.94, "decision": "ACCESS_GRANTED", "distance": 0.04,
        "windows": 7, "enrolled": true }
    """
    await ws.accept()
    event_buffer: list[KeyEvent] = []
    processed_idx = 0

    try:
        while True:
            data = await ws.receive_text()
            payload = json.loads(data)

            new_events = [
                KeyEvent(key=e["key"], event_type=e["event_type"], ts=e["ts"])
                for e in payload.get("events", [])
            ]
            event_buffer.extend(new_events)

            if not session.enrolled:
                await ws.send_json({"enrolled": False, "message": "Not enrolled yet."})
                continue

            # Process new complete windows
            while processed_idx + WINDOW_SIZE <= len(event_buffer):
                window = event_buffer[processed_idx:processed_idx + WINDOW_SIZE]
                raw_feats = extractor.extract_features(window)

                if len(raw_feats) >= 2:
                    interpolated = interpolator.process(raw_feats)
                    img          = gafmat.transform(interpolated)
                    distance     = authenticator.verify(img)
                    decision     = session.trust_manager.update(distance)
                    session.windows_processed += 1
                    session.last_distance = distance

                    await ws.send_json({
                        "enrolled":  True,
                        "trust":     session.trust_manager.trust_score,
                        "decision":  decision.value,
                        "distance":  round(distance, 4),
                        "windows":   session.windows_processed,
                    })

                processed_idx += STEP_SIZE

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Error: {e}")
        await ws.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _events_to_images(events: list[KeyEvent]) -> list:
    """Sliding window → feature extraction → interpolation → GAFMAT."""
    images = []
    start = 0
    while start + WINDOW_SIZE <= len(events):
        window    = events[start:start + WINDOW_SIZE]
        raw_feats = extractor.extract_features(window)
        if len(raw_feats) >= 2:
            interpolated = interpolator.process(raw_feats)
            img          = gafmat.transform(interpolated)
            images.append(img)
        start += STEP_SIZE
    return images

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)
