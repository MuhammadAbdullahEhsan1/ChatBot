import os
import time
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from dotenv import load_dotenv

from google import genai
from google.genai import types

from app.db import (
    create_session, save_message, get_history, set_title,
    get_sessions, delete_session, get_message_count,
    save_summary, get_latest_summary, get_messages_after
)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Abdullah AI Chat", version="2.0.0")
load_dotenv()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a helpful, smart, and friendly AI assistant named Abdullah AI. "
    "Give clear, concise, and well-formatted answers. "
    "Use markdown formatting (code blocks, bold, lists) when it improves readability. "
    "Be direct and avoid unnecessary filler text."
)

SUMMARY_PROMPT = (
    "Summarize the following conversation in 3-5 concise sentences, "
    "preserving the key facts, decisions, and context that would be needed "
    "to continue the conversation intelligently. Be brief and factual:\n\n"
)

TITLE_PROMPT = (
    "Generate a short (3-6 words) descriptive title for a chat that starts with this message. "
    "Return ONLY the title, no quotes, no punctuation at end:\n\n"
)

# ── Rate Limiting ─────────────────────────────────────────────────────────────

# { ip: [timestamp, ...] }
_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 20       # max requests
RATE_WINDOW = 60      # per N seconds
BURST_LIMIT = 5       # max in 5 seconds (anti-burst)
BURST_WINDOW = 5


def check_rate_limit(ip: str):
    now = time.time()
    # Clean old entries
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < RATE_WINDOW]

    if len(_rate_store[ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT} requests per minute."
        )

    # Burst check
    recent = [t for t in _rate_store[ip] if now - t < BURST_WINDOW]
    if len(recent) >= BURST_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many requests too fast. Please slow down."
        )

    _rate_store[ip].append(now)


# ── Token / Context Management ────────────────────────────────────────────────

SUMMARIZE_AFTER = 20      # messages before summarizing older context
RECENT_KEEP = 10          # always keep last N messages verbatim


def build_context(session_id: str) -> list[types.Content]:
    """
    Smart context builder:
    1. If we have a summary, use it + messages after summary
    2. If message count > threshold, trigger summarization
    3. Otherwise use full history
    """
    total = get_message_count(session_id)
    history = get_history(session_id, limit=SUMMARIZE_AFTER + 5)

    if not history:
        return []

    contents = []

    # Check if we have a summary
    summary_row = get_latest_summary(session_id)

    if summary_row:
        # Add summary as a synthetic exchange
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text="[Previous conversation summary for context]")]
        ))
        contents.append(types.Content(
            role="model",
            parts=[types.Part(text=summary_row["summary"])]
        ))
        # Get messages after the summary
        recent = get_messages_after(session_id, summary_row["up_to_id"])
    else:
        recent = history

    # Add recent messages
    for msg in recent[-SUMMARIZE_AFTER:]:
        role = msg["role"] if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["message"])]
        ))

    return contents


async def maybe_summarize(session_id: str):
    """Background task: summarize old messages if threshold exceeded."""
    total = get_message_count(session_id)
    summary_row = get_latest_summary(session_id)
    covered = summary_row["up_to_id"] if summary_row else 0

    # Count messages not yet summarized
    unsummarized = get_messages_after(session_id, covered)
    if len(unsummarized) < SUMMARIZE_AFTER:
        return

    # Build text to summarize (all except last RECENT_KEEP)
    to_summarize = unsummarized[:-RECENT_KEEP]
    if len(to_summarize) < 8:
        return

    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['message']}" for m in to_summarize
    )

    try:
        resp = client.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(max_output_tokens=300),
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=SUMMARY_PROMPT + convo_text)]
            )]
        )
        summary_text = resp.text.strip()
        last_id = to_summarize[-1]["id"]
        save_summary(session_id, summary_text, last_id)
    except Exception:
        pass  # Non-critical, skip silently


async def generate_title(session_id: str, first_message: str):
    """Generate a chat title from the first message."""
    try:
        resp = client.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(max_output_tokens=20),
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=TITLE_PROMPT + first_message[:300])]
            )]
        )
        title = resp.text.strip().strip('"').strip("'")
        if title:
            set_title(session_id, title)
    except Exception:
        # Fall back to truncated message
        set_title(session_id, first_message[:50])


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

    @validator("message")
    def sanitize_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        if len(v) > 8000:
            raise ValueError("Message too long (max 8000 chars)")
        return v

    @validator("session_id")
    def sanitize_session(cls, v):
        # Only allow alphanumeric, underscore, hyphen
        import re
        if not re.match(r'^[\w\-]{1,64}$', v):
            raise ValueError("Invalid session_id")
        return v


class NewChatRequest(BaseModel):
    session_id: str = "default"


class DeleteRequest(BaseModel):
    session_id: str


# ── Static ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Chat (Streaming) ──────────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    check_rate_limit(request.client.host)

    sid = body.session_id

    # Ensure session exists
    create_session(sid)

    # Check if this is the first message (for title generation)
    is_first = get_message_count(sid) == 0

    # Save user message
    save_message(sid, "user", body.message)

    # Build context
    context = build_context(sid)

    # Ensure last item is the current user message
    context.append(types.Content(
        role="user",
        parts=[types.Part(text=body.message)]
    ))

    async def generate():
        full_reply = []
        try:
            # Use streaming
            response_stream = client.models.generate_content_stream(
                model=MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=4096,
                ),
                contents=context
            )

            for chunk in response_stream:
                if chunk.text:
                    full_reply.append(chunk.text)
                    # SSE format
                    yield f"data: {json.dumps({'token': chunk.text})}\n\n"

            # After streaming completes
            reply_text = "".join(full_reply)
            save_message(sid, "model", reply_text)

            # Generate title for new chats
            if is_first:
                asyncio.create_task(generate_title(sid, body.message))

            # Maybe summarize in background
            asyncio.create_task(maybe_summarize(sid))

            yield f"data: {json.dumps({'done': True, 'session_id': sid})}\n\n"

        except Exception as e:
            err = str(e)
            yield f"data: {json.dumps({'error': err})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ── Non-streaming fallback ────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    check_rate_limit(request.client.host)

    sid = body.session_id
    create_session(sid)
    is_first = get_message_count(sid) == 0
    save_message(sid, "user", body.message)
    context = build_context(sid)
    context.append(types.Content(
        role="user",
        parts=[types.Part(text=body.message)]
    ))

    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=4096,
                ),
                contents=context
            )
            reply = response.text
            save_message(sid, "model", reply)

            if is_first:
                asyncio.create_task(generate_title(sid, body.message))
            asyncio.create_task(maybe_summarize(sid))

            return JSONResponse({"reply": reply, "session_id": sid})

        except Exception as e:
            last_error = str(e)
            if any(x in last_error.lower() for x in ["503", "unavailable", "overloaded"]):
                await asyncio.sleep(3 * (attempt + 1))
                continue
            break

    return JSONResponse(
        status_code=503,
        content={"error": f"AI service unavailable: {last_error}"}
    )


# ── Session management ────────────────────────────────────────────────────────

@app.post("/new-chat")
def new_chat(body: NewChatRequest):
    create_session(body.session_id)
    return {"status": "ok", "session_id": body.session_id}


@app.get("/sessions")
def list_sessions():
    return {"sessions": get_sessions(40)}


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    return {"status": "deleted"}


@app.get("/history/{session_id}")
def session_history(session_id: str):
    msgs = get_history(session_id, limit=100)
    return {"messages": msgs}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "running", "model": MODEL}