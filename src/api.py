"""FastAPI gateway: JWT auth, payload sanitisation, rate limiting, instant tracker tokens.

The gateway never runs the agent. It validates, enqueues a Celery job and answers in
milliseconds with a run_id the client can poll or stream.
"""
import asyncio
import json
import re
import unicodedata
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict, Field, field_validator

import auth
import run_store
import settings
from auth import User, get_current_user
from celery_app import RUN_TASK, celery_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    auth.assert_configured()
    app.state.redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        yield
    finally:
        await app.state.redis.aclose()


app = FastAPI(title="Wedding Planner Gateway", lifespan=lifespan)


def get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis


# --------------------------------------------------------------------------- schemas
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩﻿]")


def sanitize_text(value: str) -> str:
    """Normalise unicode, drop control / bidi / zero-width characters, collapse whitespace."""
    value = unicodedata.normalize("NFKC", value)
    value = _CONTROL.sub("", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


class RunPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")           # unknown fields are rejected, not ignored

    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$",
                                  description="Optional. Groups runs in Langfuse; generated if omitted.")
    assistant_id: Literal["wedding_planner"] = "wedding_planner"
    user_message: str = Field(min_length=1)

    @field_validator("user_message", mode="after")
    @classmethod
    def _clean_message(cls, v: str) -> str:
        v = sanitize_text(v)
        if not v:
            raise ValueError("user_message is empty after sanitisation")
        if len(v) > settings.MAX_MESSAGE_CHARS:
            raise ValueError(f"user_message exceeds {settings.MAX_MESSAGE_CHARS} characters")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RunAccepted(BaseModel):
    run_id: str
    thread_id: str
    status: str
    status_url: str
    stream_url: str


# --------------------------------------------------------------------------- auth
@app.post("/auth/token", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    if not auth.authenticate(form.username, form.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token, expires = auth.create_token(form.username)
    return TokenResponse(access_token=token, expires_in=expires)


# --------------------------------------------------------------------------- runs
async def _owned_run(run_id: str, user: User, r: aioredis.Redis) -> dict:
    """404 for both 'missing' and 'not yours' so run ids can't be probed."""
    run = await run_store.a_get_run(r, run_id)
    if run is None or run.get("user_id") != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run


@app.post("/runs", status_code=status.HTTP_202_ACCEPTED, response_model=RunAccepted)
async def create_run(payload: RunPayload, user: User = Depends(get_current_user),
                     r: aioredis.Redis = Depends(get_redis)):
    if await run_store.a_rate_limited(r, user.id):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many runs, slow down",
                            headers={"Retry-After": "60"})

    run_id = uuid.uuid4().hex                        # unguessable tracker token
    thread_id = payload.thread_id or uuid.uuid4().hex
    await run_store.a_create_run(r, run_id, user.id, thread_id)
    try:
        # send_task is blocking network I/O -> keep it off the event loop
        await asyncio.to_thread(
            celery_app.send_task, RUN_TASK,
            args=[run_id, user.id, thread_id, payload.user_message], task_id=run_id)
    except Exception:
        await run_store.a_fail(r, run_id, "queue_unavailable")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Could not enqueue the run, try again")

    return RunAccepted(run_id=run_id, thread_id=thread_id, status=run_store.QUEUED,
                       status_url=f"/runs/{run_id}", stream_url=f"/runs/{run_id}/stream")


@app.get("/runs/{run_id}")
async def get_run(run_id: str, user: User = Depends(get_current_user), r: aioredis.Redis = Depends(get_redis)):
    run = await _owned_run(run_id, user, r)
    run.pop("user_id", None)
    return {"run_id": run_id, **run}


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, user: User = Depends(get_current_user),
                     r: aioredis.Redis = Depends(get_redis),
                     last_event_id: str | None = Header(default=None)):
    """Server-sent events. Replays from the start (or Last-Event-ID), so a late client
    still receives everything and the stream always ends with a `done` event."""
    await _owned_run(run_id, user, r)

    async def events():
        cursor = last_event_id or "0-0"
        deadline = asyncio.get_running_loop().time() + settings.TASK_TIME_LIMIT * (settings.TASK_MAX_RETRIES + 1)
        while asyncio.get_running_loop().time() < deadline:
            batches = await r.xread({run_store.events_key(run_id): cursor}, block=15000, count=50)
            if not batches:
                yield ": keep-alive\n\n"
                run = await run_store.a_get_run(r, run_id)
                if run is None or run.get("status") in run_store.TERMINAL:   # missed/expired `done` event
                    yield f"event: done\ndata: {json.dumps({'status': (run or {}).get('status', 'expired')})}\n\n"
                    return
                continue
            for _, entries in batches:
                for entry_id, fields in entries:
                    cursor = entry_id
                    yield f"id: {entry_id}\nevent: {fields['type']}\ndata: {fields['data']}\n\n"
                    if fields["type"] == "done":
                        return

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/healthz")
async def healthz(r: aioredis.Redis = Depends(get_redis)):
    await r.ping()
    return {"status": "ok"}
