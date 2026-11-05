import json

import fakeredis
import pytest
from fastapi.testclient import TestClient

import api
import auth
import run_store
import settings


@pytest.fixture
def client(monkeypatch):
    server = fakeredis.FakeServer()
    monkeypatch.setattr(api.aioredis, "from_url",
                        lambda *a, **k: fakeredis.FakeAsyncRedis(server=server, decode_responses=True))
    sent = []
    monkeypatch.setattr(api.celery_app, "send_task", lambda name, **kw: sent.append((name, kw)))
    with TestClient(api.app) as c:
        c.sent = sent
        c.sync_redis = fakeredis.FakeRedis(server=server, decode_responses=True)
        yield c


def login(client, user="tester", password="s3cret-pass"):
    r = client.post("/auth/token", data={"username": user, "password": password})
    return r


def headers(client):
    return {"Authorization": f"Bearer {login(client).json()['access_token']}"}


def test_login_ok_and_bad_password(client):
    assert login(client).status_code == 200
    assert login(client, password="nope").status_code == 401
    assert login(client, user="other").status_code == 401


def test_runs_require_a_valid_jwt(client):
    body = {"user_message": "wedding in Paris"}
    assert client.post("/runs", json=body).status_code == 401
    assert client.post("/runs", json=body, headers={"Authorization": "Bearer garbage"}).status_code == 401
    # a token signed with another secret
    import jwt, time
    forged = jwt.encode({"sub": "tester", "iss": settings.JWT_ISSUER, "iat": 1, "exp": int(time.time()) + 99},
                        "y" * 48, algorithm="HS256")
    assert client.post("/runs", json=body, headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    # expired
    expired = jwt.encode({"sub": "tester", "iss": settings.JWT_ISSUER, "iat": 1, "exp": 2},
                         settings.JWT_SECRET, algorithm="HS256")
    assert client.post("/runs", json=body, headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    # alg=none attack
    none_tok = jwt.encode({"sub": "tester", "iss": settings.JWT_ISSUER, "iat": 1, "exp": int(time.time()) + 99},
                          None, algorithm="none")
    assert client.post("/runs", json=body, headers={"Authorization": f"Bearer {none_tok}"}).status_code == 401


def test_create_run_returns_instant_tracker_and_enqueues(client):
    r = client.post("/runs", json={"user_message": "  Wedding in\x00 Paris​ for 100  "}, headers=headers(client))
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and len(body["run_id"]) == 32
    assert body["stream_url"] == f"/runs/{body['run_id']}/stream"
    name, kw = client.sent[0]
    assert name == "tasks.execute_agent_workflow"
    run_id, user_id, thread_id, message = kw["args"]
    assert (run_id, user_id) == (body["run_id"], "tester")
    assert message == "Wedding in Paris for 100"          # control + zero-width chars removed, trimmed
    assert kw["task_id"] == run_id


@pytest.mark.parametrize("payload", [
    {"user_message": ""},
    {"user_message": "   \x00​  "},
    {"user_message": "x" * (settings.MAX_MESSAGE_CHARS + 1)},
    {"user_message": "ok", "thread_id": "../../etc/passwd"},
    {"user_message": "ok", "thread_id": "a" * 65},
    {"user_message": "ok", "is_admin": True},               # unknown field rejected
    {"user_message": "ok", "assistant_id": "other"},
    {},
])
def test_invalid_payloads_are_rejected(client, payload):
    r = client.post("/runs", json=payload, headers=headers(client))
    assert r.status_code == 422, r.text
    assert client.sent == []


def test_rate_limit(client):
    h = headers(client)
    codes = [client.post("/runs", json={"user_message": "wedding"}, headers=h).status_code for _ in range(5)]
    assert codes == [202, 202, 202, 429, 429]               # RUNS_PER_MINUTE=3 in conftest


def test_queue_outage_returns_503_and_marks_run_failed(client, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("broker down")
    monkeypatch.setattr(api.celery_app, "send_task", boom)
    r = client.post("/runs", json={"user_message": "wedding"}, headers=headers(client))
    assert r.status_code == 503


def test_runs_are_private_to_their_owner(client, monkeypatch):
    mine = headers(client)
    run_id = client.post("/runs", json={"user_message": "wedding"}, headers=mine).json()["run_id"]
    assert client.get(f"/runs/{run_id}", headers=mine).status_code == 200
    # a different (validly authenticated) user gets 404, indistinguishable from "no such run"
    other = {"Authorization": f"Bearer {auth.create_token('mallory')[0]}"}
    assert client.get(f"/runs/{run_id}", headers=other).status_code == 404
    assert client.get(f"/runs/{run_id}/stream", headers=other).status_code == 404
    assert client.get("/runs/doesnotexist", headers=mine).status_code == 404


def test_late_subscriber_replays_everything_and_stream_terminates(client):
    """The old Pub/Sub design lost events published before the client connected."""
    h = headers(client)
    run_id = client.post("/runs", json={"user_message": "wedding"}, headers=h).json()["run_id"]
    r = client.sync_redis
    run_store.set_status(r, run_id, run_store.RUNNING)
    run_store.add_event(r, run_id, "node_complete", {"node": "guard"})
    run_store.set_status(r, run_id, run_store.SUCCEEDED, result={"plan": "PLAN"})
    run_store.add_event(r, run_id, "done", {"status": "succeeded"})

    with client.stream("GET", f"/runs/{run_id}/stream", headers=h) as resp:
        text = "".join(resp.iter_text())
    events = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
    assert events == ["queued", "node_complete", "done"]

    got = client.get(f"/runs/{run_id}", headers=h).json()
    assert got["status"] == "succeeded" and got["result"] == {"plan": "PLAN"}
    assert "user_id" not in got


def test_stream_can_resume_from_last_event_id(client):
    h = headers(client)
    run_id = client.post("/runs", json={"user_message": "wedding"}, headers=h).json()["run_id"]
    r = client.sync_redis
    run_store.add_event(r, run_id, "node_complete", {"node": "guard"})
    run_store.add_event(r, run_id, "done", {"status": "succeeded"})
    first_id = r.xrange(run_store.events_key(run_id))[0][0]
    with client.stream("GET", f"/runs/{run_id}/stream", headers={**h, "Last-Event-ID": first_id}) as resp:
        text = "".join(resp.iter_text())
    assert "event: queued" not in text and "event: node_complete" in text and "event: done" in text


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}
