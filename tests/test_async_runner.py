"""Regression test for 'RuntimeError: Event loop is closed' on the 2nd job in a worker process.

LangChain's OpenAI client caches one httpx.AsyncClient for the whole process. Reusing it from a
second asyncio.run() (a new loop) fails; tasks.run_async keeps a single loop alive instead.
"""
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

import tasks


class _KeepAlive(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive, like a real API gateway

    def do_GET(self):
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _KeepAlive)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()


def test_process_wide_async_client_survives_consecutive_jobs(url):
    shared = httpx.AsyncClient(timeout=10)             # what langchain-openai caches globally

    async def job():
        return (await shared.get(url)).status_code

    assert [tasks.run_async(job()) for _ in range(5)] == [200] * 5


def test_plain_asyncio_run_is_what_broke(url):
    """Documents the bug: same shared client, one asyncio.run per job -> fails."""
    shared = httpx.AsyncClient(timeout=10)

    async def job():
        return (await shared.get(url)).status_code

    assert asyncio.run(job()) == 200
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(job())


def test_leftover_tasks_are_cancelled_after_a_failure():
    leaked = {}

    async def job():
        leaked["task"] = asyncio.ensure_future(asyncio.sleep(60))
        raise KeyError("boom")

    with pytest.raises(KeyError):
        tasks.run_async(job())
    assert leaked["task"].cancelled()
    assert tasks.run_async(asyncio.sleep(0, "still usable")) == "still usable"
