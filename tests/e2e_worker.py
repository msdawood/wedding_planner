"""Dev tool for tests/e2e_smoke.py: starts a REAL Celery worker (prefork, both queues) but with the
LLM, tools and DeepEval judge replaced by fakes, so the whole pipeline can be exercised offline."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wedding_graph as wg                      # noqa: E402
import evals                                    # noqa: E402
from test_graph import FakeAgent, FakeLLM       # noqa: E402

import asyncio                                  # noqa: E402

counter = {}
CALLS_FILE = os.getenv("E2E_CALLS_FILE")         # survives a killed child process (unlike memory)
SLOW = float(os.getenv("E2E_SLOW_VENUES", "0"))


class LoggedAgent(FakeAgent):
    async def ainvoke(self, _input, config=None):
        if CALLS_FILE:
            with open(CALLS_FILE, "a") as f:
                f.write(self.key + "\n")
        if self.key == "venues" and SLOW:
            await asyncio.sleep(SLOW)
        return await super().ainvoke(_input, config)


async def _travel(_llm):
    return LoggedAgent("FLIGHTS-DATA", counter, "flights")


wg.get_llm = lambda *a, **k: FakeLLM()
wg.travel_agent = _travel
wg.venue_agent = lambda _llm: LoggedAgent("VENUES-DATA", counter, "venues")
wg.playlist_agent = lambda _llm: LoggedAgent("PLAYLIST-DATA", counter, "playlist")
evals.run_metrics = lambda *a: {
    "hallucination_score": 0.9, "hallucination_passed": True, "hallucination_reason": "grounded",
    "relevancy_score": 0.95, "relevancy_passed": True, "relevancy_reason": "relevant"}

from tasks import celery_app                    # noqa: E402

if __name__ == "__main__":
    celery_app.worker_main(["worker", "--loglevel=info", "-Q", "runs,evals", "--concurrency=1", "-P", "prefork"])
