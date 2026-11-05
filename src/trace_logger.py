import json
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import message_to_dict
from pathlib import Path

LOG_FILE = Path("logs/trace.json")

def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text")


def _describe(m, limit=300) -> str:
    body = _text(m.content)
    for c in getattr(m, "tool_calls", None) or []:
        body += f" [calls {c['name']}({c['args']})]"
    body = " ".join(body.split())
    return f"{m.type}: {body[:limit]}{'…' if len(body) > limit else ''}"


class TraceLogger(BaseCallbackHandler):

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.n = 0
        self.agent = "?"
        self.after = "?"
        self.sent = []
        self.totals = {}
        LOG_FILE.parent.mkdir(exist_ok=True)

    def on_chat_model_start(self, model_name: str, messages: list, **kwargs) -> None:
        self.sent = messages[0]
        head = str(self.sent[0].content)[:60]
        #print(f"HEAD: {head}")
        self.agent = ("Playlist" if "playlist specialist" in head
                        else "Travel" if "travel agent" in head
                        else "Venue" if "venue specialist" in head
                        else "Coordinator")
        self.after = self.sent[-1].type

    def on_llm_end(self, response: str, **kwargs) -> None:
        self.n += 1
        msg = response.generations[0][0].message
        u = msg.usage_metadata or {}
        i, o = u.get("input_tokens", 0), u.get("output_tokens", 0)
        t = self.totals.setdefault(self.agent, [0, 0, 0])
        t[0] += 1; t[1] += i; t[2] += o

        calls = [f"{c['name']}({str(c['args'])[:100]})" for c in msg.tool_calls] \
                or ["(final answer)"]
        print(f"  #{self.n} {self.agent:<10} after={self.after:<5} "
                f"in={i:<5} out={o:<4} -> {', '.join(calls)}")
        if self.verbose:
            for m in self.sent[1:]:                    # skip system prompt (same every call)
                print(f"        SENT  {_describe(m)}")
            print(f"        GOT   {_describe(msg)}")

        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "call": self.n, "agent": self.agent,
                "tokens_in": i, "tokens_out": o,
                "sent": [message_to_dict(m) for m in self.sent],   # includes system prompt
                "got": message_to_dict(msg),
            }, default=str) + "\n")