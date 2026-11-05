"""Local memory-compaction rules for the specialist agents (no database involved).

The venue agent can run a dozen web searches and every Tavily response is large, so an
agent loop can balloon past the context window (and the token bill). Rules applied
before every model call:

  1. Any single tool result is capped at TOOL_OUTPUT_MAX_CHARS.
  2. Only the most recent KEEP_FRESH_TOOL_OUTPUTS tool results stay near-full size;
     older ones are shrunk to OLD_TOOL_OUTPUT_MAX_CHARS.

We rewrite message *content* in place (same message ids -> the reducer replaces them),
never remove messages, so AI tool-call / tool-result pairing is always preserved.
"""
import json

from langchain.agents.middleware import before_model
from langchain_core.messages import ToolMessage

import settings


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    try:
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(json.dumps(block, default=str))
        return "\n".join(parts)
    except TypeError:
        return str(content)


def _shrink(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[compacted: {omitted} characters omitted]"


def compact_messages(messages: list) -> list:
    """Return only the messages whose content changed (empty list = nothing to do)."""
    tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    fresh = set(tool_idx[-settings.KEEP_FRESH_TOOL_OUTPUTS:]) if settings.KEEP_FRESH_TOOL_OUTPUTS else set()
    changed = []
    for i in tool_idx:
        msg = messages[i]
        limit = settings.TOOL_OUTPUT_MAX_CHARS if i in fresh else settings.OLD_TOOL_OUTPUT_MAX_CHARS
        text = _as_text(msg.content)
        new_text = _shrink(text, limit)
        if new_text != text:
            changed.append(
                ToolMessage(
                    content=new_text,
                    tool_call_id=msg.tool_call_id,
                    id=msg.id,
                    name=msg.name,
                    status=getattr(msg, "status", "success"),
                )
            )
    return changed


@before_model
def compact_history(state, runtime):
    changed = compact_messages(state["messages"])
    return {"messages": changed} if changed else None
