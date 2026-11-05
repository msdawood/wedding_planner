from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import compaction
import play_list
import settings


def _tool(i, size):
    return ToolMessage(content="x" * size, tool_call_id=f"c{i}", id=f"t{i}", name="web_search")


def test_old_tool_outputs_are_shrunk_and_fresh_ones_capped():
    msgs = [HumanMessage("hi")]
    for i in range(5):
        msgs += [AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": f"c{i}"}]), _tool(i, 10_000)]
    changed = compaction.compact_messages(msgs)
    by_id = {m.id: len(m.content) for m in changed}
    # 3 old ones shrunk hard, the 2 freshest capped at TOOL_OUTPUT_MAX_CHARS
    for old in ("t0", "t1", "t2"):
        assert by_id[old] < settings.OLD_TOOL_OUTPUT_MAX_CHARS + 100
    for fresh in ("t3", "t4"):
        assert settings.OLD_TOOL_OUTPUT_MAX_CHARS < by_id[fresh] < settings.TOOL_OUTPUT_MAX_CHARS + 100
    assert all(m.tool_call_id for m in changed)          # pairing preserved
    assert len(changed) == 5


def test_small_outputs_untouched():
    assert compaction.compact_messages([HumanMessage("a"), _tool(0, 50)]) == []


def test_list_content_from_mcp_is_handled():
    m = ToolMessage(content=[{"type": "text", "text": "y" * 9000}], tool_call_id="c", id="t")
    changed = compaction.compact_messages([m])
    assert isinstance(changed[0].content, str) and len(changed[0].content) < 4200


def test_playlist_tool_is_read_only():
    q = play_list.query_playlist_db.invoke
    assert "Error" in q({"query": "DELETE FROM Track"})
    assert "Error" in q({"query": "SELECT 1; DROP TABLE Track"})
    assert "Error" in q({"query": "INSERT INTO Genre(Name) VALUES ('x')"})
    out = q({"query": "SELECT Name FROM Genre LIMIT 2"})
    assert "Error" not in out and out.strip()
    # even if the filter were bypassed, the connection itself is read-only
    import pytest
    with pytest.raises(Exception):
        play_list._database().run("INSERT INTO Genre(Name) VALUES ('x')", fetch="all")
        play_list._database()._execute("INSERT INTO Genre(Name) VALUES ('x')")
