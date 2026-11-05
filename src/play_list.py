import os
import re
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain.tools import tool
from langchain.agents import create_agent

from compaction import compact_history

_DB_PATH = Path(os.getenv("CHINOOK_DB_PATH", "resources/Chinook.db")).resolve()
_db = None

# The model writes this SQL, so the connection is read-only AND we only allow reads.
_READ_ONLY = re.compile(r"^\s*(select|with|pragma\s+table_info|explain)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|vacuum)\b", re.IGNORECASE)


def _database() -> SQLDatabase:
    global _db
    if _db is None:
        _db = SQLDatabase.from_uri(f"sqlite:///file:{_DB_PATH}?mode=ro&uri=true")
    return _db


@tool
def query_playlist_db(query: str) -> str:
    """Query the database for playlist information (read-only: SELECT statements only)"""
    statements = [s for s in query.split(";") if s.strip()]
    if len(statements) != 1 or not _READ_ONLY.match(query) or _FORBIDDEN.search(query):
        return "Error: only a single read-only SELECT query is allowed."
    try:
        return _database().run(query)
    except Exception as e:
        return f"Error querying database: {e}"


# Playlist agent
def playlist_agent(model):
    return create_agent(
        model=model,
        tools=[query_playlist_db],
        middleware=[compact_history],
        system_prompt="""
        You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
        Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
        If you run into errors when querying the database, try to fix them by making changes to the query.
        Do not come back empty handed, keep trying to query the db until you find a list of songs.

        This is a SQLite database. Before writing any data queries, first discover the schema.
        """
    )
