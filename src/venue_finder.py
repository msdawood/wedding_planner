import os
from typing import Dict, Any

from langchain.agents import create_agent
from langchain.tools import tool

from compaction import compact_history

_tavily_client = None


def _tavily():
    """Create the client lazily so importing this module never needs a key
    (the API container imports the graph package too)."""
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient

        _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily_client


@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters
      (e.g., use 'capacite' instead of 'capacité').
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        return _tavily().search(query, max_results=5)
    except Exception as e:
        return {"error": str(e)}


# Venue agent
def venue_agent(model):
    return create_agent(
        model=model,
        tools=[web_search],
        middleware=[compact_history],
        system_prompt="""
        You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
        You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
        - Price (lowest)
        - Capacity (exact match)
        - Reviews (highest)
        You may need to make multiple searches to iteratively find the best options.
        You have a suggested limit of 12 web searches. Count every web_search call you make.
        After 12 searches, you should stop searching and summarize the best options you have
        found so far.
        """
    )
