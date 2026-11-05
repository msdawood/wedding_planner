"""Central configuration. Everything comes from environment variables so the same
image runs on a laptop (docker compose) and anywhere else."""
import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# --- Infrastructure -------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

# --- LLM access (always via the LiteLLM gateway when LITELLM_BASE_URL is set) ---
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "")          # e.g. http://litellm_proxy:4000
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")            # the *virtual* key with a budget
# Model name as exposed by the proxy (see litellm-config.yaml) ...
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "claude-haiku")
# ... or the raw Anthropic model id, used only when running without the proxy.
DIRECT_MODEL = os.getenv("MODEL", "claude-haiku-4-5-20251001")

# --- Auth -----------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_ISSUER = os.getenv("JWT_ISSUER", "wedding-planner")
JWT_EXPIRES_MIN = _int("JWT_EXPIRES_MIN", 60)
API_USERNAME = os.getenv("API_USERNAME", "")
API_PASSWORD = os.getenv("API_PASSWORD", "")

# --- API guard rails ---------------------------------------------------------
MAX_MESSAGE_CHARS = _int("MAX_MESSAGE_CHARS", 2000)
RUNS_PER_MINUTE = _int("RUNS_PER_MINUTE", 10)
RUN_TTL_SECONDS = _int("RUN_TTL_SECONDS", 24 * 3600)

# --- Worker -----------------------------------------------------------------
TASK_SOFT_TIME_LIMIT = _int("TASK_SOFT_TIME_LIMIT", 600)
TASK_TIME_LIMIT = _int("TASK_TIME_LIMIT", 660)
TASK_MAX_RETRIES = _int("TASK_MAX_RETRIES", 3)
WORKER_CONCURRENCY = _int("WORKER_CONCURRENCY", 2)
# Recycle a pool process after N tasks / when it grows beyond N KiB.
WORKER_MAX_TASKS_PER_CHILD = _int("WORKER_MAX_TASKS_PER_CHILD", 20)
WORKER_MAX_MEMORY_PER_CHILD_KB = _int("WORKER_MAX_MEMORY_PER_CHILD_KB", 1_000_000)

# --- Graph guard rails ---------------------------------------------------------
GRAPH_RECURSION_LIMIT = _int("GRAPH_RECURSION_LIMIT", 40)
MAX_GUESTS = _int("MAX_GUESTS", 2000)
TOOL_OUTPUT_MAX_CHARS = _int("TOOL_OUTPUT_MAX_CHARS", 4000)      # per tool result
OLD_TOOL_OUTPUT_MAX_CHARS = _int("OLD_TOOL_OUTPUT_MAX_CHARS", 500)  # after it is no longer fresh
KEEP_FRESH_TOOL_OUTPUTS = _int("KEEP_FRESH_TOOL_OUTPUTS", 2)

# --- Observability -------------------------------------------------------------
LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
