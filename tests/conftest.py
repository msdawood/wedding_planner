import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# Must be set before `settings` is imported anywhere.
os.environ.setdefault("JWT_SECRET", "x" * 48)
os.environ.setdefault("API_USERNAME", "tester")
os.environ.setdefault("API_PASSWORD", "s3cret-pass")
os.environ.setdefault("RUNS_PER_MINUTE", "3")
os.environ["CHINOOK_DB_PATH"] = str(Path(__file__).resolve().parents[1] / "resources" / "Chinook.db")
for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LITELLM_BASE_URL"):
    os.environ.pop(k, None)
