"""Regression tests for 'works in my test, fails in the container' import problems."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Mirrors `celery -A tasks.celery_app worker`: Celery's own loader imports the app with the working
# directory temporarily on sys.path and then REMOVES it. `-I` (isolated mode) means the interpreter
# itself adds neither the cwd nor PYTHONPATH, so only what our code pins can make lazy imports work.
LOAD_LIKE_CELERY = """
import sys
from celery.app.utils import find_app
find_app("tasks.celery_app")
import wedding_graph, evals      # lazy imports done inside the tasks at job time
print("ok")
"""


def run_like_worker(src_dir: Path):
    return subprocess.run([sys.executable, "-I", "-c", LOAD_LIKE_CELERY], cwd=src_dir,
                          env={"JWT_SECRET": "x" * 40, "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True, timeout=180)


def test_lazy_imports_work_after_celery_loads_the_app():
    out = run_like_worker(SRC)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-1000:]


def test_dockerfile_sets_pythonpath_and_copies_everything_the_worker_imports():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "PYTHONPATH=/app" in dockerfile
    assert "COPY src/ /app/" in dockerfile and "COPY resources/ /app/resources/" in dockerfile
    ignore = (ROOT / ".dockerignore").read_text().split()
    for module in SRC.glob("*.py"):
        assert module.name not in ignore and module.stem not in ignore
