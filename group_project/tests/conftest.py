"""
Pytest setup — tránh access violation / stack trace ồn trên Windows
khi load pyarrow + transformers lần đầu trong test Task 5.
"""

import faulthandler
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Tắt dump stack trace "Windows fatal exception" (test vẫn pass, chỉ log ồn)
if sys.platform == "win32":
    faulthandler.disable()


def pytest_sessionstart(session):
    """Preload DLL/model một lần trước khi chạy test."""
    try:
        import pyarrow  # noqa: F401 — preload DLL sớm, tránh race khi import sau
    except ImportError:
        pass

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from src._embeddings import embed_text

        embed_text("warmup")
    except Exception:
        pass
