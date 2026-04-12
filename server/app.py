
import os
import sys

# Ensure the repo root (parent of this server/ directory) is on the path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from app import app  # noqa: F401  — re-export for `uvicorn server.app:app`

__all__ = ["app", "main"]


def main() -> None:
    """Launch the uvicorn server (used by `uv run --project . server`)."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    main()
