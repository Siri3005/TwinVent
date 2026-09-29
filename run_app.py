"""Start TwinVent's local, offline benchmark replay app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.server import serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TwinVent offline test-lung replay app.")
    parser.add_argument("--port", type=int, default=8765, help="Local browser port (default: 8765).")
    parser.add_argument("--open-browser", action="store_true", help="Open the local page in your default browser.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.open_browser:
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{args.port}")
    serve(port=args.port)


if __name__ == "__main__":
    main()
