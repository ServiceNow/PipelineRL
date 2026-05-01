from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse PipelineRL result streams.")
    parser.add_argument("--results", type=Path, default=Path("results"), help="Path to the results directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    args = parser.parse_args()

    app = create_app(args.results.resolve())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

