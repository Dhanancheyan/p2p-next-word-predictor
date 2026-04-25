"""
__main__.py
Entry point for the NWP v11 hybrid LSTM federated keyboard LM server.

Usage
-----
    python -m nwp_v12 [--host HOST] [--port PORT] [--data-dir DIR] [--static-dir DIR]

Port must be in the range 8001-8020 (the auto-discovery scan range).
"""
from __future__ import annotations

import argparse
import os

import uvicorn

from nwp_v12.Server.app import create_app
from nwp_v12.Server.settings import build_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NWP Keyboard LM -- v11 Hybrid LSTM Federated"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=8001,
        help="Port to listen on. Must be in range 8001-8020 (auto-discovery range).",
    )
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--static-dir", default="")
    args = parser.parse_args()

    if not (8001 <= args.port <= 8020):
        parser.error(f"--port must be in range 8001-8020, got {args.port}")

    base_dir = os.getcwd()
    data_dir = args.data_dir or os.path.join(base_dir, "data", "trainer")
    static_dir = args.static_dir or os.path.join(os.path.dirname(__file__), "Frontend")

    settings = build_settings(data_dir=data_dir, port=args.port)
    app = create_app(settings=settings, static_dir=static_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
