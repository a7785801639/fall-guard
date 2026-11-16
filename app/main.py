from __future__ import annotations

import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .config import Settings
from .engine import FallGuardEngine
from .web import create_web_app


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def build_app() -> FastAPI:
    configure_logging()
    settings = Settings.from_env()
    if os.environ.get("STRICT_CONFIG", "0") == "1":
        settings.require_all_secrets()

    engine = FallGuardEngine(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        engine.start()
        try:
            yield
        finally:
            engine.stop()

    web_app = create_web_app(engine)
    web_app.router.lifespan_context = lifespan
    return web_app


def main() -> None:
    configure_logging()
    settings = Settings.from_env()
    if os.environ.get("STRICT_CONFIG", "0") == "1":
        settings.require_all_secrets()

    app = build_app()
    uvicorn.run(
        app,
        host=settings.web_host,
        port=settings.web_port,
        log_level="info",
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    main()
