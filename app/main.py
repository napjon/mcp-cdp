"""App factory: db, routers, optional chat/MCP, worker, frontend."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.db import init_db
from app.lifecycle import maybe_run_retention, start_retention_loop
from app.models import AppError
from app.services.security import check_mutating_request, redact_secrets
from app.settings import get_settings
from app.worker import Worker

log = logging.getLogger("mcp_cdp")


def create_app(*, worker_enabled: bool | None = None) -> FastAPI:
    settings = get_settings()
    if worker_enabled is None:
        worker_enabled = True

    lifespan_hooks: list = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db()
        maybe_run_retention()
        retention_stop, retention_thread = start_retention_loop()
        worker = Worker()
        app.state.worker = worker
        if worker_enabled:
            worker.start()
        try:
            async with AsyncExitStack() as stack:
                for hook in lifespan_hooks:
                    await stack.enter_async_context(hook)
                yield
        finally:
            retention_stop.set()
            retention_thread.join(timeout=1)
            worker.stop()

    application = FastAPI(title="mcp-cdp", lifespan=lifespan)
    application.state.lifespan_hooks = lifespan_hooks

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            f"http://127.0.0.1:{settings.port}",
            f"http://localhost:{settings.port}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def _csrf(request: Request, call_next):
        try:
            check_mutating_request(request, get_settings())
        except AppError as exc:
            return _error_response(exc)
        return await call_next(request)

    from app.api.datasets import router as datasets_router
    from app.api.experiments import router as experiments_router
    from app.api.health import router as health_router
    from app.api.jobs import router as jobs_router
    from app.api.predict import router as predict_router
    from app.api.projects import router as projects_router

    application.include_router(health_router)
    application.include_router(projects_router)
    application.include_router(datasets_router)
    application.include_router(experiments_router)
    application.include_router(jobs_router)
    application.include_router(predict_router)

    try:
        from app.api.chat import router as chat_router

        application.include_router(chat_router)
    except ImportError:
        pass

    try:
        from app.mcp.http import mount_mcp

        mount_mcp(application, lifespan_hooks)
    except ImportError:
        pass

    @application.exception_handler(AppError)
    async def _app_error(_request: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc)

    @application.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return JSONResponse({"error": redact_secrets(detail, get_settings())}, status_code=exc.status_code)

    @application.exception_handler(RequestValidationError)
    async def _valid_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": "invalid request"}, status_code=422)

    @application.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse({"error": "internal error"}, status_code=500)

    dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        application.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")

    return application


def _error_response(exc: AppError) -> JSONResponse:
    settings = get_settings()
    body: dict = {"error": redact_secrets(exc.message, settings)}
    if exc.code:
        body["code"] = exc.code
    return JSONResponse(body, status_code=exc.status_code)


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
