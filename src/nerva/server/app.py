"""ASGI application assembly for Nerva inference server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from nerva.server.rpc import RpcHandler

if TYPE_CHECKING:
    from starlette.requests import Request


def build_app(
    pipelines: dict[str, Any],
    model_info: list[dict[str, Any]],
) -> Starlette:
    """Assemble the full Nerva ASGI application.

    Args:
        pipelines: Mapping from pipeline name to executor (with execute() method).
        model_info: List of model info dicts for the /v1/models endpoint.

    Returns:
        A Starlette ASGI application.
    """
    rpc_handler = RpcHandler(pipelines)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def models(request: Request) -> JSONResponse:
        return JSONResponse({"models": model_info})

    return Starlette(
        routes=[
            Route("/rpc/{pipeline_name}", rpc_handler.handle, methods=["POST"]),
            Route("/v1/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
        ],
    )
