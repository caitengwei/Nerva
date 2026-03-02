"""Echo server demo: simplest runnable Nerva server.

No ML dependencies required. Demonstrates full Nerva serving stack
with a trivial Echo model that returns input unchanged.

Usage:
    # Start server:
    uvicorn examples.echo_server:app --port 8080

    # Or blocking (spawns uvicorn internally):
    python -m examples.echo_server

    # Send a request (in another terminal):
    python scripts/demo_client.py

Routes:
    POST /rpc/echo    Binary RPC inference endpoint
    GET  /v1/health   Health check
    GET  /v1/models   Registered model list
    GET  /metrics     Prometheus metrics
"""

from __future__ import annotations

from typing import Any

from nerva import Model, build_nerva_app, model, trace


class EchoModel(Model):
    """Returns {"echo": inputs["value"]}. No ML, pure Python."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"echo": inputs.get("value")}


# Registered globally and loaded lazily on server startup.
echo_handle = model("echo", EchoModel, backend="pytorch", device="cpu")


def pipeline(inp: Any) -> Any:
    return echo_handle(inp)


graph = trace(pipeline)
app = build_nerva_app({"echo": graph})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.echo_server:app", host="0.0.0.0", port=8080, reload=False)
