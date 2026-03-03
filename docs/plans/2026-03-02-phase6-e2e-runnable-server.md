# E2E Runnable Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 API 不一致，提供一个可以用 `uvicorn` 直接启动的 Echo 服务器，并附带一个可独立运行的客户端脚本，跑通端到端推理请求。

**Architecture:** 新增 `build_nerva_app(pipelines)` 函数，返回带生命周期管理的 Starlette ASGI app（`on_startup` 启动 Worker 进程，`on_shutdown` 关闭），从而支持 `uvicorn examples.echo_server:app`。`serve()` 内部改为调用 `build_nerva_app()`。

**Tech Stack:** Python 3.11+, Starlette, uvicorn, msgpack, httpx (tests)

---

## 现状分析

| 组件 | 状态 | 说明 |
|------|------|------|
| Binary RPC protocol | ✅ | `server/protocol.py` 完整实现 |
| RpcHandler | ✅ | `server/rpc.py` 完整实现 |
| Worker lifecycle | ✅ | WorkerManager + WorkerProxy 完整 |
| `build_app()` | ✅ | 已有，接受已就绪的 executors dict |
| `serve(pipelines)` | ✅ 但阻塞 | 内部 `asyncio.run()`，无法返回 ASGI app |
| `build_nerva_app()` | ❌ 缺失 | 需要新增，返回带 lifespan 的 ASGI app |
| `examples/01_single_model.py` | ❌ 坏 | `serve(graph, route=...)` API 不匹配 |
| 可运行 demo server | ❌ 缺失 | 需要 `examples/echo_server.py` |
| 客户端脚本 | ❌ 缺失 | 需要 `scripts/demo_client.py` |

端到端链路（已在 `tests/test_phase4_e2e.py` 中验证通过，使用 real Worker 进程）：
```
uvicorn → Starlette(on_startup) → WorkerManager.start_worker()
         → RpcHandler.handle() → _PipelineExecutor.execute()
         → Executor → WorkerProxy.infer() → Worker subprocess → EchoModel.infer()
```

---

### Task 1: 新增 `build_nerva_app()` + 简化 `serve()` + 测试

**Files:**
- Modify: `src/nerva/server/serve.py`
- Modify: `src/nerva/__init__.py`
- Modify: `tests/test_serve.py`

**Step 1: 写失败测试**

在 `tests/test_serve.py` 末尾新增：

```python
class TestBuildNervaApp:
    async def test_health_endpoint(self) -> None:
        """build_nerva_app() 启动后 /v1/health 返回 ok。"""
        import httpx
        from nerva.server.serve import build_nerva_app
        from tests.helpers import EchoModel

        handle = model("echo_app_health", EchoModel, backend="pytorch", device="cpu")
        graph = trace(lambda inp: handle(inp))
        app = build_nerva_app({"echo": graph})

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    async def test_rpc_echo_end_to_end(self) -> None:
        """build_nerva_app() 能跑通真实 Worker 进程的推理请求。"""
        import time
        import httpx
        import msgpack
        from nerva import trace
        from nerva.server.protocol import Frame, FrameType, decode_frame, encode_frame
        from nerva.server.serve import build_nerva_app
        from tests.helpers import EchoModel

        handle = model("echo_app_rpc", EchoModel, backend="pytorch", device="cpu")
        graph = trace(lambda inp: handle(inp))
        app = build_nerva_app({"echo": graph})

        def _make_body() -> bytes:
            return (
                encode_frame(Frame(FrameType.OPEN, 1, 0, msgpack.packb({"pipeline": "echo"})))
                + encode_frame(Frame(FrameType.DATA, 1, 0, msgpack.packb({"value": "world"})))
                + encode_frame(Frame(FrameType.END, 1, 0, b""))
            )

        deadline = int(time.time() * 1000) + 30000
        headers = {
            "content-type": "application/x-nerva-rpc",
            "x-nerva-deadline-ms": str(deadline),
            "x-nerva-stream": "0",
        }

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/rpc/echo", content=_make_body(), headers=headers)

        assert resp.status_code == 200
        frames = []
        offset = 0
        while offset < len(resp.content):
            frame, consumed = decode_frame(resp.content[offset:])
            frames.append(frame)
            offset += consumed
        data_frame = next(f for f in frames if f.frame_type == FrameType.DATA)
        result = msgpack.unpackb(data_frame.payload, raw=False)
        assert result == {"echo": "world"}
```

**Step 2: 运行测试验证失败**

```bash
uv run pytest tests/test_serve.py::TestBuildNervaApp -v
```
Expected: `ImportError: cannot import name 'build_nerva_app'`

**Step 3: 在 `serve.py` 中实现 `build_nerva_app()`，并简化 `serve()`**

在 `src/nerva/server/serve.py` 中，在 `serve()` 函数前插入：

```python
def build_nerva_app(pipelines: dict[str, Graph]) -> Starlette:
    """Return an ASGI app with worker lifecycle managed via Starlette lifespan.

    Workers are started on application startup and shut down on shutdown.
    Use this when you want to control the server yourself (e.g. with uvicorn):

        app = build_nerva_app({"classify": graph})
        # uvicorn mymodule:app --port 8080

    Args:
        pipelines: Mapping from pipeline name to traced Graph.

    Returns:
        A Starlette ASGI application.
    """
    import prometheus_client
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    from nerva.server.rpc import RpcHandler

    manager = WorkerManager()
    live_executors: dict[str, Any] = {}
    live_model_info: list[dict[str, Any]] = []

    async def _on_startup() -> None:
        execs, info = await _build_pipelines(pipelines, manager)
        live_executors.update(execs)
        live_model_info.extend(info)

    async def _on_shutdown() -> None:
        await manager.shutdown_all()

    rpc_handler = RpcHandler(live_executors)

    async def _health(request: Any) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _models(request: Any) -> JSONResponse:
        return JSONResponse({"models": live_model_info})

    async def _metrics(request: Any) -> Response:
        data = prometheus_client.generate_latest()
        return Response(content=data, media_type=prometheus_client.CONTENT_TYPE_LATEST)

    return Starlette(
        routes=[
            Route("/rpc/{pipeline_name}", rpc_handler.handle, methods=["POST"]),
            Route("/v1/health", _health, methods=["GET"]),
            Route("/v1/models", _models, methods=["GET"]),
            Route("/metrics", _metrics, methods=["GET"]),
        ],
        on_startup=[_on_startup],
        on_shutdown=[_on_shutdown],
    )
```

然后将 `serve()` 简化为：

```python
def serve(
    pipelines: dict[str, Graph],
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Start the Nerva inference server (blocking).

    Scans all Graphs for model declarations, auto-spawns worker processes,
    builds the ASGI application, and starts uvicorn.

    Args:
        pipelines: Mapping from pipeline name to traced Graph.
        host: Bind address.
        port: Bind port.
    """
    app = build_nerva_app(pipelines)
    uvicorn.run(app, host=host, port=port, log_level="info")
```

同时在文件顶部加入缺失的 import（如果还没有）：
```python
from starlette.applications import Starlette
```

**Step 4: 运行测试验证通过**

```bash
uv run pytest tests/test_serve.py -v
```
Expected: all PASS（含两个新测试）

**Step 5: 导出 `build_nerva_app` 到 `__init__.py`**

在 `src/nerva/__init__.py` 中：
```python
# 修改这行：
from nerva.server.serve import serve
# 改为：
from nerva.server.serve import build_nerva_app, serve
```

在 `__all__` 列表中加入 `"build_nerva_app"`。

**Step 6: lint + 全量测试**

```bash
uv run ruff check src/ tests/
uv run pytest tests/ -q
```
Expected: 0 errors, 282+ passed

**Step 7: commit**

```bash
git add src/nerva/server/serve.py src/nerva/__init__.py tests/test_serve.py
git commit -m "feat(serve): add build_nerva_app() ASGI factory with worker lifespan"
```

---

### Task 2: 修复 `examples/01_single_model.py`

**Files:**
- Modify: `examples/01_single_model.py`

**Step 1: 修改 import 和 serve 调用**

将文件中：
```python
from nerva import Model, model, serve, trace
...
app = serve(graph, route="/rpc/classify")
```
改为：
```python
from nerva import Model, build_nerva_app, model, trace
...
app = build_nerva_app({"classify": graph})
```

底部注释改为：
```python
# --- Expected usage ---
#
# Start server:
#   uvicorn examples.01_single_model:app --port 8080
#
# or blocking:
#   python -c "from nerva import serve, trace; from examples.01_single_model import graph; serve({'classify': graph})"
#
# Client call (pseudo-code, using future client SDK):
#   import nerva.client
#   client = nerva.client.connect("http://localhost:8080")
#   result = client.call("classify", {"embedding": tensor})
#   print(result["label"])  # "positive" or "negative"
```

**Step 2: 验证文件语法正确**

```bash
uv run python -c "import ast; ast.parse(open('examples/01_single_model.py').read()); print('OK')"
```
Expected: `OK`

（不执行 import，因为 torch 可能未安装或导入 torchnet 较慢）

**Step 3: lint**

```bash
uv run ruff check examples/01_single_model.py
```
Expected: no errors

**Step 4: commit**

```bash
git add examples/01_single_model.py
git commit -m "fix(examples): update 01_single_model to use build_nerva_app()"
```

---

### Task 3: 创建 `examples/echo_server.py`（可直接运行的 demo）

**Files:**
- Create: `examples/echo_server.py`

**Step 1: 创建文件**

```python
"""Echo server demo — simplest possible Nerva server.

No ML dependencies required. Demonstrates the full Nerva serving stack
with a trivial Echo model that returns its input unchanged.

Usage:
    # Start server:
    uvicorn examples.echo_server:app --port 8080

    # Or blocking (spawns uvicorn internally):
    python -m examples.echo_server

    # Send a request (in another terminal):
    python scripts/demo_client.py

Routes:
    POST /rpc/echo       — Binary RPC inference endpoint
    GET  /v1/health      — Health check
    GET  /v1/models      — Registered model list
    GET  /metrics        — Prometheus metrics
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


# Declare model — registered globally, loaded lazily on server startup.
echo_handle = model("echo", EchoModel, backend="pytorch", device="cpu")


def pipeline(inp: Any) -> Any:
    return echo_handle(inp)


graph = trace(pipeline)

# ASGI app — used by: uvicorn examples.echo_server:app
app = build_nerva_app({"echo": graph})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("examples.echo_server:app", host="0.0.0.0", port=8080, reload=False)
```

**Step 2: 验证语法**

```bash
uv run python -c "import ast; ast.parse(open('examples/echo_server.py').read()); print('OK')"
```
Expected: `OK`

**Step 3: lint**

```bash
uv run ruff check examples/echo_server.py
```
Expected: no errors

**Step 4: commit**

```bash
git add examples/echo_server.py
git commit -m "feat(examples): add echo_server.py — minimal runnable demo server"
```

---

### Task 4: 创建 `scripts/demo_client.py`（独立请求脚本）

**Files:**
- Create: `scripts/demo_client.py`

**Step 1: 创建 scripts/ 目录并写脚本**

```python
#!/usr/bin/env python3
"""Nerva demo client — sends a Binary RPC request to a running Nerva server.

Usage:
    # First start the server:
    uvicorn examples.echo_server:app --port 8080

    # Then in another terminal:
    python scripts/demo_client.py

    # Custom host/port/input:
    python scripts/demo_client.py --url http://localhost:8080 --value "custom input"

Dependencies: msgpack (already in nerva deps)
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import urllib.request
from typing import Any

import msgpack

# Binary frame constants (mirrors server/protocol.py)
_MAGIC = 0x4E56
_VERSION = 1
_HEADER_FMT = ">HBBB3xQIIII"
_HEADER_SIZE = 32

_FRAME_OPEN = 1
_FRAME_DATA = 2
_FRAME_END = 3
_FRAME_ERROR = 4


def _encode_frame(frame_type: int, request_id: int, payload: bytes) -> bytes:
    header = struct.pack(
        _HEADER_FMT,
        _MAGIC, _VERSION, frame_type, 0,  # magic, ver, type, flags
        request_id, 1, len(payload), 0, 0  # req_id, stream_id, plen, crc, ext
    )
    return header + payload


def _decode_frames(data: bytes) -> list[dict[str, Any]]:
    frames = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < _HEADER_SIZE:
            break
        fields = struct.unpack_from(_HEADER_FMT, data, offset)
        _magic, _ver, ftype, _flags, req_id, _sid, plen, _crc, _ext = fields
        payload = data[offset + _HEADER_SIZE: offset + _HEADER_SIZE + plen]
        frames.append({"type": ftype, "request_id": req_id, "payload": payload})
        offset += _HEADER_SIZE + plen
    return frames


def call(
    url: str,
    pipeline: str,
    inputs: dict[str, Any],
    deadline_ms: int = 30000,
) -> dict[str, Any]:
    """Send a unary Binary RPC request and return the result dict.

    Args:
        url: Base URL of the Nerva server (e.g. "http://localhost:8080").
        pipeline: Pipeline name to call (e.g. "echo").
        inputs: Input dict to send.
        deadline_ms: Request deadline in milliseconds from now.

    Returns:
        Result dict from the pipeline.

    Raises:
        RuntimeError: On RPC-level error (ERROR frame received).
        urllib.error.URLError: On network error.
    """
    request_id = 42
    body = (
        _encode_frame(_FRAME_OPEN, request_id, msgpack.packb({"pipeline": pipeline}))
        + _encode_frame(_FRAME_DATA, request_id, msgpack.packb(inputs))
        + _encode_frame(_FRAME_END, request_id, b"")
    )
    deadline_epoch_ms = int(time.time() * 1000) + deadline_ms
    req = urllib.request.Request(
        f"{url}/rpc/{pipeline}",
        data=body,
        headers={
            "Content-Type": "application/x-nerva-rpc",
            "x-nerva-deadline-ms": str(deadline_epoch_ms),
            "x-nerva-stream": "0",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()

    frames = _decode_frames(raw)
    for frame in frames:
        if frame["type"] == _FRAME_ERROR:
            err = msgpack.unpackb(frame["payload"], raw=False)
            raise RuntimeError(f"RPC error {err.get('code')}: {err.get('message')}")
        if frame["type"] == _FRAME_DATA:
            return msgpack.unpackb(frame["payload"], raw=False)  # type: ignore[no-any-return]
    raise RuntimeError("No DATA frame in response")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nerva demo client")
    parser.add_argument("--url", default="http://localhost:8080", help="Server base URL")
    parser.add_argument("--pipeline", default="echo", help="Pipeline name")
    parser.add_argument("--value", default="hello from demo_client!", help="Input value")
    args = parser.parse_args()

    print(f"Calling {args.url}/rpc/{args.pipeline} ...")
    result = call(args.url, args.pipeline, {"value": args.value})
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
```

**Step 2: 确保 scripts/ 目录存在并 lint**

```bash
mkdir -p scripts
uv run ruff check scripts/demo_client.py
```
Expected: no errors

**Step 3: commit**

```bash
git add scripts/demo_client.py
git commit -m "feat(scripts): add demo_client.py — standalone Binary RPC request sender"
```

---

### Task 5: 端到端验证

**目标：** 确认 `build_nerva_app()` + real Worker + real HTTP 请求能跑通。

**Step 1: 运行全量测试（含新增的 TestBuildNervaApp）**

```bash
uv run pytest tests/ -q
```
Expected: 282+ passed, 0 failed

**Step 2: 手动启动 echo server 并发送请求**

终端 1（启动服务器）：
```bash
uv run uvicorn examples.echo_server:app --port 8080
```
Expected output:
```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080
```

终端 2（发送请求）：
```bash
uv run python scripts/demo_client.py
```
Expected output:
```
Calling http://localhost:8080/rpc/echo ...
Result: {'echo': 'hello from demo_client!'}
```

**Step 3: 验证 health + models endpoint**

```bash
curl http://localhost:8080/v1/health
# Expected: {"status":"ok"}

curl http://localhost:8080/v1/models
# Expected: {"models":[{"name":"echo","backend":"pytorch","device":"cpu"}]}
```

**Step 4: 关闭服务器（Ctrl+C），确认 Worker 进程正常退出**

No zombie processes after shutdown.

**Step 5: 最终全量测试 + commit**

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/ examples/ scripts/
```
Expected: all pass

```bash
git add .
git commit -m "chore: verify e2e runnable server — all tests passing"
```

---

## 完成标准

1. `uv run pytest tests/ -q` 全部通过
2. `uv run uvicorn examples.echo_server:app --port 8080` 正常启动，Worker 进程 READY
3. `uv run python scripts/demo_client.py` 返回 `{'echo': 'hello from demo_client!'}`
4. `curl http://localhost:8080/v1/health` 返回 `{"status":"ok"}`
