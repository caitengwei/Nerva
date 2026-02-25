# Spike S4: Async Dynamic Batcher Benchmark Report

**日期:** 2026-02-25
**脚本:** `spikes/s4_async_batcher_benchmark.py`
**状态:** 已完成

---

## 1. 目标

验证基于 `asyncio.Queue` + 定时器的动态批处理（dynamic batching）机制在 Nerva 推理服务场景下的延迟和吞吐量特性，具体需要确认以下三个命题：

1. **低并发下定时器刷新正常工作** — 当并发度极低时，请求不会因等待凑满一批而永久阻塞，timer 能够按时触发并将小批次发送出去。
2. **高并发下吞吐量显著提升** — 批处理能够有效摊薄推理的固定开销，在高并发场景下大幅提高 RPS（Requests Per Second）。
3. **asyncio 调度开销可量化** — 量化 `asyncio` 事件循环与 Queue 操作本身引入的额外延迟，为后续性能预算提供基准。

---

## 2. 测试方法

### DynamicBatcher 工作原理

`DynamicBatcher` 是本次 spike 的核心被测组件，其工作流程如下：

**请求提交路径：**

```python
async def submit(self, request: Request) -> BatchResult:
    request.submit_time = time.monotonic()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[BatchResult] = loop.create_future()
    await self._queue.put((request, future))
    return await future
```

调用方通过 `submit()` 将请求放入 `asyncio.Queue`，并等待与该请求绑定的 `asyncio.Future` 完成。提交后调用方立即挂起，控制权交还给事件循环。

**批次聚合逻辑（`_batch_loop`）：**

```python
async def _batch_loop(self) -> None:
    while True:
        batch = []
        first = await self._queue.get()   # 阻塞等待第一个请求到达
        batch.append(first)

        deadline = time.monotonic() + self._max_delay_ms / 1000.0
        while len(batch) < self._max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        # 触发条件满足后执行推理
        results = await self._handler(requests)
        for req, fut in batch:
            fut.set_result(result_map[req.request_id])
```

批次触发条件为二者取其先：
- **size 触发：** 队列中已积累 `max_batch_size` 个请求；
- **timer 触发：** 自第一个请求入队起，经过 `max_delay_ms` 毫秒仍未凑满一批。

结果通过 `Future.set_result()` 分发回各个调用方 coroutine，实现完全异步的请求-响应解耦。

**模拟 GPU 推理（sublinear scaling）：**

```python
async def simulated_inference(batch, base_compute_ms=5.0):
    n = len(batch)
    compute_s = (base_compute_ms + 0.3 * n) / 1000.0
    await asyncio.sleep(compute_s)
    return [BatchResult(request_id=req.request_id, value=f"ok-{req.request_id}") for req in batch]
```

推理耗时模型为 `5.0 + 0.3 * batch_size` 毫秒，模拟 GPU 并行计算下批次大小带来的次线性（sublinear）开销增长——单请求需 5.3 ms，32 个请求的批次仅需 14.6 ms，摊到每个请求约 0.46 ms。

---

## 3. 测试参数

测试矩阵覆盖以下参数组合：

| 参数 | 取值 |
|---|---|
| `max_batch_size` | 8, 16, 32 |
| `max_delay_ms` | 1, 5, 10 |
| `concurrency`（同时在途请求数） | 1, 32, 128, 512 |
| 每场景请求总量 | 1000 |
| **场景总数** | **3 × 3 × 4 = 36** |

并发控制通过 `asyncio.Semaphore(concurrency)` 实现，限制同时进入 batcher 的最大请求数。所有 1000 个请求的 `Task` 在场景开始时一次性创建，由信号量约束实际并发度。

---

## 4. 测试环境

| 项目 | 详情 |
|---|---|
| 平台 | darwin |
| Python 版本 | 3.14.2 |
| 运行方式 | `uv run python spikes/s4_async_batcher_benchmark.py` |
| 推理模拟 | `asyncio.sleep`（纯 asyncio，无真实 GPU） |
| 基础计算延迟 | 5.0 ms + 0.3 ms × batch_size |

---

## 5. 测试结果

所有延迟和等待时间单位为毫秒（ms）。`wait_*` 列为请求从提交到批次触发的等待时间（即在 Queue 中排队的时间），`lat_*` 列为端到端请求延迟（含队列等待 + 推理执行）。

```
batch delay  conc |  lat_p50  lat_p95  lat_p99 | wait_p50 wait_p95 wait_p99 | bs_mean bs_min bs_max |      rps
    8     1     1 |     7.22     7.90    11.84 |     1.19     1.44     2.50 |     1.0      1      1 |      135
    8     1    32 |    33.52    34.20    34.42 |    25.09    25.59    25.78 |     8.0      8      8 |      952
    8     1   128 |   133.37   143.34   144.57 |   124.98   134.89   136.00 |     7.9      3      8 |      940
    8     1   512 |   526.29   545.39   545.69 |   517.58   536.74   537.11 |     8.0      8      8 |      944
    8     5     1 |    11.73    12.57    15.06 |     5.70     5.98     7.05 |     1.0      1      1 |       84
    8     5    32 |    33.17    40.79    41.65 |    24.73    28.13    32.89 |     8.0      8      8 |      941
    8     5   128 |   133.87   136.37   137.05 |   125.51   128.14   128.75 |     8.0      8      8 |      944
    8     5   512 |   526.92   540.67   541.02 |   518.19   532.13   532.53 |     8.0      8      8 |      948
    8    10     1 |    17.12    17.74    19.86 |    11.09    11.33    12.57 |     1.0      1      1 |       58
    8    10    32 |    33.54    35.01    37.34 |    25.10    26.10    28.66 |     8.0      8      8 |      949
    8    10   128 |   133.02   138.68   139.39 |   124.65   130.22   130.98 |     8.0      8      8 |      954
    8    10   512 |   527.76   543.24   543.67 |   519.36   534.81   535.62 |     8.0      8      8 |      948
   16     1     1 |     7.24     8.15    10.11 |     1.20     1.47     2.81 |     1.0      1      1 |      135
   16     1    32 |    21.77    21.89    22.97 |    10.81    10.87    11.97 |    15.9      8     16 |     1466
   16     1   128 |    86.81    87.97    89.79 |    75.79    77.00    77.91 |    15.9      8     16 |     1458
   16     1   512 |   345.50   350.13   351.42 |   335.45   339.12   340.13 |    15.9      8     16 |     1446
   16     5     1 |    11.75    12.15    13.13 |     5.72     5.91     6.44 |     1.0      1      1 |       85
   16     5    32 |    21.80    23.90    24.67 |    10.80    11.34    13.73 |    15.9      8     16 |     1429
   16     5   128 |    86.51    88.18    88.70 |    75.75    77.42    77.73 |    15.9      8     16 |     1450
   16     5   512 |   346.03   348.21   350.43 |   335.06   337.09   339.47 |    15.9      8     16 |     1427
   16    10     1 |    17.14    18.03    21.21 |    11.10    11.41    12.79 |     1.0      1      1 |       58
   16    10    32 |    21.50    22.40    23.42 |    10.80    11.20    12.47 |    15.9      8     16 |     1430
   16    10   128 |    88.05    94.38    94.87 |    77.00    82.77    83.52 |    15.9      8     16 |     1404
   16    10   512 |   345.49   348.74   349.54 |   334.89   338.09   338.27 |    15.9      8     16 |     1437
   32     1     1 |     7.27     8.04    10.18 |     1.22     1.54     2.68 |     1.0      1      1 |      134
   32     1    32 |    16.06    18.58    18.92 |     0.28     0.98     1.49 |    31.2      8     32 |     1916
   32     1   128 |    62.52    64.19    64.66 |    46.86    48.18    48.31 |    31.2      8     32 |     2001
   32     1   512 |   250.92   253.44   254.83 |   234.80   237.05   237.40 |    31.2      8     32 |     1978
   32     5     1 |    11.78    12.84    15.92 |     5.73     6.10     8.30 |     1.0      1      1 |       84
   32     5    32 |    15.95    17.23    17.40 |     0.16     0.61     1.42 |    31.2      8     32 |     1921
   32     5   128 |    63.86    65.05    66.72 |    47.70    48.38    48.70 |    31.2      8     32 |     1935
   32     5   512 |   251.34   254.71   255.46 |   235.40   238.87   239.60 |    31.2      8     32 |     1947
   32    10     1 |    17.10    17.50    19.84 |    11.08    11.26    11.94 |     1.0      1      1 |       58
   32    10    32 |    16.10    16.84    19.06 |     0.24     0.92     3.26 |    31.2      8     32 |     1887
   32    10   128 |    62.81    64.78    65.78 |    47.15    48.15    49.87 |    31.2      8     32 |     1935
   32    10   512 |   248.90   255.87   257.02 |   232.71   239.88   240.66 |    31.2      8     32 |     1909
```

---

## 6. 分析

### 6.1 低并发下定时器刷新（验证点 1）

所有 `concurrency=1` 场景的平均批次大小均为 **1.0**，即每个请求都作为独立的大小为 1 的批次被单独处理。这表明：

- Timer flush 机制工作正常。在低并发条件下，队列中几乎不存在请求积压，第一个请求到达后，在 `max_delay_ms` 超时内没有后续请求，timer 按时触发并立即将批次发送执行。
- **请求没有因等待凑满一批而阻塞**。`max_batch_size` 配置为 8/16/32 都不影响结果——当流量不足时，batcher 不会强行等到满批才执行。
- 各 delay 参数下，`wait_p50` 分别约为 1.2 ms（delay=1）、5.7 ms（delay=5）、11.1 ms（delay=10），与 `max_delay_ms` 配置高度吻合，证明 timer 精度在 asyncio 事件循环粒度下可接受。

### 6.2 高并发下吞吐量提升（验证点 2）

批处理对吞吐量的提升效果显著：

- **最优场景（batch=32, delay=1, conc=128）RPS 达 2001，而 conc=1 最高仅 135，吞吐提升约 14.8 倍。**
- 从 `bs_mean` 列可以看出，在 `concurrency >= 32` 时，批次几乎都能填满（`batch=32` 时 `bs_mean` 约 31.2），说明请求源源不断，batcher 主要由 size 触发而非 timer 触发。
- `batch=8` 在高并发下 RPS 约 940-954，`batch=16` 约 1404-1466，`batch=32` 约 1887-2001。批次大小加倍，吞吐量大致也翻倍，这与 sublinear 的推理开销模型一致：批次越大，每个请求摊到的固定开销越低。

| batch_size | 高并发典型 RPS | 相对 batch=8 的倍数 |
|---|---|---|
| 8 | ~950 | 1.0x |
| 16 | ~1440 | ~1.5x |
| 32 | ~1950 | ~2.1x |

### 6.3 asyncio 调度开销（验证点 3）

在 `delay=1ms, concurrency=1` 的场景下，batch_size 不影响批次（因为均为 size=1），`wait_p50` 的实测值约 **1.2 ms**（三个 batch_size 配置结果一致）。

- 理论上 timer 应在 1 ms 后触发，实际中位数为 1.2 ms，**超出约 0.2 ms**。
- 这 ~0.2 ms 的额外开销来源于：asyncio 事件循环调度延迟、`asyncio.wait_for` 内部实现的 overhead、以及 `time.monotonic()` 采样误差。
- 对于目标延迟在数十毫秒级别的推理服务，0.2 ms 的调度开销可以接受，不会成为瓶颈。

### 6.4 其他关键观察

**delay 参数在低并发下影响显著，高并发下几乎无关：**

- `concurrency=1` 时，`delay=1` 对应 RPS 约 135，`delay=10` 对应 RPS 约 58，差距超过 2 倍。这是因为低并发时每个请求都要等满 timer 才触发。
- `concurrency >= 32` 时，不同 delay 下的 RPS 几乎相同（例如 batch=32 时，delay=1/5/10 在 conc=512 下分别为 1978/1947/1909，差异在 4% 以内）。高并发下请求能快速填满批次，timer 基本不需要等到超时就已触发。

**端到端延迟与并发度近似线性：**

- batch=32 的 `lat_p50` 在 conc=32/128/512 下分别约 16/63/251 ms，与并发度 32/128/512 的比值（1:4:16）基本成正比。这是因为在单一 batcher 下，推理是串行执行的，高并发下请求需要在队列中排队等待前面批次完成。这提示在真实生产系统中，需要引入**多 worker 并发推理**来打破这一瓶颈。

---

## 7. 结论与建议

### 结论

三个验证点均已得到确认：

1. Timer flush 机制在低并发下工作正常，不会造成请求永久阻塞。
2. 批处理在高并发下带来显著的吞吐提升（最高约 14.6x）。
3. asyncio 调度开销约 0.2 ms，在推理服务延迟预算内完全可接受。

### 对 Nerva 动态批处理设计的影响

**推荐采用此架构作为 Nerva batcher 的基础设计**，并结合以下建议：

1. **`max_batch_size` 是最重要的吞吐旋钮。** 应根据 GPU 显存和模型 profile 确定合理的最大批次大小，并将其作为一等配置项暴露给用户。batch=32 相比 batch=8 带来约 2x 吞吐提升。

2. **`max_delay_ms` 的合理默认值为 1-5 ms。** delay=1 ms 在低并发时显著优于 delay=10 ms（吞吐高 2x），而高并发时 delay 参数影响可忽略不计。因此除非有特殊的延迟容忍需求，默认值建议 ≤ 5 ms。

3. **单 batcher 是吞吐瓶颈，需要多 worker 并行。** 当前设计中推理串行执行，高并发下延迟线性增长。Nerva 需要支持**多 worker 并发推理**（每个 worker 持有独立的 batcher + GPU stream），以在保持批处理收益的同时降低队列排队延迟。

4. **asyncio 适合作为调度层，推理执行应 offload 到线程池或子进程。** 真实的 GPU 推理是 CPU-bound 或 I/O-bound 操作（CUDA 调用），应使用 `asyncio.to_thread` 或 `concurrent.futures.ProcessPoolExecutor` 执行，避免阻塞事件循环。

5. **Future-based 结果分发机制设计合理，可直接沿用。** 每个请求绑定独立 Future，批次完成后通过 `result_map` 精确分发，无需轮询，开销极低。

---

## 8. 复现方式

在项目根目录下执行：

```bash
uv run python spikes/s4_async_batcher_benchmark.py
```

**前置条件：**
- Python 3.12+（脚本使用 `asyncio`、`dataclasses`、`statistics` 等标准库，无第三方依赖）
- 安装 `uv`（或直接使用 `python` 替换 `uv run python`）

**预期输出：**

脚本会依次运行 36 个场景，并在最终打印完整结果表格及三条关键观察结论。单次完整运行耗时视机器性能约 2-5 分钟。
