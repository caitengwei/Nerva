#!/usr/bin/env bash
# verify_postprocess_bottleneck.sh
#
# 验证假设：STREAM=true 性能瓶颈在于 mm_postprocess 实例数只有 1。
#
# 测试方案：
#   A) 基线：instance_count=default(1) — 预期 p50 ~2100ms (已有数据，可跳过)
#   B) 对照：instance_count=20 — 如果假设成立，p50 应降至 ~130ms
#
# 用法：
#   cd ~/Nerva
#   bash scripts/bench/verify_postprocess_bottleneck.sh [--skip-baseline]
#
# 前提：
#   - 当前在远程 Linux 机器上，tritonserver 26.03 容器可用
#   - uv 已安装且在 PATH 中
#   - 无需 GPU（CPU mock 模式）

set -euo pipefail

TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:26.03-py3}"
CONTAINER_CLI="${CONTAINER_CLI:-nerdctl}"
TRITON_GRPC_PORT=8003
TRITON_HTTP_PORT=8002
TRITON_METRICS_PORT=8004
MODEL_NAME="mm_vllm"
RESULT_BASE="bench-results/verify-postprocess-bottleneck"

# 快速验证参数：短 warmup + 短 sample，只跑 C=32
WARMUP=10
SAMPLE=30
CONCURRENCY=32

SKIP_BASELINE=false
if [[ "${1:-}" == "--skip-baseline" ]]; then
    SKIP_BASELINE=true
fi

# --- helpers ---

unset_proxy() {
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
}

wait_triton_ready() {
    local url="http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready"
    local timeout=120
    local elapsed=0
    echo "  等待 Triton 就绪 ($url) ..."
    while (( elapsed < timeout )); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "  Triton 就绪 (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "  ERROR: Triton 启动超时 (${timeout}s)"
    return 1
}

stop_triton() {
    echo "停止已有 Triton 容器..."
    $CONTAINER_CLI ps -q --filter "ancestor=${TRITON_IMAGE}" 2>/dev/null | xargs -r $CONTAINER_CLI stop || true
    # 也停掉本地 tritonserver 进程（如果有）
    pkill -f "tritonserver.*--model-repository" 2>/dev/null || true
    sleep 2
}

start_triton() {
    local repo="$1"
    echo "启动 Triton (model-repo: $repo)..."
    $CONTAINER_CLI run --rm -d \
        --network host \
        -v "$repo:/models" \
        "$TRITON_IMAGE" \
        tritonserver \
        --model-repository=/models \
        --http-port=$TRITON_HTTP_PORT \
        --grpc-port=$TRITON_GRPC_PORT \
        --metrics-port=$TRITON_METRICS_PORT
    wait_triton_ready
}

run_bench() {
    local label="$1"
    local stream="$2"
    local output_dir="${RESULT_BASE}/${label}"

    echo "运行压测: $label (STREAM=$stream, C=$CONCURRENCY, warmup=${WARMUP}s, sample=${SAMPLE}s)..."
    unset_proxy

    env -u http_proxy -u https_proxy -u all_proxy \
        -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    uv run python scripts/bench/run_bench.py \
        --target triton \
        --triton-transport grpc-streaming \
        --triton-grpc-url "127.0.0.1:${TRITON_GRPC_PORT}" \
        --triton-stream "$stream" \
        --triton-model "$MODEL_NAME" \
        --workload "$MODEL_NAME" \
        --concurrency-levels "$CONCURRENCY" \
        --warmup-seconds "$WARMUP" \
        --sample-seconds "$SAMPLE" \
        --max-tokens 256 \
        --output-root "$output_dir"

    echo "--- 结果: $label ---"
    cat "$output_dir"/triton/*/summary.json 2>/dev/null || echo "(no summary.json found)"
    echo ""
}

# --- main ---

echo "============================================"
echo " 验证假设: mm_postprocess 实例数瓶颈"
echo "============================================"
echo ""

REPO_BASELINE="/tmp/triton_verify_baseline"
REPO_FIXED="/tmp/triton_verify_fixed"

# --- 准备两份 model repo ---

echo "[1/5] 生成 model repo (baseline: instance_count=default)..."
uv run python scripts/bench/infra/prepare_triton_repo.py \
    --output "$REPO_BASELINE" \
    --model-name "$MODEL_NAME" \
    --cpu-mock \
    --mock-token-latency-ms 0.5

echo "[2/5] 生成 model repo (fixed: pre-post-instance-count=20)..."
uv run python scripts/bench/infra/prepare_triton_repo.py \
    --output "$REPO_FIXED" \
    --model-name "$MODEL_NAME" \
    --cpu-mock \
    --mock-token-latency-ms 0.5 \
    --pre-post-instance-count 20

echo ""
echo "对比 mm_postprocess config.pbtxt:"
echo "  baseline:"
grep -A2 "instance_group" "$REPO_BASELINE/mm_postprocess/config.pbtxt" 2>/dev/null || echo "    (无 instance_group → 默认 1)"
echo "  fixed:"
grep -A2 "instance_group" "$REPO_FIXED/mm_postprocess/config.pbtxt"
echo ""

# --- A) Baseline: STREAM=true, instance_count=default ---

if [[ "$SKIP_BASELINE" == false ]]; then
    echo "[3/5] Baseline: STREAM=true, postprocess instance_count=default"
    stop_triton
    start_triton "$REPO_BASELINE"
    run_bench "baseline-stream-true" "true"
    stop_triton
else
    echo "[3/5] Baseline: SKIPPED (--skip-baseline)"
fi

# --- B) Fixed: STREAM=true, instance_count=20 ---

echo "[4/5] Fixed: STREAM=true, postprocess instance_count=20"
stop_triton
start_triton "$REPO_FIXED"
run_bench "fixed-stream-true-inst20" "true"

# --- C) 额外验证: STREAM=false 也跑一下作为 sanity check ---

echo "[5/5] Sanity: STREAM=false (应与 baseline stream=false 结果一致)"
run_bench "fixed-stream-false-inst20" "false"
stop_triton

# --- 汇总 ---

echo ""
echo "============================================"
echo " 汇总"
echo "============================================"
echo ""
echo "期望结果:"
echo "  baseline STREAM=true:   p50 ~2100ms, QPS ~15  (postprocess 是瓶颈)"
echo "  fixed    STREAM=true:   p50 ~130ms,  QPS ~200 (瓶颈消除)"
echo "  fixed    STREAM=false:  p50 ~130ms,  QPS ~240 (无变化，sanity check)"
echo ""
echo "所有结果保存在: $RESULT_BASE/"

# 尝试用 jq 格式化输出
if command -v jq &>/dev/null; then
    echo ""
    echo "--- 对比表 ---"
    for d in "$RESULT_BASE"/*/triton/*/summary.json; do
        label=$(echo "$d" | sed "s|$RESULT_BASE/||" | cut -d/ -f1)
        echo -n "$label: "
        jq -r '"QPS=\(.qps | tostring | .[0:7])  p50=\(.p50_ms | tostring | .[0:8])ms  err=\(.error_rate)"' "$d" 2>/dev/null || echo "(parse error)"
    done
fi
