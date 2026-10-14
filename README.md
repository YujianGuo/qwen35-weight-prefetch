# Qwen3.5 weight prefetch for vLLM-Ascend

This repository packages the experimental Qwen3.5 decode weight-prefetch implementation extracted from container `qwen35_weight1005`. It contains the runtime controller, validated configuration, the optional route-aware Ascend C extension, a kernel build script, and a focused integration patch for the source revision listed in [`TARGET.md`](TARGET.md).

The implementation adds three optional paths:

- `next_input`: prefetch the next layer's fused GDN input or full-attention QKV weight while the existing tensor-parallel all-reduce is queued.
- `routed_w2`: use the current expert routing counts to warm a bounded portion of active experts' `w2` weights before the down projection.
- `gdn_out`: prefetch the current GDN layer's output projection during its attention work.

All paths are disabled by default. They bind to post-loader weight storage and are intended only for the supported static-weight Qwen3.5 BF16 TP2 configuration on Ascend A2/910B. See the capability checks in `src/vllm_ascend/ops/weight_prefetch_qwen35.py` before adapting the code to another model or topology.

## Layout

```text
src/
  vllm_ascend/ops/weight_prefetch_config.py
  vllm_ascend/ops/weight_prefetch_qwen35.py
  csrc/qwen35_weight_prefetch/
    CMakeLists.txt
    extension.asc
patches/
  vllm-ascend-v0.26.0rc1.patch
scripts/
  build_kernel.sh
```

The patch contains only Qwen3.5 prefetch integration hunks. It excludes the separate GMM2 experiment and the container's unrelated build/debug artifacts. The compiled `_qwen35_routed_prefetch` shared library is intentionally not committed; build it for the target architecture with the supplied script.

## Integrate with the target source tree

Start from the exact vLLM-Ascend base commit in `TARGET.md`. Copy the packaged source files into the matching source tree, then apply the integration patch from that tree:

```bash
cp -a src/vllm_ascend/ops/weight_prefetch_*.py /path/to/vllm-ascend/vllm_ascend/ops/
cp -a src/csrc/qwen35_weight_prefetch /path/to/vllm-ascend/csrc/
cd /path/to/vllm-ascend
git apply /path/to/qwen35-weight-prefetch/patches/vllm-ascend-v0.26.0rc1.patch
```

The `routed_w2` target requires the Ascend C extension. Build and install it inside the target container after stopping inference services:

```bash
VLLM_ASCEND_ROOT=/path/to/vllm-ascend \
ASCEND_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh \
bash /path/to/qwen35-weight-prefetch/scripts/build_kernel.sh
```

The script checks for a running vLLM service, compiles the extension for the current A2 host, and places the resulting shared library in `vllm_ascend/ops/`.

## Configuration

Pass the object through vLLM's `--additional-config`. The default and safe state is disabled:

```json
{
  "qwen35_weight_prefetch": {
    "enabled": false
  }
}
```

An isolated experiment can select targets and budgets, for example:

```json
{
  "qwen35_weight_prefetch": {
    "enabled": true,
    "mode": "full",
    "targets": ["next_input"],
    "next_input_bytes": "full",
    "next_layer_types": ["linear_attention"],
    "min_decode_concurrency": 8
  }
}
```

`mode` accepts `full` or `control`; `targets` accepts `next_input`, `routed_w2`, and `gdn_out`. Each target has a separate byte budget. Weight reload, sleep mode, and weight transfer are unsupported while prefetch is enabled; restart with prefetch disabled before reloading weights. A captured graph does not pick up configuration changes, so restart the service and use a fresh compile-cache directory when changing settings.

This repository is a code package, not a performance acceptance report. Measure latency, tail latency, output consistency, and device-side consumer waits on the intended workload before enabling the feature for service traffic.

## RealWorldQA performance snapshot

The following descriptive comparison is transcribed from the supplied RealWorldQA summary CSVs: `base/base_realworldqa.csv` and `weight_gate8_2atten/realworldqa.csv`. Both report 765 requests. The `weight_gate8_2atten` string is kept as the input directory label; its exact runtime configuration is not specified by these CSVs.

For latency rows, a negative change means lower latency; for throughput rows, a positive change means higher throughput.

| Metric | Base | `weight_gate8_2atten` | Change |
|---|---:|---:|---:|
| E2EL mean | 18,513.3 ms | 17,728.6 ms | -4.24% |
| E2EL median | 14,981.4 ms | 13,737.9 ms | -8.30% |
| E2EL P90 | 33,340.5 ms | 35,176.1 ms | +5.51% (slower) |
| E2EL P99 | 60,003.9 ms | 56,650.2 ms | -5.59% |
| TTFT mean | 1,822.0 ms | 1,645.2 ms | -9.70% |
| TTFT median | 1,548.6 ms | 1,444.5 ms | -6.72% |
| TTFT P99 | 8,115.3 ms | 6,894.2 ms | -15.05% |
| TPOT mean | 103.9 ms | 98.2 ms | -5.49% |
| TPOT median | 104.7 ms | 99.0 ms | -5.44% |
| TPOT P90 | 124.6 ms | 115.1 ms | -7.62% |
| TPOT P99 | 140.8 ms | 133.3 ms | -5.33% |
| Output-token throughput mean | 8.7554 token/s | 9.2938 token/s | +6.15% |
| Prefill-token throughput mean | 966.4794 token/s | 1,060.1459 token/s | +9.69% |

Both CSVs report the same input-token summary (mean 1,448.4562; median 1,443; P75 1,464; P90 1,901; P99 1,927.8). Their output-token summaries differ slightly (mean 162.5516 vs. 164.9333; median 128 vs. 126).

These are aggregate summaries, not a controlled A/B acceptance result: the files do not include per-request pairing, alternating-session order or drift, output hashes/accuracy, model and service revisions, the exact variant configuration, or P95. In particular, E2EL P90 regresses even though its median and P99 improve. The CSVs therefore support reporting the observed deltas only; they do not establish output equivalence or isolate prefetch as the cause.

## Provenance

Source was collected from `qwen35_weight1005` at the base commit and environment recorded in [`TARGET.md`](TARGET.md). The overlay targets vLLM-Ascend and follows its Apache-2.0 licensing; see [`LICENSE`](LICENSE).

## Co-authors

- westbook16 <1210973614@qq.com>
