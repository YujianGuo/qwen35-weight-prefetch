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

## Provenance

Source was collected from `qwen35_weight1005` at the base commit and environment recorded in [`TARGET.md`](TARGET.md). The overlay targets vLLM-Ascend and follows its Apache-2.0 licensing; see [`LICENSE`](LICENSE).

