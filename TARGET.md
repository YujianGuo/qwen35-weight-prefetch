# Source target

- Container: `qwen35_weight1005`
- vLLM-Ascend base commit: `f2f74a16c3c50a76f4349d807918e83edec1e35c` (`v0.26.0rc1`)
- Base commit date: `2026-09-03`
- Model family: Qwen3.5-35B-A3B, 40 layers
- Device target: Ascend A2 / 910B3, BF16, TP2, DP1, PP1, expert parallelism disabled
- Runtime packages observed in the container: `torch==2.10.0+cpu`, `torch_npu==2.10.0.post4`
- Ascend environment script: `/usr/local/Ascend/ascend-toolkit/set_env.sh` (symlink to `latest/set_env.sh`)
- Source collection date: `2026-10-13`

The container working tree also contained a separate Qwen3.5 GMM2 experiment and generated compiler/debug files. Those changes are not part of this repository. The package does not include the container-built `.so`; build the routed prefetch extension on the target host.
