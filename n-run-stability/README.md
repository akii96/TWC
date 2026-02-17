# N-Run Stability - SGLang Stress Test

Stress-tests an SGLang model serving inside Docker to catch hangs during CUDA graph build or decode. Runs multiple iterations of container launch → server startup → prompt requests to identify intermittent failures.

## Quick Start

```bash
export HF_TOKEN='hf_your_token_here'
./stress_test_sglang.sh <docker_image> <num_loops>

# Example
./stress_test_sglang.sh lmsysorg/sglang:v0.5.8-rocm700-mi30x 20
```

## Prerequisites

- Docker with GPU support (ROCm)
- `HF_TOKEN` environment variable set (get one at https://huggingface.co/settings/tokens)

## What It Does

For each iteration:

1. Launches a Docker container running SGLang server
2. Waits for the server health endpoint to become ready
3. Sends 10 test prompts via `v1/chat/completions`
4. Checks responses contain expected content ("yes")
5. Scans logs for HSA errors
6. Tears down the container and records results

## Configuration

Key variables at the top of `stress_test_sglang.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER_TIMEOUT` | 900s (15 min) | Hard timeout per container |
| `SERVER_STARTUP_TIMEOUT` | 600s (10 min) | Max wait for server readiness |
| `PROMPT_TIMEOUT` | 120s (2 min) | Max wait per prompt request |
| `SERVER_PORT` | 30000 | Port for SGLang server |

## Model & Server Config

The model and SGLang launch arguments are defined in `SERVER_CMD` inside the script. Current defaults:

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp-size 8 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --mem-fraction-static 0.8 \
  ...
```

Edit `SERVER_CMD` to change the model or serving parameters.

## Output

Each run creates a timestamped directory:

```
sglang_stress_<image_slug>_<timestamp>/
├── summary.log              # Overall run summary
├── iter_1_<ts>_SUCCESS.log  # Per-iteration container logs
├── iter_2_<ts>_FAIL.log
└── ...
```

## Success Criteria

An iteration is marked **SUCCESS** if:

- Server becomes ready within timeout
- All 10 prompts return responses
- All responses contain "yes" (case-insensitive)
- No `HSA_STATUS_ERROR_EXCEPTION` in logs

Otherwise, it's marked **FAIL**.

## Exit Code

- `0` — All iterations passed
- `1` — At least one iteration failed
