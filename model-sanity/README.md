# Together We Check - Model Sanity

Launches a Docker container per model, serves it with vLLM, sends test prompts via both `v1/chat/completions` and `v1/completions`, and records everything to a timestamped CSV.

## Quick Start

```bash
# Default image (vllm/vllm-openai-rocm:v0.15.0)
./run_sanity_check.sh

# Custom image
./run_sanity_check.sh my-registry/vllm:custom-tag
```

## Adding / Removing Models

Edit `models.txt`. One model per line. Lines starting with `#` are ignored.

## Adding / Changing Environment Variables

Edit `envs.txt`. One `KEY=VALUE` per line. These are passed as `-e` flags to `docker run` and set inside every container. Example:

```
VLLM_ROCM_USE_AITER=1
VLLM_USE_V1=1
```

## Modifying Serving Args

The vLLM serve command and its flags are defined inside `run_sanity_check.sh` in the launch-script heredoc (search for `cat > "$LAUNCH_SCRIPT"`). The current template is:

```
vllm serve <model> \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --no-enable-prefix-caching \
    --quantization fp8 \
    --chat-template /tmp/fallback_chat_template.jinja \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

Edit that block to add, remove, or change any serving flags.

## Files

| File | Purpose |
|------|---------|
| `models.txt` | One HuggingFace model name per line (e.g. `Qwen/Qwen1.5-1.8B`) |
| `envs.txt` | Environment variables injected into every container, one `KEY=VALUE` per line |
| `run_sanity_check.sh` | Main script |
| `fallback_chat_template.jinja` | Default chat template for models that don't define one |

## Output

Each run produces:

- `sanity_check_results_<timestamp>.csv` -- one row per prompt/response with columns: `timestamp`, `docker_image`, `env_vars`, `serving_args`, `model`, `endpoint`, `serve_launch_status`, `prompt`, `response`, `status`
- `logs_<timestamp>/` -- per-model container logs and a `summary.log`

`serve_launch_status` is `1` if the vLLM server started successfully, `0` if it timed out or crashed.
