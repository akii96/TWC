# TWC - Together We Check

A collection of tools for testing and validating LLM serving infrastructure.

## Tools

| Directory | Description |
|-----------|-------------|
| [`model-sanity/`](model-sanity/) | Multi-model vLLM sanity checker — validates model serving across `v1/chat/completions` and `v1/completions` endpoints |
| [`n-run-stability/`](n-run-stability/) | SGLang stress tester — repeatedly launches and tests a model to catch intermittent failures |


## Quick Start

```bash
# Model sanity check (vLLM)
cd model-sanity
export HF_TOKEN='hf_your_token_here'
./run_sanity_check.sh

# Stability stress test (SGLang)
cd n-run-stability
./stress_test_sglang.sh <docker_image> <num_loops>
```

See individual tool READMEs for detailed usage.
