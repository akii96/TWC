# TWC - Together We Check

A collection of tools for testing and validating LLM serving infrastructure.

## Tools

| Directory | Description | vLLM | SGLang |
|-----------|-------------|:----:|:------:|
| [`model-sanity/`](model-sanity/) | Multi-model sanity checker — validates model serving across `v1/chat/completions` and `v1/completions` endpoints | :white_check_mark: | :x: |
| [`n-run-stability/`](n-run-stability/) | Stress tester — repeatedly launches and tests a model to catch intermittent failures | :x: | :white_check_mark: |


## Quick Start

```bash
# Model sanity check (vLLM)
cd model-sanity
export HF_TOKEN='hf_your_token_here'
./run_sanity_check.sh

# Stability stress test (SGLang)
cd n-run-stability
export HF_TOKEN='hf_your_token_here'
./stress_test_sglang.sh <docker_image> <num_loops>
```

See individual tool READMEs for detailed usage.
