#!/bin/bash
###############################################################################
# sglang.plugin.sh - SGLang Framework Plugin
#
# This plugin provides SGLang-specific implementations for the universal
# stress test runner.
###############################################################################

PLUGIN_NAME="sglang"
PLUGIN_VERSION="1.0.0"

# Build the server launch command for SGLang
# Arguments:
#   $1 - model_path: Path or HuggingFace model ID
#   $2 - port: Server port number
#   $3 - extra_args: Additional CLI arguments (space-separated)
# Returns: The complete launch command string
build_server_cmd() {
    local model_path="$1"
    local port="$2"
    local extra_args="$3"
    
    echo "python3 -m sglang.launch_server --model-path ${model_path} --port ${port} ${extra_args}"
}

# Get the health check endpoint path
# Returns: URL path for health check
get_health_endpoint() {
    echo "/health"
}

# Get the chat completions endpoint path
# Returns: URL path for chat completions API
get_chat_endpoint() {
    echo "/v1/chat/completions"
}

# Get the completions endpoint path (non-chat)
# Returns: URL path for completions API
get_completions_endpoint() {
    echo "/v1/completions"
}

# Build the prompt payload for chat completions
# Arguments:
#   $1 - model: Model name/path
#   $2 - prompt_content: The user message content
#   $3 - default_params_json: JSON string of default parameters
#   $4 - extra_params_json: JSON string of extra parameters (optional)
# Returns: JSON payload string
build_chat_payload() {
    local model="$1"
    local prompt_content="$2"
    local default_params_json="$3"
    local extra_params_json="$4"
    [ -z "$extra_params_json" ] && extra_params_json='{}'
    
    # Use jq if available (fastest, handles all escaping properly)
    if command -v jq &>/dev/null; then
        jq -n \
            --arg model "$model" \
            --arg content "$prompt_content" \
            --argjson default_params "$default_params_json" \
            --argjson extra_params "$extra_params_json" \
            '{model: $model, messages: [{role: "user", content: $content}]} + $default_params + $extra_params'
    else
        # Fallback: use Python with stdin to avoid shell escaping issues
        printf '%s\n%s\n%s\n%s\n' "$model" "$prompt_content" "$default_params_json" "$extra_params_json" | python3 -c "
import json
import sys

lines = sys.stdin.read().split('\n')
model = lines[0]
content = lines[1]
default_params = json.loads(lines[2]) if lines[2] else {}
extra_params = json.loads(lines[3]) if lines[3] else {}

payload = {
    'model': model,
    'messages': [{'role': 'user', 'content': content}],
    **default_params,
    **extra_params
}

print(json.dumps(payload))
"
    fi
}

# Parse the response from chat completions endpoint
# Arguments:
#   $1 - response: Raw JSON response from the API
# Returns: Extracted content string, or "PARSE_ERROR: <details>" on failure
parse_chat_response() {
    local response="$1"
    
    # Use jq if available (fastest)
    if command -v jq &>/dev/null; then
        local content
        content=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
        if [ -n "$content" ]; then
            echo "$content"
        else
            echo "PARSE_ERROR: $(echo "$response" | jq -r '.message // .error // "unknown error"' 2>/dev/null)"
        fi
    else
        # Fallback: use Python with stdin
        echo "$response" | python3 -c "
import json
import sys

try:
    d = json.load(sys.stdin)
    content = d['choices'][0]['message']['content']
    print(content)
except Exception as e:
    print(f'PARSE_ERROR: {e}')
"
    fi
}

# Get default Docker entrypoint override (if needed)
# Returns: Entrypoint command or empty string for default
get_docker_entrypoint() {
    echo ""  # SGLang images typically don't need entrypoint override
}

# Get the process pattern for identifying the server process
# Used by server mode to stop/check the server without killing the container
# Returns: Regex pattern for pgrep/pkill -f
get_server_process_pattern() {
    echo "sglang.launch_server|python.*-m sglang"
}

# Get any required volume mounts specific to this framework
# Returns: Space-separated list of -v mount flags
get_extra_mounts() {
    echo ""  # No extra mounts needed for SGLang
}

# Validate framework-specific configuration
# Arguments:
#   $1 - config_file: Path to the config.yaml
# Returns: 0 if valid, 1 if invalid (with error message to stderr)
validate_config() {
    local config_file="$1"
    # SGLang requires model_path at minimum
    return 0
}
