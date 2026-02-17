#!/bin/bash
###############################################################################
# stress_test.sh - Universal LLM Serving Stress Test
#
# A framework-agnostic stress testing tool for LLM serving infrastructure.
# Supports SGLang, vLLM, and other frameworks through a plugin system.
#
# Usage:
#   ./stress_test.sh --config presets/<preset>.yaml [OPTIONS]
#
# Options:
#   --config FILE       Path to preset file (REQUIRED)
#   --loops N           Override number of test loops
#   --image IMAGE       Override Docker image
#   --port PORT         Override server port
#   --framework NAME    Override framework (sglang, vllm, etc.)
#   --mode MODE         Test mode: 'container' (restart container each loop) or
#                       'server' (keep container, restart server each loop)
#   --dry-run           Show configuration without running tests
#   --help              Show this help message
#
# Environment variable overrides (prefix with STRESS_):
#   STRESS_LOOPS, STRESS_IMAGE, STRESS_PORT, STRESS_FRAMEWORK, STRESS_MODE
#
# Precedence (highest to lowest):
#   1. CLI arguments (--loops, --image, etc.)
#   2. STRESS_* environment variables
#   3. Preset file values
#
# Examples:
#   ./stress_test.sh --config presets/sglang-glm4-rocm.yaml
#   ./stress_test.sh --config presets/sglang-glm4-rocm.yaml --mode server
#   ./stress_test.sh --config presets/sglang-glm4-rocm.yaml --loops 50
#   STRESS_LOOPS=10 ./stress_test.sh --config presets/sglang-glm4-rocm.yaml
###############################################################################

set -uo pipefail

# ── Python interpreter ───────────────────────────────────────────────────────
# Use system Python with PyYAML (prefer /usr/bin/python3 if available)
if /usr/bin/python3 -c "import yaml" 2>/dev/null; then
    PYTHON="/usr/bin/python3"
elif python3 -c "import yaml" 2>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: PyYAML not found. Install with: pip3 install PyYAML" >&2
    exit 1
fi

# ── Script directory ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Default paths ────────────────────────────────────────────────────────────
PRESETS_DIR="$SCRIPT_DIR/presets"
DEFAULT_PROMPTS="$SCRIPT_DIR/prompts.json"
PLUGINS_DIR="$SCRIPT_DIR/plugins"

# ── CLI argument defaults ────────────────────────────────────────────────────
CONFIG_FILE=""
CLI_LOOPS=""
CLI_IMAGE=""
CLI_PORT=""
CLI_FRAMEWORK=""
CLI_MODE=""
DRY_RUN=false

# ── Colors for output ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

###############################################################################
# HELPER FUNCTIONS
###############################################################################

show_help() {
    head -50 "$0" | grep -E '^#' | sed 's/^# \?//'
    exit 0
}

die() {
    echo -e "${RED}ERROR:${NC} $*" >&2
    exit 1
}

warn() {
    echo -e "${YELLOW}WARNING:${NC} $*" >&2
}

info() {
    echo -e "${BLUE}INFO:${NC} $*"
}

# Check for required dependencies
check_dependencies() {
    local missing=()
    
    # Check for Python (required for YAML/JSON parsing)
    if ! command -v python3 &>/dev/null; then
        missing+=("python3")
    fi
    
    # Check for jq (optional but recommended)
    if ! command -v jq &>/dev/null; then
        warn "jq not found. Using Python for JSON parsing (slower)."
    fi
    
    # Check for docker
    if ! command -v docker &>/dev/null; then
        missing+=("docker")
    fi
    
    # Check for curl
    if ! command -v curl &>/dev/null; then
        missing+=("curl")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        die "Missing required dependencies: ${missing[*]}"
    fi
}

# Parse YAML using Python with PyYAML
yaml_get() {
    local file="$1"
    local key="$2"
    local default="${3:-}"
    
    $PYTHON -c "
import yaml
import sys
import json

def get_nested(d, keys):
    for key in keys.split('.'):
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d

try:
    with open('$file', 'r') as f:
        data = yaml.safe_load(f)
    result = get_nested(data, '$key')
    if result is None:
        print('$default')
    elif isinstance(result, list):
        print('\\n'.join(str(x) for x in result))
    elif isinstance(result, dict):
        print(json.dumps(result))
    else:
        print(result)
except Exception as e:
    print('$default', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || echo "$default"
}

# Get YAML list as array
yaml_get_list() {
    local file="$1"
    local key="$2"
    
    $PYTHON -c "
import yaml
import sys

def get_nested(d, keys):
    for key in keys.split('.'):
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return []
    return d if isinstance(d, list) else []

try:
    with open('$file', 'r') as f:
        data = yaml.safe_load(f)
    result = get_nested(data, '$key')
    for item in result:
        print(item)
except:
    pass
"
}

# Convert server_args dict to CLI flags
yaml_args_to_flags() {
    local file="$1"
    local key="$2"
    
    $PYTHON -c "
import yaml
import json

def get_nested(d, keys):
    for key in keys.split('.'):
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return {}
    return d if isinstance(d, dict) else {}

try:
    with open('$file', 'r') as f:
        data = yaml.safe_load(f)
    args = get_nested(data, '$key')
    flags = []
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                flags.append(f'--{k}')
        elif isinstance(v, dict):
            flags.append(f'--{k}')
            flags.append(\"'\" + json.dumps(v) + \"'\")
        else:
            flags.append(f'--{k}')
            flags.append(str(v))
    print(' '.join(flags))
except:
    pass
"
}

# Read JSON prompts file
json_get() {
    local file="$1"
    local key="$2"
    local default="${3:-}"
    
    if command -v jq &>/dev/null; then
        jq -r "$key // \"$default\"" "$file" 2>/dev/null || echo "$default"
    else
        $PYTHON -c "
import json
import sys

try:
    with open('$file', 'r') as f:
        data = json.load(f)
    # Simple key access (doesn't support full jq syntax)
    keys = '$key'.strip('.').split('.')
    result = data
    for k in keys:
        if k and isinstance(result, dict):
            result = result.get(k)
    if result is None:
        print('$default')
    elif isinstance(result, (dict, list)):
        print(json.dumps(result))
    else:
        print(result)
except:
    print('$default')
"
    fi
}

# Build environment flags from YAML config env section
build_env_flags_from_yaml() {
    local config_file="$1"
    
    $PYTHON -c "
import yaml

try:
    with open('$config_file', 'r') as f:
        data = yaml.safe_load(f)
    env_vars = data.get('env', {})
    if env_vars:
        flags = []
        for k, v in env_vars.items():
            flags.append(f'-e {k}={v}')
        print(' '.join(flags))
except:
    pass
"
}

# Log to both stdout and summary file
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    [ -n "${SUMMARY_LOG:-}" ] && echo "$msg" >> "$SUMMARY_LOG"
}

# Kill container and watchdog
cleanup_container() {
    local name="$1"
    if docker ps -q -f name="$name" 2>/dev/null | grep -q .; then
        log "  Stopping container $name ..."
        docker rm -f "$name" >/dev/null 2>&1 || true
    fi
    if [ -n "${WATCHDOG_PID:-}" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    if [ -n "${LOGS_PID:-}" ] && kill -0 "$LOGS_PID" 2>/dev/null; then
        kill "$LOGS_PID" 2>/dev/null || true
        wait "$LOGS_PID" 2>/dev/null || true
    fi
}

###############################################################################
# SERVER MODE FUNCTIONS (for --mode server)
###############################################################################

# Start a persistent container that stays alive via sleep infinity
# The server will be started separately via docker exec
start_persistent_container() {
    local name="$1"
    local iter_log="$2"
    
    # Clean up any leftover container
    docker rm -f "$name" >/dev/null 2>&1 || true
    
    # Build docker run command (same as container mode but with sleep infinity)
    local DOCKER_CMD=(
        docker run
        -d
        --cap-add=SYS_PTRACE
        --cap-add=CAP_SYS_ADMIN
        --security-opt seccomp=unconfined
        --user root
        --ulimit memlock=999332768:999332768
        --ipc=host
        --name "$name"
        --shm-size="$SHM_SIZE"
        --hostname "STRESS-$(echo "${HOSTNAME:-$(hostname)}" | cut -f 1 -d .)"
        --network "$NETWORK_MODE"
    )
    
    # Add devices
    for device in "${DOCKER_DEVICES[@]}"; do
        DOCKER_CMD+=(--device="$device")
    done
    
    # Add group for video (AMD GPUs)
    if [[ " ${DOCKER_DEVICES[*]} " =~ "/dev/kfd" ]]; then
        DOCKER_CMD+=(--group-add video)
    fi
    
    # Add workspace mounts
    DOCKER_CMD+=(-v "$WORKSPACE_BASE:/workspace/")
    for mount in "${WORKSPACE_MOUNTS[@]}"; do
        DOCKER_CMD+=(-v "$WORKSPACE_BASE/$mount:/workspace/$mount")
    done
    
    # Add HF token and home
    DOCKER_CMD+=(
        -e "HF_HOME=/workspace/.cache/huggingface"
        -e "HF_TOKEN=$HF_TOKEN"
    )
    
    # Add environment flags from config
    if [ -n "$ENV_FLAGS" ]; then
        # shellcheck disable=SC2206
        DOCKER_CMD+=($ENV_FLAGS)
    fi
    
    DOCKER_CMD+=(--workdir /workspace/)
    
    # Add entrypoint override if plugin specifies one
    if [ -n "$DOCKER_ENTRYPOINT" ]; then
        DOCKER_CMD+=(--entrypoint "$DOCKER_ENTRYPOINT")
    fi
    
    DOCKER_CMD+=("$DOCKER_IMAGE")
    # Use sleep infinity to keep container alive
    DOCKER_CMD+=(bash -c "sleep infinity")
    
    # Launch container
    log "  Starting persistent Docker container ..."
    "${DOCKER_CMD[@]}" 2>&1 | tee -a "$iter_log"
    
    # Verify container started
    sleep 2
    if ! docker ps -q -f name="$name" 2>/dev/null | grep -q .; then
        log "  FAIL: Persistent container failed to start."
        return 1
    fi
    
    log "  Persistent container started successfully."
    return 0
}

# Start the server process inside an already-running container
start_server_in_container() {
    local name="$1"
    local server_cmd="$2"
    local iter_log="$3"
    
    log "  Starting server inside container ..."
    
    # Start server in background inside container, redirect output to a log file
    docker exec -d "$name" bash -c "$server_cmd > /tmp/server.log 2>&1 &"
    
    # Give the server a moment to start
    sleep 2
    
    # Check if server process is running
    if is_server_running "$name"; then
        log "  Server process started."
        return 0
    else
        log "  FAIL: Server process failed to start."
        return 1
    fi
}

# Stop the server process inside the container (container stays alive)
stop_server_in_container() {
    local name="$1"
    
    log "  Stopping server inside container ..."
    
    # Get the process pattern from plugin (if available)
    local pattern
    pattern=$(get_server_process_pattern 2>/dev/null || echo "")
    
    if [ -n "$pattern" ]; then
        # Use plugin-provided pattern
        docker exec "$name" pkill -f "$pattern" 2>/dev/null || true
    else
        # Fallback: kill common LLM server processes
        docker exec "$name" pkill -f "sglang.launch_server|vllm serve|python.*-m sglang|python.*-m vllm" 2>/dev/null || true
    fi
    
    # Wait for server to stop (up to 30 seconds)
    local wait_count=0
    while [ $wait_count -lt 30 ]; do
        if ! is_server_running "$name"; then
            log "  Server stopped."
            return 0
        fi
        sleep 1
        wait_count=$(( wait_count + 1 ))
    done
    
    # Force kill if still running
    log "  Server did not stop gracefully, force killing ..."
    if [ -n "$pattern" ]; then
        docker exec "$name" pkill -9 -f "$pattern" 2>/dev/null || true
    else
        docker exec "$name" pkill -9 -f "sglang.launch_server|vllm serve|python.*-m sglang|python.*-m vllm" 2>/dev/null || true
    fi
    
    sleep 2
    if ! is_server_running "$name"; then
        log "  Server force killed."
        return 0
    fi
    
    log "  WARNING: Could not stop server process."
    return 1
}

# Check if server is running inside container
is_server_running() {
    local name="$1"
    
    # Get the process pattern from plugin (if available)
    local pattern
    pattern=$(get_server_process_pattern 2>/dev/null || echo "")
    
    if [ -n "$pattern" ]; then
        docker exec "$name" pgrep -f "$pattern" >/dev/null 2>&1
    else
        # Fallback: check for common LLM server processes
        docker exec "$name" pgrep -f "sglang.launch_server|vllm serve|python.*-m sglang|python.*-m vllm" >/dev/null 2>&1
    fi
}

# Capture server logs from inside container
capture_server_logs() {
    local name="$1"
    local iter_log="$2"
    
    # Append server logs to iteration log
    docker exec "$name" cat /tmp/server.log >> "$iter_log" 2>/dev/null || true
}

###############################################################################
# ARGUMENT PARSING
###############################################################################

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --loops)
                CLI_LOOPS="$2"
                shift 2
                ;;
            --image)
                CLI_IMAGE="$2"
                shift 2
                ;;
            --port)
                CLI_PORT="$2"
                shift 2
                ;;
            --framework)
                CLI_FRAMEWORK="$2"
                shift 2
                ;;
            --mode)
                CLI_MODE="$2"
                if [[ "$CLI_MODE" != "container" && "$CLI_MODE" != "server" ]]; then
                    die "Invalid mode '$CLI_MODE'. Must be 'container' or 'server'."
                fi
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                die "Unknown option: $1. Use --help for usage."
                ;;
        esac
    done
}

###############################################################################
# CONFIGURATION LOADING
###############################################################################

load_config() {
    # Require a preset/config file
    if [ -z "$CONFIG_FILE" ]; then
        echo ""
        echo -e "${RED}ERROR:${NC} No preset specified."
        echo ""
        echo "  Usage:"
        echo "    ./stress_test.sh --config presets/<preset>.yaml [OPTIONS]"
        echo ""
        echo "  Available presets:"
        for preset in "$PRESETS_DIR"/*.yaml; do
            [ -f "$preset" ] && echo "    - presets/$(basename "$preset")"
        done
        echo ""
        echo "  Example:"
        echo "    ./stress_test.sh --config presets/sglang-glm4-rocm.yaml"
        echo "    ./stress_test.sh --config presets/sglang-glm4-rocm.yaml --mode server"
        echo ""
        exit 1
    fi
    
    [ -f "$CONFIG_FILE" ] || die "Config file not found: $CONFIG_FILE"
    
    info "Loading configuration from: $CONFIG_FILE"
    
    # Load base config from YAML
    FRAMEWORK=$(yaml_get "$CONFIG_FILE" "framework" "sglang")
    DOCKER_IMAGE=$(yaml_get "$CONFIG_FILE" "docker.image" "")
    SHM_SIZE=$(yaml_get "$CONFIG_FILE" "docker.shm_size" "128G")
    NETWORK_MODE=$(yaml_get "$CONFIG_FILE" "docker.network" "host")
    SERVER_PORT=$(yaml_get "$CONFIG_FILE" "server.port" "30000")
    SERVER_STARTUP_TIMEOUT=$(yaml_get "$CONFIG_FILE" "server.startup_timeout" "600")
    MODEL_PATH=$(yaml_get "$CONFIG_FILE" "server.model_path" "")
    CONTAINER_TIMEOUT=$(yaml_get "$CONFIG_FILE" "timeouts.container" "900")
    PROMPT_TIMEOUT=$(yaml_get "$CONFIG_FILE" "timeouts.prompt" "120")
    NUM_LOOPS=$(yaml_get "$CONFIG_FILE" "test.num_loops" "20")
    PROMPTS_PER_LOOP=$(yaml_get "$CONFIG_FILE" "test.prompts_per_loop" "10")
    SUCCESS_PATTERN=$(yaml_get "$CONFIG_FILE" "test.success_pattern" "")
    TEST_MODE=$(yaml_get "$CONFIG_FILE" "test.mode" "container")
    WORKSPACE_BASE=$(yaml_get "$CONFIG_FILE" "workspace.base_dir" "\$HOME")
    
    # Expand $HOME in workspace base
    WORKSPACE_BASE="${WORKSPACE_BASE/\$HOME/$HOME}"
    
    # Get server args as CLI flags
    SERVER_ARGS=$(yaml_args_to_flags "$CONFIG_FILE" "server_args")
    
    # Get device list
    mapfile -t DOCKER_DEVICES < <(yaml_get_list "$CONFIG_FILE" "docker.devices")
    
    # Get error patterns
    mapfile -t ERROR_PATTERNS < <(yaml_get_list "$CONFIG_FILE" "error_patterns")
    
    # Get workspace mounts
    mapfile -t WORKSPACE_MOUNTS < <(yaml_get_list "$CONFIG_FILE" "workspace.mounts")
    
    # Apply environment variable overrides (STRESS_*)
    [ -n "${STRESS_LOOPS:-}" ] && NUM_LOOPS="$STRESS_LOOPS"
    [ -n "${STRESS_IMAGE:-}" ] && DOCKER_IMAGE="$STRESS_IMAGE"
    [ -n "${STRESS_PORT:-}" ] && SERVER_PORT="$STRESS_PORT"
    [ -n "${STRESS_FRAMEWORK:-}" ] && FRAMEWORK="$STRESS_FRAMEWORK"
    [ -n "${STRESS_MODE:-}" ] && TEST_MODE="$STRESS_MODE"
    
    # Apply CLI overrides (highest priority)
    [ -n "$CLI_LOOPS" ] && NUM_LOOPS="$CLI_LOOPS"
    [ -n "$CLI_IMAGE" ] && DOCKER_IMAGE="$CLI_IMAGE"
    [ -n "$CLI_PORT" ] && SERVER_PORT="$CLI_PORT"
    [ -n "$CLI_FRAMEWORK" ] && FRAMEWORK="$CLI_FRAMEWORK"
    [ -n "$CLI_MODE" ] && TEST_MODE="$CLI_MODE"
    
    # Validate test mode
    if [[ "$TEST_MODE" != "container" && "$TEST_MODE" != "server" ]]; then
        die "Invalid test mode '$TEST_MODE'. Must be 'container' or 'server'."
    fi
    
    # Validate required fields
    [ -z "$DOCKER_IMAGE" ] && die "Docker image not specified. Set docker.image in config or use --image"
    [ -z "$MODEL_PATH" ] && die "Model path not specified. Set server.model_path in config"
    
    # Load prompts configuration
    PROMPTS_FILE="$DEFAULT_PROMPTS"
    if [ -f "$PROMPTS_FILE" ]; then
        DEFAULT_PARAMS=$(json_get "$PROMPTS_FILE" ".default_params" "{}")
        EXTRA_PARAMS=$(json_get "$PROMPTS_FILE" ".extra_params" "{}")
        PROMPT_CONTENT=$(json_get "$PROMPTS_FILE" ".prompts[0].content" "Hello, how are you?")
    else
        warn "Prompts file not found: $PROMPTS_FILE. Using defaults."
        DEFAULT_PARAMS='{"stream": false, "max_tokens": 512}'
        EXTRA_PARAMS='{}'
        PROMPT_CONTENT="Hello, how are you?"
    fi
    
    # Load environment variables from config
    ENV_FLAGS=$(build_env_flags_from_yaml "$CONFIG_FILE")
}

###############################################################################
# PLUGIN LOADING
###############################################################################

load_plugin() {
    local plugin_file="$PLUGINS_DIR/${FRAMEWORK}.plugin.sh"
    
    if [ ! -f "$plugin_file" ]; then
        die "Plugin not found for framework '$FRAMEWORK': $plugin_file"
    fi
    
    info "Loading plugin: $plugin_file"
    # shellcheck source=/dev/null
    source "$plugin_file"
    
    # Verify required functions exist
    for func in build_server_cmd get_health_endpoint get_chat_endpoint parse_chat_response build_chat_payload; do
        if ! declare -f "$func" &>/dev/null; then
            die "Plugin '$FRAMEWORK' missing required function: $func"
        fi
    done
}

###############################################################################
# DRY RUN OUTPUT
###############################################################################

show_dry_run() {
    echo ""
    echo "============================================================"
    echo "  DRY RUN - Configuration Summary"
    echo "============================================================"
    echo ""
    echo "Framework:        $FRAMEWORK"
    echo "Docker Image:     $DOCKER_IMAGE"
    echo "Model Path:       $MODEL_PATH"
    echo "Server Port:      $SERVER_PORT"
    echo "Server Args:      $SERVER_ARGS"
    echo ""
    echo "Test Mode:        $TEST_MODE"
    echo "Test Loops:       $NUM_LOOPS"
    echo "Prompts/Loop:     $PROMPTS_PER_LOOP"
    echo "Success Pattern:  ${SUCCESS_PATTERN:-<none>}"
    echo ""
    echo "Timeouts:"
    echo "  Container:      ${CONTAINER_TIMEOUT}s"
    echo "  Server Startup: ${SERVER_STARTUP_TIMEOUT}s"
    echo "  Prompt:         ${PROMPT_TIMEOUT}s"
    echo ""
    echo "Docker Settings:"
    echo "  SHM Size:       $SHM_SIZE"
    echo "  Network:        $NETWORK_MODE"
    echo "  Devices:        ${DOCKER_DEVICES[*]:-<none>}"
    echo ""
    echo "Environment Vars: ${ENV_FLAGS:-<none>}"
    echo ""
    echo "Error Patterns:   ${ERROR_PATTERNS[*]:-<none>}"
    echo ""
    echo "Server Command:"
    echo "  $(build_server_cmd "$MODEL_PATH" "$SERVER_PORT" "$SERVER_ARGS")"
    echo ""
    echo "============================================================"
    exit 0
}

###############################################################################
# MAIN TEST LOOP
###############################################################################

# Wait for server to become ready (shared by both modes)
wait_for_server_ready() {
    local container_name="$1"
    local timeout="$2"
    
    log "  Waiting for server to become ready (up to ${timeout}s) ..."
    local elapsed=0
    while [ $elapsed -lt "$timeout" ]; do
        if ! docker ps -q -f name="$container_name" 2>/dev/null | grep -q .; then
            log "  Container died before server became ready."
            return 1
        fi
        
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 5 \
            "http://localhost:${SERVER_PORT}${HEALTH_ENDPOINT}" 2>/dev/null) || true
        
        if [ "$http_code" = "200" ]; then
            log "  Server is ready! (took ~${elapsed}s)"
            return 0
        fi
        
        sleep 5
        elapsed=$(( elapsed + 5 ))
    done
    
    log "  FAIL: Server did not become ready within ${timeout}s."
    return 1
}

# Send prompts and check responses (shared by both modes)
# Returns: sets PROMPTS_OK and ALL_MATCH variables
send_prompts() {
    local iter_log="$1"
    
    PROMPTS_OK=true
    ALL_MATCH=true
    
    for p in $(seq 1 "$PROMPTS_PER_LOOP"); do
        log "  Sending prompt $p/$PROMPTS_PER_LOOP ..."
        
        # Build payload using plugin
        local payload
        payload=$(build_chat_payload "$MODEL_PATH" "$PROMPT_CONTENT" "$DEFAULT_PARAMS" "$EXTRA_PARAMS")
        
        local response
        response=$(curl -s --max-time "$PROMPT_TIMEOUT" \
            -X POST "http://localhost:${SERVER_PORT}${CHAT_ENDPOINT}" \
            -H "accept: */*" \
            -H "Content-Type: application/json" \
            -d "$payload" 2>/dev/null) || response=""
        
        if [ -z "$response" ]; then
            log "  FAIL: Prompt $p/$PROMPTS_PER_LOOP — no response (timeout or connection error)."
            PROMPTS_OK=false
            break
        fi
        
        # Parse response using plugin
        local content
        content=$(parse_chat_response "$response")
        
        log "  Prompt $p response: $content"
        
        # Save raw response to log
        {
            echo "--- Prompt $p response ---"
            echo "$response"
            echo ""
        } >> "$iter_log"
        
        # Check success pattern if specified
        if [ -n "$SUCCESS_PATTERN" ]; then
            if ! echo "$content" | grep -qi "$SUCCESS_PATTERN"; then
                ALL_MATCH=false
                log "  WARNING: Prompt $p/$PROMPTS_PER_LOOP answer does not match pattern '$SUCCESS_PATTERN'."
            fi
        fi
    done
}

# Check for error patterns in logs (shared by both modes)
# Returns: sets ERROR_FOUND variable
check_error_patterns() {
    local iter_log="$1"
    
    ERROR_FOUND=false
    sleep 2  # Give logs time to flush
    for pattern in "${ERROR_PATTERNS[@]}"; do
        if grep -q "$pattern" "$iter_log" 2>/dev/null; then
            ERROR_FOUND=true
            log "  FAIL: Found error pattern '$pattern' in logs."
            break
        fi
    done
}

# Run stress test in container mode (restart container each iteration)
run_container_mode() {
    for (( i=1; i<=NUM_LOOPS; i++ )); do
        ITER_START=$(date +%s)
        ITER_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
        CONTAINER_NAME="${FRAMEWORK}_stress_iter${i}_$$"
        ITER_LOG="$RUN_DIR/iter_${i}_${ITER_TIMESTAMP}.log"
        ITER_STATUS="FAIL"
        
        log ""
        log "────────────────────────────────────────────────────────────"
        log "  Iteration $i / $NUM_LOOPS   (container: $CONTAINER_NAME)"
        log "────────────────────────────────────────────────────────────"
        
        # Clean up any leftover container
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        
        # Build docker run command
        DOCKER_CMD=(
            docker run
            -d
            --cap-add=SYS_PTRACE
            --cap-add=CAP_SYS_ADMIN
            --security-opt seccomp=unconfined
            --user root
            --ulimit memlock=999332768:999332768
            --ipc=host
            --name "$CONTAINER_NAME"
            --shm-size="$SHM_SIZE"
            --hostname "STRESS-$(echo "${HOSTNAME:-$(hostname)}" | cut -f 1 -d .)"
            --network "$NETWORK_MODE"
        )
        
        # Add devices
        for device in "${DOCKER_DEVICES[@]}"; do
            DOCKER_CMD+=(--device="$device")
        done
        
        # Add group for video (AMD GPUs)
        if [[ " ${DOCKER_DEVICES[*]} " =~ "/dev/kfd" ]]; then
            DOCKER_CMD+=(--group-add video)
        fi
        
        # Add workspace mounts
        DOCKER_CMD+=(-v "$WORKSPACE_BASE:/workspace/")
        for mount in "${WORKSPACE_MOUNTS[@]}"; do
            DOCKER_CMD+=(-v "$WORKSPACE_BASE/$mount:/workspace/$mount")
        done
        
        # Add HF token and home
        DOCKER_CMD+=(
            -e "HF_HOME=/workspace/.cache/huggingface"
            -e "HF_TOKEN=$HF_TOKEN"
        )
        
        # Add environment flags from config
        if [ -n "$ENV_FLAGS" ]; then
            # shellcheck disable=SC2206
            DOCKER_CMD+=($ENV_FLAGS)
        fi
        
        DOCKER_CMD+=(--workdir /workspace/)
        
        # Add entrypoint override if plugin specifies one
        if [ -n "$DOCKER_ENTRYPOINT" ]; then
            DOCKER_CMD+=(--entrypoint "$DOCKER_ENTRYPOINT")
        fi
        
        DOCKER_CMD+=("$DOCKER_IMAGE")
        DOCKER_CMD+=(bash -c "$SERVER_CMD")
        
        # Launch container
        log "  Starting Docker container ..."
        "${DOCKER_CMD[@]}" 2>&1 | tee -a "$ITER_LOG"
        
        # Watchdog: auto-kill after timeout
        (
            sleep "$CONTAINER_TIMEOUT"
            if docker ps -q -f name="$CONTAINER_NAME" 2>/dev/null | grep -q .; then
                echo "[WATCHDOG] Timeout reached (${CONTAINER_TIMEOUT}s). Killing $CONTAINER_NAME" >> "$ITER_LOG"
                docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
            fi
        ) &
        WATCHDOG_PID=$!
        
        # Stream container logs
        docker logs -f "$CONTAINER_NAME" > "$ITER_LOG" 2>&1 &
        LOGS_PID=$!
        
        # Wait for server readiness
        if ! wait_for_server_ready "$CONTAINER_NAME" "$SERVER_STARTUP_TIMEOUT"; then
            cleanup_container "$CONTAINER_NAME"
            FAIL_COUNT=$(( FAIL_COUNT + 1 ))
            ITER_END=$(date +%s)
            log "  Iteration $i finished in $(( ITER_END - ITER_START ))s — FAIL (server not ready)"
            continue
        fi
        
        # Send prompts
        send_prompts "$ITER_LOG"
        
        # Check for error patterns in logs
        check_error_patterns "$ITER_LOG"
        
        # Determine success/failure
        if [ "$PROMPTS_OK" = true ] && [ "$ALL_MATCH" = true ] && [ "$ERROR_FOUND" = false ]; then
            ITER_STATUS="SUCCESS"
            SUCCESS_COUNT=$(( SUCCESS_COUNT + 1 ))
        else
            ITER_STATUS="FAIL"
            FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        fi
        
        # Tear down
        cleanup_container "$CONTAINER_NAME"
        
        # Rename log file to include status
        FINAL_LOG="$RUN_DIR/iter_${i}_${ITER_TIMESTAMP}_${ITER_STATUS}.log"
        mv "$ITER_LOG" "$FINAL_LOG"
        
        ITER_END=$(date +%s)
        log "  Iteration $i finished in $(( ITER_END - ITER_START ))s — $ITER_STATUS"
    done
}

# Run stress test in server mode (keep container, restart server each iteration)
run_server_mode() {
    CONTAINER_NAME="${FRAMEWORK}_stress_persistent_$$"
    local container_log="$RUN_DIR/container.log"
    
    log ""
    log "────────────────────────────────────────────────────────────"
    log "  Starting persistent container: $CONTAINER_NAME"
    log "────────────────────────────────────────────────────────────"
    
    # Start persistent container
    if ! start_persistent_container "$CONTAINER_NAME" "$container_log"; then
        log "  FAIL: Could not start persistent container."
        exit 1
    fi
    
    # Set up trap to cleanup container on exit
    trap 'cleanup_container "$CONTAINER_NAME"' EXIT
    
    for (( i=1; i<=NUM_LOOPS; i++ )); do
        ITER_START=$(date +%s)
        ITER_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
        ITER_LOG="$RUN_DIR/iter_${i}_${ITER_TIMESTAMP}.log"
        ITER_STATUS="FAIL"
        
        log ""
        log "────────────────────────────────────────────────────────────"
        log "  Iteration $i / $NUM_LOOPS   (server restart)"
        log "────────────────────────────────────────────────────────────"
        
        # Clear server log inside container
        docker exec "$CONTAINER_NAME" bash -c "rm -f /tmp/server.log; touch /tmp/server.log" 2>/dev/null || true
        
        # Start server inside container
        if ! start_server_in_container "$CONTAINER_NAME" "$SERVER_CMD" "$ITER_LOG"; then
            log "  FAIL: Could not start server inside container."
            FAIL_COUNT=$(( FAIL_COUNT + 1 ))
            ITER_END=$(date +%s)
            log "  Iteration $i finished in $(( ITER_END - ITER_START ))s — FAIL (server start failed)"
            
            # Rename log file
            FINAL_LOG="$RUN_DIR/iter_${i}_${ITER_TIMESTAMP}_FAIL.log"
            mv "$ITER_LOG" "$FINAL_LOG" 2>/dev/null || true
            continue
        fi
        
        # Wait for server readiness
        if ! wait_for_server_ready "$CONTAINER_NAME" "$SERVER_STARTUP_TIMEOUT"; then
            # Capture server logs before stopping
            capture_server_logs "$CONTAINER_NAME" "$ITER_LOG"
            stop_server_in_container "$CONTAINER_NAME"
            FAIL_COUNT=$(( FAIL_COUNT + 1 ))
            ITER_END=$(date +%s)
            log "  Iteration $i finished in $(( ITER_END - ITER_START ))s — FAIL (server not ready)"
            
            # Rename log file
            FINAL_LOG="$RUN_DIR/iter_${i}_${ITER_TIMESTAMP}_FAIL.log"
            mv "$ITER_LOG" "$FINAL_LOG" 2>/dev/null || true
            continue
        fi
        
        # Send prompts
        send_prompts "$ITER_LOG"
        
        # Capture server logs
        capture_server_logs "$CONTAINER_NAME" "$ITER_LOG"
        
        # Check for error patterns in logs
        check_error_patterns "$ITER_LOG"
        
        # Determine success/failure
        if [ "$PROMPTS_OK" = true ] && [ "$ALL_MATCH" = true ] && [ "$ERROR_FOUND" = false ]; then
            ITER_STATUS="SUCCESS"
            SUCCESS_COUNT=$(( SUCCESS_COUNT + 1 ))
        else
            ITER_STATUS="FAIL"
            FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        fi
        
        # Stop server (container stays running)
        stop_server_in_container "$CONTAINER_NAME"
        
        # Rename log file to include status
        FINAL_LOG="$RUN_DIR/iter_${i}_${ITER_TIMESTAMP}_${ITER_STATUS}.log"
        mv "$ITER_LOG" "$FINAL_LOG"
        
        ITER_END=$(date +%s)
        log "  Iteration $i finished in $(( ITER_END - ITER_START ))s — $ITER_STATUS"
    done
    
    # Cleanup persistent container
    log ""
    log "  Stopping persistent container ..."
    cleanup_container "$CONTAINER_NAME"
    trap - EXIT  # Remove trap since we cleaned up manually
}

run_stress_test() {
    # Check HF token
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo ""
        echo -e "${RED}ERROR:${NC} HF_TOKEN environment variable is not set."
        echo ""
        echo "  Set it before running this script:"
        echo ""
        echo "    export HF_TOKEN='hf_your_token_here'"
        echo "    $0 [OPTIONS]"
        echo ""
        echo "  You can generate a token at: https://huggingface.co/settings/tokens"
        echo ""
        exit 1
    fi
    
    # Create output directory
    IMAGE_SLUG=$(echo "$DOCKER_IMAGE" | sed 's/[\/:]/_/g')
    RUN_DIR="$WORKSPACE_BASE/${FRAMEWORK}_stress_${IMAGE_SLUG}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_DIR"
    
    SUMMARY_LOG="$RUN_DIR/summary.log"
    
    # Counters
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    # Build server command using plugin
    SERVER_CMD=$(build_server_cmd "$MODEL_PATH" "$SERVER_PORT" "$SERVER_ARGS")
    HEALTH_ENDPOINT=$(get_health_endpoint)
    CHAT_ENDPOINT=$(get_chat_endpoint)
    
    # Get docker entrypoint override from plugin
    DOCKER_ENTRYPOINT=$(get_docker_entrypoint 2>/dev/null || echo "")
    
    log "============================================================"
    log "Universal Stress Test - $FRAMEWORK"
    log "  Test mode    : $TEST_MODE"
    log "  Docker image : $DOCKER_IMAGE"
    log "  Model        : $MODEL_PATH"
    log "  Loops        : $NUM_LOOPS"
    log "  Timeout      : ${CONTAINER_TIMEOUT}s ($(( CONTAINER_TIMEOUT / 60 )) min)"
    log "  Results dir  : $RUN_DIR"
    log "============================================================"
    
    # Run appropriate mode
    if [ "$TEST_MODE" = "server" ]; then
        run_server_mode
    else
        run_container_mode
    fi
    
    # Summary
    log ""
    log "============================================================"
    log "  STRESS TEST COMPLETE"
    log "============================================================"
    log "  Test mode    : $TEST_MODE"
    log "  Framework    : $FRAMEWORK"
    log "  Docker image : $DOCKER_IMAGE"
    log "  Model        : $MODEL_PATH"
    log "  Total runs   : $NUM_LOOPS"
    log "  Successes    : $SUCCESS_COUNT"
    log "  Failures     : $FAIL_COUNT"
    log "  Pass rate    : $(( SUCCESS_COUNT * 100 / NUM_LOOPS ))%"
    log "  Results dir  : $RUN_DIR"
    log "============================================================"
    
    # Exit with non-zero if any failures
    if [ "$FAIL_COUNT" -gt 0 ]; then
        exit 1
    fi
    exit 0
}

###############################################################################
# MAIN
###############################################################################

main() {
    parse_args "$@"
    check_dependencies
    load_config
    load_plugin
    
    if [ "$DRY_RUN" = true ]; then
        show_dry_run
    else
        run_stress_test
    fi
}

main "$@"
