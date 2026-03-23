#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/testbed/run-with-machine-state.sh OUTPUT_DIR -- COMMAND [ARGS...]

Capture machine-state snapshots before and after a command, record command
timing, and preserve stdout/stderr for benchmark reliability analysis.
EOF
}

if [[ $# -lt 3 ]]; then
    usage >&2
    exit 1
fi

output_dir=$1
shift

if [[ ${1:-} != "--" ]]; then
    usage >&2
    exit 1
fi
shift

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

mkdir -p "$output_dir"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
snapshot_script="$script_dir/capture-machine-state.sh"

command_string=$(printf '%q ' "$@")
printf '%s\n' "$command_string" >"$output_dir/command.txt"

start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
bash "$snapshot_script" "$output_dir/machine-state-before.json" before

status=0
if command -v /usr/bin/time >/dev/null 2>&1; then
    set +e
    /usr/bin/time -v -o "$output_dir/command-time.txt" -- "$@" \
        >"$output_dir/stdout.txt" 2>"$output_dir/stderr.txt"
    status=$?
    set -e
else
    set +e
    "$@" >"$output_dir/stdout.txt" 2>"$output_dir/stderr.txt"
    status=$?
    set -e
fi

end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
bash "$snapshot_script" "$output_dir/machine-state-after.json" after

timing_file_value=null
if [[ -f "$output_dir/command-time.txt" ]]; then
        timing_file_value='"command-time.txt"'
fi

cat >"$output_dir/run-summary.json" <<EOF
{
  "start_time_utc": "$start_time",
  "end_time_utc": "$end_time",
  "exit_code": $status,
  "command_file": "command.txt",
  "stdout_file": "stdout.txt",
  "stderr_file": "stderr.txt",
    "timing_file": $timing_file_value,
  "machine_state_before": "machine-state-before.json",
  "machine_state_after": "machine-state-after.json"
}
EOF

exit $status
