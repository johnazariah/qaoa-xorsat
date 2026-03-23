#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/testbed/analyze-machine-state.sh RUN_DIR

Inspect machine-state snapshots produced by run-with-machine-state.sh and emit a
JSON assessment of whether the run may have been affected by ambient load,
memory pressure, swap activity, or thermal stress.
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 1
fi

run_dir=$1
before_file="$run_dir/machine-state-before.json"
after_file="$run_dir/machine-state-after.json"

[[ -f "$before_file" ]] || { echo "missing $before_file" >&2; exit 1; }
[[ -f "$after_file" ]] || { echo "missing $after_file" >&2; exit 1; }

json_escape() {
    printf '%s' "$1" | sed \
        -e 's/\\/\\\\/g' \
        -e 's/"/\\"/g' \
        -e ':a;N;$!ba;s/\n/\\n/g' \
        -e 's/\r/\\r/g' \
        -e 's/\t/\\t/g'
}

json_number() {
    local file=$1
    local key=$2
    awk -F': ' -v target="\"$key\"" '{
            field = $1
            sub(/^ +/, "", field)
        }
        field == target {
            value = $2
            sub(/,$/, "", value)
            print value
            exit
        }' "$file"
}

warnings=()

cpu_count=$(json_number "$before_file" cpu_count)
loadavg_1=$(json_number "$before_file" loadavg_1)
mem_total_kb=$(json_number "$before_file" mem_total_kb)
mem_available_kb=$(json_number "$before_file" mem_available_kb)
swap_total_kb=$(json_number "$after_file" swap_total_kb)
swap_free_kb=$(json_number "$after_file" swap_free_kb)
root_used_percent=$(json_number "$before_file" root_filesystem_used_percent)
max_thermal_after=$(json_number "$after_file" max_thermal_temp_c)
ac_online=$(json_number "$before_file" ac_online)

load_per_cpu=$(awk -v load="$loadavg_1" -v cpu="$cpu_count" 'BEGIN {
    if (cpu == 0 || cpu == "") print "null"
    else printf "%.3f", load / cpu
}')

mem_available_fraction=$(awk -v total="$mem_total_kb" -v avail="$mem_available_kb" 'BEGIN {
    if (total == 0 || total == "") print "null"
    else printf "%.3f", avail / total
}')

swap_used_fraction=$(awk -v total="$swap_total_kb" -v free="$swap_free_kb" 'BEGIN {
    if (total == 0 || total == "") print "0.000"
    else printf "%.3f", (total - free) / total
}')

if awk -v ratio="$load_per_cpu" 'BEGIN { exit !(ratio != "null" && ratio > 0.50) }'; then
    warnings+=("high pre-run ambient CPU load")
fi

if awk -v fraction="$mem_available_fraction" 'BEGIN { exit !(fraction != "null" && fraction < 0.15) }'; then
    warnings+=("low pre-run available memory")
fi

if awk -v fraction="$swap_used_fraction" 'BEGIN { exit !(fraction > 0.05) }'; then
    warnings+=("post-run swap usage detected")
fi

if awk -v used="$root_used_percent" 'BEGIN { exit !(used > 90) }'; then
    warnings+=("root filesystem nearly full")
fi

if [[ "$max_thermal_after" != "null" ]] && awk -v temp="$max_thermal_after" 'BEGIN { exit !(temp >= 85.0) }'; then
    warnings+=("elevated post-run thermal state")
fi

if [[ "$ac_online" == "false" ]]; then
    warnings+=("system was not on AC power before run")
fi

status="ok"
if [[ ${#warnings[@]} -gt 0 ]]; then
    status="warn"
fi

printf '{\n'
printf '  "status": "%s",\n' "$status"
printf '  "load_per_cpu": %s,\n' "$load_per_cpu"
printf '  "mem_available_fraction": %s,\n' "$mem_available_fraction"
printf '  "swap_used_fraction": %s,\n' "$swap_used_fraction"
printf '  "max_thermal_temp_c_after": %s,\n' "$max_thermal_after"
printf '  "warnings": ['
for i in "${!warnings[@]}"; do
    [[ $i -gt 0 ]] && printf ', '
    printf '"%s"' "$(json_escape "${warnings[$i]}")"
done
printf ']\n'
printf '}\n'
