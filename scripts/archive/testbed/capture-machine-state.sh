#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/testbed/capture-machine-state.sh OUTPUT_PATH [PHASE]

Capture a JSON snapshot of the current Linux machine state for benchmark
reliability checks.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 1
fi

output_path=$1
phase=${2:-snapshot}

mkdir -p "$(dirname "$output_path")"

json_escape() {
    printf '%s' "$1" | sed \
        -e 's/\\/\\\\/g' \
        -e 's/"/\\"/g' \
        -e ':a;N;$!ba;s/\n/\\n/g' \
        -e 's/\r/\\r/g' \
        -e 's/\t/\\t/g'
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

meminfo_value_kb() {
    local key=$1
    awk -v target="$key" '$1 == target ":" { print $2; exit }' /proc/meminfo
}

avg_cpu_freq_khz() {
    local values
    values=$(find /sys/devices/system/cpu -path '*/cpufreq/scaling_cur_freq' -readable -exec cat {} + 2>/dev/null || true)
    if [[ -z "$values" ]]; then
        printf 'null'
        return
    fi

    awk '{ sum += $1; count += 1 } END { if (count > 0) printf "%.0f", sum / count; else printf "null" }' <<<"$values"
}

unique_cpu_governors() {
    local governors
    governors=$(find /sys/devices/system/cpu -path '*/cpufreq/scaling_governor' -readable -exec cat {} + 2>/dev/null | sort -u || true)
    if [[ -z "$governors" ]]; then
        printf '[]'
        return
    fi

    awk 'BEGIN { first = 1; printf "[" }
         { if (!first) printf ", "; gsub(/\\/, "\\\\"); gsub(/"/, "\\\""); printf "\"%s\"", $0; first = 0 }
         END { printf "]" }' <<<"$governors"
}

json_string_array_from_lines() {
    local input=$1
    if [[ -z "$input" ]]; then
        printf '[]'
        return
    fi

    awk 'BEGIN { first = 1; printf "[" }
         { if (!first) printf ", "; gsub(/\\/, "\\\\"); gsub(/"/, "\\\""); printf "\"%s\"", $0; first = 0 }
         END { printf "]" }' <<<"$input"
}

thermal_zone_lines() {
    local lines=""
    local zone
    for zone in /sys/class/thermal/thermal_zone*; do
        [[ -d "$zone" ]] || continue
        local zone_type
        local raw_temp
        local temp_c
        zone_type=$(<"$zone/type")
        raw_temp=$(<"$zone/temp")
        if [[ "$raw_temp" =~ ^-?[0-9]+$ ]]; then
            if (( raw_temp > 1000 || raw_temp < -1000 )); then
                temp_c=$(awk -v value="$raw_temp" 'BEGIN { printf "%.1f", value / 1000.0 }')
            else
                temp_c=$raw_temp
            fi
            lines+="${zone##*/}:${zone_type}:${temp_c}"$'\n'
        fi
    done
    printf '%s' "$lines"
}

max_thermal_temp_c() {
    local lines=$1
    if [[ -z "$lines" ]]; then
        printf 'null'
        return
    fi

    awk -F: 'BEGIN { found = 0 }
         NF >= 3 {
             value = $3 + 0
             if (!found || value > max) {
                 max = value
                 found = 1
             }
         }
         END {
             if (found) printf "%.1f", max
             else printf "null"
         }' <<<"$lines"
}

power_supply_lines() {
    local lines=""
    local supply
    for supply in /sys/class/power_supply/*; do
        [[ -d "$supply" ]] || continue
        local name type online status capacity
        name=${supply##*/}
        type=$(<"$supply/type" 2>/dev/null || printf 'unknown')
        online=$(<"$supply/online" 2>/dev/null || printf '')
        status=$(<"$supply/status" 2>/dev/null || printf '')
        capacity=$(<"$supply/capacity" 2>/dev/null || printf '')
        lines+="${name}:${type}:online=${online}:status=${status}:capacity=${capacity}"$'\n'
    done
    printf '%s' "$lines"
}

ac_online_value() {
    local lines=$1
    if [[ -z "$lines" ]]; then
        printf 'null'
        return
    fi

    awk -F: 'BEGIN { found = 0; online = 0 }
         $2 == "Mains" || $2 == "AC" {
             found = 1
             if ($3 ~ /online=1/) online = 1
         }
         END {
             if (!found) printf "null"
             else if (online) printf "true"
             else printf "false"
         }' <<<"$lines"
}

battery_capacity_percent() {
    local lines=$1
    if [[ -z "$lines" ]]; then
        printf 'null'
        return
    fi

    awk -F: 'BEGIN { found = 0 }
         $2 == "Battery" {
             found = 1
             if (match($0, /capacity=([0-9]+)/, captures)) {
                 print captures[1]
                 exit
             }
         }
         END {
             if (!found) printf "null"
         }' <<<"$lines"
}

top_process_lines=$(ps -eo pid=,comm=,%cpu=,%mem= --sort=-%cpu | head -n 10 | sed 's/^ *//')
thermal_lines=$(thermal_zone_lines)
power_lines=$(power_supply_lines)
sensors_output=""
if command_exists sensors; then
    sensors_output=$(sensors 2>/dev/null || true)
fi

timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
hostname_value=$(hostname)
kernel_value=$(uname -r)
arch_value=$(uname -m)
uptime_seconds=$(awk '{ printf "%.0f", $1 }' /proc/uptime)
loadavg_1=$(awk '{ print $1 }' /proc/loadavg)
loadavg_5=$(awk '{ print $2 }' /proc/loadavg)
loadavg_15=$(awk '{ print $3 }' /proc/loadavg)
mem_total_kb=$(meminfo_value_kb MemTotal)
mem_available_kb=$(meminfo_value_kb MemAvailable)
swap_total_kb=$(meminfo_value_kb SwapTotal)
swap_free_kb=$(meminfo_value_kb SwapFree)
cpu_count=$(nproc)
cpu_model=$(awk -F': ' '/model name/ { print $2; exit }
                       /Hardware/ { print $2; exit }
                       /Processor/ { print $2; exit }' /proc/cpuinfo)
if [[ -z "$cpu_model" ]] && command_exists lscpu; then
    cpu_model=$(lscpu 2>/dev/null | awk -F': *' '/Model name/ { print $2; exit }')
fi
root_used_percent=$(df -P / | awk 'NR == 2 { gsub(/%/, "", $5); print $5 }')
avg_freq_khz=$(avg_cpu_freq_khz)
governors_json=$(unique_cpu_governors)
thermal_json=$(json_string_array_from_lines "$thermal_lines")
power_json=$(json_string_array_from_lines "$power_lines")
top_processes_json=$(json_string_array_from_lines "$top_process_lines")
max_thermal_c=$(max_thermal_temp_c "$thermal_lines")
ac_online=$(ac_online_value "$power_lines")
battery_capacity=$(battery_capacity_percent "$power_lines")
sensors_json="null"
if [[ -n "$sensors_output" ]]; then
    sensors_json="\"$(json_escape "$sensors_output")\""
fi

cat >"$output_path" <<EOF
{
  "timestamp_utc": "$(json_escape "$timestamp_utc")",
  "phase": "$(json_escape "$phase")",
  "hostname": "$(json_escape "$hostname_value")",
  "kernel": "$(json_escape "$kernel_value")",
  "architecture": "$(json_escape "$arch_value")",
  "cpu_model": "$(json_escape "$cpu_model")",
  "cpu_count": $cpu_count,
  "uptime_seconds": $uptime_seconds,
  "loadavg_1": $loadavg_1,
  "loadavg_5": $loadavg_5,
  "loadavg_15": $loadavg_15,
  "mem_total_kb": $mem_total_kb,
  "mem_available_kb": $mem_available_kb,
  "swap_total_kb": $swap_total_kb,
  "swap_free_kb": $swap_free_kb,
  "root_filesystem_used_percent": $root_used_percent,
    "cpu_governors": $governors_json,
    "average_cpu_frequency_khz": $avg_freq_khz,
    "max_thermal_temp_c": $max_thermal_c,
    "ac_online": $ac_online,
    "battery_capacity_percent": $battery_capacity,
  "thermal_zones": $thermal_json,
  "power_supplies": $power_json,
  "top_cpu_processes": $top_processes_json,
  "sensors_output": $sensors_json
}
EOF

printf 'wrote %s\n' "$output_path"
