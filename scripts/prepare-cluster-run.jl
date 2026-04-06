#!/usr/bin/env julia
#
# Generate TOML warm-start configs + SLURM submission script from warm-start-angles.csv
#
# Usage (on the cluster login node):
#   cd ~/qaoa-xorsat
#   git pull origin main
#   julia --project=. scripts/prepare-cluster-run.jl
#   sbatch scripts/qaoa_warmstart_sweep.sh
#
using Printf
using Dates

csv_path = joinpath(@__DIR__, "..", "results", "warm-start-angles.csv")
config_dir = joinpath(@__DIR__, "..", "experiments", "warmstart")
mkpath(config_dir)

# Parse warm-start CSV
entries = []
for line in eachline(csv_path)
    startswith(line, '#') && continue
    startswith(line, "k,") && continue
    fields = split(line, ',')
    length(fields) >= 7 || continue
    k = parse(Int, fields[1])
    D = parse(Int, fields[2])
    p = parse(Int, fields[3])
    v = parse(Float64, fields[4])
    gamma = parse.(Float64, split(fields[6], ';'))
    beta = parse.(Float64, split(fields[7], ';'))
    push!(entries, (k=k, D=D, p=p, value=v, gamma=gamma, beta=beta))
end

@printf("Loaded %d warm-start entries from %s\n", length(entries), csv_path)

# Write per-pair TOML configs
# Each config resumes from the warm-start angles at p_start = best_p + 1
configs = []
for e in entries
    p_start = e.p + 1
    p_max = 15
    p_start > p_max && continue

    fname = "k$(e.k)-d$(e.D).toml"
    fpath = joinpath(config_dir, fname)

    # Write the warm-start angles to a companion CSV that optimize_qaoa.jl
    # can use via resume_from (we simulate a minimal results directory)
    ws_dir = joinpath(config_dir, "ws-k$(e.k)d$(e.D)")
    mkpath(ws_dir)
    open(joinpath(ws_dir, "results.csv"), "w") do io
        println(io, "run_id,run_kind,runner_label,timestamp_utc,git_commit,git_branch,git_dirty,k,D,p,clause_sign,restarts,maxiters,seed,value,wall_time_seconds,best_start_wall_time_seconds,evaluations,starts,iterations,converged,retry_count,best_start_kind,g_abstol,gamma,beta")
        gamma_str = join(string.(e.gamma), ';')
        beta_str = join(string.(e.beta), ';')
        @printf(io, "warmstart,experiment,warmstart,%s,warmstart,main,false,%d,%d,%d,1,0,1280,1234,%.12f,0.0,0.0,0,1,0,true,0,warm,1.0e-06,%s,%s\n",
            Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
            e.k, e.D, e.p, e.value, gamma_str, beta_str)
    end

    # Write TOML config
    open(fpath, "w") do io
        println(io, "k = $(e.k)")
        println(io, "D = $(e.D)")
        println(io, "p_min = $(p_start)")
        println(io, "p_max = $(p_max)")
        println(io, "restarts = 2")
        println(io, "maxiters = 1280")
        println(io, "seed = 1234")
        println(io, "preserve = true")
        println(io, "autodiff = \"adjoint\"")
        println(io, "resume_from = \"$(ws_dir)\"")
    end

    push!(configs, (k=e.k, D=e.D, p_start=p_start, p_max=p_max, value=e.value, fname=fname))
    @printf("  (%d,%d): warm-start from p=%d (c̃=%.6f), run p=%d–%d -> %s\n",
        e.k, e.D, e.p, e.value, p_start, p_max, fname)
end

# Write SLURM script
slurm_path = joinpath(@__DIR__, "qaoa_warmstart_sweep.sh")
n = length(configs)
open(slurm_path, "w") do io
    println(io, """#!/bin/bash
#
# SLURM warm-start sweep — generated $(Dates.format(today(), "yyyy-mm-dd"))
# Each task runs one (k,D) pair from its warm-start depth through p=15.
#
#SBATCH --job-name=qaoa-ws
#SBATCH --array=1-$(n)
#SBATCH --partition=c3d
#SBATCH --time=999:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=2700G
#SBATCH --output=qaoa-ws_%A-%a.out
#SBATCH --error=qaoa-ws_%A-%a.err

set -euo pipefail

CONFIGS=(""")
    for (i, c) in enumerate(configs)
        @printf(io, "    \"experiments/warmstart/%s\"  # %d: k=%d D=%d from p=%d\n", c.fname, i, c.k, c.D, c.p_start)
    end
    println(io, """)

TASK_INDEX=\$((SLURM_ARRAY_TASK_ID - 1))
CONFIG="\${CONFIGS[\$TASK_INDEX]}"

echo "=== QAOA Warm-Start Sweep ==="
echo "Task:   \$SLURM_ARRAY_TASK_ID / $(n)"
echo "Config: \$CONFIG"
echo "Node:   \$(hostname)"
echo "CPUs:   \${SLURM_CPUS_PER_TASK:-28}"
echo "Start:  \$(date -u)"
echo ""

cd ~/qaoa-xorsat
export PATH="/root/.juliaup/bin:\$HOME/.juliaup/bin:\$PATH"
julia --project=. -t \${SLURM_CPUS_PER_TASK:-28} scripts/optimize_qaoa.jl "\$CONFIG"

echo ""
echo "Done: \$(date -u)"
""")
end

chmod(slurm_path, 0o755)

println("\n=== READY ===")
@printf("Generated %d TOML configs in %s\n", length(configs), config_dir)
@printf("SLURM script: %s\n", slurm_path)
println("\nStephen: just run:")
println("  cd ~/qaoa-xorsat")
println("  git pull origin main")
println("  julia --project=. scripts/prepare-cluster-run.jl")
println("  sbatch scripts/qaoa_warmstart_sweep.sh")
