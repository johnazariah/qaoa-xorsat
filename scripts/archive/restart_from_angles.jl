#!/usr/bin/env julia
#
# Restart optimization at a single depth using known angles as warm start.
# Saves per-iteration trace CSV for post-hoc analysis.
#
# Usage:
#   julia --project=. -t 10 scripts/restart_from_angles.jl \
#     K D P RESTARTS MAXITERS SEED AUTODIFF \
#     "γ1;γ2;...;γP" "β1;β2;...;βP"
#
# Example (p=11 restart from non-converged run):
#   julia --project=. -t 10 scripts/restart_from_angles.jl \
#     3 4 11 2 640 1234 adjoint \
#     "2.9139009442638883;5.859346052024547;5.814656296787951;5.792936255637537;5.781878760117875;5.773265268851806;5.762412444996204;5.744300022681871;5.702464349202822;5.636541740111221;5.5378902710824125" \
#     "2.7591606229547816;2.8284646002114098;2.8505259922036967;2.857536596495945;2.861721530091938;2.8658231801107052;2.872380978048311;2.885243683339946;2.9124423020071335;2.9544452134082095;3.0520627823372166"

using Dates
using Printf
using Random
using QaoaXorsat

length(ARGS) == 9 || (println(stderr, "Usage: ... K D P RESTARTS MAXITERS SEED AUTODIFF \"γ1;...\" \"β1;...\""); exit(1))

k = parse(Int, ARGS[1])
D = parse(Int, ARGS[2])
p = parse(Int, ARGS[3])
restarts = parse(Int, ARGS[4])
maxiters = parse(Int, ARGS[5])
seed = parse(Int, ARGS[6])
autodiff = Symbol(lowercase(ARGS[7]))
γ_values = parse.(Float64, split(ARGS[8], ';'))
β_values = parse.(Float64, split(ARGS[9], ';'))

length(γ_values) == p || error("expected $p γ values, got $(length(γ_values))")
length(β_values) == p || error("expected $p β values, got $(length(β_values))")

warm_angles = QAOAAngles(γ_values, β_values)
clause_sign = k == 2 ? -1 : 1
rng = MersenneTwister(seed)

# Set up output directory
timestamp = Dates.format(Dates.now(Dates.UTC), dateformat"yyyymmddTHHMMSS")
run_id = "$(timestamp)-restart-k$(k)-d$(D)-p$(p)-r$(restarts)-i$(maxiters)-s$(seed)"
base_dir = joinpath(@__DIR__, "..", ".project", "results", "optimization", "runs")
run_dir = mkpath(joinpath(base_dir, run_id))

println(stderr, "run_id = $run_id")
println(stderr, "run_dir = $run_dir")
println(stderr, "warm start value = $(QaoaXorsat.qaoa_expectation(TreeParams(k, D, p), warm_angles; clause_sign))")
flush(stderr)

# Progress heartbeat
function progress_cb(start_index, evaluations, elapsed_seconds)
    @printf(stderr, "  p=%d start %d: %d evals, %.1fs elapsed\n", p, start_index, evaluations, elapsed_seconds)
    flush(stderr)
end

result = optimize_angles(
    TreeParams(k, D, p);
    clause_sign,
    restarts,
    maxiters,
    initial_guesses=[warm_angles],
    initial_guess_kind=:warm,
    autodiff,
    rng,
    on_evaluation=progress_cb,
)

# Write trace CSV
trace_path = joinpath(run_dir, "trace-p$(p).csv")
open(trace_path, "w") do io
    println(io, "start,kind,iteration,value,g_norm")
    for (i, sr) in enumerate(result.start_results)
        for entry in sr.trace
            @printf(io, "%d,%s,%d,%.12e,%.6e\n", i, sr.kind, entry.iteration, entry.value, entry.g_norm)
        end
    end
end

# Write summary
summary_path = joinpath(run_dir, "summary.txt")
open(summary_path, "w") do io
    println(io, "run_id: $run_id")
    println(io, "k=$k, D=$D, p=$p")
    println(io, "restarts=$restarts, maxiters=$maxiters, seed=$seed, autodiff=$autodiff")
    println(io, "value: $(result.value)")
    println(io, "converged: $(result.converged)")
    println(io, "iterations: $(result.iterations)")
    println(io, "evaluations: $(result.evaluations)")
    println(io, "wall_time_seconds: $(result.wall_time_seconds)")
    println(io, "g_abstol: $(result.g_abstol)")
    println(io, "best_start_kind: $(result.best_start_kind)")
    println(io, "gamma: $(join(string.(result.angles.γ), ';'))")
    println(io, "beta: $(join(string.(result.angles.β), ';'))")
    println(io, "")
    println(io, "per-start results:")
    for (i, sr) in enumerate(result.start_results)
        @printf(io, "  start %d (%s): value=%.12f, iters=%d, evals=%d, converged=%s, time=%.1fs, trace_entries=%d\n",
            i, sr.kind, sr.value, sr.iterations, sr.evaluations, sr.converged, sr.wall_time_seconds, length(sr.trace))
    end
end

# Print to stdout
@printf("p=%d value=%.12f converged=%s iterations=%d evaluations=%d wall_time=%.1fs g_abstol=%.1e\n",
    p, result.value, result.converged, result.iterations, result.evaluations, result.wall_time_seconds, result.g_abstol)
println("trace saved to: $trace_path")
println("summary saved to: $summary_path")
for (i, sr) in enumerate(result.start_results)
    @printf("  start %d (%s): value=%.12f converged=%s iters=%d trace=%d entries\n",
        i, sr.kind, sr.value, sr.converged, sr.iterations, length(sr.trace))
end
