#!/usr/bin/env julia
# Fresh MaxCut D=9 sweep with live per-depth timing.
#
# Usage: julia --project=. -t 16 scripts/maxcut_d9_timed.jl [P_MAX] [SEED]

using QaoaXorsat, Printf, Dates, Random

k, D = 2, 9
clause_sign = -1
p_max = parse(Int, get(ARGS, 1, "12"))
seed = parse(Int, get(ARGS, 2, "42"))

results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")
timing_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-timing.csv")
mkpath(dirname(results_file))

function fmt_time(seconds)
    seconds < 60 && return @sprintf("%.1fs", seconds)
    seconds < 3600 && return @sprintf("%.1fmin", seconds / 60)
    @sprintf("%.2fh", seconds / 3600)
end

function checkpoint_depth(p::Int)
    p ≥ 13
end

# Fresh CSV
open(results_file, "w") do io
    println(io, "# MaxCut k=2 D=$D TIMED sweep — $(now())")
    println(io, "k,D,p,ctilde,wall_seconds,gamma,beta")
end

open(timing_file, "w") do io
    println(io, "# MaxCut k=2 D=$D per-depth timing — $(now())")
    println(io, "# all optimized production settings: adjoint gradients, depth budgets, plateau detection, memory-bounded restarts")
    println(io, "D,p,ctilde,delta,depth_seconds,cumulative_seconds,evaluations,starts,iterations,converged,restarts,maxiters,g_abstol,best_start_kind,best_start_seconds,checkpointed")
end

println("╔══════════════════════════════════════════════════════════╗")
@printf("║  MaxCut D=%-2d — Timed Optimized Sweep p=1..%-2d          ║\n", D, p_max)
println("║  $(now())                            ║")
println("║  Threads: $(Threads.nthreads())                                          ║")
println("╚══════════════════════════════════════════════════════════╝")
println()
println("Settings: autodiff=:adjoint, depth budgets/tolerances, plateau detection, memory-bounded restarts")
@printf("  %-4s  %-14s  %-12s  %-12s  %-12s  %-8s  %-8s  %s\n",
        "p", "c̃", "Δc̃", "this depth", "cumulative", "evals", "iters", "settings")
@printf("  %-4s  %-14s  %-12s  %-12s  %-12s  %-8s  %-8s  %s\n",
        "──", "──────────────", "────────────", "────────────", "────────────", "──────", "──────", "────────")
flush(stdout)

sweep_start = time()
warm = QAOAAngles[]
prev_value = 0.5

for p in 1:p_max
    global warm, prev_value

    params = TreeParams(k, D, p)
    budget = QaoaXorsat.depth_optimization_budget(p, 8, 200)
    g_abstol = QaoaXorsat.depth_g_abstol(p)
    checkpointed = checkpoint_depth(p)
    initial_guesses = isempty(warm) ? QAOAAngles[] : [extend_angles(warm[1], p)]

    t0 = time()
    local result
    try
        result = optimize_angles(params;
            clause_sign,
            initial_guesses,
            initial_guess_kind=isempty(warm) ? :random : :warm,
            restarts=budget.restarts,
            maxiters=budget.maxiters,
            autodiff=:adjoint,
            rng=MersenneTwister(seed + p),
            g_abstol,
            checkpointed,
            on_evaluation = (chunk, evals, elapsed, val, gnorm) -> begin
                @printf("    [p=%d start=%d] eval %d: c̃=%.10f  |∇|=%.2e  %s\n",
                        p, chunk, evals, val, gnorm, fmt_time(elapsed))
                flush(stdout)
            end
        )
    catch e
        dt = time() - t0
        cumulative = time() - sweep_start
        @printf("  p=%-2d  FAILED after %s (cumulative %s): %s\n",
                p, fmt_time(dt), fmt_time(cumulative), sprint(showerror, e))
        println("\n  Stopping — likely out of memory.")
        flush(stdout)
        break
    end
    dt = time() - t0
    cumulative = time() - sweep_start
    delta = result.value - prev_value

    @printf("  p=%-2d  %.10f  %+.8f  %-12s  %-12s  %-8d  %-8d  restarts=%d maxiters=%d g=%.0e chk=%s\n",
            p, result.value, delta, fmt_time(dt), fmt_time(cumulative),
            result.evaluations, result.iterations, result.restarts,
            result.maxiters, result.g_abstol, string(checkpointed))
    flush(stdout)

    gamma_str = join(string.(result.angles.γ), ';')
    beta_str = join(string.(result.angles.β), ';')
    open(results_file, "a") do io
        @printf(io, "%d,%d,%d,%.12f,%.1f,%s,%s\n",
            k, D, p, result.value, dt, gamma_str, beta_str)
    end
    open(timing_file, "a") do io
        @printf(io, "%d,%d,%.12f,%.12f,%.1f,%.1f,%d,%d,%d,%s,%d,%d,%.1e,%s,%.1f,%s\n",
            D, p, result.value, delta, dt, cumulative, result.evaluations,
            result.starts, result.iterations, string(result.converged), result.restarts,
            result.maxiters, result.g_abstol, string(result.best_start_kind),
            result.best_start_wall_time_seconds, string(checkpointed))
    end

    warm = [result.angles]
    prev_value = result.value
end

total = time() - sweep_start
println()
println("╔══════════════════════════════════════════════════════════╗")
@printf("║  Done! Total wall time: %-33s║\n",
    total < 3600 ? @sprintf("%.1f min", total/60) : @sprintf("%.2f hours", total/3600))
println("╚══════════════════════════════════════════════════════════╝")
println("Results: $results_file")
println("Timing:  $timing_file")

# macOS notification
try
    run(`osascript -e "display notification \"MaxCut D=$D sweep done in $(round(total/60, digits=1))min\" with title \"QAOA Sweep Complete\" sound name \"Glass\""`)
catch end
