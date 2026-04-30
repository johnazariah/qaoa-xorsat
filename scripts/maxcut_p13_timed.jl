#!/usr/bin/env julia
# MaxCut p=13 timed sweep across a degree range.
#
# Usage: julia --project=. -t 16 scripts/maxcut_p13_timed.jl [D_START] [D_END] [SEED]
# Example: julia --project=. -t 16 scripts/maxcut_p13_timed.jl 3 9 42

using QaoaXorsat
using Dates
using Printf
using Random

const K = 2
const TARGET_P = 13
const CLAUSE_SIGN = -1

function fmt_time(seconds)
    seconds < 60 && return @sprintf("%.1fs", seconds)
    seconds < 3600 && return @sprintf("%.1fmin", seconds / 60)
    @sprintf("%.2fh", seconds / 3600)
end

function notify(message)
    try
        run(`osascript -e "display notification \"$message\" with title \"MaxCut p=13\" sound name \"Glass\""`)
    catch
    end
end

function results_path(D::Int)
    joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")
end

function timing_path()
    joinpath(@__DIR__, "..", "results", "maxcut-k2-p13-timing.csv")
end

function ensure_results_file!(file::String, D::Int)
    isfile(file) && return
    mkpath(dirname(file))
    open(file, "w") do io
        println(io, "# MaxCut k=2 D=$D sweep -- $(now())")
        println(io, "k,D,p,ctilde,wall_seconds,gamma,beta")
    end
end

function ensure_timing_file!(file::String)
    isfile(file) && return
    mkpath(dirname(file))
    open(file, "w") do io
        println(io, "# MaxCut p=13 per-D timing -- $(now())")
        println(io, "# checkpointed=true; autodiff=:adjoint; warm-started from p=12")
        println(io, "D,p,ctilde,wall_seconds,evaluations,starts,iterations,converged,restarts,maxiters,g_abstol,best_start_kind,best_start_seconds,checkpointed,started_at,finished_at")
    end
end

function get_best_angles(file::String, D::Int, target_p::Int)
    best_value = -Inf
    best_gamma = Float64[]
    best_beta = Float64[]
    isfile(file) || return (best_value, best_gamma, best_beta)

    for line in eachline(file)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 7 || continue

        row_k = tryparse(Int, fields[1]); row_k === nothing && continue
        row_D = tryparse(Int, fields[2]); row_D === nothing && continue
        row_p = tryparse(Int, fields[3]); row_p === nothing && continue
        value = tryparse(Float64, fields[4]); value === nothing && continue
        (row_k == K && row_D == D && row_p == target_p) || continue
        QaoaXorsat.is_valid_qaoa_value(value) || continue

        if value > best_value
            best_value = value
            best_gamma = parse.(Float64, split(fields[6], ';'))
            best_beta = parse.(Float64, split(fields[7], ';'))
        end
    end

    return (best_value, best_gamma, best_beta)
end

function append_result!(file::String, D::Int, result::AngleOptimizationResult)
    gamma_string = join(string.(result.angles.γ), ';')
    beta_string = join(string.(result.angles.β), ';')
    open(file, "a") do io
        @printf(io, "%d,%d,%d,%.12f,%.1f,%s,%s\n",
            K, D, TARGET_P, result.value, result.wall_time_seconds,
            gamma_string, beta_string)
    end
end

function append_timing!(file::String, D::Int, result::AngleOptimizationResult, started_at, finished_at)
    open(file, "a") do io
        @printf(io, "%d,%d,%.12f,%.1f,%d,%d,%d,%s,%d,%d,%.1e,%s,%.1f,%s,%s,%s\n",
            D, TARGET_P, result.value, result.wall_time_seconds,
            result.evaluations, result.starts, result.iterations,
            string(result.converged), result.restarts, result.maxiters,
            result.g_abstol, string(result.best_start_kind),
            result.best_start_wall_time_seconds, "true", string(started_at), string(finished_at))
    end
end

function run_degree(D::Int, seed::Int, timing_file::String)
    file = results_path(D)
    ensure_results_file!(file, D)

    existing_value, _, _ = get_best_angles(file, D, TARGET_P)
    if existing_value > -Inf
        @printf("D=%d p=13 already complete: ctilde=%.12f -- skipping\n", D, existing_value)
        return true
    end

    previous_value, previous_gamma, previous_beta = get_best_angles(file, D, TARGET_P - 1)
    if previous_value == -Inf
        @printf("D=%d missing p=12 warm-start -- skipping for now\n", D)
        return false
    end

    warm = extend_angles(QAOAAngles(previous_gamma, previous_beta), TARGET_P)
    params = TreeParams(K, D, TARGET_P)
    budget = QaoaXorsat.depth_optimization_budget(TARGET_P, 8, 200)
    g_abstol = QaoaXorsat.depth_g_abstol(TARGET_P)

    started_at = now()
    @printf("D=%d p=13 starting at %s (warm p=12 ctilde=%.12f, maxiters=%d, g=%.0e, checkpointed=true)\n",
        D, Dates.format(started_at, "yyyy-mm-dd HH:MM:SS"), previous_value,
        budget.maxiters, g_abstol)
    flush(stdout)

    result = optimize_angles(params;
        clause_sign=CLAUSE_SIGN,
        initial_guesses=[warm],
        initial_guess_kind=:warm,
        restarts=budget.restarts,
        maxiters=budget.maxiters,
        autodiff=:adjoint,
        rng=MersenneTwister(seed + D),
        g_abstol,
        checkpointed=true,
        on_evaluation=(start_index, evaluations, elapsed, value, gnorm) -> begin
            @printf("  [D=%d p=13 start=%d] eval %d: ctilde=%.10f |grad|=%.2e %s\n",
                D, start_index, evaluations, value, gnorm, fmt_time(elapsed))
            flush(stdout)
        end,
    )
    finished_at = now()

    append_result!(file, D, result)
    append_timing!(timing_file, D, result, started_at, finished_at)

    @printf("D=%d p=13 done: ctilde=%.12f in %s (evals=%d, iters=%d, converged=%s)\n",
        D, result.value, fmt_time(result.wall_time_seconds), result.evaluations,
        result.iterations, result.converged)
    notify("D=$D p=13 done: ctilde=$(round(result.value, digits=6)) in $(fmt_time(result.wall_time_seconds))")
    flush(stdout)
    return true
end

function main()
    D_start = parse(Int, get(ARGS, 1, "3"))
    D_end = parse(Int, get(ARGS, 2, "9"))
    seed = parse(Int, get(ARGS, 3, "42"))
    D_start <= D_end || throw(ArgumentError("D_START must be <= D_END"))

    timing_file = timing_path()
    ensure_timing_file!(timing_file)

    println("MaxCut p=13 checkpointed timed sweep")
    println("Degrees: D=$D_start..$D_end")
    println("Threads: $(Threads.nthreads())")
    println("Timing: $timing_file")
    println()

    total_start = time()
    for D in D_start:D_end
        try
            run_degree(D, seed, timing_file)
        catch error
            @printf("D=%d p=13 FAILED: %s\n", D, sprint(showerror, error))
            notify("D=$D p=13 FAILED")
        end
        println()
        flush(stdout)
    end

    total = time() - total_start
    @printf("p=13 sweep pass done in %s\n", fmt_time(total))
    notify("p=13 sweep pass done in $(fmt_time(total))")
end

main()