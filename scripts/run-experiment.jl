#!/usr/bin/env julia
#
# Cluster-ready experiment runner with checkpoint/resume and blob storage.
#
# Usage:
#   julia --project=. -t auto scripts/run-experiment.jl experiments/full-table.toml [JOB_INDEX]
#
# If JOB_INDEX is given, runs only that job (0-indexed). Otherwise runs all jobs
# in priority order. Each depth result is written to a results directory immediately,
# enabling resume after preemption.

using Dates
using Printf
using Random
using TOML
using QaoaXorsat

# ──────────────────────────────────────────────────────────────────────────────
# Result persistence
# ──────────────────────────────────────────────────────────────────────────────

function results_dir(base_dir, k, D)
    dir = joinpath(base_dir, "k$(k)-D$(D)")
    mkpath(dir)
    dir
end

function depth_result_path(dir, p)
    joinpath(dir, "p$(lpad(p, 2, '0')).json")
end

function save_depth_result(dir, p, result, job)
    path = depth_result_path(dir, p)
    ts = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS")
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"k\": $(job["k"]),")
        println(io, "  \"D\": $(job["D"]),")
        println(io, "  \"p\": $p,")
        println(io, "  \"clause_sign\": $(job["clause_sign"]),")
        println(io, "  \"value\": $(result.value),")
        println(io, "  \"wall_time_seconds\": $(result.wall_time_seconds),")
        println(io, "  \"best_start_wall_time_seconds\": $(result.best_start_wall_time_seconds),")
        println(io, "  \"evaluations\": $(result.evaluations),")
        println(io, "  \"starts\": $(result.starts),")
        println(io, "  \"iterations\": $(result.iterations),")
        println(io, "  \"converged\": $(result.converged),")
        println(io, "  \"retry_count\": $(result.retry_count),")
        println(io, "  \"best_start_kind\": \"$(result.best_start_kind)\",")
        println(io, "  \"gamma\": [$(join(result.angles.γ, ", "))],")
        println(io, "  \"beta\": [$(join(result.angles.β, ", "))],")
        println(io, "  \"timestamp\": \"$ts\",")
        println(io, "  \"hostname\": \"$(gethostname())\",")
        println(io, "  \"threads\": $(Threads.nthreads()),")
        println(io, "  \"julia_version\": \"$(VERSION)\",")
        println(io, "  \"seed\": $(job["seed"]),")
        println(io, "  \"autodiff\": \"$(job["autodiff"])\"")
        println(io, "}")
    end
    println(stderr, "  saved: $path")
    flush(stderr)
    path
end

function load_checkpoint(dir, p_max)
    # Find the highest p with a saved result
    last_p = 0
    last_angles = nothing
    for p in 1:p_max
        path = depth_result_path(dir, p)
        if isfile(path)
            # Parse the angles from JSON (simple manual parse)
            content = read(path, String)
            # Extract gamma and beta arrays
            γ_match = match(r"\"gamma\":\s*\[([^\]]+)\]", content)
            β_match = match(r"\"beta\":\s*\[([^\]]+)\]", content)
            if γ_match !== nothing && β_match !== nothing
                γ = parse.(Float64, split(γ_match[1], ","))
                β = parse.(Float64, split(β_match[1], ","))
                last_angles = QAOAAngles(γ, β)
                last_p = p
            end
        else
            break
        end
    end
    (last_p, last_angles)
end

# ──────────────────────────────────────────────────────────────────────────────
# Machine metadata
# ──────────────────────────────────────────────────────────────────────────────

function save_machine_metadata(dir)
    path = joinpath(dir, "machine.json")
    ts = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS")
    os_name = Sys.islinux() ? "linux" : Sys.isapple() ? "macos" : "other"
    mem_total = round(Sys.total_memory() / 1024^3, digits=1)
    mem_free = round(Sys.free_memory() / 1024^3, digits=1)
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"hostname\": \"$(gethostname())\",")
        println(io, "  \"julia_version\": \"$(VERSION)\",")
        println(io, "  \"threads\": $(Threads.nthreads()),")
        println(io, "  \"cpu_threads\": $(Sys.CPU_THREADS),")
        println(io, "  \"total_memory_gb\": $mem_total,")
        println(io, "  \"free_memory_gb\": $mem_free,")
        println(io, "  \"os\": \"$os_name\",")
        println(io, "  \"timestamp\": \"$ts\"")
        println(io, "}")
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Job execution
# ──────────────────────────────────────────────────────────────────────────────

function run_job(job, base_dir)
    k = job["k"]
    D = job["D"]
    p_max = job["p_max"]
    clause_sign = job["clause_sign"]
    restarts = job["restarts"]
    maxiters = job["maxiters"]
    seed = job["seed"]
    autodiff = Symbol(job["autodiff"])

    dir = results_dir(base_dir, k, D)
    save_machine_metadata(dir)

    println(stderr, "=== Job: k=$k, D=$D, p=1..$p_max, autodiff=$autodiff ===")
    flush(stderr)

    # Check for existing checkpoint
    last_p, last_angles = load_checkpoint(dir, p_max)
    if last_p > 0
        println(stderr, "  Resuming from checkpoint: p=$last_p")
        flush(stderr)
    end

    # Determine starting point
    p_start = last_p + 1
    if p_start > p_max
        println(stderr, "  Already complete through p=$p_max")
        flush(stderr)
        return
    end

    rng = MersenneTwister(seed)

    # If resuming, we need to warm-start from the last checkpoint
    warm_start = last_angles

    # Print CSV header
    println("p,k,D,value,wall_time_seconds,evaluations,iterations,converged")
    flush(stdout)

    for p in p_start:p_max
        budget = QaoaXorsat.depth_optimization_budget(p, restarts, maxiters)
        initial_guesses = isnothing(warm_start) ? QAOAAngles[] : [extend_angles(warm_start, p)]

        result = optimize_angles(
            TreeParams(k, D, p);
            clause_sign,
            restarts=budget.restarts,
            maxiters=budget.maxiters,
            initial_guesses,
            initial_guess_kind=:warm,
            autodiff,
            rng,
        )

        # Retry if not converged and warm-started
        if !isnothing(warm_start) && !result.converged
            retry_result = optimize_angles(
                TreeParams(k, D, p);
                clause_sign,
                restarts=0,
                maxiters=QaoaXorsat.retry_optimization_budget(budget.maxiters),
                initial_guesses=[result.angles],
                initial_guess_kind=:retry,
                autodiff,
                rng,
            )
            result = QaoaXorsat.merge_optimization_results(result, retry_result)
        end

        # Save immediately (survives preemption)
        save_depth_result(dir, p, result, job)

        # Stream to stdout
        @printf("%d,%d,%d,%.12f,%.6f,%d,%d,%s\n",
            p, k, D, result.value, result.wall_time_seconds,
            result.evaluations, result.iterations, string(result.converged))
        flush(stdout)

        warm_start = result.angles
    end

    println(stderr, "=== Job complete: k=$k, D=$D ===")
    flush(stderr)
end

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

function main()
    length(ARGS) >= 1 || error("Usage: julia run-experiment.jl <spec.toml> [JOB_INDEX]")

    spec = TOML.parsefile(ARGS[1])
    jobs = spec["experiments"]["jobs"]

    # Sort by priority
    sort!(jobs, by=j -> get(j, "priority", 99))

    base_dir = get(ENV, "QAOA_RESULTS_DIR", joinpath(dirname(ARGS[1]), "..", "results", "cluster"))

    if length(ARGS) >= 2
        # Run single job by index
        idx = parse(Int, ARGS[2])
        0 <= idx < length(jobs) || error("JOB_INDEX must be 0..$(length(jobs)-1)")
        run_job(jobs[idx + 1], base_dir)
    else
        # Run all jobs in priority order
        for (i, job) in enumerate(jobs)
            println(stderr, "\n--- Job $(i)/$(length(jobs)) ---")
            run_job(job, base_dir)
        end
    end
end

main()
