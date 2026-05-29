#!/usr/bin/env julia
# Run MaxCut k=2, arbitrary D, p=14 using saved p=13 warm-start from
# results/maxcut-k2-d{D}-sweep.csv.
#
# Usage:
#   julia --project=. scripts/run_p14_d_warm.jl 4
#
# Output files:
#   results/maxcut-k2-d{D}-p14-angles.txt
#   results/maxcut-k2-d{D}-p14-progress.csv
#   results/maxcut-k2-p14-timing.csv   (appended; includes D column)

using QaoaXorsat
using Printf, Dates, Random

const REPO_ROOT    = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_CSV   = joinpath(REPO_ROOT, "results", "maxcut-k2-p14-timing.csv")

const K = 2
const P = 14
const CLAUSE_SIGN = -1

function parse_D()
    if length(ARGS) < 1
        error("Usage: julia --project=. scripts/run_p14_d_warm.jl <D>")
    end
    D = tryparse(Int, ARGS[1])
    D === nothing && error("D must be an integer, got: $(ARGS[1])")
    D < 3 && error("D must be >= 3")
    return D
end

function write_angles(path, D::Int, angles::QAOAAngles, ctilde::Float64, wall::Float64, iters::Int)
    gamma = getproperty(angles, Symbol("γ"))
    beta  = getproperty(angles, Symbol("β"))
    open(path, "w") do io
        println(io, "# MaxCut k=$K D=$D p=$P angles -- $(now())")
        println(io, "# ctilde=$ctilde iterations=$iters wall_seconds=$wall")
        println(io, "gamma=", join(gamma, ';'))
        println(io, "beta=",  join(beta, ';'))
    end
end

function append_timing_row(path; row::NamedTuple)
    new_file = !isfile(path)
    open(path, "a") do io
        if new_file
            println(io, "# MaxCut per-D timing at p=$P -- created $(now())")
            println(io, "# autodiff=:charge_adjoint, warm-started from saved p=13")
            println(io, "k,D,p,ctilde,wall_seconds,evaluations,iterations,converged,",
                       "fwd_seconds,grad_seconds,grad_over_fwd,peak_rss_gb,",
                       "started_at,finished_at")
        end
        @printf(io, "%d,%d,%d,%.12f,%.1f,%d,%d,%s,%.2f,%.2f,%.2f,%.2f,%s,%s\n",
            row.k, row.D, row.p, row.ctilde, row.wall_seconds,
            row.evaluations, row.iterations, row.converged,
            row.fwd_seconds, row.grad_seconds, row.grad_over_fwd,
            row.peak_rss_gb, row.started_at, row.finished_at)
    end
end

function main()
    D = parse_D()
    sweep_csv = joinpath(REPO_ROOT, "results", "maxcut-k2-d$(D)-sweep.csv")
    angles_out = joinpath(REPO_ROOT, "results", "maxcut-k2-d$(D)-p14-angles.txt")
    progress_csv = joinpath(REPO_ROOT, "results", "maxcut-k2-d$(D)-p14-progress.csv")
    snapshots_csv = joinpath(REPO_ROOT, "results", "maxcut-k2-d$(D)-p14-angle-snapshots.csv")

    println("=" ^ 60)
    println("MaxCut k=$K D=$D p=$P -- charge_adjoint, warm-start from saved p=13")
    println("Started:    ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("PID:        ", getpid())
    println("Julia:      ", VERSION)
    println("Threads:    ", Threads.nthreads())
    println("Total RAM:  ", round(Sys.total_memory() / (1024^3), digits=1), " GB")
    println("=" ^ 60)
    flush(stdout)

    print("Loading saved p=13 angles... ")
    flush(stdout)
    store = CsvResultStore(sweep_csv, :sweep)
    p13_record = read_best_record(store, K, D, P - 1)
    p13_record === nothing && error("no p=13 row for k=$K D=$D found in $sweep_csv")
    ctilde13 = p13_record.value
    angles13 = p13_record.angles
    @printf("done.  p=13 ctilde = %.12f\n", ctilde13)
    flush(stdout)

    warm = extend_angles(angles13, P)
        warm_gamma = getproperty(warm, Symbol("γ"))
        warm_beta  = getproperty(warm, Symbol("β"))
    @printf("Extended to p=%d  (gamma length=%d, beta length=%d)\n",
            P, length(warm_gamma), length(warm_beta))
    flush(stdout)

    println()
    print("[timing] single charge forward eval at p=$P... ")
    flush(stdout)
    GC.gc(true)
    t_fwd = @elapsed v_fwd = charge_parity_expectation(TreeParams(K, D, P), warm)
    @printf("%.2fs   (warm ctilde = %.12f)\n", t_fwd, v_fwd)
    @printf("[memory] RSS after fwd: %.2f GB\n", Sys.maxrss() / 1e9)
    flush(stdout)

    print("[timing] single charge adjoint gradient at p=$P... ")
    flush(stdout)
    GC.gc(true)
    t_grad = @elapsed (vg, ggamma, gbeta) =
        QaoaXorsat.charge_expectation_and_gradient(TreeParams(K, D, P), warm; clause_sign=CLAUSE_SIGN)
    @printf("%.2fs   (%.2fx fwd, ||g|| = %.3e)\n",
            t_grad, t_grad / max(t_fwd, eps()),
            sqrt(sum(abs2, ggamma) + sum(abs2, gbeta)))
    @printf("[memory] RSS after grad: %.2f GB\n", Sys.maxrss() / 1e9)
    flush(stdout)

    progress_io = open(progress_csv, "w")
    println(progress_io, "# MaxCut k=$K D=$D p=$P optimization progress -- $(now())")
    println(progress_io, "wall_seconds,evaluations,ctilde,grad_norm,rss_gb")
    flush(progress_io)

    println()
    println("=" ^ 60)
    println("Optimizing p=$P (autodiff=:charge_adjoint, restarts=0)...")
    println("=" ^ 60)
    flush(stdout)

    started_at = now()
    t_opt = @elapsed result = optimize_angles(
        TreeParams(K, D, P);
        clause_sign = CLAUSE_SIGN,
        restarts = 0,
        maxiters = 200,
        autodiff = :charge_adjoint,
        initial_guesses = [warm],
        initial_guess_kind = :warm,
        rng = MersenneTwister(56 + D),
        on_evaluation = function(start_idx, evals, elapsed, val, gnorm)
            rss_gb = Sys.maxrss() / 1e9
            @printf(progress_io, "%.1f,%d,%.12f,%.3e,%.2f\n",
                    elapsed, evals, val, gnorm, rss_gb)
            flush(progress_io)
            @printf("  [%6.0fs] evals=%3d  ctilde=%.12f  ||g||=%.2e  rss=%.2fGB\n",
                    elapsed, evals, val, gnorm, rss_gb)
            flush(stdout)
        end,
        on_angle_snapshot = snapshot -> write_angle_snapshot!(snapshots_csv, snapshot),
    )

    finished_at = now()
    close(progress_io)

    rss_gb = Sys.maxrss() / 1e9
    println()
    println("=" ^ 60)
    @printf("  ctilde:       %.12f\n", result.value)
    @printf("  iterations:   %d\n", result.iterations)
    @printf("  evaluations:  %d\n", result.evaluations)
    @printf("  converged:    %s\n", result.converged)
    @printf("  wall time:    %.1fs  (%.2f hr)\n", t_opt, t_opt / 3600)
    @printf("  peak RSS:     %.2f GB\n", rss_gb)
    println("=" ^ 60)
    flush(stdout)

    write_angles(angles_out, D, result.angles, result.value, t_opt, result.iterations)
    append_timing_row(OUTPUT_CSV; row=(
        k = K,
        D = D,
        p = P,
        ctilde = result.value,
        wall_seconds = t_opt,
        evaluations = result.evaluations,
        iterations = result.iterations,
        converged = result.converged,
        fwd_seconds = t_fwd,
        grad_seconds = t_grad,
        grad_over_fwd = t_grad / max(t_fwd, eps()),
        peak_rss_gb = rss_gb,
        started_at = Dates.format(started_at, "yyyy-mm-ddTHH:MM:SS"),
        finished_at = Dates.format(finished_at, "yyyy-mm-ddTHH:MM:SS"),
    ))

    @printf("Wrote angles  -> %s\n", angles_out)
    @printf("Wrote timing  -> %s\n", OUTPUT_CSV)
    @printf("Wrote progress-> %s\n", progress_csv)
    @printf("Wrote snapshots-> %s\n", snapshots_csv)
    println("Finished:   ", Dates.format(finished_at, "yyyy-mm-dd HH:MM:SS"))
    flush(stdout)
end

main()
