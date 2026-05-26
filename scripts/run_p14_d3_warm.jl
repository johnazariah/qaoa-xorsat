#!/usr/bin/env julia
# Run MaxCut k=2 D=3 p=14 using a saved p=13 warm-start (skip warm-up sweep).
#
# - Reads optimal p=13 angles from results/maxcut-k2-d3-sweep.csv
# - Times a single forward eval and a single charge-adjoint gradient at p=14
# - Runs optimize_angles with autodiff=:charge_adjoint, restarts=0
# - Reports progress every ~30s via the on_evaluation callback
# - Appends a row to results/maxcut-k2-p14-timing.csv
#
# MEASURED on a 64 GB M-series Mac (2026-05-26):
#   forward eval  30.7s
#   RSS after fwd 25.9 GB     <-- jetsam killed the process during the
#                                  first adjoint gradient (peak likely ~40 GB)
# => DO NOT RUN ON THE MAC.  Run on a >= 96 GB host.
#
# Expected on a 128 GB Xeon:
#   forward eval  ~60-90s    (Xeon per-core ~1.5-2x slower than M-series)
#   adjoint grad  ~300-600s  (~5x fwd)
#   ~30 iters     ~6-12 hr
#   memory peak   ~45 GB     (fits comfortably)
#
# See scripts/HANDOFF-p14-xeon.md for end-to-end instructions.

using QaoaXorsat
using Printf, Dates, Random, DelimitedFiles

const REPO_ROOT      = normpath(joinpath(@__DIR__, ".."))
const SWEEP_CSV      = joinpath(REPO_ROOT, "results", "maxcut-k2-d3-sweep.csv")
const OUTPUT_CSV     = joinpath(REPO_ROOT, "results", "maxcut-k2-p14-timing.csv")
const ANGLES_OUT     = joinpath(REPO_ROOT, "results", "maxcut-k2-d3-p14-angles.txt")
const PROGRESS_CSV   = joinpath(REPO_ROOT, "results", "maxcut-k2-d3-p14-progress.csv")

const K = 2
const D = 3
const P = 14
const CLAUSE_SIGN = -1   # MaxCut convention used by previous sweeps

function read_p13_angles(path::AbstractString)
    isfile(path) || error("p=13 sweep CSV not found: $path")
    # Files may contain multiple appended sweeps; we want the LAST p=13 row so
    # we read the whole file and keep the latest match.
    found_ctilde = nothing
    found_angles = nothing
    for line in eachline(path)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        parts = split(line, ',')
        length(parts) >= 7 || continue
        k_ = tryparse(Int, parts[1])
        d_ = tryparse(Int, parts[2])
        p_ = tryparse(Int, parts[3])
        (k_ === nothing || d_ === nothing || p_ === nothing) && continue
        if k_ == K && d_ == D && p_ == 13
            ctilde = parse(Float64, parts[4])
            γ = parse.(Float64, split(parts[6], ';'))
            β = parse.(Float64, split(parts[7], ';'))
            length(γ) == 13 || error("expected 13 γ values, got $(length(γ))")
            length(β) == 13 || error("expected 13 β values, got $(length(β))")
            found_ctilde = ctilde
            found_angles = QAOAAngles(γ, β)
        end
    end
    found_angles === nothing &&
        error("no p=13 row for k=$K D=$D found in $path")
    return found_ctilde, found_angles
end

function write_angles(path, angles::QAOAAngles, ctilde::Float64, wall::Float64, iters::Int)
    open(path, "w") do io
        println(io, "# MaxCut k=$K D=$D p=$P angles -- $(now())")
        println(io, "# ctilde=$ctilde iterations=$iters wall_seconds=$wall")
        println(io, "gamma=", join(angles.γ, ';'))
        println(io, "beta=",  join(angles.β, ';'))
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
    println("=" ^ 60)
    println("MaxCut k=$K D=$D p=$P -- charge_adjoint, warm-start from saved p=13")
    println("Started:    ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("PID:        ", getpid())
    println("Julia:      ", VERSION)
    println("Threads:    ", Threads.nthreads())
    println("Total RAM:  ", round(Sys.total_memory() / (1024^3), digits=1), " GB")
    println("=" ^ 60)
    flush(stdout)

    # ---- Step 1: load saved p=13 angles ----
    print("Loading saved p=13 angles... ")
    flush(stdout)
    ctilde13, angles13 = read_p13_angles(SWEEP_CSV)
    @printf("done.  p=13 c̃ = %.12f\n", ctilde13)
    flush(stdout)

    # ---- Step 2: extend to p=14 ----
    warm = extend_angles(angles13, P)
    @printf("Extended to p=%d  (γ length=%d, β length=%d)\n",
            P, length(warm.γ), length(warm.β))
    flush(stdout)

    # ---- Step 3: time a single forward eval at p=14 ----
    println()
    print("[timing] single charge forward eval at p=$P... ")
    flush(stdout)
    GC.gc(true)
    t_fwd = @elapsed v_fwd = charge_parity_expectation(QaoaXorsat.TreeParams(K, D, P), warm)
    @printf("%.2fs   (warm c̃ = %.12f)\n", t_fwd, v_fwd)
    @printf("[memory] RSS after fwd: %.2f GB\n", Sys.maxrss() / 1e9)
    flush(stdout)

    # ---- Step 4: time a single adjoint gradient at p=14 ----
    print("[timing] single charge adjoint gradient at p=$P... ")
    flush(stdout)
    GC.gc(true)
    t_grad = @elapsed (vg, γg, βg) =
        QaoaXorsat.charge_expectation_and_gradient(
            QaoaXorsat.TreeParams(K, D, P), warm; clause_sign=CLAUSE_SIGN)
    @printf("%.2fs   (%.2fx fwd, ||g|| = %.3e)\n",
            t_grad, t_grad / max(t_fwd, eps()),
            sqrt(sum(abs2, γg) + sum(abs2, βg)))
    @printf("[memory] RSS after grad: %.2f GB\n", Sys.maxrss() / 1e9)
    flush(stdout)

    # ---- Step 5: open the per-evaluation progress CSV ----
    progress_io = open(PROGRESS_CSV, "w")
    println(progress_io, "# MaxCut k=$K D=$D p=$P optimization progress -- $(now())")
    println(progress_io, "wall_seconds,evaluations,ctilde,grad_norm,rss_gb")
    flush(progress_io)

    # ---- Step 6: optimize ----
    println()
    println("=" ^ 60)
    println("Optimizing p=$P (autodiff=:charge_adjoint, restarts=0)...")
    println("=" ^ 60)
    flush(stdout)

    started_at = now()
    t0 = time_ns()
    last_print = Ref(time_ns())

    t_opt = @elapsed result = optimize_angles(
        QaoaXorsat.TreeParams(K, D, P);
        clause_sign  = CLAUSE_SIGN,
        restarts     = 0,
        maxiters     = 200,
        autodiff     = :charge_adjoint,
        initial_guesses     = [warm],
        initial_guess_kind  = :warm,
        rng          = MersenneTwister(56),
        on_evaluation = function(start_idx, evals, elapsed, val, gnorm)
            rss_gb = Sys.maxrss() / 1e9
            @printf(progress_io, "%.1f,%d,%.12f,%.3e,%.2f\n",
                    elapsed, evals, val, gnorm, rss_gb)
            flush(progress_io)
            @printf("  [%6.0fs] evals=%3d  c̃=%.12f  ||g||=%.2e  rss=%.2fGB\n",
                    elapsed, evals, val, gnorm, rss_gb)
            flush(stdout)
        end,
    )

    finished_at = now()
    close(progress_io)

    # ---- Step 7: report ----
    rss_gb = Sys.maxrss() / 1e9
    println()
    println("=" ^ 60)
    @printf("  ctilde:       %.12f\n", result.value)
    @printf("  iterations:   %d\n",    result.iterations)
    @printf("  evaluations:  %d\n",    result.evaluations)
    @printf("  converged:    %s\n",    result.converged)
    @printf("  wall time:    %.1fs  (%.2f hr)\n", t_opt, t_opt/3600)
    @printf("  peak RSS:     %.2f GB\n", rss_gb)
    println("=" ^ 60)
    flush(stdout)

    # ---- Step 8: persist ----
    write_angles(ANGLES_OUT, result.angles, result.value, t_opt, result.iterations)
    append_timing_row(OUTPUT_CSV; row=(
        k             = K,
        D             = D,
        p             = P,
        ctilde        = result.value,
        wall_seconds  = t_opt,
        evaluations   = result.evaluations,
        iterations    = result.iterations,
        converged     = result.converged,
        fwd_seconds   = t_fwd,
        grad_seconds  = t_grad,
        grad_over_fwd = t_grad / max(t_fwd, eps()),
        peak_rss_gb   = rss_gb,
        started_at    = Dates.format(started_at,  "yyyy-mm-ddTHH:MM:SS"),
        finished_at   = Dates.format(finished_at, "yyyy-mm-ddTHH:MM:SS"),
    ))
    @printf("Wrote angles  -> %s\n", ANGLES_OUT)
    @printf("Wrote timing  -> %s\n", OUTPUT_CSV)
    @printf("Wrote progress-> %s\n", PROGRESS_CSV)
    println("Finished:   ", Dates.format(finished_at, "yyyy-mm-dd HH:MM:SS"))
    flush(stdout)
end

main()
