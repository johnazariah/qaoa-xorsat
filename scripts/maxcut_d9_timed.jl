#!/usr/bin/env julia
# Fresh MaxCut D=9 sweep with live timing — demonstrating the full pipeline speed.
#
# Usage: julia --project=. -t 16 scripts/maxcut_d9_timed.jl

using QaoaXorsat, Printf, Dates

k, D = 2, 9
clause_sign = -1
p_max = 13  # 30 GB adjoint cache at p=13, fits in 64 GB

results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")
mkpath(dirname(results_file))

# Fresh CSV
open(results_file, "w") do io
    println(io, "# MaxCut k=2 D=$D TIMED sweep — $(now())")
    println(io, "k,D,p,ctilde,wall_seconds,gamma,beta")
end

println("╔══════════════════════════════════════════════════════════╗")
println("║  MaxCut D=$D — Full Timed Sweep p=1..$p_max              ║")
println("║  $(now())                            ║")
println("║  Threads: $(Threads.nthreads())                                          ║")
println("╚══════════════════════════════════════════════════════════╝")
println()
@printf("  %-4s  %-14s  %-12s  %-12s  %s\n", "p", "c̃", "Δc̃", "this depth", "cumulative")
@printf("  %-4s  %-14s  %-12s  %-12s  %s\n", "──", "──────────────", "────────────", "────────────", "────────────")
flush(stdout)

sweep_start = time()
warm = QAOAAngles[]
prev_value = 0.5

for p in 1:p_max
    params = TreeParams(k, D, p)
    guesses = isempty(warm) ? QAOAAngles[] : [extend_angles(warm[1], p)]

    t0 = time()
    local result
    try
        result = optimize_angles(params;
            clause_sign,
            initial_guesses=guesses,
            restarts=(p <= 6 ? 2 : 0),  # small restarts at low p (cheap), warm-only at high p
            g_abstol=1e-6,
            on_evaluation = (chunk, evals, elapsed, val, gnorm) -> begin
                if p >= 10  # only log at high p where evals are slow
                    @printf("    [p=%d] eval %d: c̃=%.10f  |∇|=%.2e  %.0fs\n",
                            p, evals, val, gnorm, elapsed)
                    flush(stdout)
                end
            end
        )
    catch e
        dt = time() - t0
        cumulative = time() - sweep_start
        @printf("  p=%-2d  FAILED after %.1fs (cumulative %.1fs): %s\n",
                p, dt, cumulative, sprint(showerror, e))
        println("\n  Stopping — likely out of memory.")
        flush(stdout)
        break
    end
    dt = time() - t0
    cumulative = time() - sweep_start
    delta = result.value - prev_value

    # Format time nicely
    fmt_time(s) = s < 60 ? @sprintf("%.1fs", s) :
                  s < 3600 ? @sprintf("%.1fmin", s/60) :
                  @sprintf("%.1fh", s/3600)

    @printf("  p=%-2d  %.10f  %+.8f  %-12s  %s\n",
            p, result.value, delta, fmt_time(dt), fmt_time(cumulative))
    flush(stdout)

    # Save
    gamma_str = join(string.(result.angles.γ), ';')
    beta_str = join(string.(result.angles.β), ';')
    open(results_file, "a") do io
        @printf(io, "%d,%d,%d,%.12f,%.1f,%s,%s\n",
            k, D, p, result.value, dt, gamma_str, beta_str)
    end

    warm = [result.angles]
    prev_value = result.value
end

total = time() - sweep_start
println()
println("╔══════════════════════════════════════════════════════════╗")
@printf("║  Done! Total wall time: %-33s║\n",
    total < 3600 ? @sprintf("%.1f min", total/60) : @sprintf("%.1f hours", total/3600))
println("╚══════════════════════════════════════════════════════════╝")

# macOS notification
try
    run(`osascript -e "display notification \"MaxCut D=$D sweep done in $(round(total/60, digits=1))min\" with title \"QAOA Sweep Complete\" sound name \"Glass\""`)
catch end
