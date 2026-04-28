#!/usr/bin/env julia
# MaxCut p=13 sweep with CPU checkpointing — D=3..6
# Uses checkpointed=true to fit in 64 GB (√p memory instead of p)
#
# Usage: julia --project=. -t 16 scripts/maxcut_p13_checkpointed.jl

using QaoaXorsat, Printf, Dates, Random

clause_sign = -1
k = 2

fmt_time(s) = s < 60 ? @sprintf("%.1fs", s) :
              s < 3600 ? @sprintf("%.1fmin", s/60) :
              @sprintf("%.1fh", s/3600)

function notify(msg)
    try run(`osascript -e "display notification \"$msg\" with title \"QAOA p=13\" sound name \"Glass\""`) catch end
end

function get_best_angles(file, k, D, target_p)
    best_val = -Inf
    best_gamma = Float64[]
    best_beta = Float64[]
    isfile(file) || return (best_val, best_gamma, best_beta)
    for line in eachline(file)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 7 || continue
        lk = tryparse(Int, fields[1]); lk === nothing && continue
        lD = tryparse(Int, fields[2]); lD === nothing && continue
        lp = tryparse(Int, fields[3]); lp === nothing && continue
        lv = tryparse(Float64, fields[4]); lv === nothing && continue
        (lk == k && lD == D && lp == target_p) || continue
        if lv > best_val
            best_val = lv
            best_gamma = parse.(Float64, split(fields[6], ';'))
            best_beta = parse.(Float64, split(fields[7], ';'))
        end
    end
    return (best_val, best_gamma, best_beta)
end

grand_start = time()
p = 13

println("╔══════════════════════════════════════════════════════════╗")
println("║  MaxCut p=$p with CPU Checkpointing — D=3..6           ║")
println("║  $(now())                            ║")
println("║  Threads: $(Threads.nthreads())                                          ║")
println("╚══════════════════════════════════════════════════════════╝")
println()

for D in [3, 4, 5, 6]
    results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")

    prev_val, prev_gamma, prev_beta = get_best_angles(results_file, k, D, p - 1)
    if prev_val == -Inf
        @printf("  ⚠ D=%d: no p=%d warm-start — skipping\n", D, p-1)
        continue
    end
    @printf("━━━ D=%d p=%d — warm-start from p=%d (c̃=%.10f) ━━━\n", D, p, p-1, prev_val)
    @printf("  ▶ starting at %s\n", Dates.format(now(), "HH:MM:SS"))
    flush(stdout)

    warm = extend_angles(QAOAAngles(prev_gamma, prev_beta), p)
    params = TreeParams(k, D, p)

    t0 = time()
    try
        result = optimize_angles(params;
            clause_sign,
            initial_guesses=[warm],
            restarts=0,
            g_abstol=1e-6,
            checkpointed=true,
            on_evaluation = (chunk, evals, elapsed, val, gnorm) -> begin
                @printf("    [D=%d p=%d] eval %d: c̃=%.10f  |∇|=%.2e  %s\n",
                        D, p, evals, val, gnorm, fmt_time(elapsed))
                flush(stdout)
            end
        )
        dt = time() - t0

        @printf("  ✓ D=%d p=%d: c̃=%.12f in %s\n", D, p, result.value, fmt_time(dt))
        notify("D=$D p=$p: c̃=$(round(result.value, digits=6)) in $(fmt_time(dt))")

        gamma_str = join(string.(result.angles.γ), ';')
        beta_str = join(string.(result.angles.β), ';')
        open(results_file, "a") do io
            @printf(io, "%d,%d,%d,%.12f,%.1f,%s,%s\n",
                k, D, p, result.value, dt, gamma_str, beta_str)
        end
    catch e
        dt = time() - t0
        @printf("  ✗ D=%d p=%d FAILED after %s: %s\n", D, p, fmt_time(dt), sprint(showerror, e))
        notify("D=$D p=$p FAILED!")
    end
    println()
    flush(stdout)
end

grand_total = time() - grand_start
println("╔══════════════════════════════════════════════════════════╗")
@printf("║  Done! Total: %-42s║\n", fmt_time(grand_total))
println("╚══════════════════════════════════════════════════════════╝")
notify("p=13 sweep done! Total: $(fmt_time(grand_total))")
