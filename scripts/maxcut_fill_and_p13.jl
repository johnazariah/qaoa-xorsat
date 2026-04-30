#!/usr/bin/env julia
# MaxCut sweep: fill gaps and push to p=13 with checkpointing
# Runs sequentially: D=4 p=12, D=7 p=11..12, D=8 p=10..12, then p=13 for D=3..6
#
# Usage: julia --project=. -t 16 scripts/maxcut_fill_and_p13.jl 2>&1 | tee /tmp/maxcut-fill.log

using QaoaXorsat, Printf, Dates, Random

clause_sign = -1
k = 2

fmt_time(s) = s < 60 ? @sprintf("%.1fs", s) :
              s < 3600 ? @sprintf("%.1fmin", s/60) :
              @sprintf("%.1fh", s/3600)

function notify(msg)
    try run(`osascript -e "display notification \"$msg\" with title \"MaxCut Sweep\" sound name \"Glass\""`) catch end
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

function get_max_p(file, k, D)
    max_p = 0
    isfile(file) || return 0
    for line in eachline(file)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 4 || continue
        lk = tryparse(Int, fields[1]); lk === nothing && continue
        lD = tryparse(Int, fields[2]); lD === nothing && continue
        lp = tryparse(Int, fields[3]); lp === nothing && continue
        lv = tryparse(Float64, fields[4]); lv === nothing && continue
        (lk == k && lD == D) || continue
        QaoaXorsat.is_valid_qaoa_value(lv) || continue
        max_p = max(max_p, lp)
    end
    return max_p
end

function run_depth(D, p; use_checkpointing=false)
    results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")

    # Check if already done
    existing, _, _ = get_best_angles(results_file, k, D, p)
    if existing > -Inf
        @printf("  ⏭ D=%d p=%d already in CSV (c̃=%.10f) — skipping\n", D, p, existing)
        return existing
    end

    prev_val, prev_gamma, prev_beta = get_best_angles(results_file, k, D, p - 1)
    if prev_val == -Inf
        @printf("  ⚠ D=%d: no p=%d warm-start — skipping\n", D, p-1)
        return -Inf
    end

    @printf("  ▶ D=%d p=%d (warm from p=%d c̃=%.10f) checkpointed=%s at %s\n",
            D, p, p-1, prev_val, use_checkpointing, Dates.format(now(), "HH:MM:SS"))
    flush(stdout)

    warm = extend_angles(QAOAAngles(prev_gamma, prev_beta), p)
    params = TreeParams(k, D, p)

    t0 = time()
    result = optimize_angles(params;
        clause_sign,
        initial_guesses=[warm],
        restarts=0,
        g_abstol=1e-6,
        checkpointed=use_checkpointing,
        on_evaluation = (chunk, evals, elapsed, val, gnorm) -> begin
            @printf("    [D=%d p=%d] eval %d: c̃=%.10f  |∇|=%.2e  %s\n",
                    D, p, evals, val, gnorm, fmt_time(elapsed))
            flush(stdout)
        end
    )
    dt = time() - t0

    # Save to CSV
    gamma_str = join(string.(result.angles.γ), ';')
    beta_str = join(string.(result.angles.β), ';')
    open(results_file, "a") do io
        @printf(io, "%d,%d,%d,%.12f,%.1f,%s,%s\n",
            k, D, p, result.value, dt, gamma_str, beta_str)
    end

    @printf("  ✓ D=%d p=%d: c̃=%.12f in %s\n", D, p, result.value, fmt_time(dt))
    notify("D=$D p=$p: c̃=$(round(result.value, digits=6)) in $(fmt_time(dt))")
    flush(stdout)
    return result.value
end

# ═══════════════════════════════════════════════════════════════
grand_start = time()

println("╔══════════════════════════════════════════════════════════╗")
println("║  MaxCut: Fill Gaps + Push to p=13                      ║")
println("║  $(now())                            ║")
println("║  Threads: $(Threads.nthreads())                                          ║")
println("╚══════════════════════════════════════════════════════════╝")
println()

# Phase 1: Fill gaps (no checkpointing needed, p≤12)
println("━━━ Phase 1: Fill gaps ━━━")
run_depth(4, 12)           # D=4 p=12 — lost data
run_depth(7, 11)           # D=7 p=11
run_depth(7, 12)           # D=7 p=12
run_depth(8, 10)           # D=8 p=10
run_depth(8, 11)           # D=8 p=11
run_depth(8, 12)           # D=8 p=12
println()

# Phase 2: p=13 with checkpointing (D=3..6)
println("━━━ Phase 2: p=13 with checkpointing ━━━")
for D in [3, 4, 5, 6]
    run_depth(D, 13; use_checkpointing=true)
    println()
end

grand_total = time() - grand_start
println("╔══════════════════════════════════════════════════════════╗")
@printf("║  Done! Total: %-42s║\n", fmt_time(grand_total))
println("╚══════════════════════════════════════════════════════════╝")
notify("Full sweep done! Total: $(fmt_time(grand_total))")
