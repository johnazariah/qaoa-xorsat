#!/usr/bin/env julia
# Overnight MaxCut sweep: D=3..6, push each to p=13
# Runs sequentially (p=13 needs ~30 GB, can't parallel)
# Picks up from existing CSV data, warm-starts from best available.
#
# Usage: julia --project=. -t 16 scripts/overnight_maxcut_p13.jl 2>&1 | tee /tmp/overnight-maxcut.log

using QaoaXorsat, Printf, Dates

clause_sign = -1
k = 2

function fmt_time(s)
    s < 60    && return @sprintf("%.1fs", s)
    s < 3600  && return @sprintf("%.1fmin", s/60)
    return @sprintf("%.1fh", s/3600)
end

function notify(msg)
    try
        run(`osascript -e "display notification \"$msg\" with title \"QAOA Overnight\" sound name \"Glass\""`)
    catch end
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

function run_single(D, p, results_file)
    prev_val, prev_gamma, prev_beta = get_best_angles(results_file, k, D, p - 1)
    if prev_val == -Inf
        @printf("  ⚠ No p=%d warm-start for D=%d — skipping\n", p-1, D)
        return nothing
    end

    warm = extend_angles(QAOAAngles(prev_gamma, prev_beta), p)
    params = TreeParams(k, D, p)

    t0 = time()
    result = optimize_angles(params;
        clause_sign,
        initial_guesses=[warm],
        restarts=0,
        g_abstol=1e-6,
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

    return (value=result.value, time=dt, angles=result.angles)
end

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
grand_start = time()

println("╔══════════════════════════════════════════════════════════╗")
println("║  Overnight MaxCut Sweep — D=3..6, target p=13          ║")
println("║  $(now())                            ║")
println("║  Threads: $(Threads.nthreads())                                          ║")
println("╚══════════════════════════════════════════════════════════╝")
println()

for D in [3, 4, 5, 6]
    results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")
    current_max_p = get_max_p(results_file, k, D)

    println("━━━ D=$D: have p=1..$current_max_p, targeting p=13 ━━━")
    flush(stdout)

    for p in (current_max_p + 1):13
        @printf("  ▶ D=%d p=%d starting at %s\n", D, p, Dates.format(now(), "HH:MM:SS"))
        flush(stdout)

        try
            r = run_single(D, p, results_file)
            if r === nothing
                println("  ✗ Skipped (no warm-start)")
                break
            end
            @printf("  ✓ D=%d p=%d: c̃=%.10f in %s\n", D, p, r.value, fmt_time(r.time))
            notify("D=$D p=$p done: c̃=$(round(r.value, digits=6)) in $(fmt_time(r.time))")
        catch e
            @printf("  ✗ D=%d p=%d FAILED: %s\n", D, p, sprint(showerror, e))
            notify("D=$D p=$p FAILED!")
            break
        end
        flush(stdout)
    end
    println()
end

grand_total = time() - grand_start
println("╔══════════════════════════════════════════════════════════╗")
@printf("║  All done! Total: %-38s║\n", fmt_time(grand_total))
println("╚══════════════════════════════════════════════════════════╝")
notify("Overnight sweep complete! Total: $(fmt_time(grand_total))")
