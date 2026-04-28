using QaoaXorsat, Printf, Dates

k, D, p = 2, 6, 12
clause_sign = -1

# Check what we have for D=6
results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d6-sweep.csv")
if isfile(results_file)
    for line in reverse(readlines(results_file))
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 4 || continue
        lp = tryparse(Int, fields[3])
        lv = tryparse(Float64, fields[4])
        if lp !== nothing && lv !== nothing
            @printf("Latest in CSV: p=%d c̃=%.10f\n", lp, lv)
            break
        end
    end
end

# Warm-start: need p=11 angles for D=6
# Read from CSV
function get_best_angles(file, k, D, target_p)
    best_val = -Inf
    best_gamma = Float64[]
    best_beta = Float64[]
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
    return best_val, best_gamma, best_beta
end

prev_val, prev_gamma, prev_beta = get_best_angles(results_file, k, D, p - 1)
if prev_val == -Inf
    error("No p=$(p-1) result found for D=$D — can't warm-start")
end
@printf("Warm-start from p=%d: c̃=%.10f\n", p - 1, prev_val)

warm = extend_angles(QAOAAngles(prev_gamma, prev_beta), p)
params = TreeParams(k, D, p)

println("=== MaxCut D=$D p=$p — $(now()) ===")
println("Threads: $(Threads.nthreads())")
flush(stdout)

t0 = time()
result = optimize_angles(params;
    clause_sign,
    initial_guesses=[warm],
    restarts=0,
    g_abstol=1e-6,
    on_evaluation = (chunk, evals, elapsed, val, gnorm) -> begin
        @printf("  chunk %d, eval %d: c̃=%.10f  |∇|=%.2e  elapsed=%.0fs\n",
                chunk, evals, val, gnorm, elapsed)
        flush(stdout)
    end
)
dt = time() - t0
@printf("p=%d  c̃=%.12f  time=%.1fs\n", p, result.value, dt)
flush(stdout)

# Append to CSV
open(results_file, "a") do io
    gamma_str = join(string.(result.angles.γ), ';')
    beta_str = join(string.(result.angles.β), ';')
    @printf(io, "%d,%d,%d,%.12f,%.1f,%s,%s\n",
        k, D, p, result.value, dt, gamma_str, beta_str)
end

# macOS notification
run(`osascript -e "display notification \"MaxCut D=$D p=$p done: c̃=$(round(result.value, digits=6)) in $(round(dt/60, digits=1))min\" with title \"QAOA Compute Complete\""`)
