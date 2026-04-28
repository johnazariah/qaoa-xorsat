using QaoaXorsat, Printf, Dates

k, D, p = 2, 4, 12
clause_sign = -1

gamma11 = [3.354346203192871,3.566965198690236,3.6057871770707037,3.638933128619131,3.6606445991578647,3.677703374198545,3.699595200338122,3.7288891097628385,3.7729276260846794,3.8519679911352247,3.9383725747677105]
beta11 = [2.220170157856824,2.118029720294113,2.067673327258194,2.0514204517470547,2.026588989451891,2.0045299186861,1.9750972926015984,1.9394127508380479,1.8731037214166093,1.7952521890162878,1.673116765886924]
warm = extend_angles(QAOAAngles(gamma11, beta11), p)

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

# macOS notification
run(`osascript -e "display notification \"MaxCut D=$D p=$p done: c̃=$(round(result.value, digits=6)) in $(round(dt/60, digits=1))min\" with title \"QAOA Compute Complete\""`)
