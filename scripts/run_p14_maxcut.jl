using QaoaXorsat, Printf, Dates, Random

function main()
    println("MaxCut p=14 D=3 -- charge evaluator throughout")
    println("Started: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("RAM: ", Sys.total_memory() / (1024^3), " GB")
    println()
    flush(stdout)

    # All phases use charge + central FD
    println("=== Warm-up: p=1..13 with autodiff=:charge ===")
    flush(stdout)

    warm_start = nothing

    for p in 1:13
        params = TreeParams(2, 3, p)
        guesses = isnothing(warm_start) ? QAOAAngles[] : [extend_angles(warm_start, p)]
        t = @elapsed r = optimize_angles(params;
            clause_sign=-1, restarts=(p <= 5 ? 4 : 0), maxiters=400,
            autodiff=:charge,
            initial_guesses=guesses, initial_guess_kind=:warm,
            rng=MersenneTwister(42 + p))
        @printf("  p=%2d: ctilde=%.10f  %3d iters  %7.1fs\n", p, r.value, r.iterations, t)
        flush(stdout)
        if QaoaXorsat.is_valid_qaoa_value(r.value)
            warm_start = r.angles
        end
    end

    # Phase 2: p=14
    println()
    println("=== p=14 with autodiff=:charge ===")
    flush(stdout)

    params_14 = TreeParams(2, 3, 14)
    warm_14 = extend_angles(warm_start, 14)

    @printf("  Single charge eval at p=14... ")
    flush(stdout)
    t_eval = @elapsed charge_parity_expectation(params_14, warm_14)
    @printf("%.1fs\n", t_eval)
    @printf("  Estimated per-iteration: %.0fs\n", 29*t_eval)
    println()
    flush(stdout)

    println("  Running optimizer (restarts=0, maxiters=100)...")
    flush(stdout)

    t_p14 = @elapsed r14 = optimize_angles(params_14;
        clause_sign=-1, restarts=0, maxiters=100,
        autodiff=:charge,
        initial_guesses=[warm_14], initial_guess_kind=:warm,
        rng=MersenneTwister(42 + 14),
        on_evaluation=(si, evals, elapsed, val, gnorm) -> begin
            @printf("    [eval %d] %.0fs: ctilde=%.10f gnorm=%.2e\n",
                    evals, elapsed, val, gnorm)
            flush(stdout)
        end)

    println()
    println("===================================================")
    @printf("  p=14 RESULT: ctilde = %.10f\n", r14.value)
    @printf("  Iterations:  %d\n", r14.iterations)
    @printf("  Wall time:   %.1fs (%.1f min)\n", t_p14, t_p14/60)
    @printf("  Converged:   %s\n", r14.converged)
    println("===================================================")
    println()

    # Print angles
    println("  gamma = [")
    for (i, g) in enumerate(r14.angles.γ)
        @printf("    %.10f%s\n", g, i < 14 ? "," : "")
    end
    println("  ]")
    println("  beta = [")
    for (i, b) in enumerate(r14.angles.β)
        @printf("    %.10f%s\n", b, i < 14 ? "," : "")
    end
    println("  ]")

    println()
    println("Finished: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    flush(stdout)
end

main()
