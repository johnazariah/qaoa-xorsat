using QaoaXorsat, Printf, Dates, Random

function main()
    println("MaxCut p=14 D=3 -- manual charge adjoint (~5x fwd cost)")
    println("Started: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("RAM: ", Sys.total_memory() / (1024^3), " GB")
    println()

    # FD baseline timings (from killed run)
    println("FD baseline (from previous run):")
    println("  p=10: 772.0s  p=11: 4331.4s  (p=12+ was running)")
    println()
    flush(stdout)

    println("=== Warm-start p=1..13 with charge_adjoint ===")
    flush(stdout)

    warm = nothing
    for p in 1:13
        params = TreeParams(2, 3, p)
        guesses = isnothing(warm) ? QAOAAngles[] : [extend_angles(warm, p)]
        t = @elapsed r = optimize_angles(params;
            clause_sign=-1, restarts=(p <= 5 ? 2 : 0), maxiters=200,
            autodiff=:charge_adjoint,
            initial_guesses=guesses, initial_guess_kind=:warm,
            rng=MersenneTwister(42 + p))
        @printf("  p=%2d: ctilde=%.10f  %3d iters  %8.1fs\n", p, r.value, r.iterations, t)
        flush(stdout)
        if QaoaXorsat.is_valid_qaoa_value(r.value)
            warm = r.angles
        end
    end

    # p=14
    println()
    println("=== p=14 optimization (manual adjoint) ===")
    flush(stdout)

    params_14 = TreeParams(2, 3, 14)
    warm_14 = extend_angles(warm, 14)

    @printf("  Single charge eval at p=14... ")
    flush(stdout)
    t_eval = @elapsed charge_parity_expectation(params_14, warm_14)
    @printf("%.1fs\n", t_eval)

    @printf("  Single adjoint gradient at p=14... ")
    flush(stdout)
    t_grad = @elapsed charge_expectation_and_gradient(params_14, warm_14; clause_sign=-1)
    @printf("%.1fs (%.1fx fwd)\n", t_grad, t_grad / t_eval)
    @printf("  Memory: %.1f GB RSS\n", Sys.maxrss() / 1e9)
    flush(stdout)

    println("  Optimizing...")
    flush(stdout)

    t_p14 = @elapsed r14 = optimize_angles(params_14;
        clause_sign=-1, restarts=0, maxiters=100,
        autodiff=:charge_adjoint,
        initial_guesses=[warm_14], initial_guess_kind=:warm,
        rng=MersenneTwister(56),
        on_evaluation=(si, evals, elapsed, val, gnorm) -> begin
            @printf("    [%d evals, %.0fs] ctilde=%.10f gnorm=%.2e\n",
                    evals, elapsed, val, gnorm)
            flush(stdout)
        end)

    println()
    println("===================================================")
    @printf("  p=14 RESULT: ctilde = %.10f\n", r14.value)
    @printf("  Iterations:  %d\n", r14.iterations)
    @printf("  Wall time:   %.1fs (%.1f hr)\n", t_p14, t_p14/3600)
    @printf("  Converged:   %s\n", r14.converged)
    @printf("  Memory:      %.1f GB RSS\n", Sys.maxrss() / 1e9)
    println("===================================================")
    println("Finished: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    flush(stdout)
end

main()
