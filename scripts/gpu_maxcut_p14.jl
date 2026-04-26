#!/usr/bin/env julia
# GPU MaxCut sweep using the PRODUCTION optimizer with gpu_evaluator.
# Uses warm-start chain from p=11, minimal restarts (MaxCut is single-basin).

using QaoaXorsat, Metal, Printf, Random

include(joinpath(@__DIR__, "..", "src", "gpu_checkpointed.jl"))

gpu_array_fn(x::AbstractVector{<:Complex}) = MtlArray(ComplexF32.(x))
gpu_array_fn(x::AbstractVector{<:Real}) = MtlArray(ComplexF32.(complex.(x)))

# GPU evaluator closure for the production optimizer
function make_gpu_evaluator(gpu_fn; checkpoint_interval=0)
    function gpu_eval(params, angles; clause_sign)
        gpu_checkpointed_forward_backward(params, angles, gpu_fn;
            clause_sign, checkpoint_interval)
    end
    gpu_eval
end

gpu_eval = make_gpu_evaluator(gpu_array_fn)

# p=11 optimal angles from CPU sweep
γ11 = [0.25771657761408784, 0.5302956055768999, 3.7347249730268595,
       0.6378773756992225, 0.673454112578702, 0.6975828686883527,
       0.7345746342881904, 0.7767605300287548, 0.8864535229410492,
       1.049438627577906, 1.1159567942744113]
β11 = [0.6586876811062338, 0.5657110881830981, 2.624922911399236,
       2.6394034232110237, 2.66220812578448, 2.686488948535113,
       2.718374507146742, 2.766755120305004, 2.862090979321062,
       2.939569684149758, 3.033248523560461]

global warm_angles = QAOAAngles(γ11, β11)

for target_p in [12, 13, 14]
    println("=" ^ 60)
    println("=== MaxCut D=3 p=$target_p (production optimizer + GPU) ===")
    println("=" ^ 60)

    params = TreeParams(2, 3, target_p)
    warm = [extend_angles(warm_angles, target_p)]

    t0 = time()
    # Use the PRODUCTION optimizer with gpu_evaluator.
    # Warm-start + 2 random restarts (MaxCut is single-basin, no need for swarm).
    result = optimize_angles(params;
        clause_sign=-1,
        restarts=2,
        maxiters=200,
        initial_guesses=warm,
        rng=MersenneTwister(42 + target_p),
        g_abstol=1e-6,
        gpu_evaluator=gpu_eval,
    )
    dt = time() - t0

    @printf("\np=%d  c̃=%.12f  converged=%s  time=%.1fs  evals=%d\n",
            target_p, result.value, result.converged, dt, result.evaluations)
    @printf("γ = %s\n", join(string.(result.angles.γ), ";"))
    @printf("β = %s\n\n", join(string.(result.angles.β), ";"))
    flush(stdout)

    global warm_angles = result.angles
end

