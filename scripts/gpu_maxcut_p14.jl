#!/usr/bin/env julia
# GPU MaxCut sweep from p=12 to p=14 using checkpointed backward pass.
# Warm-starts from p=11 CPU result.

using QaoaXorsat, Metal, Printf, Random
include(joinpath(@__DIR__, "..", "src", "gpu_optimizer.jl"))

gpu_array(x::AbstractVector{<:Complex}) = MtlArray(ComplexF32.(x))
gpu_array(x::AbstractVector{<:Real}) = MtlArray(ComplexF32.(complex.(x)))

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
    println("=== MaxCut D=3 p=$target_p (GPU, checkpointed) ===")
    println("=" ^ 60)

    params = TreeParams(2, 3, target_p)
    warm = [extend_angles(warm_angles, target_p)]

    t0 = time()
    best_angles, best_val = gpu_optimize_angles(params, gpu_array;
        clause_sign=-1, restarts=4, maxiters=200,
        initial_guesses=warm, rng=MersenneTwister(42),
        checkpoint_interval=0, verbose=true)
    dt = time() - t0

    @printf("\np=%d  c̃=%.12f  time=%.1fs\n\n", target_p, best_val, dt)
    flush(stdout)

    global warm_angles = best_angles
end
