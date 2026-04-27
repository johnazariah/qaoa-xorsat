#!/usr/bin/env julia
# GPU MaxCut sweep using SWARM optimizer with gpu_evaluator.
# Uses warm-start chain from p=11/p=12, memetic swarm for basin discovery.

using QaoaXorsat, Metal, Printf, Random

include(joinpath(@__DIR__, "..", "src", "gpu_checkpointed.jl"))

gpu_array_fn(x::AbstractVector{<:Complex}) = MtlArray(ComplexF32.(x))
gpu_array_fn(x::AbstractVector{<:Real}) = MtlArray(ComplexF32.(complex.(x)))

# GPU evaluator closure
function make_gpu_evaluator(gpu_fn; checkpoint_interval=0)
    function gpu_eval(params, angles; clause_sign)
        gpu_checkpointed_forward_backward(params, angles, gpu_fn;
            clause_sign, checkpoint_interval)
    end
    gpu_eval
end

gpu_eval = make_gpu_evaluator(gpu_array_fn)

# p=12 optimal angles from the GPU run we just did
γ12 = [0.25799199576076925, 0.5275737565219489, 3.731774233707554,
       0.6381214021980847, 0.669407834600715, 0.6826420778797152,
       0.7216418998660016, 0.7803873052252396, 0.8428447097055709,
       0.9702736660151621, 1.125802496527615, 1.106079670361682]
β12 = [0.6531372954232731, 0.5607554425084307, 2.628140767434414,
       2.6368194724189187, 2.65990296637837, 2.683205892553897,
       2.701771349062728, 2.744226280598693, 2.8073568134696325,
       2.920397692865255, 2.995945182305031, 3.062289983902837]

global warm_angles = QAOAAngles(γ12, β12)

for target_p in [13, 14]
    println("=" ^ 60)
    println("=== MaxCut D=3 p=$target_p (swarm + GPU) ===")
    println("=" ^ 60)
    flush(stdout)

    params = TreeParams(2, 3, target_p)
    warm = [extend_angles(warm_angles, target_p)]

    t0 = time()
    # SWARM optimizer: 100 candidates, short bursts, early exit.
    # For MaxCut single-basin landscape, the swarm exits after 1-2
    # generations (early exit on stagnation), then polishes the winner.
    # Reduced population for GPU (each eval is expensive at high p).
    result = swarm_optimize(params;
        clause_sign=-1,
        population=20,
        generations=5,
        burst_iters=20,
        warm_starts=warm,
        rng=MersenneTwister(42 + target_p),
        g_abstol=1e-6,
        gpu_evaluator=gpu_eval,
        on_generation=(gen, best, npop) -> begin
            elapsed = time() - t0
            @printf("  gen %2d: best=%.10f  pop=%d  elapsed=%.0fs\n",
                    gen, best, npop, elapsed)
            flush(stdout)
        end,
    )
    dt = time() - t0

    @printf("\np=%d  c̃=%.12f  converged=%s  time=%.1fs  evals=%d\n",
            target_p, result.value, result.converged, dt, result.evaluations)
    @printf("γ = %s\n", join(string.(result.angles.γ), ";"))
    @printf("β = %s\n\n", join(string.(result.angles.β), ";"))
    flush(stdout)

    global warm_angles = result.angles
end

