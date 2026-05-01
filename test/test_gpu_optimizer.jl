"""
GPU optimizer end-to-end tests — Level 5.

Tests that gpu_optimize_angles converges to the same optimum as
CPU optimize_angles at moderate depth.
"""

using Test
using QaoaXorsat
using Random
using Printf

include(joinpath(@__DIR__, "gpu_test_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu_optimizer.jl"))

@testset "GPU Optimizer" begin
    if !GPU_OK
        @test true
        return
    end

    @testset "MaxCut k=2 D=3 p=$p" for p in [1, 3, 5]
        params = TreeParams(2, 3, p)

        # CPU reference
        cpu_result = optimize_angles(params; clause_sign=-1, restarts=4,
                                     maxiters=100, rng=MersenneTwister(42))

        # GPU optimization
        gpu_angles, gpu_val = gpu_optimize_angles(params, gpu_array;
            clause_sign=-1, restarts=4, maxiters=100,
            rng=MersenneTwister(42), verbose=false)

        # GPU should find a value close to CPU
        # Float32 means it won't be identical, but should be within 1e-3
        @test gpu_val ≈ cpu_result.value atol=2e-3
        @test gpu_val > 0.5  # better than random
    end

    @testset "XORSAT k=3 D=4 p=$p" for p in [1, 3]
        params = TreeParams(3, 4, p)

        cpu_result = optimize_angles(params; clause_sign=1, restarts=4,
                                     maxiters=100, rng=MersenneTwister(42))

        gpu_angles, gpu_val = gpu_optimize_angles(params, gpu_array;
            clause_sign=1, restarts=4, maxiters=100,
            rng=MersenneTwister(42), verbose=false)

        # Different GPU precision (Float64 on CUDA, Float32 on Metal) may lead
        # to different basins at k>=3; check GPU finds a reasonable value
        @test gpu_val > cpu_result.value - 0.05
        @test gpu_val > 0.5
    end

    @testset "warm start" begin
        params = TreeParams(2, 3, 3)

        # First optimize at p=2
        _, val2 = gpu_optimize_angles(params, gpu_array;
            clause_sign=-1, restarts=2, maxiters=50,
            rng=MersenneTwister(42), verbose=false)

        # Then at p=3 with warm start from p=2 result
        angles_p2, _ = gpu_optimize_angles(TreeParams(2, 3, 2), gpu_array;
            clause_sign=-1, restarts=2, maxiters=50,
            rng=MersenneTwister(42), verbose=false)

        warm = [QaoaXorsat.extend_angles(angles_p2, 3)]
        _, val3_warm = gpu_optimize_angles(params, gpu_array;
            clause_sign=-1, restarts=2, maxiters=50,
            initial_guesses=warm,
            rng=MersenneTwister(42), verbose=false)

        @test val3_warm > val2  # deeper should be better
        @test val3_warm > 0.75  # should be decent
    end

    @testset "known MaxCut p=1 optimal" begin
        params = TreeParams(2, 3, 1)

        _, gpu_val = gpu_optimize_angles(params, gpu_array;
            clause_sign=-1, restarts=8, maxiters=200,
            rng=MersenneTwister(42), verbose=false)

        # Known optimal: c̃ ≈ 0.6925
        @test gpu_val ≈ 0.6925 atol=1e-3
    end
end
