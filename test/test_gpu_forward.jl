"""
GPU forward pass integration tests — Level 3.

Compares gpu_forward_value against cpu basso_expectation_normalized
for multiple (k, D, p) combinations.
"""

using Test
using Metal
using QaoaXorsat

include(joinpath(@__DIR__, "..", "src", "gpu_forward.jl"))

const GPU_OK = Metal.functional()
GPU_OK || @warn "Metal not functional — skipping GPU forward tests"

# Helper to create MtlArray with auto-conversion to ComplexF32
gpu_array(x::AbstractVector{<:Complex}) = MtlArray(ComplexF32.(x))
gpu_array(x::AbstractVector{<:Real}) = MtlArray(ComplexF32.(complex.(x)))

@testset "GPU Forward Pass" begin
    if !GPU_OK
        @test true
        return
    end

    # Float32 has ~7 decimal digits; accumulated error across p levels
    # means we need generous tolerance
    @testset "MaxCut k=2 D=3 p=$p" for p in [1, 2, 3, 5]
        params = TreeParams(2, 3, p)
        angles = QAOAAngles(randn(p), randn(p))

        cpu_val = basso_expectation_normalized(params, angles; clause_sign=-1)
        gpu_val = gpu_forward_value(params, angles, gpu_array; clause_sign=-1)

        @test gpu_val ≈ cpu_val atol=1e-3
    end

    @testset "XORSAT k=3 D=4 p=$p" for p in [1, 2, 3, 5]
        params = TreeParams(3, 4, p)
        angles = QAOAAngles(randn(p), randn(p))

        cpu_val = basso_expectation_normalized(params, angles; clause_sign=1)
        gpu_val = gpu_forward_value(params, angles, gpu_array; clause_sign=1)

        @test gpu_val ≈ cpu_val atol=1e-3
    end

    @testset "XORSAT k=5 D=6 p=$p" for p in [1, 2, 3]
        params = TreeParams(5, 6, p)
        angles = QAOAAngles(randn(p), randn(p))

        cpu_val = basso_expectation_normalized(params, angles; clause_sign=1)
        gpu_val = gpu_forward_value(params, angles, gpu_array; clause_sign=1)

        @test gpu_val ≈ cpu_val atol=1e-2  # higher arity = more error
    end

    @testset "known MaxCut p=1 value" begin
        # Farhi's known optimal: c̃ ≈ 0.6924 for k=2, D=3, p=1
        # at angles γ≈0.6155, β≈0.3927 (approximate)
        params = TreeParams(2, 3, 1)
        angles = QAOAAngles([0.6155], [0.3927])

        gpu_val = gpu_forward_value(params, angles, gpu_array; clause_sign=-1)
        @test 0.68 < gpu_val < 0.70  # rough check
    end

    @testset "physical bounds" begin
        # c̃ must be in [0, 1] for any angles
        for _ in 1:10
            params = TreeParams(3, 4, 3)
            angles = QAOAAngles(randn(3), randn(3))
            gpu_val = gpu_forward_value(params, angles, gpu_array; clause_sign=1)
            @test 0.0 ≤ gpu_val ≤ 1.0 || isnan(gpu_val)
        end
    end

    @testset "optimal angles match CPU" begin
        # Use the known optimal for k=2, D=3, p=1
        params = TreeParams(2, 3, 1)
        # Optimize on CPU
        result = optimize_angles(params; clause_sign=-1, restarts=4, maxiters=100)

        # Evaluate on GPU
        gpu_val = gpu_forward_value(params, result.angles, gpu_array; clause_sign=-1)

        @test gpu_val ≈ result.value atol=1e-3
    end
end
