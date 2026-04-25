"""
GPU backward pass (gradient) integration tests — Level 4.

Compares gpu_forward_backward against CPU basso_expectation_and_gradient
and finite-difference gradients.
"""

using Test
using Metal
using QaoaXorsat

include(joinpath(@__DIR__, "..", "src", "gpu_backward.jl"))

const GPU_OK = Metal.functional()
GPU_OK || @warn "Metal not functional — skipping GPU gradient tests"

gpu_array(x::AbstractVector{<:Complex}) = MtlArray(ComplexF32.(x))
gpu_array(x::AbstractVector{<:Real}) = MtlArray(ComplexF32.(complex.(x)))

@testset "GPU Backward Pass" begin
    if !GPU_OK
        @test true
        return
    end

    @testset "gradient matches CPU k=$k D=$D p=$p" for (k, D, cs) in
            [(2, 3, -1), (3, 4, 1)],
            p in [1, 2, 3]

        params = TreeParams(k, D, p)
        angles = QAOAAngles(randn(p) .* 0.5, randn(p) .* 0.3 .+ 0.5)

        cpu_val, cpu_γg, cpu_βg = basso_expectation_and_gradient(params, angles; clause_sign=cs)
        gpu_val, gpu_γg, gpu_βg = gpu_forward_backward(params, angles, gpu_array; clause_sign=cs)

        @test gpu_val ≈ cpu_val atol=1e-3
        @test gpu_γg ≈ cpu_γg atol=1e-2
        @test gpu_βg ≈ cpu_βg atol=1e-2
    end

    @testset "gradient vs finite difference p=$p" for p in [1, 2, 3]
        params = TreeParams(2, 3, p)
        γ = randn(p) .* 0.5
        β = randn(p) .* 0.3 .+ 0.5
        angles = QAOAAngles(γ, β)

        _, gpu_γg, gpu_βg = gpu_forward_backward(params, angles, gpu_array; clause_sign=-1)

        # Finite difference
        ε = 1e-5
        fd_γg = zeros(p)
        fd_βg = zeros(p)

        for i in 1:p
            γp = copy(γ); γp[i] += ε
            γm = copy(γ); γm[i] -= ε
            vp = basso_expectation_normalized(params, QAOAAngles(γp, β); clause_sign=-1)
            vm = basso_expectation_normalized(params, QAOAAngles(γm, β); clause_sign=-1)
            fd_γg[i] = (vp - vm) / (2ε)
        end

        for i in 1:p
            βp = copy(β); βp[i] += ε
            βm = copy(β); βm[i] -= ε
            vp = basso_expectation_normalized(params, QAOAAngles(γ, βp); clause_sign=-1)
            vm = basso_expectation_normalized(params, QAOAAngles(γ, βm); clause_sign=-1)
            fd_βg[i] = (vp - vm) / (2ε)
        end

        @test gpu_γg ≈ fd_γg atol=1e-2
        @test gpu_βg ≈ fd_βg atol=1e-2
    end

    @testset "gradient at higher depth p=5" begin
        params = TreeParams(2, 3, 5)
        angles = QAOAAngles(randn(5) .* 0.3, randn(5) .* 0.2 .+ 0.4)

        cpu_val, cpu_γg, cpu_βg = basso_expectation_and_gradient(params, angles; clause_sign=-1)
        gpu_val, gpu_γg, gpu_βg = gpu_forward_backward(params, angles, gpu_array; clause_sign=-1)

        @test gpu_val ≈ cpu_val atol=1e-3
        @test gpu_γg ≈ cpu_γg atol=5e-2  # Float32 accumulation
        @test gpu_βg ≈ cpu_βg atol=5e-2
    end
end
