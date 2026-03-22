using LinearAlgebra
using QaoaXorsat
using Test

@testset "QAOAAngles" begin
    @test QAOAAngles([0.1], [0.2]) isa QAOAAngles
    @test depth(QAOAAngles([0.1, 0.2], [0.3, 0.4])) == 2
    @test QAOAAngles([1, 2], [3, 4]).γ == [1.0, 2.0]
    @test_throws ArgumentError QAOAAngles([0.1, 0.2], [0.3])
    @test_throws ArgumentError QAOAAngles(Float64[], Float64[])
end

@testset "hyperindex utilities" begin
    @test hyperindex_dimension(1) == 4
    @test hyperindex_dimension(3) == 64
    @test_throws ArgumentError hyperindex_dimension(0)
    @test_throws ArgumentError hyperindex_dimension(-1)
    @test slice_bit_positions(1, 3) == (1, 2)
    @test slice_bit_positions(3, 3) == (5, 6)
    @test round_bit_positions(1, 3) == (1, 2)
    @test round_bit_positions(3, 3) == (5, 6)
    @test slice_from_physical_round(1, 3) == 3
    @test slice_from_physical_round(3, 3) == 1
    @test physical_round_from_slice(1, 3) == 3
    @test physical_round_from_slice(3, 3) == 1
    @test_throws ArgumentError round_bit_positions(0, 3)
    @test_throws ArgumentError round_bit_positions(4, 3)
    @test_throws ArgumentError slice_from_physical_round(0, 3)
    @test_throws ArgumentError physical_round_from_slice(4, 3)

    @test hyperindex_bit(0b1010, 1) == 0
    @test hyperindex_bit(0b1010, 2) == 1
    @test hyperindex_bit(0b1010, 3) == 0
    @test hyperindex_bit(0b1010, 4) == 1
    @test hyperindex_parity(0b1010, [1, 2]) == 1
    @test hyperindex_parity(0b1010, [2, 4]) == 0
    @test hyperindex_parity(0b1111, [1, 2, 3]) == 1
    @test_throws ArgumentError hyperindex_bit(-1, 1)
    @test_throws ArgumentError hyperindex_bit(0, 0)
end

@testset "leaf tensor" begin
    @testset "dimensions" begin
        for p in 1:4
            angles = QAOAAngles(zeros(p), zeros(p))
            tensor = leaf_tensor(angles)
            @test length(tensor) == 4^p
        end
    end

    @testset "constant value" begin
        for p in 1:4
            angles = QAOAAngles(rand(p) .* π, rand(p) .* π)
            tensor = leaf_tensor(angles)
            @test eltype(tensor) <: Real
            @test all(isfinite, tensor)
            @test all(x -> isapprox(x, exp2(-p); atol = 1e-12), tensor)
        end
    end

    @testset "golden values p=1" begin
        tensor = leaf_tensor(QAOAAngles([0.37], [0.29]))
        @test tensor ≈ fill(0.5, 4) atol = 1e-12
    end

    @testset "periodicity" begin
        p = 2
        γ = [0.3, 0.7]
        β = [0.4, 0.8]
        @test leaf_tensor(QAOAAngles(γ, β)) ≈
            leaf_tensor(QAOAAngles(γ .+ 2π, β)) atol = 1e-12
        @test leaf_tensor(QAOAAngles(γ, β)) ≈
            leaf_tensor(QAOAAngles(γ, β .+ 2π)) atol = 1e-12
    end
end

@testset "mixer tensor" begin
    @testset "dimensions" begin
        for p in 1:4
            tensor = mixer_tensor(0.0, 1, p)
            @test size(tensor) == (4^p, 4^p)
            @test eltype(tensor) == ComplexF64
        end
    end

    @testset "identity at zero angle" begin
        for p in 1:4
            tensor = mixer_tensor(0.0, 1, p)
            @test tensor ≈ Matrix{ComplexF64}(I, 4^p, 4^p) atol = 1e-12
        end
    end

    @testset "local p=1 block" begin
        β = π / 6
        c = cos(β)
        s = sin(β)
        tensor = mixer_tensor(β, 1, 1)
        @test tensor[1, 1] ≈ c^2 atol = 1e-12
        @test tensor[2, 1] ≈ -im * c * s atol = 1e-12
        @test tensor[3, 1] ≈ im * c * s atol = 1e-12
        @test tensor[4, 1] ≈ s^2 atol = 1e-12
    end

    @testset "periodicity" begin
        @test mixer_tensor(0.4, 1, 2) ≈ mixer_tensor(0.4 + 2π, 1, 2) atol = 1e-12
    end

    @testset "multi-round p=2" begin
        β = π / 5
        c = cos(β)
        s = sin(β)
        M2 = mixer_tensor(β, 2, 2)
        # Round 2 acts on bits 3,4 only; bits 1,2 untouched
        # input σ=0 → output σ=4 (bit 3 flipped): -i*cos*sin
        @test M2[5, 1] ≈ -im * c * s atol = 1e-12
        # input σ=0 → output σ=2 (bit 2 flipped, wrong round): 0
        @test M2[3, 1] ≈ 0.0 + 0.0im atol = 1e-12
    end

    @testset "superoperator unitarity" begin
        for p in 1:3
            M = mixer_tensor(rand() * 2π, 1, p)
            @test M * M' ≈ I(4^p) atol = 1e-10
        end
    end
end

@testset "problem tensor" begin
    @testset "dimensions" begin
        for p in 1:2, k in (2, 3)
            tensor = problem_tensor(k, 0.3, 1, p)
            @test length(tensor) == (4^p)^k
            @test eltype(tensor) == ComplexF64
        end
    end

    @testset "zero angle" begin
        for p in 1:2, k in (2, 3)
            tensor = problem_tensor(k, 0.0, 1, p)
            @test all(x -> isapprox(x, 1.0 + 0.0im; atol = 1e-12), tensor)
        end
    end

    @testset "golden values maxcut p=1" begin
        γ = π / 3
        tensor = reshape(problem_tensor(2, γ, 1, 1), 4, 4)
        @test tensor[1, 1] ≈ 1.0 + 0.0im atol = 1e-12
        @test tensor[2, 1] ≈ cis(γ) atol = 1e-12
        @test tensor[3, 1] ≈ cis(-γ) atol = 1e-12
        @test tensor[4, 1] ≈ 1.0 + 0.0im atol = 1e-12
        @test tensor[2, 4] ≈ cis(-γ) atol = 1e-12
    end

    @testset "odd-clause sign handling" begin
        γ = π / 3
        tensor = reshape(problem_tensor(2, γ, 1, 1; clause_sign = -1), 4, 4)
        @test tensor[2, 1] ≈ cis(-γ) atol = 1e-12
        @test tensor[3, 1] ≈ cis(γ) atol = 1e-12
    end

    @testset "golden values xorsat k=3 p=1" begin
        γ = π / 3
        tensor = reshape(problem_tensor(3, γ, 1, 1), 4, 4, 4)
        @test tensor[1, 1, 1] ≈ 1.0 + 0.0im atol = 1e-12
        @test tensor[2, 1, 1] ≈ cis(γ) atol = 1e-12
        @test tensor[3, 1, 1] ≈ cis(-γ) atol = 1e-12
        @test tensor[4, 1, 1] ≈ 1.0 + 0.0im atol = 1e-12
        @test tensor[2, 4, 1] ≈ cis(-γ) atol = 1e-12
        @test tensor[2, 4, 4] ≈ cis(γ) atol = 1e-12
    end

    @testset "periodicity" begin
        @test problem_tensor(3, 0.4, 1, 2) ≈ problem_tensor(3, 0.4 + 2π, 1, 2) atol = 1e-12
    end
end

@testset "parity observable tensor" begin
    @testset "dimensions" begin
        for p in 1:2, k in (2, 3)
            tensor = parity_observable_tensor(k, p)
            @test length(tensor) == (4^p)^k
            @test eltype(tensor) <: Real
        end
    end

    @testset "p=1 values" begin
        maxcut = reshape(parity_observable_tensor(2, 1), 4, 4)
        @test maxcut[1, 1] == 1.0
        @test maxcut[4, 4] == 1.0
        @test maxcut[1, 4] == -1.0
        @test maxcut[4, 1] == -1.0
        @test maxcut[2, 1] == 0.0
        @test maxcut[1, 2] == 0.0
        @test maxcut[3, 3] == 0.0

        xorsat = reshape(parity_observable_tensor(3, 1), 4, 4, 4)
        @test xorsat[1, 1, 1] == 1.0
        @test xorsat[4, 4, 4] == -1.0
        @test xorsat[4, 4, 1] == 1.0
        @test xorsat[4, 1, 1] == -1.0
        @test xorsat[2, 1, 1] == 0.0
    end

    @testset "completeness" begin
        O = parity_observable_tensor(2, 1)
        @test sum(O) == 0.0
        @test count(!iszero, O) == 4
    end
end

@testset "observable tensor" begin
    @testset "dimensions" begin
        for p in 1:2, k in (2, 3)
            tensor = observable_tensor(k, p)
            @test length(tensor) == (4^p)^k
            @test eltype(tensor) <: Real
        end
    end

    @testset "p=1 even-clause values" begin
        maxcut = reshape(observable_tensor(2, 1), 4, 4)
        @test maxcut[1, 1] == 1.0
        @test maxcut[4, 4] == 1.0
        @test maxcut[1, 4] == 0.0
        @test maxcut[4, 1] == 0.0
        @test maxcut[2, 1] == 0.0
        @test maxcut[1, 2] == 0.0
        @test maxcut[3, 3] == 0.0
    end

    @testset "p=1 odd-clause values" begin
        maxcut = reshape(observable_tensor(2, 1; clause_sign = -1), 4, 4)
        @test maxcut[1, 1] == 0.0
        @test maxcut[4, 4] == 0.0
        @test maxcut[1, 4] == 1.0
        @test maxcut[4, 1] == 1.0
        @test maxcut[2, 1] == 0.0
    end
end
