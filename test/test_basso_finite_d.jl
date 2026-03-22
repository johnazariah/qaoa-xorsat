using QaoaXorsat
using Test

function manual_basso_phase_argument(angles::QAOAAngles, configurations::Vararg{T}) where {T<:Integer}
    gamma_vector = QaoaXorsat.build_gamma_vector(angles)
    p = depth(angles)
    phase_positions = [collect(1:p); collect((p+2):(2p+1))]
    bit_vectors = [QaoaXorsat.decode_bits(configuration, QaoaXorsat.basso_bit_count(p))
                   for configuration in configurations]

    total = 0.0
    for (gamma_index, bit_index) in pairs(phase_positions)
        parity = prod(bit_vectors) do bits
            bit = bits[bit_index]
            bit == 0 ? 1 : -1
        end
        total += gamma_vector[gamma_index] * parity
    end

    total
end

function manual_basso_branch_tensor_step(params::TreeParams, angles::QAOAAngles, previous)
    configuration_count = QaoaXorsat.basso_configuration_count(params.p)
    branch_degree = QaoaXorsat.basso_branching_degree(params)
    phase_scale = inv(sqrt(float(branch_degree)))
    child_arity = params.k - 1
    all_configurations = collect(0:configuration_count-1)

    function child_sum(parent_configuration, selected, remaining, weight_product)
        if iszero(remaining)
            θ = phase_scale * manual_basso_phase_argument(angles, parent_configuration, selected...)
            return cos(θ) * weight_product
        end

        total = 0.0 + 0.0im
        for child_configuration in all_configurations
            child_weight = QaoaXorsat.f_function(angles, child_configuration) * previous[child_configuration+1]
            total += child_sum(
                parent_configuration,
                (selected..., child_configuration),
                remaining - 1,
                weight_product * child_weight,
            )
        end

        total
    end

    [child_sum(parent_configuration, (), child_arity, 1.0 + 0.0im)^branch_degree
     for parent_configuration in all_configurations]
end

@testset "Basso finite-D helpers" begin
    @testset "gamma vector" begin
        angles = QAOAAngles([0.21, 0.64], [0.17, 0.39])
        @test QaoaXorsat.build_gamma_vector(angles) == [0.21, 0.64, -0.64, -0.21]
    end

    @testset "bit counts" begin
        @test QaoaXorsat.basso_bit_count(1) == 3
        @test QaoaXorsat.basso_bit_count(3) == 7
        @test QaoaXorsat.basso_root_bit_index(1) == 2
        @test QaoaXorsat.basso_root_bit_index(3) == 4
        @test QaoaXorsat.basso_configuration_count(1) == 8
        @test QaoaXorsat.basso_configuration_count(2) == 32
    end

    @testset "decode bits" begin
        @test QaoaXorsat.decode_bits(0b10110, 5) == [0, 1, 1, 0, 1]
        @test QaoaXorsat.decode_bits(0, 3) == [0, 0, 0]
        @test_throws ArgumentError QaoaXorsat.decode_bits(-1, 3)
        @test_throws ArgumentError QaoaXorsat.decode_bits(8, 3)
    end

    @testset "f(a) zero beta support" begin
        angles = QAOAAngles([0.31], [0.0])

        @test QaoaXorsat.f_function(angles, 0b000) ≈ 0.5 + 0.0im atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b111) ≈ 0.5 + 0.0im atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b001) ≈ 0.0 + 0.0im atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b010) ≈ 0.0 + 0.0im atol = 1e-12
    end

    @testset "f(a) p=1 values" begin
        β = 0.37
        angles = QAOAAngles([0.21], [β])
        c = cos(β)
        s = sin(β)

        @test QaoaXorsat.f_function(angles, 0b000) ≈ 0.5 * c^2 atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b001) ≈ 0.5im * c * s atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b011) ≈ -0.5im * c * s atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b010) ≈ 0.5 * s^2 atol = 1e-12
    end

    @testset "phase argument" begin
        angles = QAOAAngles([0.31], [0.17])
        gamma_vector = QaoaXorsat.build_gamma_vector(angles)
        parent_bits = QaoaXorsat.decode_bits(0b000, 3)
        child_bits = QaoaXorsat.decode_bits(0b001, 3)

        @test QaoaXorsat.basso_phase_argument(gamma_vector, parent_bits, child_bits) ≈ -0.62 atol = 1e-12
        @test QaoaXorsat.basso_phase_argument(gamma_vector, parent_bits, child_bits) ≈
              manual_basso_phase_argument(angles, 0b000, 0b001) atol = 1e-12

        parent_flipped_center = QaoaXorsat.decode_bits(0b010, 3)
        @test QaoaXorsat.basso_phase_argument(gamma_vector, parent_flipped_center, child_bits) ≈
              QaoaXorsat.basso_phase_argument(gamma_vector, parent_bits, child_bits) atol = 1e-12
    end

    @testset "branch tensor initial state" begin
        params = TreeParams(3, 4, 1)
        angles = QAOAAngles([0.23], [0.41])
        tensor = QaoaXorsat.basso_branch_tensor(params, angles; steps=0)

        @test length(tensor) == QaoaXorsat.basso_configuration_count(1)
        @test all(value -> isapprox(value, 1.0 + 0.0im; atol=1e-12), tensor)
    end

    @testset "branch tensor zero angles stays uniform" begin
        for params in (TreeParams(2, 3, 1), TreeParams(3, 4, 1))
            angles = QAOAAngles(zeros(params.p), zeros(params.p))
            tensor = QaoaXorsat.basso_branch_tensor(params, angles)
            @test all(value -> isapprox(value, 1.0 + 0.0im; atol=1e-12), tensor)
        end
    end

    @testset "k=2 exact branch step matches manual sum" begin
        params = TreeParams(2, 3, 1)
        angles = QAOAAngles([0.31], [0.17])
        previous = ones(ComplexF64, QaoaXorsat.basso_configuration_count(params.p))

        @test QaoaXorsat.basso_branch_tensor_step(params, angles, previous) ≈
              manual_basso_branch_tensor_step(params, angles, previous) atol = 1e-12
    end

    @testset "k=3 exact branch step matches manual sum" begin
        params = TreeParams(3, 4, 1)
        angles = QAOAAngles([0.23], [0.19])
        previous = ones(ComplexF64, QaoaXorsat.basso_configuration_count(params.p))

        @test QaoaXorsat.basso_branch_tensor_step(params, angles, previous) ≈
              manual_basso_branch_tensor_step(params, angles, previous) atol = 1e-12
    end

    @testset "root kernel decomposition" begin
        angles = QAOAAngles([0.31, 0.64], [0.17, 0.39])
        branch_degree = 2

        parity_kernel = QaoaXorsat.basso_root_parity_kernel(depth(angles))
        positive_phase = QaoaXorsat.basso_root_problem_kernel(angles, branch_degree; clause_sign=1)
        negative_phase = QaoaXorsat.basso_root_problem_kernel(angles, branch_degree; clause_sign=-1)

        @test negative_phase ≈ conj.(positive_phase) atol = 1e-12
        @test QaoaXorsat.basso_root_kernel(angles, branch_degree) ≈
              parity_kernel .* positive_phase atol = 1e-12
    end

    @testset "zero-angle root parity sum vanishes" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1),
            (2, 3, 2),
            (3, 2, 1),
            (3, 4, 1),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(zeros(p), zeros(p))
            branch_tensor = QaoaXorsat.basso_branch_tensor(params, angles)

            @test QaoaXorsat.basso_root_parity_sum(params, angles, branch_tensor) ≈
                  0.0 + 0.0im atol = 1e-12
        end
    end
end
