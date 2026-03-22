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

function manual_root_local_factor(
    angles::QAOAAngles,
    configurations::AbstractVector{<:Integer};
    clause_sign::Int=1,
)
    p = depth(angles)
    hyperindices = map(configuration -> QaoaXorsat.basso_configuration_to_hyperindex(configuration, p), configurations)

    phase = prod(1:p) do physical_round
        slice = QaoaXorsat.slice_from_physical_round(physical_round, p)
        QaoaXorsat.problem_phase(hyperindices, angles.γ[physical_round], slice, p; clause_sign)
    end

    parity = prod(configuration -> QaoaXorsat.basso_root_parity(configuration, p), configurations)
    parity * phase
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

    @testset "smallest finite-D branch tensor closed form" begin
        params = TreeParams(3, 2, 1)
        γ = 0.31
        angles = QAOAAngles([γ], [0.17])
        tensor = QaoaXorsat.basso_branch_tensor(params, angles)

        @testset "configuration=$configuration" for configuration in 0:QaoaXorsat.basso_configuration_count(params.p)-1
            bits = QaoaXorsat.decode_bits(configuration, QaoaXorsat.basso_bit_count(params.p))
            expected = bits[1] == bits[3] ? 1.0 : cos(2γ)

            @test tensor[configuration+1] ≈ expected + 0.0im atol = 1e-12
        end
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

    @testset "branch-to-hyperindex mapping" begin
        @test QaoaXorsat.basso_configuration_to_hyperindex(0b000, 1) == 0b00
        @test QaoaXorsat.basso_configuration_to_hyperindex(0b101, 1) == 0b11
        @test QaoaXorsat.basso_configuration_to_hyperindex(0b00000, 2) == 0b0000
        @test QaoaXorsat.basso_configuration_to_hyperindex(0b10001, 2) == 0b1100
        @test QaoaXorsat.basso_configuration_to_hyperindex(0b01010, 2) == 0b0011
    end

    @testset "hyperindex-to-branch lift" begin
        @testset "leaf boundary reproduces f(a) at p=$p" for p in [1, 2]
            angles = p == 1 ? QAOAAngles([0.31], [0.17]) : QAOAAngles([0.31, 0.64], [0.17, 0.39])
            lifted = QaoaXorsat.lift_hyperindex_message_to_branch_basis(leaf_tensor(angles), angles)

            @test lifted ≈ ComplexF64[
                QaoaXorsat.f_function(angles, configuration)
                for configuration in 0:QaoaXorsat.basso_configuration_count(p)-1
            ] atol = 1e-12
        end
    end

    @testset "lift does not commute with k-body constraint contraction" begin
        params = TreeParams(3, 2, 1)
        angles = QAOAAngles([0.31], [0.17])
        leaf = ComplexF64.(leaf_tensor(angles))

        lifted_leaf = QaoaXorsat.lift_hyperindex_message_to_branch_basis(leaf, angles)
        branch_tensor = QaoaXorsat.basso_branch_tensor(params, angles)
        raw_parent = QaoaXorsat.contract_constraint_message([leaf, leaf], angles.γ[1], 1, 1)
        lifted_parent = QaoaXorsat.lift_hyperindex_message_to_branch_basis(raw_parent, angles)

        @test lifted_leaf ≈ ComplexF64[
            QaoaXorsat.f_function(angles, configuration)
            for configuration in 0:QaoaXorsat.basso_configuration_count(params.p)-1
        ] atol = 1e-12
        @test branch_tensor ≈ QaoaXorsat.basso_branch_tensor_step(
            params,
            angles,
            ones(ComplexF64, QaoaXorsat.basso_configuration_count(params.p)),
        ) atol = 1e-12
        @test !isapprox(lifted_parent, branch_tensor; atol=1e-12)
    end

    @testset "mixed raw-to-branch clause oracle" begin
        params = TreeParams(3, 2, 1)
        angles = QAOAAngles([0.31], [0.17])
        leaf = ComplexF64.(leaf_tensor(angles))

        mixed = QaoaXorsat.contract_hyperindex_messages_to_branch_basis(
            [leaf, leaf],
            angles,
            QaoaXorsat.basso_branching_degree(params),
        )
        direct = QaoaXorsat.basso_constraint_fold(
            QaoaXorsat.lift_hyperindex_message_to_branch_basis(leaf, angles),
            QaoaXorsat.basso_constraint_kernel(angles, QaoaXorsat.basso_branching_degree(params)),
            2,
        )
        branch_tensor = QaoaXorsat.basso_branch_tensor(params, angles)

        @test mixed ≈ direct atol = 1e-12
        @test mixed ≈ branch_tensor atol = 1e-12
    end

    @testset "root local factor matches raw tensor semantics" begin
        @testset "k=$k, p=$p, clause_sign=$clause_sign" for (k, p, clause_sign, tuples) in [
            (2, 1, -1, [(0b000, 0b000), (0b001, 0b011), (0b101, 0b010)]),
            (2, 2, -1, [(0b00000, 0b00000), (0b10001, 0b01010), (0b11100, 0b00111)]),
            (3, 1, 1, [(0b000, 0b000, 0b000), (0b001, 0b010, 0b011), (0b101, 0b110, 0b011)]),
        ]
            angles = p == 1 ? QAOAAngles([0.31], [0.17]) : QAOAAngles([0.31, 0.64], [0.17, 0.39])
            gamma_full = QaoaXorsat.build_gamma_full_vector(angles)
            bit_count = QaoaXorsat.basso_bit_count(p)

            for configs in tuples
                delta = foldl(xor, configs; init=0)
                raw = manual_root_local_factor(angles, collect(configs); clause_sign)
                unscaled = QaoaXorsat.basso_root_parity(delta, p) *
                           cis(-0.5 * clause_sign * sum(gamma_full .* QaoaXorsat.configuration_spins(delta, bit_count)))

                @test raw ≈ unscaled atol = 1e-12
                @test raw ≈ QaoaXorsat.basso_root_kernel(angles, 1, p; clause_sign)[delta+1] atol = 1e-12
            end
        end
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
