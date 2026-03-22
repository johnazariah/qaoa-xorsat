using QaoaXorsat
using Test

z_sign(bit::Int) = bit == 0 ? 1 : -1

function build_maxcut_light_cone_edges(D::Int, p::Int)
    next_qubit = Ref(2)
    edges = Tuple{Int,Int}[(1, 2)]

    function expand_variable(parent::Int, depth::Int)
        depth == p && return

        for _ in 1:(D-1)
            next_qubit[] += 1
            child = next_qubit[]
            push!(edges, (parent, child))
            expand_variable(child, depth + 1)
        end
    end

    expand_variable(1, 0)
    expand_variable(2, 0)

    edges, next_qubit[]
end

function apply_maxcut_problem_layer!(state::Vector{ComplexF64}, edges, γ::Real)
    γf = Float64(γ)
    iszero(γf) && return state

    for basis_state in 0:length(state)-1
        parity_sum = sum(
            z_sign((basis_state >> (u - 1)) & 1) * z_sign((basis_state >> (v - 1)) & 1)
            for (u, v) in edges
        )
        state[basis_state+1] *= cis(0.5 * γf * parity_sum)
    end

    state
end

function apply_reference_mixer_layer!(state::Vector{ComplexF64}, β::Real, qubit_count::Int)
    βf = Float64(β)
    iszero(βf) && return state

    c = ComplexF64(cos(βf))
    s = ComplexF64(0.0, -sin(βf))

    for qubit in 1:qubit_count
        mask = one(Int) << (qubit - 1)
        for basis_state in 0:length(state)-1
            iszero(basis_state & mask) || continue

            zero_index = basis_state + 1
            one_index = (basis_state | mask) + 1
            amplitude_zero = state[zero_index]
            amplitude_one = state[one_index]

            state[zero_index] = c * amplitude_zero + s * amplitude_one
            state[one_index] = s * amplitude_zero + c * amplitude_one
        end
    end

    state
end

function exact_maxcut_reference(γs, βs; D::Int=3)
    p = length(γs)
    edges, qubit_count = build_maxcut_light_cone_edges(D, p)
    state = fill(ComplexF64(inv(sqrt(float(one(Int) << qubit_count)))), one(Int) << qubit_count)

    for physical_round in 1:p
        apply_maxcut_problem_layer!(state, edges, γs[physical_round])
        apply_reference_mixer_layer!(state, βs[physical_round], qubit_count)
    end

    parity = 0.0
    for basis_state in 0:length(state)-1
        parity += abs2(state[basis_state+1]) *
                  z_sign((basis_state >> 0) & 1) *
                  z_sign((basis_state >> 1) & 1)
    end

    parity, 0.5 * (1 - parity)
end

function build_k3_d2_p1_clauses()
    [
        [1, 2, 3],
        [1, 4, 5],
        [2, 6, 7],
        [3, 8, 9],
    ]
end

function apply_reference_xorsat_problem_layer!(
    state::Vector{ComplexF64},
    clauses,
    γ::Real;
    clause_sign::Int=1,
)
    γf = Float64(γ)
    iszero(γf) && return state

    for basis_state in 0:length(state)-1
        parity_sum = sum(
            clause_sign * prod(z_sign((basis_state >> (qubit - 1)) & 1) for qubit in clause)
            for clause in clauses
        )
        state[basis_state+1] *= cis(-0.5 * γf * parity_sum)
    end

    state
end

function exact_k3_d2_p1_reference(γ, β; clause_sign::Int=1)
    clauses = build_k3_d2_p1_clauses()
    qubit_count = 9
    state = fill(ComplexF64(inv(sqrt(float(one(Int) << qubit_count)))), one(Int) << qubit_count)

    apply_reference_xorsat_problem_layer!(state, clauses, γ; clause_sign)
    apply_reference_mixer_layer!(state, β, qubit_count)

    parity = 0.0
    for basis_state in 0:length(state)-1
        parity += abs2(state[basis_state+1]) *
                  prod(z_sign((basis_state >> (qubit - 1)) & 1) for qubit in clauses[1])
    end

    parity, 0.5 * (1 + clause_sign * parity)
end

@testset "QAOA evaluation" begin
    @testset "zero-angle baseline" begin
        params = TreeParams(3, 4, 1)
        angles = QAOAAngles([0.0], [0.0])
        @test parity_expectation(params, angles) ≈ 0.0 atol = 1e-12
        @test qaoa_expectation(params, angles) ≈ 0.5 atol = 1e-12
    end

    @testset "MaxCut p=1 exact formula" begin
        params = TreeParams(2, 3, 1)

        @testset "γ=$γ, β=$β" for (γ, β) in [
            (0.31, 0.17),
            (0.73, 0.29),
        ]
            angles = QAOAAngles([γ], [β])
            parity = parity_expectation(params, angles; clause_sign=-1)
            formula = -sin(4β) * cos(γ)^2 * sin(γ)
            @test parity ≈ formula atol = 1e-10
            @test qaoa_expectation(params, angles; clause_sign=-1) ≈
                  (1 - formula) / 2 atol = 1e-10
        end
    end

    @testset "MaxCut p=1 optimum" begin
        params = TreeParams(2, 3, 1)
        γopt = atan(1 / sqrt(2))
        βopt = π / 8
        optimum = 0.5 + sqrt(3) / 9
        angles = QAOAAngles([γopt], [βopt])
        @test qaoa_expectation(params, angles; clause_sign=-1) ≈ optimum atol = 1e-10
    end

    @testset "MaxCut p=2 exact-statevector comparison" begin
        params = TreeParams(2, 3, 2)
        angles = QAOAAngles([0.21, 0.64], [0.17, 0.39])
        reference_parity, reference_cost = exact_maxcut_reference(angles.γ, angles.β)

        @test parity_expectation(params, angles; clause_sign=-1) ≈
              reference_parity atol = 1e-10
        @test qaoa_expectation(params, angles; clause_sign=-1) ≈
              reference_cost atol = 1e-10
    end

    @testset "k=3, D=2, p=1 exact-statevector comparison" begin
        params = TreeParams(3, 2, 1)

        @testset "γ=$γ, β=$β, clause_sign=$clause_sign" for (γ, β, clause_sign) in [
            (0.31, 0.17, 1),
            (0.73, 0.29, 1),
            (0.31, 0.17, -1),
        ]
            angles = QAOAAngles([γ], [β])
            reference_parity, reference_cost = exact_k3_d2_p1_reference(γ, β; clause_sign)

            @test parity_expectation(params, angles; clause_sign) ≈
                  reference_parity atol = 1e-10
            @test qaoa_expectation(params, angles; clause_sign) ≈
                  reference_cost atol = 1e-10
        end
    end

    @testset "Tier 2 finite-D overlap comparison" begin
        @testset "k=$k, D=$D, p=$p, clause_sign=$clause_sign" for (k, D, p, clause_sign, angles) in [
            (3, 2, 1, 1, QAOAAngles([0.31], [0.17])),
            (3, 2, 1, 1, QAOAAngles([0.73], [0.29])),
            (3, 2, 1, -1, QAOAAngles([0.31], [0.17])),
            (2, 3, 1, -1, QAOAAngles([0.31], [0.17])),
            (2, 3, 2, -1, QAOAAngles([0.21, 0.64], [0.17, 0.39])),
        ]
            params = TreeParams(k, D, p)

            @test parity_expectation(params, angles; clause_sign) ≈
                  basso_parity_expectation(params, angles; clause_sign) atol = 1e-10
            @test qaoa_expectation(params, angles; clause_sign) ≈
                  basso_expectation(params, angles; clause_sign) atol = 1e-10
            @test basso_parity_expectation(params, angles; clause_sign) ≈
                  parity_expectation(params, angles; clause_sign) atol = 1e-10
            @test basso_expectation(params, angles; clause_sign) ≈
                  qaoa_expectation(params, angles; clause_sign) atol = 1e-10
        end
    end

    @testset "Tier 2 removes public light-cone guard" begin
        params = TreeParams(3, 4, 2)
        angles = QAOAAngles([0.1, 0.2], [0.3, 0.4])

        @test isfinite(parity_expectation(params, angles))
        @test 0.0 ≤ qaoa_expectation(params, angles) ≤ 1.0
    end

    @testset "reference light-cone guard" begin
        params = TreeParams(3, 4, 2)
        angles = QAOAAngles([0.1, 0.2], [0.3, 0.4])
        @test_throws ArgumentError QaoaXorsat.reference_parity_expectation(params, angles)
    end
end
