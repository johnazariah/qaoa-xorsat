const MAX_EXACT_LIGHTCONE_QUBITS = 22

struct LightConeTree
    root_clause::Vector{Int}
    clauses::Vector{Vector{Int}}
    qubit_count::Int
end

function build_light_cone_tree(params::TreeParams)
    next_qubit = Ref(0)
    fresh_qubit!() = (next_qubit[] += 1)

    root_clause = [fresh_qubit!() for _ in 1:params.k]
    clauses = Vector{Vector{Int}}()
    push!(clauses, copy(root_clause))

    function expand_variable(parent::Int, depth::Int)
        depth == params.p && return

        for _ in 1:(params.D-1)
            clause = Int[parent]
            for _ in 1:(params.k-1)
                push!(clause, fresh_qubit!())
            end
            push!(clauses, clause)

            for child in clause[2:end]
                expand_variable(child, depth + 1)
            end
        end
    end

    foreach(root_variable -> expand_variable(root_variable, 0), root_clause)

    tree = LightConeTree(root_clause, clauses, next_qubit[])
    tree.qubit_count == total_variables(params) ||
        throw(AssertionError("light-cone qubit count mismatch"))
    length(tree.clauses) == total_constraints(params) ||
        throw(AssertionError("light-cone constraint count mismatch"))

    tree
end

qubit_state_sign(state::Int, qubit::Int) = z_eigenvalue((state >> (qubit - 1)) & 1)

clause_parity_sign(state::Int, clause) =
    foldl(*, (qubit_state_sign(state, qubit) for qubit in clause); init=1)

function validate_exact_light_cone(tree::LightConeTree)
    tree.qubit_count ≤ MAX_EXACT_LIGHTCONE_QUBITS || throw(ArgumentError(
        "exact light-cone reference evaluation is limited to " *
        "$(MAX_EXACT_LIGHTCONE_QUBITS) qubits; got $(tree.qubit_count). " *
        "The O(4^p) transfer contraction is not yet derived from the current raw tensors.",
    ))
    tree
end

function plus_state(qubit_count::Int)
    amplitude = inv(sqrt(float(one(Int) << qubit_count)))
    fill(ComplexF64(amplitude), one(Int) << qubit_count)
end

function apply_problem_layer!(
    state::Vector{ComplexF64},
    clauses,
    γ::Real;
    clause_sign::Int=1,
)
    γf = Float64(γ)
    iszero(γf) && return state

    validate_clause_sign(clause_sign)
    phase_scale = -0.5 * clause_sign * γf

    for basis_state in 0:length(state)-1
        parity_sum = sum(clause_parity_sign(basis_state, clause) for clause in clauses)
        state[basis_state+1] *= cis(phase_scale * parity_sum)
    end

    state
end

function apply_single_qubit_mixer!(
    state::Vector{ComplexF64},
    β::Real,
    qubit::Int,
)
    βf = Float64(β)
    iszero(βf) && return state

    c = ComplexF64(cos(βf))
    s = ComplexF64(0.0, -sin(βf))
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

    state
end

function apply_mixer_layer!(
    state::Vector{ComplexF64},
    β::Real,
    qubit_count::Int,
)
    βf = Float64(β)
    iszero(βf) && return state

    for qubit in 1:qubit_count
        apply_single_qubit_mixer!(state, βf, qubit)
    end

    state
end

function simulate_light_cone_state(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
)
    depth(angles) == params.p ||
        throw(ArgumentError("angle depth must match tree depth"))
    validate_clause_sign(clause_sign)

    tree = build_light_cone_tree(params) |> validate_exact_light_cone
    state = plus_state(tree.qubit_count)

    for physical_round in 1:params.p
        apply_problem_layer!(state, tree.clauses, angles.γ[physical_round]; clause_sign)
        apply_mixer_layer!(state, angles.β[physical_round], tree.qubit_count)
    end

    tree, state
end

function root_parity_expectation(state::Vector{ComplexF64}, root_clause)
    expectation = 0.0

    for basis_state in 0:length(state)-1
        expectation += abs2(state[basis_state+1]) * clause_parity_sign(basis_state, root_clause)
    end

    expectation
end

"""
    reference_parity_expectation(params, angles; clause_sign=1) -> Float64

Evaluate the exact root-clause parity correlator `⟨Z₁⋯Z_k⟩` on the physical
light-cone tree defined by `params` by explicit statevector simulation.

This path is retained as a correctness reference and is intentionally guarded to
small trees.
"""
function reference_parity_expectation(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
)
    tree, state = simulate_light_cone_state(params, angles; clause_sign)
    root_parity_expectation(state, tree.root_clause)
end

"""
    parity_expectation(params, angles; clause_sign=1) -> Float64

Evaluate the exact finite-D root-clause parity correlator `⟨Z₁⋯Z_k⟩` using the
Tier 2 branch-transfer contraction in the physical `γ/2` convention.
"""
function parity_expectation(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
)
    basso_parity_expectation(params, angles; clause_sign)
end

"""
    qaoa_expectation(algebra, params, angles) -> Float64

Evaluate the exact expected satisfaction of the root clause using the fold engine
parametrised by the given `CostAlgebra`.

This is the primary algebra-aware entry point.
"""
function qaoa_expectation(
    algebra::XORSATAlgebra,
    params::TreeParams,
    angles::QAOAAngles,
)
    arity(algebra) == params.k || throw(ArgumentError(
        "algebra arity $(arity(algebra)) does not match tree arity $(params.k)"
    ))
    basso_expectation(params, angles; clause_sign=default_clause_sign(algebra))
end

"""
    qaoa_expectation(params, angles; clause_sign=1) -> Float64

Evaluate the exact expected satisfaction of the root clause

`(1 + clause_sign * Z₁⋯Z_k) / 2`

using the exact finite-D Tier 2 branch-transfer contraction in the physical
`γ/2` convention.

Set `clause_sign = -1` for odd clauses such as MaxCut edges.

This is a convenience wrapper that constructs an `XORSATAlgebra` internally.
"""
function qaoa_expectation(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
)
    algebra = algebra_from_clause_sign(params.k, clause_sign)
    qaoa_expectation(algebra, params, angles)
end
