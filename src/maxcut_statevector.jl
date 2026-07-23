"""
    MaxCutStatevectorEval

Prepared exact finite-`N` MaxCut QAOA evaluator. The evaluator stores the graph,
the diagonal cut value of every computational basis state, `|+⟩^N`, and reusable
state, adjoint, and gradient buffers.

An evaluator is not thread-safe: evaluation calls mutate its scratch buffers.
Use one evaluator per concurrent task.
"""
mutable struct MaxCutStatevectorEval
    N::Int
    edges::Vector{NTuple{2,Int}}
    weights::Vector{Float64}
    cut_values::Vector{Float64}
    initial_state::Vector{ComplexF64}
    state::Vector{ComplexF64}
    adjoint::Vector{ComplexF64}
    gradient::Vector{Float64}
end

function _maxcut_edges(N::Int, edges)
    result = NTuple{2,Int}[]
    seen = Set{NTuple{2,Int}}()

    for (edge_index, edge) in enumerate(edges)
        (edge isa Tuple || edge isa AbstractVector) || throw(ArgumentError(
            "edge $edge_index must be a 2-element tuple or vector"))
        length(edge) == 2 || throw(ArgumentError(
            "edge $edge_index must have exactly 2 endpoints"))

        u, v = edge
        (u isa Integer && !(u isa Bool) && v isa Integer && !(v isa Bool)) ||
            throw(ArgumentError("edge $edge_index endpoints must be integers"))
        1 ≤ u ≤ N || throw(ArgumentError(
            "edge $edge_index endpoint $u is outside 1:$N"))
        1 ≤ v ≤ N || throw(ArgumentError(
            "edge $edge_index endpoint $v is outside 1:$N"))
        u != v || throw(ArgumentError("self-loop at vertex $u is not allowed"))

        normalized = u < v ? (Int(u), Int(v)) : (Int(v), Int(u))
        normalized ∉ seen || throw(ArgumentError(
            "duplicate undirected edge $(normalized)"))
        push!(seen, normalized)
        push!(result, normalized)
    end

    result
end

function _maxcut_weights(weights, edge_count::Int)
    weights === nothing && return ones(Float64, edge_count)
    weights isa AbstractVector || throw(ArgumentError("weights must be a vector"))
    length(weights) == edge_count || throw(ArgumentError(
        "weight count $(length(weights)) does not match edge count $edge_count"))

    result = Vector{Float64}(undef, edge_count)
    for index in eachindex(weights)
        weight = weights[index]
        weight isa Real || throw(ArgumentError("weight $index must be real"))
        value = Float64(weight)
        isfinite(value) || throw(ArgumentError("weight $index must be finite"))
        value ≥ 0 || throw(ArgumentError("weight $index must be nonnegative"))
        result[index] = value
    end
    result
end

function _maxcut_diagonal(
    N::Int,
    edges::Vector{NTuple{2,Int}},
    weights::Vector{Float64},
)
    dimension = one(Int) << N
    cut_values = zeros(Float64, dimension)

    @inbounds for basis in 0:(dimension-1)
        value = 0.0
        for edge_index in eachindex(edges)
            u, v = edges[edge_index]
            u_bit = (basis >> (u - 1)) & 1
            v_bit = (basis >> (v - 1)) & 1
            value += weights[edge_index] * xor(u_bit, v_bit)
        end
        cut_values[basis+1] = value
    end
    cut_values
end

"""
    prepare_maxcut_eval(N, edges; weights=nothing) -> MaxCutStatevectorEval

Prepare an exact matrix-free finite-graph MaxCut evaluator. Vertices are
1-indexed. `edges` contains two-endpoint tuples or vectors; omitted weights
default to one. Explicit weights must be finite, nonnegative real values.

The objective convention is
`C(z) = Σ_(i,j) w_ij (1 - z_i z_j) / 2`.
No `2^N × 2^N` matrix is constructed.
"""
function prepare_maxcut_eval(N::Integer, edges; weights=nothing)
    N isa Bool && throw(ArgumentError("N must be an integer qubit count"))
    N ≥ 1 || throw(ArgumentError("N must be at least 1, got $N"))
    N < Sys.WORD_SIZE - 1 || throw(ArgumentError(
        "N=$N is too large for state indexing on a $(Sys.WORD_SIZE)-bit system"))

    vertex_count = Int(N)
    validated_edges = _maxcut_edges(vertex_count, edges)
    validated_weights = _maxcut_weights(weights, length(validated_edges))
    cut_values = _maxcut_diagonal(vertex_count, validated_edges, validated_weights)

    dimension = length(cut_values)
    amplitude = inv(sqrt(Float64(dimension)))
    initial_state = fill(ComplexF64(amplitude), dimension)
    MaxCutStatevectorEval(
        vertex_count,
        validated_edges,
        validated_weights,
        cut_values,
        initial_state,
        similar(initial_state),
        similar(initial_state),
        Float64[],
    )
end

function _validate_maxcut_angles(angles::QAOAAngles)
    depth(angles) ≥ 1 || throw(ArgumentError("angle depth must be at least 1"))
    all(isfinite, angles.γ) || throw(ArgumentError("γ angles must be finite"))
    all(isfinite, angles.β) || throw(ArgumentError("β angles must be finite"))
    angles
end

function _apply_maxcut_cost!(
    state::Vector{ComplexF64},
    cut_values::Vector{Float64},
    gamma::Float64,
)
    iszero(gamma) && return state
    @inbounds for index in eachindex(state, cut_values)
        state[index] *= cis(-gamma * cut_values[index])
    end
    state
end

function _apply_maxcut_mixer!(
    state::Vector{ComplexF64},
    beta::Float64,
    N::Int,
)
    iszero(beta) && return state
    cosine = cos(beta)
    sine = ComplexF64(0.0, -sin(beta))

    for qubit in 1:N
        half_block = one(Int) << (qubit - 1)
        block_size = 2 * half_block
        @inbounds for block_start in 1:block_size:length(state)
            for offset in 0:(half_block-1)
                zero_index = block_start + offset
                one_index = zero_index + half_block
                zero_amplitude = state[zero_index]
                one_amplitude = state[one_index]
                state[zero_index] = cosine * zero_amplitude + sine * one_amplitude
                state[one_index] = sine * zero_amplitude + cosine * one_amplitude
            end
        end
    end
    state
end

function _maxcut_forward!(ev::MaxCutStatevectorEval, angles::QAOAAngles)
    _validate_maxcut_angles(angles)
    copyto!(ev.state, ev.initial_state)
    for layer in 1:depth(angles)
        _apply_maxcut_cost!(ev.state, ev.cut_values, Float64(angles.γ[layer]))
        _apply_maxcut_mixer!(ev.state, Float64(angles.β[layer]), ev.N)
    end
    ev.state
end

@inline function _maxcut_expectation(
    state::Vector{ComplexF64},
    cut_values::Vector{Float64},
)
    value = 0.0
    @inbounds for index in eachindex(state, cut_values)
        value += cut_values[index] * abs2(state[index])
    end
    value
end

"""
    maxcut_state(ev, angles) -> Vector{ComplexF64}

Return the exact QAOA state obtained from `|+⟩^N` by applying
`exp(-i γ_l C)` and then `exp(-i β_l Σ_i X_i)` at every layer. The returned
state is an independent allocation and is not overwritten by later calls.
"""
maxcut_state(ev::MaxCutStatevectorEval, angles::QAOAAngles) =
    copy(_maxcut_forward!(ev, angles))

"""
    maxcut_expectation(ev, angles) -> Float64

Evaluate `⟨C⟩` exactly using the evaluator's reusable state buffer.
"""
function maxcut_expectation(ev::MaxCutStatevectorEval, angles::QAOAAngles)
    state = _maxcut_forward!(ev, angles)
    _maxcut_expectation(state, ev.cut_values)
end

"""
    maxcut_value_max(ev) -> Float64

Return the exact maximum cut value from the precomputed diagonal table.
"""
maxcut_value_max(ev::MaxCutStatevectorEval) = maximum(ev.cut_values)

"""
    maxcut_success_probability(ev, angles) -> Float64

Return the probability of measuring any computational basis state attaining
the exact maximum cut value.
"""
function maxcut_success_probability(ev::MaxCutStatevectorEval, angles::QAOAAngles)
    state = _maxcut_forward!(ev, angles)
    maximum_value = maxcut_value_max(ev)
    probability = 0.0
    @inbounds for index in eachindex(state, ev.cut_values)
        ev.cut_values[index] == maximum_value || continue
        probability += abs2(state[index])
    end
    probability
end

function _maxcut_mixer_overlap(
    adjoint::Vector{ComplexF64},
    state::Vector{ComplexF64},
    N::Int,
)
    overlap = ComplexF64(0.0)
    for qubit in 1:N
        half_block = one(Int) << (qubit - 1)
        block_size = 2 * half_block
        @inbounds for block_start in 1:block_size:length(state)
            for offset in 0:(half_block-1)
                zero_index = block_start + offset
                one_index = zero_index + half_block
                overlap += conj(adjoint[zero_index]) * state[one_index]
                overlap += conj(adjoint[one_index]) * state[zero_index]
            end
        end
    end
    overlap
end

function _maxcut_cost_overlap(
    adjoint::Vector{ComplexF64},
    state::Vector{ComplexF64},
    cut_values::Vector{Float64},
)
    overlap = ComplexF64(0.0)
    @inbounds for index in eachindex(adjoint, state, cut_values)
        overlap += conj(adjoint[index]) * cut_values[index] * state[index]
    end
    overlap
end

"""
    maxcut_expectation_and_gradient!(gradient, ev, angles) -> Float64

Write the analytic gradient of `⟨C⟩` into `gradient` in the layout
`[∂/∂γ₁, …, ∂/∂γ_p, ∂/∂β₁, …, ∂/∂β_p]` and return the expectation.

The reverse-mode pass uses only the evaluator's state and adjoint buffers. It
reversibly undoes each unitary layer, so working memory is `O(2^N)` independent
of depth and runtime is `O(p N 2^N)`.
"""
function maxcut_expectation_and_gradient!(
    gradient::AbstractVector{Float64},
    ev::MaxCutStatevectorEval,
    angles::QAOAAngles,
)
    _validate_maxcut_angles(angles)
    p = depth(angles)
    length(gradient) == 2p || throw(ArgumentError(
        "gradient length must be 2p=$(2p), got $(length(gradient))"))

    state = _maxcut_forward!(ev, angles)
    value = _maxcut_expectation(state, ev.cut_values)
    @inbounds for index in eachindex(ev.adjoint, state, ev.cut_values)
        ev.adjoint[index] = ev.cut_values[index] * state[index]
    end

    for layer in p:-1:1
        gradient[p+layer] = 2 * imag(_maxcut_mixer_overlap(
            ev.adjoint, state, ev.N))

        beta = Float64(angles.β[layer])
        _apply_maxcut_mixer!(state, -beta, ev.N)
        _apply_maxcut_mixer!(ev.adjoint, -beta, ev.N)

        gradient[layer] = 2 * imag(_maxcut_cost_overlap(
            ev.adjoint, state, ev.cut_values))

        gamma = Float64(angles.γ[layer])
        _apply_maxcut_cost!(state, ev.cut_values, -gamma)
        _apply_maxcut_cost!(ev.adjoint, ev.cut_values, -gamma)
    end

    value
end

"""
    maxcut_expectation_and_gradient(ev, angles) -> (value, gradient)

Return `⟨C⟩` and its analytic gradient in
`[∂/∂γ₁, …, ∂/∂γ_p, ∂/∂β₁, …, ∂/∂β_p]` order. The returned gradient is an
evaluator-owned scratch vector and is overwritten by the next gradient call.
Use `maxcut_expectation_and_gradient!` with caller-owned storage when retaining
multiple gradients.
"""
function maxcut_expectation_and_gradient(
    ev::MaxCutStatevectorEval,
    angles::QAOAAngles,
)
    required_length = 2depth(angles)
    length(ev.gradient) == required_length || resize!(ev.gradient, required_length)
    value = maxcut_expectation_and_gradient!(ev.gradient, ev, angles)
    value, ev.gradient
end
