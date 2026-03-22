"""
    build_gamma_vector(angles)

Build the mirrored `Γ` vector used by the Basso / Villalonga transfer-style
recurrences.

For depth `p`, this returns a length-`2p` vector

`(γ₁, γ₂, …, γₚ, -γₚ, …, -γ₂, -γ₁)`.

These `2p` entries live on the transitions between the `2p + 1` bits of the
branch bitstring used by the recurrence.
"""
function build_gamma_vector(angles::QAOAAngles)::Vector{Float64}
    p = depth(angles)
    gamma_vector = zeros(Float64, 2p)

    for round in 1:p
        mirror = 2p - round + 1
        gamma_vector[round] = angles.γ[round]
        gamma_vector[mirror] = -angles.γ[round]
    end

    gamma_vector
end

"""Number of Basso branch bits, namely `2p + 1`."""
basso_bit_count(p::Int) = 2 * p + 1

"""Index of the central Basso bit `a^[0]` inside the `(2p + 1)`-bit branch string."""
basso_root_bit_index(p::Int) = p + 1

function basso_phase_bit_positions(p::Int)::Vector{Int}
    [collect(1:p); collect((p+2):(2p+1))]
end

function build_gamma_full_vector(angles::QAOAAngles)::Vector{Float64}
    p = depth(angles)
    gamma_full = zeros(Float64, basso_bit_count(p))
    gamma_vector = build_gamma_vector(angles)

    for (gamma_index, bit_index) in pairs(basso_phase_bit_positions(p))
        gamma_full[bit_index] = gamma_vector[gamma_index]
    end

    gamma_full
end

"""Number of Basso branch configurations, namely `2^(2p + 1)`."""
function basso_configuration_count(p::Int)
    validate_depth(p)
    one(Int) << basso_bit_count(p)
end

"""
    decode_bits(configuration, bit_count)

Decode the least-significant-bit-first representation of a Basso branch
configuration into a vector of `0/1` integers.
"""
function decode_bits(configuration::Integer, bit_count::Int)::Vector{Int}
    validate_hyperindex(configuration)
    bit_count ≥ 1 || throw(ArgumentError("bit_count must be ≥ 1, got $bit_count"))

    bits = Vector{Int}(undef, bit_count)
    encoded = Int(configuration)

    encoded < (one(Int) << bit_count) ||
        throw(ArgumentError("configuration $(configuration) exceeds $bit_count bits"))

    for index in 1:bit_count
        bits[index] = (encoded >> (index - 1)) & 1
    end

    bits
end

function basso_trig_table(angles::QAOAAngles)::Matrix{ComplexF64}
    p = depth(angles)
    trigs = zeros(ComplexF64, 2, 2p)

    for round in 1:p
        mirror = 2p - round + 1
        β = angles.β[round]

        trigs[1, round] = cos(β)
        trigs[2, round] = ComplexF64(0.0, sin(β))
        trigs[1, mirror] = cos(-β)
        trigs[2, mirror] = ComplexF64(0.0, sin(-β))
    end

    trigs
end

"""
    f_function(angles, configuration)

Evaluate the Basso / Villalonga mixer weight `f(a)` for a `(2p + 1)`-bit branch
configuration `a`.

For each of the `2p` transitions between adjacent bits, the factor contributes
`cos(β)` if the two neighbouring bits agree and `i sin(β)` if they differ,
using the mirrored angle convention of the recurrence. A global `1/2` factor is
included, matching the upstream MaxCut implementation.
"""
function f_function(angles::QAOAAngles, configuration::Integer)::ComplexF64
    p = depth(angles)
    bit_count = basso_bit_count(p)
    bits = decode_bits(configuration, bit_count)
    trigs = basso_trig_table(angles)

    weight = ComplexF64(0.5)
    for index in 1:(bit_count-1)
        bit_difference = xor(bits[index], bits[index+1])
        weight *= trigs[bit_difference+1, index]
    end

    weight
end

"""Number of branching child constraints in the finite-D Basso iteration."""
basso_branching_degree(params::TreeParams) = params.D - 1

"""
    basso_phase_argument(gamma_vector, parent_bits, child_bits...)

Evaluate the finite-D Basso phase dot product

`Γ ⋅ (a b¹ … b^{k-1})`

using the first `2p` positions of the `(2p + 1)`-bit branch strings. Bit values
are interpreted as `Z` eigenvalues via `0 ↦ +1` and `1 ↦ -1`.
"""
function basso_phase_argument(
    gamma_vector::AbstractVector{<:Real},
    parent_bits::AbstractVector{<:Integer},
    child_bits::Vararg{AbstractVector{<:Integer}},
)::Float64
    bit_count = length(gamma_vector) + 1
    length(parent_bits) == bit_count || throw(ArgumentError(
        "parent bit count $(length(parent_bits)) does not match gamma length $(length(gamma_vector))",
    ))

    for bits in child_bits
        length(bits) == bit_count || throw(ArgumentError(
            "child bit count $(length(bits)) does not match gamma length $(length(gamma_vector))",
        ))
    end

    phase = 0.0
    for (gamma_index, bit_index) in pairs(basso_phase_bit_positions(length(gamma_vector) ÷ 2))
        parity = z_eigenvalue(Int(parent_bits[bit_index]))
        for bits in child_bits
            parity *= z_eigenvalue(Int(bits[bit_index]))
        end
        phase += gamma_vector[gamma_index] * parity
    end

    phase
end

function _basso_child_sum(
    parent_bits::Vector{Int},
    all_bits::Vector{Vector{Int}},
    child_weights::Vector{ComplexF64},
    gamma_vector::Vector{Float64},
    phase_scale::Float64,
    child_arity::Int,
)::ComplexF64
    selections = Int[]

    function recurse(weight_product::ComplexF64, remaining::Int)::ComplexF64
        if iszero(remaining)
            selected_bits = map(index -> all_bits[index], selections)
            θ = phase_scale * basso_phase_argument(gamma_vector, parent_bits, selected_bits...)
            return cos(θ) * weight_product
        end

        total = 0.0 + 0.0im
        for child_index in eachindex(all_bits)
            push!(selections, child_index)
            total += recurse(weight_product * child_weights[child_index], remaining - 1)
            pop!(selections)
        end

        total
    end

    recurse(1.0 + 0.0im, child_arity)
end

function configuration_spins(configuration::Integer, bit_count::Int)::Vector{Int}
    [z_eigenvalue((Int(configuration) >> (index - 1)) & 1) for index in 1:bit_count]
end

"""
    basso_configuration_to_hyperindex(configuration, p)

Map a `(2p + 1)`-bit Basso branch configuration

`(a^[1], …, a^[p], a^[0], a^[-p], …, a^[-1])`

to the raw P1.2 hyperindex convention

`(ket₁, bra₁, ket₂, bra₂, …, ket_p, bra_p)`

where slice `1` is the innermost / root slice and slice `p` is the outermost
/ boundary slice.
"""
function basso_configuration_to_hyperindex(configuration::Integer, p::Int)::Int
    bits = decode_bits(configuration, basso_bit_count(p))
    hyperindex = 0

    for physical_round in 1:p
        slice = slice_from_physical_round(physical_round, p)
        ket_position, bra_position = slice_bit_positions(slice, p)
        negative_index = 2p - physical_round + 2

        hyperindex |= bits[physical_round] << (ket_position - 1)
        hyperindex |= bits[negative_index] << (bra_position - 1)
    end

    hyperindex
end

"""
    lift_hyperindex_message_to_branch_basis(message, angles)

Lift a raw P1.2 hyperindex-space message on one variable line into the expanded
`(2p + 1)`-bit Basso branch basis.

For a branch configuration `a`, this applies the local variable-line sandwich
factor associated with the inserted complete sets and reads the raw message at
the corresponding hyperindex `σ(a)`.

At the leaf boundary, where `leaf_tensor(angles)` is the constant `2^{-p}`
vector, this lift reproduces `f(a)` exactly.
"""
function lift_hyperindex_message_to_branch_basis(
    message::AbstractVector{<:Number},
    angles::QAOAAngles,
)::Vector{ComplexF64}
    dimension = hyperindex_dimension(depth(angles))
    length(message) == dimension || throw(ArgumentError(
        "message length $(length(message)) does not match hyperindex dimension $(dimension)",
    ))

    scale = float(one(Int) << depth(angles))
    ComplexF64[
        scale * ComplexF64(message[basso_configuration_to_hyperindex(configuration, depth(angles)) + 1]) *
        f_function(angles, configuration)
        for configuration in 0:basso_configuration_count(depth(angles))-1
    ]
end

function basso_constraint_kernel(
    angles::QAOAAngles,
    branch_degree::Int,
)::Vector{ComplexF64}
    bit_count = basso_bit_count(depth(angles))
    configuration_count = basso_configuration_count(depth(angles))
    gamma_full = build_gamma_full_vector(angles)
    phase_scale = inv(sqrt(float(branch_degree)))

    ComplexF64[
        cos(phase_scale * sum(gamma_full .* configuration_spins(configuration, bit_count)))
        for configuration in 0:configuration_count-1
    ]
end

function basso_constraint_fold(
    child_message::AbstractVector{<:Number},
    kernel::AbstractVector{<:Number},
    child_arity::Int,
)::Vector{ComplexF64}
    child_arity ≥ 1 || throw(ArgumentError("child_arity must be ≥ 1, got $child_arity"))
    length(child_message) == length(kernel) || throw(ArgumentError(
        "child_message and kernel must have equal length",
    ))

    child_hat = wht(ComplexF64.(child_message))
    kernel_hat = wht(ComplexF64.(kernel))
    iwht(kernel_hat .* (child_hat .^ child_arity))
end

function basso_root_parity(configuration::Integer, p::Int)::Int
    root_bit = basso_root_bit_index(p)
    z_eigenvalue((Int(configuration) >> (root_bit - 1)) & 1)
end

function basso_root_message(
    params::TreeParams,
    angles::QAOAAngles,
    branch_tensor::AbstractVector{<:Number},
)::Vector{ComplexF64}
    configuration_count = basso_configuration_count(params.p)
    length(branch_tensor) == configuration_count || throw(ArgumentError(
        "branch tensor length $(length(branch_tensor)) does not match configuration count $(configuration_count)",
    ))

    ComplexF64[
        f_function(angles, configuration) * ComplexF64(branch_tensor[configuration + 1])
        for configuration in 0:configuration_count-1
    ]
end

function basso_root_parity_kernel(p::Int)::Vector{ComplexF64}
    configuration_count = basso_configuration_count(p)

    ComplexF64[
        basso_root_parity(configuration, p)
        for configuration in 0:configuration_count-1
    ]
end

function basso_root_problem_kernel(
    angles::QAOAAngles,
    branch_degree::Int;
    clause_sign::Int=1,
)::Vector{ComplexF64}
    validate_clause_sign(clause_sign)

    gamma_full = build_gamma_full_vector(angles)
    configuration_count = basso_configuration_count(depth(angles))
    bit_count = basso_bit_count(depth(angles))
    phase_scale = -0.5 * clause_sign * inv(sqrt(float(branch_degree)))

    ComplexF64[
        cis(phase_scale * sum(gamma_full .* configuration_spins(configuration, bit_count)))
        for configuration in 0:configuration_count-1
    ]
end

function basso_root_kernel(
    angles::QAOAAngles,
    branch_degree::Int,
    p::Int=depth(angles);
    clause_sign::Int=1,
)::Vector{ComplexF64}
    basso_root_parity_kernel(p) .* basso_root_problem_kernel(
        angles,
        branch_degree;
        clause_sign,
    )
end

function basso_root_fold(
    root_message::AbstractVector{<:Number},
    kernel::AbstractVector{<:Number},
    arity::Int,
)::ComplexF64
    arity ≥ 1 || throw(ArgumentError("arity must be ≥ 1, got $arity"))
    length(root_message) == length(kernel) || throw(ArgumentError(
        "root_message and kernel must have equal length",
    ))

    sum(ComplexF64.(kernel) .* xor_convolution_power(ComplexF64.(root_message), arity))
end

function basso_root_parity_sum(
    params::TreeParams,
    angles::QAOAAngles,
    branch_tensor::AbstractVector{<:Number};
    clause_sign::Int=1,
)::ComplexF64
    root_message = basso_root_message(params, angles, branch_tensor)
    root_kernel = basso_root_kernel(
        angles,
        basso_branching_degree(params),
        params.p;
        clause_sign,
    )

    basso_root_fold(root_message, root_kernel, params.k)
end

"""
    basso_branch_tensor_step(params, angles, previous)

Apply one exact finite-D Basso branch-tensor update (Eq. 8.7) to the branch
tensor `previous`.

The returned vector has one entry for each `(2p + 1)`-bit branch configuration.
"""
function basso_branch_tensor_step(
    params::TreeParams,
    angles::QAOAAngles,
    previous::AbstractVector{<:Number},
)::Vector{ComplexF64}
    depth(angles) == params.p || throw(ArgumentError("angle depth must match tree depth"))

    p = params.p
    configuration_count = basso_configuration_count(p)
    length(previous) == configuration_count || throw(ArgumentError(
        "previous tensor length $(length(previous)) does not match configuration count $(configuration_count)",
    ))

    child_arity = params.k - 1
    branch_degree = basso_branching_degree(params)
    child_weights = ComplexF64[
        f_function(angles, configuration) * ComplexF64(previous[configuration+1])
        for configuration in 0:configuration_count-1
    ]
    kernel = basso_constraint_kernel(angles, branch_degree)
    branch_sum = basso_constraint_fold(child_weights, kernel, child_arity)

    branch_sum .^ branch_degree
end

"""
    basso_branch_tensor(params, angles; steps=params.p)

Iterate the exact finite-D Basso branch tensor for `steps` levels, starting from
`H_D^(0) = 1`.

This object lives in the proof's expanded `(2p + 1)`-bit branch basis

`(a^[1], …, a^[p], a^[0], a^[-p], …, a^[-1])`

after inserting complete sets along the variable line. It should be interpreted
as a partially contracted subtree partition function in that basis, not as a
raw parent-facing hyperindex-space transfer message.
"""
function basso_branch_tensor(
    params::TreeParams,
    angles::QAOAAngles;
    steps::Int=params.p,
)::Vector{ComplexF64}
    0 ≤ steps ≤ params.p || throw(ArgumentError("steps must lie in 0:$(params.p), got $steps"))
    depth(angles) == params.p || throw(ArgumentError("angle depth must match tree depth"))

    current = ones(ComplexF64, basso_configuration_count(params.p))
    for _ in 1:steps
        current = basso_branch_tensor_step(params, angles, current)
    end

    current
end
