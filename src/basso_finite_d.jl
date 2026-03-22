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
    for index in 1:(bit_count - 1)
        bit_difference = xor(bits[index], bits[index + 1])
        weight *= trigs[bit_difference + 1, index]
    end

    weight
end