struct MaxCutTransferParams
    degree::Int
    angles::QAOAAngles
    mirrored_gammas::Vector{Float64}
    trigs::Matrix{ComplexF64}
    ratios::Vector{ComplexF64}

    function MaxCutTransferParams(degree::Int, angles::QAOAAngles)
        degree ≥ 2 || throw(ArgumentError("degree must be ≥ 2, got $degree"))

        p = depth(angles)
        mirrored_gammas = zeros(Float64, 2p)
        trigs = zeros(ComplexF64, 2, 2p)
        ratios = zeros(ComplexF64, 2p)

        for round in 1:p
            mirror = 2p - round + 1
            β = angles.β[round]

            mirrored_gammas[round] = angles.γ[round]
            mirrored_gammas[mirror] = -angles.γ[round]

            trigs[1, round] = cos(β)
            trigs[2, round] = ComplexF64(0.0, sin(β))
            trigs[1, mirror] = cos(-β)
            trigs[2, mirror] = ComplexF64(0.0, sin(-β))

            ratios[round] = trigs[2, round] / trigs[1, round]
            ratios[mirror] = trigs[2, mirror] / trigs[1, mirror]
        end

        new(degree, angles, mirrored_gammas, trigs, ratios)
    end
end

function broadcast_maxcut_transfer_corner!(matrix::Matrix{ComplexF64}, p::Int)
    last_index = 2p

    for row in 0:p, column in row:p
        matrix[column + 1, row + 1] = matrix[row + 1, column + 1]
        matrix[row + 1, last_index - column + 1] = matrix[row + 1, column + 1]
        matrix[last_index - column + 1, row + 1] = matrix[row + 1, column + 1]
        matrix[column + 1, last_index - row + 1] = conj(matrix[row + 1, column + 1])
        matrix[last_index - column + 1, last_index - row + 1] =
            conj(matrix[row + 1, column + 1])
        matrix[last_index - row + 1, last_index - column + 1] =
            conj(matrix[row + 1, column + 1])
        matrix[last_index - row + 1, column + 1] = conj(matrix[row + 1, column + 1])
    end

    matrix
end

function maxcut_transfer_exponent(
    reduced_matrix::Matrix{ComplexF64},
    reduced_gammas::Vector{Float64},
    configuration::Int,
)
    size(reduced_matrix, 1) == length(reduced_gammas) ||
        throw(ArgumentError("matrix/gamma dimensions must match"))

    exponent = 0.0 + 0.0im
    bit_count = length(reduced_gammas)

    for left in 0:bit_count-1
        left_bit = !iszero(configuration & (1 << left))
        element = -0.5 * reduced_matrix[left + 1, left + 1] * reduced_gammas[left + 1]

        for right in left+1:bit_count-1
            right_bit = !iszero(configuration & (1 << right))
            sign = left_bit == right_bit ? -1 : 1
            element += sign * reduced_matrix[left + 1, right + 1] * reduced_gammas[right + 1]
        end

        exponent += element * reduced_gammas[left + 1]
    end

    exponent
end

function maxcut_transfer_ratio(ratios::Vector{ComplexF64}, bit_difference::Int)
    ratio = 1.0 + 0.0im

    for index in 0:length(ratios)-1
        if !iszero(bit_difference & (1 << index))
            ratio *= ratios[index + 1]
        end
    end

    ratio
end

function build_maxcut_transfer_column!(
    matrix::Matrix{ComplexF64},
    params::MaxCutTransferParams,
    column::Int,
)
    p = depth(params.angles)
    reduced_size = 2column
    dimension = one(Int) << reduced_size

    if iszero(reduced_size)
        matrix[1, 1] = 1.0 + 0.0im
        return matrix
    end

    reduced_indices = Vector{Int}(undef, reduced_size)
    reduced_matrix_indices = Vector{Int}(undef, reduced_size)

    for offset in 0:column-1
        reduced_indices[offset + 1] = offset + 1
        reduced_indices[reduced_size - offset] = 2p - offset
        reduced_matrix_indices[offset + 1] = offset + 1
        reduced_matrix_indices[reduced_size - offset] = 2p - offset + 1
    end

    reduced_matrix = matrix[reduced_matrix_indices, reduced_matrix_indices] .^ (params.degree - 1)
    reduced_gammas = params.mirrored_gammas[reduced_indices]
    reduced_trigs = params.trigs[:, reduced_indices]
    reduced_ratios = params.ratios[reduced_indices]

    trig_product = ComplexF64(0.5)
    for index in 1:reduced_size
        trig_product *= reduced_trigs[1, index]
    end

    column_entries = zeros(ComplexF64, column + 1)

    for configuration in 0:dimension-1
        h_value = exp(maxcut_transfer_exponent(reduced_matrix, reduced_gammas, configuration))

        left_bits = (configuration >> column) << column
        right_bits = configuration - left_bits
        extended = (left_bits << 1) + right_bits
        bit_difference = xor(extended, extended >> 1)
        f_value = trig_product * maxcut_transfer_ratio(reduced_ratios, bit_difference)

        pivot_bit = !iszero(extended & (1 << column))
        for row in 0:column
            row_bit = !iszero(extended & (1 << row))
            sign = pivot_bit == row_bit ? 1 : -1
            column_entries[row + 1] += sign * f_value * h_value
        end
    end

    for row in 0:column
        matrix[row + 1, column + 1] = 2 * column_entries[row + 1]
    end

    matrix
end

"""
    build_maxcut_transfer_matrix(degree, angles) -> Matrix{ComplexF64}

Build the compact `(2p + 1) × (2p + 1)` transfer matrix used by the upstream
large-girth MaxCut recursion of Basso et al. / Villalonga et al.

This object is kept internal to the current branch because it is not yet wired
to the repository's exact finite-`D` `qaoa_expectation` API.
"""
function build_maxcut_transfer_matrix(
    degree::Int,
    angles::QAOAAngles,
)::Matrix{ComplexF64}
    params = MaxCutTransferParams(degree, angles)
    p = depth(angles)
    matrix = zeros(ComplexF64, 2p + 1, 2p + 1)

    for column in 0:p
        build_maxcut_transfer_column!(matrix, params, column)
        broadcast_maxcut_transfer_corner!(matrix, p)
    end

    matrix
end

"""
    maxcut_transfer_objective(degree, angles) -> Float64

Evaluate the scalar objective associated with the compact large-girth MaxCut
transfer matrix.

This follows the upstream recursion exactly and is currently used as an
experimental bridge toward a future exact transfer-tensor implementation.
"""
function maxcut_transfer_objective(
    degree::Int,
    angles::QAOAAngles,
)::Float64
    params = MaxCutTransferParams(degree, angles)
    matrix = build_maxcut_transfer_matrix(degree, angles)
    p = depth(angles)

    objective = 0.0
    for row in 1:p
        objective += -imag(matrix[row, p + 1]^degree) * params.mirrored_gammas[row]
    end

    objective * sqrt(2 / degree)
end