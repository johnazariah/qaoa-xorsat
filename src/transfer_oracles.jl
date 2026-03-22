function flattened_hyperindex_position(configuration, dimension::Int)
    index = 1
    stride = 1

    for hyperindex in configuration
        index += hyperindex * stride
        stride *= dimension
    end

    index
end

"""
    contract_diagonal_tensor_messages(messages, tensor)

Contract a flattened diagonal `k`-body tensor against one message per tensor
leg and return the resulting scalar.

Each message must have the same length `d`, and the tensor must have length
`d^k` in the same row-major flattening convention used throughout the raw
transfer helpers.
"""
function contract_diagonal_tensor_messages(
    messages::AbstractVector{<:AbstractVector},
    tensor::AbstractVector{<:Number},
)::ComplexF64
    !isempty(messages) || throw(ArgumentError("need at least one message"))

    dimension = length(first(messages))
    all(length(message) == dimension for message in messages) || throw(ArgumentError(
        "all messages must have length $dimension",
    ))
    length(tensor) == dimension^length(messages) || throw(ArgumentError(
        "tensor length $(length(tensor)) does not match $(length(messages)) messages of length $dimension",
    ))

    ranges = ntuple(_ -> 0:dimension-1, length(messages))
    total = 0.0 + 0.0im

    for configuration in Iterators.product(ranges...)
        weight = ComplexF64(tensor[flattened_hyperindex_position(configuration, dimension)])

        for (message, hyperindex) in zip(messages, configuration)
            weight *= message[hyperindex+1]
        end

        total += weight
    end

    total
end

"""
    contract_tensor_messages(messages, tensor)

Contract a flattened `k`-body tensor against one message per tensor leg and
return the resulting scalar.

This is the dense analogue of `contract_diagonal_tensor_messages`. The tensor is
flattened in the same row-major convention used by the raw transfer helpers and
has length `d^k`, where `d` is the common message length.
"""
function contract_tensor_messages(
    messages::AbstractVector{<:AbstractVector},
    tensor::AbstractVector{<:Number},
)::ComplexF64
    !isempty(messages) || throw(ArgumentError("need at least one message"))

    dimension = length(first(messages))
    all(length(message) == dimension for message in messages) || throw(ArgumentError(
        "all messages must have length $dimension",
    ))
    length(tensor) == dimension^length(messages) || throw(ArgumentError(
        "tensor length $(length(tensor)) does not match $(length(messages)) messages of length $dimension",
    ))

    ranges = ntuple(_ -> 0:dimension-1, length(messages))
    total = 0.0 + 0.0im

    for configuration in Iterators.product(ranges...)
        weight = ComplexF64(tensor[flattened_hyperindex_position(configuration, dimension)])

        for (message, hyperindex) in zip(messages, configuration)
            weight *= message[hyperindex+1]
        end

        total += weight
    end

    total
end

"""
    apply_local_matrix_to_tensor(tensor, local_map, arity)

Apply the same one-leg linear map `local_map` to every leg of a flattened
`arity`-body tensor.

The input tensor is interpreted as a dense `arity`-way tensor with equal leg
dimension `size(local_map, 1)`. The returned tensor is flattened back into the
same row-major convention used throughout the raw transfer helpers.
"""
function apply_local_matrix_to_tensor(
    tensor::AbstractVector{<:Number},
    local_map::AbstractMatrix{<:Number},
    arity::Int,
)::Vector{ComplexF64}
    arity ≥ 1 || throw(ArgumentError("arity must be ≥ 1, got $arity"))

    dimension = size(local_map, 1)
    size(local_map, 2) == dimension || throw(ArgumentError(
        "local_map must be square, got size $(size(local_map))",
    ))
    length(tensor) == dimension^arity || throw(ArgumentError(
        "tensor length $(length(tensor)) does not match dimension $dimension and arity $arity",
    ))

    data = reshape(ComplexF64.(tensor), ntuple(_ -> dimension, arity)...)
    for leg in 1:arity
        permutation = [leg; collect(1:(leg-1)); collect((leg+1):arity)]
        inverse_permutation = invperm(permutation)
        permuted = permutedims(data, permutation)
        reshaped = reshape(permuted, dimension, :)
        mapped = ComplexF64.(local_map) * reshaped
        data = permutedims(reshape(mapped, size(permuted)...), inverse_permutation)
    end

    vec(data)
end

"""
    contract_constraint_message(child_messages, γ, slice, p; clause_sign=1)

Contract the raw diagonal problem tensor of one `k`-body constraint against the
`k - 1` child branch messages and return the message passed toward the parent
variable.

This is the multilinear constraint update that replaces the incorrect draft rule
`branch .^ (k - 1)` at non-root constraints. It is intentionally written as a
small exact oracle for derivation work rather than an optimized production
routine.
"""
function contract_constraint_message(
    child_messages::AbstractVector{<:AbstractVector},
    γ::Real,
    slice::Int,
    p::Int;
    clause_sign::Int=1,
)::Vector{ComplexF64}
    validate_depth(p)
    validate_slice(slice, p)
    validate_clause_sign(clause_sign)
    !isempty(child_messages) || throw(ArgumentError("need at least one child message"))

    dimension = hyperindex_dimension(p)
    all(length(message) == dimension for message in child_messages) ||
        throw(ArgumentError("all child messages must have length $dimension"))

    child_count = length(child_messages)
    arity = child_count + 1
    tensor = problem_tensor(arity, γ, slice, p; clause_sign)
    parent_message = zeros(ComplexF64, dimension)
    parent_basis = zeros(ComplexF64, dimension)

    for parent_hyperindex in 0:dimension-1
        fill!(parent_basis, 0.0 + 0.0im)
        parent_basis[parent_hyperindex+1] = 1.0 + 0.0im
        parent_message[parent_hyperindex+1] = contract_diagonal_tensor_messages(
            AbstractVector{ComplexF64}[parent_basis, child_messages...],
            tensor,
        )
    end

    parent_message
end
