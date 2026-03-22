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
    ranges = ntuple(_ -> 0:dimension-1, child_count)

    for parent_hyperindex in 0:dimension-1
        contribution = 0.0 + 0.0im

        for child_configuration in Iterators.product(ranges...)
            tensor_index = flattened_hyperindex_position(
                (parent_hyperindex, child_configuration...),
                dimension,
            )
            weight = tensor[tensor_index]

            for (message, child_hyperindex) in zip(child_messages, child_configuration)
                weight *= message[child_hyperindex+1]
            end

            contribution += weight
        end

        parent_message[parent_hyperindex+1] = contribution
    end

    parent_message
end
