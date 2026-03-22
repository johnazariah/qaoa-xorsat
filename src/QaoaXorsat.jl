module QaoaXorsat

# Tree structure
include("tree.jl")
export TreeParams
export branching_factor, variable_count_at_level, constraint_count_at_level
export total_variables, total_constraints, total_nodes, leaf_count

# Tensor network
include("tensors.jl")
export QAOAAngles, depth
export hyperindex_dimension
export slice_from_physical_round, physical_round_from_slice
export slice_bit_positions, round_bit_positions
export hyperindex_bit, hyperindex_parity
export leaf_tensor, mixer_tensor, problem_tensor
export parity_observable_tensor, observable_tensor

# Experimental MaxCut transfer recursion
include("maxcut_transfer.jl")

# QAOA evaluation
include("qaoa.jl")
export parity_expectation, qaoa_expectation

# Comparison data
# export load_comparison_data

end # module
