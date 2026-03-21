module QaoaXorsat

# Tree structure
include("tree.jl")
export TreeParams
export branching_factor, variable_count_at_level, constraint_count_at_level
export total_variables, total_constraints, total_nodes, leaf_count

# Tensor network
include("tensors.jl")
export QAOAAngles, depth
export hyperindex_dimension, round_bit_positions
export hyperindex_bit, hyperindex_parity
export leaf_tensor, mixer_tensor, problem_tensor, observable_tensor

# QAOA evaluation
# export qaoa_expectation, optimize_angles

# Comparison data
# export load_comparison_data

end # module
