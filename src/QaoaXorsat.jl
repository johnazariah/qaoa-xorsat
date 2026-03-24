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

# Cost algebra — pluggable problem definition
include("cost_algebra.jl")
export CostAlgebra, XORSATAlgebra, MaxCutAlgebra
export arity, default_clause_sign, algebra_from_clause_sign

# Raw transfer oracles
include("transfer_oracles.jl")

# XOR-convolution / Walsh-Hadamard utilities
include("wht.jl")

# Tier 2 Basso finite-D helpers
include("basso_finite_d.jl")
export basso_parity_expectation, basso_expectation

# Experimental MaxCut transfer recursion
include("maxcut_transfer.jl")

# Optimisation helpers
include("optimization.jl")
export AngleOptimizationResult
export canonicalize_angles, random_angles, extend_angles
export optimize_angles, optimize_depth_sequence

# QAOA evaluation
include("qaoa.jl")
export parity_expectation, qaoa_expectation

# Manual adjoint differentiation
include("adjoint.jl")
export basso_expectation_and_gradient

# Comparison data
# export load_comparison_data

end # module
