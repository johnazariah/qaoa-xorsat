module QaoaXorsat

# Runtime diagnostics (controlled by QAOA_DIAG env vars)
include("diagnostics.jl")
using .Diagnostics

# Optional CUDA, AMDGPU, and Metal execution backends
include("gpu_backend.jl")
export GPUBackend, GPUBackendError
export gpu_backend, gpu_backend_available, validate_gpu_backend, detect_gpu_backend
export gpu_array, make_gpu_evaluator

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
export constraint_kernel, root_observable_kernel, expectation_from_parity

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
export AngleOptimizationResult, AngleSnapshot, PlateauPolicy, OptimizationPolicy
export OptimizationRunSpec, OptimizationCallbacks, OptimizationEvent
export EvaluationEvent, PlateauEvent, AngleSnapshotEvent, DepthResultEvent
export run_optimization
export AngleRecord, CsvResultStore, PreviousDepthWarmStart
export canonicalize_angles, random_angles, extend_angles
export format_angles, parse_angles, snapshot_csv_header, snapshot_csv_row, write_angle_snapshot!
export write_latest_angle_snapshot!, read_angle_snapshots, read_best_angle_snapshot
export read_records, read_best_record, append_record!, resolve_warm_start
export optimize_angles, optimize_depth_sequence, swarm_optimize, optimize_angles_chebyshev

# QAOA evaluation
include("qaoa.jl")
export parity_expectation, qaoa_expectation

# Exact finite-N MaxCut statevector evaluation
include("maxcut_statevector.jl")
export MaxCutStatevectorEval, prepare_maxcut_eval
export maxcut_state, maxcut_expectation, maxcut_success_probability
export maxcut_expectation_and_gradient, maxcut_expectation_and_gradient!
export maxcut_value_max

# Heisenberg (XYZ) cost — Stage 0 statevector oracle on the 1D chain
include("heisenberg_statevector.jl")
export HeisenbergCouplings
export chain_edges, central_edge, exact_chain_length
export heisenberg_chain_state, heisenberg_edge_correlators, heisenberg_energy_density

# Heisenberg (XYZ) cost — Stage 0.5 matrix-free Schrödinger evolver (RK4/Euler)
include("sparse_qaoa.jl")
export PauliTerm
export x_term, y_term, z_term
export xx_term, xy_term, xz_term, yx_term, yy_term, yz_term, zx_term, zy_term, zz_term
export heisenberg_terms, x_mixer_terms, xy_mixer_terms, swap_mixer_terms
export apply_terms!, hamiltonian_expectation
export evolve_rk4!, evolve_euler!
export sparse_qaoa_state, sparse_qaoa_energy

# Manual adjoint differentiation
include("adjoint.jl")
export basso_expectation_and_gradient, basso_expectation_normalized

# CPU gradient checkpointing (√p memory for p≥13)
include("checkpointed_adjoint.jl")
export basso_expectation_and_gradient_checkpointed, basso_expectation_checkpointed

# 4× reduced branch-tensor iteration
include("reduced_basis.jl")
export ReducedBasis, basso_branch_tensor_reduced, basso_expectation_reduced
export expand_symmetric

# Charge decomposition evaluator — O(p·4^p)
include("charge.jl")
export charge_parity_expectation, charge_expectation

# Manual charge adjoint — exact gradients in ~3x forward cost
include("charge_manual_adjoint.jl")
export charge_expectation_and_gradient

# Spectral analysis of branch tensor iteration
include("spectral_analysis.jl")
export SpectralSnapshot, SpectralProfile
export spectral_snapshot, spectral_decay_rate
export basso_branch_tensor_instrumented
export format_spectral_report, write_spectral_csv, write_effective_ranks_csv

# Comparison data
# export load_comparison_data

end # module
