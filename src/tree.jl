"""
    TreeParams(k, D, p)

Parameters defining a regular hypergraph QAOA light-cone tree.

- `k`: constraint arity (hyperedge size). k=2 is MaxCut.
- `D`: variable degree (regularity).
- `p`: physical QAOA depth (number of rounds).

At physical depth `p`, the light cone contains the root clause plus `p` additional
constraint shells, and therefore `p + 1` variable shells. Under the compressed
shell indexing used here:

- constraint shells are indexed by `j = 0, …, p`, with `j = 0` the root clause
- variable shells are indexed by `j = 0, …, p`, with `j = p` the boundary leaves

The branching factor per two-shell step is `(D-1)(k-1)`.
"""
struct TreeParams
    k::Int
    D::Int
    p::Int

    function TreeParams(k::Int, D::Int, p::Int)
        k ≥ 2 || throw(ArgumentError("k must be ≥ 2, got $k"))
        D ≥ 2 || throw(ArgumentError("D must be ≥ 2, got $D"))
        p ≥ 1 || throw(ArgumentError("p must be ≥ 1, got $p"))
        new(k, D, p)
    end
end

"""
    branching_factor(t::TreeParams) -> Int

Branching factor per two-level step: `(D-1)(k-1)`.
"""
branching_factor(t::TreeParams) = (t.D - 1) * (t.k - 1)

"""
    variable_count_at_level(t::TreeParams, j::Int) -> Int

Number of variable nodes at variable shell `j` (0-indexed, `j = 0, …, p`).

Shell `j = 0` has the `k` root variables.
Shell `j ≥ 1` has `k · b^j` variables, where `b = branching_factor(t)`.
"""
function variable_count_at_level(t::TreeParams, j::Int)
    0 ≤ j ≤ t.p || throw(ArgumentError("j must be in 0:$(t.p), got $j"))
    t.k * branching_factor(t)^j
end

"""
    constraint_count_at_level(t::TreeParams, j::Int) -> Int

Number of constraint nodes at constraint shell `j` (0-indexed, `j = 0, …, p`).

Shell `j = 0` has the root constraint.
Shell `j ≥ 1` has `k(D-1) · b^(j-1)`, where `b = branching_factor(t)`.
"""
function constraint_count_at_level(t::TreeParams, j::Int)
    0 ≤ j ≤ t.p || throw(ArgumentError("j must be in 0:$(t.p), got $j"))
    j == 0 ? 1 : t.k * (t.D - 1) * branching_factor(t)^(j - 1)
end

"""
    total_variables(t::TreeParams) -> Int

Total number of variable (qubit) nodes across all `p + 1` variable shells.
"""
total_variables(t::TreeParams) =
    sum(variable_count_at_level(t, j) for j in 0:t.p)

"""
    total_constraints(t::TreeParams) -> Int

Total number of constraint nodes across all `p + 1` constraint shells.
"""
total_constraints(t::TreeParams) =
    sum(constraint_count_at_level(t, j) for j in 0:t.p)

"""
    total_nodes(t::TreeParams) -> Int

Total nodes (variables + constraints) in the tree.
"""
total_nodes(t::TreeParams) = total_variables(t) + total_constraints(t)

"""
    leaf_count(t::TreeParams) -> Int

Number of boundary leaf nodes (the variable shell `j = p`).
These are the boundary qubits initialised to |+⟩.
"""
leaf_count(t::TreeParams) = variable_count_at_level(t, t.p)
