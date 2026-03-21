"""
    TreeParams(k, D, p)

Parameters defining a regular hypergraph QAOA light-cone tree.

- `k`: constraint arity (hyperedge size). k=2 is MaxCut.
- `D`: variable degree (regularity).
- `p`: QAOA depth (number of rounds).

The tree alternates constraint→variable levels for 2p total levels (0 through 2p-1).
The branching factor per two-level step is `(D-1)(k-1)`.
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

Number of variable nodes at variable level `j` (0-indexed, j = 0, …, p-1).

Level j=0 has k variables (children of the root constraint).
Level j≥1 has k · b^j variables, where b = branching_factor(t).
"""
function variable_count_at_level(t::TreeParams, j::Int)
    0 ≤ j < t.p || throw(ArgumentError("j must be in 0:$(t.p-1), got $j"))
    j == 0 ? t.k : t.k * branching_factor(t)^j
end

"""
    constraint_count_at_level(t::TreeParams, j::Int) -> Int

Number of constraint nodes at constraint level `j` (0-indexed, j=0 is the root).

Level j=0 has 1 (the root constraint).
Level j≥1 has k(D-1) · b^(j-1), where b = branching_factor(t).
"""
function constraint_count_at_level(t::TreeParams, j::Int)
    0 ≤ j < t.p || throw(ArgumentError("j must be in 0:$(t.p-1), got $j"))
    j == 0 ? 1 : t.k * (t.D - 1) * branching_factor(t)^(j - 1)
end

"""
    total_variables(t::TreeParams) -> Int

Total number of variable (qubit) nodes in the tree across all p variable levels.
"""
total_variables(t::TreeParams) =
    sum(variable_count_at_level(t, j) for j in 0:t.p-1)

"""
    total_constraints(t::TreeParams) -> Int

Total number of constraint nodes in the tree across all p constraint levels.
"""
total_constraints(t::TreeParams) =
    sum(constraint_count_at_level(t, j) for j in 0:t.p-1)

"""
    total_nodes(t::TreeParams) -> Int

Total nodes (variables + constraints) in the tree.
"""
total_nodes(t::TreeParams) = total_variables(t) + total_constraints(t)

"""
    leaf_count(t::TreeParams) -> Int

Number of leaf nodes (variable nodes at the deepest level, p-1).
These are the boundary qubits initialised to |+⟩.
"""
leaf_count(t::TreeParams) = variable_count_at_level(t, t.p - 1)
