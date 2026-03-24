"""
    CostAlgebra{K}

Abstract type defining a k-local cost function for QAOA evaluation on D-regular
k-uniform hypergraphs. The type parameter K is the constraint arity.

A CostAlgebra parametrises the fold engine through two pluggable components:
- `constraint_kernel`: the problem gate's contribution at each constraint node
- `root_observable`: how to extract the satisfaction fraction at the root

Everything else — tree structure, leaf tensor, mixer, WHT acceleration, angle
optimisation — is problem-agnostic.

See `.project/specs/generic-fold-engine.md` for the full specification.
"""
abstract type CostAlgebra{K} end

"""Constraint arity of the algebra."""
arity(::CostAlgebra{K}) where {K} = K

"""
    XORSATAlgebra{K} <: CostAlgebra{K}

Max-k-XORSAT on D-regular k-uniform hypergraphs.

Cost operator per constraint: C_α = (1 + clause_sign · Z_{i₁}···Z_{iₖ}) / 2

For k=2 with clause_sign=-1 this gives MaxCut (cut fraction = (1 - Z_iZ_j) / 2).
"""
struct XORSATAlgebra{K} <: CostAlgebra{K}
    clause_sign::Int

    function XORSATAlgebra{K}(clause_sign::Int) where {K}
        K ≥ 2 || throw(ArgumentError("constraint arity must be ≥ 2, got $K"))
        clause_sign ∈ (-1, 1) || throw(ArgumentError("clause_sign must be ±1, got $clause_sign"))
        new{K}(clause_sign)
    end
end

XORSATAlgebra(k::Int; clause_sign::Int=1) = XORSATAlgebra{k}(clause_sign)

"""Default clause sign for the algebra."""
default_clause_sign(a::XORSATAlgebra) = a.clause_sign

"""
    MaxCutAlgebra()

Convenience constructor: MaxCut is Max-2-XORSAT with clause_sign = -1.
"""
MaxCutAlgebra() = XORSATAlgebra(2; clause_sign=-1)

"""
    algebra_from_clause_sign(k, clause_sign) -> CostAlgebra

Construct the appropriate CostAlgebra from the legacy (k, clause_sign) convention.
"""
algebra_from_clause_sign(k::Int, clause_sign::Int) = XORSATAlgebra(k; clause_sign)

# ── Dispatched kernel methods ────────────────────────────────────────────────

"""
    constraint_kernel(algebra, angles, branch_degree) -> Vector{ComplexF64}

Build the constraint kernel for the fold. The kernel encodes the problem gate's
contribution at each non-root constraint node.

For XORSAT: κ(a) = cos(Γ · spins(a) / 2)
"""
function constraint_kernel(::XORSATAlgebra, angles::QAOAAngles, branch_degree::Int)
    basso_constraint_kernel(angles, branch_degree)
end

"""
    root_observable_kernel(algebra, angles, branch_degree) -> Vector{ComplexF64}

Build the root observable kernel. This encodes both the root problem gate and
the observable (Z₁···Zₖ expectation) used to extract the satisfaction fraction.

For XORSAT with clause_sign s: κ_root(a) = i·sin(s · Γ · spins(a) / 2)
"""
function root_observable_kernel(algebra::XORSATAlgebra, angles::QAOAAngles, branch_degree::Int)
    basso_root_problem_kernel(angles, branch_degree; clause_sign=algebra.clause_sign)
end

"""
    expectation_from_parity(algebra, parity) -> Float64

Convert a root parity correlator ⟨Z₁···Zₖ⟩ into the satisfaction fraction
(1 + clause_sign · parity) / 2.
"""
function expectation_from_parity(algebra::XORSATAlgebra, parity::Real)
    (1 + algebra.clause_sign * parity) / 2
end
