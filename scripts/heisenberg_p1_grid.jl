#!/usr/bin/env julia
#
# Diagnostic for the Stage 0 smoothness sweep: is ε = -1 (the trivial γ=0 value,
# where the state stays |+⟩^L and ε = Jx⟨XX⟩ = Jx) the true p=1 optimum for the
# isotropic point, or is the multi-start optimiser failing to escape it?
#
# Does a dense grid search over (γ, β) at p=1 for each coupling and reports the
# minimum ε and its argmin, alongside the trivial baseline. Cheap (L=8).
#
# Usage: julia --project=. scripts/heisenberg_p1_grid.jl [NGRID]
# Output: stderr report only.

using QaoaXorsat
using Printf

ngrid = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 121

couplings = [
    ("iso-FM", HeisenbergCouplings(-1.0, -1.0, -1.0)),
    ("iso-AFM", HeisenbergCouplings(1.0, 1.0, 1.0)),
    ("xxz-AFM", HeisenbergCouplings(1.0, 1.0, 0.5)),
    ("ising-AFM", HeisenbergCouplings(0.0, 0.0, 1.0)),
]

γs = range(0, 2π; length=ngrid)
βs = range(0, π; length=ngrid)

for (label, J) in couplings
    trivial = heisenberg_energy_density(QAOAAngles([0.0], [0.0]), J)  # γ=0 baseline
    best_ε = Inf
    best_γ = 0.0
    best_β = 0.0
    for γ in γs, β in βs
        ε = heisenberg_energy_density(QAOAAngles([γ], [β]), J)
        if ε < best_ε
            best_ε = ε
            best_γ = γ
            best_β = β
        end
    end
    @printf(stderr, "%-6s J=(%.2f,%.2f,%.2f)  trivial(γ=0) ε=%.6f | grid-min ε=%.6f at γ=%.4f β=%.4f | improvement=%.6f\n",
        label, J.Jx, J.Jy, J.Jz, trivial, best_ε, best_γ, best_β, trivial - best_ε)
end
