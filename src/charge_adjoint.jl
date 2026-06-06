# Charge adjoint — simple central FD on charge_expectation.
#
# Computes the gradient of charge_expectation by perturbing each angle
# independently and re-evaluating.  No Dual numbers, no child Jacobians,
# no type promotion issues.
#
# Cost: (4p + 1) × charge_expectation evaluations
# Memory: O(4^p) — same as a single evaluation
# Accuracy: ~1e-8 (central FD with h = √ε)

const _FD_STEP = 1.5e-8

"""
    charge_expectation_and_gradient(params, angles; clause_sign=1) -> (value, γ_grad, β_grad)

Compute the expected satisfaction fraction and its gradient using the charge
decomposition evaluator with central finite differences.

Cost: `(4p + 1)` charge evaluations.  Memory: `O(4^p)`.
"""
function charge_expectation_and_gradient(
    params::TreeParams,
    angles::QAOAAngles{T};
    clause_sign::Int=1,
) where T
    p = depth(angles)
    h = T(_FD_STEP)

    value = charge_expectation(params, angles; clause_sign)

    γ_grad = Vector{T}(undef, p)
    β_grad = Vector{T}(undef, p)

    for r in 1:p
        # γ gradient
        γ_p = copy(angles.γ); γ_p[r] += h
        γ_m = copy(angles.γ); γ_m[r] -= h
        v_p = charge_expectation(params, QAOAAngles(γ_p, angles.β); clause_sign)
        v_m = charge_expectation(params, QAOAAngles(γ_m, angles.β); clause_sign)
        γ_grad[r] = (v_p - v_m) / (2h)

        # β gradient
        β_p = copy(angles.β); β_p[r] += h
        β_m = copy(angles.β); β_m[r] -= h
        v_p = charge_expectation(params, QAOAAngles(angles.γ, β_p); clause_sign)
        v_m = charge_expectation(params, QAOAAngles(angles.γ, β_m); clause_sign)
        β_grad[r] = (v_p - v_m) / (2h)
    end

    (value, γ_grad, β_grad)
end
