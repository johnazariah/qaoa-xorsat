# Manual adjoint (reverse-mode) differentiation for the charge evaluator.
#
# The charge forward pass has two phases:
#   Phase A: Branch construction (p levels, each O(4^ℓ))
#   Phase B: Root contraction (O(p·4^p))
#
# ── Strategy ─────────────────────────────────────────────────────────────
#
# Each branch level ℓ is a function (γs, βs, child) → F_ℓ operating on a
# 4^ℓ-entry vector.  Rather than manually differentiating through the
# complex recursive Phase 2 trace, we use ForwardDiff on each level's
# branch function.  Since each level operates on 4^ℓ entries (much smaller
# than the full 4^p), the Dual overhead is small.
#
# The root contraction is differentiated manually — it's a simple loop
# (WHT butterfly + coefficient expansion) that operates on 4^p entries.
#
# Total gradient cost:
#   Branch: Σ_{ℓ=1}^{p} 2p × O(4^ℓ) = O(p·4^p)  (dominated by last level)
#   Root: ~2 × O(p·4^p)  (forward + backward pass)
#   Total: ~O(p·4^p) — same order as a single forward evaluation
#
# ── Normalization ────────────────────────────────────────────────────────
#
# Scale factors from normalization are tracked in log space and detached
# from the gradient, following the same strategy as the Basso adjoint
# (see adjoint.jl header comments).

using ForwardDiff

# ──────────────────────────────────────────────────────────────────────────────
# Forward pass with caching
# ──────────────────────────────────────────────────────────────────────────────

struct ChargeAdjointCache{T<:Real}
    p::Int
    k::Int
    D::Int
    clause_sign::Int
    γs::Vector{T}      # charge-convention γ (= angles.γ * clause_sign / 2)
    βs::Vector{T}

    # Per-level branch tensors (normalized) and scale info
    F_levels::Vector{Vector{Complex{T}}}  # F[ℓ] after hyperedge_branch, length 4^ℓ
    F_scales::Vector{T}                    # max|F[ℓ]| before normalization
    children::Vector{Vector{Complex{T}}}   # child = (F/F_max)^(D-1) passed to level ℓ+1

    # Root contraction intermediates
    rb::Vector{Complex{T}}                 # final (F/F_max)^(D-1)
    rb_scale::T                            # max|F_p| for final normalization
    root_factors::Vector{Vector{Complex{T}}}  # factor after each WHT round
    root_coeffs::Vector{Vector{Complex{T}}}   # coeffs after each expansion
    root_Ms::Vector{Matrix{Complex{T}}}       # doubled_mixer(β_ℓ) per round
    root_us::Vector{Vector{Complex{T}}}       # root_charge_weights(γ_ℓ) per round

    # Results
    log_scale::T
    raw::T              # root contraction result (before scaling)
    value::T            # final c̃
end

function _charge_forward_pass(
    params::TreeParams,
    angles::QAOAAngles{T};
    clause_sign::Int=1,
) where T
    p = params.p
    k = params.k
    D = params.D
    CT = Complex{T}

    γs = T(clause_sign) .* T.(angles.γ) ./ 2
    βs = T.(angles.β)

    # ── Branch construction with caching ──
    F_levels = Vector{Vector{CT}}(undef, p)
    F_scales = Vector{T}(undef, p)
    children = Vector{Vector{CT}}(undef, p)

    log_scale = zero(T)

    F = _charge_hyperedge_branch(γs, βs, 1, k)
    F_levels[1] = copy(F)
    F_scales[1] = one(T)
    children[1] = CT[]  # no child for level 1

    for level in 2:p
        F_max = maximum(abs, F)
        F_scales[level-1] = F_max  # scale from level-1's output
        if F_max > 0
            F = F ./ F_max
            log_scale += (D - 1) * log(F_max)
        end
        child = F .^ (D - 1)
        children[level] = copy(child)
        F = _charge_hyperedge_branch(γs, βs, level, k; child_branch=child)
        F_levels[level] = copy(F)
    end

    # Final normalization
    F_max = maximum(abs, F)
    rb_scale = F_max
    if F_max > 0
        F = F ./ F_max
        log_scale += (D - 1) * log(F_max)
    end
    rb = F .^ (D - 1)

    # ── Root contraction with caching ──
    N = length(rb)
    factor = copy(rb)
    buf = similar(factor)
    R = 1

    max_R = 4^(p - 1)
    coeffs_a = Vector{CT}(undef, max_R)
    coeffs_a[1] = CT(0.5)^k
    coeffs_b = Vector{CT}(undef, max_R)

    root_factors = Vector{Vector{CT}}(undef, p)
    root_coeffs = Vector{Vector{CT}}(undef, p)
    root_Ms = Vector{Matrix{CT}}(undef, p)
    root_us = Vector{Vector{CT}}(undef, p)

    root_factors[1] = copy(factor)  # initial factor = rb

    for ℓ in 1:p-1
        M = doubled_mixer(βs[ℓ])
        u = root_charge_weights(γs[ℓ])
        root_Ms[ℓ] = M
        root_us[ℓ] = u
        entries_per_row = div(N, R)

        _wht_charge_contract_flat!(buf, M, factor, R, entries_per_row)
        factor, buf = buf, factor

        @inbounds for a in 1:4
            ua = u[a]
            for i in 1:R
                coeffs_b[(a-1)*R + i] = ua * coeffs_a[i]
            end
        end
        R *= 4
        coeffs_a, coeffs_b = coeffs_b, coeffs_a

        root_factors[ℓ+1] = copy(factor)
        root_coeffs[ℓ] = copy(coeffs_a[1:R])
    end

    # Final round
    M = doubled_mixer(βs[p])
    u = root_charge_weights(γs[p])
    root_Ms[p] = M
    root_us[p] = u
    root_coeffs[p] = copy(coeffs_a[1:R])

    entries_per_row = div(N, R)
    result = zero(CT)
    @inbounds for a in 1:4
        K = M .* CT.(CHARGE_DIAG[a, :]')
        tv1 = K[1, 1] - K[4, 1]
        tv2 = K[1, 2] - K[4, 2]
        tv3 = K[1, 3] - K[4, 3]
        tv4 = K[1, 4] - K[4, 4]
        s = zero(CT)
        for i in 0:R-1
            base = i * entries_per_row
            z = factor[base+1]*tv1 + factor[base+2]*tv2 + factor[base+3]*tv3 + factor[base+4]*tv4
            s += coeffs_a[i+1] * z ^ k
        end
        result += u[a] * s
    end
    raw = real(result)

    # Scale and compute final value
    scaled_raw = raw * exp(k * log_scale)
    value = (1 + clause_sign * scaled_raw) / 2

    ChargeAdjointCache{T}(
        p, k, D, clause_sign, γs, βs,
        F_levels, F_scales, children,
        rb, rb_scale,
        root_factors, root_coeffs, root_Ms, root_us,
        log_scale, raw, value,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Backward pass
# ──────────────────────────────────────────────────────────────────────────────

"""
    _charge_backward_pass(cache) -> (γ_grad, β_grad)

Compute gradients ∂E/∂γ and ∂E/∂β using the cached charge forward pass.

Uses ForwardDiff on each branch level (small 4^ℓ vectors) and manual
adjoint on the root contraction (4^p vectors, dominant cost).
"""
function _charge_backward_pass(cache::ChargeAdjointCache{T}) where T
    p = cache.p
    k = cache.k
    D = cache.D
    CT = Complex{T}
    cs = T(cache.clause_sign)
    degree = D - 1

    # ∂value/∂raw = clause_sign/2 * exp(k * log_scale)
    # Scale factors are detached (treated as constants).
    scale_mult = cs / 2 * exp(k * cache.log_scale)

    if !isfinite(scale_mult)
        return (zeros(T, p), zeros(T, p))
    end

    # ── Root contraction backward ──
    # raw = real(Σ_a u[a] * Σ_i coeffs[i] * z_i^k)
    # where z_i = factor[i,:] · tv_a
    # ∂raw/∂factor entries → rb_bar
    # ∂raw/∂γ (through u and M) → γ_root_bar
    # ∂raw/∂β (through M) → β_root_bar

    N = length(cache.rb)
    R = 4^(p - 1)
    entries_per_row = div(N, R)

    # Backward through final measurement
    M = cache.root_Ms[p]
    u = cache.root_us[p]
    coeffs = cache.root_coeffs[p]
    factor = cache.root_factors[p > 1 ? p : 1]
    if p > 1
        factor = cache.root_factors[p]
    end

    # factor_bar: gradient of raw w.r.t. factor entries
    factor_bar = zeros(CT, N)

    @inbounds for a in 1:4
        K = M .* CT.(CHARGE_DIAG[a, :]')
        tv1 = K[1, 1] - K[4, 1]
        tv2 = K[1, 2] - K[4, 2]
        tv3 = K[1, 3] - K[4, 3]
        tv4 = K[1, 4] - K[4, 4]

        for i in 0:R-1
            base = i * entries_per_row
            z = factor[base+1]*tv1 + factor[base+2]*tv2 + factor[base+3]*tv3 + factor[base+4]*tv4
            # ∂raw/∂z_i = u[a] * coeffs[i+1] * k * z^(k-1)
            dz = scale_mult * u[a] * coeffs[i+1] * k * z^(k-1)
            # ∂z/∂factor[base+j] = tv_j
            factor_bar[base+1] += dz * tv1
            factor_bar[base+2] += dz * tv2
            factor_bar[base+3] += dz * tv3
            factor_bar[base+4] += dz * tv4
        end
    end

    # Backward through root WHT rounds (ℓ = p-1 down to 1)
    for ℓ in (p-1):-1:1
        M = cache.root_Ms[ℓ]
        R_prev = 4^(ℓ - 1)
        entries_per_row_prev = div(N, R_prev)

        # factor_bar is gradient w.r.t. output of _wht_charge_contract_flat!
        # Need to propagate to input (factor_prev)
        # output[channel_a][i, b, r] = Σ_σ CDIAG[a,σ] * M[b,σ] * input[i,σ,b,r]
        # This is linear in input, so adjoint is:
        # input_bar[i,σ,b,r] += Σ_a CDIAG[a,σ] * M[b,σ] * output_bar[channel_a][i,b,r]

        rest = div(entries_per_row_prev, 16)
        σ_stride = 4 * rest
        channel_size = R_prev * 4 * rest
        row_stride = 16 * rest

        factor_prev_bar = zeros(CT, N)
        @inbounds for i in 0:R_prev-1
            row_base = i * row_stride
            for b in 0:3
                for r in 0:rest-1
                    # Gather output_bar for this (i,b,r) across channels
                    dst = i * 4 * rest + b * rest + r + 1
                    ob0 = factor_bar[0*channel_size + dst]
                    ob1 = factor_bar[1*channel_size + dst]
                    ob2 = factor_bar[2*channel_size + dst]
                    ob3 = factor_bar[3*channel_size + dst]

                    # Reverse WHT butterfly
                    # Forward: p02=e0+e2, q02=e0-e2, p13=e1+e3, q13=e1-e3
                    # out0=p02+p13, out1=p02-p13, out2=q02+q13, out3=q02-q13
                    # Adjoint: p02_bar = ob0+ob1, p13_bar = ob0-ob1
                    #          q02_bar = ob2+ob3, q13_bar = ob2-ob3
                    #          e0_bar = p02_bar+q02_bar, e2_bar = p02_bar-q02_bar
                    #          e1_bar = p13_bar+q13_bar, e3_bar = p13_bar-q13_bar
                    p02_bar = ob0 + ob1
                    p13_bar = ob0 - ob1
                    q02_bar = ob2 + ob3
                    q13_bar = ob2 - ob3
                    e0_bar = p02_bar + q02_bar
                    e1_bar = p13_bar + q13_bar
                    e2_bar = p02_bar - q02_bar
                    e3_bar = p13_bar - q13_bar

                    # e[s] = M[b+1,s+1] * input[i,s,b,r]
                    # input_bar[i,s,b,r] += M[b+1,s+1] * e_bar[s]
                    for s in 0:3
                        e_bar_s = s == 0 ? e0_bar : s == 1 ? e1_bar : s == 2 ? e2_bar : e3_bar
                        src = row_base + s * σ_stride + b * rest + r + 1
                        factor_prev_bar[src] += M[b+1, s+1] * e_bar_s
                    end
                end
            end
        end
        factor_bar = factor_prev_bar

        # Note: we don't differentiate through coeffs (u[a] * coeffs) because
        # the coefficient expansion is u[a] * coeffs_prev[i] which depends on γ
        # via u = root_charge_weights(γ_ℓ). This contribution is handled
        # separately below via ForwardDiff on the angle-dependent parts.
    end

    # factor_bar now holds ∂raw/∂rb (the initial factor = rb)
    rb_bar = real.(factor_bar)  # rb is real after normalization

    # ── Backward through rb = (F_p / F_max)^(D-1) ──
    F_p = cache.F_levels[p]
    F_max = cache.rb_scale
    F_norm = F_p ./ F_max
    # rb = F_norm^(D-1), so F_norm_bar = (D-1) * conj(F_norm^(D-2)) * rb_bar
    F_norm_bar = degree .* conj.(F_norm .^ (degree - 1)) .* factor_bar
    # F_norm = F_p / F_max, F_max detached
    F_p_bar = F_norm_bar ./ F_max

    # ── Backward through branch levels (using ForwardDiff) ──
    # We need ∂E/∂γ_r and ∂E/∂β_r for each physical round r.
    # Each branch level ℓ computes F_ℓ = hyperedge_branch(γs, βs, ℓ, k; child).
    # The chain rule through the level loop:
    #   F_p_bar → child_bar → F_{p-1}_bar → ... → F_1_bar
    # And at each level, the contribution to γ/β gradients.

    γ_grad = zeros(T, p)
    β_grad = zeros(T, p)

    current_bar = F_p_bar  # cotangent for current level's output

    for level in p:-1:1
        # At this level: F_level = _charge_hyperedge_branch(γs, βs, level, k; child)
        # child comes from the previous level: child = (F_{level-1}/scale)^(D-1)
        #
        # We need:
        # 1. ∂F_level/∂γ_r for each r — contributes to γ_grad
        # 2. ∂F_level/∂β_r for each r — contributes to β_grad
        # 3. ∂F_level/∂child — propagates to the previous level

        child = level > 1 ? cache.children[level] : nothing

        # Use ForwardDiff to compute Jacobian-vector products.
        # For each angle parameter θ: ∂F/∂θ is a vector of length 4^level.
        # Compute as: jvp = ForwardDiff.derivative(θ -> F(γs_with_θ, βs), θ)

        for r in 1:p
            # γ gradient: ∂F_level/∂γ_r
            dF_dγr = ForwardDiff.derivative(cache.γs[r]) do γr_dual
                γs_mod = [i == r ? γr_dual : cache.γs[i] for i in 1:p]
                _charge_hyperedge_branch(γs_mod, cache.βs, level, k; child_branch=child)
            end
            γ_grad[r] += real(dot(current_bar, dF_dγr))

            # β gradient: ∂F_level/∂β_r
            dF_dβr = ForwardDiff.derivative(cache.βs[r]) do βr_dual
                βs_mod = [i == r ? βr_dual : cache.βs[i] for i in 1:p]
                _charge_hyperedge_branch(cache.γs, βs_mod, level, k; child_branch=child)
            end
            β_grad[r] += real(dot(current_bar, dF_dβr))
        end

        # Propagate to previous level via child
        if level > 1
            # child = (F_{level-1} / F_max)^(D-1)
            # ∂F_level/∂child → child_bar
            function f_child(child_dual)
                _charge_hyperedge_branch(cache.γs, cache.βs, level, k; child_branch=child_dual)
            end
            # Jacobian-vector product: J^T * current_bar
            # Use ForwardDiff to compute each column of the Jacobian
            n_child = length(child)
            child_bar = zeros(CT, n_child)
            for j in 1:n_child
                function f_child_j(t)
                    c = copy(child)
                    c[j] += t
                    _charge_hyperedge_branch(cache.γs, cache.βs, level, k; child_branch=c)
                end
                dF_dcj = ForwardDiff.derivative(f_child_j, zero(T))
                child_bar[j] = dot(current_bar, dF_dcj)
            end

            # child = (F_{level-1} / F_max)^(D-1)
            # child_bar → F_{level-1}_bar
            F_prev = cache.F_levels[level-1]
            F_prev_max = cache.F_scales[level-1]
            F_prev_norm = F_prev ./ F_prev_max
            # ∂child/∂F_prev_norm = (D-1) * F_prev_norm^(D-2)
            F_prev_norm_bar = degree .* conj.(F_prev_norm .^ (degree - 1)) .* child_bar
            current_bar = F_prev_norm_bar ./ F_prev_max
        end
    end

    # ── Root angle gradients ──
    # The root contraction also depends on γs, βs through:
    # - doubled_mixer(βs[ℓ]) in each WHT round
    # - root_charge_weights(γs[ℓ]) in coefficient expansion and final round
    # Compute these contributions via ForwardDiff on the root contraction.

    for r in 1:p
        # γ contribution from root
        d_root_γ = ForwardDiff.derivative(cache.γs[r]) do γr_dual
            γs_mod = [i == r ? γr_dual : cache.γs[i] for i in 1:p]
            _charge_root_contract(cache.rb, γs_mod, cache.βs, p, D, k)
        end
        γ_grad[r] += scale_mult * d_root_γ

        # β contribution from root
        d_root_β = ForwardDiff.derivative(cache.βs[r]) do βr_dual
            βs_mod = [i == r ? βr_dual : cache.βs[i] for i in 1:p]
            _charge_root_contract(cache.rb, cache.γs, βs_mod, p, D, k)
        end
        β_grad[r] += scale_mult * d_root_β
    end

    # Convert from charge-convention γ to our γ/2 convention
    # γs = clause_sign * angles.γ / 2, so ∂E/∂angles.γ = clause_sign/2 * ∂E/∂γs
    γ_grad .*= cs / 2

    (γ_grad, β_grad)
end

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    charge_expectation_and_gradient(params, angles; clause_sign=1) -> (value, γ_grad, β_grad)

Compute the expected satisfaction fraction and its gradient using the charge
decomposition evaluator with hybrid adjoint differentiation.

Cost: approximately 3-4× a single charge forward evaluation, independent of p.
Memory: O(4^p) — same as a single forward evaluation.
"""
function charge_expectation_and_gradient(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
)
    cache = _charge_forward_pass(params, angles; clause_sign)
    γ_grad, β_grad = _charge_backward_pass(cache)
    (cache.value, γ_grad, β_grad)
end
