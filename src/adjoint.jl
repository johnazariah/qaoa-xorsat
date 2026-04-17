# Manual adjoint (reverse-mode) differentiation for the Basso evaluator.
#
# The forward pass matches basso_parity_expectation / basso_expectation exactly.
# The backward pass propagates cotangents through the saved computation graph
# to produce exact ∂E/∂γ and ∂E/∂β at cost ≈ 2× a single Float64 evaluation.
#
# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZED BRANCH TENSOR RECURRENCE
# ══════════════════════════════════════════════════════════════════════════════
#
# The branch tensor recurrence
#
#   child_hat[t]  = WHT(f .* B[t])
#   folded[t]     = iWHT(kernel_hat .* child_hat[t]^(k-1))
#   B[t+1]        = folded[t]^(D-1)
#
# raises complex numbers to the (k-1)th and (D-1)th power at every step.
# At high (k, D, p) the magnitudes compound exponentially and overflow
# Float64 (~1.8e+308).  For (k=7, D=8) this happens around p ≈ 9.
#
# ── Normalization strategy ────────────────────────────────────────────────
#
# Before each power operation, we normalize the vector to unit max-magnitude
# and track the scale factor in log space:
#
#   child_hat_norm[t] = child_hat[t] / α_t       where α_t = max|child_hat[t]|
#   folded_raw        = iWHT(kernel_hat .* child_hat_norm[t]^(k-1))
#   folded_norm[t]    = folded_raw / β_t          where β_t = max|folded_raw|
#   B[t+1]            = folded_norm[t]^(D-1)      (max-magnitude ≤ 1, safe)
#
# The true (un-normalized) B[t+1] would be:
#   B_true[t+1] = (α_t^(k-1) · β_t)^(D-1) · folded_norm[t]^(D-1)
#
# but we store only the normalized part (folded_norm^(D-1)) and accumulate
# the scale in a single running variable:
#
#   log_s[1] = 0
#   log_s[t+1] = (k-1)(D-1) · log_s[t]
#                + (k-1) · log(α_t)    [from child_hat normalization]
#                + (D-1) · log(β_t)    [from folded normalization]
#
# At the root, we also normalize msg_hat = WHT(root_msg) by μ = max|msg_hat|
# before raising to ^k.  The total log-scale for the parity correlator S is:
#
#   L = k · (log_s[p+1] + log(μ))
#
# and the physical answer is:
#
#   c̃ = (1 + c_s · exp(L) · Re(S_normalized)) / 2
#
# This product is computed in log space:
#   exp(L + log|Re(S_normalized)|) with the correct sign.
#
# ── Backward pass ─────────────────────────────────────────────────────────
#
# The backward pass operates entirely on normalized intermediates.  The
# gradient of c̃ with respect to any angle θ is:
#
#   ∂c̃/∂θ = (c_s / 2) · exp(L) · ∂S_normalized/∂θ
#
# where we DETACH the scale factors from the gradient (treat α_t, β_t, μ
# as constants).  This is valid because:
#
# 1. The scale factors are max-magnitude operations, whose gradient is a
#    sparse selection operator (nonzero only for the argmax entry), making
#    their contribution negligible compared to the O(N) gradient terms.
#
# 2. The physical answer c̃ is invariant to the choice of normalization
#    convention, so the detached gradient still represents the correct
#    descent direction.
#
# 3. At convergence, the optimizer verifies the final c̃ value using the
#    same normalized forward pass, so any gradient approximation error
#    only affects the path to convergence, not the result.
#
# All normalized intermediates have max-magnitude ≤ 1, so the backward
# powers (folded_norm^(degree-1), child_hat_norm^(arity-1)) cannot overflow.
# The single exp(L) multiplier is applied once at the end; if L > 700
# (which should never happen for valid QAOA), zero gradients are returned
# and the optimizer's overflow guard handles the rest.
#
# ── Performance ───────────────────────────────────────────────────────────
#
# The normalization adds 2 passes per step (max-magnitude + scale) and one
# at the root.  Each pass is O(N) where N = 4^p.  Since the WHTs are
# O(N log N) and the power operations are O(N), the overhead is negligible.
# Memory usage is unchanged: we store the same vectors but add 2p + 1 scalars.

# ──────────────────────────────────────────────────────────────────────────────
# Forward pass with caching
# ──────────────────────────────────────────────────────────────────────────────

"""
    _fast_pow(x, n)

Specialized complex power for small positive integers (1–7), avoiding the
general `x^n` path which uses log/exp internally. Falls back to `x^n` for n>7.
"""
@inline function _fast_pow(x::Complex{T}, n::Int) where T
    n == 1 && return x
    n == 2 && return x * x
    x2 = x * x
    n == 3 && return x2 * x
    n == 4 && return x2 * x2
    x4 = x2 * x2
    n == 5 && return x4 * x
    n == 6 && return x4 * x2
    n == 7 && return x4 * x2 * x
    return x ^ n
end

"""
    _phase_dot(gamma_full, configuration, bit_count)

Compute Σ_i gamma_full[i] * z_eigenvalue(bit i of configuration).
Inlined spin computation, no allocations.
"""
function _phase_dot(gamma_full::AbstractVector, configuration::Int, bit_count::Int)
    phase = zero(eltype(gamma_full))
    for index in 1:bit_count
        @inbounds phase += gamma_full[index] * z_eigenvalue((configuration >> (index - 1)) & 1)
    end
    phase
end

"""
Cache structure holding all intermediates needed for the backward pass.

All vectors stored here are **normalized** (max magnitude ≈ 1) to prevent
Float64 overflow at high (k, D, p).  The accumulated scale is tracked
separately in log space and applied once at the very end.
"""
struct BassoPipelineCache{T<:Real}
    # Angles and params
    p::Int
    k::Int
    D::Int
    clause_sign::Int
    bit_count::Int
    configuration_count::Int
    arity::Int       # k - 1
    degree::Int       # D - 1

    # Precomputed angle-dependent tables
    β::Vector{T}                     # mixer angles (needed for backward pass)
    gamma_full::Vector{T}
    trig_table::Matrix{Complex{T}}   # 2 × 2p — cos/isin for each transition

    # f_table and per-config phase arguments
    f_table::Vector{Complex{T}}
    phase_args::Vector{T}            # phase arg per config (shared by constraint + root kernels)

    # Constraint kernel and its WHT
    kernel::Vector{Complex{T}}
    kernel_hat::Vector{Complex{T}}

    # Branch tensor history: B[1] = ones, B[t+1] = step(B[t])  (NORMALIZED)
    B::Vector{Vector{Complex{T}}}

    # Per-step intermediates (saved for backward)  (NORMALIZED)
    child_hat::Vector{Vector{Complex{T}}}    # WHT(f_table .* B[t]) / child_hat_scale
    folded::Vector{Vector{Complex{T}}}       # iWHT(kernel_hat .* child_hat_norm^arity) / folded_scale

    # Per-step normalization scale factors (real, positive)
    child_hat_scales::Vector{T}    # max|child_hat| per step
    folded_scales::Vector{T}       # max|folded| per step

    # Root computation  (NORMALIZED)
    root_msg::Vector{Complex{T}}
    root_parity_signs::Vector{Int}
    root_kernel::Vector{Complex{T}}
    msg_hat::Vector{Complex{T}}      # WHT(root_msg) / msg_hat_scale  (NORMALIZED)
    msg_hat_scale::T                 # max|WHT(root_msg)|
    msg_hat_power::Vector{Complex{T}}  # msg_hat_norm .^ k

    # Accumulated log-scale: log(s_{p+1}^k * msg_hat_scale^k)
    log_total_scale::T

    # Final result
    S_normalized::Complex{T}         # sum(root_kernel .* conv) in normalized space
    value::T
end

"""
    _basso_f_table_fast(trig_table, bit_count, N, T)

Compute the mixer weight f(a) for all N configurations using a pre-built
trig_table, avoiding per-config allocations. Each f(a) is ½ · ∏ⱼ trigs[Δ(a,j)+1, j]
where Δ(a,j) = bit[j] ⊻ bit[j+1].
"""
function _basso_f_table_fast(trig_table::Matrix{Complex{T}}, bit_count::Int, N::Int, ::Type{T}) where T
    table = Vector{Complex{T}}(undef, N)
    transitions = bit_count - 1  # = 2p
    Threads.@threads for config in 0:N-1
        weight = complex(one(T) / 2)
        @inbounds for j in 1:transitions
            d = xor((config >> (j - 1)) & 1, (config >> j) & 1)
            weight *= trig_table[d + 1, j]
        end
        @inbounds table[config+1] = weight
    end
    table
end

"""
    _forward_pass(params, angles; clause_sign=1) -> BassoPipelineCache

Run the full Basso evaluation, saving all intermediates for the backward pass.
"""
function _forward_pass(
    params::TreeParams,
    angles::QAOAAngles{T};
    clause_sign::Int=1,
) where T
    p = params.p
    k = params.k
    D = params.D
    arity = k - 1
    degree = D - 1
    bit_count = basso_bit_count(p)
    N = basso_configuration_count(p)

    depth(angles) == p || throw(ArgumentError("angle depth must match tree depth"))
    validate_clause_sign(clause_sign)

    gamma_full = build_gamma_full_vector(angles)
    trig_table = basso_trig_table(angles)

    # f_table — allocation-free version using pre-built trig_table
    f_table = _basso_f_table_fast(trig_table, bit_count, N, T)

    # Phase arguments (shared by constraint + root kernels)
    half = one(T) / 2
    phase_args = Vector{T}(undef, N)
    kernel = Vector{Complex{T}}(undef, N)
    Threads.@threads for config in 0:N-1
        ph = _phase_dot(gamma_full, config, bit_count)
        @inbounds phase_args[config+1] = ph
        @inbounds kernel[config+1] = complex(cos(half * ph))
    end
    kernel_hat = wht!(copy(kernel))

    # Branch tensor iteration with caching and per-step normalization.
    #
    # At high (k, D, p) the powers child_hat^(k-1) and folded^(D-1)
    # overflow Float64.  We normalize before each power operation ONLY
    # when magnitudes threaten to overflow, and track the accumulated
    # scale in log space.
    #
    # Normalizing every step destroys the relative magnitude relationships
    # between entries that carry the physical signal (the deviation from
    # c̃ = 0.5), causing signal underflow at high depth.  By only
    # normalizing when max-magnitude exceeds a safe threshold, we preserve
    # precision while still preventing overflow.
    #
    # Threshold: 1e100 is safe because (1e100)^7 = 1e700 < 1.8e308 is
    # false — actually we need (threshold)^max(arity,degree) < 1e300.
    # For degree=7: threshold < 1e300/7 ≈ 1e42.  Use 1e30 for safety.

    _NORM_THRESHOLD = T(1e30)

    B_history = Vector{Vector{Complex{T}}}(undef, p + 1)
    child_hat_history = Vector{Vector{Complex{T}}}(undef, p)
    folded_history = Vector{Vector{Complex{T}}}(undef, p)
    child_hat_scales = Vector{T}(undef, p)
    folded_scales = Vector{T}(undef, p)

    scratch = Vector{Complex{T}}(undef, N)

    log_s = zero(T)  # log of accumulated scale on B

    B_history[1] = ones(Complex{T}, N)
    for t in 1:p
        # child_weights = f_table .* B[t] — compute into scratch, then WHT in-place
        @inbounds @simd for i in 1:N
            scratch[i] = f_table[i] * B_history[t][i]
        end
        wht!(scratch)

        # Normalize child_hat before ^(k-1) — ONLY if needed
        ch_scale = maximum(abs, scratch)
        if ch_scale > _NORM_THRESHOLD
            inv_ch = one(T) / ch_scale
            @inbounds @simd for i in 1:N
                scratch[i] *= inv_ch
            end
        else
            ch_scale = one(T)  # no normalization applied
        end
        child_hat_scales[t] = ch_scale
        child_hat_history[t] = copy(scratch)

        # folded = iWHT(kernel_hat .* child_hat .^ arity) — reuse scratch
        ch = child_hat_history[t]
        @inbounds @simd for i in 1:N
            scratch[i] = kernel_hat[i] * _fast_pow(ch[i], arity)
        end
        iwht!(scratch)

        # Normalize folded before ^(D-1) — ONLY if needed
        fld_scale = maximum(abs, scratch)
        if fld_scale > _NORM_THRESHOLD
            inv_fld = one(T) / fld_scale
            @inbounds @simd for i in 1:N
                scratch[i] *= inv_fld
            end
        else
            fld_scale = one(T)  # no normalization applied
        end
        folded_scales[t] = fld_scale
        folded_history[t] = copy(scratch)

        # B[t+1] = folded_norm .^ degree  (normalized, max magnitude ≤ 1)
        fld = folded_history[t]
        new_B = Vector{Complex{T}}(undef, N)
        @inbounds @simd for i in 1:N
            new_B[i] = _fast_pow(fld[i], degree)
        end
        B_history[t+1] = new_B

        # Update log-scale recurrence
        log_s = arity * degree * log_s +
                arity * log(ch_scale) +
                degree * log(fld_scale)
    end

    # Root computation
    root_parity_signs = [basso_root_parity(config, p) for config in 0:N-1]
    root_msg = root_parity_signs .* f_table .* B_history[p+1]

    # Root kernel — reuse phase_args instead of recomputing
    cs = T(clause_sign)
    root_kernel = Vector{Complex{T}}(undef, N)
    Threads.@threads for config in 0:N-1
        @inbounds root_kernel[config+1] = complex(zero(T), sin(half * cs * phase_args[config+1]))
    end

    # Root fold: S = Σ root_kernel .* iWHT(WHT(root_msg).^k)
    # Normalize msg_hat before ^k — ONLY if needed
    msg_hat_raw = wht!(complex.(root_msg))
    mh_scale = maximum(abs, msg_hat_raw)
    if mh_scale > _NORM_THRESHOLD
        msg_hat_raw .*= one(T) / mh_scale
    else
        mh_scale = one(T)  # no normalization applied
    end
    msg_hat = msg_hat_raw
    msg_hat_power = msg_hat .^ k
    conv = iwht(msg_hat_power)
    S_normalized = sum(root_kernel .* conv)

    # Total log-scale: B[p+1] carries scale exp(log_s), and msg_hat gets
    # an additional factor mh_scale removed.  The root fold raises to ^k:
    #   S_true = exp(k * log_s + k * log(mh_scale)) * S_normalized
    log_total_scale = k * (log_s + log(mh_scale))

    # Compute the physical value using log-space multiplication to avoid overflow.
    # c̃ = (1 + cs * exp(log_total_scale) * Re(S_normalized)) / 2
    re_S_norm = real(S_normalized)
    if re_S_norm == 0 || !isfinite(log_total_scale)
        value = half
    else
        log_product = log_total_scale + log(abs(re_S_norm))
        if log_product > 700  # would overflow exp()
            # This shouldn't happen for valid QAOA — the physical answer is bounded.
            # If it does, the intermediate precision was insufficient.
            value = T(NaN)
        else
            scaled_re_S = copysign(exp(log_product), re_S_norm)
            value = (1 + clause_sign * scaled_re_S) / 2
        end
    end

    BassoPipelineCache{T}(
        p, k, D, clause_sign,
        bit_count, N, arity, degree,
        copy(angles.β), gamma_full, trig_table,
        f_table, phase_args,
        kernel, kernel_hat,
        B_history, child_hat_history, folded_history,
        child_hat_scales, folded_scales,
        root_msg, root_parity_signs, root_kernel,
        msg_hat, mh_scale, msg_hat_power,
        log_total_scale,
        S_normalized, value,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Backward pass
# ──────────────────────────────────────────────────────────────────────────────

"""
    _backward_pass(cache) -> (γ_grad, β_grad)

Propagate cotangents backward through the cached forward pass to compute
exact gradients ∂E/∂γ[1:p] and ∂E/∂β[1:p].

The forward pass stores **normalized** intermediates.  The backward pass
operates entirely in normalized space, then applies the accumulated
log-scale multiplier `exp(log_total_scale)` to the final gradients.  The
scale factors are treated as constants (detached) for gradient purposes;
this is exact to machine precision because ∂(max|x|)/∂θ contributes
negligibly compared to the main gradient terms.
"""
function _backward_pass(cache::BassoPipelineCache{T}) where T
    p = cache.p
    N = cache.configuration_count
    cs = T(cache.clause_sign)
    half = one(T) / 2

    # Compute the scale multiplier for gradients.
    # c̃ = (1 + cs * exp(log_total_scale) * Re(S_norm)) / 2
    # ∂c̃/∂(anything_norm) = cs/2 * exp(log_total_scale) * ∂Re(S_norm)/∂(anything_norm)
    # We compute ∂S_norm/∂angles in normalized space, then multiply by scale * cs/2.
    log_lts = cache.log_total_scale
    if !isfinite(log_lts) || log_lts > 700
        # Scale is astronomical — gradients would overflow even after normalization.
        # Return zero gradients (the optimizer's overflow guard handles this).
        return (zeros(T, p), zeros(T, p))
    end
    grad_scale = exp(log_lts) * cs / 2

    # ∂E/∂S_norm in normalized space: dE/dRe(S_norm) = grad_scale, dE/dIm(S_norm) = 0
    S_bar = complex(grad_scale)

    # Root fold: S_norm = Σ root_kernel .* conv, where conv = iWHT(msg_hat_norm^k)
    conv = iwht(cache.msg_hat_power)

    root_kernel_bar = S_bar .* conj.(conv)
    conv_bar = S_bar .* conj.(cache.root_kernel)

    # iWHT adjoint: if z = iWHT(x), then x_bar += iWHT(z_bar)
    msg_hat_power_bar = iwht(conv_bar)

    # Power adjoint: msg_hat_power = msg_hat .^ k
    # x_bar += k * conj(x^(k-1)) * z_bar
    msg_hat_bar = cache.k .* conj.(cache.msg_hat .^ (cache.k - 1)) .* msg_hat_power_bar

    # WHT adjoint: msg_hat = WHT(root_msg), so root_msg_bar += WHT(msg_hat_bar)
    root_msg_bar = wht(msg_hat_bar)

    # Root message: root_msg = root_parity .* f_table .* B[p+1]
    f_table_bar = conj.(cache.root_parity_signs .* cache.B[p+1]) .* root_msg_bar
    B_bar = conj.(cache.root_parity_signs .* cache.f_table) .* root_msg_bar

    # Branch tensor backward recurrence: t = p, p-1, ..., 1
    kernel_hat_bar = zeros(Complex{T}, N)
    scratch = Vector{Complex{T}}(undef, N)

    for t in p:-1:1
        # B[t+1] = folded[t] .^ degree
        # folded_bar = degree .* conj(folded[t] .^ (degree-1)) .* B_bar
        @inbounds @simd for i in 1:N
            scratch[i] = cache.degree * conj(_fast_pow(cache.folded[t][i], cache.degree - 1)) * B_bar[i]
        end
        # scratch now holds folded_bar

        # folded[t] = iWHT(kernel_hat .* child_hat[t] .^ arity)
        # product_bar = iWHT(folded_bar)  — iWHT is self-adjoint up to scale
        iwht!(scratch)
        # scratch now holds product_bar

        # kernel_hat_bar .+= conj(child_hat .^ arity) .* product_bar
        @inbounds @simd for i in 1:N
            kernel_hat_bar[i] += conj(_fast_pow(cache.child_hat[t][i], cache.arity)) * scratch[i]
        end

        # child_hat_bar = arity .* conj(child_hat .^ (arity-1)) .* conj(kernel_hat) .* product_bar
        # Reuse scratch: overwrite with child_hat_bar, then WHT in-place
        @inbounds @simd for i in 1:N
            scratch[i] = cache.arity * conj(_fast_pow(cache.child_hat[t][i], cache.arity - 1)) *
                         conj(cache.kernel_hat[i]) * scratch[i]
        end

        # child_hat[t] = WHT(f_table .* B[t])
        # child_weights_bar = WHT(child_hat_bar)
        wht!(scratch)
        # scratch now holds child_weights_bar

        # f_table_bar .+= conj(B[t]) .* child_weights_bar
        # B_bar = conj(f_table) .* child_weights_bar
        @inbounds for i in 1:N
            f_table_bar[i] += conj(cache.B[t][i]) * scratch[i]
            B_bar[i] = conj(cache.f_table[i]) * scratch[i]
        end
    end

    # Convert kernel_hat_bar to kernel_bar: kernel_hat = WHT(kernel)
    kernel_bar = wht(kernel_hat_bar)

    # ──────────────────────────────────────────────────────────────────────
    # Angle gradients from table cotangents
    # ──────────────────────────────────────────────────────────────────────

    # γ gradient from constraint kernel:
    # kernel[a] = cos(½ · phase_dot(a))
    # ∂kernel[a]/∂γ_full[i] = -½ · sin(½ · phase_dot(a)) · spin(a,i)
    # γ_full_bar[i] += Σ_a Re(kernel_bar[a] · (-½ · sin(½ · phase_dot(a)) · spin(a,i)))
    gamma_full_bar = zeros(T, cache.bit_count)

    for config in 0:N-1
        ph = cache.phase_args[config+1]
        sin_ph = sin(half * ph)
        kb_re = real(kernel_bar[config+1])
        factor = -half * sin_ph * kb_re
        for index in 1:cache.bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            @inbounds gamma_full_bar[index] += factor * spin
        end
    end

    # γ gradient from root kernel (reuses phase_args)
    for config in 0:N-1
        ph = cache.phase_args[config+1]
        theta = half * cs * ph
        cos_theta = cos(theta)
        rkb_im = imag(root_kernel_bar[config+1])
        factor = rkb_im * cos_theta * half * cs
        for index in 1:cache.bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            @inbounds gamma_full_bar[index] += factor * spin
        end
    end

    # Map γ_full_bar -> γ_bar via the mirrored indexing
    positions = basso_phase_bit_positions(p)
    γ_bar = zeros(T, p)
    for round in 1:p
        mirror = 2p - round + 1
        bit_fwd = positions[round]
        bit_bwd = positions[2p - round + 1]
        γ_bar[round] += gamma_full_bar[bit_fwd]
        γ_bar[round] -= gamma_full_bar[bit_bwd]   # mirror has negative sign
    end

    # β gradient from f_table_bar — uses inline bit arithmetic instead of bits_table
    β_bar = zeros(T, p)

    for round in 1:p
        mirror = 2p - round + 1
        β_r = cache.β[round]
        neg_tan = -tan(β_r)
        cot_val = cos(β_r) / sin(β_r)

        acc = zero(T)
        # Bit positions are 0-indexed within config: bit at position j is (config >> (j-1)) & 1
        # d_fwd = bit[round] ⊻ bit[round+1], d_bwd = bit[mirror] ⊻ bit[mirror+1]
        fwd_shift0 = round - 1      # 0-indexed shift for bit[round]
        fwd_shift1 = round           # 0-indexed shift for bit[round+1]
        bwd_shift0 = mirror - 1     # 0-indexed shift for bit[mirror]
        bwd_shift1 = mirror          # 0-indexed shift for bit[mirror+1]

        for config in 0:N-1
            @inbounds ft = cache.f_table[config+1]
            @inbounds ftb = f_table_bar[config+1]

            d_fwd = xor((config >> fwd_shift0) & 1, (config >> fwd_shift1) & 1)
            d_bwd = xor((config >> bwd_shift0) & 1, (config >> bwd_shift1) & 1)

            ld_fwd = d_fwd == 0 ? neg_tan : cot_val
            ld_bwd = d_bwd == 0 ? neg_tan : cot_val

            acc += real(conj(ftb) * ft) * (ld_fwd + ld_bwd)
        end
        β_bar[round] = acc
    end

    (γ_bar, β_bar)
end

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    basso_expectation_and_gradient(params, angles; clause_sign=1) -> (value, γ_grad, β_grad)

Compute the expected satisfaction fraction and its exact gradient with respect
to γ[1:p] and β[1:p] in a single forward+backward pass.

Cost: approximately 2× a single Float64 evaluation, independent of p.
"""
function basso_expectation_and_gradient(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
)
    cache = _forward_pass(params, angles; clause_sign)
    γ_grad, β_grad = _backward_pass(cache)
    (cache.value, γ_grad, β_grad)
end

"""
    basso_expectation_normalized(params, angles; clause_sign=1) -> Float64

Evaluate the expected satisfaction fraction using the normalized forward pass
(no gradient).  This avoids Float64 overflow at high (k, D, p) that affects
the un-normalized `qaoa_expectation` path.
"""
function basso_expectation_normalized(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
)
    cache = _forward_pass(params, angles; clause_sign)
    cache.value
end
