# Manual adjoint (reverse-mode) differentiation for the Basso evaluator.
#
# The forward pass matches basso_parity_expectation / basso_expectation exactly.
# The backward pass propagates cotangents through the saved computation graph
# to produce exact ∂E/∂γ and ∂E/∂β at cost ≈ 2× a single Float64 evaluation.

# ──────────────────────────────────────────────────────────────────────────────
# Forward pass with caching
# ──────────────────────────────────────────────────────────────────────────────

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
    bits_table::Matrix{Int}          # bit_count × configuration_count (columns = configs)

    # f_table and per-config phase arguments
    f_table::Vector{Complex{T}}
    constraint_phase::Vector{T}      # phase arg per config for constraint kernel

    # Constraint kernel and its WHT
    kernel::Vector{Complex{T}}
    kernel_hat::Vector{Complex{T}}

    # Branch tensor history: B[1] = ones, B[t+1] = step(B[t])
    B::Vector{Vector{Complex{T}}}

    # Per-step intermediates (saved for backward)
    child_hat::Vector{Vector{Complex{T}}}    # WHT(f_table .* B[t])
    folded::Vector{Vector{Complex{T}}}       # iWHT(kernel_hat .* child_hat^arity)

    # Root computation
    root_msg::Vector{Complex{T}}
    root_parity_signs::Vector{Int}
    root_kernel::Vector{Complex{T}}
    root_phase::Vector{T}            # phase arg per config for root kernel
    msg_hat::Vector{Complex{T}}      # WHT(root_msg)
    msg_hat_power::Vector{Complex{T}}  # msg_hat .^ k

    # Final result
    S::Complex{T}
    value::T
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

    # Precompute bits table (integer, independent of angles)
    bits_table = Matrix{Int}(undef, bit_count, N)
    for config in 0:N-1
        for index in 1:bit_count
            @inbounds bits_table[index, config+1] = (config >> (index - 1)) & 1
        end
    end

    # f_table (threaded)
    f_table = basso_f_table(angles)

    # Constraint kernel with cached phase arguments
    half = one(T) / 2
    constraint_phase = Vector{T}(undef, N)
    kernel = Vector{Complex{T}}(undef, N)
    Threads.@threads for config in 0:N-1
        ph = _phase_dot(gamma_full, config, bit_count)
        @inbounds constraint_phase[config+1] = ph
        @inbounds kernel[config+1] = complex(cos(half * ph))
    end
    kernel_hat = wht(complex.(kernel))

    # Branch tensor iteration with caching
    # Pre-allocate scratch buffer to avoid repeated allocations in the hot loop.
    # Each step previously allocated ~5 temporary vectors of size N.
    B_history = Vector{Vector{Complex{T}}}(undef, p + 1)
    child_hat_history = Vector{Vector{Complex{T}}}(undef, p)
    folded_history = Vector{Vector{Complex{T}}}(undef, p)

    scratch = Vector{Complex{T}}(undef, N)

    B_history[1] = ones(Complex{T}, N)
    for t in 1:p
        # child_weights = f_table .* B[t] — compute into scratch, then WHT in-place
        @inbounds @simd for i in 1:N
            scratch[i] = f_table[i] * B_history[t][i]
        end
        wht!(scratch)
        child_hat_history[t] = copy(scratch)

        # folded = iWHT(kernel_hat .* child_hat .^ arity) — reuse scratch
        ch = child_hat_history[t]
        @inbounds @simd for i in 1:N
            scratch[i] = kernel_hat[i] * ch[i] ^ arity
        end
        iwht!(scratch)
        folded_history[t] = copy(scratch)

        # B[t+1] = folded .^ degree
        fld = folded_history[t]
        new_B = Vector{Complex{T}}(undef, N)
        @inbounds @simd for i in 1:N
            new_B[i] = fld[i] ^ degree
        end
        B_history[t+1] = new_B
    end

    # Root computation
    root_parity_signs = [basso_root_parity(config, p) for config in 0:N-1]
    root_msg = root_parity_signs .* f_table .* B_history[p+1]

    # Root kernel with cached phase
    cs = T(clause_sign)
    root_phase = Vector{T}(undef, N)
    root_kernel = Vector{Complex{T}}(undef, N)
    Threads.@threads for config in 0:N-1
        ph = _phase_dot(gamma_full, config, bit_count)
        @inbounds root_phase[config+1] = ph
        @inbounds root_kernel[config+1] = complex(zero(T), sin(half * cs * ph))
    end

    # Root fold: S = Σ root_kernel .* iWHT(WHT(root_msg).^k)
    msg_hat = wht(complex.(root_msg))
    msg_hat_power = msg_hat .^ k
    conv = iwht(msg_hat_power)
    S = sum(root_kernel .* conv)
    value = (1 + clause_sign * real(S)) / 2

    BassoPipelineCache{T}(
        p, k, D, clause_sign,
        bit_count, N, arity, degree,
        copy(angles.β), gamma_full, trig_table, bits_table,
        f_table, constraint_phase,
        kernel, kernel_hat,
        B_history, child_hat_history, folded_history,
        root_msg, root_parity_signs, root_kernel, root_phase,
        msg_hat, msg_hat_power,
        S, value,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Backward pass
# ──────────────────────────────────────────────────────────────────────────────

"""
    _backward_pass(cache) -> (γ_grad, β_grad)

Propagate cotangents backward through the cached forward pass to compute
exact gradients ∂E/∂γ[1:p] and ∂E/∂β[1:p].
"""
function _backward_pass(cache::BassoPipelineCache{T}) where T
    p = cache.p
    N = cache.configuration_count
    cs = T(cache.clause_sign)
    half = one(T) / 2

    # ∂E/∂S: E = (1 + cs * Re(S)) / 2, so dE/dRe(S) = cs/2, dE/dIm(S) = 0
    S_bar = complex(cs / 2)

    # Root fold: S = Σ root_kernel .* conv, where conv = iWHT(msg_hat^k)
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
            scratch[i] = cache.degree * conj(cache.folded[t][i] ^ (cache.degree - 1)) * B_bar[i]
        end
        # scratch now holds folded_bar

        # folded[t] = iWHT(kernel_hat .* child_hat[t] .^ arity)
        # product_bar = iWHT(folded_bar)  — iWHT is self-adjoint up to scale
        iwht!(scratch)
        # scratch now holds product_bar

        # kernel_hat_bar .+= conj(child_hat .^ arity) .* product_bar
        @inbounds @simd for i in 1:N
            kernel_hat_bar[i] += conj(cache.child_hat[t][i] ^ cache.arity) * scratch[i]
        end

        # child_hat_bar = arity .* conj(child_hat .^ (arity-1)) .* conj(kernel_hat) .* product_bar
        # Reuse scratch: overwrite with child_hat_bar, then WHT in-place
        @inbounds @simd for i in 1:N
            scratch[i] = cache.arity * conj(cache.child_hat[t][i] ^ (cache.arity - 1)) *
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
        ph = cache.constraint_phase[config+1]
        sin_ph = sin(half * ph)
        kb_re = real(kernel_bar[config+1])
        factor = -half * sin_ph * kb_re
        for index in 1:cache.bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            @inbounds gamma_full_bar[index] += factor * spin
        end
    end

    # γ gradient from root kernel:
    # root_kernel[a] = i · sin(½ · cs · phase_dot(a))
    # ∂root_kernel[a]/∂γ_full[i] = i · ½ · cs · cos(½ · cs · phase_dot(a)) · spin(a,i)
    # Contribution: Re(root_kernel_bar[a]* · i · ½ · cs · cos(½ · cs · phase_dot(a)) · spin(a,i)))
    # = -Im(root_kernel_bar[a]) · ½ · cs · cos(½ · cs · phase_dot(a)) · spin(a,i)
    # (because Re(z* · i·w) = -Im(z) · Re(w) + Re(z) · Im(w), and cos is real)
    # Actually: root_kernel_bar is complex. Let rkb = root_kernel_bar[a].
    # derivative direction = i · ½ · cs · cos(½·cs·ph) · spin
    # cotangent contribution = Re(conj(rkb) · i · ½ · cs · cos(½·cs·ph) · spin)
    # = Re((Re(rkb) - i·Im(rkb)) · (i · ½ · cs · cos(½·cs·ph) · spin))
    # = Re(i·Re(rkb) · ... + Im(rkb) · ...)
    # = ½ · cs · cos(½·cs·ph) · spin · (-Im(rkb) + 0) ... wait, let me redo.
    # Actually for a real-valued loss, the correct formula is:
    # If z = complex(0, sin(θ)), dz/dθ = complex(0, cos(θ))
    # Then θ_bar += Re(z_bar · conj(dz/dθ)) = Re(z_bar · complex(0, -cos(θ)))
    #            = Im(z_bar) · (-cos(θ))  ... no.
    # Re(z_bar · conj(complex(0, cos(θ)))) = Re(z_bar · complex(0, -cos(θ)))
    # = Re((a+bi)(0 - ci)) = Re(-aci + bci²) = Re(-aci - bc) = -bc
    # where z_bar = a + bi, so = -Im(z_bar) · cos(θ) ... hmm that's not right either.
    # Let me use the real chain rule directly:
    # z = 0 + i·sin(θ), so Re(z) = 0, Im(z) = sin(θ)
    # ∂L/∂θ = ∂L/∂Re(z) · 0 + ∂L/∂Im(z) · cos(θ) = Im_bar(z) · cos(θ)
    # But z_bar encodes: z_bar = ∂L/∂Re(z) + i·∂L/∂Im(z)
    # So Im_bar(z) = Im(z_bar)
    # Therefore: θ_bar = Im(z_bar) · cos(θ)

    for config in 0:N-1
        ph = cache.root_phase[config+1]
        theta = half * cs * ph
        cos_theta = cos(theta)
        rkb_im = imag(root_kernel_bar[config+1])
        # θ = ½ · cs · Σ_i γ_full[i] · spin(a,i)
        # ∂θ/∂γ_full[i] = ½ · cs · spin(a,i)
        factor = rkb_im * cos_theta * half * cs
        for index in 1:cache.bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            @inbounds gamma_full_bar[index] += factor * spin
        end
    end

    # Map γ_full_bar -> γ_bar via the mirrored indexing
    # gamma_full[bit_index] = gamma_vector[gamma_index]
    # gamma_vector[round] = γ[round], gamma_vector[mirror] = -γ[round]
    positions = basso_phase_bit_positions(p)
    γ_bar = zeros(T, p)
    for round in 1:p
        mirror = 2p - round + 1
        # gamma_vector[round] = γ[round] → γ_bar[round] += gamma_full_bar[positions[round]]
        # gamma_vector[mirror] = -γ[round] → γ_bar[round] -= gamma_full_bar[positions[mirror]]
        # But positions maps gamma_index -> bit_index, and gamma_full_bar is indexed by bit_index.
        # positions[round] is the bit_index for the forward direction
        # positions[p + round'] where round' = 2p - round + 1 - p = p - round + 1
        # Actually: positions = [1:p; (p+2):(2p+1)]
        # gamma_vector has 2p entries: gamma_vector[round] and gamma_vector[2p-round+1]
        # gamma_full is 2p+1 entries, and gamma_full[positions[gi]] = gamma_vector[gi]
        bit_fwd = positions[round]
        bit_bwd = positions[2p - round + 1]
        γ_bar[round] += gamma_full_bar[bit_fwd]
        γ_bar[round] -= gamma_full_bar[bit_bwd]   # mirror has negative sign
    end

    # β gradient from f_table_bar:
    # f[a] = ½ · ∏_j trigs[Δ(a,j)+1, j]
    # ∂f/∂β_r = f · [(dtrig_fwd/trig_fwd) at position r + (dtrig_bwd/trig_bwd) at mirror r]
    # The log-derivative ratio dtrig/trig depends only on Δ and β_r:
    #   Δ=0: dtrig/trig = -sin(β)/cos(β) = -tan(β)          (forward)
    #   Δ=1: dtrig/trig = i·cos(β)/(i·sin(β)) = cot(β)      (forward)
    #   Δ=0 mirror: dtrig/trig = -sin(β)/cos(β) = -tan(β)    (same as forward, since cos(-β)=cos(β))
    #   Δ=1 mirror: dtrig/trig = -i·cos(β)/(i·(-sin(β))) = cos(β)/sin(β) = cot(β) (same!)
    # So the log-derivative ratio is: Δ=0 → -tan(β_r), Δ=1 → cot(β_r), for both positions.
    #
    # Therefore: ∂f/∂β_r = f · (logderiv[Δ_fwd] + logderiv[Δ_bwd])
    # And: β_bar[r] = Σ_a Re(conj(f_table_bar[a]) · f[a] · (logderiv[Δ_fwd(a)] + logderiv[Δ_bwd(a)]))

    β_bar = zeros(T, p)

    for round in 1:p
        mirror = 2p - round + 1
        β_r = cache.β[round]
        neg_tan = -tan(β_r)
        cot_val = cos(β_r) / sin(β_r)
        # logderiv: Δ=0 → neg_tan, Δ=1 → cot_val

        acc = zero(T)
        for config in 0:N-1
            @inbounds ft = cache.f_table[config+1]
            @inbounds ftb = f_table_bar[config+1]

            d_fwd = xor(cache.bits_table[round, config+1], cache.bits_table[round+1, config+1])
            d_bwd = xor(cache.bits_table[mirror, config+1], cache.bits_table[mirror+1, config+1])

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
