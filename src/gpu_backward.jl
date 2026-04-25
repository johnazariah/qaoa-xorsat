"""
GPU-accelerated backward pass for the Basso QAOA evaluation pipeline.

Computes exact gradients ∂c̃/∂γ and ∂c̃/∂β using reverse-mode adjoint
through the cached forward pass, with all tensor operations on GPU.

The angle gradient extraction (phase_dot and trig derivatives) runs on
CPU since it's not the bottleneck and requires complex bit arithmetic.
"""

include("gpu_forward.jl")

# ── Additional GPU kernels for backward pass ──────────────────────────

"""Adjoint of power: out[i] = n * conj(x[i]^(n-1)) * z_bar[i]"""
@kernel function _power_adjoint_kernel!(out, @Const(x), @Const(z_bar), @Const(n))
    i = @index(Global)
    @inbounds begin
        val = x[i]
        # Compute val^(n-1) via repeated multiply
        result = one(val)
        for _ in 1:(n-1)
            result *= val
        end
        out[i] = n * conj(result) * z_bar[i]
    end
end

"""Accumulate: a[i] += conj(x[i]^n) * b[i]"""
@kernel function _accum_conj_power_mul_kernel!(a, @Const(x), @Const(b), @Const(n))
    i = @index(Global)
    @inbounds begin
        val = x[i]
        result = val
        for _ in 2:n
            result *= val
        end
        a[i] += conj(result) * b[i]
    end
end

"""child_hat_bar: out[i] = n * conj(x[i]^(n-1)) * conj(khat[i]) * prod_bar[i]"""
@kernel function _child_hat_adjoint_kernel!(out, @Const(x), @Const(khat), @Const(prod_bar), @Const(n))
    i = @index(Global)
    @inbounds begin
        val = x[i]
        result = one(val)
        for _ in 1:(n-1)
            result *= val
        end
        out[i] = n * conj(result) * conj(khat[i]) * prod_bar[i]
    end
end

"""Fan-out: f_bar[i] += conj(B[i]) * s[i]; B_bar[i] = conj(f[i]) * s[i]"""
@kernel function _fanout_kernel!(f_bar, B_bar, @Const(B), @Const(f), @Const(s))
    i = @index(Global)
    @inbounds begin
        f_bar[i] += conj(B[i]) * s[i]
        B_bar[i] = conj(f[i]) * s[i]
    end
end

# ── GPU forward+backward combined ─────────────────────────────────────

"""
    gpu_forward_backward(params, angles, gpu_array_fn; clause_sign=1)
        -> (value, γ_grad, β_grad)

Compute c̃ and its gradient using GPU-accelerated forward and backward passes.
Returns Float64 value and gradient vectors.

The forward pass caches all intermediates on GPU; the backward pass
propagates cotangents through them. Angle gradients (phase_dot derivatives)
are extracted on CPU.
"""
function gpu_forward_backward(
    params::TreeParams,
    angles::QAOAAngles,
    gpu_array_fn::Function;
    clause_sign::Int=1,
)
    p = params.p
    k = params.k
    D = params.D
    arity = k - 1
    degree = D - 1
    bit_count = basso_bit_count(p)
    N = basso_configuration_count(p)

    depth(angles) == p || throw(ArgumentError("angle depth must match tree depth"))
    validate_clause_sign(clause_sign)

    # ── CPU precomputation ────────────────────────────────────────
    gamma_full = build_gamma_full_vector(angles)
    trig_table = basso_trig_table(angles)
    f_table_cpu = _basso_f_table_fast(trig_table, bit_count, N, Float64)

    half = 0.5
    phase_args = Vector{Float64}(undef, N)
    kernel_cpu = Vector{ComplexF64}(undef, N)
    for config in 0:N-1
        ph = _phase_dot(gamma_full, config, bit_count)
        phase_args[config+1] = ph
        kernel_cpu[config+1] = complex(cos(half * ph))
    end
    kernel_hat_cpu = QaoaXorsat.wht!(copy(kernel_cpu))

    root_parity_cpu = Float64[basso_root_parity(config, p) for config in 0:N-1]
    cs = Float64(clause_sign)
    root_kernel_cpu = [complex(zero(Float64), sin(half * cs * phase_args[config+1]))
                       for config in 0:N-1]

    # ── Transfer to GPU ───────────────────────────────────────────
    f_table_gpu = gpu_array_fn(f_table_cpu)
    kernel_hat_gpu = gpu_array_fn(kernel_hat_cpu)
    root_parity_gpu = gpu_array_fn(complex.(root_parity_cpu))
    root_kernel_gpu = gpu_array_fn(root_kernel_cpu)

    GT = eltype(f_table_gpu)
    backend = KernelAbstractions.get_backend(f_table_gpu)
    _NORM_THRESHOLD = real(GT)(1e15)

    # ── Forward pass with caching ─────────────────────────────────
    B_history = Vector{typeof(f_table_gpu)}(undef, p + 1)
    child_hat_history = Vector{typeof(f_table_gpu)}(undef, p)
    folded_history = Vector{typeof(f_table_gpu)}(undef, p)
    child_hat_scales = Vector{Float64}(undef, p)
    folded_scales = Vector{Float64}(undef, p)

    B_history[1] = gpu_array_fn(ones(ComplexF64, N))
    scratch = similar(f_table_gpu)
    log_s = 0.0

    for t in 1:p
        gpu_elemwise_mul!(scratch, f_table_gpu, B_history[t])
        gpu_wht!(scratch)

        ch_scale = Float64(maximum(abs.(scratch)))
        if ch_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / ch_scale)
        else
            ch_scale = 1.0
        end
        child_hat_scales[t] = ch_scale
        child_hat_history[t] = copy(scratch)

        gpu_fold!(scratch, kernel_hat_gpu, child_hat_history[t], arity)
        gpu_iwht!(scratch)

        fld_scale = Float64(maximum(abs.(scratch)))
        if fld_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / fld_scale)
        else
            fld_scale = 1.0
        end
        folded_scales[t] = fld_scale
        folded_history[t] = copy(scratch)

        B_history[t+1] = gpu_complex_power(folded_history[t], degree)

        log_s = arity * degree * log_s +
                arity * log(ch_scale) +
                degree * log(fld_scale)
    end

    # Root fold
    root_msg = similar(B_history[1])
    mul3! = _elemwise_mul3_kernel!(backend)
    mul3!(root_msg, root_parity_gpu, f_table_gpu, B_history[p+1]; ndrange=N)
    KernelAbstractions.synchronize(backend)

    gpu_wht!(root_msg)
    mh_scale = Float64(maximum(abs.(root_msg)))
    if mh_scale > Float64(_NORM_THRESHOLD)
        root_msg .*= real(GT)(1.0 / mh_scale)
    else
        mh_scale = 1.0
    end
    msg_hat_gpu = copy(root_msg)

    msg_hat_power = gpu_complex_power(msg_hat_gpu, k)
    conv_gpu = gpu_iwht(copy(msg_hat_power))

    product_tmp = similar(conv_gpu)
    gpu_elemwise_mul!(product_tmp, root_kernel_gpu, conv_gpu)
    S_normalized = ComplexF64(sum(Array(product_tmp)))

    log_total_scale = k * (log_s + log(mh_scale))

    # Compute value
    re_S_norm = real(S_normalized)
    if re_S_norm == 0 || !isfinite(log_total_scale)
        value = 0.5
    else
        log_product = log_total_scale + log(abs(re_S_norm))
        if log_product > 700
            value = NaN
        else
            scaled_re_S = copysign(exp(log_product), re_S_norm)
            value = (1 + clause_sign * scaled_re_S) / 2
        end
    end

    # ── Backward pass on GPU ──────────────────────────────────────
    if !isfinite(log_total_scale) || log_total_scale > 700
        return (value, zeros(Float64, p), zeros(Float64, p))
    end
    grad_scale = exp(log_total_scale) * cs / 2
    S_bar_val = GT(grad_scale)

    # Root fold backward
    root_kernel_bar = S_bar_val .* conj.(conv_gpu)
    conv_bar = S_bar_val .* conj.(root_kernel_gpu)

    msg_hat_power_bar = gpu_iwht(copy(conv_bar))

    # Power adjoint: msg_hat_power = msg_hat^k
    pa_kernel! = _power_adjoint_kernel!(backend)
    msg_hat_bar = similar(msg_hat_gpu)
    pa_kernel!(msg_hat_bar, msg_hat_gpu, msg_hat_power_bar, k; ndrange=N)
    KernelAbstractions.synchronize(backend)

    root_msg_bar = gpu_wht(copy(msg_hat_bar))

    # root_msg = root_parity .* f_table .* B[p+1]
    f_table_bar = similar(f_table_gpu)
    B_bar = similar(f_table_gpu)

    # f_table_bar = conj(root_parity .* B[p+1]) .* root_msg_bar
    @kernel function _root_fanout_kernel!(f_bar, B_bar, @Const(rp), @Const(ft), @Const(B), @Const(rmb))
        i = @index(Global)
        @inbounds begin
            f_bar[i] = conj(rp[i] * B[i]) * rmb[i]
            B_bar[i] = conj(rp[i] * ft[i]) * rmb[i]
        end
    end
    rf_kernel! = _root_fanout_kernel!(backend)
    rf_kernel!(f_table_bar, B_bar, root_parity_gpu, f_table_gpu, B_history[p+1], root_msg_bar; ndrange=N)
    KernelAbstractions.synchronize(backend)

    kernel_hat_bar = gpu_array_fn(zeros(ComplexF64, N))

    # Branch backward recurrence
    for t in p:-1:1
        # folded_bar = degree * conj(folded^(degree-1)) * B_bar
        folded_bar = similar(scratch)
        pa2_kernel! = _power_adjoint_kernel!(backend)
        pa2_kernel!(folded_bar, folded_history[t], B_bar, degree; ndrange=N)
        KernelAbstractions.synchronize(backend)

        # product_bar = iWHT(folded_bar)
        gpu_iwht!(folded_bar)

        # kernel_hat_bar += conj(child_hat^arity) * product_bar
        accum_kernel! = _accum_conj_power_mul_kernel!(backend)
        accum_kernel!(kernel_hat_bar, child_hat_history[t], folded_bar, arity; ndrange=N)
        KernelAbstractions.synchronize(backend)

        # child_hat_bar = arity * conj(child_hat^(arity-1)) * conj(kernel_hat) * product_bar
        cha_kernel! = _child_hat_adjoint_kernel!(backend)
        cha_kernel!(scratch, child_hat_history[t], kernel_hat_gpu, folded_bar, arity; ndrange=N)
        KernelAbstractions.synchronize(backend)

        # child_weights_bar = WHT(child_hat_bar)
        gpu_wht!(scratch)

        # Fan-out
        fanout_kernel! = _fanout_kernel!(backend)
        fanout_kernel!(f_table_bar, B_bar, B_history[t], f_table_gpu, scratch; ndrange=N)
        KernelAbstractions.synchronize(backend)
    end

    # ── Angle gradients on CPU ────────────────────────────────────
    kernel_bar_cpu = ComplexF64.(Array(gpu_wht(copy(kernel_hat_bar))))
    root_kernel_bar_cpu = ComplexF64.(Array(root_kernel_bar))
    f_table_bar_cpu = ComplexF64.(Array(f_table_bar))

    # γ gradient from constraint kernel
    gamma_full_bar = zeros(Float64, bit_count)
    for config in 0:N-1
        ph = phase_args[config+1]
        sin_ph = sin(half * ph)
        kb_re = real(kernel_bar_cpu[config+1])
        factor = -half * sin_ph * kb_re
        for index in 1:bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            gamma_full_bar[index] += factor * spin
        end
    end

    # γ gradient from root kernel
    for config in 0:N-1
        ph = phase_args[config+1]
        theta = half * cs * ph
        cos_theta = cos(theta)
        rkb_im = imag(root_kernel_bar_cpu[config+1])
        factor = rkb_im * cos_theta * half * cs
        for index in 1:bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            gamma_full_bar[index] += factor * spin
        end
    end

    # Map γ_full_bar -> γ_bar
    positions = QaoaXorsat.basso_phase_bit_positions(p)
    γ_bar = zeros(Float64, p)
    for round in 1:p
        bit_fwd = positions[round]
        bit_bwd = positions[2p - round + 1]
        γ_bar[round] += gamma_full_bar[bit_fwd]
        γ_bar[round] -= gamma_full_bar[bit_bwd]
    end

    # β gradient from f_table_bar
    β_bar = zeros(Float64, p)
    for round in 1:p
        mirror = 2p - round + 1
        β_r = angles.β[round]
        neg_tan = -tan(β_r)
        cot_val = cos(β_r) / sin(β_r)

        fwd_shift0 = round - 1
        fwd_shift1 = round
        bwd_shift0 = mirror - 1
        bwd_shift1 = mirror

        acc = 0.0
        for config in 0:N-1
            ft = f_table_cpu[config+1]
            ftb = f_table_bar_cpu[config+1]

            d_fwd = xor((config >> fwd_shift0) & 1, (config >> fwd_shift1) & 1)
            d_bwd = xor((config >> bwd_shift0) & 1, (config >> bwd_shift1) & 1)

            ld_fwd = d_fwd == 0 ? neg_tan : cot_val
            ld_bwd = d_bwd == 0 ? neg_tan : cot_val

            acc += real(conj(ftb) * ft) * (ld_fwd + ld_bwd)
        end
        β_bar[round] = acc
    end

    (value, γ_bar, β_bar)
end
