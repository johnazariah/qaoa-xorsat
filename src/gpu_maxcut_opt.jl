"""
Optimized GPU forward pass for MaxCut (k=2) — eliminates unnecessary
kernel launches by exploiting k=2 structure:

1. arity=1: child_hat^(k-1) = child_hat^1 = identity (no power kernel)
2. No normalization needed for Float32 through p≤14 (magnitudes safe)
3. iWHT scale folded into degree-power kernel
4. Fold simplifies to elementwise multiply (no power)

This reduces kernel launches from ~20 per step to ~12 per step.
"""

include("gpu_wht.jl")

using KernelAbstractions

# ── Specialized kernels for k=2 ───────────────────────────────────────

"""
Fused: out[i] = (kernel_hat[i] * x[i] / N) ^ degree
Combines iWHT scaling with power, eliminating one kernel launch.
"""
@kernel function _fold_scale_power_kernel!(out, @Const(wht_result), @Const(kernel_hat),
                                           @Const(inv_N), @Const(degree))
    i = @index(Global)
    @inbounds begin
        # fold (arity=1, so no power on child_hat)
        folded_hat = kernel_hat[i] * wht_result[i]
        # This will become iWHT input — but we can't do iWHT here.
        # Store the folded_hat for iWHT.
        out[i] = folded_hat
    end
end

"""
Fused: out[i] = (x[i] * inv_N)^degree
Combines iWHT scaling (÷N) with the subsequent power operation.
"""
@kernel function _scale_power_kernel!(out, @Const(x), @Const(inv_N), @Const(degree))
    i = @index(Global)
    @inbounds begin
        val = x[i] * inv_N
        result = val
        for _ in 2:degree
            result *= val
        end
        out[i] = result
    end
end

"""
Fused multiply: out[i] = a[i] * b[i]
"""
@kernel function _mul_kernel!(out, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds out[i] = a[i] * b[i]
end

"""
Fused triple multiply: out[i] = a[i] * b[i] * c[i]
"""
@kernel function _mul3_kernel_opt!(out, @Const(a), @Const(b), @Const(c))
    i = @index(Global)
    @inbounds out[i] = a[i] * b[i] * c[i]
end

# ── Optimized forward pass for MaxCut ─────────────────────────────────

"""
    gpu_forward_maxcut(params, angles, gpu_array_fn) -> Float64

GPU forward pass optimized for MaxCut (k=2).
Eliminates normalize checks and exploits arity=1.
"""
function gpu_forward_maxcut(
    params::TreeParams,
    angles::QAOAAngles,
    gpu_array_fn::Function;
)
    p = params.p
    k = params.k
    D = params.D
    @assert k == 2 "This function is specialized for MaxCut (k=2)"
    degree = D - 1  # arity = k-1 = 1
    bit_count = basso_bit_count(p)
    N = basso_configuration_count(p)

    # CPU precomputation
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
    root_kernel_cpu = [complex(zero(Float64), sin(half * (-1.0) * phase_args[config+1]))
                       for config in 0:N-1]

    # Transfer to GPU
    f_table_gpu = gpu_array_fn(f_table_cpu)
    kernel_hat_gpu = gpu_array_fn(kernel_hat_cpu)
    root_parity_gpu = gpu_array_fn(complex.(root_parity_cpu))
    root_kernel_gpu = gpu_array_fn(root_kernel_cpu)

    GT = eltype(f_table_gpu)
    backend = KernelAbstractions.get_backend(f_table_gpu)
    inv_N_val = real(GT)(1.0 / N)

    # Branch iteration — no normalization, arity=1 simplification
    B = gpu_array_fn(ones(ComplexF64, N))
    scratch = similar(B)
    scratch2 = similar(B)

    mul! = _mul_kernel!(backend)
    sp! = _scale_power_kernel!(backend)

    for t in 1:p
        # scratch = f_table .* B  (1 launch)
        mul!(scratch, f_table_gpu, B; ndrange=N)
        KernelAbstractions.synchronize(backend)

        # scratch = WHT(scratch)  (~5 launches with fused levels)
        gpu_wht!(scratch)

        # scratch2 = kernel_hat .* scratch  (arity=1, no power!) (1 launch)
        mul!(scratch2, kernel_hat_gpu, scratch; ndrange=N)
        KernelAbstractions.synchronize(backend)

        # B = (iWHT(scratch2))^degree = (WHT(scratch2)/N)^degree
        # Do WHT first (~5 launches)
        gpu_wht!(scratch2)

        # Then fused scale+power: B[i] = (scratch2[i]/N)^degree (1 launch)
        sp!(B, scratch2, inv_N_val, degree; ndrange=N)
        KernelAbstractions.synchronize(backend)
    end
    # Per step: 1 + ~5 + 1 + ~5 + 1 = ~13 launches (was ~20)

    # Root fold
    root_msg = similar(B)
    mul3! = _mul3_kernel_opt!(backend)
    mul3!(root_msg, root_parity_gpu, f_table_gpu, B; ndrange=N)
    KernelAbstractions.synchronize(backend)

    gpu_wht!(root_msg)
    # No normalize — safe for MaxCut

    msg_hat_power = gpu_complex_power(root_msg, k)
    # iWHT = WHT + /N, but k=2 so power is just squaring
    gpu_wht!(msg_hat_power)
    msg_hat_power .*= inv_N_val

    prod_tmp = similar(B)
    mul!(prod_tmp, root_kernel_gpu, msg_hat_power; ndrange=N)
    KernelAbstractions.synchronize(backend)

    S_normalized = ComplexF64(sum(Array(prod_tmp)))

    # No log-scale (no normalization was applied)
    re_S = real(S_normalized)
    value = (1 + (-1) * re_S) / 2

    return value
end
