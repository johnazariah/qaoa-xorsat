"""
GPU-accelerated forward pass for the Basso QAOA evaluation pipeline.

Performs the branch tensor iteration and root fold on GPU, using the
GPU WHT kernel from gpu_wht.jl. Precomputation (angles, trig tables,
phase args) is done on CPU; only the O(p² · 4^p) hot loop runs on GPU.

The GPU forward pass computes c̃ only (no gradient). For gradient
computation, see gpu_backward.jl (Phase 3).

Metal backend uses Float32; CUDA uses Float64.
"""

include("gpu_wht.jl")

using KernelAbstractions
using QaoaXorsat: TreeParams, QAOAAngles, depth, validate_clause_sign,
    basso_bit_count, basso_configuration_count, basso_trig_table,
    basso_root_parity, build_gamma_full_vector, _phase_dot,
    _basso_f_table_fast, z_eigenvalue, wht!

# ── GPU element-wise kernels ──────────────────────────────────────────

"""Element-wise multiply: out[i] = a[i] * b[i]"""
@kernel function _elemwise_mul_kernel!(out, @Const(a), @Const(b))
    i = @index(Global)
    @inbounds out[i] = a[i] * b[i]
end

"""Element-wise multiply-accumulate: out[i] = a[i] * b[i] * c[i]"""
@kernel function _elemwise_mul3_kernel!(out, @Const(a), @Const(b), @Const(c))
    i = @index(Global)
    @inbounds out[i] = a[i] * b[i] * c[i]
end

"""Element-wise: out[i] = kernel_hat[i] * power(child_hat[i], arity)"""
@kernel function _fold_kernel!(out, @Const(kernel_hat), @Const(child_hat), @Const(arity))
    i = @index(Global)
    @inbounds begin
        val = child_hat[i]
        result = val
        for _ in 2:arity
            result *= val
        end
        out[i] = kernel_hat[i] * result
    end
end

function gpu_elemwise_mul!(out, a, b)
    backend = KernelAbstractions.get_backend(out)
    kernel! = _elemwise_mul_kernel!(backend)
    kernel!(out, a, b; ndrange=length(out))
    KernelAbstractions.synchronize(backend)
    out
end

function gpu_fold!(out, kernel_hat, child_hat, arity)
    backend = KernelAbstractions.get_backend(out)
    kernel! = _fold_kernel!(backend)
    kernel!(out, kernel_hat, child_hat, arity; ndrange=length(out))
    KernelAbstractions.synchronize(backend)
    out
end

# ── GPU forward pass ──────────────────────────────────────────────────

"""
    gpu_forward_value(params, angles; clause_sign=1, gpu_array_fn) -> Float64

Compute the QAOA satisfaction fraction c̃ using GPU-accelerated
branch tensor iteration.

`gpu_array_fn` converts a CPU array to GPU (e.g., MtlArray for Metal).
The element type is determined by the GPU backend (Float32 for Metal,
Float64 for CUDA).

Returns the c̃ value as a Float64.
"""
function gpu_forward_value(
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

    # ── CPU precomputation (small, angle-dependent) ───────────────
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
    kernel_hat_cpu = wht!(copy(kernel_cpu))

    root_parity_cpu = Float64[basso_root_parity(config, p) for config in 0:N-1]
    cs = Float64(clause_sign)
    root_kernel_cpu = [complex(zero(Float64), sin(half * cs * phase_args[config+1]))
                       for config in 0:N-1]

    # ── Transfer to GPU ───────────────────────────────────────────
    f_table_gpu = gpu_array_fn(f_table_cpu)
    kernel_hat_gpu = gpu_array_fn(kernel_hat_cpu)
    root_parity_gpu = gpu_array_fn(complex.(root_parity_cpu))
    root_kernel_gpu = gpu_array_fn(root_kernel_cpu)

    # Determine GPU element type
    GT = eltype(f_table_gpu)

    # ── Branch iteration on GPU ───────────────────────────────────
    _NORM_THRESHOLD = real(GT)(1e15)  # lower for Float32 (max ~1e38)

    B_gpu = gpu_array_fn(ones(ComplexF64, N))
    scratch = similar(B_gpu)
    log_s = 0.0  # keep scale tracking in Float64

    for t in 1:p
        # child_weights = f_table .* B
        gpu_elemwise_mul!(scratch, f_table_gpu, B_gpu)

        # child_hat = WHT(child_weights)
        gpu_wht!(scratch)

        # Normalize child_hat before ^(k-1)
        ch_scale = Float64(maximum(abs.(scratch)))
        if ch_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / ch_scale)
        else
            ch_scale = 1.0
        end

        # folded = iWHT(kernel_hat .* child_hat^arity)
        gpu_fold!(scratch, kernel_hat_gpu, scratch, arity)
        gpu_iwht!(scratch)

        # Normalize folded before ^(D-1)
        fld_scale = Float64(maximum(abs.(scratch)))
        if fld_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / fld_scale)
        else
            fld_scale = 1.0
        end

        # B[t+1] = folded^degree
        B_gpu = gpu_complex_power(scratch, degree)

        # Update log-scale
        log_s = arity * degree * log_s +
                arity * log(ch_scale) +
                degree * log(fld_scale)
    end

    # ── Root fold on GPU ──────────────────────────────────────────
    # root_msg = root_parity .* f_table .* B
    root_msg = similar(B_gpu)
    backend = KernelAbstractions.get_backend(B_gpu)
    mul3_kernel! = _elemwise_mul3_kernel!(backend)
    mul3_kernel!(root_msg, root_parity_gpu, f_table_gpu, B_gpu; ndrange=N)
    KernelAbstractions.synchronize(backend)

    # msg_hat = WHT(root_msg), normalized
    gpu_wht!(root_msg)
    mh_scale = Float64(maximum(abs.(root_msg)))
    if mh_scale > Float64(_NORM_THRESHOLD)
        root_msg .*= real(GT)(1.0 / mh_scale)
    else
        mh_scale = 1.0
    end

    # msg_hat_power = msg_hat^k
    msg_power = gpu_complex_power(root_msg, k)

    # conv = iWHT(msg_hat_power)
    gpu_iwht!(msg_power)

    # S = sum(root_kernel .* conv) — reduction
    product = similar(msg_power)
    gpu_elemwise_mul!(product, root_kernel_gpu, msg_power)
    S_normalized = ComplexF64(sum(Array(product)))

    # ── Compute c̃ from S and log-scale ───────────────────────────
    log_total_scale = k * (log_s + log(mh_scale))
    re_S_norm = real(S_normalized)

    if re_S_norm == 0 || !isfinite(log_total_scale)
        return 0.5
    end

    log_product = log_total_scale + log(abs(re_S_norm))
    if log_product > 700
        return NaN
    end

    scaled_re_S = copysign(exp(log_product), re_S_norm)
    value = (1 + clause_sign * scaled_re_S) / 2

    return value
end
