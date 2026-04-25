"""
Fused branch-step kernels: compose element-wise operations with WHT
to eliminate intermediate kernel launches.

Instead of:  mul → WHT → mul+power → iWHT → power
(5 operations, ~20+ kernel launches)

We do:  fused_mul_wht → fused_fold → fused_iwht_power
(3 operations, fewer launches)
"""

using KernelAbstractions

# ── Fused multiply-then-first-butterfly ───────────────────────────────

"""
Fused: out[i] = butterfly of (a[i]*b[i]) at first level.
Combines element-wise multiply with the first WHT butterfly level,
saving one kernel launch per branch step.
"""
@kernel function _fused_mul_butterfly_kernel!(out, @Const(a), @Const(b))
    tid = @index(Global) - 1
    stride = 1
    block_idx = tid ÷ stride
    j = tid % stride
    base = block_idx * 2 + 1

    left = base + j
    right = left + stride

    @inbounds begin
        xl = a[left] * b[left]
        xr = a[right] * b[right]
        out[left] = xl + xr
        out[right] = xl - xr
    end
end

# ── Fused power-multiply (fold) ──────────────────────────────────────

"""
Fused: out[i] = kernel_hat[i] * child_hat[i]^arity
Single kernel instead of separate power + multiply.
Already exists as _fold_kernel! in gpu_forward.jl.
"""

# ── Fused iWHT-last-butterfly + power ────────────────────────────────

"""
Fused: apply last butterfly of iWHT and then power in one kernel.
out[left]  = ((values[left] + values[right]) / N) ^ degree
out[right] = ((values[left] - values[right]) / N) ^ degree
"""
@kernel function _fused_butterfly_scale_power_kernel!(out, @Const(values),
        @Const(stride), @Const(inv_N), @Const(degree))
    tid = @index(Global) - 1
    block_idx = tid ÷ stride
    j = tid % stride
    base = block_idx * (2 * stride) + 1

    left = base + j
    right = left + stride

    @inbounds begin
        x = values[left]
        y = values[right]
        lval = (x + y) * inv_N
        rval = (x - y) * inv_N

        # Inline power
        lresult = lval
        for _ in 2:degree
            lresult *= lval
        end
        rresult = rval
        for _ in 2:degree
            rresult *= rval
        end

        out[left] = lresult
        out[right] = rresult
    end
end

# ── Composed branch step ──────────────────────────────────────────────

"""
    gpu_branch_step!(B_next, scratch, B, f_table, kernel_hat, arity, degree)

Perform one complete branch tensor iteration step on GPU:
  1. scratch = WHT(f_table .* B)
  2. scratch = iWHT(kernel_hat .* scratch^arity)  
  3. B_next = scratch^degree

Uses fused kernels where possible to minimise launches.
Returns (B_next, ch_scale, fld_scale) for log-scale tracking.
"""
function gpu_branch_step!(
    B_next, scratch, B, f_table, kernel_hat,
    arity::Int, degree::Int, threshold,
)
    N = length(B)
    GT = eltype(B)
    backend = KernelAbstractions.get_backend(B)
    half_n = N ÷ 2

    # Step 1: scratch = f_table .* B, then WHT
    # Fuse the multiply into the first butterfly level
    fmb! = _fused_mul_butterfly_kernel!(backend)
    fmb!(scratch, f_table, B; ndrange=half_n)
    KernelAbstractions.synchronize(backend)

    # Remaining WHT levels (level 2 onwards)
    n_levels = trailing_zeros(N)
    stride = 2
    for level in 2:n_levels
        bfly! = _wht_butterfly_kernel!(backend)
        bfly!(scratch, half_n, stride; ndrange=half_n)
        KernelAbstractions.synchronize(backend)
        stride *= 2
    end

    # Normalize child_hat
    ch_scale = Float64(maximum(abs.(scratch)))
    if ch_scale > Float64(threshold)
        scratch .*= real(GT)(1.0 / ch_scale)
    else
        ch_scale = 1.0
    end

    # Step 2: folded = iWHT(kernel_hat .* scratch^arity)
    # Fused fold (power + multiply)
    gpu_fold!(scratch, kernel_hat, scratch, arity)

    # iWHT = WHT then divide by N
    gpu_wht!(scratch)
    scratch .*= real(GT)(1.0 / N)

    # Normalize folded
    fld_scale = Float64(maximum(abs.(scratch)))
    if fld_scale > Float64(threshold)
        scratch .*= real(GT)(1.0 / fld_scale)
    else
        fld_scale = 1.0
    end

    # Step 3: B_next = scratch^degree
    # Could fuse with last iWHT butterfly, but for now use separate kernel
    tmp = gpu_complex_power(scratch, degree)
    copyto!(B_next, tmp)

    (ch_scale, fld_scale)
end
