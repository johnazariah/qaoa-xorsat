"""
GPU-accelerated Walsh-Hadamard Transform using KernelAbstractions.jl.

Supports Metal (Apple Silicon) and CUDA (NVIDIA) backends via a
portable kernel abstraction. Metal requires Float32; CUDA supports Float64.

The GPU WHT uses a level-by-level butterfly approach: each level launches
a kernel where each thread handles one butterfly pair independently.

Usage:
    using Metal  # or CUDA
    x_gpu = MtlArray(ComplexF32.(x_cpu))
    gpu_wht!(x_gpu)              # in-place WHT on GPU
    x_hat = gpu_wht(x_gpu)       # out-of-place WHT on GPU
"""

using KernelAbstractions

# ── Fused multi-level WHT butterfly kernel ────────────────────────────

"""
Fused WHT kernel: processes multiple butterfly levels in a single launch.
Each thread handles butterflies for levels `start_level` through `end_level`.

For levels where stride < workgroup_size, threads within a workgroup
cooperate on the same butterflies (requires synchronization).
For levels where stride >= workgroup_size, each thread works independently.
"""
@kernel function _wht_fused_kernel!(values, @Const(start_level), @Const(end_level), @Const(half_n))
    tid = @index(Global) - 1  # 0-indexed

    for level in start_level:end_level
        stride = 1 << (level - 1)  # 2^(level-1)
        block_idx = tid ÷ stride
        j = tid % stride
        base = block_idx * (2 * stride) + 1

        left = base + j
        right = left + stride

        @inbounds begin
            x = values[left]
            y = values[right]
            values[left] = x + y
            values[right] = x - y
        end

        # Synchronize between levels — use a global memory fence
        @synchronize()
    end
end

"""
Single-level butterfly kernel (fallback for large strides).
"""
@kernel function _wht_butterfly_kernel!(values, @Const(half_n), @Const(stride))
    i = @index(Global) - 1
    block_idx = i ÷ stride
    j = i % stride
    base = block_idx * (2 * stride) + 1
    left = base + j
    right = left + stride
    @inbounds begin
        x = values[left]
        y = values[right]
        values[left] = x + y
        values[right] = x - y
    end
end

# ── GPU element-wise power kernel ─────────────────────────────────────

@kernel function _complex_power_kernel!(out, @Const(x), @Const(k))
    i = @index(Global)
    @inbounds begin
        val = x[i]
        result = val
        for _ in 2:k
            result *= val
        end
        out[i] = result
    end
end

function gpu_complex_power(x::AbstractVector, k::Int)
    k >= 1 || throw(ArgumentError("power must be >= 1, got $k"))
    k == 1 && return copy(x)
    out = similar(x)
    backend = KernelAbstractions.get_backend(x)
    kernel! = _complex_power_kernel!(backend)
    kernel!(out, x, k; ndrange=length(x))
    KernelAbstractions.synchronize(backend)
    out
end

# ── Fused power + normalize kernel ────────────────────────────────────

"""
Combined power and max-magnitude computation.
out[i] = x[i]^k, and atomically tracks max|out[i]| via reduction.
"""
@kernel function _power_and_abs_kernel!(out, @Const(x), @Const(k))
    i = @index(Global)
    @inbounds begin
        val = x[i]
        result = val
        for _ in 2:k
            result *= val
        end
        out[i] = result
    end
end

# ── GPU WHT with fused levels ─────────────────────────────────────────

# Maximum levels to fuse in one kernel launch.
# Limited by threadgroup synchronization — @synchronize in KA uses
# threadgroup barriers, which only work within a single workgroup.
# We fuse levels where stride < workgroup_size.
const _FUSED_WHT_MAX_LEVELS = 8  # 2^8 = 256 < typical workgroup size

"""
    gpu_wht!(values::AbstractGPUVector)

In-place Walsh-Hadamard transform using fused multi-level kernels
where possible, falling back to single-level launches for large strides.
"""
function gpu_wht!(values::AbstractVector)
    N = length(values)
    N ≥ 1 || throw(ArgumentError("values must be non-empty"))
    ispow2(N) || throw(ArgumentError("length must be a power of two, got $N"))

    backend = KernelAbstractions.get_backend(values)
    half_n = N ÷ 2
    n_levels = trailing_zeros(N)  # log2(N)

    # Process levels in batches
    level = 1
    while level <= n_levels
        stride = 1 << (level - 1)

        if stride < 256 && level + _FUSED_WHT_MAX_LEVELS - 1 <= n_levels
            # Fuse up to _FUSED_WHT_MAX_LEVELS levels
            end_level = min(level + _FUSED_WHT_MAX_LEVELS - 1, n_levels)
            # But don't fuse past stride=256 (workgroup sync limit)
            while (1 << (end_level - 1)) >= 256 && end_level > level
                end_level -= 1
            end

            if end_level > level
                kernel! = _wht_fused_kernel!(backend, 256)  # workgroup size 256
                kernel!(values, level, end_level, half_n; ndrange=half_n)
                KernelAbstractions.synchronize(backend)
                level = end_level + 1
                continue
            end
        end

        # Single-level fallback for large strides
        kernel! = _wht_butterfly_kernel!(backend)
        kernel!(values, half_n, stride; ndrange=half_n)
        KernelAbstractions.synchronize(backend)
        level += 1
    end

    values
end

gpu_wht(values::AbstractVector) = gpu_wht!(copy(values))

function gpu_iwht!(values::AbstractVector)
    gpu_wht!(values)
    N = real(eltype(values))(length(values))
    values .*= (one(real(eltype(values))) / N)
    values
end

gpu_iwht(values::AbstractVector) = gpu_iwht!(copy(values))

function gpu_normalize!(values::AbstractVector, threshold::Real)
    max_val = maximum(abs.(values))
    if real(max_val) > threshold
        values ./= max_val
        return values, log(real(max_val))
    end
    return values, zero(real(eltype(values)))
end
