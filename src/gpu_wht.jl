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

# ── GPU WHT butterfly kernel ──────────────────────────────────────────

"""
Single-level butterfly kernel for the Walsh-Hadamard transform.
"""
@kernel function _wht_butterfly_kernel!(values, @Const(half_n), @Const(stride))
    i = @index(Global) - 1  # 0-indexed thread ID
    block_idx = i ÷ stride
    j = i % stride
    base = block_idx * (2 * stride) + 1  # 1-indexed
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

"""
Element-wise integer power kernel — avoids Metal's lack of Complex^Int support.
Computes out[i] = x[i]^k using repeated multiplication.
"""
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

"""
    gpu_complex_power(x::AbstractGPUVector, k::Int) -> AbstractGPUVector

Element-wise integer power for complex GPU arrays.
Works around Metal's lack of native Complex^Int support.
"""
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

"""
    gpu_wht!(values::AbstractGPUVector)

In-place Walsh-Hadamard transform on a GPU array.
The array length must be a power of 2.
"""
function gpu_wht!(values::AbstractVector)
    N = length(values)
    N ≥ 1 || throw(ArgumentError("values must be non-empty"))
    ispow2(N) || throw(ArgumentError("length must be a power of two, got $N"))

    backend = KernelAbstractions.get_backend(values)
    half_n = N ÷ 2

    # Process each butterfly level sequentially (levels are data-dependent)
    stride = 1
    while stride < N
        kernel! = _wht_butterfly_kernel!(backend)
        kernel!(values, half_n, stride; ndrange=half_n)
        KernelAbstractions.synchronize(backend)
        stride *= 2
    end

    values
end

"""
    gpu_wht(values::AbstractGPUVector)

Out-of-place Walsh-Hadamard transform on a GPU array.
"""
gpu_wht(values::AbstractVector) = gpu_wht!(copy(values))

"""
    gpu_iwht!(values::AbstractGPUVector)

In-place inverse Walsh-Hadamard transform on a GPU array.
"""
function gpu_iwht!(values::AbstractVector)
    gpu_wht!(values)
    # Use real scalar to avoid Metal ComplexF32 division IR bug
    N = real(eltype(values))(length(values))
    values .*= (one(real(eltype(values))) / N)
    values
end

"""
    gpu_iwht(values::AbstractGPUVector)

Out-of-place inverse Walsh-Hadamard transform on a GPU array.
"""
gpu_iwht(values::AbstractVector) = gpu_iwht!(copy(values))


# ── GPU utility operations ────────────────────────────────────────────

"""
    gpu_normalize!(values, threshold) -> (values, log_scale)

Normalize a GPU array if max magnitude exceeds threshold.
Returns the array and the log of the scale factor applied (0.0 if no normalization).
"""
function gpu_normalize!(values::AbstractVector, threshold::Real)
    max_val = maximum(abs.(values))
    if real(max_val) > threshold
        values ./= max_val
        return values, log(real(max_val))
    end
    return values, zero(real(eltype(values)))
end
