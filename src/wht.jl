"""
    wht!(values)

Apply the in-place Walsh-Hadamard transform to a vector of length `2^n`.

The transform uses the XOR-character basis on `(ℤ₂^n, ⊕)`:

`ĝ(s) = Σₓ g(x) (-1)^{⟨s, x⟩}`.

Uses a recursive cache-oblivious decomposition: for large vectors, the transform
splits into a butterfly merge followed by two independent sub-transforms on each
half. Sub-problems that fit in L1 cache (~32KB) use the iterative SIMD kernel.
Since WHT butterfly levels operate on independent bit positions, the
level-ordering is free — the recursive and iterative approaches produce
identical results.
"""
function wht!(values::AbstractVector)
    N = length(values)
    N ≥ 1 || throw(ArgumentError("values must be non-empty"))
    ispow2(N) || throw(ArgumentError("length must be a power of two"))
    _wht_recursive!(values, 1, N)
    values
end

# Cutoff: sub-problems of this size or smaller use the iterative kernel.
# 2048 ComplexF64 elements = 32KB, fits comfortably in L1 cache on all targets
# (M4 Apple Silicon: 192KB L1, Intel/AMD Xeon: 32-48KB L1).
const _WHT_RECURSIVE_CUTOFF = 2048

"""
Recursive cache-oblivious WHT on values[offset : offset+n-1].
Splits the transform at the top butterfly level, then recurses on each half.
"""
function _wht_recursive!(values::AbstractVector, offset::Int, n::Int)
    if n ≤ _WHT_RECURSIVE_CUTOFF
        _wht_iterative!(values, offset, n)
        return
    end
    half = n >> 1
    # Top-level butterfly: combine the two halves
    @inbounds @simd for i in 0:half-1
        left = offset + i
        right = left + half
        x = values[left]
        y = values[right]
        values[left] = x + y
        values[right] = x - y
    end
    # Recurse on each half — sub-problems stay in cache longer
    _wht_recursive!(values, offset, half)
    _wht_recursive!(values, offset + half, half)
end

"""
Iterative SIMD WHT kernel for contiguous sub-arrays. Used as the base case
of the recursive decomposition.
"""
function _wht_iterative!(values::AbstractVector, offset::Int, n::Int)
    block = 1
    @inbounds while block < n
        stride = 2 * block
        for base in offset:stride:(offset + n - 1)
            @simd for j in 0:(block-1)
                left = base + j
                right = left + block
                x = values[left]
                y = values[right]
                values[left] = x + y
                values[right] = x - y
            end
        end
        block = stride
    end
end

"""Out-of-place Walsh-Hadamard transform."""
wht(values::AbstractVector) = wht!(copy(values))

"""
    iwht!(values)

Apply the inverse Walsh-Hadamard transform in place.
"""
function iwht!(values::AbstractVector)
    wht!(values)
    values ./= length(values)
end

"""Out-of-place inverse Walsh-Hadamard transform."""
iwht(values::AbstractVector) = iwht!(copy(values))

"""
    xor_convolution(left, right)

Compute the convolution on `(ℤ₂^n, ⊕)`:

`(left ★ right)(x) = Σ_y left(y) right(x ⊻ y)`.
"""
function xor_convolution(left::AbstractVector, right::AbstractVector)
    length(left) == length(right) || throw(ArgumentError("convolution inputs must have equal length"))
    iwht(wht(left) .* wht(right))
end

"""
    xor_autoconvolution(values)

Compute `values ★ values` on `(ℤ₂^n, ⊕)`.
"""
xor_autoconvolution(values::AbstractVector) = iwht(wht(values) .^ 2)

"""
    xor_convolution_power(values, exponent)

Compute the repeated XOR-convolution power

`values ★ values ★ ⋯ ★ values`

with `exponent` factors on `(ℤ₂^n, ⊕)`.
"""
function xor_convolution_power(values::AbstractVector, exponent::Int)
    exponent ≥ 1 || throw(ArgumentError("exponent must be ≥ 1, got $exponent"))
    iwht(wht(values) .^ exponent)
end
