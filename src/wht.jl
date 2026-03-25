"""
    wht!(values)

Apply the in-place Walsh-Hadamard transform to a vector of length `2^n`.

The transform uses the XOR-character basis on `(ℤ₂^n, ⊕)`:

`ĝ(s) = Σₓ g(x) (-1)^{⟨s, x⟩}`.

Uses `@inbounds @simd` for the inner butterfly loop, which gives significant
speedups at large N by enabling SIMD vectorization of the complex add/subtract.
"""
function wht!(values::AbstractVector)
    N = length(values)
    N ≥ 1 || throw(ArgumentError("values must be non-empty"))
    ispow2(N) || throw(ArgumentError("length must be a power of two"))

    block = 1
    @inbounds while block < N
        stride = 2 * block
        for base in 1:stride:N
            @simd for offset in 0:(block-1)
                left = base + offset
                right = left + block
                x = values[left]
                y = values[right]
                values[left] = x + y
                values[right] = x - y
            end
        end
        block = stride
    end

    values
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
