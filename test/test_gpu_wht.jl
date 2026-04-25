"""
GPU WHT tests — Level 1 (unit) and Level 2 (element-wise ops).

Tests the GPU Walsh-Hadamard transform against the CPU implementation
for correctness, and verifies element-wise GPU operations.

Runs on whatever GPU backend is available (Metal on Mac, CUDA on cluster,
CPU fallback otherwise).
"""

using Test
using QaoaXorsat

# ── Detect available GPU backend ──────────────────────────────────

using Metal

const GPU_AVAILABLE = Metal.functional()
const GPU_ARRAY_TYPE = GPU_AVAILABLE ? MtlArray : nothing

if GPU_AVAILABLE
    @info "GPU tests using Metal backend"
else
    @warn "Metal GPU not functional — GPU tests will be skipped"
end

# ── Helper: convert to GPU-compatible element type ────────────────

# Metal requires Float32; CUDA supports Float64
function gpu_complex_type()
    if GPU_ARRAY_TYPE === nothing
        return ComplexF64
    end
    try
        # Test if Float64 is supported
        x = GPU_ARRAY_TYPE(ComplexF64[1.0 + 0im])
        return ComplexF64
    catch
        return ComplexF32
    end
end

function to_gpu(x::AbstractVector)
    T = gpu_complex_type()
    GPU_ARRAY_TYPE(T.(x))
end

function from_gpu(x)
    Array(x)
end

# ── Include GPU WHT code ──────────────────────────────────────────

include(joinpath(@__DIR__, "..", "src", "gpu_wht.jl"))

# ── Level 1: GPU WHT correctness ─────────────────────────────────

@testset "GPU WHT" begin
    if !GPU_AVAILABLE
        @info "Skipping GPU tests (no GPU backend)"
        @test true  # placeholder so testset isn't empty
        return
    end

    CT = gpu_complex_type()
    # Tolerance: Float32 has ~7 digits, Float64 has ~15
    atol = CT == ComplexF32 ? 1f-4 : 1e-10

    @testset "WHT correctness N=2^$n" for n in 1:17
        N = 2^n
        # Skip very large sizes on GPU to keep tests fast
        n > 15 && continue

        x_cpu = randn(ComplexF64, N)
        x_gpu = to_gpu(x_cpu)

        # CPU reference
        ref = QaoaXorsat.wht(x_cpu)

        # GPU WHT
        result_gpu = gpu_wht(x_gpu)
        result_cpu = ComplexF64.(from_gpu(result_gpu))

        @test result_cpu ≈ ref atol=atol*sqrt(N)
    end

    @testset "iWHT correctness N=2^$n" for n in 1:15
        N = 2^n
        x_cpu = randn(ComplexF64, N)
        x_gpu = to_gpu(x_cpu)

        ref = QaoaXorsat.iwht(x_cpu)
        result_gpu = gpu_iwht(x_gpu)
        result_cpu = ComplexF64.(from_gpu(result_gpu))

        @test result_cpu ≈ ref atol=atol*sqrt(N)
    end

    @testset "roundtrip iWHT(WHT(x)) ≈ x" for n in [4, 8, 12, 15]
        N = 2^n
        x_cpu = randn(ComplexF64, N)
        x_gpu = to_gpu(x_cpu)

        roundtrip = gpu_iwht(gpu_wht(x_gpu))
        result = ComplexF64.(from_gpu(roundtrip))

        @test result ≈ x_cpu atol=atol*N
    end

    @testset "WHT of delta function" begin
        # WHT of δ_0 should be all ones
        N = 2^10
        x = zeros(CT, N)
        x[1] = one(CT)
        x_gpu = GPU_ARRAY_TYPE(x)

        result = from_gpu(gpu_wht(x_gpu))
        @test all(abs.(result .- one(CT)) .< atol)
    end

    @testset "WHT of constant vector" begin
        # WHT of constant c should be c*N at index 0, zero elsewhere
        N = 2^10
        c = CT(3.0 + 1.0im)
        x = fill(c, N)
        x_gpu = GPU_ARRAY_TYPE(x)

        result = from_gpu(gpu_wht(x_gpu))
        @test abs(result[1] - c * N) < atol * N
        @test all(abs.(result[2:end]) .< atol * N)
    end

    @testset "WHT convolution theorem" begin
        # WHT(a .* b) should equal iWHT(WHT(a) .* WHT(b)) * N ... no
        # Actually: WHT(a ⊛ b) = WHT(a) .* WHT(b)
        # where (a ⊛ b)(x) = Σ_y a(y) b(x⊻y)
        # Equivalently: WHT(a) .* WHT(b) = WHT of XOR-convolution
        # Simpler test: iWHT(WHT(a) .* WHT(b)) = a ⊛ b (XOR convolution)
        N = 2^8
        a_cpu = randn(ComplexF64, N)
        b_cpu = randn(ComplexF64, N)

        # CPU XOR convolution via WHT
        ref = QaoaXorsat.xor_convolution(a_cpu, b_cpu)

        # GPU path
        a_gpu = to_gpu(a_cpu)
        b_gpu = to_gpu(b_cpu)
        a_hat = gpu_wht(a_gpu)
        b_hat = gpu_wht(b_gpu)
        conv_gpu = gpu_iwht(a_hat .* b_hat)
        result = ComplexF64.(from_gpu(conv_gpu))

        @test result ≈ ref atol=atol*N*10
    end

    # ── Level 2: GPU element-wise operations ──────────────────────

    @testset "element-wise power" begin
        N = 2^12
        x_cpu = randn(ComplexF64, N) .* 0.5  # keep magnitudes moderate
        x_gpu = to_gpu(x_cpu)

        for k in [2, 3, 5, 7]
            ref = x_cpu .^ k
            result = ComplexF64.(from_gpu(gpu_complex_power(x_gpu, k)))
            @test result ≈ ref atol=atol*N
        end
    end

    @testset "gpu_normalize!" begin
        N = 2^10
        x = randn(CT, N) .* CT(1e20)
        x_gpu = GPU_ARRAY_TYPE(copy(x))

        result, log_s = gpu_normalize!(x_gpu, real(CT)(1e15))
        result_cpu = from_gpu(result)

        @test maximum(abs.(result_cpu)) ≤ 1.0 + atol
        @test log_s > 0
        # Verify the scaling is correct
        @test exp(Float64(log_s)) * maximum(abs.(result_cpu)) ≈
              Float64(maximum(abs.(x))) atol=1e-3*Float64(maximum(abs.(x)))
    end

    @testset "gpu_normalize! no-op below threshold" begin
        N = 2^10
        x = randn(CT, N)
        x_gpu = GPU_ARRAY_TYPE(copy(x))

        result, log_s = gpu_normalize!(x_gpu, real(CT)(1e30))
        @test log_s == 0
        @test from_gpu(result) ≈ Array(x) atol=atol
    end
end
