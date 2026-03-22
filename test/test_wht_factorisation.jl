using QaoaXorsat
using Test

# ── WHT primitives ───────────────────────────────────────────────────────────

"""
    wht!(v)

In-place Walsh-Hadamard transform on a vector of length 2^n.
Computes ĝ(s) = Σ_x g(x) (-1)^{⟨s,x⟩}.
"""
function wht!(v::AbstractVector)
    N = length(v)
    @assert ispow2(N) "length must be a power of 2"
    h = 1
    while h < N
        for i in 0:2h:N-1
            for j in 0:h-1
                x = v[i+j+1]
                y = v[i+j+h+1]
                v[i+j+1] = x + y
                v[i+j+h+1] = x - y
            end
        end
        h *= 2
    end
    v
end

"""Out-of-place Walsh-Hadamard transform."""
wht(v::AbstractVector) = wht!(copy(v))

"""Inverse WHT: IWHT(v) = (1/N) WHT(v)."""
iwht(v::AbstractVector) = wht(v) ./ length(v)

# ── Helpers ──────────────────────────────────────────────────────────────────

"""
    build_gamma_full(angles)

Build the (2p+1)-component Γ vector for the constraint fold:
Γ = (γ₁, …, γₚ, 0, -γₚ, …, -γ₁).
"""
function build_gamma_full(angles::QAOAAngles)
    p = depth(angles)
    Γ = zeros(Float64, 2p + 1)
    for r in 1:p
        Γ[r] = angles.γ[r]
        Γ[2p + 2 - r] = -angles.γ[r]
    end
    Γ
end

"""
    to_spins(config, n)

Convert integer configuration to {-1,+1}^n spin vector (LSB first).
Bit 0 → spin +1, bit 1 → spin -1.
"""
function to_spins(config::Integer, n::Int)
    [1 - 2 * ((config >> (i - 1)) & 1) for i in 1:n]
end

"""Build g(b) = f(b) for all 2^(2p+1) branch configurations (first iteration)."""
function build_g_vector(angles::QAOAAngles)
    N = QaoaXorsat.basso_configuration_count(depth(angles))
    [QaoaXorsat.f_function(angles, c) for c in 0:N-1]
end

"""Build κ(d) = cos(Γ_full · spins(d) / √D) for all d."""
function build_kappa_vector(angles::QAOAAngles, D::Int)
    p = depth(angles)
    n = QaoaXorsat.basso_bit_count(p)
    N = QaoaXorsat.basso_configuration_count(p)
    Γ = build_gamma_full(angles)
    sqrtD = sqrt(D)
    [cos(sum(Γ .* to_spins(d, n)) / sqrtD) for d in 0:N-1]
end

# ── Naive constraint fold ────────────────────────────────────────────────────

"""
    naive_constraint_fold(angles, D)

Brute-force constraint fold for k=3:
S(a) = Σ_{b1,b2} cos(Γ·(a⊙b1⊙b2)/√D) · g(b1) · g(b2)

Uses bitwise XOR (≡ spin-wise ⊙) to compute the three-way parity product.
"""
function naive_constraint_fold(angles::QAOAAngles, D::Int)
    N = QaoaXorsat.basso_configuration_count(depth(angles))
    g = build_g_vector(angles)
    κ = build_kappa_vector(angles, D)

    S = zeros(ComplexF64, N)
    for a in 0:N-1
        s = zero(ComplexF64)
        for b1 in 0:N-1
            for b2 in 0:N-1
                d = xor(a, xor(b1, b2))
                s += κ[d+1] * g[b1+1] * g[b2+1]
            end
        end
        S[a+1] = s
    end
    S
end

# ── WHT-based constraint fold ───────────────────────────────────────────────

"""
    wht_constraint_fold(angles, D)

Walsh-Hadamard factorised constraint fold for k=3:
1. ĝ = WHT(g),  Ŵ = ĝ²           (auto-convolution in WHT domain)
2. κ̂ = WHT(κ)                      (transform cosine kernel)
3. S = IWHT(κ̂ · Ŵ)                 (convolution theorem)
"""
function wht_constraint_fold(angles::QAOAAngles, D::Int)
    g = build_g_vector(angles)
    κ = ComplexF64.(build_kappa_vector(angles, D))

    ĝ = wht(g)
    Ŵ = ĝ .^ 2
    κ̂ = wht(κ)

    iwht(κ̂ .* Ŵ)
end

# ── Tests ────────────────────────────────────────────────────────────────────

@testset "WHT Factorisation Verification" begin

    @testset "WHT sanity checks" begin
        @testset "round-trip n=$n" for n in [3, 5]
            N = 1 << n
            v = randn(ComplexF64, N)
            @test iwht(wht(v)) ≈ v atol = 1e-12
        end

        @testset "convolution theorem" begin
            N = 8
            f = randn(ComplexF64, N)
            g = randn(ComplexF64, N)

            # Direct Z₂ⁿ convolution: (f★g)(x) = Σ_y f(y) g(x⊕y)
            conv_direct = zeros(ComplexF64, N)
            for x in 0:N-1
                for y in 0:N-1
                    conv_direct[x+1] += f[y+1] * g[xor(x, y)+1]
                end
            end

            conv_wht = iwht(wht(f) .* wht(g))
            @test conv_wht ≈ conv_direct atol = 1e-10
        end

        @testset "autoconvolution theorem" begin
            N = 8
            g = randn(ComplexF64, N)

            # Direct: W(c) = Σ_b g(b) g(b⊕c)
            W_direct = zeros(ComplexF64, N)
            for c in 0:N-1
                for b in 0:N-1
                    W_direct[c+1] += g[b+1] * g[xor(b, c)+1]
                end
            end

            ĝ = wht(g)
            W_wht = iwht(ĝ .^ 2)
            @test W_wht ≈ W_direct atol = 1e-10
        end
    end

    @testset "naive vs WHT — p=$p, D=$D" for (p, D, n_trials) in [
        (1, 3, 100),
        (1, 4, 50),
        (2, 3, 20),
        (2, 4, 10),
        (3, 3, 5),
        (3, 4, 3),
    ]
        max_err = 0.0
        @testset "trial $t" for t in 1:n_trials
            γ = π .* rand(p)
            β = (π / 2) .* rand(p)
            angles = QAOAAngles(γ, β)

            S_naive = naive_constraint_fold(angles, D)
            S_wht = wht_constraint_fold(angles, D)
            err = maximum(abs.(S_naive .- S_wht))
            max_err = max(max_err, err)

            @test S_wht ≈ S_naive atol = 1e-10
        end
        @info "p=$p, D=$D: max |S_naive - S_wht| = $max_err over $n_trials trials"
    end
end
