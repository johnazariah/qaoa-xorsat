# Robustness / property-based hardening for the matrix-free statevector engine.
#
# Complements test_sparse_qaoa.jl. Encodes the ADR-0012 stance (code correctness
# by tests, not designated review) and the SPEC-pfqe.md §11 validation ladder:
# energy linearity over terms, full-layer unitarity, spectrum bounds, the Ising
# limit, end-to-end agreement with the dense propagator, symmetry conservation,
# edge cases, determinism, and Hermiticity. Fixed seeds and a fixed 3-regular
# graph keep every check deterministic.

using QaoaXorsat
using Test
using LinearAlgebra
using Random

using QaoaXorsat: x_term, xx_term, yy_term, zz_term, plus_state, evolve_rk4!, edge_zz,
    HeisenbergCouplings, heisenberg_terms, x_mixer_terms, xy_mixer_terms, swap_mixer_terms,
    apply_terms!, hamiltonian_expectation, sparse_qaoa_state, sparse_qaoa_energy,
    QAOAAngles, PauliTerm

# ── Independent dense references (qubit 1 = least-significant bit), rb-prefixed
#    to avoid clashing with the identically-shaped helpers in test_sparse_qaoa.jl.
const rbX = ComplexF64[0 1; 1 0]
const rbY = ComplexF64[0 -im; im 0]
const rbZ = ComplexF64[1 0; 0 -1]
rbid(n) = Matrix{ComplexF64}(I, n, n)
rbon(O, q, N) = reduce(kron, (rbid(1 << (N - q)), O, rbid(1 << (q - 1))))
function rbterm(t::PauliTerm, N)
    t.kind === :x && return t.coeff * rbon(rbX, t.i, N)
    t.kind === :xx && return t.coeff * rbon(rbX, t.i, N) * rbon(rbX, t.j, N)
    t.kind === :yy && return t.coeff * rbon(rbY, t.i, N) * rbon(rbY, t.j, N)
    t.kind === :zz && return t.coeff * rbon(rbZ, t.i, N) * rbon(rbZ, t.j, N)
    error("bad kind")
end
rbH(terms, N) = isempty(terms) ? zeros(ComplexF64, 1 << N, 1 << N) :
                sum(rbterm(t, N) for t in terms)

# Triangular prism: a fixed simple 3-regular graph on N=6 (each vertex degree 3).
const PRISM = [[1, 2], [2, 3], [1, 3], [4, 5], [5, 6], [4, 6], [1, 4], [2, 5], [3, 6]]

# Total magnetization ⟨Σ Zᵢ⟩ from the amplitudes (Z=+1 for bit 0, -1 for bit 1).
rbmag(ψ, N) = sum(abs2(ψ[b+1]) * (N - 2 * count_ones(b)) for b in 0:length(ψ)-1)

@testset "sparse_qaoa robustness" begin

    @testset "energy is linear over terms  ⟨H⟩ = Σ ⟨H_j⟩" begin
        rng = MersenneTwister(11)
        N = 5
        terms = heisenberg_terms([[1, 2], [2, 3], [3, 4], [4, 5], [1, 5]],
            HeisenbergCouplings(1.0, 0.7, -0.4))
        push!(terms, x_term(3, 0.5))
        ψ = randn(rng, ComplexF64, 1 << N)
        ψ ./= norm(ψ)
        total = hamiltonian_expectation(terms, ψ, N)
        piecewise = sum(hamiltonian_expectation([t], ψ, N) for t in terms)
        @test total ≈ piecewise atol = 1e-12
    end

    @testset "apply_terms! acts Hermitian  ⟨φ|Hψ⟩ = conj⟨ψ|Hφ⟩" begin
        rng = MersenneTwister(18)
        N = 4
        terms = heisenberg_terms([[1, 2], [2, 3], [3, 4]], HeisenbergCouplings(1.0, 0.7, -0.4))
        push!(terms, x_term(2, 0.5))
        φ = randn(rng, ComplexF64, 1 << N)
        ψ = randn(rng, ComplexF64, 1 << N)
        Hψ = similar(ψ); apply_terms!(Hψ, terms, ψ, N)
        Hφ = similar(φ); apply_terms!(Hφ, terms, φ, N)
        @test dot(φ, Hψ) ≈ conj(dot(ψ, Hφ)) atol = 1e-12
    end

    @testset "full QAOA layers preserve the norm (unitarity)" begin
        rng = MersenneTwister(12)
        N = 6
        for J in (HeisenbergCouplings(1.0, 1.0, 1.0),
            HeisenbergCouplings(1.0, 1.0, 0.5),
            HeisenbergCouplings(0.0, 0.0, 1.0))
            cost = heisenberg_terms(PRISM, J)
            for _ in 1:3
                p = 3
                γ = 2π .* rand(rng, p)
                β = π .* rand(rng, p)
                ψ = sparse_qaoa_state(N, cost, x_mixer_terms(N), QAOAAngles(γ, β))
                # default step count targets ~1e-4 state accuracy; the norm is
                # preserved to a comparable order, so assert 1e-5.
                @test norm(ψ) ≈ 1.0 atol = 1e-5
            end
        end
    end

    @testset "cost energy lies within the spectrum" begin
        N = 6
        cost = heisenberg_terms(PRISM, HeisenbergCouplings(1.0, 1.0, 1.0))
        λ = eigvals(Hermitian(rbH(cost, N)))
        lo, hi = minimum(real, λ), maximum(real, λ)
        rng = MersenneTwister(13)
        for _ in 1:5
            γ = 2π .* rand(rng, 2)
            β = π .* rand(rng, 2)
            E = sparse_qaoa_energy(N, cost, x_mixer_terms(N), QAOAAngles(γ, β))
            @test lo - 1e-6 ≤ E ≤ hi + 1e-6
        end
    end

    @testset "Ising limit (0,0,1): energy = Σ ⟨ZᵢZⱼ⟩ (diagonal observable)" begin
        N = 6
        cost = heisenberg_terms(PRISM, HeisenbergCouplings(0.0, 0.0, 1.0))
        @test all(t.kind === :zz for t in cost)          # Jx=Jy dropped by the builder
        @test length(cost) == length(PRISM)
        γ = [0.4, 0.9]; β = [0.8, 0.3]
        ψ = sparse_qaoa_state(N, cost, x_mixer_terms(N), QAOAAngles(γ, β))
        direct = sum(edge_zz(ψ, e[1], e[2]) for e in PRISM)
        @test hamiltonian_expectation(cost, ψ, N) ≈ direct atol = 1e-9
    end

    @testset "full QAOA state matches the dense propagator (3-regular, N=6)" begin
        N = 6
        cost = heisenberg_terms(PRISM, HeisenbergCouplings(1.0, 1.0, 1.0))
        mix = x_mixer_terms(N)
        Hc = rbH(cost, N)
        Hb = rbH(mix, N)
        γ = [0.4, 0.9]; β = [0.8, 0.3]
        dense = plus_state(N)
        for r in 1:2
            dense = exp(-im * γ[r] * Hc) * dense
            dense = exp(-im * β[r] * Hb) * dense
        end
        ψ = sparse_qaoa_state(N, cost, mix, QAOAAngles(γ, β); cost_steps=1500, mixer_steps=1500)
        @test ψ ≈ dense atol = 1e-6
        @test hamiltonian_expectation(cost, ψ, N) ≈ real(dense' * Hc * dense) atol = 1e-6
    end

    @testset "XY and SWAP mixers conserve magnetization; X does not" begin
        N = 6
        ψ0 = zeros(ComplexF64, 1 << N)
        ψ0[2] = 1.0                                       # basis state b=1: one flipped spin
        m0 = rbmag(ψ0, N)                                 # = N - 2 = 4
        for mix in (xy_mixer_terms(PRISM), swap_mixer_terms(PRISM))
            ψ = evolve_rk4!(copy(ψ0), mix, 0.9, N; steps=600)
            @test norm(ψ) ≈ 1.0 atol = 1e-7
            @test rbmag(ψ, N) ≈ m0 atol = 1e-6           # stays in the fixed-weight sector
        end
        ψx = evolve_rk4!(copy(ψ0), x_mixer_terms(N), 0.9, N; steps=600)
        @test abs(rbmag(ψx, N) - m0) > 1e-2              # X leaves the sector
    end

    @testset "determinism: identical inputs give identical output" begin
        N = 6
        cost = heisenberg_terms(PRISM, HeisenbergCouplings(1.0, 1.0, 0.5))
        a = sparse_qaoa_state(N, cost, x_mixer_terms(N), QAOAAngles([0.3, 0.7], [0.9, 0.2]))
        b = sparse_qaoa_state(N, cost, x_mixer_terms(N), QAOAAngles([0.3, 0.7], [0.9, 0.2]))
        @test a == b
    end

    @testset "edge cases and input validation" begin
        N = 4
        rng = MersenneTwister(17)
        ψ = randn(rng, ComplexF64, 1 << N); ψ ./= norm(ψ)

        # empty Hamiltonian: zero energy, zero action
        @test hamiltonian_expectation(PauliTerm[], ψ, N) == 0.0
        out = similar(ψ); apply_terms!(out, PauliTerm[], ψ, N)
        @test all(iszero, out)

        cost = heisenberg_terms([[1, 2], [2, 3]], HeisenbergCouplings(1.0, 1.0, 1.0))

        # θ = 0 evolution is the identity
        @test evolve_rk4!(copy(ψ), cost, 0.0, N) == ψ

        # a zero-angle QAOA layer returns |+⟩^N untouched
        ψ0 = sparse_qaoa_state(N, cost, x_mixer_terms(N), QAOAAngles([0.0], [0.0]))
        @test ψ0 ≈ plus_state(N) atol = 1e-12

        # invalid integrator method rejected
        @test_throws ArgumentError sparse_qaoa_state(
            N, cost, x_mixer_terms(N), QAOAAngles([0.1], [0.1]); method=:bogus)
        # unknown Pauli kind rejected
        @test_throws ArgumentError apply_terms!(similar(ψ), [PauliTerm(:zzz, 1, 2, 1.0)], ψ, N)
        # malformed angles rejected
        @test_throws ArgumentError QAOAAngles([0.1, 0.2], [0.1])
        @test_throws ArgumentError QAOAAngles(Float64[], Float64[])
    end
end
