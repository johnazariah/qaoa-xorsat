using QaoaXorsat
using Test
using LinearAlgebra
using Random

using QaoaXorsat: apply_terms!, evolve_rk4!, evolve_euler!, x_term, z_term, xx_term, yy_term, zz_term,
    apply_heisenberg_cost_layer!, apply_mixer_layer!, plus_state, edge_xx, edge_yy, edge_zz

# ── Independent dense references (qubit 1 = least-significant bit) ────────────
const _X = ComplexF64[0 1; 1 0]
const _Y = ComplexF64[0 -im; im 0]
const _Z = ComplexF64[1 0; 0 -1]
_id(n) = Matrix{ComplexF64}(I, n, n)
_op_on(O, q, N) = reduce(kron, (_id(1 << (N - q)), O, _id(1 << (q - 1))))

function _dense_term(t::PauliTerm, N)
    t.kind === :x && return t.coeff * _op_on(_X, t.i, N)
    t.kind === :z && return t.coeff * _op_on(_Z, t.i, N)
    t.kind === :xx && return t.coeff * _op_on(_X, t.i, N) * _op_on(_X, t.j, N)
    t.kind === :yy && return t.coeff * _op_on(_Y, t.i, N) * _op_on(_Y, t.j, N)
    t.kind === :zz && return t.coeff * _op_on(_Z, t.i, N) * _op_on(_Z, t.j, N)
    error("bad kind")
end
_dense_H(terms, N) = sum(_dense_term(t, N) for t in terms)

@testset "sparse_qaoa" begin

    @testset "apply_terms! matches independent dense H·ψ" begin
        rng = MersenneTwister(1)
        N = 4
        terms = [x_term(2, 0.7), z_term(3, -0.2), xx_term(1, 3, -1.1),
            yy_term(2, 4, 0.5), zz_term(1, 2, 0.9)]
        H = _dense_H(terms, N)
        ψ = randn(rng, ComplexF64, 1 << N)
        out = similar(ψ)
        apply_terms!(out, terms, ψ, N)
        @test out ≈ H * ψ atol = 1e-12
    end

    @testset "one-body Z acts diagonally on computational basis states" begin
        N = 3
        coefficient = -0.7
        for qubit in 1:N, basis in 0:(1 << N)-1
            ψ = zeros(ComplexF64, 1 << N)
            ψ[basis+1] = 1
            out = similar(ψ)
            apply_terms!(out, [z_term(qubit, coefficient)], ψ, N)

            sign = iszero((basis >> (qubit - 1)) & 1) ? 1 : -1
            expected = zeros(ComplexF64, 1 << N)
            expected[basis+1] = coefficient * sign
            @test out == expected
        end
    end

    @testset "hamiltonian_expectation matches ψ'Hψ" begin
        rng = MersenneTwister(2)
        N = 4
        terms = heisenberg_terms([[1, 2], [2, 3], [3, 4]], HeisenbergCouplings(1.0, 0.6, -0.3))
        H = _dense_H(terms, N)
        ψ = randn(rng, ComplexF64, 1 << N)
        ψ ./= norm(ψ)
        @test hamiltonian_expectation(terms, ψ, N) ≈ real(ψ' * H * ψ) atol = 1e-12
    end

    @testset "RK4 converges to exp(-iθH) and preserves norm" begin
        rng = MersenneTwister(3)
        N = 4
        terms = heisenberg_terms([[1, 2], [2, 3], [3, 4]], HeisenbergCouplings(1.0, 1.0, 0.5))
        H = _dense_H(terms, N)
        θ = 0.7
        ψ0 = randn(rng, ComplexF64, 1 << N)
        ψ0 ./= norm(ψ0)
        exact = exp(-im * θ * H) * ψ0

        rk4 = evolve_rk4!(copy(ψ0), terms, θ, N; steps=4000)
        @test rk4 ≈ exact atol = 1e-8
        @test norm(rk4) ≈ 1.0 atol = 1e-9

        # 4th-order: halving Δs cuts error by ~16×.
        e1 = norm(evolve_rk4!(copy(ψ0), terms, θ, N; steps=50) .- exact)
        e2 = norm(evolve_rk4!(copy(ψ0), terms, θ, N; steps=100) .- exact)
        @test e2 < e1 / 8
    end

    @testset "Euler converges too (slower) " begin
        rng = MersenneTwister(4)
        N = 3
        terms = heisenberg_terms([[1, 2], [2, 3]], HeisenbergCouplings(0.5, 0.5, 0.5))
        H = _dense_H(terms, N)
        θ = 0.3
        ψ0 = randn(rng, ComplexF64, 1 << N)
        ψ0 ./= norm(ψ0)
        exact = exp(-im * θ * H) * ψ0
        euler = evolve_euler!(copy(ψ0), terms, θ, N; steps=200_000)
        @test euler ≈ exact atol = 1e-4
    end

    @testset "X-mixer leaves |+⟩^N invariant (up to phase)" begin
        N = 5
        ψ = plus_state(N)
        evolve_rk4!(ψ, x_mixer_terms(N), 1.3, N; steps=500)
        @test all(abs.(abs.(ψ) .- exp2(-N / 2)) .< 1e-8)
    end

    @testset "longitudinal-field QAOA matches dense propagation and energy" begin
        N = 3
        cost = [z_term(1, 0.7), z_term(2, -0.4), z_term(3, 1.1)]
        mixer = x_mixer_terms(N)
        angles = QAOAAngles([0.3, -0.6], [0.8, 0.2])
        Hcost = _dense_H(cost, N)
        Hmixer = _dense_H(mixer, N)

        dense = plus_state(N)
        for layer in 1:depth(angles)
            dense = exp(-im * angles.γ[layer] * Hcost) * dense
            dense = exp(-im * angles.β[layer] * Hmixer) * dense
        end
        sparse = sparse_qaoa_state(
            N, cost, mixer, angles; cost_steps=1200, mixer_steps=1200)

        @test sparse ≈ dense atol=1e-8
        @test sparse_qaoa_energy(
            N, cost, mixer, angles; cost_steps=1200, mixer_steps=1200) ≈
              real(dense' * Hcost * dense) atol=1e-8
    end

    @testset "single edge: sparse RK4 == gate-based == dense (Trotter exact)" begin
        rng = MersenneTwister(5)
        for _ in 1:4
            Jx, Jy, Jz = 2 .* rand(rng, 3) .- 1
            γ, β = 2π * rand(rng), π * rand(rng)
            J = HeisenbergCouplings(Jx, Jy, Jz)
            edge = [[1, 2]]

            # gate-based Stage 0 primitive (single-edge Trotter is exact)
            gate = plus_state(2)
            apply_heisenberg_cost_layer!(gate, edge, 2, γ, J)
            apply_mixer_layer!(gate, β, 2)
            gate_corr = (edge_xx(gate, 1, 2), edge_yy(gate, 1, 2), edge_zz(gate, 1, 2))

            # sparse matrix-free RK4 on the full (non-split) cost
            cost = heisenberg_terms(edge, J)
            ψ = sparse_qaoa_state(2, cost, x_mixer_terms(2), QAOAAngles([γ], [β]);
                cost_steps=3000, mixer_steps=3000)
            sparse_corr = (edge_xx(ψ, 1, 2), edge_yy(ψ, 1, 2), edge_zz(ψ, 1, 2))

            @test all(abs.(sparse_corr .- gate_corr) .< 1e-7)
        end
    end
end
