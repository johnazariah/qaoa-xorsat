using QaoaXorsat
using Test
using LinearAlgebra
using Random

# Non-exported helpers used as an independent oracle / for gate-level checks.
using QaoaXorsat: apply_zz_layer!, apply_heisenberg_cost_layer!, apply_mixer_layer!,
    plus_state, edge_xx, edge_yy, edge_zz, reference_parity_expectation

@testset "heisenberg_statevector" begin

    @testset "sanity: zero angles give |+⟩^L" begin
        # γ=β=0 ⇒ identity circuit ⇒ |+⟩^L ⇒ ⟨XX⟩=1, ⟨YY⟩=⟨ZZ⟩=0.
        angles = QAOAAngles([0.0], [0.0])
        J = HeisenbergCouplings(1.0, 1.0, 1.0)
        xx, yy, zz = heisenberg_edge_correlators(angles, J)
        @test xx ≈ 1.0 atol = 1e-12
        @test yy ≈ 0.0 atol = 1e-12
        @test zz ≈ 0.0 atol = 1e-12
    end

    @testset "correlators are real and bounded" begin
        rng = MersenneTwister(20260702)
        J = HeisenbergCouplings(-1.0, -0.7, -0.4)
        @testset "p=$p" for p in 1:2
            angles = QAOAAngles(2π .* rand(rng, p), 2π .* rand(rng, p))
            xx, yy, zz = heisenberg_edge_correlators(angles, J)
            for c in (xx, yy, zz)
                @test -1.0 - 1e-10 ≤ c ≤ 1.0 + 1e-10
            end
        end
    end

    @testset "reduction: Jx=Jy=0 dressing cancels to bare Ising" begin
        # doc 23 §4/§8.0.6 regression: with Jx=Jy=0 the Clifford dressing must
        # cancel, leaving U_ZZ(γ Jz) + mixer. Compare full state to bare state.
        rng = MersenneTwister(11)
        Jz = -1.0
        J = HeisenbergCouplings(0.0, 0.0, Jz)
        @testset "p=$p" for p in 1:3
            γ = 2π .* rand(rng, p)
            β = 2π .* rand(rng, p)
            angles = QAOAAngles(γ, β)
            L = exact_chain_length(p)

            full = heisenberg_chain_state(angles, J; L)

            bare = plus_state(L)
            edges = chain_edges(L)
            for r in 1:p
                apply_zz_layer!(bare, edges, γ[r] * Jz)
                apply_mixer_layer!(bare, β[r], L)
            end

            @test maximum(abs.(full .- bare)) < 1e-12
        end
    end

    @testset "cross-engine: bare Ising ⟨ZZ⟩ matches the Basso reference" begin
        # Convention map (see comment in derivation): reference clause_sign=-1
        # applies e^{+i·0.5·γ_ref·ΣZZ}; our U_ZZ(γ Jz) with Jz=-1 applies
        # e^{+i·γ_mine·ΣZZ}. So γ_mine = 0.5·γ_ref, β_mine = β_ref.
        rng = MersenneTwister(2718)
        J = HeisenbergCouplings(0.0, 0.0, -1.0)
        @testset "p=$p" for p in 1:3
            γ_ref = 2π .* rand(rng, p)
            β_ref = 2π .* rand(rng, p)
            ref = reference_parity_expectation(
                TreeParams(2, 2, p), QAOAAngles(γ_ref, β_ref); clause_sign=-1)

            angles = QAOAAngles(0.5 .* γ_ref, β_ref)
            _, _, zz = heisenberg_edge_correlators(angles, J)

            @test zz ≈ ref atol = 1e-9
        end
    end

    @testset "single edge: Trotter split is exact (dimer vs dense)" begin
        # On one edge XX, YY, ZZ commute, so the Pauli-split layer equals
        # exp(-iγ H_edge). Validate the gate sequence against dense matrices.
        rng = MersenneTwister(99)
        X = ComplexF64[0 1; 1 0]
        Y = ComplexF64[0 -im; im 0]
        Z = ComplexF64[1 0; 0 -1]
        Id = ComplexF64[1 0; 0 1]
        XX, YY, ZZ = kron(X, X), kron(Y, Y), kron(Z, Z)
        plus = ComplexF64[1, 1] / sqrt(2)
        ψ0 = kron(plus, plus)

        for _ in 1:5
            Jx, Jy, Jz = 2 .* rand(rng, 3) .- 1
            γ = 2π * rand(rng)
            β = 2π * rand(rng)
            J = HeisenbergCouplings(Jx, Jy, Jz)

            Ucost = exp(-im * γ * (Jx * XX + Jy * YY + Jz * ZZ))
            Umix = exp(-im * β * (kron(X, Id) + kron(Id, X)))
            ψ = Umix * Ucost * ψ0
            dense = (real(ψ' * XX * ψ), real(ψ' * YY * ψ), real(ψ' * ZZ * ψ))

            state = plus_state(2)
            edges = chain_edges(2)
            apply_heisenberg_cost_layer!(state, edges, 2, γ, J)
            apply_mixer_layer!(state, β, 2)
            sim = (edge_xx(state, 1, 2), edge_yy(state, 1, 2), edge_zz(state, 1, 2))

            @test all(abs.(sim .- dense) .< 1e-10)
        end
    end

    @testset "cone exactness: ε independent of L beyond 6p+2" begin
        rng = MersenneTwister(7)
        J = HeisenbergCouplings(-1.0, -1.0, -0.5)
        @testset "p=$p" for p in 1:2
            angles = QAOAAngles(2π .* rand(rng, p), 2π .* rand(rng, p))
            ε_min = heisenberg_energy_density(angles, J; L=exact_chain_length(p))
            ε_pad = heisenberg_energy_density(angles, J; L=exact_chain_length(p) + 2)
            @test ε_min ≈ ε_pad atol = 1e-10
        end
    end

    @testset "guards" begin
        angles = QAOAAngles([0.1], [0.2])
        @test_throws ArgumentError heisenberg_chain_state(
            angles, HeisenbergCouplings(0, 0, -1); L=4)  # L < 6p+2
    end
end
