using QaoaXorsat
using Test

@testset "Reduced basis iteration" begin
    @testset "ReducedBasis construction" begin
        @testset "p=$p" for p in 1:5
            basis = ReducedBasis(p)
            @test basis.M == 1 << (2p - 1)
            @test basis.N == 1 << (2p + 1)
            @test basis.M * 4 == basis.N
            @test length(basis.free_positions) == 2p - 1
        end

        @test_throws ArgumentError ReducedBasis(0)
    end

    @testset "index mapping roundtrip" begin
        @testset "p=$p" for p in 1:4
            basis = ReducedBasis(p)
            # Every reduced index maps to a valid full index and back
            for j in 0:basis.M-1
                full = QaoaXorsat.reduced_to_full(basis, j)
                # Coset rep has bit 0 = 0 and bit p = 0
                @test (full & 1) == 0
                @test ((full >> p) & 1) == 0
                @test full < basis.N
            end
            # All M representatives are distinct
            reps = [QaoaXorsat.reduced_to_full(basis, j) for j in 0:basis.M-1]
            @test length(unique(reps)) == basis.M
        end
    end

    @testset "coset coverage" begin
        @testset "p=$p" for p in 1:3
            basis = ReducedBasis(p)
            covered = Set{Int}()
            for j in 0:basis.M-1
                r = QaoaXorsat.reduced_to_full(basis, j)
                for v in QaoaXorsat.coset_elements(basis, r)
                    push!(covered, v)
                end
            end
            # Every configuration in 0:N-1 is covered exactly once
            @test length(covered) == basis.N
            @test covered == Set(0:basis.N-1)
        end
    end

    @testset "reduce and expand roundtrip" begin
        @testset "p=$p" for p in 1:4
            basis = ReducedBasis(p)
            # Build a symmetric full vector (constant on cosets)
            reduced = randn(ComplexF64, basis.M)
            full = expand_symmetric(reduced, basis)
            recovered = QaoaXorsat.reduce_sample(full, basis)
            @test recovered ≈ reduced atol = 1e-14
        end
    end

    @testset "branch tensor symmetry verification" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [(2, 3, 2), (3, 4, 2), (3, 4, 3), (4, 5, 2)]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(rand(p), rand(p))
            basis = ReducedBasis(p)

            B_full = QaoaXorsat.basso_branch_tensor(params, angles)

            # Root-bit independence
            for j in 0:basis.M-1
                r = QaoaXorsat.reduced_to_full(basis, j)
                @test B_full[r+1] ≈ B_full[(r ⊻ basis.root_mask)+1] atol = 1e-12
            end

            # Complement invariance
            for j in 0:basis.M-1
                r = QaoaXorsat.reduced_to_full(basis, j)
                @test B_full[r+1] ≈ B_full[(r ⊻ basis.complement_mask)+1] atol = 1e-12
            end
        end
    end

    @testset "reduced branch tensor matches full" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (3, 4, 1), (3, 4, 2), (3, 4, 3), (3, 4, 4),
            (4, 5, 2), (5, 6, 2),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(rand(p), rand(p))

            f_table = QaoaXorsat.basso_f_table(angles)
            B_full = QaoaXorsat.basso_branch_tensor(params, angles; f_table)

            B_red, basis = basso_branch_tensor_reduced(params, angles; f_table)
            B_expanded = expand_symmetric(B_red, basis)

            @test B_expanded ≈ B_full atol = 1e-10
        end
    end

    @testset "reduced expectation matches full" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (3, 4, 1), (3, 4, 2), (3, 4, 3),
            (4, 5, 2), (5, 6, 2),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(rand(p), rand(p))
            clause_sign = k == 2 ? -1 : 1

            val_full = basso_expectation(params, angles; clause_sign)
            val_reduced = basso_expectation_reduced(params, angles; clause_sign)

            @test val_reduced ≈ val_full atol = 1e-10
        end
    end

    @testset "validation: known MaxCut results" begin
        # MaxCut k=2, D=3, p=1 exact optimum: 0.5 + √3/9
        params = TreeParams(2, 3, 1)
        γopt = atan(1 / sqrt(2))
        βopt = π / 8
        optimum = 0.5 + sqrt(3) / 9
        angles = QAOAAngles([γopt], [βopt])
        val = basso_expectation_reduced(params, angles; clause_sign=-1)
        @test val ≈ optimum atol = 1e-10
    end

    @testset "4× size reduction" begin
        @testset "p=$p" for p in 1:5
            basis = ReducedBasis(p)
            B_red = ones(ComplexF64, basis.M)
            B_full = ones(ComplexF64, basis.N)
            @test length(B_red) * 4 == length(B_full)
        end
    end
end
