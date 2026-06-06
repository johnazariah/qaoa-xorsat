using QaoaXorsat
using Test

@testset "Charge decomposition" begin
    @testset "matches basso_parity_expectation" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (3, 4, 1), (3, 4, 2), (3, 4, 3),
            (3, 2, 1), (3, 2, 2),
            (4, 3, 1), (4, 3, 2),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(
                [0.3 + 0.1 * i for i in 1:p],
                [0.4 - 0.05 * i for i in 1:p],
            )

            basso_val = basso_parity_expectation(params, angles)
            charge_val = charge_parity_expectation(params, angles)

            @test charge_val ≈ basso_val atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "matches basso_expectation" begin
        @testset "k=$k, D=$D, p=$p, cs=$cs" for (k, D, p, cs) in [
            (2, 3, 1, -1), (2, 3, 2, -1),  # MaxCut uses clause_sign=-1
            (3, 4, 1, 1), (3, 4, 2, 1),
            (3, 4, 1, -1), (3, 4, 2, -1),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(rand(p), rand(p))

            basso_val = basso_expectation(params, angles; clause_sign=cs)
            charge_val = charge_expectation(params, angles; clause_sign=cs)

            @test charge_val ≈ basso_val atol = 1e-10
        end
    end

    @testset "zero angles gives 0.5 expectation" begin
        for (k, D) in [(2, 3), (3, 4)]
            params = TreeParams(k, D, 1)
            angles = QAOAAngles([0.0], [0.0])
            @test charge_expectation(params, angles) ≈ 0.5 atol = 1e-12
        end
    end

    @testset "MaxCut p=1 optimal" begin
        # Known: k=2, D=3, p=1 optimal c̃ ≈ 0.6924 (Farhi 2014)
        params = TreeParams(2, 3, 1)
        angles = QAOAAngles([0.6155580653], [0.3927292003])
        c_basso = basso_expectation(params, angles; clause_sign=-1)
        c_charge = charge_expectation(params, angles; clause_sign=-1)
        @test c_charge ≈ c_basso atol = 1e-8
        @test c_charge ≈ 0.6924500847885 atol = 1e-6
    end
end
