using QaoaXorsat
using ForwardDiff
using Test

@testset "Charge adjoint differentiation" begin
    # ── 1. Value consistency: gradient call matches standalone charge_expectation
    @testset "value matches charge_expectation" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (3, 4, 1), (3, 4, 2), (3, 4, 3),
            (4, 3, 1), (4, 3, 2),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(
                [0.3 + 0.1 * i for i in 1:p],
                [0.4 - 0.05 * i for i in 1:p],
            )
            cs = k == 2 ? -1 : 1

            val_standalone = charge_expectation(params, angles; clause_sign=cs)
            val_grad, _, _ = charge_expectation_and_gradient(params, angles; clause_sign=cs)

            @test val_grad ≈ val_standalone atol = 1e-12
        end
    end

    # ── 2. Value matches basso_expectation (cross-evaluator consistency)
    @testset "value matches basso_expectation" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (3, 4, 1), (3, 4, 2),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(
                [0.5 + 0.15 * i for i in 1:p],
                [0.35 - 0.08 * i for i in 1:p],
            )
            cs = k == 2 ? -1 : 1

            val_basso = basso_expectation(params, angles; clause_sign=cs)
            val_charge, _, _ = charge_expectation_and_gradient(params, angles; clause_sign=cs)

            @test val_charge ≈ val_basso atol = 1e-10
        end
    end

    # ── 3. Gradient matches ForwardDiff on basso_expectation (gold standard)
    @testset "gradient matches ForwardDiff" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (3, 4, 1), (3, 4, 2), (3, 4, 3),
            (4, 3, 1), (4, 3, 2),
        ]
            params = TreeParams(k, D, p)
            γ = [0.3 + 0.1 * i for i in 1:p]
            β = [0.4 - 0.05 * i for i in 1:p]
            cs = k == 2 ? -1 : 1

            # ForwardDiff reference via basso (exact)
            function fd_objective(values)
                a = QAOAAngles(values[1:p], values[p+1:2p])
                basso_expectation(params, a; clause_sign=cs)
            end
            fd_grad = ForwardDiff.gradient(fd_objective, [γ; β])

            # Charge gradient
            angles = QAOAAngles(γ, β)
            _, γ_grad, β_grad = charge_expectation_and_gradient(params, angles; clause_sign=cs)

            @test γ_grad ≈ fd_grad[1:p] atol = 1e-6
            @test β_grad ≈ fd_grad[p+1:2p] atol = 1e-6
        end
    end

    # ── 4. Both clause_sign values
    @testset "clause_sign=$cs" for cs in [1, -1]
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2),
            (3, 4, 1), (3, 4, 2),
        ]
            params = TreeParams(k, D, p)
            γ = [0.6 + 0.1 * i for i in 1:p]
            β = [0.25 + 0.05 * i for i in 1:p]

            function fd_obj(values)
                a = QAOAAngles(values[1:p], values[p+1:2p])
                basso_expectation(params, a; clause_sign=cs)
            end
            fd_grad = ForwardDiff.gradient(fd_obj, [γ; β])

            angles = QAOAAngles(γ, β)
            val, γg, βg = charge_expectation_and_gradient(params, angles; clause_sign=cs)

            @test val ≈ basso_expectation(params, angles; clause_sign=cs) atol = 1e-10
            @test γg ≈ fd_grad[1:p] atol = 1e-6
            @test βg ≈ fd_grad[p+1:2p] atol = 1e-6
        end
    end

    # ── 5. Zero angles: value=0.5, gradients finite
    @testset "zero angles" begin
        for (k, D) in [(2, 3), (3, 4)]
            params = TreeParams(k, D, 1)
            angles = QAOAAngles([0.0], [0.0])
            val, γg, βg = charge_expectation_and_gradient(params, angles)

            @test val ≈ 0.5 atol = 1e-12
            @test length(γg) == 1
            @test length(βg) == 1
            @test all(isfinite, γg)
            @test all(isfinite, βg)
        end
    end

    # ── 6. Gradient matches basso adjoint (exact adjoint as second reference)
    @testset "matches basso adjoint gradient" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (3, 4, 1), (3, 4, 2),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(
                [0.4 + 0.12 * i for i in 1:p],
                [0.35 - 0.06 * i for i in 1:p],
            )
            cs = k == 2 ? -1 : 1

            _, basso_γg, basso_βg = basso_expectation_and_gradient(params, angles; clause_sign=cs)
            _, charge_γg, charge_βg = charge_expectation_and_gradient(params, angles; clause_sign=cs)

            @test charge_γg ≈ basso_γg atol = 1e-6
            @test charge_βg ≈ basso_βg atol = 1e-6
        end
    end

    # ── 7. MaxCut p=1 optimal: gradient near zero at known optimum
    @testset "near-zero gradient at MaxCut optimum" begin
        params = TreeParams(2, 3, 1)
        angles = QAOAAngles([0.6155580653], [0.3927292003])
        val, γg, βg = charge_expectation_and_gradient(params, angles; clause_sign=-1)

        @test val ≈ 0.6924500847885 atol = 1e-6
        @test abs(γg[1]) < 1e-4
        @test abs(βg[1]) < 1e-4
    end
end
