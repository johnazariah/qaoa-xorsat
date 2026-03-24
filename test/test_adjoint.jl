using QaoaXorsat
using ForwardDiff
using Test

@testset "Adjoint differentiation" begin
    @testset "value matches basso_expectation" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [(2, 3, 1), (3, 4, 1), (2, 3, 2), (3, 4, 2), (3, 4, 3)]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(rand(p), rand(p))
            clause_sign = k == 2 ? -1 : 1

            val_basso = basso_expectation(params, angles; clause_sign)
            val_adj, _, _ = basso_expectation_and_gradient(params, angles; clause_sign)

            @test val_adj ≈ val_basso atol = 1e-12
        end
    end

    @testset "gradient matches ForwardDiff" begin
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [(2, 3, 1), (3, 4, 1), (2, 3, 2), (3, 4, 2), (3, 4, 3)]
            params = TreeParams(k, D, p)
            γ = rand(p)
            β = rand(p)
            clause_sign = k == 2 ? -1 : 1

            # ForwardDiff reference gradient
            function objective(values)
                angles = QAOAAngles(values[1:p], values[p+1:2p])
                basso_expectation(params, angles; clause_sign)
            end
            x = [γ; β]
            fd_grad = ForwardDiff.gradient(objective, x)

            # Adjoint gradient
            angles = QAOAAngles(γ, β)
            _, γ_grad, β_grad = basso_expectation_and_gradient(params, angles; clause_sign)

            @test γ_grad ≈ fd_grad[1:p] atol = 1e-10
            @test β_grad ≈ fd_grad[p+1:2p] atol = 1e-10
        end
    end

    @testset "gradient at zero angles" begin
        params = TreeParams(3, 4, 1)
        angles = QAOAAngles([0.0], [0.0])
        val, γ_grad, β_grad = basso_expectation_and_gradient(params, angles)

        @test val ≈ 0.5 atol = 1e-12
        @test length(γ_grad) == 1
        @test length(β_grad) == 1
    end

    @testset "gradient symmetry: negating clause_sign flips γ gradient" begin
        params = TreeParams(2, 3, 1)
        angles = QAOAAngles([0.7], [0.3])

        _, γ_pos, β_pos = basso_expectation_and_gradient(params, angles; clause_sign=1)
        _, γ_neg, β_neg = basso_expectation_and_gradient(params, angles; clause_sign=-1)

        # E_+ + E_- = 1, so ∂E_+/∂γ + ∂E_-/∂γ = 0
        @test γ_pos .+ γ_neg ≈ zeros(1) atol = 1e-10
        @test β_pos .+ β_neg ≈ zeros(1) atol = 1e-10
    end
end
