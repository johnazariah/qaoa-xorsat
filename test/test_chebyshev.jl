using QaoaXorsat
using Test

@testset "Chebyshev warm-start" begin
    @testset "_chebyshev_basis" begin
        # T₀(x) = 1, T₁(x) = 2x-1 on [0,1]
        t = [0.0, 0.25, 0.5, 0.75, 1.0]
        B = QaoaXorsat._chebyshev_basis(t, 2)
        @test size(B) == (5, 3)
        @test all(B[:, 1] .≈ 1.0)  # T₀ = 1
        @test B[1, 2] ≈ -1.0  # T₁(0) = 2*0-1 = -1
        @test B[3, 2] ≈ 0.0   # T₁(0.5) = 0
        @test B[5, 2] ≈ 1.0   # T₁(1) = 1
    end

    @testset "_chebyshev_interp identity at same depth" begin
        # Interpolating p→p should recover original angles
        angles = [0.3, 0.5, 0.7, 0.9, 1.1]
        result = QaoaXorsat._chebyshev_interp(angles, 5)
        @test result ≈ angles atol = 1e-12
    end

    @testset "_chebyshev_interp linear angles" begin
        # For a linear function f(t) = a + b*t, any interpolation to higher p
        # should exactly reproduce the linear function at the new sample points
        p_src = 3
        a, b = 0.5, 1.2
        t_src = [(i - 0.5) / p_src for i in 1:p_src]
        angles_src = a .+ b .* t_src

        p_tgt = 7
        t_tgt = [(i - 0.5) / p_tgt for i in 1:p_tgt]
        expected = a .+ b .* t_tgt

        result = QaoaXorsat._chebyshev_interp(angles_src, p_tgt)
        @test result ≈ expected atol = 1e-12
    end

    @testset "_chebyshev_interp with truncation" begin
        # With num_coeffs=2, should fit a linear function through any data
        angles = [0.1, 0.5, 0.3, 0.8, 0.2]  # non-smooth
        result = QaoaXorsat._chebyshev_interp(angles, 5; num_coeffs=2)
        # Result should be smoother (linear fit), not identical
        @test length(result) == 5
        # Check it's actually linear: differences should be constant
        diffs = diff(result)
        @test all(isapprox.(diffs, diffs[1]; atol=1e-12))
    end

    @testset "chebyshev_extend_angles basic" begin
        base = QAOAAngles([0.2, 0.4, 0.6], [0.1, 0.3, 0.5])
        extended = chebyshev_extend_angles(base, 5)

        @test depth(extended) == 5
        @test length(extended.γ) == 5
        @test length(extended.β) == 5
    end

    @testset "chebyshev_extend_angles identity at same depth" begin
        base = QAOAAngles([0.2, 0.4, 0.6], [0.1, 0.3, 0.5])
        same = chebyshev_extend_angles(base, 3)
        @test same.γ == base.γ
        @test same.β == base.β
    end

    @testset "chebyshev_extend_angles p=1 source" begin
        base = QAOAAngles([0.5], [0.3])
        extended = chebyshev_extend_angles(base, 4)
        @test depth(extended) == 4
        @test all(extended.γ .≈ 0.5)
        @test all(extended.β .≈ 0.3)
    end

    @testset "interp_extend_angles basic" begin
        base = QAOAAngles([0.2, 0.4, 0.6], [0.1, 0.3, 0.5])
        extended = interp_extend_angles(base, 5)

        @test depth(extended) == 5
        # Values should be within the range of the source
        @test all(0.2 - 0.01 ≤ g ≤ 0.6 + 0.01 for g in extended.γ)
        @test all(0.1 - 0.01 ≤ b ≤ 0.5 + 0.01 for b in extended.β)
    end

    @testset "interp_extend_angles identity at same depth" begin
        base = QAOAAngles([0.2, 0.4, 0.6], [0.1, 0.3, 0.5])
        same = interp_extend_angles(base, 3)
        @test same.γ ≈ base.γ atol = 1e-12
        @test same.β ≈ base.β atol = 1e-12
    end

    @testset "interp_extend_angles p=1 source" begin
        base = QAOAAngles([0.5], [0.3])
        extended = interp_extend_angles(base, 4)
        @test depth(extended) == 4
        @test all(extended.γ .≈ 0.5)
        @test all(extended.β .≈ 0.3)
    end

    @testset "chebyshev vs linear warm-start quality" begin
        # Compare warm-start quality: evaluate the objective at the warm-start
        # point and check that both produce valid starting points.
        # Note: Chebyshev is not always better than linear for individually
        # optimized angles (which may not follow a smooth curve). The real
        # benefit shows at higher p with depth-sequence optimization.
        params_3 = TreeParams(2, 3, 3)
        result_3 = optimize_angles(params_3; clause_sign=-1, restarts=4, maxiters=200)
        angles_3 = result_3.angles

        params_5 = TreeParams(2, 3, 5)

        # Linear warm-start
        linear_5 = extend_angles(angles_3, 5)
        val_linear = basso_expectation(params_5, linear_5; clause_sign=-1)

        # Chebyshev warm-start
        cheb_5 = chebyshev_extend_angles(angles_3, 5)
        val_cheb = basso_expectation(params_5, cheb_5; clause_sign=-1)

        # Both should be valid expectations (between 0 and 1)
        @test 0.0 < val_linear < 1.0
        @test 0.0 < val_cheb < 1.0
    end

    @testset "optimize_depth_sequence with $strategy strategy" for strategy in [:chebyshev, :interp]
        results = optimize_depth_sequence(2, 3, 1:3;
            clause_sign=-1,
            restarts=2,
            maxiters=100,
            warm_start_strategy=strategy,
        )
        @test length(results) == 3
        @test all(r -> 0.0 < r.value < 1.0, results)
        # Values should improve with depth
        @test results[2].value ≥ results[1].value - 0.01
        @test results[3].value ≥ results[2].value - 0.01
    end
end
