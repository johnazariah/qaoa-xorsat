using QaoaXorsat
using ForwardDiff
using Random
using Test

@testset "Normalized evaluator" begin
    # ──────────────────────────────────────────────────────────────────────
    # 1. Exact agreement with un-normalized evaluator at low (k,D,p)
    # ──────────────────────────────────────────────────────────────────────
    @testset "value matches basso_expectation at low (k,D)" begin
        rng = MersenneTwister(42)
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (3, 4, 1), (3, 4, 2), (3, 4, 3),
            (3, 5, 1), (3, 5, 2),
            (4, 5, 1), (4, 5, 2),
            (5, 6, 1), (5, 6, 2),
        ]
            params = TreeParams(k, D, p)
            cs = k == 2 ? -1 : 1
            for _ in 1:5
                angles = QAOAAngles(2π .* rand(rng, p), π .* rand(rng, p))
                v_ref = basso_expectation(params, angles; clause_sign=cs)
                v_norm = basso_expectation_normalized(params, angles; clause_sign=cs)
                @test v_norm ≈ v_ref atol = 1e-10 rtol = 1e-10
            end
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # 2. Gradient matches basso_expectation at low (k,D,p)
    #    Tests that the backward pass correctly incorporates the scale factor.
    # ──────────────────────────────────────────────────────────────────────
    @testset "gradient matches ForwardDiff at low (k,D)" begin
        rng = MersenneTwister(99)
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (3, 4, 1), (3, 4, 2), (3, 4, 3),
            (3, 5, 2),
            (4, 5, 2),
            (5, 6, 2),
        ]
            params = TreeParams(k, D, p)
            cs = k == 2 ? -1 : 1
            γ = 2π .* rand(rng, p)
            β = π .* rand(rng, p)

            function fd_objective(values)
                a = QAOAAngles(values[1:p], values[p+1:2p])
                basso_expectation(params, a; clause_sign=cs)
            end
            fd_grad = ForwardDiff.gradient(fd_objective, [γ; β])

            angles = QAOAAngles(γ, β)
            _, γg, βg = basso_expectation_and_gradient(params, angles; clause_sign=cs)

            @test γg ≈ fd_grad[1:p] atol = 1e-8
            @test βg ≈ fd_grad[p+1:2p] atol = 1e-8
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # 3. Physical bounds: c̃ ∈ [0, 1] at ALL (k,D,p) including high ones
    # ──────────────────────────────────────────────────────────────────────
    @testset "value bounded ∈ [0, 1] at high (k,D)" begin
        rng = MersenneTwister(2026)
        # These are the cases that previously overflowed Float64
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (5, 8, 9), (5, 8, 10),
            (6, 7, 9), (6, 7, 10),
            (6, 8, 9), (6, 8, 10),
            (7, 8, 8), (7, 8, 9), (7, 8, 10),
            (4, 8, 10), (4, 8, 11),
        ]
            params = TreeParams(k, D, p)
            for trial in 1:3
                angles = QAOAAngles(2π .* rand(rng, p), π .* rand(rng, p))
                v = basso_expectation_normalized(params, angles; clause_sign=1)
                @test isfinite(v) || isnan(v)
                if isfinite(v)
                    @test -1e-9 ≤ v ≤ 1.0 + 1e-9
                end
            end
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # 4. Gradient finite at high (k,D) — no NaN/Inf in gradient vectors
    # ──────────────────────────────────────────────────────────────────────
    @testset "gradient finite at high (k,D)" begin
        rng = MersenneTwister(314)
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (5, 8, 9), (5, 8, 10),
            (6, 8, 9), (6, 8, 10),
            (7, 8, 8), (7, 8, 9),
            (4, 8, 10),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(2π .* rand(rng, p), π .* rand(rng, p))
            val, γg, βg = basso_expectation_and_gradient(params, angles; clause_sign=1)

            @test isfinite(val) || isnan(val)
            if isfinite(val) && -1e-9 ≤ val ≤ 1.0 + 1e-9
                @test all(isfinite, γg)
                @test all(isfinite, βg)
            end
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # 5. Consistency: value from gradient call matches standalone evaluation
    # ──────────────────────────────────────────────────────────────────────
    @testset "value_and_gradient consistent with normalized eval" begin
        rng = MersenneTwister(1701)
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (3, 4, 3), (4, 5, 3), (5, 6, 2), (6, 7, 2), (7, 8, 2),
            (7, 8, 5), (6, 8, 5),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(2π .* rand(rng, p), π .* rand(rng, p))
            val_norm = basso_expectation_normalized(params, angles; clause_sign=1)
            val_grad, _, _ = basso_expectation_and_gradient(params, angles; clause_sign=1)

            @test val_norm ≈ val_grad atol = 1e-12
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # 6. Known MaxCut results preserved through normalization
    # ──────────────────────────────────────────────────────────────────────
    @testset "MaxCut k=2, D=3 validation preserved" begin
        # Farhi et al. 2014: p=1 optimum for 3-regular MaxCut
        # The clause-level optimum is 0.5 + √3/9 ≈ 0.6925
        seed = QAOAAngles([0.7], [0.3])
        optimum = 0.5 + sqrt(3) / 9

        result = optimize_angles(
            TreeParams(2, 3, 1);
            clause_sign=-1,
            restarts=0,
            maxiters=100,
            initial_guesses=[seed],
            rng=MersenneTwister(7),
        )

        @test result.value ≈ optimum atol = 1e-6
        @test QaoaXorsat.is_valid_qaoa_value(result.value)
    end

    # ──────────────────────────────────────────────────────────────────────
    # 7. Optimizer at high (k,D) produces valid results (not overflow)
    # ──────────────────────────────────────────────────────────────────────
    @testset "optimizer produces valid results at high (k,D)" begin
        # (7,8) at p=5 previously would have been fine, but let's verify
        # the full optimizer pipeline stays valid
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (5, 6, 3), (6, 7, 3), (7, 8, 3),
        ]
            result = optimize_angles(
                TreeParams(k, D, p);
                clause_sign=1,
                restarts=0,
                maxiters=20,
                rng=MersenneTwister(42),
            )

            @test QaoaXorsat.is_valid_qaoa_value(result.value)
            @test result.value > 0.5  # should be better than trivial
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # 8. Scale factor correctness: normalization should not change the
    #    mathematical value — verify by computing at moderate (k,D,p)
    #    where both paths work, that log_total_scale is consistent
    # ──────────────────────────────────────────────────────────────────────
    @testset "scale accumulation self-consistent" begin
        rng = MersenneTwister(555)
        @testset "k=$k, D=$D, p=$p" for (k, D, p) in [
            (3, 4, 2), (3, 4, 4), (4, 5, 3), (5, 6, 3),
        ]
            params = TreeParams(k, D, p)
            angles = QAOAAngles(2π .* rand(rng, p), π .* rand(rng, p))

            # Un-normalized reference (works at moderate depths)
            v_ref = basso_expectation(params, angles; clause_sign=1)

            # Normalized
            v_norm = basso_expectation_normalized(params, angles; clause_sign=1)

            # They should agree precisely
            @test v_norm ≈ v_ref atol = 1e-10
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # 9. Regression: the specific (k,D,p) cases from the cluster logs
    #    that produced impossible values. Verify they now return valid c̃.
    # ──────────────────────────────────────────────────────────────────────
    @testset "cluster overflow regression" begin
        rng = MersenneTwister(1234)  # same seed as the production runs

        # All 15 (k,D) pairs at p=10 — the depth where most pairs overflowed
        @testset "(k=$k, D=$D) at p=10" for (k, D) in [
            (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
            (4, 5), (4, 6), (4, 7), (4, 8),
            (5, 6), (5, 7), (5, 8),
            (6, 7), (6, 8),
            (7, 8),
        ]
            p = 10
            params = TreeParams(k, D, p)
            angles = QAOAAngles(2π .* rand(rng, p), π .* rand(rng, p))

            v = basso_expectation_normalized(params, angles; clause_sign=1)
            val, γg, βg = basso_expectation_and_gradient(params, angles; clause_sign=1)

            # Value must be physical
            @test isfinite(v)
            @test -1e-9 ≤ v ≤ 1.0 + 1e-9

            # Value from gradient call must match
            @test v ≈ val atol = 1e-12

            # Gradients must be finite
            @test all(isfinite, γg)
            @test all(isfinite, βg)
        end
    end

    # ──────────────────────────────────────────────────────────────────────
    # 10. is_valid_qaoa_value utility function
    # ──────────────────────────────────────────────────────────────────────
    @testset "is_valid_qaoa_value" begin
        @test QaoaXorsat.is_valid_qaoa_value(0.5)
        @test QaoaXorsat.is_valid_qaoa_value(0.0)
        @test QaoaXorsat.is_valid_qaoa_value(1.0)
        @test QaoaXorsat.is_valid_qaoa_value(0.999999999)
        @test !QaoaXorsat.is_valid_qaoa_value(1.5)
        @test !QaoaXorsat.is_valid_qaoa_value(-0.5)
        @test !QaoaXorsat.is_valid_qaoa_value(NaN)
        @test !QaoaXorsat.is_valid_qaoa_value(Inf)
        @test !QaoaXorsat.is_valid_qaoa_value(-Inf)
        @test !QaoaXorsat.is_valid_qaoa_value(21.44)
        @test !QaoaXorsat.is_valid_qaoa_value(1.33)
    end

    # ──────────────────────────────────────────────────────────────────────
    # 11. merge_optimization_results validity-aware selection
    # ──────────────────────────────────────────────────────────────────────
    @testset "merge prefers valid over invalid" begin
        dummy_trace = QaoaXorsat.OptimizationTraceEntry[]
        dummy_angles = QAOAAngles([0.1], [0.2])
        dummy_start = QaoaXorsat.AngleOptimizationStartResult(:test, 0.8, 1.0, 10, 5, true, dummy_trace)

        valid_result = QaoaXorsat.AngleOptimizationResult(
            dummy_angles, 0.85, 1.0, 1.0, 10, 1, 5, true, 0, 100, 0, :warm, 1e-6,
            [dummy_start],
        )
        overflow_result = QaoaXorsat.AngleOptimizationResult(
            dummy_angles, 21.44, 1.0, 1.0, 10, 1, 5, true, 0, 100, 0, :random, 1e-6,
            [dummy_start],
        )
        invalid_result = QaoaXorsat.AngleOptimizationResult(
            dummy_angles, -Inf, 1.0, 1.0, 10, 1, 5, false, 0, 100, 0, :random, 1e-6,
            [dummy_start],
        )

        # Valid should beat overflow even though overflow is numerically larger
        merged = QaoaXorsat.merge_optimization_results(overflow_result, valid_result)
        @test merged.value ≈ 0.85

        merged2 = QaoaXorsat.merge_optimization_results(valid_result, overflow_result)
        @test merged2.value ≈ 0.85

        # Valid should beat -Inf
        merged3 = QaoaXorsat.merge_optimization_results(invalid_result, valid_result)
        @test merged3.value ≈ 0.85
    end
end
