using QaoaXorsat
using Random
using Test

@testset "Optimization" begin
    @testset "canonicalize_angles" begin
        angles = QAOAAngles([-0.5, 2π + 0.25], [-0.1, π + 0.2])
        canonical = canonicalize_angles(angles)

        @test canonical.γ[1] ≈ 2π - 0.5 atol = 1e-12
        @test canonical.γ[2] ≈ 0.25 atol = 1e-12
        @test canonical.β[1] ≈ π - 0.1 atol = 1e-12
        @test canonical.β[2] ≈ 0.2 atol = 1e-12
    end

    @testset "random_angles" begin
        rng = MersenneTwister(1234)
        angles = random_angles(3; rng)

        @test depth(angles) == 3
        @test all(0.0 ≤ γ < 2π for γ in angles.γ)
        @test all(0.0 ≤ β < π for β in angles.β)
    end

    @testset "extend_angles" begin
        base = QAOAAngles([0.2, 0.4], [0.1, 0.3])
        extended = extend_angles(base, 4)

        @test extended.γ == [0.2, 0.4, 0.4, 0.4]
        @test extended.β == [0.1, 0.3, 0.3, 0.3]
    end

    @testset "depth_optimization_budget" begin
        budget_p3 = QaoaXorsat.depth_optimization_budget(3, 8, 200)
        budget_p4 = QaoaXorsat.depth_optimization_budget(4, 8, 200)
        budget_p5 = QaoaXorsat.depth_optimization_budget(5, 8, 200)

        @test budget_p3.restarts == 8
        @test budget_p3.maxiters == 200
        @test budget_p4.restarts == 4
        @test budget_p4.maxiters == 400
        @test budget_p5.restarts == 2
        @test budget_p5.maxiters == 800
        @test QaoaXorsat.retry_optimization_budget(200) == 400
    end

    @testset "optimize_angles MaxCut p=1" begin
        params = TreeParams(2, 3, 1)
        optimum = 0.5 + sqrt(3) / 9
        seed = QAOAAngles([0.7], [0.3])

        result = optimize_angles(
            params;
            clause_sign=-1,
            restarts=0,
            maxiters=100,
            initial_guesses=[seed],
            rng=MersenneTwister(7),
        )

        @test result.value ≈ optimum atol = 1e-6
        @test depth(result.angles) == 1
        @test result.starts == 1
        @test result.evaluations ≥ 1
        @test result.wall_time_seconds ≥ 0.0
        @test result.best_start_wall_time_seconds ≥ 0.0
        @test result.best_start_wall_time_seconds ≤ result.wall_time_seconds
        @test result.restarts == 0
        @test result.maxiters == 100
        @test result.retry_count == 0
        @test result.best_start_kind == :seeded
        @test length(result.start_results) == 1
    end

    @testset "optimize_angles start telemetry" begin
        result = optimize_angles(
            TreeParams(2, 3, 1);
            clause_sign=-1,
            restarts=1,
            maxiters=5,
            initial_guesses=[QAOAAngles([0.7], [0.3])],
            rng=MersenneTwister(19),
        )

        @test result.starts == 2
        @test result.restarts == 1
        @test result.maxiters == 5
        @test length(result.start_results) == 2
        @test [start.kind for start in result.start_results] == [:seeded, :random]
        @test all(start -> start.evaluations ≥ 1, result.start_results)
        @test all(start -> start.wall_time_seconds ≥ 0.0, result.start_results)
    end

    @testset "optimize_angles canonicalizes stored result angles" begin
        result = optimize_angles(
            TreeParams(2, 3, 1);
            clause_sign=-1,
            restarts=0,
            maxiters=5,
            initial_guesses=[QAOAAngles([-0.7], [π + 0.3])],
            rng=MersenneTwister(23),
        )

        @test all(0.0 ≤ γ < 2π for γ in result.angles.γ)
        @test all(0.0 ≤ β < π for β in result.angles.β)
    end

    @testset "optimize_depth_sequence warm starts" begin
        callback_results = QaoaXorsat.AngleOptimizationResult[]
        results = optimize_depth_sequence(
            2,
            3,
            [1, 2];
            clause_sign=-1,
            restarts=0,
            maxiters=5,
            rng=MersenneTwister(11),
            on_result=result -> push!(callback_results, result),
        )

        @test length(results) == 2
        @test length(callback_results) == 2
        @test depth.(getfield.(callback_results, :angles)) == [1, 2]
        @test depth(results[1].angles) == 1
        @test depth(results[2].angles) == 2
        @test all(result -> isfinite(result.value), results)
        @test all(result -> result.wall_time_seconds ≥ 0.0, results)
        @test all(result -> result.best_start_wall_time_seconds ≥ 0.0, results)
        @test all(result -> result.best_start_wall_time_seconds ≤ result.wall_time_seconds, results)
        @test results[1].best_start_kind in (:random, :seeded)
        @test results[2].best_start_kind in (:warm, :random, :retry)
        @test results[2].restarts == 0
        @test results[2].maxiters in (5, 10)
        @test all(result -> !isempty(result.start_results), results)
    end
end
