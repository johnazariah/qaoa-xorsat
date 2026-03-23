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
    end

    @testset "optimize_depth_sequence warm starts" begin
        results = optimize_depth_sequence(
            2,
            3,
            [1, 2];
            clause_sign=-1,
            restarts=0,
            maxiters=5,
            rng=MersenneTwister(11),
        )

        @test length(results) == 2
        @test depth(results[1].angles) == 1
        @test depth(results[2].angles) == 2
        @test all(result -> isfinite(result.value), results)
    end
end
