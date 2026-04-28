using QaoaXorsat, Test, Random, Printf

println("=== CPU Checkpointing Correctness Tests ===")
flush(stdout)

@testset "checkpointed value matches full" begin
    for (k, D, p) in [(2,3,5), (2,4,6), (3,4,4), (2,3,8)]
        params = TreeParams(k, D, p)
        cs = k == 2 ? -1 : 1
        angles = random_angles(p; rng=MersenneTwister(42))

        v_full = basso_expectation_normalized(params, angles; clause_sign=cs)
        v_ckpt = basso_expectation_checkpointed(params, angles; clause_sign=cs)

        @test v_full ≈ v_ckpt atol=1e-12
        @printf("  (k=%d,D=%d,p=%d): diff=%.2e ✓\n", k, D, p, abs(v_full-v_ckpt))
    end
end

@testset "checkpointed gradient matches full" begin
    for (k, D, p) in [(2,3,5), (2,4,6), (3,4,4), (2,3,8)]
        params = TreeParams(k, D, p)
        cs = k == 2 ? -1 : 1
        angles = random_angles(p; rng=MersenneTwister(42))

        v1, γg1, βg1 = basso_expectation_and_gradient(params, angles; clause_sign=cs)
        v2, γg2, βg2 = basso_expectation_and_gradient_checkpointed(params, angles; clause_sign=cs)

        @test v1 ≈ v2 atol=1e-12
        @test γg1 ≈ γg2 atol=1e-10
        @test βg1 ≈ βg2 atol=1e-10
        @printf("  (k=%d,D=%d,p=%d): max γ=%.2e β=%.2e ✓\n", k, D, p,
                maximum(abs.(γg1.-γg2)), maximum(abs.(βg1.-βg2)))
    end
end

@testset "checkpoint intervals" begin
    params = TreeParams(2, 3, 8)
    angles = random_angles(8; rng=MersenneTwister(42))
    v_ref, γ_ref, β_ref = basso_expectation_and_gradient(params, angles; clause_sign=-1)

    for ci in [1, 2, 3, 4, 8]
        v, γg, βg = basso_expectation_and_gradient_checkpointed(params, angles;
            clause_sign=-1, checkpoint_interval=ci)
        @test v ≈ v_ref atol=1e-12
        @test γg ≈ γ_ref atol=1e-10
        @test βg ≈ β_ref atol=1e-10
        @printf("  ci=%d: ok ✓\n", ci)
    end
end

println("\nAll checkpointing tests passed!")
