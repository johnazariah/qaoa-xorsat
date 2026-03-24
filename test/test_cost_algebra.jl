@testset "CostAlgebra" begin

    @testset "XORSATAlgebra construction" begin
        a = XORSATAlgebra(3)
        @test arity(a) == 3
        @test default_clause_sign(a) == 1

        a2 = XORSATAlgebra(3; clause_sign=-1)
        @test default_clause_sign(a2) == -1

        @test_throws ArgumentError XORSATAlgebra(1)
        @test_throws ArgumentError XORSATAlgebra(3; clause_sign=0)
    end

    @testset "MaxCutAlgebra" begin
        a = MaxCutAlgebra()
        @test arity(a) == 2
        @test default_clause_sign(a) == -1
        @test a isa XORSATAlgebra{2}
    end

    @testset "algebra_from_clause_sign" begin
        a = algebra_from_clause_sign(3, 1)
        @test a isa XORSATAlgebra{3}
        @test default_clause_sign(a) == 1
    end

    @testset "algebra-parameterised qaoa_expectation matches legacy API" begin
        for (k, D, cs) in [(2, 3, -1), (3, 4, 1), (3, 5, 1)]
            for p in 1:3
                params = TreeParams(k, D, p)
                angles = QAOAAngles(randn(p), randn(p))
                algebra = algebra_from_clause_sign(k, cs)

                legacy = qaoa_expectation(params, angles; clause_sign=cs)
                via_algebra = qaoa_expectation(algebra, params, angles)

                @test via_algebra ≈ legacy atol=1e-14
            end
        end
    end

    @testset "MaxCutAlgebra matches clause_sign=-1 for MaxCut" begin
        algebra = MaxCutAlgebra()
        for p in 1:3
            params = TreeParams(2, 3, p)
            angles = QAOAAngles(randn(p), randn(p))

            legacy = qaoa_expectation(params, angles; clause_sign=-1)
            via_algebra = qaoa_expectation(algebra, params, angles)

            @test via_algebra ≈ legacy atol=1e-14
        end
    end

    @testset "arity mismatch throws" begin
        algebra = XORSATAlgebra(3)
        params = TreeParams(2, 3, 1)
        angles = QAOAAngles([1.0], [1.0])
        @test_throws ArgumentError qaoa_expectation(algebra, params, angles)
    end

    @testset "algebra-parameterised optimize_angles matches legacy" begin
        algebra = XORSATAlgebra(3)
        params = TreeParams(3, 4, 1)
        result_legacy = optimize_angles(params; clause_sign=1, restarts=1, maxiters=20)
        result_algebra = optimize_angles(algebra, params; restarts=1, maxiters=20)

        @test result_algebra.value ≈ result_legacy.value atol=1e-8
    end
end
