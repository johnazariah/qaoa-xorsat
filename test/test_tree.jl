using QaoaXorsat
using Test

@testset "TreeParams" begin
    @testset "construction" begin
        @test TreeParams(2, 3, 1) isa TreeParams
        @test TreeParams(3, 4, 5) isa TreeParams
        @test_throws ArgumentError TreeParams(1, 3, 1)  # k < 2
        @test_throws ArgumentError TreeParams(2, 1, 1)  # D < 2
        @test_throws ArgumentError TreeParams(2, 3, 0)  # p < 1
    end

    @testset "branching_factor" begin
        @test branching_factor(TreeParams(2, 3, 1)) == 2   # MaxCut D=3
        @test branching_factor(TreeParams(3, 4, 1)) == 6   # Our target
        @test branching_factor(TreeParams(4, 5, 1)) == 12
    end

    # Golden values from spec — MaxCut (k=2, D=3)
    @testset "node counts k=2, D=3 (MaxCut)" begin
        @testset "p=$p" for (p, exp_var, exp_con, exp_total, exp_leaf) in [
            (1, 2,  1,  3,  2),
            (2, 6,  5,  11, 4),
            (3, 14, 13, 27, 8),
            (4, 30, 29, 59, 16),
        ]
            t = TreeParams(2, 3, p)
            @test total_variables(t)   == exp_var
            @test total_constraints(t) == exp_con
            @test total_nodes(t)       == exp_total
            @test leaf_count(t)        == exp_leaf
        end
    end

    # Golden values from spec — (k=3, D=4)
    @testset "node counts k=3, D=4" begin
        @testset "p=$p" for (p, exp_var, exp_con, exp_total) in [
            (1, 3,    1,    4),
            (2, 21,   10,   31),
            (3, 129,  64,   193),
            (4, 777,  388,  1165),
            (5, 4665, 2332, 6997),
        ]
            t = TreeParams(3, 4, p)
            @test total_variables(t)   == exp_var
            @test total_constraints(t) == exp_con
            @test total_nodes(t)       == exp_total
        end
    end

    @testset "leaf_count" begin
        @test leaf_count(TreeParams(2, 3, 1)) == 2
        @test leaf_count(TreeParams(3, 4, 1)) == 3
        @test leaf_count(TreeParams(3, 4, 2)) == 18
    end

    @testset "level count bounds" begin
        @test_throws ArgumentError variable_count_at_level(TreeParams(2, 3, 1), -1)
        @test_throws ArgumentError variable_count_at_level(TreeParams(2, 3, 1), 1)
        @test_throws ArgumentError constraint_count_at_level(TreeParams(2, 3, 1), -1)
        @test_throws ArgumentError constraint_count_at_level(TreeParams(2, 3, 1), 1)
    end

    @testset "monotonicity" begin
        for k in [2, 3, 4], D in [3, 4, 5]
            for p in 1:4
                tp = TreeParams(k, D, p)
                tp_next = TreeParams(k, D, p + 1)
                @test total_variables(tp_next) > total_variables(tp)
                @test total_nodes(tp_next) > total_nodes(tp)
            end
        end
    end
end
