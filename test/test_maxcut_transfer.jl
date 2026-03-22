using QaoaXorsat
using Test

@testset "MaxCut transfer recursion" begin
    @testset "p=1 regression" begin
        angles = QAOAAngles([0.31], [0.17])
        matrix = QaoaXorsat.build_maxcut_transfer_matrix(3, angles)

        @test size(matrix) == (3, 3)
        @test matrix[1, 1] ≈ 1.0 + 0.0im atol = 1e-12
        @test matrix[1, 2] ≈ 0.9427546655283464 - 0.27517414736381im atol = 1e-12
        @test matrix[2, 2] ≈ 1.0 + 0.0im atol = 1e-12
        @test matrix[2, 3] ≈ 0.9427546655283464 + 0.27517414736381im atol = 1e-12
        @test QaoaXorsat.maxcut_transfer_objective(3, angles) ≈
            0.18043902430464884 atol = 1e-12
    end

    @testset "p=2 regression" begin
        angles = QAOAAngles([0.21, 0.64], [0.17, 0.39])
        matrix = QaoaXorsat.build_maxcut_transfer_matrix(3, angles)

        @test size(matrix) == (5, 5)
        @test matrix[1, 2] ≈ 0.9427546655283464 - 0.3053333590958912im atol = 1e-12
        @test matrix[1, 3] ≈ 0.5717499119206679 - 0.38436593706558586im atol = 1e-12
        @test matrix[2, 3] ≈ 0.7109135380122775 - 0.22370705787128325im atol = 1e-12
        @test matrix[3, 3] ≈ 1.0 + 0.0im atol = 1e-12
        @test matrix[4, 5] ≈ 0.9427546655283464 + 0.3053333590958912im atol = 1e-12
        @test QaoaXorsat.maxcut_transfer_objective(3, angles) ≈
            0.22628870336650803 atol = 1e-12
    end

    @testset "corner symmetry" begin
        angles = QAOAAngles([0.31, 0.57], [0.17, 0.23])
        matrix = QaoaXorsat.build_maxcut_transfer_matrix(4, angles)
        p = depth(angles)
        last_index = 2p + 1

        for row in 1:(p + 1), column in row:(p + 1)
            @test matrix[column, row] ≈ matrix[row, column] atol = 1e-12
            @test matrix[row, last_index - column + 1] ≈ matrix[row, column] atol = 1e-12
            @test matrix[column, last_index - row + 1] ≈ conj(matrix[row, column]) atol = 1e-12
        end
    end
end