using QaoaXorsat
using Test

@testset "Basso finite-D helpers" begin
    @testset "gamma vector" begin
        angles = QAOAAngles([0.21, 0.64], [0.17, 0.39])
        @test QaoaXorsat.build_gamma_vector(angles) == [0.21, 0.64, -0.64, -0.21]
    end

    @testset "bit counts" begin
        @test QaoaXorsat.basso_bit_count(1) == 3
        @test QaoaXorsat.basso_bit_count(3) == 7
        @test QaoaXorsat.basso_configuration_count(1) == 8
        @test QaoaXorsat.basso_configuration_count(2) == 32
    end

    @testset "decode bits" begin
        @test QaoaXorsat.decode_bits(0b10110, 5) == [0, 1, 1, 0, 1]
        @test QaoaXorsat.decode_bits(0, 3) == [0, 0, 0]
        @test_throws ArgumentError QaoaXorsat.decode_bits(-1, 3)
        @test_throws ArgumentError QaoaXorsat.decode_bits(8, 3)
    end

    @testset "f(a) zero beta support" begin
        angles = QAOAAngles([0.31], [0.0])

        @test QaoaXorsat.f_function(angles, 0b000) ≈ 0.5 + 0.0im atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b111) ≈ 0.5 + 0.0im atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b001) ≈ 0.0 + 0.0im atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b010) ≈ 0.0 + 0.0im atol = 1e-12
    end

    @testset "f(a) p=1 values" begin
        β = 0.37
        angles = QAOAAngles([0.21], [β])
        c = cos(β)
        s = sin(β)

        @test QaoaXorsat.f_function(angles, 0b000) ≈ 0.5 * c^2 atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b001) ≈ 0.5im * c * s atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b011) ≈ -0.5im * c * s atol = 1e-12
        @test QaoaXorsat.f_function(angles, 0b010) ≈ 0.5 * s^2 atol = 1e-12
    end
end