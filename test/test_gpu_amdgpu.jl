using AMDGPU
using QaoaXorsat
using Random
using Test

@testset "Live AMDGPU backend" begin
    backend = gpu_backend(:amdgpu)
    @test backend.kind == :amdgpu
    @test backend.complex_type == ComplexF64
    @test backend.device !== nothing
    @test gpu_backend_available(:amdgpu)
    @test validate_gpu_backend(backend)
    @test gpu_backend(:amd).kind == :amdgpu
    @test gpu_backend(:rocm).kind == :amdgpu
    @test gpu_backend(:hip).kind == :amdgpu

    evaluator = make_gpu_evaluator(backend)
    Random.seed!(0x860)
    values = randn(ComplexF64, 4096)
    transformed = Array(QaoaXorsat.gpu_wht(gpu_array(backend, values)))
    @test transformed ≈ QaoaXorsat.wht(values) rtol=2e-12 atol=2e-10

    cases = [
        (TreeParams(2, 3, 1), QAOAAngles([0.6155], [0.3927]), -1),
        (TreeParams(2, 3, 3), QAOAAngles([0.31, -0.22, 0.14], [0.42, 0.33, -0.18]), -1),
        (TreeParams(3, 4, 3), QAOAAngles([0.27, -0.19, 0.11], [0.36, 0.24, -0.13]), 1),
    ]
    for (params, angles, clause_sign) in cases
        cpu_value, cpu_gamma, cpu_beta =
            basso_expectation_and_gradient(params, angles; clause_sign)
        amd_value, amd_gamma, amd_beta = evaluator(params, angles; clause_sign)
        @test amd_value ≈ cpu_value rtol=2e-10 atol=2e-11
        @test amd_gamma ≈ cpu_gamma rtol=2e-9 atol=2e-10
        @test amd_beta ≈ cpu_beta rtol=2e-9 atol=2e-10
    end
end
