using AMDGPU
using QaoaXorsat
using Random
using Test

include("statevector_test_utils.jl")

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

    @testset "Physical statevector N=$N p=$p" for N in (4, 6, 8), p in (1, 2, 3)
        diagonal, terms = _test_diagonal(N)
        angles = QAOAAngles(
            [0.17 * layer - 0.04 for layer in 1:p],
            [0.11 * layer + 0.07 for layer in 1:p],
        )
        reference_value, reference_gradient =
            _cpu_fast_value_gradient(N, diagonal, angles)
        statevector = make_statevector_evaluator(backend, N, terms)
        launches_before = statevector_execution_stats(statevector).kernel_launches
        value, gamma_gradient, beta_gradient = statevector(angles)
        stats = statevector_execution_stats(statevector)

        @test isapprox(value, reference_value; atol=1e-10, rtol=1e-10)
        @test isapprox(
            [gamma_gradient; beta_gradient],
            reference_gradient;
            atol=1e-10,
            rtol=1e-10,
        )
        @test stats.backend == :amdgpu
        @test stats.device !== nothing
        @test occursin("Radeon", stats.device)
        @test stats.complex_type == ComplexF64
        @test stats.kernel_launches > launches_before
        @test stats.kernel_execution_seconds !== nothing
        @test stats.kernel_execution_seconds > 0
        @test stats.kernel_timing_source == :hip_events
        @test stats.reservation_after <= stats.memory_cap_bytes
        @test synchronize(statevector) === statevector
    end
end
