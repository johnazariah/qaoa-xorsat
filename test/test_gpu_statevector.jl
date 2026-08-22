using QaoaXorsat
using Test

include("statevector_test_utils.jl")

@testset "Device statevector evaluator" begin
    cpu_backend = gpu_backend(:cpu)

    @testset "ComplexF64 CPU oracle N=$N p=$p" for N in (4, 6, 8), p in (1, 2, 3)
        diagonal, terms = _test_diagonal(N)
        if p == 1
            applied = Vector{ComplexF64}(undef, 1 << N)
            apply_terms!(applied, terms, ones(ComplexF64, 1 << N), N)
            @test diagonal == real.(applied)
        end
        angles = QAOAAngles(
            [0.17 * layer - 0.04 for layer in 1:p],
            [0.11 * layer + 0.07 for layer in 1:p],
        )
        reference_value, reference_gradient =
            _cpu_fast_value_gradient(N, diagonal, angles)
        evaluator = make_statevector_evaluator(cpu_backend, N, terms)
        value, gamma_gradient, beta_gradient = evaluator(angles)
        gradient = [gamma_gradient; beta_gradient]

        @test isapprox(value, reference_value; atol=1e-10, rtol=1e-10)
        @test isapprox(gradient, reference_gradient; atol=1e-10, rtol=1e-10)
        stats = statevector_execution_stats(evaluator)
        @test stats.backend == :cpu
        @test stats.complex_type == ComplexF64
        @test stats.kernel_launches > 0
        @test stats.predicted_live_bytes > 0
        @test stats.reservation_after <= stats.memory_cap_bytes
        @test !isempty(stats.source_revision)
        @test synchronize(evaluator) === evaluator
        if N == 4 && p == 1
            evaluator(angles)
            updated = statevector_execution_stats(evaluator)
            @test updated.evaluation_count == 2
            @test updated.first_evaluation_seconds !== nothing
            @test updated.steady_state_seconds > 0
            @test updated.synchronized_launch_seconds > 0
            @test updated.kernel_execution_seconds !== nothing
            @test updated.kernel_timing_source == :synchronized_wall
        end
    end

    @testset "ComplexF32 tolerance is separate" begin
        N = 8
        p = 3
        diagonal, _ = _test_diagonal(N)
        angles = QAOAAngles([0.13, -0.21, 0.37], [0.19, 0.28, -0.16])
        reference_value, reference_gradient =
            _cpu_fast_value_gradient(N, diagonal, angles)
        cpu32 = GPUBackend(:cpu, nothing, ComplexF32, "KA CPU ComplexF32", nothing)
        evaluator = make_statevector_evaluator(cpu32, N, diagonal)
        value, gamma_gradient, beta_gradient = evaluator(angles)
        @test isapprox(value, reference_value; atol=5e-5, rtol=5e-5)
        @test isapprox(
            [gamma_gradient; beta_gradient],
            reference_gradient;
            atol=2e-4,
            rtol=2e-4,
        )
    end

    @testset "admission and fail-closed behavior" begin
        cpu_admission = statevector_memory_admission(cpu_backend, 4)
        @test cpu_admission.backend == :cpu
        @test cpu_admission.dimension == 16
        @test cpu_admission.predicted_live_bytes == 120 * 16
        @test cpu_admission.admission_total_bytes ==
            cpu_admission.reported_total_bytes

        telemetry = GPUMemoryStatus(
            47_102_148_608 - 43_117_445_120,
            47_102_148_608,
            43_117_445_120,
        )
        @test_throws GPUBackendError QaoaXorsat._admit_statevector_memory(
            GPUBackend(:amdgpu, identity, ComplexF64, "test", "test"),
            telemetry,
            0,
            0.8,
        )
        @test_throws GPUBackendError QaoaXorsat._assert_observed_memory(
            :amdgpu,
            43_117_445_120,
            floor(Int, 0.8 * 47_102_148_608),
        )
        @test QaoaXorsat._statevector_memory_bytes(256, ComplexF64) ==
            120 * 256
        @test QaoaXorsat._statevector_memory_bytes(256, ComplexF32) ==
            60 * 256

        amd_backend = GPUBackend(
            :amdgpu,
            identity,
            ComplexF64,
            "AMD UMA regression",
            "test",
        )
        device_total = 47_102_148_608
        host_total = 33_981_000_000
        uma_status = GPUMemoryStatus(device_total, device_total, 0)
        @test QaoaXorsat._statevector_admission_total(
            amd_backend,
            uma_status,
            host_total,
        ) == host_total
        @test QaoaXorsat._admit_statevector_memory(
            amd_backend,
            uma_status,
            0,
            0.8;
            host_total_bytes=host_total,
        ) == floor(Int, 0.8 * host_total)
        n28_predicted = QaoaXorsat._statevector_memory_bytes(
            1 << 28,
            ComplexF64,
        )
        @test n28_predicted == 32_212_254_720
        @test_throws GPUBackendError QaoaXorsat._admit_statevector_memory(
            amd_backend,
            uma_status,
            n28_predicted,
            0.8;
            host_total_bytes=host_total,
        )

        @test_throws ArgumentError make_statevector_evaluator(
            cpu_backend,
            4,
            zeros(16);
            memory_fraction=0.81,
        )
        oom_backend = GPUBackend(
            :cuda,
            _ -> throw(OutOfMemoryError()),
            ComplexF64,
            "OOM test",
            "test",
        )
        old_status = get(QaoaXorsat._GPU_MEMORY_STATUS_FACTORIES, :cuda, nothing)
        QaoaXorsat._register_gpu_memory_status!(
            :cuda,
            () -> GPUMemoryStatus(1 << 30, 1 << 30, 0),
        )
        try
            error = try
                make_statevector_evaluator(oom_backend, 4, zeros(16))
                nothing
            catch caught
                caught
            end
            @test error isa GPUBackendError
            @test error.kind == :cuda
            @test occursin("device validation failed", error.message)
        finally
            if old_status === nothing
                delete!(QaoaXorsat._GPU_MEMORY_STATUS_FACTORIES, :cuda)
            else
                QaoaXorsat._GPU_MEMORY_STATUS_FACTORIES[:cuda] = old_status
            end
        end
        @test_throws ArgumentError make_statevector_evaluator(
            cpu_backend,
            4,
            [x_term(1)],
        )
    end
end
