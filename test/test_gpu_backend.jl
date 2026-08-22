using Test
using QaoaXorsat

@testset "GPU backend API" begin
    cpu = gpu_backend(:cpu)
    @test cpu.kind == :cpu
    @test cpu.complex_type == ComplexF64
    @test validate_gpu_backend(cpu)
    @test gpu_backend_available(:cpu)
    @test make_gpu_evaluator(cpu) === nothing
    @test_throws GPUBackendError gpu_array(cpu, [1.0])

    @test_throws ArgumentError gpu_backend(:unknown)
    @test gpu_backend(:auto; validate=false) isa GPUBackend
end
