"""
Test the optimized MaxCut GPU forward pass against the generic one.
"""

using Test
using QaoaXorsat
using Statistics, Printf

include(joinpath(@__DIR__, "gpu_test_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu_maxcut_opt.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu_forward.jl"))

@testset "Optimized MaxCut Forward" begin
    if !GPU_OK
        @test true
        return
    end

    @testset "matches generic GPU forward p=$p" for p in [1, 3, 5, 7, 9]
        params = TreeParams(2, 3, p)
        angles = QAOAAngles(randn(p) .* 0.3, randn(p) .* 0.2 .+ 0.4)

        val_generic = gpu_forward_value(params, angles, gpu_array; clause_sign=-1)
        val_opt = gpu_forward_maxcut(params, angles, gpu_array)

        @test val_opt ≈ val_generic atol=1e-3
    end

    @testset "matches CPU p=$p" for p in [1, 3, 5, 7]
        params = TreeParams(2, 3, p)
        angles = QAOAAngles(randn(p) .* 0.3, randn(p) .* 0.2 .+ 0.4)

        val_cpu = basso_expectation_normalized(params, angles; clause_sign=-1)
        val_opt = gpu_forward_maxcut(params, angles, gpu_array)

        @test val_opt ≈ val_cpu atol=1e-3
    end

    @testset "benchmark optimized vs generic at p=$p" for p in [9, 10]
        params = TreeParams(2, 3, p)
        angles = QAOAAngles(randn(p) .* 0.3, randn(p) .* 0.2 .+ 0.4)

        # Warmup
        gpu_forward_maxcut(params, angles, gpu_array)
        gpu_forward_value(params, angles, gpu_array; clause_sign=-1)

        opt_times = [(@elapsed gpu_forward_maxcut(params, angles, gpu_array)) for _ in 1:5]
        gen_times = [(@elapsed gpu_forward_value(params, angles, gpu_array; clause_sign=-1)) for _ in 1:5]
        cpu_times = [(@elapsed basso_expectation_normalized(params, angles; clause_sign=-1)) for _ in 1:3]

        opt_med = median(opt_times)
        gen_med = median(gen_times)
        cpu_med = median(cpu_times)

        @info @sprintf("p=%d: opt=%.4fs gen=%.4fs cpu=%.4fs  opt_speedup=%.2f× vs_cpu=%.2f×",
                       p, opt_med, gen_med, cpu_med, gen_med/opt_med, cpu_med/opt_med)

        @test true
    end
end
