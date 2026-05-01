"""
Gradient checkpointing tests — validates that checkpointed backward
produces identical gradients to full backward, with less memory.
"""

using Test
using QaoaXorsat

include(joinpath(@__DIR__, "gpu_test_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu_checkpointed.jl"))

@testset "Gradient Checkpointing" begin
    if !GPU_OK
        @test true
        return
    end

    @testset "checkpointed matches full k=$k D=$D p=$p ci=$ci" for
            (k, D, cs) in [(2, 3, -1), (3, 4, 1)],
            p in [3, 5, 7],
            ci in [1, 2, 3]  # different checkpoint intervals

        params = TreeParams(k, D, p)
        angles = QAOAAngles(randn(p) .* 0.3, randn(p) .* 0.2 .+ 0.4)

        # Full backward (reference)
        val_full, γg_full, βg_full = gpu_forward_backward(
            params, angles, gpu_array; clause_sign=cs)

        # Checkpointed backward
        val_cp, γg_cp, βg_cp = gpu_checkpointed_forward_backward(
            params, angles, gpu_array; clause_sign=cs,
            checkpoint_interval=ci)

        @test val_cp ≈ val_full atol=1e-4
        @test γg_cp ≈ γg_full atol=1e-2
        @test βg_cp ≈ βg_full atol=1e-2
    end

    @testset "checkpointed matches CPU p=$p" for p in [3, 5]
        params = TreeParams(2, 3, p)
        angles = QAOAAngles(randn(p) .* 0.3, randn(p) .* 0.2 .+ 0.4)

        cpu_val, cpu_γg, cpu_βg = basso_expectation_and_gradient(
            params, angles; clause_sign=-1)

        cp_val, cp_γg, cp_βg = gpu_checkpointed_forward_backward(
            params, angles, gpu_array; clause_sign=-1)

        @test cp_val ≈ cpu_val atol=1e-3
        @test cp_γg ≈ cpu_γg atol=5e-2
        @test cp_βg ≈ cpu_βg atol=5e-2
    end

    @testset "disk spillover" begin
        params = TreeParams(2, 3, 5)
        angles = QAOAAngles(randn(5) .* 0.3, randn(5) .* 0.2 .+ 0.4)

        disk_dir = mktempdir()

        val_disk, γg_disk, βg_disk = gpu_checkpointed_forward_backward(
            params, angles, gpu_array; clause_sign=-1,
            checkpoint_interval=2,
            disk_dir=disk_dir,
            max_ram_checkpoints=2)

        val_full, γg_full, βg_full = gpu_forward_backward(
            params, angles, gpu_array; clause_sign=-1)

        @test val_disk ≈ val_full atol=1e-4
        @test γg_disk ≈ γg_full atol=1e-2
        @test βg_disk ≈ βg_full atol=1e-2

        # Verify disk files were cleaned up
        @test isempty(readdir(disk_dir))
        rm(disk_dir; force=true)
    end

    @testset "auto interval (√p)" begin
        params = TreeParams(3, 4, 9)
        angles = QAOAAngles(randn(9) .* 0.3, randn(9) .* 0.2 .+ 0.4)

        # Auto interval should be ceil(√9) = 3
        val_auto, γg_auto, βg_auto = gpu_checkpointed_forward_backward(
            params, angles, gpu_array; clause_sign=1)

        val_full, γg_full, βg_full = gpu_forward_backward(
            params, angles, gpu_array; clause_sign=1)

        @test val_auto ≈ val_full atol=1e-3
        @test γg_auto ≈ γg_full atol=5e-2
        @test βg_auto ≈ βg_full atol=5e-2
    end
end
