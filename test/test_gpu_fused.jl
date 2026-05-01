"""
Tests for fused branch-step kernels.
Validates that gpu_branch_step! produces identical results to the
unfused sequence of operations.
"""

using Test
using QaoaXorsat

include(joinpath(@__DIR__, "gpu_test_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu_fused.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu_forward.jl"))

@testset "Fused Branch Step" begin
    if !GPU_OK
        @test true
        return
    end

    @testset "fused matches unfused k=$k D=$D p=$p" for
            (k, D) in [(2, 3), (3, 4), (5, 6)],
            p in [3, 5]

        arity = k - 1
        degree = D - 1
        N = QaoaXorsat.basso_configuration_count(p)
        GT = GPU_CT
        threshold = GPU_RT(1e15)

        # Random tensors
        B = gpu_array(randn(ComplexF64, N))
        f_table = gpu_array(randn(ComplexF64, N))
        kernel_hat = gpu_array(randn(ComplexF64, N))

        # ── Unfused path ──────────────────────────────────────────
        scratch_u = similar(B)
        gpu_elemwise_mul!(scratch_u, f_table, B)
        gpu_wht!(scratch_u)

        ch_scale_u = Float64(maximum(abs.(scratch_u)))
        if ch_scale_u > Float64(threshold)
            scratch_u .*= GPU_RT(1.0 / ch_scale_u)
        else
            ch_scale_u = 1.0
        end

        child_hat_u = copy(scratch_u)
        gpu_fold!(scratch_u, kernel_hat, child_hat_u, arity)
        gpu_iwht!(scratch_u)

        fld_scale_u = Float64(maximum(abs.(scratch_u)))
        if fld_scale_u > Float64(threshold)
            scratch_u .*= GPU_RT(1.0 / fld_scale_u)
        else
            fld_scale_u = 1.0
        end

        B_next_u = gpu_complex_power(scratch_u, degree)

        # ── Fused path ────────────────────────────────────────────
        B_next_f = similar(B)
        scratch_f = similar(B)
        ch_scale_f, fld_scale_f = gpu_branch_step!(
            B_next_f, scratch_f, B, f_table, kernel_hat,
            arity, degree, threshold)

        # ── Compare ───────────────────────────────────────────────
        B_next_u_cpu = ComplexF64.(Array(B_next_u))
        B_next_f_cpu = ComplexF64.(Array(B_next_f))

        @test B_next_f_cpu ≈ B_next_u_cpu atol=1e-2 * maximum(abs.(B_next_u_cpu))
        @test ch_scale_f ≈ ch_scale_u rtol=1e-4
        @test fld_scale_f ≈ fld_scale_u rtol=1e-4
    end

    @testset "fused forward matches gpu_forward_value" for p in [3, 5]
        params = TreeParams(2, 3, p)
        angles = QAOAAngles(randn(p) .* 0.3, randn(p) .* 0.2 .+ 0.4)

        # Reference: unfused GPU forward
        val_unfused = gpu_forward_value(params, angles, gpu_array; clause_sign=-1)

        # Fused forward: manually build using gpu_branch_step!
        k, D = 2, 3
        arity, degree = k-1, D-1
        bit_count = QaoaXorsat.basso_bit_count(p)
        N = QaoaXorsat.basso_configuration_count(p)

        gamma_full = QaoaXorsat.build_gamma_full_vector(angles)
        trig_table = QaoaXorsat.basso_trig_table(angles)
        f_table_cpu = QaoaXorsat._basso_f_table_fast(trig_table, bit_count, N, Float64)

        half = 0.5
        kernel_cpu = [complex(cos(half * QaoaXorsat._phase_dot(gamma_full, c, bit_count)))
                      for c in 0:N-1]
        kernel_hat_cpu = QaoaXorsat.wht!(copy(kernel_cpu))

        f_gpu = gpu_array(f_table_cpu)
        kh_gpu = gpu_array(kernel_hat_cpu)
        threshold = GPU_RT(1e15)

        B = gpu_array(ones(ComplexF64, N))
        B_next = similar(B)
        scratch = similar(B)
        log_s = 0.0

        for t in 1:p
            ch_s, fld_s = gpu_branch_step!(B_next, scratch, B, f_gpu, kh_gpu,
                                           arity, degree, threshold)
            log_s = arity * degree * log_s + arity * log(ch_s) + degree * log(fld_s)
            B, B_next = B_next, B  # swap
        end

        # Root fold (same as unfused)
        phase_args = [QaoaXorsat._phase_dot(gamma_full, c, bit_count) for c in 0:N-1]
        root_parity = gpu_array(complex.(Float64[QaoaXorsat.basso_root_parity(c, p) for c in 0:N-1]))
        root_kernel = gpu_array([complex(0.0, sin(half * (-1.0) * phase_args[c+1])) for c in 0:N-1])

        root_msg = similar(B)
        backend = KernelAbstractions.get_backend(B)
        mul3! = _elemwise_mul3_kernel!(backend)
        mul3!(root_msg, root_parity, f_gpu, B; ndrange=N)
        KernelAbstractions.synchronize(backend)

        gpu_wht!(root_msg)
        mh_scale = Float64(maximum(abs.(root_msg)))
        if mh_scale > Float64(threshold)
            root_msg .*= GPU_RT(1.0 / mh_scale)
        else
            mh_scale = 1.0
        end

        msg_power = gpu_complex_power(root_msg, k)
        gpu_iwht!(msg_power)

        prod_tmp = similar(msg_power)
        gpu_elemwise_mul!(prod_tmp, root_kernel, msg_power)
        S_norm = ComplexF64(sum(Array(prod_tmp)))

        log_total = k * (log_s + log(mh_scale))
        re_S = real(S_norm)
        if re_S != 0 && isfinite(log_total)
            val_fused = (1 + (-1) * copysign(exp(log_total + log(abs(re_S))), re_S)) / 2
        else
            val_fused = 0.5
        end

        @test val_fused ≈ val_unfused atol=1e-3
    end

    @testset "benchmark fused vs unfused" begin
        p = 9
        params = TreeParams(2, 3, p)
        N = QaoaXorsat.basso_configuration_count(p)
        arity, degree = 1, 2
        threshold = GPU_RT(1e15)

        B = gpu_array(randn(ComplexF64, N))
        f_table = gpu_array(randn(ComplexF64, N))
        kernel_hat = gpu_array(randn(ComplexF64, N))
        B_next = similar(B)
        scratch = similar(B)

        # Warmup
        gpu_branch_step!(B_next, scratch, B, f_table, kernel_hat, arity, degree, threshold)

        # Fused timing
        fused_times = Float64[]
        for _ in 1:5
            t = @elapsed gpu_branch_step!(B_next, scratch, B, f_table, kernel_hat,
                                          arity, degree, threshold)
            push!(fused_times, t)
        end

        # Unfused timing
        unfused_times = Float64[]
        for _ in 1:5
            t = @elapsed begin
                gpu_elemwise_mul!(scratch, f_table, B)
                gpu_wht!(scratch)
                ch = copy(scratch)
                gpu_fold!(scratch, kernel_hat, ch, arity)
                gpu_iwht!(scratch)
                copyto!(B_next, gpu_complex_power(scratch, degree))
            end
            push!(unfused_times, t)
        end

        using Statistics
        fused_med = median(fused_times)
        unfused_med = median(unfused_times)
        speedup = unfused_med / fused_med

        @info "Branch step at p=9: fused=$(round(fused_med*1000,digits=1))ms " *
              "unfused=$(round(unfused_med*1000,digits=1))ms " *
              "speedup=$(round(speedup,digits=2))×"

        @test true  # benchmark always passes
    end
end
