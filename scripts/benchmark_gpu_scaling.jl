#!/usr/bin/env julia

using AMDGPU
using Dates
using Printf
using QaoaXorsat

const CPU_PRACTICAL_LIMIT_SECONDS = 60.0
const CPU_MEASUREMENT_LIMIT_SECONDS = 180.0
const CPU_MEMORY_FRACTION_LIMIT = 0.60
const GPU_MEMORY_FRACTION_LIMIT = 0.80
const COMPLEX_BYTES = sizeof(ComplexF64)
const CPU_ADJOINT_VECTORS = 40

function benchmark_angles(p)
    positions = collect(1:p) ./ (p + 1)
    QAOAAngles(
        0.35 .* sinpi.(positions),
        0.30 .+ 0.10 .* cospi.(positions),
    )
end

configuration_count(p) = 4^p
state_bytes(p) = configuration_count(p) * COMPLEX_BYTES
cpu_adjoint_bytes(p) = CPU_ADJOINT_VECTORS * state_bytes(p)

function projected_cpu_seconds(cpu_times)
    isempty(cpu_times) && return 0.0
    length(cpu_times) == 1 && return 4.0 * last(cpu_times)
    ratios = cpu_times[2:end] ./ cpu_times[1:end-1]
    last(cpu_times) * max(4.0, sum(ratios) / length(ratios))
end

function timed(call)
    result = nothing
    elapsed = @elapsed result = call()
    elapsed, result
end

function main()
    output = length(ARGS) >= 1 ? ARGS[1] :
        joinpath(@__DIR__, "..", "results", "amd-gpu-scaling.csv")
    sweep_max = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    large_p = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 12
    sweep_max >= 3 || error("sweep_max must be at least 3")
    large_p > sweep_max || error("large_p must be greater than sweep_max")

    backend_compile_seconds, backend = timed(() -> gpu_backend(:amdgpu))
    evaluator = make_gpu_evaluator(backend)
    params_warmup = TreeParams(2, 3, 1)
    angles_warmup = benchmark_angles(1)
    kernel_compile_seconds, _ = timed(
        () -> evaluator(params_warmup, angles_warmup; clause_sign=-1),
    )
    warm_steady_seconds, _ = timed(
        () -> evaluator(params_warmup, angles_warmup; clause_sign=-1),
    )
    cpu_compile_seconds, _ = timed(
        () -> basso_expectation_and_gradient(
            params_warmup, angles_warmup; clause_sign=-1,
        ),
    )
    cpu_warm_steady_seconds, _ = timed(
        () -> basso_expectation_and_gradient(
            params_warmup, angles_warmup; clause_sign=-1,
        ),
    )

    mkpath(dirname(output))
    cpu_times = Float64[]
    crossover_p = nothing
    last_gpu_p = 0
    last_gpu_reserved = 0
    open(output, "w") do io
        println(io, "# generated_utc=$(now(UTC))")
        println(io, "# device=$(backend.device)")
        println(io, "# amdgpu_version=$(pkgversion(AMDGPU))")
        println(io, "# backend_compile_seconds=$(backend_compile_seconds)")
        println(io, "# kernel_compile_warmup_seconds=$(kernel_compile_seconds)")
        println(io, "# warm_steady_seconds=$(warm_steady_seconds)")
        println(io, "# cpu_compile_warmup_seconds=$(cpu_compile_seconds)")
        println(io, "# cpu_warm_steady_seconds=$(cpu_warm_steady_seconds)")
        println(io, "# cpu_practical_limit_seconds=$(CPU_PRACTICAL_LIMIT_SECONDS)")
        println(io, "# cpu_measurement_limit_seconds=$(CPU_MEASUREMENT_LIMIT_SECONDS)")
        println(io, "# cpu_memory_fraction_limit=$(CPU_MEMORY_FRACTION_LIMIT)")
        println(io, "# gpu_memory_fraction_limit=$(GPU_MEMORY_FRACTION_LIMIT)")
        println(io, "# host_memory_bytes=$(Sys.total_memory())")
        _, hip_total_bytes = AMDGPU.info()
        println(io, "# hip_total_bytes=$(hip_total_bytes)")
        println(io,
            "depth_p,configuration_count,physical_problem_n,state_bytes," *
            "cpu_adjoint_estimated_bytes,cpu_status,amd_status," *
            "cpu_first_seconds,cpu_seconds," *
            "amd_first_seconds,amd_seconds," *
            "speedup,reference_kind,value_abs_error,gamma_max_abs_error," *
            "beta_max_abs_error,amd_value,hip_pool_reserved_bytes",
        )

        for p in vcat(collect(3:sweep_max), [large_p])
            params = TreeParams(2, 3, p)
            angles = benchmark_angles(p)
            estimated_cpu_bytes = cpu_adjoint_bytes(p)
            predicted_cpu = projected_cpu_seconds(cpu_times)
            cpu_allowed = estimated_cpu_bytes <=
                CPU_MEMORY_FRACTION_LIMIT * Sys.total_memory() &&
                (p <= 3 || predicted_cpu <= CPU_MEASUREMENT_LIMIT_SECONDS)

            cpu_status = cpu_allowed ? "measured" :
                estimated_cpu_bytes > CPU_MEMORY_FRACTION_LIMIT * Sys.total_memory() ?
                    "skipped_memory_limit" : "skipped_measurement_limit"
            cpu_first_seconds = NaN
            cpu_seconds = NaN
            cpu_result = nothing
            if cpu_allowed
                cpu_first_seconds, _ = timed(
                    () -> basso_expectation_and_gradient(
                        params, angles; clause_sign=-1,
                    ),
                )
                cpu_seconds, cpu_result = timed(
                    () -> basso_expectation_and_gradient(
                        params, angles; clause_sign=-1,
                    ),
                )
                push!(cpu_times, cpu_seconds)
                if cpu_seconds > CPU_PRACTICAL_LIMIT_SECONDS
                    cpu_status = "measured_impractical_runtime"
                end
            end

            _, hip_total_bytes = AMDGPU.info()
            projected_gpu_reserved = last_gpu_reserved == 0 ? 0 :
                last_gpu_reserved * 4^(p - last_gpu_p)
            gpu_allowed = projected_gpu_reserved <=
                GPU_MEMORY_FRACTION_LIMIT * hip_total_bytes
            amd_status = gpu_allowed ? "measured" : "skipped_memory_safety"
            amd_first_seconds = NaN
            amd_seconds = NaN
            amd_result = nothing
            pool_reserved = last_gpu_reserved
            if gpu_allowed
                AMDGPU.reclaim()
                GC.gc(true)
                amd_first_seconds, _ = timed(
                    () -> evaluator(params, angles; clause_sign=-1),
                )
                amd_seconds, amd_result = timed(
                    () -> evaluator(params, angles; clause_sign=-1),
                )
                pool_reserved = AMDGPU.cached_memory()
                last_gpu_p = p
                last_gpu_reserved = pool_reserved
            end

            if cpu_result === nothing || amd_result === nothing
                reference_kind = "none_resource_limited"
                value_error = NaN
                gamma_error = NaN
                beta_error = NaN
                speedup = NaN
            else
                reference_kind = "cpu_full_adjoint"
                value_error = abs(amd_result[1] - cpu_result[1])
                gamma_error = maximum(abs.(amd_result[2] .- cpu_result[2]))
                beta_error = maximum(abs.(amd_result[3] .- cpu_result[3]))
                speedup = cpu_seconds / amd_seconds
                if crossover_p === nothing && speedup > 1
                    crossover_p = p
                end
            end

            @printf(
                io,
                "%d,%d,not_applicable,%d,%d,%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%s,%.9g,%.9g,%.9g,%.17g,%d\n",
                p,
                configuration_count(p),
                state_bytes(p),
                estimated_cpu_bytes,
                cpu_status,
                amd_status,
                cpu_first_seconds,
                cpu_seconds,
                amd_first_seconds,
                amd_seconds,
                speedup,
                reference_kind,
                value_error,
                gamma_error,
                beta_error,
                amd_result === nothing ? NaN : amd_result[1],
                pool_reserved,
            )
            flush(io)
            @info "completed scaling point" p cpu_status amd_status cpu_seconds amd_seconds pool_reserved
        end
        println(io, "# measured_cpu_to_amd_crossover_p=$(something(crossover_p, "not_observed"))")
    end
    println(output)
end

main()
