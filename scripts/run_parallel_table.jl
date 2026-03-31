#!/usr/bin/env julia
#
# Run multiple (k,D) pairs in parallel, each as a separate Julia process.
# Designed for high-memory machines (M128s: 128 cores, 2TB RAM).
#
# Usage:
#   julia scripts/run_parallel_table.jl [P_MAX] [MAX_PARALLEL]
#
# MAX_PARALLEL defaults to fitting all pairs in available RAM.
# Each child process gets floor(total_threads / active_pairs) threads.
#
# Examples:
#   julia scripts/run_parallel_table.jl 13      # auto-detect parallelism
#   julia scripts/run_parallel_table.jl 14 4    # 4 pairs at a time
#   julia scripts/run_parallel_table.jl 12 15   # all 15 in parallel

using Dates
using Printf

p_max = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 13
max_parallel = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0

# All 15 (k,D) pairs
pairs = [
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 5), (4, 6), (4, 7), (4, 8),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (6, 8),
    (7, 8),
]

# Memory per pair (GB) at each depth — adjoint cache dominates
function adjoint_memory_gb(p)
    N = 2^(2p + 1)
    arrays = (p + 1) + p + p + 1 + 1 + 2  # B, child_hat, folded, f_table, kernel, scratch
    return arrays * N * 16 / 1024^3
end

# Detect available memory
total_memory_gb = try
    parse(Float64, strip(read(`free -g`, String) |> x -> split(x, '\n')[2] |> x -> split(x)[2]))
catch
    Sys.total_memory() / 1024^3
end
total_cores = Sys.CPU_THREADS

mem_per_pair = adjoint_memory_gb(p_max)
# Reserve 10% for OS + Julia overhead
usable_memory = total_memory_gb * 0.90
max_by_memory = max(1, floor(Int, usable_memory / mem_per_pair))

if max_parallel == 0
    max_parallel = min(max_by_memory, length(pairs))
end

threads_per_pair = max(1, div(total_cores, max_parallel))

@printf("=== QAOA Parallel Table Runner ===\n")
@printf("Pairs:          %d\n", length(pairs))
@printf("P_MAX:          %d\n", p_max)
@printf("Memory/pair:    %.1f GB\n", mem_per_pair)
@printf("Total memory:   %.0f GB\n", total_memory_gb)
@printf("Total cores:    %d\n", total_cores)
@printf("Max parallel:   %d\n", max_parallel)
@printf("Threads/pair:   %d\n", threads_per_pair)
@printf("Est. total mem: %.0f GB (%.0f%% of %.0f GB)\n",
    min(max_parallel, length(pairs)) * mem_per_pair,
    min(max_parallel, length(pairs)) * mem_per_pair / total_memory_gb * 100,
    total_memory_gb)
println()

# Create results directory
mkpath(joinpath(@__DIR__, "..", "results", "logs"))

# Launch pairs in batches
remaining = copy(pairs)
active = Dict{Tuple{Int,Int}, Base.Process}()
completed = Tuple{Int,Int}[]
failed = Tuple{Int,Int}[]

function launch_pair(k, d, threads)
    logfile = joinpath(@__DIR__, "..", "results", "logs",
        "parallel-k$(k)-d$(d)-p$(p_max)-$(Dates.format(now(), "yyyymmddTHHMMSS")).log")
    cmd = `julia --project=. -t $threads scripts/optimize_qaoa.jl $k $d 1 $p_max 2 320 1234 true adjoint`
    @printf("  Launching (k=%d, D=%d) with %d threads → %s\n", k, d, threads, basename(logfile))
    proc = open(cmd, logfile, write=true)
    return proc
end

@printf("Starting at %s\n\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

while !isempty(remaining) || !isempty(active)
    # Launch new pairs if we have capacity
    while !isempty(remaining) && length(active) < max_parallel
        k, d = popfirst!(remaining)
        proc = launch_pair(k, d, threads_per_pair)
        active[(k, d)] = proc
    end

    # Check for completed processes
    for ((k, d), proc) in collect(active)
        if !process_running(proc)
            wait(proc)
            if proc.exitcode == 0
                @printf("  ✓ (k=%d, D=%d) completed successfully\n", k, d)
                push!(completed, (k, d))
            else
                @printf("  ✗ (k=%d, D=%d) FAILED (exit=%d)\n", k, d, proc.exitcode)
                push!(failed, (k, d))
            end
            delete!(active, (k, d))
        end
    end

    # Status update every 60 seconds
    if !isempty(active)
        @printf("  [%s] Active: %d, Remaining: %d, Completed: %d, Failed: %d\n",
            Dates.format(now(), "HH:MM:SS"),
            length(active), length(remaining), length(completed), length(failed))
        sleep(60)
    end
end

@printf("\n=== Complete at %s ===\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
@printf("Completed: %d\n", length(completed))
@printf("Failed:    %d\n", length(failed))
if !isempty(failed)
    println("Failed pairs: ", join(["($(k),$(d))" for (k,d) in failed], ", "))
end
