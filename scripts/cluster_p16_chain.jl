#!/usr/bin/env julia
# Double64 + CPU-checkpointed warm-start chain for Stephen's SLURM cluster.
#
# Usage:
#   julia --project=. scripts/cluster_p16_chain.jl K D P_TARGET [POP] [GENS] [BURST] [SEED]

using QaoaXorsat
using Printf
using Random
using Dates
using DoubleFloats

k = parse(Int, get(ARGS, 1, "3"))
D = parse(Int, get(ARGS, 2, "4"))
p_target = parse(Int, get(ARGS, 3, "16"))
population = parse(Int, get(ARGS, 4, "100"))
generations = parse(Int, get(ARGS, 5, "10"))
burst_iters = parse(Int, get(ARGS, 6, "20"))
seed = parse(Int, get(ARGS, 7, "42"))

clause_sign = k == 2 ? -1 : 1

repo_root = normpath(joinpath(@__DIR__, ".."))
results_file = get(ENV, "QAOA_RESULTS_FILE",
    joinpath(repo_root, "results", "cluster-p16-k$(k)d$(D).csv"))
progress_file = get(ENV, "QAOA_PROGRESS_FILE", results_file * ".progress.log")
checkpoint_file = results_file * ".checkpoint"

checkpoint_disk_dir_env = get(ENV, "QAOA_CHECKPOINT_DIR", "")
checkpoint_disk_dir = isempty(checkpoint_disk_dir_env) ? nothing : checkpoint_disk_dir_env
max_ram_checkpoints = parse(Int, get(ENV, "QAOA_MAX_RAM_CHECKPOINTS", "4"))
swarm_concurrency = parse(Int, get(ENV, "QAOA_SWARM_CONCURRENCY", "1"))

mkpath(dirname(results_file))
mkpath(dirname(progress_file))
checkpoint_disk_dir !== nothing && mkpath(checkpoint_disk_dir)

struct WarmStartRecord
    p::Int
    value::Float64
    angles::QAOAAngles
    source::String
end

function utcstamp()
    Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
end

function progress(msg)
    line = "$(utcstamp()) $msg"
    println(line)
    flush(stdout)
    open(progress_file, "a") do io
        println(io, line)
        flush(io)
    end
end

valid_value(value::Float64) = QaoaXorsat.is_valid_qaoa_value(value) && value > 0.501

function parse_angles(gamma_field::AbstractString, beta_field::AbstractString)
    gamma = tryparse.(Float64, split(gamma_field, ';'))
    beta = tryparse.(Float64, split(beta_field, ';'))
    any(isnothing, gamma) && return nothing
    any(isnothing, beta) && return nothing
    length(gamma) == length(beta) || return nothing
    QAOAAngles(Float64.(gamma), Float64.(beta))
end

function maybe_better(candidate::WarmStartRecord, current::Union{WarmStartRecord,Nothing})
    current === nothing && return true
    candidate.p > current.p || (candidate.p == current.p && candidate.value > current.value)
end

function ingest_swarm_line(line::AbstractString, source::String, current)
    startswith(line, '#') && return current
    startswith(line, "k,") && return current
    fields = split(line, ',')
    length(fields) >= 8 || return current
    lk = tryparse(Int, fields[1]); lk === nothing && return current
    lD = tryparse(Int, fields[2]); lD === nothing && return current
    lp = tryparse(Int, fields[3]); lp === nothing && return current
    value = tryparse(Float64, fields[4]); value === nothing && return current
    lk == k && lD == D && valid_value(value) || return current
    angles = parse_angles(fields[7], fields[8])
    angles === nothing && return current
    candidate = WarmStartRecord(lp, value, angles, source)
    maybe_better(candidate, current) ? candidate : current
end

function ingest_warm_start_line(line::AbstractString, source::String, current)
    startswith(line, '#') && return current
    startswith(line, "k,") && return current
    fields = split(line, ',')
    length(fields) >= 7 || return current
    lk = tryparse(Int, fields[1]); lk === nothing && return current
    lD = tryparse(Int, fields[2]); lD === nothing && return current
    lp = tryparse(Int, fields[3]); lp === nothing && return current
    value = tryparse(Float64, fields[4]); value === nothing && return current
    lk == k && lD == D && valid_value(value) || return current
    angles = parse_angles(fields[6], fields[7])
    angles === nothing && return current
    candidate = WarmStartRecord(lp, value, angles, source)
    maybe_better(candidate, current) ? candidate : current
end

function best_from_file(path::String, current::Union{WarmStartRecord,Nothing}=nothing)
    isfile(path) || return current
    for line in eachline(path)
        if basename(path) == "warm-start-angles.csv"
            current = ingest_warm_start_line(line, path, current)
        else
            current = ingest_swarm_line(line, path, current)
        end
    end
    current
end

function find_warm_start()
    candidates = String[
        get(ENV, "QAOA_WARM_START_FILE", joinpath(repo_root, "results", "warm-start-angles.csv")),
        joinpath(repo_root, "results", "swarm-d64-k$(k)d$(D).csv"),
        joinpath(repo_root, "results", "swarm-k$(k)d$(D).csv"),
    ]
    best = nothing
    for path in unique(candidates)
        best = best_from_file(path, best)
    end
    best
end

function best_from_results_file()
    best_from_file(results_file, nothing)
end

function write_header_if_needed()
    isfile(results_file) && filesize(results_file) > 0 && return
    open(results_file, "w") do io
        println(io, "# cluster-p16 chain: k=$k, D=$D, target=$p_target, pop=$population, gens=$generations, burst=$burst_iters, seed=$seed")
        println(io, "# eval_eltype=Double64, checkpointed=true, max_ram_checkpoints=$max_ram_checkpoints")
        println(io, "k,D,p,ctilde,evals,wall_seconds,gamma,beta")
    end
end

function append_result(result::AngleOptimizationResult, p::Int, wall_seconds::Float64)
    gamma_str = join(string.(result.angles.γ), ';')
    beta_str = join(string.(result.angles.β), ';')
    open(results_file, "a") do io
        @printf(io, "%d,%d,%d,%.12f,%d,%.1f,%s,%s\n",
            k, D, p, result.value, result.evaluations, wall_seconds, gamma_str, beta_str)
        flush(io)
    end
end

function save_checkpoint(p::Int, value::Float64, angles::QAOAAngles)
    gamma_str = join(string.(angles.γ), ';')
    beta_str = join(string.(angles.β), ';')
    open(checkpoint_file, "w") do io
        @printf(io, "%d,%d,%d,%.12f,%s,%s\n", k, D, p, value, gamma_str, beta_str)
    end
end

function load_checkpoint()
    isfile(checkpoint_file) || return nothing
    line = strip(readline(checkpoint_file))
    fields = split(line, ',')
    length(fields) >= 6 || return nothing
    lk = tryparse(Int, fields[1]); lk === nothing && return nothing
    lD = tryparse(Int, fields[2]); lD === nothing && return nothing
    lp = tryparse(Int, fields[3]); lp === nothing && return nothing
    value = tryparse(Float64, fields[4]); value === nothing && return nothing
    lk == k && lD == D && valid_value(value) || return nothing
    angles = parse_angles(fields[5], fields[6])
    angles === nothing && return nothing
    WarmStartRecord(lp, value, angles, checkpoint_file)
end

function main()
    progress("=== QAOA cluster p16 chain ===")
    progress("pair k=$k D=$D target_p=$p_target population=$population generations=$generations burst=$burst_iters")
    progress("results=$results_file")
    progress("progress=$progress_file")
    progress("checkpoint_disk_dir=$(checkpoint_disk_dir === nothing ? "none" : checkpoint_disk_dir)")
    progress("max_ram_checkpoints=$max_ram_checkpoints swarm_concurrency=$swarm_concurrency threads=$(Threads.nthreads())")

    write_header_if_needed()

    resume = best_from_results_file()
    warm = resume === nothing ? find_warm_start() : resume

    if warm === nothing
        p_start = 1
        current_warm = QAOAAngles[]
        progress("no warm start found; starting at p=1")
    else
        p_start = warm.p + 1
        current_warm = [warm.angles]
        progress(@sprintf("warm start from p=%d c=%.12f source=%s", warm.p, warm.value, warm.source))
    end

    checkpoint = load_checkpoint()
    if checkpoint !== nothing && checkpoint.p == p_start
        current_warm = [checkpoint.angles]
        progress(@sprintf("using interrupted-depth checkpoint at p=%d c=%.12f", checkpoint.p, checkpoint.value))
    end

    if p_start > p_target
        progress("target already complete; nothing to do")
        return
    end

    for p in p_start:p_target
        params = TreeParams(k, D, p)
        starts = isempty(current_warm) ? QAOAAngles[] : [extend_angles(current_warm[1], p)]
        progress("starting p=$p with $(length(starts)) warm start(s)")
        started_at = time()

        result = swarm_optimize(
            params;
            clause_sign,
            population,
            generations,
            burst_iters,
            rng=MersenneTwister(seed + p),
            warm_starts=starts,
            eval_eltype=Double64,
            checkpointed=true,
            checkpoint_disk_dir,
            checkpoint_max_ram_checkpoints=max_ram_checkpoints,
            candidate_concurrency=swarm_concurrency,
            on_generation=(gen, best, npop, best_angles) -> begin
                progress(@sprintf("p=%d gen=%d best=%.12f population=%d", p, gen, best, npop))
                valid_value(best) && save_checkpoint(p, best, best_angles)
            end,
        )

        wall_seconds = time() - started_at
        append_result(result, p, wall_seconds)
        isfile(checkpoint_file) && rm(checkpoint_file)
        progress(@sprintf("completed p=%d c=%.12f evals=%d wall_seconds=%.1f", p, result.value, result.evaluations, wall_seconds))

        if valid_value(result.value)
            current_warm = [result.angles]
        else
            progress(@sprintf("p=%d produced invalid/weak value %.12f; next depth will use random starts", p, result.value))
            current_warm = QAOAAngles[]
        end
    end

    progress("DONE")
end

main()