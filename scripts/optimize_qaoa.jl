#!/usr/bin/env julia

using Dates
using Printf
using Random
using QaoaXorsat

function parse_bool_flag(value::String)
    lowercase(value) in ("1", "true", "yes", "y") && return true
    lowercase(value) in ("0", "false", "no", "n") && return false
    error("invalid boolean flag: $(value)")
end

function parse_int(name::String, value::String)
    try
        parse(Int, value)
    catch
        error("invalid $(name): $(value)")
    end
end

function usage()
    println(stderr, "Usage: julia --project=. scripts/optimize_qaoa.jl K D P_MIN P_MAX [RESTARTS] [MAXITERS] [SEED] [PRESERVE] [AUTODIFF]")
    println(stderr, "  AUTODIFF: adjoint (default), forward, or finite")
end

json_escape(text::AbstractString) = escape_string(text)

function ensure_directory(path::AbstractString)
    mkpath(path)
    path
end

function git_commit_sha()
    try
        readchomp(`git rev-parse HEAD`)
    catch
        "unknown"
    end
end

function git_branch_name()
    try
        readchomp(`git rev-parse --abbrev-ref HEAD`)
    catch
        "unknown"
    end
end

function git_is_dirty()
    try
        !isempty(readchomp(`git status --porcelain`))
    catch
        true
    end
end

run_kind() = get(ENV, "QAOA_RUN_KIND", "experiment")

runner_label() = get(ENV, "QAOA_RUNNER_LABEL", get(ENV, "RUNNER_NAME", "unknown"))

reliability_artifacts_dir() = get(ENV, "QAOA_RELIABILITY_DIR", "")

function reliability_artifact_names(dir::AbstractString)
    isempty(dir) && return String[]
    isdir(dir) || return String[]
    sort(filter(name -> isfile(joinpath(dir, name)), readdir(dir)))
end

function format_angle_list(values)
    join(string.(values), ';')
end

function result_csv_header()
    "run_id,run_kind,runner_label,timestamp_utc,git_commit,git_branch,git_dirty,k,D,p,clause_sign,restarts,maxiters,seed,value,wall_time_seconds,best_start_wall_time_seconds,evaluations,starts,iterations,converged,retry_count,best_start_kind,gamma,beta"
end

function result_csv_row(run_id, run_kind_value, runner_label_value, timestamp_utc, git_commit, git_branch, git_dirty, k, D, clause_sign, restarts, maxiters, seed, result)
    @sprintf(
        "%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%.12f,%.6f,%.6f,%d,%d,%d,%s,%d,%s,%s,%s",
        run_id,
        run_kind_value,
        runner_label_value,
        timestamp_utc,
        git_commit,
        git_branch,
        string(git_dirty),
        k,
        D,
        depth(result.angles),
        clause_sign,
        result.restarts,
        result.maxiters,
        seed,
        result.value,
        result.wall_time_seconds,
        result.best_start_wall_time_seconds,
        result.evaluations,
        result.starts,
        result.iterations,
        string(result.converged),
        result.retry_count,
        string(result.best_start_kind),
        format_angle_list(result.angles.γ),
        format_angle_list(result.angles.β),
    )
end

function write_results_csv(file_path, rows)
    open(file_path, "w") do io
        println(io, result_csv_header())
        foreach(row -> println(io, row), rows)
    end
end

function append_results_csv_row(file_path, row)
    file_exists = isfile(file_path)
    open(file_path, file_exists ? "a" : "w") do io
        file_exists || println(io, result_csv_header())
        println(io, row)
    end
end

function manifest_string_field(file_path, field_name; default="")
    if !isfile(file_path)
        return default
    end

    pattern = Regex("^\\s*\"" * field_name * "\": \"(.*)\",?")
    for line in eachline(file_path)
        match_result = match(pattern, line)
        isnothing(match_result) || return String(match_result.captures[1])
    end

    default
end

function normalize_results_row(header, row, run_kind_value, runner_label_value)
    header == result_csv_header() && return row

    legacy_header = "run_id,timestamp_utc,git_commit,git_branch,git_dirty,k,D,p,clause_sign,restarts,maxiters,seed,value,wall_time_seconds,best_start_wall_time_seconds,evaluations,starts,iterations,converged,gamma,beta"
    if header == legacy_header
        fields = split(row, ',')
        return join(vcat(fields[1:1], [run_kind_value, runner_label_value], fields[2:19], ["0", "unknown"], fields[20:end]), ',')
    end

    previous_header = "run_id,run_kind,runner_label,timestamp_utc,git_commit,git_branch,git_dirty,k,D,p,clause_sign,restarts,maxiters,seed,value,wall_time_seconds,best_start_wall_time_seconds,evaluations,starts,iterations,converged,gamma,beta"
    if header == previous_header
        fields = split(row, ',')
        return join(vcat(fields[1:21], ["0", "unknown"], fields[22:end]), ',')
    end

    error("unsupported results.csv header: $(header)")
end

function rebuild_results_index(file_path, runs_dir)
    rows = String[]
    if isdir(runs_dir)
        for run_name in sort(readdir(runs_dir))
            run_dir = joinpath(runs_dir, run_name)
            isdir(run_dir) || continue

            manifest_path = joinpath(run_dir, "manifest.json")
            results_path = joinpath(run_dir, "results.csv")
            isfile(results_path) || continue

            run_kind_value = manifest_string_field(manifest_path, "run_kind"; default="experiment")
            runner_label_value = manifest_string_field(manifest_path, "runner_label"; default="unknown")

            file_lines = readlines(results_path)
            isempty(file_lines) && continue
            header = first(file_lines)
            for row in file_lines[2:end]
                push!(rows, normalize_results_row(header, row, run_kind_value, runner_label_value))
            end
        end
    end

    write_results_csv(file_path, rows)
end

function append_results_index(file_path, runs_dir, rows)
    file_exists = isfile(file_path)
    rebuilt = false
    if file_exists
        first_line = open(file_path, "r") do io
            eof(io) ? "" : readline(io)
        end
        if first_line != result_csv_header()
            rebuild_results_index(file_path, runs_dir)
            rebuilt = true
        end
    end

    rebuilt && return nothing

    open(file_path, file_exists ? "a" : "w") do io
        file_exists || println(io, result_csv_header())
        foreach(row -> println(io, row), rows)
    end
end

function write_manifest_json(file_path; run_id, run_kind_value, runner_label_value, reliability_dir, reliability_files, timestamp_utc, git_commit, git_branch, git_dirty, k, D, p_min, p_max, clause_sign, restarts, maxiters, seed, preserve, result_count)
    open(file_path, "w") do io
        write(io, "{\n")
        write(io, "  \"run_id\": \"$(json_escape(run_id))\",\n")
        write(io, "  \"run_kind\": \"$(json_escape(run_kind_value))\",\n")
        write(io, "  \"runner_label\": \"$(json_escape(runner_label_value))\",\n")
        write(io, "  \"timestamp_utc\": \"$(json_escape(timestamp_utc))\",\n")
        write(io, "  \"git_commit\": \"$(json_escape(git_commit))\",\n")
        write(io, "  \"git_branch\": \"$(json_escape(git_branch))\",\n")
        write(io, "  \"git_dirty\": $(string(git_dirty)),\n")
        write(io, "  \"reliability_artifacts_dir\": \"$(json_escape(reliability_dir))\",\n")
        write(io, "  \"reliability_artifacts\": [")
        for (index, file_name) in enumerate(reliability_files)
            index > 1 && write(io, ", ")
            write(io, "\"$(json_escape(file_name))\"")
        end
        write(io, "],\n")
        write(io, "  \"k\": $(k),\n")
        write(io, "  \"D\": $(D),\n")
        write(io, "  \"p_min\": $(p_min),\n")
        write(io, "  \"p_max\": $(p_max),\n")
        write(io, "  \"clause_sign\": $(clause_sign),\n")
        write(io, "  \"restarts\": $(restarts),\n")
        write(io, "  \"maxiters\": $(maxiters),\n")
        write(io, "  \"seed\": $(seed),\n")
        write(io, "  \"preserve\": $(string(preserve)),\n")
        write(io, "  \"result_count\": $(result_count)\n")
        write(io, "}\n")
    end
end

mutable struct RunPreservationContext
    run_dir::String
    run_id::String
    run_kind_value::String
    runner_label_value::String
    reliability_dir::String
    reliability_files::Vector{String}
    timestamp_utc::String
    git_commit::String
    git_branch::String
    git_dirty::Bool
    base_dir::String
    runs_dir::String
    results_path::String
    manifest_path::String
    index_path::String
    k::Int
    D::Int
    p_min::Int
    p_max::Int
    clause_sign::Int
    restarts::Int
    maxiters::Int
    seed::Int
    preserve::Bool
    result_count::Int
end

function write_manifest!(context::RunPreservationContext)
    write_manifest_json(
        context.manifest_path;
        run_id=context.run_id,
        run_kind_value=context.run_kind_value,
        runner_label_value=context.runner_label_value,
        reliability_dir=context.reliability_dir,
        reliability_files=context.reliability_files,
        timestamp_utc=context.timestamp_utc,
        git_commit=context.git_commit,
        git_branch=context.git_branch,
        git_dirty=context.git_dirty,
        k=context.k,
        D=context.D,
        p_min=context.p_min,
        p_max=context.p_max,
        clause_sign=context.clause_sign,
        restarts=context.restarts,
        maxiters=context.maxiters,
        seed=context.seed,
        preserve=context.preserve,
        result_count=context.result_count,
    )
end

function initialize_run_preservation(k, D, p_min, p_max, clause_sign, restarts, maxiters, seed, preserve)
    timestamp = Dates.now(Dates.UTC)
    timestamp_utc = Dates.format(timestamp, dateformat"yyyy-mm-ddTHH:MM:SS")
    run_stamp = Dates.format(timestamp, dateformat"yyyymmddTHHMMSS")
    run_id = "$(run_stamp)-k$(k)-d$(D)-p$(p_min)-$(p_max)-r$(restarts)-i$(maxiters)-s$(seed)"
    run_kind_value = run_kind()
    runner_label_value = runner_label()
    reliability_dir = reliability_artifacts_dir()
    reliability_files = reliability_artifact_names(reliability_dir)
    git_commit = git_commit_sha()
    git_branch = git_branch_name()
    git_dirty = git_is_dirty()
    base_dir = ensure_directory(joinpath(@__DIR__, "..", ".project", "results", "optimization"))
    runs_dir = ensure_directory(joinpath(base_dir, "runs"))
    run_dir = ensure_directory(joinpath(runs_dir, run_id))
    results_path = joinpath(run_dir, "results.csv")
    manifest_path = joinpath(run_dir, "manifest.json")
    index_path = joinpath(base_dir, "index.csv")

    open(results_path, "w") do io
        println(io, result_csv_header())
    end

    context = RunPreservationContext(
        run_dir,
        run_id,
        run_kind_value,
        runner_label_value,
        reliability_dir,
        reliability_files,
        timestamp_utc,
        git_commit,
        git_branch,
        git_dirty,
        base_dir,
        runs_dir,
        results_path,
        manifest_path,
        index_path,
        k,
        D,
        p_min,
        p_max,
        clause_sign,
        restarts,
        maxiters,
        seed,
        preserve,
        0,
    )
    write_manifest!(context)

    context
end

function append_preserved_result!(context::RunPreservationContext, result)
    row = result_csv_row(
        context.run_id,
        context.run_kind_value,
        context.runner_label_value,
        context.timestamp_utc,
        context.git_commit,
        context.git_branch,
        context.git_dirty,
        context.k,
        context.D,
        context.clause_sign,
        context.restarts,
        context.maxiters,
        context.seed,
        result,
    )
    append_results_csv_row(context.results_path, row)
    append_results_index(context.index_path, context.runs_dir, [row])
    context.result_count += 1
    write_manifest!(context)
end

function preserve_run(results, k, D, p_min, p_max, clause_sign, restarts, maxiters, seed, preserve)
    context = initialize_run_preservation(k, D, p_min, p_max, clause_sign, restarts, maxiters, seed, preserve)
    foreach(result -> append_preserved_result!(context, result), results)

    context.run_dir, context.run_id
end

length(ARGS) ≥ 4 || (usage(); exit(1))

k = parse_int("K", ARGS[1])
D = parse_int("D", ARGS[2])
p_min = parse_int("P_MIN", ARGS[3])
p_max = parse_int("P_MAX", ARGS[4])
restarts = length(ARGS) ≥ 5 ? parse_int("RESTARTS", ARGS[5]) : 8
maxiters = length(ARGS) ≥ 6 ? parse_int("MAXITERS", ARGS[6]) : 200
seed = length(ARGS) ≥ 7 ? parse_int("SEED", ARGS[7]) : 1234
preserve = length(ARGS) ≥ 8 ? parse_bool_flag(ARGS[8]) : true
autodiff = if length(ARGS) ≥ 9
    s = lowercase(ARGS[9])
    s in ("adjoint", "forward", "finite") || error("AUTODIFF must be adjoint, forward, or finite; got $s")
    Symbol(s)
else
    :adjoint
end

p_max ≥ p_min || error("P_MAX must be ≥ P_MIN")

rng = MersenneTwister(seed)
clause_sign = k == 2 ? -1 : 1
println("p,value,wall_time_seconds,best_start_wall_time_seconds,evaluations,starts,iterations,converged,retry_count,best_start_kind,gamma,beta")
flush(stdout)

preservation_context = preserve ? initialize_run_preservation(k, D, p_min, p_max, clause_sign, restarts, maxiters, seed, preserve) : nothing
if preserve
    println(stderr, "preserved run_id=$(preservation_context.run_id)")
    println(stderr, "preserved directory=$(preservation_context.run_dir)")
else
    println(stderr, "preservation disabled")
end
flush(stderr)

function emit_result(result)
    @printf(
        "%d,%.12f,%.6f,%.6f,%d,%d,%d,%s,%d,%s,%s,%s\n",
        depth(result.angles),
        result.value,
        result.wall_time_seconds,
        result.best_start_wall_time_seconds,
        result.evaluations,
        result.starts,
        result.iterations,
        string(result.converged),
        result.retry_count,
        string(result.best_start_kind),
        join(string.(result.angles.γ), ';'),
        join(string.(result.angles.β), ';'),
    )
    flush(stdout)

    if !isnothing(preservation_context)
        append_preserved_result!(preservation_context, result)
    end
end

results = optimize_depth_sequence(
    k,
    D,
    collect(p_min:p_max);
    clause_sign,
    restarts,
    maxiters,
    autodiff,
    rng,
    on_result=emit_result,
)
