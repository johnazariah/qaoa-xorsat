using Optim
using ADTypes: AutoForwardDiff
using ForwardDiff
using Random

struct AngleOptimizationStartSpec
    kind::Symbol
    angles::QAOAAngles
end

struct OptimizationTraceEntry
    iteration::Int
    value::Float64
    g_norm::Float64
end

struct AngleOptimizationStartResult
    kind::Symbol
    value::Float64
    wall_time_seconds::Float64
    evaluations::Int
    iterations::Int
    converged::Bool
    trace::Vector{OptimizationTraceEntry}
end

const DEFAULT_G_ABSTOL = 1.0e-6
const RELAXED_G_ABSTOL_FLOOR = 1.0e-3
const F_RELTOL = 1.0e-10
const PLATEAU_CHECK_SECONDS = 300  # check plateau every 5 minutes wall time
const PLATEAU_WINDOW_SIZE = 30     # rolling window of recent values for plateau detection
const GRADIENT_PLATEAU_WINDOW = 20 # secondary: if gradient norm has been < 100×g_abstol for this many iters, exit

"""
    plateau_chunk_size(p) -> Int

Depth-dependent chunk size for plateau detection. Smaller chunks at high p
where evaluations are expensive.
"""
function plateau_chunk_size(p::Int)
    p ≤ 8  ? 100 :
    p ≤ 10 ? 50 :
    p ≤ 12 ? 20 :
             10
end

struct DepthOptimizationBudget
    restarts::Int
    maxiters::Int
end

struct AngleOptimizationResult
    angles::QAOAAngles
    value::Float64
    wall_time_seconds::Float64
    best_start_wall_time_seconds::Float64
    evaluations::Int
    starts::Int
    iterations::Int
    converged::Bool
    restarts::Int
    maxiters::Int
    retry_count::Int
    best_start_kind::Symbol
    g_abstol::Float64
    start_results::Vector{AngleOptimizationStartResult}
end

default_clause_sign(k::Int) = k == 2 ? -1 : 1

function angle_vector(angles::QAOAAngles)
    [angles.γ; angles.β]
end

function angles_from_vector(values::AbstractVector{<:Real}, p::Int)
    length(values) == 2p || throw(ArgumentError(
        "need exactly $(2p) angle values for depth $p, got $(length(values))",
    ))

    QAOAAngles(values[1:p], values[(p+1):(2p)])
end

canonicalize_problem_angle(γ::Real) = mod(Float64(γ), 2π)
canonicalize_mixer_angle(β::Real) = mod(Float64(β), π)

"""
    canonicalize_angles(angles) -> QAOAAngles

Map the QAOA angles into a canonical periodic window:

- `γ_r ∈ [0, 2π)`
- `β_r ∈ [0, π)`
"""
function canonicalize_angles(angles::QAOAAngles)
    QAOAAngles(
        canonicalize_problem_angle.(angles.γ),
        canonicalize_mixer_angle.(angles.β),
    )
end

"""
    random_angles(p; rng=Random.default_rng()) -> QAOAAngles

Sample a random depth-`p` angle vector in the canonical search box.
"""
function random_angles(p::Int; rng=Random.default_rng())
    validate_depth(p)

    QAOAAngles(2π .* rand(rng, p), π .* rand(rng, p))
end

"""
    extend_angles(previous, target_depth=depth(previous)+1) -> QAOAAngles

Warm-start a deeper optimisation by reusing the existing angles and padding the
tail with the last known values.
"""
function extend_angles(
    previous::QAOAAngles,
    target_depth::Int=depth(previous) + 1,
)
    previous_depth = depth(previous)
    target_depth ≥ previous_depth || throw(ArgumentError(
        "target_depth must be ≥ $(previous_depth), got $target_depth",
    ))

    target_depth == previous_depth && return QAOAAngles(previous.γ, previous.β)

    extra = target_depth - previous_depth
    QAOAAngles(
        [previous.γ; fill(last(previous.γ), extra)],
        [previous.β; fill(last(previous.β), extra)],
    )
end

function build_initial_guesses(
    p::Int,
    initial_guesses::AbstractVector{<:QAOAAngles},
    initial_guess_kind::Symbol,
    restarts::Int,
    rng,
)::Vector{AngleOptimizationStartSpec}
    restarts ≥ 0 || throw(ArgumentError("restarts must be ≥ 0, got $restarts"))
    all(depth(guess) == p for guess in initial_guesses) || throw(ArgumentError(
        "all initial guesses must have depth $p",
    ))

    guesses = AngleOptimizationStartSpec[
        AngleOptimizationStartSpec(initial_guess_kind, canonicalize_angles(guess))
        for guess in initial_guesses
    ]
    isempty(guesses) && push!(guesses, AngleOptimizationStartSpec(:random, random_angles(p; rng)))

    for _ in 1:restarts
        push!(guesses, AngleOptimizationStartSpec(:random, random_angles(p; rng)))
    end

    guesses
end

function depth_optimization_budget(
    p::Int,
    restarts::Int,
    maxiters::Int,
)::DepthOptimizationBudget
    validate_depth(p)
    restarts ≥ 0 || throw(ArgumentError("restarts must be ≥ 0, got $restarts"))
    maxiters ≥ 1 || throw(ArgumentError("maxiters must be ≥ 1, got $maxiters"))

    if p ≤ 3
        return DepthOptimizationBudget(restarts, maxiters)
    elseif p == 4
        return DepthOptimizationBudget(min(restarts, 4), 2 * maxiters)
    elseif p ≤ 10
        return DepthOptimizationBudget(min(restarts, 2), 4 * maxiters)
    else
        # At p≥11, random starts can't compete with warm start — the landscape
        # is too large. Run warm start only to avoid waiting hours for hopeless
        # random starts to hit maxiters.
        return DepthOptimizationBudget(0, 4 * maxiters)
    end
end

retry_optimization_budget(maxiters::Int) = maxiters ≥ 1 ? maxiters : throw(ArgumentError(
    "maxiters must be ≥ 1, got $maxiters",
))

"""
    depth_g_abstol(p) -> Float64

Depth-dependent gradient tolerance. At high p the gradient noise floor
grows (cross-run agreement is ~1e-8 in c̃ at p=10), making tight tolerances
unreachable. Starting with a looser tolerance avoids wasting a full
iteration budget before the adaptive escalation kicks in.

| Depth    | g_abstol | Rationale                                |
|----------|----------|------------------------------------------|
| p ≤ 10   | 1e-6     | Converges in 5-45 iterations reliably    |
| p = 11   | 1e-5     | Gradient noise floor makes 1e-6 marginal |
| p ≥ 12   | 1e-4     | g_norm oscillates ~1e-4 at p=12          |
"""
function depth_g_abstol(p::Int)
    p ≤ 10 ? DEFAULT_G_ABSTOL :
    p == 11 ? 1.0e-5 :
              1.0e-4
end

"""
    is_valid_qaoa_value(v) -> Bool

Return `true` if `v` is a physically plausible QAOA satisfaction fraction.
Values outside [0, 1] (with a tiny tolerance for floating-point noise)
indicate evaluator overflow and must never be accepted as results."""
is_valid_qaoa_value(v::Real) = isfinite(v) && -1.0e-9 ≤ v ≤ 1.0 + 1.0e-9

function merge_optimization_results(
    primary::AngleOptimizationResult,
    secondary::AngleOptimizationResult,
)::AngleOptimizationResult
    pv = is_valid_qaoa_value(primary.value)
    sv = is_valid_qaoa_value(secondary.value)
    best = if sv && !pv
        secondary
    elseif pv && !sv
        primary
    elseif sv && pv && secondary.value > primary.value
        secondary
    elseif sv && pv && secondary.value ≈ primary.value && secondary.converged && !primary.converged
        secondary
    else
        primary
    end

    AngleOptimizationResult(
        best.angles,
        best.value,
        primary.wall_time_seconds + secondary.wall_time_seconds,
        best.best_start_wall_time_seconds,
        primary.evaluations + secondary.evaluations,
        primary.starts + secondary.starts,
        best.iterations,
        best.converged,
        primary.restarts + secondary.restarts,
        max(primary.maxiters, secondary.maxiters),
        primary.retry_count + secondary.retry_count + 1,
        best.best_start_kind,
        best.g_abstol,
        vcat(primary.start_results, secondary.start_results),
    )
end

"""
    optimize_angles(params; kwargs...) -> AngleOptimizationResult

Run a multistart local optimisation of `qaoa_expectation(params, angles)` over
the `2p` QAOA angles using `Optim.LBFGS`.

Keyword arguments:

- `clause_sign`: defaults to `-1` for MaxCut (`k=2`) and `+1` otherwise
- `restarts`: number of additional random starts beyond any supplied seeds
- `maxiters`: per-start optimiser iteration cap
- `initial_guesses`: optional seeded starting points of depth `p`
- `autodiff`: gradient method — `:adjoint` (default, fastest), `:charge` (charge evaluator + ForwardDiff, lower memory), `:forward` (ForwardDiff on raw evaluator), or `:finite` (finite differences)
- `rng`: random number generator for restart sampling
- `g_abstol`: gradient absolute tolerance for convergence (default: `DEFAULT_G_ABSTOL`)
- `on_evaluation`: optional callback `(start_index, evaluations, elapsed_seconds, value, g_norm) -> nothing` throttled to at most once per 30 seconds per start
- `on_chunk`: optional callback `(start_index, iterations, trace_entries, current_angles_vector) -> nothing` called after each plateau-detection chunk for incremental trace flushing
- `eval_eltype`: element type for evaluation arithmetic (default: `Float64`; use `Double64` for k≥6)
- `checkpointed`: use the CPU gradient checkpointer for adjoint evaluations
- `checkpoint_disk_dir`: optional parent directory for spilled checkpoint tensors
- `checkpoint_max_ram_checkpoints`: maximum branch-tensor checkpoints to keep in RAM when spilling
"""
# Promote angles to a different precision for evaluation.
# Optim.jl works in Float64; this promotes to e.g. Double64 for the evaluator.
_promote_angles(a::QAOAAngles, ::Type{Float64}) = a
_promote_angles(a::QAOAAngles, ::Type{T}) where T<:Real = QAOAAngles(T.(a.γ), T.(a.β))

function optimize_angles(
    params::TreeParams;
    clause_sign::Int=default_clause_sign(params.k),
    restarts::Int=8,
    maxiters::Int=200,
    initial_guesses::AbstractVector{<:QAOAAngles}=QAOAAngles[],
    initial_guess_kind::Symbol=:seeded,
    autodiff::Symbol=:adjoint,
    rng=Random.default_rng(),
    g_abstol::Float64=DEFAULT_G_ABSTOL,
    on_evaluation=nothing,
    on_chunk=nothing,
    eval_eltype::Type=Float64,
    gpu_evaluator::Union{Function,Nothing}=nothing,
    checkpointed::Bool=false,
    checkpoint_disk_dir::Union{String,Nothing}=nothing,
    checkpoint_max_ram_checkpoints::Int=typemax(Int),
)::AngleOptimizationResult
    validate_clause_sign(clause_sign)
    maxiters ≥ 1 || throw(ArgumentError("maxiters must be ≥ 1, got $maxiters"))

    guesses = build_initial_guesses(params.p, initial_guesses, initial_guess_kind, restarts, rng)

    optimization_started_at = time_ns()

    # Memory-bounded concurrency: limit parallel restarts to avoid OOM.
    # At p=11 each adjoint cache is ~5-7 GB; running too many in parallel
    # exhausts RAM. Cap at available_ram / estimated_per_eval_bytes.
    N = basso_configuration_count(params.p)
    # Rough estimate: ~40 vectors of size N × 16 bytes (ComplexF64) per eval
    est_bytes_per_eval = 40 * N * sizeof(ComplexF64)
    available_ram = Sys.total_memory() * 0.75  # leave 25% headroom
    max_concurrent = max(1, min(Threads.nthreads(), floor(Int, available_ram / est_bytes_per_eval)))

    # Use a semaphore to limit concurrency
    sem = Base.Semaphore(max_concurrent)

    # Run each restart independently — thread-parallel with memory-bounded concurrency
    per_start_results = Vector{Tuple{QAOAAngles,Float64,AngleOptimizationStartResult}}(undef, length(guesses))
    Threads.@threads for i in eachindex(guesses)
        Base.acquire(sem)
        local_checkpoint_dir = nothing
        try
            guess = guesses[i]
            local_evaluations = Ref(0)
            last_progress_at = Ref(time_ns())
            started_at = time_ns()
            last_value = Ref(NaN)
            last_g_norm = Ref(NaN)
            local_checkpoint_dir = checkpointed && checkpoint_disk_dir !== nothing ?
                                   mktempdir(checkpoint_disk_dir; prefix="qaoa-start$(i)-") :
                                   nothing

        function maybe_report_progress!()
            isnothing(on_evaluation) && return
            now = time_ns()
            elapsed_since_report = (now - last_progress_at[]) / 1.0e9
            elapsed_since_report ≥ 30.0 || return
            last_progress_at[] = now
            elapsed = (now - started_at) / 1.0e9
            on_evaluation(i, local_evaluations[], elapsed, last_value[], last_g_norm[])
        end

        if autodiff == :adjoint
            # Manual adjoint: combined value+gradient via fg! for efficiency.
            # Optim's line search calls fg! when it needs both f and g together,
            # avoiding the redundant forward pass that separate f/g! would require.
            #
            # Overflow guard: at high (k,D,p) the branch tensor iteration can
            # overflow Float64, producing c̃ > 1 or c̃ < 0 or NaN/Inf.
            # We return a large positive objective with a large gradient pointing
            # back toward the origin, so L-BFGS backtracks. Returning zero
            # gradient is dangerous because it fakes convergence.
            function _overflow_gradient!(G, values, p)
                # Point gradient toward the origin so L-BFGS moves away
                # from the overflow region, not park at it.
                for j in eachindex(G)
                    G[j] = values[j] > 0 ? 1.0 : -1.0
                end
                last_g_norm[] = 1.0
            end

            function f_adjoint(values)
                local_evaluations[] += 1
                maybe_report_progress!()
                candidate = _promote_angles(angles_from_vector(values, params.p), eval_eltype)
                if gpu_evaluator !== nothing
                    val, _, _ = gpu_evaluator(params, candidate; clause_sign)
                elseif checkpointed
                    val = Float64(basso_expectation_checkpointed(params, candidate;
                        clause_sign,
                        disk_dir=local_checkpoint_dir,
                        max_ram_checkpoints=checkpoint_max_ram_checkpoints))
                else
                    val = Float64(basso_expectation_normalized(params, candidate; clause_sign))
                end
                if !is_valid_qaoa_value(val)
                    return 1.0e6
                end
                -val
            end

            function g_adjoint!(G, values)
                candidate = _promote_angles(angles_from_vector(values, params.p), eval_eltype)
                if gpu_evaluator !== nothing
                    _, γg, βg = gpu_evaluator(params, candidate; clause_sign)
                elseif checkpointed
                    _, γg, βg = basso_expectation_and_gradient_checkpointed(params, candidate;
                        clause_sign,
                        disk_dir=local_checkpoint_dir,
                        max_ram_checkpoints=checkpoint_max_ram_checkpoints)
                else
                    _, γg, βg = basso_expectation_and_gradient(params, candidate; clause_sign)
                end
                if any(!isfinite, γg) || any(!isfinite, βg)
                    _overflow_gradient!(G, values, params.p)
                    return
                end
                G[1:params.p] .= .-Float64.(γg)
                G[params.p+1:2*params.p] .= .-Float64.(βg)
                last_g_norm[] = maximum(abs, G)
            end

            function fg_adjoint!(G, values)
                local_evaluations[] += 1
                maybe_report_progress!()
                candidate = _promote_angles(angles_from_vector(values, params.p), eval_eltype)

                # GPU path: use gpu_evaluator if provided
                if gpu_evaluator !== nothing
                    fval, γg, βg = gpu_evaluator(params, candidate; clause_sign)
                elseif checkpointed
                    val, γg, βg = basso_expectation_and_gradient_checkpointed(params, candidate;
                        clause_sign,
                        disk_dir=local_checkpoint_dir,
                        max_ram_checkpoints=checkpoint_max_ram_checkpoints)
                    fval = Float64(val)
                else
                    val, γg, βg = basso_expectation_and_gradient(params, candidate; clause_sign)
                    fval = Float64(val)
                end

                if !is_valid_qaoa_value(fval) ||
                   any(!isfinite, γg) || any(!isfinite, βg)
                    _overflow_gradient!(G, values, params.p)
                    last_value[] = NaN
                    return 1.0e6  # overflow — large but finite
                end
                G[1:params.p] .= .-Float64.(γg)
                G[params.p+1:2*params.p] .= .-Float64.(βg)
                last_g_norm[] = maximum(abs, G)
                last_value[] = fval
                -fval
            end

            od = Optim.OnceDifferentiable(f_adjoint, g_adjoint!, fg_adjoint!, angle_vector(guess.angles))

            # Single Optim run with a per-iteration callback that:
            #   1. Maintains a circular buffer of recent values
            #   2. Every PLATEAU_CHECK_SECONDS, fires on_chunk for trace visibility
            #      and checks if max-min over the buffer < g_abstol (plateau)
            #   3. Returns true to stop Optim if plateau detected
            all_trace = OptimizationTraceEntry[]
            converged_flag = false
            last_check_time = time_ns()
            value_buffer = Float64[]

            iteration_count = Ref(0)
            gnorm_buffer = Float64[]

            function plateau_callback(state)
                iteration_count[] += 1
                iter = iteration_count[]
                val = state.f_x           # raw objective (negated c̃)
                gnorm = maximum(abs, state.g_x)

                push!(all_trace, OptimizationTraceEntry(iter, val, gnorm))

                # Update circular buffers
                push!(value_buffer, val)
                if length(value_buffer) > PLATEAU_WINDOW_SIZE
                    popfirst!(value_buffer)
                end
                push!(gnorm_buffer, gnorm)
                if length(gnorm_buffer) > GRADIENT_PLATEAU_WINDOW
                    popfirst!(gnorm_buffer)
                end

                # Primary: value plateau (30 iters with range < g_abstol)
                if length(value_buffer) == PLATEAU_WINDOW_SIZE
                    buffer_min = minimum(value_buffer)
                    buffer_max = maximum(value_buffer)
                    vrange = buffer_max - buffer_min
                    if vrange < g_abstol
                        # Flush trace before stopping
                        if !isnothing(on_chunk)
                            try
                                on_chunk(i, iter, all_trace, state.x)
                            catch e
                                @warn "on_chunk callback failed" exception=e
                            end
                        end
                        converged_flag = true
                        return true  # stop Optim immediately
                    end
                end

                # Secondary: gradient plateau (20 iters with all gnorms < 100×g_abstol
                # AND value range < 10×g_abstol — gradient stuck but value converged)
                if length(gnorm_buffer) == GRADIENT_PLATEAU_WINDOW &&
                   length(value_buffer) >= GRADIENT_PLATEAU_WINDOW
                    if maximum(gnorm_buffer) < 100 * g_abstol
                        recent_vals = value_buffer[end-GRADIENT_PLATEAU_WINDOW+1:end]
                        if maximum(recent_vals) - minimum(recent_vals) < 10 * g_abstol
                            if !isnothing(on_chunk)
                                try
                                    on_chunk(i, iter, all_trace, state.x)
                                catch e
                                    @warn "on_chunk callback failed" exception=e
                                end
                            end
                            converged_flag = true
                            return true  # gradient plateau exit
                        end
                    end
                end

                # Periodic trace flush for visibility (independent of convergence)
                elapsed_since_check = (time_ns() - last_check_time) / 1.0e9
                if elapsed_since_check >= PLATEAU_CHECK_SECONDS
                    if !isnothing(on_chunk)
                        try
                            on_chunk(i, iter, all_trace, state.x)
                        catch e
                            @warn "on_chunk callback failed" exception=e
                        end
                    end
                    last_check_time = time_ns()
                end

                return false  # keep going
            end

            result = Optim.optimize(
                od,
                angle_vector(guess.angles),
                Optim.LBFGS(),
                Optim.Options(
                    iterations=maxiters,
                    g_abstol=g_abstol,
                    f_reltol=F_RELTOL,
                    store_trace=false,  # we track trace ourselves in the callback
                    show_trace=false,
                    callback=plateau_callback,
                ),
            )

            # If Optim converged via g_abstol (not our plateau), mark it
            if Optim.converged(result) && !converged_flag
                converged_flag = true
            end

            # Final on_chunk flush
            if !isnothing(on_chunk)
                try
                    on_chunk(i, Optim.iterations(result), all_trace, Optim.minimizer(result))
                catch e
                    @warn "on_chunk callback failed" exception=e
                end
            end

            # Package results — re-evaluate using normalized path to avoid overflow
            elapsed_seconds_start = (time_ns() - started_at) / 1.0e9
            candidate_angles_start = angles_from_vector(Optim.minimizer(result), params.p) |> canonicalize_angles
            if gpu_evaluator !== nothing
                candidate_value_start, _, _ = gpu_evaluator(params,
                    _promote_angles(candidate_angles_start, eval_eltype); clause_sign)
            elseif checkpointed
                candidate_value_start = Float64(basso_expectation_checkpointed(params,
                    _promote_angles(candidate_angles_start, eval_eltype);
                    clause_sign,
                    disk_dir=local_checkpoint_dir,
                    max_ram_checkpoints=checkpoint_max_ram_checkpoints))
            else
                candidate_value_start = Float64(basso_expectation_normalized(params,
                    _promote_angles(candidate_angles_start, eval_eltype); clause_sign))
            end
            if !is_valid_qaoa_value(candidate_value_start)
                @warn "start $(i) ($(guess.kind)) produced invalid value $(candidate_value_start); marking failed"
                candidate_value_start = -Inf
                converged_flag = false
            end
            start_result = AngleOptimizationStartResult(
                guess.kind,
                candidate_value_start,
                elapsed_seconds_start,
                local_evaluations[],
                Optim.iterations(result),
                converged_flag,
                all_trace,
            )
            per_start_results[i] = (candidate_angles_start, candidate_value_start, start_result)
        else
            # Non-adjoint path (ForwardDiff or finite differences) — no chunking
            function objective(values)
                local_evaluations[] += 1
                maybe_report_progress!()
                candidate = angles_from_vector(values, params.p)
                if autodiff == :charge
                    -charge_expectation(params, candidate; clause_sign)
                else
                    -qaoa_expectation(params, candidate; clause_sign)
                end
            end

            result = Optim.optimize(
                objective,
                angle_vector(guess.angles),
                Optim.LBFGS(),
                Optim.Options(iterations=maxiters, g_abstol=g_abstol, f_reltol=F_RELTOL, store_trace=true, show_trace=false);
                autodiff=AutoForwardDiff(),
            )

            elapsed_seconds = (time_ns() - started_at) / 1.0e9
            candidate_angles = angles_from_vector(Optim.minimizer(result), params.p) |> canonicalize_angles
            candidate_value = if autodiff == :charge
                charge_expectation(params, candidate_angles; clause_sign)
            else
                basso_expectation_normalized(params, candidate_angles; clause_sign)
            end
            if !is_valid_qaoa_value(candidate_value)
                @warn "start $(i) ($(guess.kind), non-adjoint) produced invalid value $(candidate_value); marking failed"
                candidate_value = -Inf
            end
            optim_trace = Optim.trace(result)
            trace_entries = [
                OptimizationTraceEntry(state.iteration, state.value, state.g_norm)
                for state in optim_trace
            ]
            start_result = AngleOptimizationStartResult(
                guess.kind,
                candidate_value,
                elapsed_seconds,
                local_evaluations[],
                Optim.iterations(result),
                Optim.converged(result),
                trace_entries,
            )
            per_start_results[i] = (candidate_angles, candidate_value, start_result)
        end
        finally
            local_checkpoint_dir !== nothing && rm(local_checkpoint_dir; force=true, recursive=true)
            Base.release(sem)
        end
    end

    # Collect results — pick the best start, preferring valid values over invalid
    start_results = [r[3] for r in per_start_results]
    total_evaluations = sum(r.evaluations for r in start_results)
    valid_mask = [is_valid_qaoa_value(r[2]) for r in per_start_results]
    best_idx = if any(valid_mask)
        argmax(j -> valid_mask[j] ? per_start_results[j][2] : -Inf, eachindex(per_start_results))
    else
        # All starts overflowed — pick the least-bad one
        argmax(j -> isfinite(per_start_results[j][2]) ? per_start_results[j][2] : -Inf, eachindex(per_start_results))
    end
    best_angles, best_value, best_start = per_start_results[best_idx]

    wall_time_seconds = (time_ns() - optimization_started_at) / 1.0e9
    AngleOptimizationResult(
        best_angles,
        best_value,
        wall_time_seconds,
        best_start.wall_time_seconds,
        total_evaluations,
        length(guesses),
        best_start.iterations,
        best_start.converged,
        restarts,
        maxiters,
        0,
        best_start.kind,
        g_abstol,
        start_results,
    )
end

function validate_depth_sequence(p_values)
    !isempty(p_values) || throw(ArgumentError("need at least one target depth"))
    all(p -> p ≥ 1, p_values) || throw(ArgumentError("all target depths must be ≥ 1"))

    for (left, right) in zip(p_values, p_values[2:end])
        right > left || throw(ArgumentError("target depths must be strictly increasing"))
    end

    p_values
end

"""
    optimize_depth_sequence(k, D, p_values; kwargs...) -> Vector{AngleOptimizationResult}

Optimise a sequence of depths, seeding each depth from the best angles found at
the previous depth by repeating the final angle pair.
"""
function optimize_depth_sequence(
    k::Int,
    D::Int,
    p_values::AbstractVector{<:Integer};
    clause_sign::Int=default_clause_sign(k),
    restarts::Int=8,
    maxiters::Int=200,
    autodiff::Symbol=:adjoint,
    rng=Random.default_rng(),
    on_result=nothing,
    on_evaluation=nothing,
    on_chunk=nothing,
    warm_start::Union{Nothing,QAOAAngles}=nothing,
    eval_eltype::Type=Float64,
    gpu_evaluator::Union{Function,Nothing}=nothing,
    checkpointed::Bool=false,
    checkpoint_disk_dir::Union{String,Nothing}=nothing,
    checkpoint_max_ram_checkpoints::Int=typemax(Int),
)::Vector{AngleOptimizationResult}
    validate_clause_sign(clause_sign)
    validated_p_values = validate_depth_sequence(collect(Int, p_values))

    results = AngleOptimizationResult[]

    for p in validated_p_values
        budget = depth_optimization_budget(p, restarts, maxiters)
        start_tol = depth_g_abstol(p)
        initial_guesses = isnothing(warm_start) ? QAOAAngles[] : [extend_angles(warm_start, p)]
        result = optimize_angles(
            TreeParams(k, D, p);
            clause_sign,
            restarts=budget.restarts,
            maxiters=budget.maxiters,
            initial_guesses,
            initial_guess_kind=:warm,
            autodiff,
            rng,
            g_abstol=start_tol,
            on_evaluation,
            on_chunk,
            eval_eltype,
            gpu_evaluator,
            checkpointed,
            checkpoint_disk_dir,
            checkpoint_max_ram_checkpoints,
        )

        if !isnothing(warm_start) && !result.converged
            # Adaptive tolerance escalation: relax g_abstol by 10× each retry
            # until convergence or the floor is hit. At high p the gradient noise
            # floor is ~1e-8 in c̃, so tight tolerances may be unreachable. Instead
            # of doubling iterations (which wastes hours), we accept a slightly
            # looser gradient norm. The trace is preserved for post-hoc analysis.
            escalated_tol = start_tol
            while !result.converged && escalated_tol < RELAXED_G_ABSTOL_FLOOR
                escalated_tol = min(escalated_tol * 10, RELAXED_G_ABSTOL_FLOOR)
                retry_result = optimize_angles(
                    TreeParams(k, D, p);
                    clause_sign,
                    restarts=0,
                    maxiters=retry_optimization_budget(budget.maxiters),
                    initial_guesses=[result.angles],
                    initial_guess_kind=:retry,
                    autodiff,
                    rng,
                    g_abstol=escalated_tol,
                    on_evaluation,
                    on_chunk,
                    eval_eltype,
                    gpu_evaluator,
                    checkpointed,
                    checkpoint_disk_dir,
                    checkpoint_max_ram_checkpoints,
                )
                result = merge_optimization_results(result, retry_result)
            end
        end

        push!(results, result)
        isnothing(on_result) || on_result(result)

        # Only propagate warm-start if the result is physically valid.
        # A poisoned warm-start will corrupt all subsequent depths.
        if is_valid_qaoa_value(result.value)
            warm_start = result.angles
        else
            @warn "depth p=$(p): invalid result c̃=$(result.value); not propagating as warm start"
            # Keep previous warm_start (from last valid depth) so the next
            # depth at least gets a plausible starting point.
        end
    end

    results
end

"""
    optimize_angles(algebra, params; kwargs...) -> AngleOptimizationResult

Algebra-parameterised optimisation entry point. Delegates to the clause_sign-based
implementation.
"""
function optimize_angles(
    algebra::CostAlgebra,
    params::TreeParams;
    kwargs...,
)::AngleOptimizationResult
    arity(algebra) == params.k || throw(ArgumentError(
        "algebra arity $(arity(algebra)) does not match tree arity $(params.k)"
    ))
    optimize_angles(params; clause_sign=default_clause_sign(algebra), kwargs...)
end

# ──────────────────────────────────────────────────────────────────────────────
# Swarm optimizer — memetic algorithm for rugged landscapes at high (k,D)
# ──────────────────────────────────────────────────────────────────────────────

struct SwarmCandidate
    angles::QAOAAngles
    value::Float64
end

"""
    swarm_optimize(params; kwargs...) -> AngleOptimizationResult

Memetic (evolutionary + local search) optimizer for QAOA angles.

At high (k, D) the loss landscape is extremely rugged: most starting points
see c̃ ≈ 0.5 (flat), and only specific basins carry signal.  Standard
multi-start L-BFGS with a handful of restarts misses these basins.  The
swarm strategy maintains a population of candidates and repeatedly:

1. Runs short L-BFGS bursts on all candidates (local improvement)
2. Culls the worst half
3. Replenishes with random starts + midpoint crossovers from the survivors

This explores far more basins than a fixed-restart strategy while concentrating
L-BFGS effort on promising regions.

# Keyword arguments
- `population`: initial population size (default: 100)
- `generations`: number of cull/replenish cycles (default: 10)
- `burst_iters`: L-BFGS iterations per burst (default: 20)
- `cull_fraction`: fraction of population to kill each generation (default: 0.5)
- `random_fraction`: fraction of replacements that are fresh random (default: 0.4)
- `clause_sign`, `autodiff`, `rng`, `g_abstol`: as in `optimize_angles`
- `on_generation`: optional callback `(gen, best_value, population_size, best_angles) -> nothing`
- `warm_starts`: optional initial angles to seed the population
- `checkpointed`, `checkpoint_disk_dir`, `checkpoint_max_ram_checkpoints`: CPU checkpointing controls passed to `optimize_angles`
- `candidate_concurrency`: maximum number of swarm candidates to burst-optimise at once (`0` = thread count)
"""
function swarm_optimize(
    params::TreeParams;
    clause_sign::Int=default_clause_sign(params.k),
    population::Int=100,
    generations::Int=10,
    burst_iters::Int=20,
    cull_fraction::Float64=0.5,
    random_fraction::Float64=0.4,
    autodiff::Symbol=:adjoint,
    rng=Random.default_rng(),
    g_abstol::Float64=DEFAULT_G_ABSTOL,
    on_generation=nothing,
    warm_starts::AbstractVector{<:QAOAAngles}=QAOAAngles[],
    eval_eltype::Type=Float64,
    gpu_evaluator::Union{Function,Nothing}=nothing,
    checkpointed::Bool=false,
    checkpoint_disk_dir::Union{String,Nothing}=nothing,
    checkpoint_max_ram_checkpoints::Int=typemax(Int),
    candidate_concurrency::Int=0,
)::AngleOptimizationResult
    p = params.p
    started_at = time_ns()
    total_evaluations = 0
    total_evaluations_lock = ReentrantLock()
    burst_concurrency = candidate_concurrency > 0 ? candidate_concurrency : Threads.nthreads()
    burst_semaphore = Base.Semaphore(max(1, burst_concurrency))

    # ── Evaluate a single candidate with a short L-BFGS burst ────────────
    function burst_optimize(angles::QAOAAngles)
        result = optimize_angles(
            params;
            clause_sign,
            restarts=0,
            maxiters=burst_iters,
            initial_guesses=[angles],
            initial_guess_kind=:swarm,
            autodiff,
            rng,
            g_abstol,
            eval_eltype,
            gpu_evaluator,
            checkpointed,
            checkpoint_disk_dir,
            checkpoint_max_ram_checkpoints,
        )
        (result.angles, result.value, result.evaluations, result.converged)
    end

    # ── Initialize population ─────────────────────────────────────────────
    candidates = SwarmCandidate[]

    # Seed with warm starts
    for ws in warm_starts
        depth(ws) == p || continue
        if gpu_evaluator !== nothing
            val, _, _ = gpu_evaluator(params, _promote_angles(ws, eval_eltype); clause_sign)
        elseif checkpointed
            warm_checkpoint_dir = checkpoint_disk_dir === nothing ? nothing :
                                  mktempdir(checkpoint_disk_dir; prefix="qaoa-warm-")
            try
                val = Float64(basso_expectation_checkpointed(params, _promote_angles(ws, eval_eltype);
                    clause_sign,
                    disk_dir=warm_checkpoint_dir,
                    max_ram_checkpoints=checkpoint_max_ram_checkpoints))
            finally
                warm_checkpoint_dir !== nothing && rm(warm_checkpoint_dir; force=true, recursive=true)
            end
        else
            val = Float64(basso_expectation_normalized(params, _promote_angles(ws, eval_eltype); clause_sign))
        end
        push!(candidates, SwarmCandidate(ws, is_valid_qaoa_value(val) ? val : -Inf))
    end

    # Fill the rest with random starts
    while length(candidates) < population
        push!(candidates, SwarmCandidate(random_angles(p; rng), -Inf))
    end

    best_ever = SwarmCandidate(candidates[1].angles, -Inf)
    all_trace = OptimizationTraceEntry[]
    stagnant_generations = 0

    for gen in 1:generations
        # ── Local improvement: short L-BFGS burst on each candidate ───────
        new_candidates = Vector{SwarmCandidate}(undef, length(candidates))
        candidates_done = Threads.Atomic{Int}(0)
        Threads.@threads for i in eachindex(candidates)
            Base.acquire(burst_semaphore)
            try
                angles_out, val_out, evals, _ = burst_optimize(candidates[i].angles)
                if is_valid_qaoa_value(val_out) && val_out > candidates[i].value
                    new_candidates[i] = SwarmCandidate(angles_out, val_out)
                else
                    # Keep the original if burst didn't improve
                    orig_val = candidates[i].value
                    if orig_val == -Inf
                        # First evaluation — use burst result even if mediocre
                        new_candidates[i] = SwarmCandidate(angles_out,
                            is_valid_qaoa_value(val_out) ? val_out : -Inf)
                    else
                        new_candidates[i] = candidates[i]
                    end
                end
                lock(total_evaluations_lock) do
                    total_evaluations += evals
                end
                Threads.atomic_add!(candidates_done, 1)
                done = candidates_done[]
                if done % max(1, length(candidates) ÷ 5) == 0 || done == length(candidates)
                    @info "  gen $gen: $done/$(length(candidates)) candidates burst-optimized"
                end
            finally
                Base.release(burst_semaphore)
            end
        end
        candidates = new_candidates

        # Track best
        prev_best = best_ever.value
        for c in candidates
            if is_valid_qaoa_value(c.value) && c.value > best_ever.value
                best_ever = c
            end
        end

        push!(all_trace, OptimizationTraceEntry(gen, -best_ever.value, 0.0))

        if !isnothing(on_generation)
            on_generation(gen, best_ever.value, length(candidates), best_ever.angles)
        end

        # ── Early exit: if the swarm isn't improving, stop exploring ──────
        # and switch to a full L-BFGS polish on the best candidate.
        if best_ever.value <= prev_best + 1e-10
            stagnant_generations += 1
        else
            stagnant_generations = 0
        end

        if stagnant_generations >= 3 && gen >= 3
            # Swarm is wandering — polish the winner and exit
            if !isnothing(on_generation)
                on_generation(gen, best_ever.value, -1, best_ever.angles)  # -1 signals early exit
            end
            break
        end

        gen == generations && break  # don't cull on the last generation

        # ── Sort by value (descending) and cull ───────────────────────────
        sort!(candidates, by=c -> -c.value)  # best first
        n_cull = round(Int, length(candidates) * cull_fraction)
        survivors = candidates[1:end-n_cull]

        # ── Replenish ─────────────────────────────────────────────────────
        n_new = n_cull
        n_random = round(Int, n_new * random_fraction)
        n_crossover = n_new - n_random

        new_members = SwarmCandidate[]

        # Random fresh starts
        for _ in 1:n_random
            push!(new_members, SwarmCandidate(random_angles(p; rng), -Inf))
        end

        # Midpoint crossovers from top survivors
        n_top = min(length(survivors), 30)
        for _ in 1:n_crossover
            i = rand(rng, 1:n_top)
            j = rand(rng, 1:n_top)
            while j == i && n_top > 1
                j = rand(rng, 1:n_top)
            end
            parent_a = angle_vector(survivors[i].angles)
            parent_b = angle_vector(survivors[j].angles)
            # Midpoint with small perturbation
            child_vec = 0.5 .* (parent_a .+ parent_b) .+ 0.1 .* randn(rng, 2p)
            child = angles_from_vector(child_vec, p) |> canonicalize_angles
            push!(new_members, SwarmCandidate(child, -Inf))
        end

        candidates = vcat(survivors, new_members)
    end

    # ── Polish: run a full L-BFGS on the best candidate found ───────────
    # The swarm finds the right basin; L-BFGS converges it properly.
    if is_valid_qaoa_value(best_ever.value) && best_ever.value > 0.501
        @info "  Polishing best candidate (c̃=$(best_ever.value))..."
        polish_result = optimize_angles(
            params;
            clause_sign,
            restarts=0,
            maxiters=1280,
            initial_guesses=[best_ever.angles],
            initial_guess_kind=:swarm_polish,
            autodiff,
            rng,
            g_abstol,
            eval_eltype,
            gpu_evaluator,
            checkpointed,
            checkpoint_disk_dir,
            checkpoint_max_ram_checkpoints,
            on_evaluation=(start_idx, evals, elapsed, val, gnorm) -> begin
                @info "  polish: $(evals) evals, $(round(elapsed,digits=0))s, c̃=$(round(val,digits=10)), |g|=$(round(gnorm,sigdigits=2))"
            end,
        )
        total_evaluations += polish_result.evaluations
        if is_valid_qaoa_value(polish_result.value) && polish_result.value > best_ever.value
            best_ever = SwarmCandidate(polish_result.angles, polish_result.value)
        end
    end

    # ── Package best result ───────────────────────────────────────────────
    wall_time = (time_ns() - started_at) / 1.0e9
    best_start_result = AngleOptimizationStartResult(
        :swarm, best_ever.value, wall_time, total_evaluations,
        generations * burst_iters, is_valid_qaoa_value(best_ever.value), all_trace,
    )

    AngleOptimizationResult(
        best_ever.angles,
        best_ever.value,
        wall_time,
        wall_time,
        total_evaluations,
        population * generations,
        generations * burst_iters,
        is_valid_qaoa_value(best_ever.value) && best_ever.value > 0.501,
        0,
        burst_iters,
        0,
        :swarm,
        g_abstol,
        [best_start_result],
    )
end
