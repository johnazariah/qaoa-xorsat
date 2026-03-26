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
const RELAXED_G_ABSTOL_FLOOR = 1.0e-4
const F_RELTOL = 1.0e-10

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
| p = 11-12| 1e-5     | Gradient noise floor makes 1e-6 marginal |
| p ≥ 13   | 1e-4     | 2^27 element WHT, higher noise floor     |
"""
function depth_g_abstol(p::Int)
    p ≤ 10 ? DEFAULT_G_ABSTOL :
    p ≤ 12 ? 1.0e-5 :
             RELAXED_G_ABSTOL_FLOOR
end

function merge_optimization_results(
    primary::AngleOptimizationResult,
    secondary::AngleOptimizationResult,
)::AngleOptimizationResult
    best = if secondary.value > primary.value
        secondary
    elseif secondary.value ≈ primary.value && secondary.converged && !primary.converged
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
- `autodiff`: gradient method — `:adjoint` (default, fastest), `:forward` (ForwardDiff), or `:finite` (finite differences)
- `rng`: random number generator for restart sampling
- `g_abstol`: gradient absolute tolerance for convergence (default: `DEFAULT_G_ABSTOL`)
- `on_evaluation`: optional callback `(start_index, evaluations, elapsed_seconds) -> nothing` throttled to at most once per 30 seconds per start
"""
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
)::AngleOptimizationResult
    validate_clause_sign(clause_sign)
    maxiters ≥ 1 || throw(ArgumentError("maxiters must be ≥ 1, got $maxiters"))

    guesses = build_initial_guesses(params.p, initial_guesses, initial_guess_kind, restarts, rng)

    optimization_started_at = time_ns()

    # Run each restart independently — thread-parallel when multiple threads available
    per_start_results = Vector{Tuple{QAOAAngles,Float64,AngleOptimizationStartResult}}(undef, length(guesses))
    Threads.@threads for i in eachindex(guesses)
        guess = guesses[i]
        local_evaluations = Ref(0)
        last_progress_at = Ref(time_ns())
        started_at = time_ns()

        function maybe_report_progress!()
            isnothing(on_evaluation) && return
            now = time_ns()
            elapsed_since_report = (now - last_progress_at[]) / 1.0e9
            elapsed_since_report ≥ 30.0 || return
            last_progress_at[] = now
            elapsed = (now - started_at) / 1.0e9
            on_evaluation(i, local_evaluations[], elapsed)
        end

        if autodiff == :adjoint
            # Manual adjoint: combined value+gradient via fg! for efficiency.
            # Optim's line search calls fg! when it needs both f and g together,
            # avoiding the redundant forward pass that separate f/g! would require.
            function f_adjoint(values)
                local_evaluations[] += 1
                maybe_report_progress!()
                candidate = angles_from_vector(values, params.p)
                -qaoa_expectation(params, candidate; clause_sign)
            end

            function g_adjoint!(G, values)
                candidate = angles_from_vector(values, params.p)
                _, γg, βg = basso_expectation_and_gradient(params, candidate; clause_sign)
                G[1:params.p] .= .-γg
                G[params.p+1:2*params.p] .= .-βg
            end

            function fg_adjoint!(G, values)
                local_evaluations[] += 1
                maybe_report_progress!()
                candidate = angles_from_vector(values, params.p)
                val, γg, βg = basso_expectation_and_gradient(params, candidate; clause_sign)
                G[1:params.p] .= .-γg
                G[params.p+1:2*params.p] .= .-βg
                -val
            end

            od = Optim.OnceDifferentiable(f_adjoint, g_adjoint!, fg_adjoint!, angle_vector(guess.angles))
            result = Optim.optimize(
                od,
                angle_vector(guess.angles),
                Optim.LBFGS(),
                Optim.Options(iterations=maxiters, g_abstol=g_abstol, f_reltol=F_RELTOL, store_trace=true, show_trace=false),
            )
        else
            function objective(values)
                local_evaluations[] += 1
                maybe_report_progress!()
                candidate = angles_from_vector(values, params.p)
                -qaoa_expectation(params, candidate; clause_sign)
            end

            result = Optim.optimize(
                objective,
                angle_vector(guess.angles),
                Optim.LBFGS(),
                Optim.Options(iterations=maxiters, g_abstol=g_abstol, f_reltol=F_RELTOL, store_trace=true, show_trace=false);
                autodiff=autodiff == :forward ? AutoForwardDiff() : :finite,
            )
        end

        elapsed_seconds = (time_ns() - started_at) / 1.0e9
        candidate_angles = angles_from_vector(Optim.minimizer(result), params.p) |> canonicalize_angles
        candidate_value = qaoa_expectation(params, candidate_angles; clause_sign)
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

    # Collect results — pick the best start
    start_results = [r[3] for r in per_start_results]
    total_evaluations = sum(r.evaluations for r in start_results)
    best_idx = argmax(r[2] for r in per_start_results)
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
)::Vector{AngleOptimizationResult}
    validate_clause_sign(clause_sign)
    validated_p_values = validate_depth_sequence(collect(Int, p_values))

    results = AngleOptimizationResult[]
    warm_start = nothing

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
                )
                result = merge_optimization_results(result, retry_result)
            end
        end

        push!(results, result)
        isnothing(on_result) || on_result(result)
        warm_start = result.angles
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
