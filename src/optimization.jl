using Optim
using Random

struct AngleOptimizationResult
    angles::QAOAAngles
    value::Float64
    evaluations::Int
    starts::Int
    iterations::Int
    converged::Bool
end

default_clause_sign(k::Int) = k == 2 ? -1 : 1

function angle_vector(angles::QAOAAngles)::Vector{Float64}
    [angles.γ; angles.β]
end

function angles_from_vector(values::AbstractVector{<:Real}, p::Int)::QAOAAngles
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
function canonicalize_angles(angles::QAOAAngles)::QAOAAngles
    QAOAAngles(
        canonicalize_problem_angle.(angles.γ),
        canonicalize_mixer_angle.(angles.β),
    )
end

"""
    random_angles(p; rng=Random.default_rng()) -> QAOAAngles

Sample a random depth-`p` angle vector in the canonical search box.
"""
function random_angles(p::Int; rng=Random.default_rng())::QAOAAngles
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
)::QAOAAngles
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
    restarts::Int,
    rng,
)::Vector{QAOAAngles}
    restarts ≥ 0 || throw(ArgumentError("restarts must be ≥ 0, got $restarts"))
    all(depth(guess) == p for guess in initial_guesses) || throw(ArgumentError(
        "all initial guesses must have depth $p",
    ))

    guesses = canonicalize_angles.(collect(initial_guesses))
    isempty(guesses) && push!(guesses, random_angles(p; rng))

    for _ in 1:restarts
        push!(guesses, random_angles(p; rng))
    end

    guesses
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
- `rng`: random number generator for restart sampling
"""
function optimize_angles(
    params::TreeParams;
    clause_sign::Int=default_clause_sign(params.k),
    restarts::Int=8,
    maxiters::Int=200,
    initial_guesses::AbstractVector{<:QAOAAngles}=QAOAAngles[],
    rng=Random.default_rng(),
)::AngleOptimizationResult
    validate_clause_sign(clause_sign)
    maxiters ≥ 1 || throw(ArgumentError("maxiters must be ≥ 1, got $maxiters"))

    guesses = build_initial_guesses(params.p, initial_guesses, restarts, rng)

    best_angles = first(guesses)
    best_value = -Inf
    best_iterations = 0
    best_converged = false
    total_evaluations = 0

    for guess in guesses
        local_evaluations = Ref(0)

        function objective(values)
            local_evaluations[] += 1
            candidate = angles_from_vector(values, params.p) |> canonicalize_angles
            -qaoa_expectation(params, candidate; clause_sign)
        end

        result = Optim.optimize(
            objective,
            angle_vector(guess),
            Optim.LBFGS(),
            Optim.Options(iterations=maxiters, show_trace=false),
        )

        total_evaluations += local_evaluations[]

        candidate_angles = angles_from_vector(Optim.minimizer(result), params.p) |> canonicalize_angles
        candidate_value = qaoa_expectation(params, candidate_angles; clause_sign)

        if candidate_value > best_value
            best_angles = candidate_angles
            best_value = candidate_value
            best_iterations = Optim.iterations(result)
            best_converged = Optim.converged(result)
        end
    end

    AngleOptimizationResult(
        best_angles,
        best_value,
        total_evaluations,
        length(guesses),
        best_iterations,
        best_converged,
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
    rng=Random.default_rng(),
)::Vector{AngleOptimizationResult}
    validate_clause_sign(clause_sign)
    validated_p_values = validate_depth_sequence(collect(Int, p_values))

    results = AngleOptimizationResult[]
    warm_start = nothing

    for p in validated_p_values
        initial_guesses = isnothing(warm_start) ? QAOAAngles[] : [extend_angles(warm_start, p)]
        result = optimize_angles(
            TreeParams(k, D, p);
            clause_sign,
            restarts,
            maxiters,
            initial_guesses,
            rng,
        )
        push!(results, result)
        warm_start = result.angles
    end

    results
end
