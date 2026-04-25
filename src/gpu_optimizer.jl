"""
GPU-accelerated QAOA angle optimizer.

Wraps gpu_checkpointed_forward_backward into Optim.jl's L-BFGS,
enabling GPU-accelerated optimization at high depth p.

Usage:
    include("src/gpu_optimizer.jl")
    result = gpu_optimize_angles(TreeParams(2,3,10), gpu_array;
                                 clause_sign=-1, restarts=8)
"""

include("gpu_checkpointed.jl")

using Optim

"""
    gpu_optimize_angles(params, gpu_array_fn;
        clause_sign, restarts, maxiters, initial_guesses,
        g_abstol, checkpoint_interval, disk_dir) -> (angles, value)

Optimize QAOA angles using GPU-accelerated evaluation and gradient.

Returns (best_angles::QAOAAngles, best_value::Float64).
"""
function gpu_optimize_angles(
    params::TreeParams,
    gpu_array_fn::Function;
    clause_sign::Int=QaoaXorsat.default_clause_sign(params.k),
    restarts::Int=8,
    maxiters::Int=200,
    initial_guesses::AbstractVector{<:QAOAAngles}=QAOAAngles[],
    g_abstol::Float64=1e-6,
    checkpoint_interval::Int=0,
    disk_dir::Union{String,Nothing}=nothing,
    rng=Random.default_rng(),
    verbose::Bool=true,
)
    p = params.p

    # Build initial guesses
    guesses = QAOAAngles[]

    # Add user-provided guesses
    for g in initial_guesses
        if QaoaXorsat.depth(g) == p
            push!(guesses, g)
        end
    end

    # Fill with random starts
    while length(guesses) < restarts + length(initial_guesses)
        push!(guesses, QaoaXorsat.random_angles(p; rng))
    end

    best_angles = guesses[1]
    best_value = -Inf
    total_evals = 0

    for (start_idx, guess) in enumerate(guesses)
        eval_count = Ref(0)

        function fg_gpu!(G, values)
            eval_count[] += 1
            candidate = QaoaXorsat.angles_from_vector(values, p)

            val, γg, βg = gpu_checkpointed_forward_backward(
                params, candidate, gpu_array_fn;
                clause_sign,
                checkpoint_interval,
                disk_dir,
            )

            fval = Float64(val)
            if !QaoaXorsat.is_valid_qaoa_value(fval) ||
               any(!isfinite, γg) || any(!isfinite, βg)
                # Overflow guard
                for j in eachindex(G)
                    G[j] = values[j] > 0 ? 1.0 : -1.0
                end
                return 1.0e6
            end

            G[1:p] .= .-Float64.(γg)
            G[p+1:2p] .= .-Float64.(βg)
            return -fval
        end

        x0 = QaoaXorsat.angle_vector(guess)
        od = Optim.OnceDifferentiable(
            x -> fg_gpu!(similar(x), x),  # f only (unused but required)
            (G, x) -> fg_gpu!(G, x),       # g! only
            fg_gpu!,                        # fg! combined
            x0,
        )

        result = Optim.optimize(
            od, x0,
            Optim.LBFGS(),
            Optim.Options(
                iterations=maxiters,
                g_abstol=g_abstol,
                f_reltol=1e-12,
                store_trace=false,
                show_trace=false,
            ),
        )

        total_evals += eval_count[]
        candidate = QaoaXorsat.angles_from_vector(Optim.minimizer(result), p)
        candidate_val = -Optim.minimum(result)

        if QaoaXorsat.is_valid_qaoa_value(candidate_val) && candidate_val > best_value
            best_value = candidate_val
            best_angles = candidate
        end

        if verbose
            converged = Optim.converged(result) ? "✓" : "✗"
            @printf("  start %2d/%d: c̃=%.10f  iters=%d  evals=%d  %s\n",
                    start_idx, length(guesses), candidate_val,
                    Optim.iterations(result), eval_count[], converged)
        end
    end

    if verbose
        @printf("  BEST: c̃=%.10f  total_evals=%d\n", best_value, total_evals)
    end

    (best_angles, best_value)
end
