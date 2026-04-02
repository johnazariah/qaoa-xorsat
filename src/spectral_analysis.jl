"""
    SpectralSnapshot(step, magnitudes, effective_ranks)

WHT-domain spectrum of the branch tensor at a single iteration step.

- `step`: iteration index `t` (0 = initial all-ones)
- `magnitudes`: sorted descending absolute values of `WHT(B^(t))`
- `effective_ranks`: dictionary mapping tolerance `δ` → smallest `r` such that
  the `r`-term truncation has relative `ℓ∞` error ≤ `δ`
"""
struct SpectralSnapshot
    step::Int
    magnitudes::Vector{Float64}
    effective_ranks::Dict{Float64,Int}
end

const DEFAULT_TOLERANCES = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

"""
    spectral_snapshot(branch_tensor, step; tolerances)

Compute the WHT spectrum of `branch_tensor` and measure its effective rank at
each tolerance level.
"""
function spectral_snapshot(
    branch_tensor::AbstractVector{<:Number},
    step::Int;
    tolerances::AbstractVector{Float64}=DEFAULT_TOLERANCES,
)
    spectrum = wht(complex.(branch_tensor))
    magnitudes = sort(abs.(spectrum); rev=true)

    peak = magnitudes[1]
    effective_ranks = Dict{Float64,Int}()
    for δ in tolerances
        threshold = δ * peak
        # Smallest r such that all terms beyond r are below threshold
        r = length(magnitudes)
        for i in length(magnitudes):-1:1
            if magnitudes[i] ≥ threshold
                r = i
                break
            end
        end
        effective_ranks[δ] = r
    end

    SpectralSnapshot(step, magnitudes, effective_ranks)
end

"""
    spectral_decay_rate(snapshot)

Fit an exponential decay to the sorted magnitude spectrum.

Returns `(rate, r²)` where `rate` is the log₁₀ decay per decade of rank
(i.e., `|ĉ_r| ∝ 10^{-rate · log₁₀(r)}`), and `r²` is the coefficient of
determination for the linear fit in log-log space.

Only fits over entries with magnitude > `eps()` to avoid log(0).
"""
function spectral_decay_rate(snapshot::SpectralSnapshot)
    mags = snapshot.magnitudes
    # Filter to nonzero entries
    nonzero_mask = mags .> eps()
    nonzero_count = count(nonzero_mask)
    nonzero_count ≥ 2 || return (0.0, 0.0)

    log_ranks = log10.(1:nonzero_count)
    log_mags = log10.(mags[nonzero_mask])

    # Least squares fit: log_mag = a + b * log_rank
    n = nonzero_count
    sx = sum(log_ranks)
    sy = sum(log_mags)
    sxx = sum(log_ranks .^ 2)
    sxy = sum(log_ranks .* log_mags)

    denom = n * sxx - sx^2
    abs(denom) > eps() || return (0.0, 0.0)

    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n

    # R² coefficient of determination
    predicted = a .+ b .* log_ranks
    ss_res = sum((log_mags .- predicted) .^ 2)
    ss_tot = sum((log_mags .- mean(log_mags)) .^ 2)
    r_squared = ss_tot > eps() ? 1.0 - ss_res / ss_tot : 0.0

    (-b, r_squared)
end

"""
    mean(xs)

Arithmetic mean (avoids Statistics.jl dependency).
"""
mean(xs) = sum(xs) / length(xs)

"""
    SpectralProfile(params, angles, snapshots, wall_time_seconds)

Full spectral profile of the Basso iteration for a given `(k, D, p)` and angles.
"""
struct SpectralProfile
    params::TreeParams
    angles::QAOAAngles
    snapshots::Vector{SpectralSnapshot}
    wall_time_seconds::Float64
end

"""
    basso_branch_tensor_instrumented(params, angles; tolerances, f_table)

Run the Basso branch-tensor iteration with spectral instrumentation.

Returns the final branch tensor and a `SpectralProfile` containing a
`SpectralSnapshot` at each step `t = 0, 1, …, p`.
"""
function basso_branch_tensor_instrumented(
    params::TreeParams,
    angles::QAOAAngles{T};
    tolerances::AbstractVector{Float64}=DEFAULT_TOLERANCES,
    f_table::AbstractVector=basso_f_table(angles),
) where T
    depth(angles) == params.p || throw(ArgumentError("angle depth must match tree depth"))

    start_time = time()
    kernel = basso_constraint_kernel(angles, basso_branching_degree(params))

    current = ones(Complex{T}, basso_configuration_count(params.p))
    snapshots = SpectralSnapshot[]

    push!(snapshots, spectral_snapshot(current, 0; tolerances))

    for step in 1:params.p
        current = basso_branch_tensor_step(params, angles, current, f_table, kernel)
        push!(snapshots, spectral_snapshot(current, step; tolerances))
    end

    elapsed = time() - start_time
    profile = SpectralProfile(params, angles, snapshots, elapsed)

    (current, profile)
end

"""
    format_spectral_report(profile) -> String

Format a human-readable report of the spectral profile.
"""
function format_spectral_report(profile::SpectralProfile)
    k, D, p = profile.params.k, profile.params.D, profile.params.p
    N = basso_configuration_count(p)

    lines = String[]
    push!(lines, "Spectral Analysis: k=$k, D=$D, p=$p")
    push!(lines, "Configuration space: 2^$(basso_bit_count(p)) = $N")
    push!(lines, "Wall time: $(round(profile.wall_time_seconds; digits=2))s")
    push!(lines, "")

    # Header
    tols = sort(collect(keys(first(profile.snapshots).effective_ranks)); rev=true)
    tol_labels = ["δ=1e$(Int(log10(δ)))" for δ in tols]
    header = rpad("step", 6) * join([rpad(label, 12) for label in tol_labels])
    push!(lines, header)
    push!(lines, "-"^length(header))

    for snapshot in profile.snapshots
        row = rpad("t=$(snapshot.step)", 6)
        for δ in tols
            r = snapshot.effective_ranks[δ]
            pct = round(100.0 * r / N; digits=1)
            row *= rpad("$r ($(pct)%)", 12)
        end
        push!(lines, row)
    end

    push!(lines, "")
    push!(lines, "Decay analysis (log-log fit of sorted WHT magnitudes):")
    for snapshot in profile.snapshots
        rate, r² = spectral_decay_rate(snapshot)
        push!(lines, "  t=$(snapshot.step): decay_rate=$(round(rate; digits=3)), R²=$(round(r²; digits=4))")
    end

    join(lines, "\n")
end

"""
    write_spectral_csv(io, profile)

Write the full sorted magnitude spectrum to CSV for plotting.
"""
function write_spectral_csv(io::IO, profile::SpectralProfile)
    println(io, "step,rank,magnitude,relative_magnitude")
    for snapshot in profile.snapshots
        peak = snapshot.magnitudes[1]
        for (rank, mag) in enumerate(snapshot.magnitudes)
            println(io, "$(snapshot.step),$rank,$mag,$(mag/peak)")
        end
    end
end

"""
    write_effective_ranks_csv(io, profile)

Write effective ranks at each step × tolerance to CSV.
"""
function write_effective_ranks_csv(io::IO, profile::SpectralProfile)
    tols = sort(collect(keys(first(profile.snapshots).effective_ranks)))
    print(io, "step,N")
    for δ in tols
        print(io, ",rank_1e$(Int(log10(δ)))")
    end
    println(io)

    N = basso_configuration_count(profile.params.p)
    for snapshot in profile.snapshots
        print(io, "$(snapshot.step),$N")
        for δ in tols
            print(io, ",$(snapshot.effective_ranks[δ])")
        end
        println(io)
    end
end
