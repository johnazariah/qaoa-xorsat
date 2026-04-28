# CPU gradient checkpointing for the Basso evaluator.
#
# Instead of storing all p+1 branch tensors + p child_hat + p folded
# (O(3p · N) memory), stores only ceil(√p) branch-tensor checkpoints
# (O(√p · N) memory) and recomputes the intermediates during backward.
#
# Trade-off: ~2× backward compute for √p/3p memory reduction.
# At p=13, N=2^27: full cache = ~42 GB, checkpointed = ~8 GB.
#
# Supports optional disk spillover: checkpoints that exceed a RAM cap
# are serialised to disk and read back during backward. This enables
# p=17 on a 1.4 TB machine (each checkpoint is 549 GB).
#
# This enables p=13 on a 64 GB machine, p=16 on 1.4 TB, p=17 with disk.

using Serialization

"""
    CheckpointedForwardCache{T}

Minimal cache for checkpointed evaluation: stores only √p branch-tensor
checkpoints plus the tables and root-fold intermediates needed for backward.
"""
struct CheckpointedForwardCache{T<:Real}
    # Params
    p::Int
    k::Int
    D::Int
    clause_sign::Int
    bit_count::Int
    configuration_count::Int
    arity::Int
    degree::Int

    # Precomputed tables (shared, not per-step)
    β::Vector{T}
    gamma_full::Vector{T}
    trig_table::Matrix{Complex{T}}
    f_table::Vector{Complex{T}}
    phase_args::Vector{T}
    kernel::Vector{Complex{T}}
    kernel_hat::Vector{Complex{T}}

    # Branch-tensor checkpoints: B at selected levels, with log-scale
    checkpoint_levels::Vector{Int}        # sorted levels where B is stored (1-indexed)
    checkpoint_B::Dict{Int, Vector{Complex{T}}}  # RAM checkpoints
    checkpoint_log_s::Dict{Int, T}
    checkpoint_disk_paths::Dict{Int, String}     # disk-spilled checkpoints

    # Root computation (always stored — needed for backward)
    root_msg::Vector{Complex{T}}
    root_parity_signs::Vector{Int}
    root_kernel::Vector{Complex{T}}
    msg_hat::Vector{Complex{T}}
    msg_hat_scale::T
    msg_hat_power::Vector{Complex{T}}
    B_final::Vector{Complex{T}}          # B[p+1] for root backward

    # Final result
    log_total_scale::T
    S_normalized::Complex{T}
    value::T
end

"""
    _forward_pass_checkpointed(params, angles; clause_sign=1, checkpoint_interval=0)

Run the forward pass storing only √p checkpoints instead of all intermediates.
"""
function _forward_pass_checkpointed(
    params::TreeParams,
    angles::QAOAAngles{T};
    clause_sign::Int=1,
    checkpoint_interval::Int=0,
    disk_dir::Union{String,Nothing}=nothing,
    max_ram_checkpoints::Int=typemax(Int),
) where T
    p = params.p
    k = params.k
    D = params.D
    arity = k - 1
    degree = D - 1
    bit_count = basso_bit_count(p)
    N = basso_configuration_count(p)

    ci = checkpoint_interval > 0 ? checkpoint_interval : max(1, ceil(Int, sqrt(p)))

    depth(angles) == p || throw(ArgumentError("angle depth must match tree depth"))
    validate_clause_sign(clause_sign)

    # Precompute tables (same as _forward_pass)
    gamma_full = build_gamma_full_vector(angles)
    trig_table = basso_trig_table(angles)
    f_table = _basso_f_table_fast(trig_table, bit_count, N, T)

    half = one(T) / 2
    phase_args = Vector{T}(undef, N)
    kernel = Vector{Complex{T}}(undef, N)
    Threads.@threads for config in 0:N-1
        ph = _phase_dot(gamma_full, config, bit_count)
        @inbounds phase_args[config+1] = ph
        @inbounds kernel[config+1] = complex(cos(half * ph))
    end
    kernel_hat = wht!(copy(kernel))

    # Forward iteration with selective checkpointing
    _NORM_THRESHOLD = T(1e30)

    checkpoint_B = Dict{Int, Vector{Complex{T}}}()
    checkpoint_log_s = Dict{Int, T}()
    checkpoint_disk_paths = Dict{Int, String}()
    checkpoint_levels = Int[]
    ram_count = 0

    if disk_dir !== nothing
        mkpath(disk_dir)
    end

    B = ones(Complex{T}, N)
    log_s = zero(T)
    scratch = Vector{Complex{T}}(undef, N)

    # Store initial checkpoint (level 1 = B[1]) — always in RAM
    checkpoint_B[1] = copy(B)
    checkpoint_log_s[1] = log_s
    push!(checkpoint_levels, 1)
    ram_count += 1

    for t in 1:p
        # child_weights = f_table .* B
        @inbounds @simd for i in 1:N
            scratch[i] = f_table[i] * B[i]
        end
        wht!(scratch)

        ch_scale = maximum(abs, scratch)
        if ch_scale > _NORM_THRESHOLD
            inv_ch = one(T) / ch_scale
            @inbounds @simd for i in 1:N
                scratch[i] *= inv_ch
            end
        else
            ch_scale = one(T)
        end

        # folded = iWHT(kernel_hat .* child_hat .^ arity)
        @inbounds @simd for i in 1:N
            scratch[i] = kernel_hat[i] * _fast_pow(scratch[i], arity)
        end
        iwht!(scratch)

        fld_scale = maximum(abs, scratch)
        if fld_scale > _NORM_THRESHOLD
            inv_fld = one(T) / fld_scale
            @inbounds @simd for i in 1:N
                scratch[i] *= inv_fld
            end
        else
            fld_scale = one(T)
        end

        # B = folded .^ degree
        @inbounds @simd for i in 1:N
            B[i] = _fast_pow(scratch[i], degree)
        end

        log_s = arity * degree * log_s +
                arity * log(ch_scale) +
                degree * log(fld_scale)

        # Store checkpoint at interval boundaries and at end
        if t % ci == 0 || t == p
            checkpoint_log_s[t + 1] = log_s
            push!(checkpoint_levels, t + 1)

            if ram_count >= max_ram_checkpoints && disk_dir !== nothing
                # Spill to disk
                path = joinpath(disk_dir, "checkpoint_B_$(t+1).bin")
                open(path, "w") do io
                    serialize(io, B)
                end
                checkpoint_disk_paths[t + 1] = path
            else
                checkpoint_B[t + 1] = copy(B)
                ram_count += 1
            end
        end
    end
    sort!(checkpoint_levels)

    # Root computation (same as _forward_pass)
    root_parity_signs = [basso_root_parity(config, p) for config in 0:N-1]
    root_msg = root_parity_signs .* f_table .* B

    cs = T(clause_sign)
    root_kernel = Vector{Complex{T}}(undef, N)
    Threads.@threads for config in 0:N-1
        @inbounds root_kernel[config+1] = complex(zero(T), sin(half * cs * phase_args[config+1]))
    end

    msg_hat_raw = wht!(complex.(root_msg))
    mh_scale = maximum(abs, msg_hat_raw)
    if mh_scale > _NORM_THRESHOLD
        msg_hat_raw .*= one(T) / mh_scale
    else
        mh_scale = one(T)
    end
    msg_hat = msg_hat_raw
    msg_hat_power = msg_hat .^ k
    conv = iwht(msg_hat_power)
    S_normalized = sum(root_kernel .* conv)

    log_total_scale = k * (log_s + log(mh_scale))

    re_S_norm = real(S_normalized)
    if re_S_norm == 0 || !isfinite(log_total_scale)
        value = half
    else
        log_product = log_total_scale + log(abs(re_S_norm))
        if log_product > 700
            value = T(NaN)
        else
            scaled_re_S = copysign(exp(log_product), re_S_norm)
            value = (1 + clause_sign * scaled_re_S) / 2
        end
    end

    CheckpointedForwardCache{T}(
        p, k, D, clause_sign,
        bit_count, N, arity, degree,
        copy(angles.β), gamma_full, trig_table,
        f_table, phase_args,
        kernel, kernel_hat,
        checkpoint_levels, checkpoint_B, checkpoint_log_s, checkpoint_disk_paths,
        root_msg, root_parity_signs, root_kernel,
        msg_hat, mh_scale, msg_hat_power,
        copy(B),  # B_final
        log_total_scale, S_normalized, value,
    )
end

"""
    _recompute_segment_cpu(B_start, from_t, to_t, f_table, kernel_hat, arity, degree, N, T)

Recompute forward pass from level `from_t` to `to_t`, returning per-step
intermediates (B, child_hat, folded, scales) needed for backward.
"""
function _recompute_segment_cpu(
    B_start::Vector{Complex{T}},
    from_t::Int, to_t::Int,
    f_table::Vector{Complex{T}},
    kernel_hat::Vector{Complex{T}},
    arity::Int, degree::Int,
    N::Int,
) where T
    _NORM_THRESHOLD = T(1e30)
    seg_len = to_t - from_t + 1

    B_seg = Vector{Vector{Complex{T}}}(undef, seg_len + 1)
    child_hat_seg = Vector{Vector{Complex{T}}}(undef, seg_len)
    folded_seg = Vector{Vector{Complex{T}}}(undef, seg_len)
    child_hat_scales_seg = Vector{T}(undef, seg_len)
    folded_scales_seg = Vector{T}(undef, seg_len)

    B_seg[1] = copy(B_start)
    scratch = Vector{Complex{T}}(undef, N)

    for i in 1:seg_len
        @inbounds @simd for j in 1:N
            scratch[j] = f_table[j] * B_seg[i][j]
        end
        wht!(scratch)

        ch_scale = maximum(abs, scratch)
        if ch_scale > _NORM_THRESHOLD
            inv_ch = one(T) / ch_scale
            @inbounds @simd for j in 1:N
                scratch[j] *= inv_ch
            end
        else
            ch_scale = one(T)
        end
        child_hat_scales_seg[i] = ch_scale
        child_hat_seg[i] = copy(scratch)

        @inbounds @simd for j in 1:N
            scratch[j] = kernel_hat[j] * _fast_pow(scratch[j], arity)
        end
        iwht!(scratch)

        fld_scale = maximum(abs, scratch)
        if fld_scale > _NORM_THRESHOLD
            inv_fld = one(T) / fld_scale
            @inbounds @simd for j in 1:N
                scratch[j] *= inv_fld
            end
        else
            fld_scale = one(T)
        end
        folded_scales_seg[i] = fld_scale
        folded_seg[i] = copy(scratch)

        new_B = Vector{Complex{T}}(undef, N)
        @inbounds @simd for j in 1:N
            new_B[j] = _fast_pow(scratch[j], degree)
        end
        B_seg[i + 1] = new_B
    end

    (B_seg, child_hat_seg, folded_seg, child_hat_scales_seg, folded_scales_seg)
end

"""
    _backward_pass_checkpointed(cache) -> (γ_grad, β_grad)

Backward pass with gradient checkpointing: recomputes forward segments
between checkpoints during backward to avoid storing all intermediates.
"""
function _backward_pass_checkpointed(cache::CheckpointedForwardCache{T}) where T
    p = cache.p
    N = cache.configuration_count
    cs = T(cache.clause_sign)
    half = one(T) / 2

    log_lts = cache.log_total_scale
    if !isfinite(log_lts) || log_lts > 700
        return (zeros(T, p), zeros(T, p))
    end
    grad_scale = exp(log_lts) * cs / 2
    S_bar = complex(grad_scale)

    # Root backward (identical to non-checkpointed)
    conv = iwht(cache.msg_hat_power)
    root_kernel_bar = S_bar .* conj.(conv)
    conv_bar = S_bar .* conj.(cache.root_kernel)

    msg_hat_power_bar = iwht(conv_bar)
    msg_hat_bar = cache.k .* conj.(cache.msg_hat .^ (cache.k - 1)) .* msg_hat_power_bar
    root_msg_bar = wht(msg_hat_bar)

    f_table_bar = conj.(cache.root_parity_signs .* cache.B_final) .* root_msg_bar
    B_bar = conj.(cache.root_parity_signs .* cache.f_table) .* root_msg_bar

    kernel_hat_bar = zeros(Complex{T}, N)
    scratch = Vector{Complex{T}}(undef, N)

    # Process backward through segments between checkpoints
    cp_levels = cache.checkpoint_levels  # sorted: [1, ci+1, 2ci+1, ..., p+1]

    for seg_idx in length(cp_levels):-1:2
        seg_end_level = cp_levels[seg_idx]      # checkpoint exists here
        seg_start_level = cp_levels[seg_idx - 1] # checkpoint exists here

        from_t = seg_start_level       # first iteration in segment (1-indexed)
        to_t = seg_end_level - 1       # last iteration in segment
        seg_len = to_t - from_t + 1

        if seg_len == 0
            continue
        end

        # Recompute forward through this segment
        # Load checkpoint from RAM or disk
        if haskey(cache.checkpoint_B, seg_start_level)
            B_start = cache.checkpoint_B[seg_start_level]
        elseif haskey(cache.checkpoint_disk_paths, seg_start_level)
            B_start = open(cache.checkpoint_disk_paths[seg_start_level]) do io
                deserialize(io)
            end
        else
            error("No checkpoint at level $seg_start_level")
        end
        B_seg, child_hat_seg, folded_seg, _, _ = _recompute_segment_cpu(
            B_start, from_t, to_t,
            cache.f_table, cache.kernel_hat,
            cache.arity, cache.degree, N,
        )

        # Backward through this segment (same as _backward_pass loop body)
        for local_t in seg_len:-1:1
            # B[t+1] = folded[t] .^ degree
            @inbounds @simd for i in 1:N
                scratch[i] = cache.degree * conj(_fast_pow(folded_seg[local_t][i], cache.degree - 1)) * B_bar[i]
            end

            iwht!(scratch)

            @inbounds @simd for i in 1:N
                kernel_hat_bar[i] += conj(_fast_pow(child_hat_seg[local_t][i], cache.arity)) * scratch[i]
            end

            @inbounds @simd for i in 1:N
                scratch[i] = cache.arity * conj(_fast_pow(child_hat_seg[local_t][i], cache.arity - 1)) *
                             conj(cache.kernel_hat[i]) * scratch[i]
            end

            wht!(scratch)

            @inbounds for i in 1:N
                f_table_bar[i] += conj(B_seg[local_t][i]) * scratch[i]
                B_bar[i] = conj(cache.f_table[i]) * scratch[i]
            end
        end
    end

    # Angle gradients (identical to non-checkpointed _backward_pass)
    kernel_bar = wht(kernel_hat_bar)

    gamma_full_bar = zeros(T, cache.bit_count)
    for config in 0:N-1
        ph = cache.phase_args[config+1]
        sin_ph = sin(half * ph)
        kb_re = real(kernel_bar[config+1])
        factor = -half * sin_ph * kb_re
        for index in 1:cache.bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            @inbounds gamma_full_bar[index] += factor * spin
        end
    end

    for config in 0:N-1
        ph = cache.phase_args[config+1]
        theta = half * cs * ph
        cos_theta = cos(theta)
        rkb_im = imag(root_kernel_bar[config+1])
        factor = rkb_im * cos_theta * half * cs
        for index in 1:cache.bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            @inbounds gamma_full_bar[index] += factor * spin
        end
    end

    positions = basso_phase_bit_positions(p)
    γ_bar = zeros(T, p)
    for round in 1:p
        bit_fwd = positions[round]
        bit_bwd = positions[2p - round + 1]
        γ_bar[round] += gamma_full_bar[bit_fwd]
        γ_bar[round] -= gamma_full_bar[bit_bwd]
    end

    β_bar = zeros(T, p)
    for round in 1:p
        β_r = cache.β[round]
        neg_tan = -tan(β_r)
        cot_val = cos(β_r) / sin(β_r)

        acc = zero(T)
        fwd_shift0 = round - 1
        fwd_shift1 = round
        bwd_shift0 = 2p - round
        bwd_shift1 = 2p - round + 1

        for config in 0:N-1
            @inbounds ft = cache.f_table[config+1]
            @inbounds ftb = f_table_bar[config+1]

            d_fwd = xor((config >> fwd_shift0) & 1, (config >> fwd_shift1) & 1)
            d_bwd = xor((config >> bwd_shift0) & 1, (config >> bwd_shift1) & 1)

            ld_fwd = d_fwd == 0 ? neg_tan : cot_val
            ld_bwd = d_bwd == 0 ? neg_tan : cot_val

            acc += real(conj(ftb) * ft) * (ld_fwd + ld_bwd)
        end
        β_bar[round] = acc
    end

    (γ_bar, β_bar)
end

# ── Public API ────────────────────────────────────────────────────────

"""
    _cleanup_checkpoints!(cache::CheckpointedForwardCache)

Remove disk-spilled checkpoint files.
"""
function _cleanup_checkpoints!(cache::CheckpointedForwardCache)
    for (_, path) in cache.checkpoint_disk_paths
        rm(path; force=true)
    end
end

"""
    basso_expectation_and_gradient_checkpointed(params, angles;
        clause_sign=1, checkpoint_interval=0, disk_dir=nothing, max_ram_checkpoints=typemax(Int))
        -> (value, γ_grad, β_grad)

Same as `basso_expectation_and_gradient` but uses gradient checkpointing
to reduce memory from O(p · N) to O(√p · N) at ~2× backward compute cost.

Use this for p ≥ 13 on memory-limited hardware.

`disk_dir`: if set, spill checkpoints to this directory when RAM cap is reached.
`max_ram_checkpoints`: maximum number of checkpoints to keep in RAM.
"""
function basso_expectation_and_gradient_checkpointed(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
    checkpoint_interval::Int=0,
    disk_dir::Union{String,Nothing}=nothing,
    max_ram_checkpoints::Int=typemax(Int),
)
    cache = _forward_pass_checkpointed(params, angles;
        clause_sign, checkpoint_interval, disk_dir, max_ram_checkpoints)
    γ_grad, β_grad = _backward_pass_checkpointed(cache)
    _cleanup_checkpoints!(cache)
    (cache.value, γ_grad, β_grad)
end

"""
    basso_expectation_checkpointed(params, angles; clause_sign=1)

Forward-only checkpointed evaluation (no gradient, minimal memory).
"""
function basso_expectation_checkpointed(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
)
    cache = _forward_pass_checkpointed(params, angles; clause_sign)
    _cleanup_checkpoints!(cache)
    cache.value
end
