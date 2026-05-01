"""
Gradient checkpointing for the QAOA evaluation pipeline.

Instead of storing all p+1 intermediate branch tensors (O(p) memory),
stores only √p checkpoints and recomputes the rest during the backward
pass. This trades ~2× compute for √p memory, enabling higher depth p
on limited-memory hardware.

Supports optional disk spillover: checkpoints that don't fit in RAM
are written to NVMe and read back during backward. At p=15 this
enables computation on a 64GB machine.

Works with both CPU and GPU arrays.
"""

include("gpu_backward.jl")

using Serialization

# ── Checkpoint storage ────────────────────────────────────────────────

struct CheckpointStore{A}
    """Branch tensor checkpoints indexed by tree level."""
    tensors::Dict{Int, A}
    """Log-scale at each checkpoint level."""
    log_scales::Dict{Int, Float64}
    """Disk paths for spilled checkpoints (empty if all in RAM)."""
    disk_paths::Dict{Int, String}
    """Checkpoint levels (sorted)."""
    levels::Vector{Int}
end

function CheckpointStore(::Type{A}) where A
    CheckpointStore{A}(Dict{Int,A}(), Dict{Int,Float64}(), Dict{Int,String}(), Int[])
end

function store_checkpoint!(cs::CheckpointStore, level::Int, tensor, log_s::Float64;
                          disk_dir::Union{String,Nothing}=nothing)
    if disk_dir !== nothing
        path = joinpath(disk_dir, "checkpoint_B_$level.bin")
        open(path, "w") do io
            serialize(io, Array(tensor))
        end
        cs.disk_paths[level] = path
        # Don't keep in RAM
    else
        cs.tensors[level] = copy(tensor)
    end
    cs.log_scales[level] = log_s
    if !(level in cs.levels)
        push!(cs.levels, level)
        sort!(cs.levels)
    end
end

function load_checkpoint(cs::CheckpointStore, level::Int, gpu_array_fn)
    if haskey(cs.tensors, level)
        return copy(cs.tensors[level]), cs.log_scales[level]
    elseif haskey(cs.disk_paths, level)
        data = open(cs.disk_paths[level]) do io
            deserialize(io)
        end
        return gpu_array_fn(data), cs.log_scales[level]
    else
        error("No checkpoint at level $level")
    end
end

function cleanup_disk!(cs::CheckpointStore)
    for (_, path) in cs.disk_paths
        rm(path; force=true)
    end
end

# ── Recompute segment ─────────────────────────────────────────────────

"""
Recompute forward pass from checkpoint level `from` to level `to`,
returning intermediates needed for backward pass in that segment.

Returns: (B_segment, child_hat_segment, folded_segment, log_s_final)
where B_segment[t-from+1] = B[t] for t in from:to
"""
function recompute_segment(
    B_start, log_s_start::Float64,
    from::Int, to::Int,
    f_table_gpu, kernel_hat_gpu,
    arity::Int, degree::Int,
    _NORM_THRESHOLD,
)
    GT = eltype(B_start)
    N = length(B_start)
    segment_len = to - from

    B_seg = Vector{typeof(B_start)}(undef, segment_len + 1)
    child_hat_seg = Vector{typeof(B_start)}(undef, segment_len)
    folded_seg = Vector{typeof(B_start)}(undef, segment_len)

    B_seg[1] = copy(B_start)
    scratch = similar(B_start)
    log_s = log_s_start

    for i in 1:segment_len
        gpu_elemwise_mul!(scratch, f_table_gpu, B_seg[i])
        gpu_wht!(scratch)

        ch_scale = Float64(maximum(abs.(scratch)))
        if ch_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / ch_scale)
        else
            ch_scale = 1.0
        end
        child_hat_seg[i] = copy(scratch)

        gpu_fold!(scratch, kernel_hat_gpu, child_hat_seg[i], arity)
        gpu_iwht!(scratch)

        fld_scale = Float64(maximum(abs.(scratch)))
        if fld_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / fld_scale)
        else
            fld_scale = 1.0
        end
        folded_seg[i] = copy(scratch)

        B_seg[i+1] = gpu_complex_power(folded_seg[i], degree)

        log_s = arity * degree * log_s +
                arity * log(ch_scale) +
                degree * log(fld_scale)
    end

    (B_seg, child_hat_seg, folded_seg, log_s)
end

# ── Checkpointed forward+backward ────────────────────────────────────

"""
    gpu_checkpointed_forward_backward(params, angles, gpu_array_fn;
        clause_sign=1, checkpoint_interval=0, disk_dir=nothing)
        -> (value, γ_grad, β_grad)

GPU forward+backward with gradient checkpointing.

`checkpoint_interval`: levels between checkpoints (0 = auto = ceil(√p)).
`disk_dir`: if set, spill checkpoints to this directory when RAM is tight.

Memory: O(√p · N) instead of O(p · N), at ~2× compute cost for backward.
"""
function gpu_checkpointed_forward_backward(
    params::TreeParams,
    angles::QAOAAngles,
    gpu_array_fn::Function;
    clause_sign::Int=1,
    checkpoint_interval::Int=0,
    disk_dir::Union{String,Nothing}=nothing,
    max_ram_checkpoints::Int=typemax(Int),
)
    p = params.p
    k = params.k
    D = params.D
    arity = k - 1
    degree = D - 1
    bit_count = basso_bit_count(p)
    N = basso_configuration_count(p)

    # Auto checkpoint interval: ceil(√p)
    ci = checkpoint_interval > 0 ? checkpoint_interval : max(1, ceil(Int, sqrt(p)))

    depth(angles) == p || throw(ArgumentError("angle depth must match tree depth"))
    validate_clause_sign(clause_sign)

    # ── CPU precomputation ────────────────────────────────────────
    gamma_full = build_gamma_full_vector(angles)
    trig_table = basso_trig_table(angles)
    f_table_cpu = _basso_f_table_fast(trig_table, bit_count, N, Float64)

    half = 0.5
    phase_args = Vector{Float64}(undef, N)
    kernel_cpu = Vector{ComplexF64}(undef, N)
    for config in 0:N-1
        ph = _phase_dot(gamma_full, config, bit_count)
        phase_args[config+1] = ph
        kernel_cpu[config+1] = complex(cos(half * ph))
    end
    kernel_hat_cpu = QaoaXorsat.wht!(copy(kernel_cpu))

    root_parity_cpu = Float64[basso_root_parity(config, p) for config in 0:N-1]
    cs = Float64(clause_sign)
    root_kernel_cpu = [complex(zero(Float64), sin(half * cs * phase_args[config+1]))
                       for config in 0:N-1]

    # ── Transfer to GPU ───────────────────────────────────────────
    f_table_gpu = gpu_array_fn(f_table_cpu)
    kernel_hat_gpu = gpu_array_fn(kernel_hat_cpu)
    root_parity_gpu = gpu_array_fn(complex.(root_parity_cpu))
    root_kernel_gpu = gpu_array_fn(root_kernel_cpu)

    GT = eltype(f_table_gpu)
    backend = KernelAbstractions.get_backend(f_table_gpu)
    _NORM_THRESHOLD = real(GT)(1e15)

    # ── Forward pass with checkpointing ───────────────────────────
    checkpoints = CheckpointStore(typeof(f_table_gpu))
    if disk_dir !== nothing
        mkpath(disk_dir)
    end

    B_gpu = gpu_array_fn(ones(ComplexF64, N))
    scratch = similar(B_gpu)
    log_s = 0.0

    # Store initial checkpoint
    ram_count = 0
    store_checkpoint!(checkpoints, 1, B_gpu, log_s)
    ram_count += 1

    for t in 1:p
        gpu_elemwise_mul!(scratch, f_table_gpu, B_gpu)
        gpu_wht!(scratch)

        ch_scale = Float64(maximum(abs.(scratch)))
        if ch_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / ch_scale)
        else
            ch_scale = 1.0
        end

        gpu_fold!(scratch, kernel_hat_gpu, scratch, arity)
        gpu_iwht!(scratch)

        fld_scale = Float64(maximum(abs.(scratch)))
        if fld_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / fld_scale)
        else
            fld_scale = 1.0
        end

        B_gpu = gpu_complex_power(scratch, degree)

        log_s = arity * degree * log_s +
                arity * log(ch_scale) +
                degree * log(fld_scale)

        # Store checkpoint at interval boundaries and at end
        if t % ci == 0 || t == p
            use_disk = (ram_count >= max_ram_checkpoints && disk_dir !== nothing)
            store_checkpoint!(checkpoints, t + 1, B_gpu, log_s;
                            disk_dir=use_disk ? disk_dir : nothing)
            if !use_disk
                ram_count += 1
            end
        end
    end

    # ── Root fold ─────────────────────────────────────────────────
    root_msg = similar(B_gpu)
    mul3! = _elemwise_mul3_kernel!(backend)
    mul3!(root_msg, root_parity_gpu, f_table_gpu, B_gpu; ndrange=N)
    KernelAbstractions.synchronize(backend)

    gpu_wht!(root_msg)
    mh_scale = Float64(maximum(abs.(root_msg)))
    if mh_scale > Float64(_NORM_THRESHOLD)
        root_msg .*= real(GT)(1.0 / mh_scale)
    else
        mh_scale = 1.0
    end
    msg_hat_gpu = copy(root_msg)

    msg_hat_power = gpu_complex_power(msg_hat_gpu, k)
    conv_gpu = gpu_iwht(copy(msg_hat_power))

    product_tmp = similar(conv_gpu)
    gpu_elemwise_mul!(product_tmp, root_kernel_gpu, conv_gpu)
    S_normalized = ComplexF64(sum(Array(product_tmp)))

    log_total_scale = k * (log_s + log(mh_scale))

    # Compute value
    re_S_norm = real(S_normalized)
    if re_S_norm == 0 || !isfinite(log_total_scale)
        value = 0.5
    else
        log_product = log_total_scale + log(abs(re_S_norm))
        if log_product > 700
            value = NaN
        else
            scaled_re_S = copysign(exp(log_product), re_S_norm)
            value = (1 + clause_sign * scaled_re_S) / 2
        end
    end

    # ── Checkpointed backward pass ────────────────────────────────
    if !isfinite(log_total_scale) || log_total_scale > 700
        cleanup_disk!(checkpoints)
        return (value, zeros(Float64, p), zeros(Float64, p))
    end
    grad_scale = exp(log_total_scale) * cs / 2
    S_bar_val = GT(grad_scale)

    # Root backward (same as non-checkpointed)
    root_kernel_bar = S_bar_val .* conj.(conv_gpu)
    conv_bar = S_bar_val .* conj.(root_kernel_gpu)

    msg_hat_power_bar = gpu_iwht(copy(conv_bar))

    pa_kernel! = _power_adjoint_kernel!(backend)
    msg_hat_bar = similar(msg_hat_gpu)
    pa_kernel!(msg_hat_bar, msg_hat_gpu, msg_hat_power_bar, k; ndrange=N)
    KernelAbstractions.synchronize(backend)

    root_msg_bar = gpu_wht(copy(msg_hat_bar))

    f_table_bar = similar(f_table_gpu)
    B_bar = similar(f_table_gpu)

    rf_kernel! = _root_fanout_kernel!(backend)
    rf_kernel!(f_table_bar, B_bar, root_parity_gpu, f_table_gpu, B_gpu, root_msg_bar; ndrange=N)
    KernelAbstractions.synchronize(backend)

    kernel_hat_bar = gpu_array_fn(zeros(ComplexF64, N))

    # Process backward in segments between checkpoints
    cp_levels = checkpoints.levels  # sorted: [1, ci+1, 2ci+1, ..., p+1]

    for seg_idx in length(cp_levels):-1:2
        seg_end_level = cp_levels[seg_idx]      # B[seg_end_level] exists
        seg_start_level = cp_levels[seg_idx - 1] # B[seg_start_level] exists

        # Recompute forward from seg_start to seg_end
        B_start, log_s_start = load_checkpoint(checkpoints, seg_start_level, gpu_array_fn)

        from_t = seg_start_level      # tree level (1-indexed)
        to_t = seg_end_level - 1      # last iteration in this segment
        seg_len = to_t - from_t + 1

        if seg_len == 0
            continue
        end

        B_seg, ch_seg, fld_seg, _ = recompute_segment(
            B_start, log_s_start,
            from_t, to_t + 1,  # recompute from_t..to_t (to_t+1 exclusive)
            f_table_gpu, kernel_hat_gpu,
            arity, degree, _NORM_THRESHOLD,
        )

        # Backward through this segment
        for local_t in seg_len:-1:1
            global_t = from_t + local_t - 1

            # folded_bar = degree * conj(folded^(degree-1)) * B_bar
            folded_bar = similar(scratch)
            pa2! = _power_adjoint_kernel!(backend)
            pa2!(folded_bar, fld_seg[local_t], B_bar, degree; ndrange=N)
            KernelAbstractions.synchronize(backend)

            gpu_iwht!(folded_bar)

            accum! = _accum_conj_power_mul_kernel!(backend)
            accum!(kernel_hat_bar, ch_seg[local_t], folded_bar, arity; ndrange=N)
            KernelAbstractions.synchronize(backend)

            cha! = _child_hat_adjoint_kernel!(backend)
            cha!(scratch, ch_seg[local_t], kernel_hat_gpu, folded_bar, arity; ndrange=N)
            KernelAbstractions.synchronize(backend)

            gpu_wht!(scratch)

            fanout! = _fanout_kernel!(backend)
            fanout!(f_table_bar, B_bar, B_seg[local_t], f_table_gpu, scratch; ndrange=N)
            KernelAbstractions.synchronize(backend)
        end
    end

    cleanup_disk!(checkpoints)

    # ── Angle gradients on CPU (same as non-checkpointed) ─────────
    kernel_bar_cpu = ComplexF64.(Array(gpu_wht(copy(kernel_hat_bar))))
    root_kernel_bar_cpu = ComplexF64.(Array(root_kernel_bar))
    f_table_bar_cpu = ComplexF64.(Array(f_table_bar))

    gamma_full_bar = zeros(Float64, bit_count)
    for config in 0:N-1
        ph = phase_args[config+1]
        sin_ph = sin(half * ph)
        kb_re = real(kernel_bar_cpu[config+1])
        factor = -half * sin_ph * kb_re
        for index in 1:bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            gamma_full_bar[index] += factor * spin
        end
    end

    for config in 0:N-1
        ph = phase_args[config+1]
        theta = half * cs * ph
        cos_theta = cos(theta)
        rkb_im = imag(root_kernel_bar_cpu[config+1])
        factor = rkb_im * cos_theta * half * cs
        for index in 1:bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            gamma_full_bar[index] += factor * spin
        end
    end

    positions = QaoaXorsat.basso_phase_bit_positions(p)
    γ_bar = zeros(Float64, p)
    for round in 1:p
        bit_fwd = positions[round]
        bit_bwd = positions[2p - round + 1]
        γ_bar[round] += gamma_full_bar[bit_fwd]
        γ_bar[round] -= gamma_full_bar[bit_bwd]
    end

    β_bar = zeros(Float64, p)
    for round in 1:p
        mirror = 2p - round + 1
        β_r = angles.β[round]
        neg_tan = -tan(β_r)
        cot_val = cos(β_r) / sin(β_r)

        fwd_shift0 = round - 1
        fwd_shift1 = round
        bwd_shift0 = mirror - 1
        bwd_shift1 = mirror

        acc = 0.0
        for config in 0:N-1
            ft = f_table_cpu[config+1]
            ftb = f_table_bar_cpu[config+1]
            d_fwd = xor((config >> fwd_shift0) & 1, (config >> fwd_shift1) & 1)
            d_bwd = xor((config >> bwd_shift0) & 1, (config >> bwd_shift1) & 1)
            ld_fwd = d_fwd == 0 ? neg_tan : cot_val
            ld_bwd = d_bwd == 0 ? neg_tan : cot_val
            acc += real(conj(ftb) * ft) * (ld_fwd + ld_bwd)
        end
        β_bar[round] = acc
    end

    (value, γ_bar, β_bar)
end

# ── Streaming checkpointed forward+backward ──────────────────────────

"""
    gpu_streaming_forward_backward(params, angles, gpu_array_fn;
        clause_sign=1) -> (value, γ_grad, β_grad)

GPU forward+backward with **streaming** gradient checkpointing.

Stores every level's branch tensor in **CPU RAM** during the forward pass,
then recomputes one step at a time during the backward pass. All GPU
working buffers are pre-allocated — zero allocations inside the backward
loop.

Memory:
  - GPU: O(1) — fixed working set of ~9 vectors (no segment storage)
  - CPU: O(p · N) — one branch tensor per level in host RAM

This trades PCIe transfer bandwidth for GPU VRAM, enabling gradient
computation at higher depth p on memory-limited GPUs.

Compared to `gpu_checkpointed_forward_backward`:
  - Same compute cost (2× forward)
  - Much lower GPU peak memory (9 vectors vs √p·segment + overhead)
  - Higher CPU RAM usage (p+1 checkpoints vs √p)
"""
function gpu_streaming_forward_backward(
    params::TreeParams,
    angles::QAOAAngles,
    gpu_array_fn::Function;
    clause_sign::Int=1,
)
    p = params.p
    k = params.k
    D = params.D
    arity = k - 1
    degree = D - 1
    bit_count = basso_bit_count(p)
    N = basso_configuration_count(p)

    depth(angles) == p || throw(ArgumentError("angle depth must match tree depth"))
    validate_clause_sign(clause_sign)

    # ── CPU precomputation ────────────────────────────────────────
    gamma_full = build_gamma_full_vector(angles)
    trig_table = basso_trig_table(angles)
    f_table_cpu = _basso_f_table_fast(trig_table, bit_count, N, Float64)

    half = 0.5
    phase_args = Vector{Float64}(undef, N)
    kernel_cpu = Vector{ComplexF64}(undef, N)
    for config in 0:N-1
        ph = _phase_dot(gamma_full, config, bit_count)
        phase_args[config+1] = ph
        kernel_cpu[config+1] = complex(cos(half * ph))
    end
    kernel_hat_cpu = QaoaXorsat.wht!(copy(kernel_cpu))

    root_parity_cpu = Float64[basso_root_parity(config, p) for config in 0:N-1]
    cs = Float64(clause_sign)
    root_kernel_cpu = [complex(zero(Float64), sin(half * cs * phase_args[config+1]))
                       for config in 0:N-1]

    # ── Transfer to GPU ───────────────────────────────────────────
    f_table_gpu = gpu_array_fn(f_table_cpu)
    kernel_hat_gpu = gpu_array_fn(kernel_hat_cpu)
    root_parity_gpu = gpu_array_fn(complex.(root_parity_cpu))
    root_kernel_gpu = gpu_array_fn(root_kernel_cpu)

    GT = eltype(f_table_gpu)
    backend = KernelAbstractions.get_backend(f_table_gpu)
    _NORM_THRESHOLD = real(GT)(1e15)

    # ── Forward pass with CPU-offloaded checkpoints ───────────────
    # Checkpoint indexing: cpu_cp[t] = branch tensor B BEFORE step t
    # cpu_cp[1] = initial (ones), cpu_cp[t+1] = result after step t
    cpu_cp = Vector{Vector{GT}}(undef, p + 1)

    B_gpu = gpu_array_fn(ones(ComplexF64, N))
    scratch = similar(B_gpu)
    log_s = 0.0

    cpu_cp[1] = Array(B_gpu)

    for t in 1:p
        gpu_elemwise_mul!(scratch, f_table_gpu, B_gpu)
        gpu_wht!(scratch)

        ch_scale = Float64(maximum(abs.(scratch)))
        if ch_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / ch_scale)
        else
            ch_scale = 1.0
        end

        gpu_fold!(scratch, kernel_hat_gpu, scratch, arity)
        gpu_iwht!(scratch)

        fld_scale = Float64(maximum(abs.(scratch)))
        if fld_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / fld_scale)
        else
            fld_scale = 1.0
        end

        B_gpu = gpu_complex_power(scratch, degree)

        log_s = arity * degree * log_s +
                arity * log(ch_scale) +
                degree * log(fld_scale)

        cpu_cp[t + 1] = Array(B_gpu)
    end

    # ── Root fold ─────────────────────────────────────────────────
    root_msg = similar(B_gpu)
    mul3! = _elemwise_mul3_kernel!(backend)
    mul3!(root_msg, root_parity_gpu, f_table_gpu, B_gpu; ndrange=N)
    KernelAbstractions.synchronize(backend)

    gpu_wht!(root_msg)
    mh_scale = Float64(maximum(abs.(root_msg)))
    if mh_scale > Float64(_NORM_THRESHOLD)
        root_msg .*= real(GT)(1.0 / mh_scale)
    else
        mh_scale = 1.0
    end
    msg_hat_gpu = copy(root_msg)

    msg_hat_power = gpu_complex_power(msg_hat_gpu, k)
    conv_gpu = gpu_iwht(copy(msg_hat_power))

    product_tmp = similar(conv_gpu)
    gpu_elemwise_mul!(product_tmp, root_kernel_gpu, conv_gpu)
    S_normalized = ComplexF64(sum(Array(product_tmp)))

    log_total_scale = k * (log_s + log(mh_scale))

    # Compute value
    re_S_norm = real(S_normalized)
    if re_S_norm == 0 || !isfinite(log_total_scale)
        value = 0.5
    else
        log_product = log_total_scale + log(abs(re_S_norm))
        if log_product > 700
            value = NaN
        else
            scaled_re_S = copysign(exp(log_product), re_S_norm)
            value = (1 + clause_sign * scaled_re_S) / 2
        end
    end

    # ── Streaming backward pass ───────────────────────────────────
    if !isfinite(log_total_scale) || log_total_scale > 700
        return (value, zeros(Float64, p), zeros(Float64, p))
    end
    grad_scale = exp(log_total_scale) * cs / 2
    S_bar_val = GT(grad_scale)

    # Root backward
    root_kernel_bar = S_bar_val .* conj.(conv_gpu)
    conv_bar = S_bar_val .* conj.(root_kernel_gpu)

    msg_hat_power_bar = gpu_iwht(copy(conv_bar))

    pa_kernel! = _power_adjoint_kernel!(backend)
    msg_hat_bar = similar(msg_hat_gpu)
    pa_kernel!(msg_hat_bar, msg_hat_gpu, msg_hat_power_bar, k; ndrange=N)
    KernelAbstractions.synchronize(backend)

    root_msg_bar = gpu_wht(copy(msg_hat_bar))

    f_table_bar = similar(f_table_gpu)
    B_bar = similar(f_table_gpu)

    rf_kernel! = _root_fanout_kernel!(backend)
    rf_kernel!(f_table_bar, B_bar, root_parity_gpu, f_table_gpu, B_gpu, root_msg_bar; ndrange=N)
    KernelAbstractions.synchronize(backend)

    kernel_hat_bar = gpu_array_fn(zeros(ComplexF64, N))

    # Move root_kernel_bar to CPU (only needed for angle gradients)
    root_kernel_bar_cpu = ComplexF64.(Array(root_kernel_bar))

    # Free root backward temporaries — reuse as streaming buffers
    # buf_Bt: reloaded checkpoint; buf_child: child_hat; buf_fbar: folded_bar
    buf_Bt      = root_msg       # reuse allocation
    buf_child   = msg_hat_gpu    # reuse allocation
    buf_fbar    = conv_gpu       # reuse allocation
    # scratch already available

    # Hoist kernel objects outside the loop
    _pa2!     = _power_adjoint_kernel!(backend)
    _accum!   = _accum_conj_power_mul_kernel!(backend)
    _cha!     = _child_hat_adjoint_kernel!(backend)
    _fanout!  = _fanout_kernel!(backend)

    # ── Streaming backward loop (zero GPU allocations) ────────────
    for t in p:-1:1
        # Load B[t] from CPU checkpoint (branch tensor BEFORE step t)
        copyto!(buf_Bt, cpu_cp[t])

        # Recompute child_hat = WHT(f_table .* B[t]), normalized
        gpu_elemwise_mul!(buf_child, f_table_gpu, buf_Bt)
        gpu_wht!(buf_child)
        ch_scale = Float64(maximum(abs.(buf_child)))
        if ch_scale > Float64(_NORM_THRESHOLD)
            buf_child .*= real(GT)(1.0 / ch_scale)
        else
            ch_scale = 1.0
        end

        # Recompute folded = iWHT(fold(kernel_hat, child_hat)), normalized
        # Store in scratch (no need to keep beyond this step)
        gpu_fold!(scratch, kernel_hat_gpu, buf_child, arity)
        gpu_iwht!(scratch)
        fld_scale = Float64(maximum(abs.(scratch)))
        if fld_scale > Float64(_NORM_THRESHOLD)
            scratch .*= real(GT)(1.0 / fld_scale)
        else
            fld_scale = 1.0
        end
        # scratch = normalized folded[t]

        # Backward: folded_bar = degree * conj(folded^(degree-1)) * B_bar
        _pa2!(buf_fbar, scratch, B_bar, degree; ndrange=N)
        KernelAbstractions.synchronize(backend)
        gpu_iwht!(buf_fbar)

        # kernel_hat_bar += conj(child_hat^arity) * folded_bar
        _accum!(kernel_hat_bar, buf_child, buf_fbar, arity; ndrange=N)
        KernelAbstractions.synchronize(backend)

        # child_hat_bar (in scratch) = arity * conj(child_hat^(arity-1)) * conj(kernel_hat) * folded_bar
        _cha!(scratch, buf_child, kernel_hat_gpu, buf_fbar, arity; ndrange=N)
        KernelAbstractions.synchronize(backend)
        gpu_wht!(scratch)

        # Fan-out: f_table_bar += conj(B[t]) * scratch; B_bar = conj(f_table) * scratch
        _fanout!(f_table_bar, B_bar, buf_Bt, f_table_gpu, scratch; ndrange=N)
        KernelAbstractions.synchronize(backend)
    end

    # ── Angle gradients on CPU ────────────────────────────────────
    kernel_bar_cpu = ComplexF64.(Array(gpu_wht(copy(kernel_hat_bar))))
    f_table_bar_cpu = ComplexF64.(Array(f_table_bar))

    gamma_full_bar = zeros(Float64, bit_count)
    for config in 0:N-1
        ph = phase_args[config+1]
        sin_ph = sin(half * ph)
        kb_re = real(kernel_bar_cpu[config+1])
        factor = -half * sin_ph * kb_re
        for index in 1:bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            gamma_full_bar[index] += factor * spin
        end
    end

    for config in 0:N-1
        ph = phase_args[config+1]
        theta = half * cs * ph
        cos_theta = cos(theta)
        rkb_im = imag(root_kernel_bar_cpu[config+1])
        factor = rkb_im * cos_theta * half * cs
        for index in 1:bit_count
            spin = z_eigenvalue((config >> (index - 1)) & 1)
            gamma_full_bar[index] += factor * spin
        end
    end

    positions = QaoaXorsat.basso_phase_bit_positions(p)
    γ_bar = zeros(Float64, p)
    for round in 1:p
        bit_fwd = positions[round]
        bit_bwd = positions[2p - round + 1]
        γ_bar[round] += gamma_full_bar[bit_fwd]
        γ_bar[round] -= gamma_full_bar[bit_bwd]
    end

    β_bar = zeros(Float64, p)
    for round in 1:p
        mirror = 2p - round + 1
        β_r = angles.β[round]
        neg_tan = -tan(β_r)
        cot_val = cos(β_r) / sin(β_r)

        fwd_shift0 = round - 1
        fwd_shift1 = round
        bwd_shift0 = mirror - 1
        bwd_shift1 = mirror

        acc = 0.0
        for config in 0:N-1
            ft = f_table_cpu[config+1]
            ftb = f_table_bar_cpu[config+1]
            d_fwd = xor((config >> fwd_shift0) & 1, (config >> fwd_shift1) & 1)
            d_bwd = xor((config >> bwd_shift0) & 1, (config >> bwd_shift1) & 1)
            ld_fwd = d_fwd == 0 ? neg_tan : cot_val
            ld_bwd = d_bwd == 0 ? neg_tan : cot_val
            acc += real(conj(ftb) * ft) * (ld_fwd + ld_bwd)
        end
        β_bar[round] = acc
    end

    (value, γ_bar, β_bar)
end
