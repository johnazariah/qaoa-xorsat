# Manual adjoint for the charge decomposition evaluator.
#
# Translates JPM's C++ adjoint into Julia. Computes exact gradients
# in ~3x one forward eval, replacing the (4p+1)x FD approach.
#
# Architecture:
#   forward_pass_cached  -> run all p branch levels + root, save intermediates
#   backward_root        -> adjoint of root contraction -> adj_rb
#   backward_branch      -> adjoint of one branch level -> adj_child (per level)

using Serialization

# ── Derivative primitives ────────────────────────────────────────

"""d/dgamma charge_weight_matrix(gamma)"""
function _charge_weight_deriv(gamma::Float64)
    c = cos(gamma); s = sin(gamma)
    dc2 = -2*c*s; ds2 = 2*c*s
    dics = im*(c^2 - s^2)
    zk = ComplexF64[1, 1, -1, -1]
    zb = ComplexF64[1, -1, 1, -1]
    hcat(fill(complex(dc2), 4), dics .* zb, -dics .* zk, ds2 .* zk .* zb)
end

"""d/dbeta doubled_mixer(beta)"""
function _doubled_mixer_deriv(beta::Float64)
    c = cos(beta); s = sin(beta)
    dRx = ComplexF64[-s -im*c; -im*c -s]
    Rx = ComplexF64[c -im*s; -im*s c]
    kron(dRx, conj.(Rx)) .+ kron(Rx, conj.(dRx))
end

"""d/dgamma root_charge_weights(gamma)"""
function _root_charge_deriv(gamma::Float64)
    c = cos(gamma); s = sin(gamma)
    ComplexF64[-2*c*s, im*(c^2 - s^2), -im*(c^2 - s^2), 2*c*s]
end

# ── Adjoint of mode_product_flat! ────────────────────────────────

function _mode_product_adjoint!(
    adj_src::Vector{ComplexF64},
    adj_W::Matrix{ComplexF64},
    W::Matrix{ComplexF64},
    src::Vector{ComplexF64},
    adj_dst::Vector{ComplexF64},
    stride::Int, N::Int,
)
    block = stride * 4
    @inbounds for outer in 0:block:N-1
        for inner in 0:stride-1
            base = outer + inner + 1
            for j in 1:4
                s = zero(ComplexF64)
                for h in 1:4
                    s += conj(W[h, j]) * adj_dst[base + (h-1)*stride]
                end
                adj_src[base + (j-1)*stride] = s
            end
            for h in 1:4
                av = adj_dst[base + (h-1)*stride]
                for j in 1:4
                    adj_W[h, j] += conj(src[base + (j-1)*stride]) * av
                end
            end
        end
    end
end

# ── Adjoint of wht_charge_contract_flat! ─────────────────────────

function _wht_adjoint!(
    adj_T::Vector{ComplexF64},
    adj_M::Matrix{ComplexF64},
    M::Matrix{ComplexF64},
    T_in::Vector{ComplexF64},
    adj_out::Vector{ComplexF64},
    R::Int, rest::Int,
)
    vec_len = 4 * rest
    sigma_stride = 4 * rest
    channel_size = R * vec_len
    row_stride = 16 * rest

    fill!(adj_T, zero(ComplexF64))

    @inbounds for i in 0:R-1
        t_base = i * row_stride
        for b in 0:3, r in 0:rest-1
            base_out = i * vec_len + b * rest + r + 1

            a0 = adj_out[0*channel_size + base_out]
            a1 = adj_out[1*channel_size + base_out]
            a2 = adj_out[2*channel_size + base_out]
            a3 = adj_out[3*channel_size + base_out]

            p02 = a0 + a1; q02 = a0 - a1
            p13 = a2 + a3; q13 = a2 - a3
            ae = (p02 + p13, q02 + q13, p02 - p13, q02 - q13)

            for s in 1:4
                src_idx = t_base + (s-1) * sigma_stride + b * rest + r + 1
                adj_T[src_idx] += conj(M[b+1, s]) * ae[s]
                adj_M[b+1, s] += conj(T_in[src_idx]) * ae[s]
            end
        end
    end
end

# ── Forward cache ────────────────────────────────────────────────

struct BranchState
    V_flat::Vector{ComplexF64}
    n_ch::Int
    t_after_perm::Vector{ComplexF64}
    t_max::Float64
    t_normalized::Vector{ComplexF64}
    F_powered::Vector{ComplexF64}
end

struct FwdCache
    p::Int; k::Int; D::Int; cs::Int
    gs::Vector{Float64}; bs::Vector{Float64}
    F_levels::Vector{Vector{ComplexF64}}
    F_maxs::Vector{Float64}
    children::Vector{Vector{ComplexF64}}
    states::Vector{BranchState}
    rb::Vector{ComplexF64}; rb_max::Float64
    log_s::Float64; raw::Float64; val::Float64
    spill_dir::Union{Nothing,String}
    own_spill_dir::Bool
    f_level_paths::Vector{String}
    child_paths::Vector{String}
    state_paths::Vector{String}
end

function _spill_path(dir::String, kind::String, level::Int)
    joinpath(dir, "$(kind)_$(level).bin")
end

function _spill_payload!(path::String, payload)
    open(path, "w") do io
        serialize(io, payload)
    end
end

function _load_payload(path::String)
    open(path, "r") do io
        deserialize(io)
    end
end

function _maybe_rm(path::String)
    isempty(path) && return
    isfile(path) && rm(path)
end

"""
Instrumented forward: compute branch F *and* save intermediates for backward.
Returns `(F_flat, BranchState)` in a single pass — no redundant replay.
"""
function _charge_branch_instrumented(gs, bs, nr, k, child)
    m = k - 1; CT = ComplexF64
    child_rounds = child !== nothing ? nr - 1 : 0

    # ── Phase 1: coupled contractions consuming child branch ──
    if child !== nothing && child_rounds >= 2
        V_flat = CT(0.5) .* copy(child)
        n_ch = 1; N_child = length(V_flat)
        scratch_v = similar(V_flat)
        for el in 1:child_rounds - 1
            epr = div(N_child, n_ch)
            _wht_charge_contract_flat!(scratch_v, doubled_mixer(bs[el]), V_flat, n_ch, epr)
            V_flat, scratch_v = scratch_v, V_flat
            n_ch *= 4
        end
        V = _reshape_c(V_flat, n_ch, 4)
    elseif child !== nothing
        V = reshape(CT(0.5) .* child, 1, 4); n_ch = 1
        V_flat = _vec_c(V)
    else
        V = fill(CT(0.5), 1, 4); n_ch = 1
        V_flat = _vec_c(V)
    end
    saved_V_flat = copy(V_flat)

    # ── Phase 2: fused mixer + trace ──
    start_mv = max(child_rounds - 1, 0)
    MD = [let Ml = doubled_mixer(bs[el])
        [Ml .* CT.(CHARGE_DIAG[a, :]') for a in 1:4]
    end for el in 1:nr]
    trace_vecs = [MD[nr][a][1, :] .+ MD[nr][a][4, :] for a in 1:4]
    trace_matrix = hcat(trace_vecs...)

    function _p2(V_local, l0)
        if l0 == nr - 1
            result = V_local * trace_matrix
            return vcat([result[:, a] for a in 1:4]...)
        end
        el = l0 + 1
        vcat([_p2(V_local * transpose(MD[el][a]), l0 + 1) for a in 1:4]...)
    end

    t_flat = _p2(V, start_mv)
    t = _reshape_c(t_flat, ntuple(_ -> 4, nr)...)

    remaining = nr - start_mv
    if nr > 1
        perm = vcat(collect(nr:-1:remaining+1), collect(1:remaining))
        t = permutedims(t, perm)
    end

    # ── Entrywise power with normalization — save intermediates ──
    if m > 1
        tmax = maximum(abs, t)
        tmax > 0 && (t = t ./ tmax)
    else
        tmax = 0.0
    end
    F_powered = _vec_c(t .^ m)
    # When m == 1 the backward never reads state.t_normalized (the m==1 branch
    # of the power backward returns `ac` directly without touching it).  Alias
    # to F_powered to save one full 4^lv vector per branch level.  Combined
    # over all p levels at k=2 this saves ~5.7 GB at p=14.
    t_normalized = m == 1 ? F_powered : _vec_c(copy(t))

    # ── Mode products ──
    W = [charge_weight_matrix(gs[el]) for el in 1:nr]
    N = length(F_powered)
    F_flat = copy(F_powered)
    scratch = similar(F_flat)
    for el in 1:nr
        stride = 4^(nr - el)
        _mode_product_flat!(scratch, F_flat, W[el], stride, N)
        F_flat, scratch = scratch, F_flat
    end

    state = BranchState(saved_V_flat, n_ch, F_powered, tmax, t_normalized, F_powered)
    (F_flat, state)
end

function _fwd_cached(params, angles; clause_sign=1, cache_disk_dir::Union{Nothing,String}=nothing)
    p = params.p; k = params.k; D = params.D; deg = D - 1
    gs = Float64(clause_sign) .* Float64.(angles.γ) ./ 2
    bs = Float64.(angles.β); CT = ComplexF64

    FL = Vector{Vector{CT}}(undef, p)
    FM = Vector{Float64}(undef, p)
    CH = Vector{Vector{CT}}(undef, p)
    ST = Vector{BranchState}(undef, p)
    f_level_paths = fill("", p)
    child_paths = fill("", p)
    state_paths = fill("", p)
    spill_dir = cache_disk_dir
    own_spill_dir = false
    if spill_dir === nothing
        spill_dir = nothing
    else
        mkpath(spill_dir)
    end

    function maybe_spill_level!(lv::Int)
        spill_dir === nothing && return

        if !isempty(FL[lv])
            path = _spill_path(spill_dir, "F", lv)
            _spill_payload!(path, FL[lv])
            f_level_paths[lv] = path
            FL[lv] = CT[]
        end

        if !isempty(CH[lv])
            path = _spill_path(spill_dir, "child", lv)
            _spill_payload!(path, CH[lv])
            child_paths[lv] = path
            CH[lv] = CT[]
        end

        if !isempty(ST[lv].F_powered)
            path = _spill_path(spill_dir, "state", lv)
            _spill_payload!(path, ST[lv])
            state_paths[lv] = path
            ST[lv] = BranchState(CT[], 0, CT[], 0.0, CT[], CT[])
        end
    end

    ls = 0.0

    F, st = _charge_branch_instrumented(gs, bs, 1, k, nothing)
    FL[1] = copy(F); FM[1] = 1.0; CH[1] = CT[]
    ST[1] = st

    for lv in 2:p
        fmx = maximum(abs, F); FM[lv-1] = fmx
        fmx > 0 && (F ./= fmx; ls += deg * log(fmx))
        ch = F .^ deg; CH[lv] = copy(ch)
        F, st = _charge_branch_instrumented(gs, bs, lv, k, ch)
        FL[lv] = copy(F)
        ST[lv] = st

        maybe_spill_level!(lv)
        if lv > 1
            maybe_spill_level!(lv - 1)
        end
    end

    if p == 1
        maybe_spill_level!(1)
    end

    fmx = maximum(abs, F); rbm = fmx
    fmx > 0 && (F ./= fmx; ls += deg * log(fmx))
    rb = F .^ deg
    raw = _charge_root_contract(rb, gs, bs, p, D, k; scratch=F)
    sv = raw * exp(k * ls)
    v = (1 + clause_sign * sv) / 2

    maybe_spill_level!(p)

    FwdCache(p, k, D, clause_sign, gs, bs, FL, FM, CH, ST, rb, rbm, ls, raw, v,
        spill_dir, own_spill_dir, f_level_paths, child_paths, state_paths)
end

function _load_f_level!(cache::FwdCache, lv::Int)
    if !isempty(cache.F_levels[lv])
        return cache.F_levels[lv]
    end
    path = cache.f_level_paths[lv]
    isempty(path) && return cache.F_levels[lv]
    cache.F_levels[lv] = _load_payload(path)
    cache.F_levels[lv]
end

function _load_child_level!(cache::FwdCache, lv::Int)
    if !isempty(cache.children[lv])
        return cache.children[lv]
    end
    path = cache.child_paths[lv]
    isempty(path) && return cache.children[lv]
    cache.children[lv] = _load_payload(path)
    cache.children[lv]
end

function _load_state_level!(cache::FwdCache, lv::Int)
    if !isempty(cache.states[lv].F_powered)
        return cache.states[lv]
    end
    path = cache.state_paths[lv]
    isempty(path) && return cache.states[lv]
    cache.states[lv] = _load_payload(path)
    cache.states[lv]
end

function _release_f_level!(cache::FwdCache, lv::Int)
    cache.F_levels[lv] = ComplexF64[]
    _maybe_rm(cache.f_level_paths[lv])
    cache.f_level_paths[lv] = ""
end

function _release_child_level!(cache::FwdCache, lv::Int)
    cache.children[lv] = ComplexF64[]
    _maybe_rm(cache.child_paths[lv])
    cache.child_paths[lv] = ""
end

function _release_state_level!(cache::FwdCache, lv::Int)
    cache.states[lv] = BranchState(ComplexF64[], 0, ComplexF64[], 0.0, ComplexF64[], ComplexF64[])
    _maybe_rm(cache.state_paths[lv])
    cache.state_paths[lv] = ""
end

# ── Backward branch ─────────────────────────────────────────────

function _bwd_branch!(adj_F, gs, bs, nr, k, child, state, gg, gb)
    m = k - 1; tt = 4^nr; CT = ComplexF64
    W = [charge_weight_matrix(gs[el]) for el in 1:nr]

    # Mode products backward
    ac = copy(adj_F)
    as_buf = similar(ac)
    for el in nr:-1:1
        stride = 4^(nr - el)
        mp_in = copy(state.F_powered)
        mp_sc = similar(mp_in)
        for ll in 1:el-1
            s = 4^(nr - ll)
            _mode_product_flat!(mp_sc, mp_in, W[ll], s, tt)
            mp_in, mp_sc = mp_sc, mp_in
        end
        aw = zeros(CT, 4, 4)
        _mode_product_adjoint!(as_buf, aw, W[el], mp_in, ac, stride, tt)
        dW = _charge_weight_deriv(gs[el])
        gg[el] += real(sum(conj.(aw) .* dW))
        ac = copy(as_buf)
    end

    # Power backward
    atn = if m == 0; zeros(CT, tt)
    elseif m == 1; ac
    else
        [abs(state.t_normalized[i]) > 0 ?
            ac[i] * conj(CT(m) * state.t_normalized[i]^(m-1)) : zero(CT)
         for i in 1:tt]
    end
    atp = m > 1 && state.t_max > 0 ? atn ./ state.t_max : atn

    # Permutation backward
    cr = child !== nothing ? nr - 1 : 0
    sm = max(cr - 1, 0); rem = nr - sm
    if nr > 1
        pm = vcat(collect(nr:-1:rem+1), collect(1:rem))
        ip = sortperm(pm)
        atp_nd = _reshape_c(atp, ntuple(_ -> 4, nr)...)
        at2_nd = permutedims(atp_nd, ip)
        at2 = _vec_c(at2_nd)
    else
        at2 = atp
    end

    # Phase 2 backward
    MD = [let Ml = doubled_mixer(bs[el])
        [Ml .* CT.(CHARGE_DIAG[a, :]') for a in 1:4]
    end for el in 1:nr]
    tv = [MD[nr][a][1,:] .+ MD[nr][a][4,:] for a in 1:4]
    tm = hcat(tv...)

    adj_md = [zeros(CT, 4, 4) for _ in 1:nr, _ in 1:4]
    Vp1 = reshape(state.V_flat, state.n_ch, 4)

    function p2b!(Vl, nrows, e0, aout, aV)
        if e0 == nr - 1
            for i in 1:nrows, s in 1:4
                su = zero(CT)
                for a in 1:4
                    idx = (a-1)*nrows + i
                    su += aout[idx] * conj(tm[s, a])
                    # Backward through tm: adj_tm[s,a] += conj(Vl[i,s]) * aout[idx]
                    # tm[s,a] = MD[nr][a][1,s] + MD[nr][a][4,s]
                    adj_tm_sa = conj(Vl[(i-1)*4 + s]) * aout[idx]
                    adj_md[nr, a][1, s] += adj_tm_sa
                    adj_md[nr, a][4, s] += adj_tm_sa
                end
                aV[(i-1)*4 + s] += su
            end
            return
        end
        el = e0 + 1; sub_sz = nrows
        for ll in e0+2:nr; sub_sz *= 4; end
        off = 0
        for a in 1:4
            Vs = zeros(CT, nrows * 4)
            for i in 1:nrows, col in 1:4
                su = zero(CT)
                for s in 1:4
                    su += Vl[(i-1)*4 + s] * MD[el][a][col, s]
                end
                Vs[(i-1)*4 + col] = su
            end
            aVs = zeros(CT, nrows * 4)
            p2b!(Vs, nrows, e0 + 1, view(aout, off+1:off+sub_sz), aVs)
            for i in 1:nrows, s in 1:4
                su = zero(CT)
                for col in 1:4
                    su += aVs[(i-1)*4 + col] * conj(MD[el][a][col, s])
                    adj_md[el, a][col, s] += conj(Vl[(i-1)*4 + s]) * aVs[(i-1)*4 + col]
                end
                aV[(i-1)*4 + s] += su
            end
            off += sub_sz
        end
    end

    aV = zeros(CT, length(state.V_flat))
    p2b!(vec(Vp1), state.n_ch, sm, at2, aV)

    for el in 1:nr
        aM = zeros(CT, 4, 4)
        for a in 1:4, row in 1:4, col in 1:4
            aM[row, col] += adj_md[el, a][row, col] * CT(CHARGE_DIAG[a, col])
        end
        dM = _doubled_mixer_deriv(bs[el])
        gb[el] += real(sum(conj.(aM) .* dM))
    end

    # Phase 1 backward
    child === nothing && return nothing
    cr < 2 && return aV .* 0.5

    # Memory-frugal Phase 1 backward.
    #
    # The previous version stored the full forward history `p1s` (cr vectors
    # of length 4^cr ComplexF64).  At cr = p-1 = 13 that is ~13 GB transient
    # inside a single _bwd_branch! call.  We instead keep only the seed
    # `vi = 0.5 * child` (1 vector) plus two ping-pong replay buffers and
    # recompute each historical state on demand at the start of the relevant
    # backward iteration.  Cost: O(cr²) extra _wht_charge_contract_flat! calls,
    # bounded by ~p²/2 ≪ p · 4^p work.
    csz = 4^cr
    vi          = CT(0.5) .* copy(child)
    replay_a    = similar(vi)
    replay_b    = similar(vi)

    ap1 = copy(aV); ncb = state.n_ch
    for el in (cr-1):-1:1
        ncb = div(ncb, 4); rest = div(csz, ncb * 16)
        Mb = doubled_mixer(bs[el])

        # Replay forward from vi for (el-1) WHT steps to reconstruct p1s[el].
        copyto!(replay_a, vi)
        ncf_r = 1
        for j in 1:el-1
            epr_r = div(csz, ncf_r)
            _wht_charge_contract_flat!(replay_b, doubled_mixer(bs[j]),
                                       replay_a, ncf_r, epr_r)
            replay_a, replay_b = replay_b, replay_a
            ncf_r *= 4
        end

        aT = zeros(CT, csz); aMw = zeros(CT, 4, 4)
        _wht_adjoint!(aT, aMw, Mb, replay_a, ap1, ncb, rest)
        dM = _doubled_mixer_deriv(bs[el])
        gb[el] += real(sum(conj.(aMw) .* dM))
        ap1 = aT
    end

    return ap1 .* 0.5
end

# ── Backward root ────────────────────────────────────────────────

function _bwd_root!(adj_raw, cache, gg, gb)
    p = cache.p; k = cache.k; D = cache.D
    gs = cache.gs; bs = cache.bs; CT = ComplexF64; N = length(cache.rb)

    # ── Phase A: forward chain from cache.rb (no history saved) ──
    # The previous implementation pushed a copy of `factor` at every round
    # into `fhist`, costing p · N ComplexF64 ≈ 60 GB at p=14, N=4^14.
    # We instead retain only the final state and replay the chain on demand
    # in the backward loop (Phase D).  Replay cost is O(p²) WHTs which is
    # negligible compared to the p · 4^p inner work.
    factor = copy(cache.rb); buf = similar(factor)
    R = 1; maxR = 4^(p-1)
    ca = Vector{CT}(undef, maxR); ca[1] = CT(0.5)^k
    cb = Vector{CT}(undef, maxR)

    for el in 1:p-1
        M = doubled_mixer(bs[el]); u = root_charge_weights(gs[el])
        epr = div(N, R)
        _wht_charge_contract_flat!(buf, M, factor, R, epr)
        factor, buf = buf, factor
        @inbounds for a in 1:4
            ua = u[a]
            for i in 1:R; cb[(a-1)*R+i] = ua * ca[i]; end
        end
        R *= 4; ca, cb = cb, ca
    end

    # `factor` now holds factor_final (read-only for the rest of this fn).
    # `buf` is a spare length-N buffer — we'll reuse it in Phase D as one
    # leg of the replay ping-pong, saving a 4.3 GB allocation at p=14.
    # `ca` holds ca_final at R = 4^(p-1); `cb` is the spare coefficient
    # buffer — we'll reuse the pair as the coefficient-replay scratch.

    Mf = doubled_mixer(bs[p]); uf = root_charge_weights(gs[p])
    eprf = div(N, R)
    R_final = R                          # = 4^(p-1)

    # ── Phase B: bs[p]/gs[p] gradients and seed fb ──
    fb = zeros(CT, N)
    for a in 1:4
        K = Mf .* CT.(CHARGE_DIAG[a, :]')
        tv = (K[1,1]-K[4,1], K[1,2]-K[4,2], K[1,3]-K[4,3], K[1,4]-K[4,4])
        for i in 0:R-1
            b = i * eprf
            z = factor[b+1]*tv[1] + factor[b+2]*tv[2] + factor[b+3]*tv[3] + factor[b+4]*tv[4]
            dz = adj_raw * uf[a] * ca[i+1] * k * z^(k-1)
            cdz = conj(dz)
            ctv1 = conj(tv[1]); ctv2 = conj(tv[2]); ctv3 = conj(tv[3]); ctv4 = conj(tv[4])
            fb[b+1] += cdz*ctv1; fb[b+2] += cdz*ctv2
            fb[b+3] += cdz*ctv3; fb[b+4] += cdz*ctv4
        end
        du = _root_charge_deriv(gs[p])
        sv = zero(CT)
        for i in 0:R-1
            b = i * eprf
            z = factor[b+1]*tv[1] + factor[b+2]*tv[2] + factor[b+3]*tv[3] + factor[b+4]*tv[4]
            sv += ca[i+1] * z^k
        end
        gg[p] += adj_raw * real(du[a] * sv)

        dM = _doubled_mixer_deriv(bs[p])
        dK = dM .* CT.(CHARGE_DIAG[a, :]')
        dtv = (dK[1,1]-dK[4,1], dK[1,2]-dK[4,2], dK[1,3]-dK[4,3], dK[1,4]-dK[4,4])
        for i in 0:R-1
            b = i * eprf
            z = factor[b+1]*tv[1] + factor[b+2]*tv[2] + factor[b+3]*tv[3] + factor[b+4]*tv[4]
            dzb = factor[b+1]*dtv[1] + factor[b+2]*dtv[2] + factor[b+3]*dtv[3] + factor[b+4]*dtv[4]
            gb[p] += adj_raw * real(uf[a] * ca[i+1] * k * z^(k-1) * dzb)
        end
    end

    # ── Phase C: precompute w_final ONCE (used by every backward iter) ──
    # w_final[j+1] = d(raw)/d(ca_final[j+1]) = Σ_a uf[a] · z_{a,j}^k
    # where z_{a,j} is the per-cell linear combination over factor_final.
    # The previous code recomputed this inside the loop over `el`, costing
    # O(p · R_final) redundant work and one R_final allocation per iter.
    w_final = zeros(CT, R_final)
    for a in 1:4
        K2 = Mf .* CT.(CHARGE_DIAG[a, :]')
        tv2 = (K2[1,1]-K2[4,1], K2[1,2]-K2[4,2], K2[1,3]-K2[4,3], K2[1,4]-K2[4,4])
        for j in 0:R_final-1
            bb = j * eprf
            z2 = factor[bb+1]*tv2[1] + factor[bb+2]*tv2[2] +
                 factor[bb+3]*tv2[3] + factor[bb+4]*tv2[4]
            w_final[j+1] += uf[a] * z2^k
        end
    end

    # ── Phase D: backward chain (el = p-1 → 1) ──
    # All scratch buffers are allocated ONCE outside the loop and reused.
    # Big buffers are aliased to Phase-A buffers that are no longer needed:
    #   factor (= factor_final, read-only past Phase C)  ↦  factor_replay
    #   buf    (spare from Phase A)                      ↦  buf_replay
    #   ca, cb (coefficient pair from Phase A)           ↦  ca_replay, cb_replay
    # Net new Phase-D allocations: fb_next (length N), aMw (4×4),
    # dca / dca_buf (length R_final each, 1 GB at p=14 — unavoidable).
    factor_replay = factor                   # alias: read-only past Phase C
    buf_replay    = buf                      # reuse Phase-A spare
    ca_replay     = ca                       # reuse Phase-A coefficients
    cb_replay     = cb
    fb_next       = similar(fb)
    aMw           = Matrix{CT}(undef, 4, 4)
    dca           = Vector{CT}(undef, R_final)
    dca_buf       = Vector{CT}(undef, R_final)

    for el in (p-1):-1:1
        M = doubled_mixer(bs[el]); Rpv = 4^(el-1)
        eprp = div(N, Rpv); rest = div(eprp, 16)

        # Reconstruct factor_at[el] (= fhist[el] in the old code) by
        # replaying the forward WHT chain from cache.rb for (el-1) steps.
        copyto!(factor_replay, cache.rb)
        R_replay = 1
        for el2 in 1:el-1
            M2 = doubled_mixer(bs[el2])
            epr2 = div(N, R_replay)
            _wht_charge_contract_flat!(buf_replay, M2, factor_replay,
                                       R_replay, epr2)
            factor_replay, buf_replay = buf_replay, factor_replay
            R_replay *= 4
        end

        # bs[el] gradient and propagated factor adjoint.
        fill!(aMw, zero(CT))            # _wht_adjoint! zeros adj_T but uses += on adj_M
        _wht_adjoint!(fb_next, aMw, M, factor_replay, fb, Rpv, rest)
        dM = _doubled_mixer_deriv(bs[el])
        gb[el] += real(sum(conj.(aMw) .* dM))

        # gs[el] gradient via coefficient-chain replay + propagation.
        # ca_before[i] = ca at start of round `el` (after rounds 1..el-1).
        ca_replay[1] = CT(0.5)^k
        R_tmp = 1
        for el2 in 1:el-1
            u2 = root_charge_weights(gs[el2])
            @inbounds for a2 in 1:4
                ua = u2[a2]
                for ii in 1:R_tmp
                    cb_replay[(a2-1)*R_tmp + ii] = ua * ca_replay[ii]
                end
            end
            R_tmp *= 4
            ca_replay, cb_replay = cb_replay, ca_replay
        end

        # Seed dca = ∂ca_after_el / ∂gs[el]  =  du[a] · ca_before[i].
        du = _root_charge_deriv(gs[el])
        R_el = 4^(el-1)
        fill!(dca, zero(CT))
        @inbounds for a in 1:4
            duA = du[a]
            for ii in 1:R_el
                dca[(a-1)*R_el + ii] = duA * ca_replay[ii]
            end
        end

        # Propagate dca through later rounds el+1..p-1 (ping-pong).
        R_prop = 4 * R_el
        for el2 in (el+1):(p-1)
            u2 = root_charge_weights(gs[el2])
            fill!(dca_buf, zero(CT))
            @inbounds for a2 in 1:4
                ua = u2[a2]
                for ii in 1:R_prop
                    dca_buf[(a2-1)*R_prop + ii] = ua * dca[ii]
                end
            end
            R_prop *= 4
            dca, dca_buf = dca_buf, dca
        end

        s = zero(CT)
        @inbounds for j in 1:R_final
            s += dca[j] * w_final[j]
        end
        gg[el] += adj_raw * real(s)

        # Promote fb_next into fb for the next (smaller-el) iter.
        fb, fb_next = fb_next, fb
    end
    return fb
end

# ── Public API ───────────────────────────────────────────────────

"""
    charge_expectation_and_gradient(params, angles; clause_sign=1)

Compute value and gradient via manual adjoint. Cost: ~3-5x forward.
"""
function charge_expectation_and_gradient(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
    cache_disk_dir::Union{Nothing,String}=nothing,
)
    p = params.p
    Diagnostics.diag_info("charge_expectation_and_gradient: k=$(params.k) D=$(params.D) p=$p"; level=1)

    local_cache_dir = cache_disk_dir
    own_cache_dir = false
    if local_cache_dir === nothing
        local_cache_dir = nothing
    else
        mkpath(local_cache_dir)
    end

    cache = Diagnostics.diag_phase("forward pass p=$p") do
        _fwd_cached(params, angles; clause_sign, cache_disk_dir=local_cache_dir)
    end
    cache = FwdCache(cache.p, cache.k, cache.D, cache.cs, cache.gs, cache.bs,
        cache.F_levels, cache.F_maxs, cache.children, cache.states,
        cache.rb, cache.rb_max, cache.log_s, cache.raw, cache.val,
        cache.spill_dir, own_cache_dir, cache.f_level_paths,
        cache.child_paths, cache.state_paths)
    k = cache.k; D = cache.D; deg = D - 1
    cs = Float64(cache.cs)

    gg = zeros(p); gb = zeros(p)
    sm = cs / 2 * exp(k * cache.log_s)

    t_root = @elapsed arb = _bwd_root!(sm, cache, gg, gb)
    Diagnostics.diag_time("backward root p=$p", t_root)

    Fp = _load_f_level!(cache, p); fmx = cache.rb_max
    Fn = Fp ./ fmx
    aFn = deg .* conj.(Fn .^ (deg-1)) .* arb
    aF = aFn ./ fmx
    # Free F_levels[p] eagerly — only Fn/aFn copies are needed below, and
    # those are local to this scope.  At p=14 this drops ~4.3 GB before the
    # branch backward begins, where many other large buffers will allocate.
    _release_f_level!(cache, p)
    Fp = Fn = aFn = nothing

    Diagnostics.diag_phase("backward branches p=$p") do
        for lv in p:-1:1
            ch = lv > 1 ? _load_child_level!(cache, lv) : nothing
            state = _load_state_level!(cache, lv)
            ach = _bwd_branch!(aF, cache.gs, cache.bs, lv, k, ch, state, gg, gb)

            # Eagerly drop the per-level branch payload now that _bwd_branch!
            # has finished consuming it.  Each BranchState at level lv holds
            # up to ~5.4 GB worth of ComplexF64 vectors at p=14, and children
            # add another ~4.3 GB at the top level.
            if lv > 1
                _release_child_level!(cache, lv)
            end
            _release_state_level!(cache, lv)
            ch = nothing

            if lv > 1 && ach !== nothing
                Fpv = _load_f_level!(cache, lv - 1); fmpv = cache.F_maxs[lv-1]
                Fnpv = Fpv ./ fmpv
                aFnpv = deg .* conj.(Fnpv .^ (deg-1)) .* ach
                aF = aFnpv ./ fmpv
                _release_f_level!(cache, lv - 1)
                Fpv = Fnpv = aFnpv = nothing
            end
            ach = nothing
        end
    end

    gg .*= cs / 2
    Diagnostics.diag_mem("gradient done p=$p")
    if cache.own_spill_dir && cache.spill_dir !== nothing && isdir(cache.spill_dir)
        rm(cache.spill_dir; recursive=true, force=true)
    end
    (cache.val, gg, gb)
end
