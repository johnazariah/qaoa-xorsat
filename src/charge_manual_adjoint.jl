# Manual adjoint for the charge decomposition evaluator.
#
# Translates JPM's C++ adjoint into Julia. Computes exact gradients
# in ~3x one forward eval, replacing the (4p+1)x FD approach.
#
# Architecture:
#   forward_pass_cached  -> run all p branch levels + root, save intermediates
#   backward_root        -> adjoint of root contraction -> adj_rb
#   backward_branch      -> adjoint of one branch level -> adj_child (per level)

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
    t_normalized = _vec_c(copy(t))
    F_powered = _vec_c(t .^ m)

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

function _fwd_cached(params, angles; clause_sign=1)
    p = params.p; k = params.k; D = params.D; deg = D - 1
    gs = Float64(clause_sign) .* Float64.(angles.γ) ./ 2
    bs = Float64.(angles.β); CT = ComplexF64

    FL = Vector{Vector{CT}}(undef, p)
    FM = Vector{Float64}(undef, p)
    CH = Vector{Vector{CT}}(undef, p)
    ST = Vector{BranchState}(undef, p)
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
    end

    fmx = maximum(abs, F); rbm = fmx
    fmx > 0 && (F ./= fmx; ls += deg * log(fmx))
    rb = F .^ deg
    raw = _charge_root_contract(rb, gs, bs, p, D, k; scratch=F)
    sv = raw * exp(k * ls)
    v = (1 + clause_sign * sv) / 2

    FwdCache(p, k, D, clause_sign, gs, bs, FL, FM, CH, ST, rb, rbm, ls, raw, v)
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

    csz = 4^cr
    vi = CT(0.5) .* copy(child)
    p1s = [copy(vi)]; ncf = 1; vc = copy(vi); sc = similar(vc)
    for el in 1:cr-1
        epr = div(csz, ncf)
        _wht_charge_contract_flat!(sc, doubled_mixer(bs[el]), vc, ncf, epr)
        vc, sc = sc, vc; ncf *= 4
        push!(p1s, copy(vc))
    end

    ap1 = copy(aV); ncb = state.n_ch
    for el in (cr-1):-1:1
        ncb = div(ncb, 4); rest = div(csz, ncb * 16)
        Mb = doubled_mixer(bs[el])
        aT = zeros(CT, csz); aMw = zeros(CT, 4, 4)
        _wht_adjoint!(aT, aMw, Mb, p1s[el], ap1, ncb, rest)
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

    factor = copy(cache.rb); buf = similar(factor)
    R = 1; maxR = 4^(p-1)
    ca = Vector{CT}(undef, maxR); ca[1] = CT(0.5)^k
    cb = Vector{CT}(undef, maxR)
    fhist = [copy(factor)]

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
        push!(fhist, copy(factor))
    end

    Mf = doubled_mixer(bs[p]); uf = root_charge_weights(gs[p])
    eprf = div(N, R)

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

    for el in (p-1):-1:1
        M = doubled_mixer(bs[el]); Rpv = 4^(el-1)
        eprp = div(N, Rpv); rest = div(eprp, 16)
        aT = zeros(CT, N); aMw = zeros(CT, 4, 4)
        _wht_adjoint!(aT, aMw, M, fhist[el], fb, Rpv, rest)
        dM = _doubled_mixer_deriv(bs[el])
        gb[el] += real(sum(conj.(aMw) .* dM))

        # Coefficient adjoint: gamma gradient through root_charge_weights(gs[el])
        # At round el, coefficient update was: cb[(a-1)*R_el + i] = u[a] * ca_el[i]
        # The final raw depends on the coefficients, so d(raw)/d(u[a]) contributes to gg[el]
        # We need adj_ca at this level: how much does each ca affect the final raw?
        # Propagate adj_ca backward from the final measurement
        du = _root_charge_deriv(gs[el])
        R_el = 4^(el-1)
        # Reconstruct adj_ca at level el from fb (adjoint of factor at level el)
        # The coefficient adjoint is: d(raw)/d(u_el[a]) = Σ_i adj_ca_after[(a-1)*R_el+i] * ca_before[i]
        # But adj_ca is entangled with the forward pass in complex ways.
        # Simple approach: use the stored ca values and the final measurement structure.
        # The coefficient after round el is: ca_after[(a-1)*R_el + i] = u_el[a] * ca_before[i]
        # The final raw uses these coefficients linearly: raw = Σ_i ca_final[i] * w[i]
        # where w[i] = Σ_a uf[a] * z_{a,i}^k (weighted by final-round u)
        # So d(raw)/d(ca_final[i]) = w[i]
        # Chain backward through subsequent coefficient rounds to get d(raw)/d(u_el[a])

        # For now, compute numerically via the scalar relationship:
        # raw depends on gs[el] through u = root_charge_weights(gs[el])
        # u enters only through the coefficient array, not through factor
        # So d(raw)/d(gs[el]) = Σ_a du[a] * Σ_i ca_partial[i] * z_final[...] where the
        # ca_partial terms collect the contribution from this specific u[a]

        # Efficient approach: replay the coefficient chain from round el onward
        # and accumulate the gradient of raw w.r.t. u_el[a]
        u_el = root_charge_weights(gs[el])
        R_after = 4 * R_el  # R after round el
        # Reconstruct ca_before from ca_after and u
        # ca_before[i] = ca_after[(a-1)*R_el + i] / u_el[a] for any a where u_el[a] != 0
        # But it's easier to just recompute by replaying forward up to el-1
        ca_replay = Vector{CT}(undef, maxR)
        ca_replay[1] = CT(0.5)^k
        cb_replay = Vector{CT}(undef, maxR)
        R_tmp = 1
        for el2 in 1:el-1
            u2 = root_charge_weights(gs[el2])
            @inbounds for a2 in 1:4
                for ii in 1:R_tmp; cb_replay[(a2-1)*R_tmp+ii] = u2[a2] * ca_replay[ii]; end
            end
            R_tmp *= 4; ca_replay, cb_replay = cb_replay, ca_replay
        end
        # ca_replay now holds ca_before (coefficients before round el)
        # At round el: ca_after[(a-1)*R_el + i] = u_el[a] * ca_before[i]
        # d(ca_after[(a-1)*R_el + i])/d(gs[el]) = du[a] * ca_before[i]
        # Need to propagate through subsequent rounds to get d(ca_final)/d(gs[el])
        # Then: d(raw)/d(gs[el]) = Σ_j d(ca_final[j])/d(gs[el]) * w_final[j]
        # where w_final[j] = Σ_a uf[a] * z_{a,j}^k (the final measurement contribution per coeff)

        # Compute w_final: d(raw)/d(ca_final[j])
        # From the final measurement: raw = Σ_a uf[a] * Σ_j ca_final[j+1] * z_{a,j}^k
        # So d(raw)/d(ca_final[j+1]) = Σ_a uf[a] * z_{a,j}^k
        R_final = 4^(p-1)
        eprf_final = div(N, R_final)
        w_final = zeros(CT, R_final)
        factor_final = fhist[end]
        for a in 1:4
            K2 = Mf .* CT.(CHARGE_DIAG[a, :]')
            tv2 = (K2[1,1]-K2[4,1], K2[1,2]-K2[4,2], K2[1,3]-K2[4,3], K2[1,4]-K2[4,4])
            for j in 0:R_final-1
                bb = j * eprf_final
                z2 = factor_final[bb+1]*tv2[1] + factor_final[bb+2]*tv2[2] +
                     factor_final[bb+3]*tv2[3] + factor_final[bb+4]*tv2[4]
                w_final[j+1] += uf[a] * z2^k
            end
        end

        # Propagate d(ca)/d(gs[el]) forward through rounds el+1..p-1
        # Starting: d(ca_after_el)/d(gs[el])[(a-1)*R_el+i] = du[a] * ca_before[i]
        dca = zeros(CT, R_final)  # will hold d(ca_final)/d(gs[el])
        # Initialize at round el
        for a in 1:4
            for ii in 1:R_el
                dca[(a-1)*R_el + ii] = du[a] * ca_replay[ii]
            end
        end
        # Propagate through rounds el+1..p-1
        R_prop = R_after
        for el2 in (el+1):(p-1)
            u2 = root_charge_weights(gs[el2])
            dca_new = zeros(CT, R_final)
            for a2 in 1:4
                for ii in 1:R_prop
                    dca_new[(a2-1)*R_prop + ii] = u2[a2] * dca[ii]
                end
            end
            R_prop *= 4; dca = dca_new
        end

        # d(raw)/d(gs[el]) = Σ_j dca[j] * w_final[j]
        gg[el] += adj_raw * real(sum(dca .* w_final))

        fb = aT
    end
    return fb
end

# ── Public API ───────────────────────────────────────────────────

"""
    charge_expectation_and_gradient(params, angles; clause_sign=1)

Compute value and gradient via manual adjoint. Cost: ~3-5x forward.
"""
function charge_expectation_and_gradient(
    params::TreeParams, angles::QAOAAngles; clause_sign::Int=1,
)
    p = params.p
    Diagnostics.diag_info("charge_expectation_and_gradient: k=$(params.k) D=$(params.D) p=$p"; level=1)

    cache = Diagnostics.diag_phase("forward pass p=$p") do
        _fwd_cached(params, angles; clause_sign)
    end
    k = cache.k; D = cache.D; deg = D - 1
    cs = Float64(cache.cs)

    gg = zeros(p); gb = zeros(p)
    sm = cs / 2 * exp(k * cache.log_s)

    t_root = @elapsed arb = _bwd_root!(sm, cache, gg, gb)
    Diagnostics.diag_time("backward root p=$p", t_root)

    Fp = cache.F_levels[p]; fmx = cache.rb_max
    Fn = Fp ./ fmx
    aFn = deg .* conj.(Fn .^ (deg-1)) .* arb
    aF = aFn ./ fmx

    Diagnostics.diag_phase("backward branches p=$p") do
        for lv in p:-1:1
            ch = lv > 1 ? cache.children[lv] : nothing
            ach = _bwd_branch!(aF, cache.gs, cache.bs, lv, k, ch, cache.states[lv], gg, gb)
            if lv > 1 && ach !== nothing
                Fpv = cache.F_levels[lv-1]; fmpv = cache.F_maxs[lv-1]
                Fnpv = Fpv ./ fmpv
                aFnpv = deg .* conj.(Fnpv .^ (deg-1)) .* ach
                aF = aFnpv ./ fmpv
            end
        end
    end

    gg .*= cs / 2
    Diagnostics.diag_mem("gradient done p=$p")
    (cache.val, gg, gb)
end
