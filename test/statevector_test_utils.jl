using LinearAlgebra
using QaoaXorsat

function _cpu_cost_phase!(state, diagonal, gamma)
    @inbounds @. state *= cis(-gamma * diagonal)
    state
end

function _cpu_x_mixer!(state, beta, N)
    cosine = cos(beta)
    sine = complex(0, -sin(beta))
    @inbounds for qubit in 0:(N-1)
        bit = 1 << qubit
        step = bit << 1
        for base in 0:step:(length(state)-1), offset in 0:(bit-1)
            zero_index = base + offset + 1
            one_index = zero_index + bit
            zero_amplitude = state[zero_index]
            one_amplitude = state[one_index]
            state[zero_index] = cosine * zero_amplitude + sine * one_amplitude
            state[one_index] = sine * zero_amplitude + cosine * one_amplitude
        end
    end
    state
end

function _cpu_x_sum!(out, state, N)
    fill!(out, zero(eltype(out)))
    @inbounds for basis in 0:(length(state)-1), qubit in 0:(N-1)
        out[basis+1] += state[(basis ⊻ (1 << qubit))+1]
    end
    out
end

function _cpu_fast_value_gradient(N, diagonal, angles)
    p = depth(angles)
    state = fill(ComplexF64(inv(sqrt(1 << N))), 1 << N)
    adjoint = similar(state)
    scratch = similar(state)
    for layer in 1:p
        _cpu_cost_phase!(state, diagonal, angles.γ[layer])
        _cpu_x_mixer!(state, angles.β[layer], N)
    end
    @. adjoint = diagonal * state
    value = real(dot(state, adjoint))
    gradient = zeros(2p)
    for gate in 2p:-1:1
        if isodd(gate)
            layer = (gate + 1) >> 1
            @. scratch = diagonal * adjoint
            gradient[layer] = -2 * imag(dot(state, scratch))
            _cpu_cost_phase!(adjoint, diagonal, -angles.γ[layer])
            _cpu_cost_phase!(state, diagonal, -angles.γ[layer])
        else
            layer = gate >> 1
            _cpu_x_sum!(scratch, adjoint, N)
            gradient[p+layer] = -2 * imag(dot(state, scratch))
            _cpu_x_mixer!(adjoint, -angles.β[layer], N)
            _cpu_x_mixer!(state, -angles.β[layer], N)
        end
    end
    value, gradient
end

function _test_diagonal(N)
    terms = PauliTerm[
        z_term(qubit, (-1)^qubit * (0.1 + qubit / 17))
        for qubit in 1:N
    ]
    append!(terms, [
        zz_term(qubit, qubit + 1, 0.2 - qubit / 31)
        for qubit in 1:(N-1)
    ])
    QaoaXorsat._diagonal_from_pauli_terms(N, terms), terms
end
