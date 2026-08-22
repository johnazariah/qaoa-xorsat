using KernelAbstractions

const _STATEVECTOR_MAX_MEMORY_FRACTION = 0.8
const _STATEVECTOR_TRANSIENT_FRACTION = 0.25

function _statevector_source_revision()
    root = dirname(@__DIR__)
    try
        revision = readchomp(`git -C $root rev-parse HEAD`)
        dirty = !isempty(readchomp(`git -C $root status --porcelain`))
        dirty ? "$revision-dirty" : revision
    catch
        "QaoaXorsat-$(Base.pkgversion(@__MODULE__))"
    end
end

struct StatevectorExecutionStats
    backend::Symbol
    device::Union{String,Nothing}
    complex_type::Type
    kernel_launches::Int
    predicted_live_bytes::Int
    reservation_before::Int
    reservation_after::Int
    memory_cap_bytes::Int
    source_revision::String
    evaluation_count::Int
    first_evaluation_seconds::Union{Float64,Nothing}
    steady_state_seconds::Float64
    synchronized_launch_seconds::Float64
    compile_seconds::Union{Float64,Nothing}
    launch_seconds::Union{Float64,Nothing}
    kernel_execution_seconds::Union{Float64,Nothing}
    kernel_timing_source::Symbol
    allocation_seconds::Float64
    transfer_seconds::Float64
    verification_seconds::Float64
end

struct StatevectorMemoryAdmission
    backend::Symbol
    N::Int
    dimension::Int
    complex_type::Type
    predicted_live_bytes::Int
    free_bytes::Int
    reported_total_bytes::Int
    admission_total_bytes::Int
    reserved_bytes::Int
    memory_cap_bytes::Int
    memory_fraction::Float64
end

mutable struct _StatevectorRouteCounter
    kernel_launches::Int
    reservation_after::Int
    evaluation_count::Int
    first_evaluation_seconds::Union{Float64,Nothing}
    steady_state_seconds::Float64
    synchronized_launch_seconds::Float64
    kernel_execution_seconds::Union{Float64,Nothing}
    kernel_timing_source::Symbol
    allocation_seconds::Float64
    transfer_seconds::Float64
    verification_seconds::Float64
end

"""
Prepared exact QAOA evaluator for a diagonal cost and `sum(X)` mixer.

The device owns the cost diagonal, normalized `|+>^N` state, working state,
adjoint, and one scratch vector. Evaluation mutates scratch and is not
thread-safe; create one evaluator per concurrent task.
"""
struct DeviceStatevectorEvaluator{B,D,C,R}
    backend::B
    N::Int
    diagonal::D
    reduction::C
    state::C
    adjoint::C
    scratch::C
    gradient::Vector{Float64}
    memory_fraction::Float64
    predicted_live_bytes::Int
    reservation_before::Int
    memory_cap_bytes::Int
    route::R
    source_revision::String
end

@kernel function _sv_cost_phase_kernel!(state, @Const(diagonal), @Const(gamma))
    index = @index(Global)
    @inbounds begin
        angle = -gamma * diagonal[index]
        state[index] *= complex(cos(angle), sin(angle))
    end
end

@kernel function _sv_x_mixer_kernel!(state, @Const(bit), @Const(cosine), @Const(sine))
    pair = @index(Global)
    pair -= 1
    block = pair ÷ bit
    offset = pair % bit
    zero_index = block * (bit << 1) + offset + 1
    one_index = zero_index + bit
    @inbounds begin
        zero_amplitude = state[zero_index]
        one_amplitude = state[one_index]
        state[zero_index] = cosine * zero_amplitude + sine * one_amplitude
        state[one_index] = sine * zero_amplitude + cosine * one_amplitude
    end
end

@kernel function _sv_diagonal_apply_kernel!(out, @Const(diagonal), @Const(state))
    index = @index(Global)
    @inbounds out[index] = diagonal[index] * state[index]
end

@kernel function _sv_x_sum_kernel!(out, @Const(state), @Const(N))
    index = @index(Global)
    basis = index - 1
    value = zero(eltype(state))
    @inbounds for qubit in 0:(N-1)
        value += state[(basis ⊻ (1 << qubit)) + 1]
    end
    @inbounds out[index] = value
end

@kernel function _sv_inner_terms_kernel!(out, @Const(left), @Const(right))
    index = @index(Global)
    @inbounds out[index] = conj(left[index]) * right[index]
end

@kernel function _sv_expectation_terms_kernel!(out, @Const(diagonal), @Const(state))
    index = @index(Global)
    @inbounds out[index] = diagonal[index] * abs2(state[index])
end

@kernel function _sv_fill_kernel!(state, @Const(value))
    index = @index(Global)
    @inbounds state[index] = value
end

@kernel function _sv_pair_reduce_kernel!(out, @Const(input), @Const(input_length))
    index = @index(Global)
    first = (index << 1) - 1
    @inbounds begin
        value = input[first]
        second = first + 1
        if second <= input_length
            value += input[second]
        end
        out[index] = value
    end
end

function _recorded_launch!(ev::DeviceStatevectorEvaluator, kernel, args...; ndrange)
    ka_backend = KernelAbstractions.get_backend(ev.state)
    launch = () -> kernel(args...; ndrange)
    timer = _gpu_kernel_timer(ev.backend)
    started = time_ns()
    device_seconds = if timer === nothing
        launch()
        nothing
    else
        Float64(Base.invokelatest(timer, launch))
    end
    KernelAbstractions.synchronize(ka_backend)
    synchronized_seconds = (time_ns() - started) / 1e9
    ev.route.synchronized_launch_seconds += synchronized_seconds
    ev.route.kernel_launches += 1
    if device_seconds !== nothing
        current = something(ev.route.kernel_execution_seconds, 0.0)
        ev.route.kernel_execution_seconds = current + device_seconds
    elseif ev.backend.kind == :cpu
        ev.route.kernel_execution_seconds += synchronized_seconds
    end
    nothing
end

function _statevector_memory_bytes(dimension::Int, ::Type{CT}) where {CT<:Complex}
    RT = real(CT)
    device_bytes = Base.checked_mul(dimension, 4 * sizeof(CT) + sizeof(RT))
    host_shared_bytes = Base.checked_mul(dimension, sizeof(CT) + sizeof(RT))
    transient_bytes = ceil(Int,
        _STATEVECTOR_TRANSIENT_FRACTION * (device_bytes + host_shared_bytes))
    Base.checked_add(Base.checked_add(device_bytes, host_shared_bytes), transient_bytes)
end

function _validate_memory_fraction(memory_fraction::Real)
    fraction = Float64(memory_fraction)
    isfinite(fraction) || throw(ArgumentError("memory_fraction must be finite"))
    0 < fraction <= _STATEVECTOR_MAX_MEMORY_FRACTION || throw(ArgumentError(
        "memory_fraction must be in (0, $(_STATEVECTOR_MAX_MEMORY_FRACTION)]",
    ))
    fraction
end

function _admit_statevector_memory(
    backend::GPUBackend,
    status::GPUMemoryStatus,
    predicted_live_bytes::Int,
    memory_fraction::Float64,
    ;
    host_total_bytes::Integer=Sys.total_memory(),
)
    admission_total = _statevector_admission_total(
        backend,
        status,
        host_total_bytes,
    )
    cap = floor(Int, memory_fraction * admission_total)
    status.reserved_bytes <= cap || throw(GPUBackendError(
        backend.kind,
        "allocator reservation $(status.reserved_bytes) bytes already exceeds " *
        "the memory cap $cap bytes",
    ))
    required = Base.checked_add(status.reserved_bytes, predicted_live_bytes)
    required <= cap || throw(GPUBackendError(
        backend.kind,
        "statevector admission rejected before allocation: reserved " *
        "$(status.reserved_bytes) + predicted live $predicted_live_bytes = " *
        "$required bytes exceeds cap $cap bytes",
    ))
    predicted_live_bytes <= status.free_bytes || throw(GPUBackendError(
        backend.kind,
        "statevector admission rejected before allocation: predicted live " *
        "$predicted_live_bytes bytes exceeds reported free memory " *
        "$(status.free_bytes) bytes",
    ))
    cap
end

function _statevector_admission_total(
    backend::GPUBackend,
    status::GPUMemoryStatus,
    host_total_bytes::Integer,
)
    host_total = Int(host_total_bytes)
    host_total > 0 || throw(ArgumentError("host total memory must be positive"))
    backend.kind == :amdgpu ?
        min(status.total_bytes, host_total) :
        status.total_bytes
end

function _statevector_postcheck!(ev::DeviceStatevectorEvaluator)
    started = time_ns()
    status = gpu_memory_status(ev.backend)
    ev.route.reservation_after = status.reserved_bytes
    _assert_observed_memory(
        ev.backend.kind,
        status.reserved_bytes,
        ev.memory_cap_bytes,
    )
    ev.route.verification_seconds += (time_ns() - started) / 1e9
    nothing
end

function _assert_observed_memory(kind::Symbol, reserved_bytes::Int, cap_bytes::Int)
    reserved_bytes <= cap_bytes || throw(GPUBackendError(
        kind,
        "statevector result rejected: observed allocator reservation " *
        "$reserved_bytes bytes exceeds cap $cap_bytes bytes",
    ))
    true
end

function _statevector_dimension(N::Int)
    N >= 1 || throw(ArgumentError("N must be at least 1"))
    N < Sys.WORD_SIZE - 1 || throw(ArgumentError(
        "N=$N is too large for state indexing on a $(Sys.WORD_SIZE)-bit system",
    ))
    one(Int) << N
end

function _validate_statevector_size(N::Int, diagonal)
    dimension = _statevector_dimension(N)
    length(diagonal) == dimension || throw(DimensionMismatch(
        "diagonal length $(length(diagonal)) does not equal 2^N = $dimension",
    ))
    all(isfinite, diagonal) || throw(ArgumentError("cost diagonal must be finite"))
    dimension
end

"""
    statevector_memory_admission(
        backend::GPUBackend,
        N::Int;
        memory_fraction=0.8,
    ) -> StatevectorMemoryAdmission

Check statevector memory admission without allocating any `O(2^N)` arrays.
For AMD unified memory, the admission total is the lesser of the HIP-reported
aperture and physical host RAM. Throws `GPUBackendError` when inadmissible.
"""
function statevector_memory_admission(
    backend::GPUBackend,
    N::Int;
    memory_fraction::Real=_STATEVECTOR_MAX_MEMORY_FRACTION,
)
    fraction = _validate_memory_fraction(memory_fraction)
    dimension = _statevector_dimension(N)
    CT = backend.complex_type
    CT in (ComplexF32, ComplexF64) || throw(ArgumentError(
        "statevector backend requires ComplexF32 or ComplexF64, got $CT",
    ))
    predicted = _statevector_memory_bytes(dimension, CT)
    status = gpu_memory_status(backend)
    admission_total = _statevector_admission_total(
        backend,
        status,
        Sys.total_memory(),
    )
    cap = _admit_statevector_memory(
        backend,
        status,
        predicted,
        fraction,
    )
    StatevectorMemoryAdmission(
        backend.kind,
        N,
        dimension,
        CT,
        predicted,
        status.free_bytes,
        status.total_bytes,
        admission_total,
        status.reserved_bytes,
        cap,
        fraction,
    )
end

"""
    make_statevector_evaluator(
        backend::GPUBackend,
        N::Int,
        diagonal::AbstractVector{<:Real};
        memory_fraction=0.8,
    ) -> DeviceStatevectorEvaluator

Prepare an exact diagonal-cost/`sum(X)` statevector evaluator. The returned
object is callable as `evaluator(angles)` and returns
`(value, gamma_gradient, beta_gradient)`. Explicit vendor backends never fall
back to CPU.
"""
function make_statevector_evaluator(
    backend::GPUBackend,
    N::Int,
    diagonal::AbstractVector{<:Real};
    memory_fraction::Real=_STATEVECTOR_MAX_MEMORY_FRACTION,
)
    dimension = _statevector_dimension(N)
    length(diagonal) == dimension || throw(DimensionMismatch(
        "diagonal length $(length(diagonal)) does not equal 2^N = $dimension",
    ))
    admission = statevector_memory_admission(
        backend,
        N;
        memory_fraction,
    )
    all(isfinite, diagonal) || throw(ArgumentError("cost diagonal must be finite"))
    fraction = admission.memory_fraction
    CT = admission.complex_type
    RT = real(CT)
    backend.kind == :cpu || validate_gpu_backend(backend)

    to_backend = backend.kind == :cpu ? copy :
        array -> gpu_array(backend, array)
    try
        allocation_started = time_ns()
        diagonal_host = RT.(diagonal)
        initial_host = fill(CT(inv(sqrt(RT(dimension)))), dimension)
        host_allocation_seconds = (time_ns() - allocation_started) / 1e9
        transfer_started = time_ns()
        diagonal_device = to_backend(diagonal_host)
        initial_device = to_backend(initial_host)
        transfer_seconds = backend.kind == :cpu ? 0.0 :
            (time_ns() - transfer_started) / 1e9
        device_allocation_started = time_ns()
        state = similar(initial_device)
        adjoint = similar(initial_device)
        scratch = similar(initial_device)
        device_allocation_seconds =
            (time_ns() - device_allocation_started) / 1e9
        evaluator = DeviceStatevectorEvaluator(
            backend,
            N,
            diagonal_device,
            initial_device,
            state,
            adjoint,
            scratch,
            Float64[],
            fraction,
            admission.predicted_live_bytes,
            admission.reserved_bytes,
            admission.memory_cap_bytes,
            _StatevectorRouteCounter(
                0,
                admission.reserved_bytes,
                0,
                nothing,
                0.0,
                0.0,
                backend.kind == :cpu ? 0.0 : nothing,
                backend.kind == :cpu ? :synchronized_wall : (
                    _gpu_kernel_timer(backend) === nothing ?
                    :unavailable : :hip_events
                ),
                host_allocation_seconds + device_allocation_seconds +
                    (backend.kind == :cpu ?
                     (time_ns() - transfer_started) / 1e9 : 0.0),
                transfer_seconds,
                0.0,
            ),
            String(_statevector_source_revision()),
        )
        _statevector_postcheck!(evaluator)
        evaluator
    catch error
        error isa GPUBackendError && rethrow()
        throw(GPUBackendError(
            backend.kind,
            "statevector allocation failed after admission; no CPU fallback was attempted",
            error,
        ))
    end
end

function _fill_initial_state!(ev::DeviceStatevectorEvaluator)
    ka_backend = KernelAbstractions.get_backend(ev.state)
    kernel! = _sv_fill_kernel!(ka_backend)
    CT = eltype(ev.state)
    amplitude = CT(inv(sqrt(real(CT)(length(ev.state)))))
    _recorded_launch!(ev, kernel!, ev.state, amplitude; ndrange=length(ev.state))
    ev.state
end

function _diagonal_from_pauli_terms(N::Int, terms)
    dimension = _statevector_dimension(N)
    validated = PauliTerm[]
    for (index, term) in enumerate(terms)
        term isa PauliTerm || throw(ArgumentError(
            "cost term $index is not a PauliTerm",
        ))
        term.kind in (:z, :zz) || throw(ArgumentError(
            "device statevector evaluator supports only Z and ZZ costs; " *
            "term $index has kind :$(term.kind)",
        ))
        1 <= term.i || throw(ArgumentError(
            "PauliTerm qubit $(term.i) must be positive",
        ))
        term.i <= N || throw(ArgumentError(
            "PauliTerm qubit $(term.i) exceeds N = $N",
        ))
        1 <= term.j || throw(ArgumentError(
            "PauliTerm qubit $(term.j) must be positive",
        ))
        term.j <= N || throw(ArgumentError(
            "PauliTerm qubit $(term.j) exceeds N = $N",
        ))
        isfinite(term.coeff) || throw(ArgumentError(
            "PauliTerm coefficient must be finite",
        ))
        push!(validated, term)
    end
    diagonal = zeros(Float64, dimension)
    for term in validated
        mask = one(Int) << (term.i - 1)
        term.kind === :zz && (mask |= one(Int) << (term.j - 1))
        @inbounds for basis in 0:(dimension-1)
            sign = isodd(count_ones(basis & mask)) ? -1.0 : 1.0
            diagonal[basis+1] += term.coeff * sign
        end
    end
    diagonal
end

function make_statevector_evaluator(
    backend::GPUBackend,
    N::Int,
    terms::AbstractVector{PauliTerm};
    memory_fraction::Real=_STATEVECTOR_MAX_MEMORY_FRACTION,
)
    admission = statevector_memory_admission(
        backend,
        N;
        memory_fraction,
    )
    make_statevector_evaluator(
        backend,
        N,
        _diagonal_from_pauli_terms(N, terms);
        memory_fraction=admission.memory_fraction,
    )
end

function _cost_phase!(ev::DeviceStatevectorEvaluator, state, gamma::Real)
    iszero(gamma) && return state
    ka_backend = KernelAbstractions.get_backend(state)
    kernel! = _sv_cost_phase_kernel!(ka_backend)
    RT = real(eltype(state))
    _recorded_launch!(ev, kernel!, state, ev.diagonal, RT(gamma);
        ndrange=length(state))
    state
end

function _x_mixer!(ev::DeviceStatevectorEvaluator, state, beta::Real)
    iszero(beta) && return state
    ka_backend = KernelAbstractions.get_backend(state)
    kernel! = _sv_x_mixer_kernel!(ka_backend)
    CT = eltype(state)
    cosine = CT(cos(beta))
    sine = CT(complex(0, -sin(beta)))
    for qubit in 0:(ev.N-1)
        _recorded_launch!(
            ev, kernel!, state, one(Int) << qubit, cosine, sine;
            ndrange=length(state) >> 1,
        )
    end
    state
end

function _diagonal_apply!(ev::DeviceStatevectorEvaluator, out, state)
    ka_backend = KernelAbstractions.get_backend(state)
    kernel! = _sv_diagonal_apply_kernel!(ka_backend)
    _recorded_launch!(ev, kernel!, out, ev.diagonal, state;
        ndrange=length(state))
    out
end

function _x_sum!(ev::DeviceStatevectorEvaluator, out, state)
    ka_backend = KernelAbstractions.get_backend(state)
    kernel! = _sv_x_sum_kernel!(ka_backend)
    _recorded_launch!(ev, kernel!, out, state, ev.N; ndrange=length(state))
    out
end

function _device_inner(ev::DeviceStatevectorEvaluator, left, right)
    ka_backend = KernelAbstractions.get_backend(left)
    kernel! = _sv_inner_terms_kernel!(ka_backend)
    _recorded_launch!(ev, kernel!, ev.scratch, left, right;
        ndrange=length(left))
    _device_reduce(ev)
end

function _device_expectation(ev::DeviceStatevectorEvaluator)
    ka_backend = KernelAbstractions.get_backend(ev.state)
    kernel! = _sv_expectation_terms_kernel!(ka_backend)
    _recorded_launch!(ev, kernel!, ev.scratch, ev.diagonal, ev.state;
        ndrange=length(ev.state))
    Float64(real(_device_reduce(ev)))
end

function _device_reduce(ev::DeviceStatevectorEvaluator)
    input = ev.scratch
    output = ev.reduction
    input_length = length(input)
    ka_backend = KernelAbstractions.get_backend(input)
    kernel! = _sv_pair_reduce_kernel!(ka_backend)
    while input_length > 1
        output_length = (input_length + 1) >> 1
        _recorded_launch!(
            ev,
            kernel!,
            output,
            input,
            input_length;
            ndrange=output_length,
        )
        input, output = output, input
        input_length = output_length
    end
    transfer_started = time_ns()
    value = Array(input[1:1])[1]
    ev.route.transfer_seconds += (time_ns() - transfer_started) / 1e9
    value
end

function (ev::DeviceStatevectorEvaluator)(angles::QAOAAngles)
    evaluation_started = time_ns()
    p = depth(angles)
    p >= 1 || throw(ArgumentError("angle depth must be at least 1"))
    all(isfinite, angles.γ) || throw(ArgumentError("gamma angles must be finite"))
    all(isfinite, angles.β) || throw(ArgumentError("beta angles must be finite"))

    _fill_initial_state!(ev)
    for layer in 1:p
        _cost_phase!(ev, ev.state, angles.γ[layer])
        _x_mixer!(ev, ev.state, angles.β[layer])
    end

    _diagonal_apply!(ev, ev.adjoint, ev.state)
    value = _device_expectation(ev)
    resize!(ev.gradient, 2p)

    for gate in 2p:-1:1
        if isodd(gate)
            layer = (gate + 1) >> 1
            _diagonal_apply!(ev, ev.scratch, ev.adjoint)
            ev.gradient[layer] = -2 * Float64(imag(
                _device_inner(ev, ev.state, ev.scratch)))
            _cost_phase!(ev, ev.adjoint, -angles.γ[layer])
            _cost_phase!(ev, ev.state, -angles.γ[layer])
        else
            layer = gate >> 1
            _x_sum!(ev, ev.scratch, ev.adjoint)
            ev.gradient[p + layer] = -2 * Float64(imag(
                _device_inner(ev, ev.state, ev.scratch)))
            _x_mixer!(ev, ev.adjoint, -angles.β[layer])
            _x_mixer!(ev, ev.state, -angles.β[layer])
        end
    end

    _statevector_postcheck!(ev)
    elapsed = (time_ns() - evaluation_started) / 1e9
    ev.route.evaluation_count += 1
    if ev.route.first_evaluation_seconds === nothing
        ev.route.first_evaluation_seconds = elapsed
    else
        ev.route.steady_state_seconds += elapsed
    end
    value, copy(ev.gradient[1:p]), copy(ev.gradient[(p+1):(2p)])
end

"""
    synchronize(evaluator::DeviceStatevectorEvaluator)

Wait for all queued evaluator work. Evaluator calls already synchronize before
returning; this method is provided for explicit consumer-side barriers.
"""
function KernelAbstractions.synchronize(ev::DeviceStatevectorEvaluator)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(ev.state))
    ev
end

"""
    statevector_execution_stats(evaluator) -> StatevectorExecutionStats

Return immutable, source-bound route and memory telemetry. `kernel_launches`
is incremented only by actual KernelAbstractions launch sites.
"""
function statevector_execution_stats(ev::DeviceStatevectorEvaluator)
    StatevectorExecutionStats(
        ev.backend.kind,
        ev.backend.device,
        ev.backend.complex_type,
        ev.route.kernel_launches,
        ev.predicted_live_bytes,
        ev.reservation_before,
        ev.route.reservation_after,
        ev.memory_cap_bytes,
        ev.source_revision,
        ev.route.evaluation_count,
        ev.route.first_evaluation_seconds,
        ev.route.steady_state_seconds,
        ev.route.synchronized_launch_seconds,
        nothing,
        nothing,
        ev.route.kernel_execution_seconds,
        ev.route.kernel_timing_source,
        ev.route.allocation_seconds,
        ev.route.transfer_seconds,
        ev.route.verification_seconds,
    )
end
