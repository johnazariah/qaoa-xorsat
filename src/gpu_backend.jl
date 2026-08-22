"""
Optional GPU backend selection for the QAOA evaluation pipeline.

CUDA, AMDGPU, and Metal are weak dependencies loaded through package
extensions. Explicit backend selection throws `GPUBackendError` when the
runtime, device, or validation kernel is unavailable. Automatic selection
tries CUDA, AMDGPU, and Metal in that order before returning the CPU backend.
"""

using KernelAbstractions
using UUIDs

struct GPUBackend
    kind::Symbol
    gpu_array_fn::Union{Function,Nothing}
    complex_type::Type
    label::String
    device::Union{String,Nothing}
end

GPUBackend(kind::Symbol, gpu_array_fn::Union{Function,Nothing},
    complex_type::Type, label::String) =
    GPUBackend(kind, gpu_array_fn, complex_type, label, nothing)

struct GPUBackendError <: Exception
    kind::Symbol
    message::String
    cause::Any
end

GPUBackendError(kind::Symbol, message::String) =
    GPUBackendError(kind, message, nothing)

function Base.showerror(io::IO, error::GPUBackendError)
    print(io, "GPU backend ", repr(error.kind), ": ", error.message)
    if error.cause !== nothing
        print(io, "\nCaused by: ")
        showerror(io, error.cause)
    end
end

const _CPU_BACKEND = GPUBackend(
    :cpu,
    nothing,
    ComplexF64,
    "CPU checkpointed path",
    nothing,
)

const _GPU_PROVIDER_IDS = Dict(
    :cuda => Base.PkgId(UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"),
    :amdgpu => Base.PkgId(UUID("21141c5a-9bdb-4563-92ae-f87d6854732e"), "AMDGPU"),
    :metal => Base.PkgId(UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"),
)
const _GPU_PROVIDER_FACTORIES = Dict{Symbol,Function}()
const _GPU_PROVIDER_PRIORITY = (:cuda, :amdgpu, :metal)

function _canonical_gpu_kind(kind::Symbol)
    kind in (:amd, :rocm, :hip) && return :amdgpu
    kind in (:auto, :cpu, :cuda, :amdgpu, :metal) && return kind
    throw(ArgumentError(
        "unknown GPU backend $(repr(kind)); expected :auto, :cpu, :cuda, :amdgpu, or :metal",
    ))
end

function _register_gpu_backend!(kind::Symbol, factory::Function)
    canonical = _canonical_gpu_kind(kind)
    canonical in _GPU_PROVIDER_PRIORITY || throw(ArgumentError(
        "cannot register non-GPU backend $(repr(kind))",
    ))
    _GPU_PROVIDER_FACTORIES[canonical] = factory
    nothing
end

function _load_gpu_provider!(kind::Symbol)
    haskey(_GPU_PROVIDER_FACTORIES, kind) && return
    package = _GPU_PROVIDER_IDS[kind]
    try
        Base.require(package)
    catch error
        throw(GPUBackendError(
            kind,
            "$(package.name).jl is not available in the active environment; " *
            "install it with `using Pkg; Pkg.add(\"$(package.name)\")`",
            error,
        ))
    end
    haskey(_GPU_PROVIDER_FACTORIES, kind) || throw(GPUBackendError(
        kind,
        "$(package.name).jl loaded, but the QaoaXorsat package extension did not activate",
    ))
end

function _create_gpu_backend(kind::Symbol)
    _load_gpu_provider!(kind)
    try
        backend = Base.invokelatest(_GPU_PROVIDER_FACTORIES[kind])
        backend isa GPUBackend || throw(TypeError(
            :_create_gpu_backend, GPUBackend, backend,
        ))
        backend.kind == kind || throw(ArgumentError(
            "provider returned backend $(repr(backend.kind)) for $(repr(kind))",
        ))
        backend
    catch error
        error isa GPUBackendError && rethrow()
        throw(GPUBackendError(kind, "runtime or device initialization failed", error))
    end
end

@kernel function _gpu_validation_kernel!(output, @Const(input))
    index = @index(Global)
    @inbounds output[index] = input[index] + input[index]
end

"""
    validate_gpu_backend(backend::GPUBackend) -> Bool

Validate a backend by allocating an array, launching a KernelAbstractions
kernel, synchronizing the device, and checking the host result. Returns `true`
or throws `GPUBackendError`; it never silently substitutes another backend.
"""
function validate_gpu_backend(backend::GPUBackend)
    Base.invokelatest(_validate_gpu_backend_latest, backend)
end

function _validate_gpu_backend_latest(backend::GPUBackend)
    backend.kind == :cpu && return true
    backend.gpu_array_fn === nothing && throw(GPUBackendError(
        backend.kind, "backend has no device-array constructor",
    ))

    try
        input_cpu = backend.complex_type[complex(1, 2), complex(-3, 0.5)]
        input = Base.invokelatest(backend.gpu_array_fn, input_cpu)
        output = similar(input)
        ka_backend = KernelAbstractions.get_backend(input)
        ka_backend isa KernelAbstractions.CPU && throw(ErrorException(
            "device-array constructor returned a CPU array",
        ))
        kernel! = _gpu_validation_kernel!(ka_backend)
        kernel!(output, input; ndrange=length(input))
        KernelAbstractions.synchronize(ka_backend)
        Array(output) == input_cpu .+ input_cpu || throw(ErrorException(
            "validation kernel returned an incorrect result",
        ))
        true
    catch error
        error isa GPUBackendError && rethrow()
        throw(GPUBackendError(
            backend.kind,
            "device validation failed for $(backend.label)",
            error,
        ))
    end
end

"""
    gpu_backend(kind::Symbol=:auto; validate::Bool=true) -> GPUBackend

Select a CPU, CUDA, AMDGPU/ROCm/HIP, or Metal backend. `:amd`, `:rocm`, and
`:hip` are aliases for `:amdgpu`. An explicit vendor request throws when that
backend is unavailable. `:auto` tries CUDA, AMDGPU, and Metal, then returns CPU.
"""
function gpu_backend(kind::Symbol=:auto; validate::Bool=true)
    canonical = _canonical_gpu_kind(kind)
    canonical == :cpu && return _CPU_BACKEND

    if canonical == :auto
        for candidate in _GPU_PROVIDER_PRIORITY
            try
                backend = _create_gpu_backend(candidate)
                validate && Base.invokelatest(validate_gpu_backend, backend)
                return backend
            catch error
                error isa GPUBackendError || rethrow()
            end
        end
        return _CPU_BACKEND
    end

    backend = _create_gpu_backend(canonical)
    validate && Base.invokelatest(validate_gpu_backend, backend)
    backend
end

"""
    gpu_backend_available(kind::Symbol; validate::Bool=true) -> Bool

Return whether an explicit backend can be constructed and, by default,
validated with a real device kernel. This query never throws for a known kind.
"""
function gpu_backend_available(kind::Symbol; validate::Bool=true)
    canonical = _canonical_gpu_kind(kind)
    canonical == :cpu && return true
    canonical == :auto && return gpu_backend(:auto; validate).kind != :cpu
    try
        gpu_backend(canonical; validate)
        true
    catch error
        error isa GPUBackendError || rethrow()
        false
    end
end

detect_gpu_backend(kind::Symbol=:auto; validate::Bool=true) =
    gpu_backend(kind; validate)

"""
    gpu_array(backend::GPUBackend, array)

Copy an array to the selected GPU using the precision required by the backend.
"""
function gpu_array(backend::GPUBackend, array::AbstractArray)
    backend.gpu_array_fn === nothing && throw(GPUBackendError(
        backend.kind, "the CPU backend does not create GPU arrays",
    ))
    Base.invokelatest(backend.gpu_array_fn, array)
end

"""
    make_gpu_evaluator(backend::GPUBackend=gpu_backend();
                       checkpoint_interval::Int=0) -> Function or nothing

Build a `gpu_evaluator(params, angles; clause_sign)` closure for
`optimize_angles`, `optimize_depth_sequence`, or `swarm_optimize`. Returns
`nothing` for the CPU backend.
"""
function make_gpu_evaluator(
    backend::GPUBackend=gpu_backend();
    checkpoint_interval::Int=0,
)
    Base.invokelatest(
        _make_gpu_evaluator_latest,
        backend;
        checkpoint_interval,
    )
end

function _make_gpu_evaluator_latest(
    backend::GPUBackend;
    checkpoint_interval::Int=0,
)
    backend.kind == :cpu && return nothing
    checkpoint_interval >= 0 || throw(ArgumentError(
        "checkpoint_interval must be nonnegative, got $checkpoint_interval",
    ))
    validate_gpu_backend(backend)

    if !@isdefined(gpu_checkpointed_forward_backward)
        include(joinpath(@__DIR__, "gpu_checkpointed.jl"))
    end

    array_fn = backend.gpu_array_fn
    function gpu_evaluator(params, angles; clause_sign)
        Base.invokelatest(
            gpu_checkpointed_forward_backward,
            params,
            angles,
            array_fn;
            clause_sign,
            checkpoint_interval,
        )
    end
    gpu_evaluator
end
