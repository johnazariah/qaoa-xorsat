"""
GPU backend auto-detection for the QAOA evaluation pipeline.

Picks the first available backend in order:

    1. CUDA  (NVIDIA, ComplexF64) — Stephen's SLURM cluster
    2. Metal (Apple Silicon, ComplexF32) — Mac Studio
    3. nothing (CPU fallback)

Loads the chosen backend lazily so the package can be used on machines
that have neither GPU SDK installed.

Usage:

    include("src/gpu_backend.jl")
    if GPU_BACKEND.kind != :cpu
        eval = make_gpu_evaluator(GPU_BACKEND)
        # pass `gpu_evaluator=eval` into swarm_optimize / optimize_angles
    end
"""

struct GPUBackend
    kind::Symbol                                # :cuda | :metal | :cpu
    gpu_array_fn::Union{Function,Nothing}       # converts CPU array -> GPU array
    complex_type::Type                          # ComplexF64 (CUDA) or ComplexF32 (Metal)
    label::String                               # human-readable status string
end

const _CPU_BACKEND = GPUBackend(:cpu, nothing, ComplexF64, "off (CPU checkpointed path)")

function detect_gpu_backend()
    # ── CUDA (NVIDIA) ────────────────────────────────────────────────
    try
        @eval Main using CUDA
        if Base.invokelatest(getfield(Main.CUDA, :functional))
            cu_array = getfield(Main.CUDA, :CuArray)
            fn(x) = Base.invokelatest(cu_array, x)
            return GPUBackend(:cuda, fn, ComplexF64, "CUDA GPU (ComplexF64)")
        end
    catch
        # CUDA not installed or not functional — fall through
    end

    # ── Metal (Apple Silicon) ────────────────────────────────────────
    try
        @eval Main using Metal
        if Base.invokelatest(getfield(Main.Metal, :functional))
            mtl_array = getfield(Main.Metal, :MtlArray)
            # Metal requires Float32; we narrow on the way to the GPU.
            fn(x::AbstractArray{<:Complex}) = Base.invokelatest(mtl_array, ComplexF32.(x))
            fn(x::AbstractArray{<:Real})    = Base.invokelatest(mtl_array, ComplexF32.(complex.(x)))
            return GPUBackend(:metal, fn, ComplexF32, "Metal GPU (ComplexF32)")
        end
    catch
        # Metal not installed or not functional — fall through
    end

    return _CPU_BACKEND
end

const GPU_BACKEND = detect_gpu_backend()

"""
    make_gpu_evaluator(backend::GPUBackend; checkpoint_interval=0) -> Function or nothing

Build a `gpu_evaluator(params, angles; clause_sign)` closure suitable for
passing into `swarm_optimize` / `optimize_angles`. Returns `nothing` if no
GPU backend is available, in which case callers should fall back to the
CPU checkpointed path.
"""
function make_gpu_evaluator(backend::GPUBackend=GPU_BACKEND; checkpoint_interval::Int=0)
    backend.kind == :cpu && return nothing

    # Lazy include: only loaded once, only when a GPU is present
    if !@isdefined(gpu_checkpointed_forward_backward)
        include(joinpath(@__DIR__, "gpu_checkpointed.jl"))
    end

    fn = backend.gpu_array_fn
    function gpu_eval(params, angles; clause_sign)
        gpu_checkpointed_forward_backward(params, angles, fn;
            clause_sign, checkpoint_interval)
    end
    return gpu_eval
end
