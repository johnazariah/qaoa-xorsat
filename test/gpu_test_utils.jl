"""
Shared GPU test utilities — auto-detects CUDA or Metal backend.

Include this before GPU test code:

    include(joinpath(@__DIR__, "gpu_test_utils.jl"))

Provides:
    GPU_OK::Bool          — true if any GPU backend is functional
    GPU_LABEL::String     — "CUDA" or "Metal" or "none"
    GPU_CT::Type          — ComplexF64 (CUDA) or ComplexF32 (Metal)
    GPU_RT::Type          — Float64 (CUDA) or Float32 (Metal)
    GPU_ARRAY_TYPE        — CuArray or MtlArray or nothing
    gpu_array(x)          — convert CPU array to GPU array
    to_gpu(x)             — alias for gpu_array
    from_gpu(x)           — convert GPU array back to CPU
"""

if !@isdefined(_GPU_TEST_UTILS_LOADED)

const _GPU_TEST_UTILS_LOADED = true

# ── Detect backend (let block avoids soft-scope issues) ───────────

const _GPU_DETECT = let
    local ok = false
    local label = "none"
    local ct = ComplexF64
    local rt = Float64
    local arr_type = nothing
    local arr_fn = nothing

    # Try CUDA first (NVIDIA — higher precision, ComplexF64)
    if !ok
        try
            @eval Main using CUDA
            if Base.invokelatest(() -> Main.CUDA.functional())
                ok = true
                label = "CUDA"
                ct = ComplexF64
                rt = Float64
                arr_type = Main.CUDA.CuArray
                arr_fn = (x) -> Base.invokelatest(Main.CUDA.CuArray, x)
            end
        catch; end
    end

    # Try Metal (Apple Silicon — ComplexF32 only)
    if !ok
        try
            @eval Main using Metal
            if Base.invokelatest(() -> Main.Metal.functional())
                ok = true
                label = "Metal"
                ct = ComplexF32
                rt = Float32
                arr_type = Main.Metal.MtlArray
                arr_fn = (x) -> Base.invokelatest(Main.Metal.MtlArray, x)
            end
        catch; end
    end

    (ok=ok, label=label, ct=ct, rt=rt, arr_type=arr_type, arr_fn=arr_fn)
end

const GPU_OK = _GPU_DETECT.ok
const GPU_LABEL = _GPU_DETECT.label
const GPU_CT = _GPU_DETECT.ct
const GPU_RT = _GPU_DETECT.rt
const GPU_ARRAY_TYPE = _GPU_DETECT.arr_type

if GPU_OK
    @info "GPU tests using $GPU_LABEL backend (element type: $GPU_CT)"
else
    @warn "No GPU backend available — GPU tests will be skipped"
end

# ── GPU array factory ─────────────────────────────────────────────

if GPU_OK
    const _gpu_arr_fn = _GPU_DETECT.arr_fn
    gpu_array(x::AbstractVector{<:Complex}) = _gpu_arr_fn(GPU_CT.(x))
    gpu_array(x::AbstractVector{<:Real})    = _gpu_arr_fn(GPU_CT.(complex.(x)))
else
    gpu_array(x) = error("No GPU backend — cannot create GPU arrays")
end

to_gpu(x::AbstractVector) = gpu_array(x)
from_gpu(x) = Array(x)

end # @isdefined guard
