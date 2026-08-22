module QaoaXorsatMetalExt

using Metal
using QaoaXorsat

metal_array(array::AbstractArray{<:Complex}) =
    Metal.MtlArray(ComplexF32.(array))
metal_array(array::AbstractArray{<:Real}) =
    Metal.MtlArray(Float32.(array))

function metal_backend()
    Metal.functional() || throw(ErrorException("Metal.functional() returned false"))
    QaoaXorsat.GPUBackend(
        :metal,
        metal_array,
        ComplexF32,
        "Metal GPU (ComplexF32)",
        nothing,
    )
end

function metal_memory_status()
    device = Metal.device()
    total_bytes = Metal.total_memory(device)
    free_bytes = Metal.free_memory(device)
    QaoaXorsat.GPUMemoryStatus(
        free_bytes,
        total_bytes,
        total_bytes - free_bytes,
    )
end

function __init__()
    QaoaXorsat._register_gpu_backend!(:metal, metal_backend)
    QaoaXorsat._register_gpu_memory_status!(:metal, metal_memory_status)
end

end
