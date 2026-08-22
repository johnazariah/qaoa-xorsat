module QaoaXorsatCUDAExt

using CUDA
using QaoaXorsat

cuda_array(array::AbstractArray) = CUDA.CuArray(array)

function cuda_backend()
    CUDA.functional() || throw(ErrorException("CUDA.functional() returned false"))
    device_name = string(CUDA.name(CUDA.device()))
    QaoaXorsat.GPUBackend(
        :cuda,
        cuda_array,
        ComplexF64,
        "CUDA GPU (ComplexF64)",
        device_name,
    )
end

function cuda_memory_status()
    QaoaXorsat.GPUMemoryStatus(
        CUDA.free_memory(),
        CUDA.total_memory(),
        CUDA.cached_memory(),
    )
end

function __init__()
    QaoaXorsat._register_gpu_backend!(:cuda, cuda_backend)
    QaoaXorsat._register_gpu_memory_status!(:cuda, cuda_memory_status)
end

end
