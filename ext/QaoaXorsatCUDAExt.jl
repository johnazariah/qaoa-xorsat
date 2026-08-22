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

function __init__()
    QaoaXorsat._register_gpu_backend!(:cuda, cuda_backend)
end

end
