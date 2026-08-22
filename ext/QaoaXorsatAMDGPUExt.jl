module QaoaXorsatAMDGPUExt

using AMDGPU
using QaoaXorsat

function amdgpu_array(array::AbstractArray)
    AMDGPU.ROCArray(array)
end

function amdgpu_backend()
    AMDGPU.functional() || throw(ErrorException(
        "AMDGPU.functional() returned false; HIP, LLD, and ROCm device libraries are required",
    ))
    devices = AMDGPU.devices()
    isempty(devices) && throw(ErrorException("AMDGPU found no HIP devices"))
    device = AMDGPU.device()
    device_name = String(AMDGPU.HIP.name(device))
    QaoaXorsat.GPUBackend(
        :amdgpu,
        amdgpu_array,
        ComplexF64,
        "AMDGPU/ROCm GPU (ComplexF64)",
        device_name,
    )
end

function amdgpu_memory_status()
    free_bytes, total_bytes = AMDGPU.info()
    QaoaXorsat.GPUMemoryStatus(
        free_bytes,
        total_bytes,
        AMDGPU.cached_memory(),
    )
end

function amdgpu_timed_launch(launch::Function)
    AMDGPU.@elapsed Base.invokelatest(launch)
end

function __init__()
    QaoaXorsat._register_gpu_backend!(:amdgpu, amdgpu_backend)
    QaoaXorsat._register_gpu_memory_status!(:amdgpu, amdgpu_memory_status)
    QaoaXorsat._register_gpu_kernel_timer!(:amdgpu, amdgpu_timed_launch)
end

end
