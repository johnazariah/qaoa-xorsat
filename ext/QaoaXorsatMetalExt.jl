module QaoaXorsatMetalExt

using Metal
using QaoaXorsat

metal_array(array::AbstractArray{<:Complex}) =
    Metal.MtlArray(ComplexF32.(array))
metal_array(array::AbstractArray{<:Real}) =
    Metal.MtlArray(ComplexF32.(complex.(array)))

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

function __init__()
    QaoaXorsat._register_gpu_backend!(:metal, metal_backend)
end

end
