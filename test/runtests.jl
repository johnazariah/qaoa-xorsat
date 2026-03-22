using QaoaXorsat
using Test

@testset "QaoaXorsat" begin
    include("test_tree.jl")
    include("test_tensors.jl")
    include("test_maxcut_transfer.jl")
    include("test_qaoa.jl")
end
