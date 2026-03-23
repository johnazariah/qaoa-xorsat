using QaoaXorsat
using Test

@testset "QaoaXorsat" begin
    include("test_tree.jl")
    include("test_tensors.jl")
    include("test_basso_finite_d.jl")
    include("test_transfer_oracles.jl")
    include("test_maxcut_transfer.jl")
    include("test_qaoa.jl")
    include("test_optimization.jl")
    include("test_wht_factorisation.jl")
end
