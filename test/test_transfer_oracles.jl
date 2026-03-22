using QaoaXorsat
using Test

@testset "Transfer oracles" begin
    @testset "k=2 reduction" begin
        γ = 0.37
        slice = 1
        p = 1
        child = ComplexF64[1+0im, 2-im, -0.5+0.25im, 0.75-0.5im]

        contracted = QaoaXorsat.contract_constraint_message([child], γ, slice, p)
        expected = reshape(problem_tensor(2, γ, slice, p), 4, 4) * child

        @test contracted ≈ expected atol = 1e-12
    end

    @testset "multilinearity for k=3" begin
        γ = 0.41
        slice = 1
        p = 1
        first_a = ComplexF64[1+0im, 0.5-0.25im, -1+0.75im, 0.2+0.1im]
        first_b = ComplexF64[-0.3+0.8im, 1.2+0im, 0.4-0.5im, -0.6+0.2im]
        second = ComplexF64[0.7-0.1im, -1.1+0.4im, 0.25+0.5im, 0.9-0.3im]
        α = 1.3 - 0.2im
        β = -0.4 + 0.7im

        combined = α .* first_a .+ β .* first_b

        contracted_combined =
            QaoaXorsat.contract_constraint_message([combined, second], γ, slice, p)
        contracted_split = α .* QaoaXorsat.contract_constraint_message([first_a, second], γ, slice, p) .+
                           β .* QaoaXorsat.contract_constraint_message([first_b, second], γ, slice, p)

        @test contracted_combined ≈ contracted_split atol = 1e-12
    end

    @testset "zero-angle factorization" begin
        slice = 1
        p = 1
        first = ComplexF64[1+0im, 2-0.5im, -0.5+0.2im, 0.25-0.1im]
        second = ComplexF64[0.5+0.3im, -1+0im, 0.75-0.4im, 1.5+0.2im]

        contracted = QaoaXorsat.contract_constraint_message([first, second], 0.0, slice, p)
        expected_entry = sum(first) * sum(second)

        @test contracted ≈ fill(expected_entry, length(contracted)) atol = 1e-12
    end
end
