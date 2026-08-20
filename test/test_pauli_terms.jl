using QaoaXorsat
using Test
using LinearAlgebra
using Random
using Serialization

const ptI = Matrix{ComplexF64}(I, 2, 2)
const ptX = ComplexF64[0 1; 1 0]
const ptY = ComplexF64[0 -im; im 0]
const ptZ = ComplexF64[1 0; 0 -1]
const ptaxis = Dict(:x => ptX, :y => ptY, :z => ptZ)

function pton(axis, q, N)
    factors = [q == site ? ptaxis[axis] : ptI for site in N:-1:1]
    reduce(kron, factors)
end

function ptdense(t::PauliTerm, N)
    axes = Tuple(Symbol(c) for c in String(t.kind))
    operator = t.coeff * pton(axes[1], t.i, N)
    length(axes) == 1 ? operator : operator * pton(axes[2], t.j, N)
end

function ptaction(terms, ψ, N)
    out = similar(ψ)
    apply_terms!(out, terms, ψ, N)
end

@testset "ordered Pauli terms" begin
    constructors = (
        x=x_term, y=y_term, z=z_term,
        xx=xx_term, xy=xy_term, xz=xz_term,
        yx=yx_term, yy=yy_term, yz=yz_term,
        zx=zx_term, zy=zy_term, zz=zz_term,
    )

    @testset "dense-matrix and Kronecker-product action oracle" begin
        rng = MersenneTwister(41)
        N = 3
        ψ = randn(rng, ComplexF64, 1 << N)
        for kind in (:x, :y, :z)
            term = constructors[kind](2, -0.7)
            @test ptaction([term], ψ, N) ≈ ptdense(term, N) * ψ atol=1e-12
        end
        for kind in (:xx, :xy, :xz, :yx, :yy, :yz, :zx, :zy, :zz)
            term = constructors[kind](1, 3, -0.7)
            @test ptaction([term], ψ, N) ≈ ptdense(term, N) * ψ atol=1e-12
        end
    end

    @testset "mixed-axis endpoint reversal preserves axis association" begin
        rng = MersenneTwister(44)
        ψ = randn(rng, ComplexF64, 8)
        for (forward, reverse, first_axis, second_axis) in (
            (xy_term, yx_term, :x, :y), (xz_term, zx_term, :x, :z),
            (yz_term, zy_term, :y, :z), (yx_term, xy_term, :y, :x),
            (zx_term, xz_term, :z, :x), (zy_term, yz_term, :z, :y),
        )
            direct = forward(1, 3, 0.4)
            reversed = reverse(3, 1, 0.4)
            @test direct == reversed
            @test ptdense(direct, 3) == ptdense(reversed, 3)

            reversed_input = forward(3, 1, 0.4)
            expected = 0.4 * pton(first_axis, 3, 3) * pton(second_axis, 1, 3) * ψ
            @test ptaction([reversed_input], ψ, 3) ≈ expected atol=1e-12
        end
        @test PauliTerm(:xy, 3, 1, 0.4) == PauliTerm(:yx, 1, 3, 0.4)
        @test PauliTerm(:xz, 3, 1, 0.4) == PauliTerm(:zx, 1, 3, 0.4)
        @test PauliTerm(:yz, 3, 1, 0.4) == PauliTerm(:zy, 1, 3, 0.4)
    end

    @testset "linearity, coefficients, duplicates, and Hermiticity" begin
        rng = MersenneTwister(42)
        N = 3
        ψ = randn(rng, ComplexF64, 1 << N)
        φ = randn(rng, ComplexF64, 1 << N)
        kinds = (:x, :y, :z, :xx, :xy, :xz, :yx, :yy, :yz, :zx, :zy, :zz)
        terms = PauliTerm[
            length(String(kind)) == 1 ?
            constructors[kind](2, (-1.0)^n * (n / 7)) :
            constructors[kind](1, 3, (-1.0)^n * (n / 7))
            for (n, kind) in enumerate(kinds)
        ]

        total = ptaction(terms, ψ, N)
        pieces = sum((ptaction([term], ψ, N) for term in terms);
            init=zeros(ComplexF64, 1 << N))
        @test total ≈ pieces atol=1e-12

        base = xy_term(1, 3)
        @test ptaction([xy_term(1, 3, -2.5)], ψ, N) ≈
              -2.5 .* ptaction([base], ψ, N) atol=1e-12
        @test ptaction([base, base], ψ, N) ≈
              2 .* ptaction([base], ψ, N) atol=1e-12

        H = sum((ptdense(term, N) for term in terms);
            init=zeros(ComplexF64, 1 << N, 1 << N))
        @test H ≈ H' atol=1e-12
        Hψ = ptaction(terms, ψ, N)
        Hφ = ptaction(terms, φ, N)
        @test dot(φ, Hψ) ≈ conj(dot(ψ, Hφ)) atol=1e-12
        @test hamiltonian_expectation(terms, ψ, N) ≈ real(dot(ψ, Hψ)) atol=1e-12
    end

    @testset "evolution adjoint consistency" begin
        rng = MersenneTwister(43)
        N = 3
        terms = [y_term(2, 0.3), xy_term(1, 3, -0.8), yz_term(2, 3, 0.4)]
        ψ = randn(rng, ComplexF64, 1 << N)
        ψ ./= norm(ψ)
        evolved = evolve_rk4!(copy(ψ), terms, 0.35, N; steps=800)
        restored = evolve_rk4!(copy(evolved), terms, -0.35, N; steps=800)
        @test norm(evolved) ≈ 1.0 atol=1e-12
        @test restored ≈ ψ atol=1e-11
    end

    @testset "single-pass iterables and pure-X fast path" begin
        rng = MersenneTwister(45)
        N = 3
        ψ = randn(rng, ComplexF64, 1 << N)

        terms = [x_term(1, 0.2), xy_term(3, 1, -0.7), yz_term(2, 3, 0.4)]
        expected = ptaction(terms, ψ, N)
        single_pass = Iterators.Stateful((term for term in terms))
        actual = similar(ψ)
        apply_terms!(actual, single_pass, ψ, N)
        @test actual == expected
        @test isempty(single_pass)

        invalid = Iterators.Stateful((term for term in (x_term(1), x_term(4))))
        fill!(actual, 3)
        @test_throws ArgumentError apply_terms!(actual, invalid, ψ, N)
        @test all(==(3), actual)
        @test isempty(invalid)

        x_terms = x_mixer_terms(N)
        expected_x = zeros(ComplexF64, 1 << N)
        for term in x_terms
            mask = one(Int) << (term.i - 1)
            coefficient = ComplexF64(term.coeff)
            @inbounds for b in 0:(1 << N)-1
                expected_x[b+1] += coefficient * ψ[(b ⊻ mask)+1]
            end
        end
        @test ptaction(x_terms, ψ, N) == expected_x
    end

    @testset "normalisation, serialization, and validation" begin
        term = xy_term(3, 1, -0.25)
        @test term == yx_term(1, 3, -0.25)
        io = IOBuffer()
        serialize(io, term)
        seekstart(io)
        @test deserialize(io) == term

        @test_throws ArgumentError PauliTerm(:bad, 1, 2, 1.0)
        @test_throws ArgumentError xy_term(1, 1)
        @test_throws ArgumentError y_term(0)
        @test_throws ArgumentError z_term(1, Inf)
        @test_throws MethodError y_term(1, 1 + im)

        ψ = zeros(ComplexF64, 8)
        out = similar(ψ)
        @test_throws ArgumentError apply_terms!(out, [x_term(4)], ψ, 3)
        fill!(out, 2)
        @test_throws ArgumentError apply_terms!(out, [x_term(1), x_term(4)], ψ, 3)
        @test all(==(2), out)
        @test_throws DimensionMismatch apply_terms!(zeros(ComplexF64, 4), [y_term(1)], ψ, 3)
        @test_throws DimensionMismatch apply_terms!(out, [y_term(1)], ψ, 2)
        @test_throws ArgumentError apply_terms!(ψ, [y_term(1)], ψ, 3)
    end
end
