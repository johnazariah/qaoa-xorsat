using QaoaXorsat
using Test
using LinearAlgebra
using Random

function dense_maxcut_reference(N, edges, weights, angles)
    dimension = 1 << N
    costs = zeros(Float64, dimension)
    for basis in 0:(dimension-1)
        for (edge, weight) in zip(edges, weights)
            u, v = edge
            zu = iszero((basis >> (u - 1)) & 1) ? 1.0 : -1.0
            zv = iszero((basis >> (v - 1)) & 1) ? 1.0 : -1.0
            costs[basis+1] += weight * (1 - zu * zv) / 2
        end
    end

    cost_matrix = Matrix(Diagonal(ComplexF64.(costs)))
    mixer_matrix = zeros(ComplexF64, dimension, dimension)
    for basis in 0:(dimension-1), qubit in 1:N
        flipped = basis ⊻ (1 << (qubit - 1))
        mixer_matrix[flipped+1, basis+1] += 1
    end

    state = fill(ComplexF64(inv(sqrt(dimension))), dimension)
    for layer in 1:depth(angles)
        state = exp(-im * angles.β[layer] * mixer_matrix) *
                exp(-im * angles.γ[layer] * cost_matrix) * state
    end

    expectation = real(dot(state, cost_matrix * state))
    maximum_value = maximum(costs)
    success_probability = sum(
        abs2(state[index]) for index in eachindex(state) if costs[index] == maximum_value)
    (; state, expectation, maximum_value, success_probability)
end

function finite_difference_maxcut(ev, angles; step=1e-6)
    p = depth(angles)
    gradient = zeros(2p)
    for index in 1:p
        gamma_plus = copy(angles.γ)
        gamma_minus = copy(angles.γ)
        gamma_plus[index] += step
        gamma_minus[index] -= step
        gradient[index] = (
            maxcut_expectation(ev, QAOAAngles(gamma_plus, angles.β)) -
            maxcut_expectation(ev, QAOAAngles(gamma_minus, angles.β))
        ) / (2step)
    end
    for index in 1:p
        beta_plus = copy(angles.β)
        beta_minus = copy(angles.β)
        beta_plus[index] += step
        beta_minus[index] -= step
        gradient[p+index] = (
            maxcut_expectation(ev, QAOAAngles(angles.γ, beta_plus)) -
            maxcut_expectation(ev, QAOAAngles(angles.γ, beta_minus))
        ) / (2step)
    end
    gradient
end

@testset "finite-N MaxCut statevector" begin
    @testset "dense weighted and unweighted references" begin
        rng = MersenneTwister(20260723)
        edges = [(1, 2), (1, 3), (2, 4), (3, 4), (1, 4)]

        for p in 1:4, weighted in (false, true)
            weights = weighted ? 2 .* rand(rng, length(edges)) : ones(length(edges))
            angles = QAOAAngles(
                2π .* rand(rng, p) .- π,
                π .* rand(rng, p) .- π / 2,
            )
            ev = prepare_maxcut_eval(4, edges; weights)
            reference = dense_maxcut_reference(4, edges, weights, angles)
            state = maxcut_state(ev, angles)

            @test norm(state - reference.state) ≤ 1e-10
            @test maxcut_expectation(ev, angles) ≈ reference.expectation atol=1e-10
            @test maxcut_value_max(ev) ≈ reference.maximum_value atol=1e-12
            @test maxcut_success_probability(ev, angles) ≈
                  reference.success_probability atol=1e-10
        end
    end

    @testset "analytic reversible adjoint" begin
        rng = MersenneTwister(8675309)
        graph_cases = [
            ([(1, 2), (2, 3), (3, 4), (1, 4)], ones(4)),
            ([(1, 2), (1, 3), (2, 4), (3, 4), (1, 4)], rand(rng, 5)),
        ]

        for (edges, weights) in graph_cases, p in 1:4
            ev = prepare_maxcut_eval(4, edges; weights)
            angles = QAOAAngles(randn(rng, p), randn(rng, p))
            expected_value = maxcut_expectation(ev, angles)
            value, gradient_buffer = maxcut_expectation_and_gradient(ev, angles)
            gradient = copy(gradient_buffer)
            finite_difference = finite_difference_maxcut(ev, angles)

            @test value ≈ expected_value atol=1e-12
            @test maximum(abs.(gradient - finite_difference)) ≤ 1e-8

            caller_gradient = similar(gradient)
            inplace_value = maxcut_expectation_and_gradient!(
                caller_gradient, ev, angles)
            @test inplace_value ≈ value atol=1e-12
            @test caller_gradient ≈ gradient atol=1e-12
        end
    end

    @testset "norm preservation" begin
        rng = MersenneTwister(314159)
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6), (2, 5)]
        ev = prepare_maxcut_eval(6, edges; weights=rand(rng, length(edges)))
        retained_state = maxcut_state(
            ev, QAOAAngles([0.2, -0.1], [0.3, 0.4]))
        retained_copy = copy(retained_state)
        maxcut_expectation(ev, QAOAAngles([-0.7], [0.8]))
        @test retained_state == retained_copy
        @test retained_state !== ev.state

        for p in 1:4
            angles = QAOAAngles(randn(rng, p), randn(rng, p))
            @test abs(norm(maxcut_state(ev, angles)) - 1) ≤ 1e-12
        end
    end

    @testset "p=1 triangle-free 3-regular sanity" begin
        edges = [(u, v) for u in 1:3 for v in 4:6]
        ev = prepare_maxcut_eval(6, edges)
        angles = QAOAAngles([atan(inv(sqrt(2.0)))], [π / 8])
        expected_fraction = 1 / 2 + 1 / (3sqrt(3.0))

        @test maxcut_expectation(ev, angles) / length(edges) ≈
              expected_fraction atol=1e-12
        @test maxcut_value_max(ev) == 9.0
    end

    @testset "validation and empty graph" begin
        @test_throws ArgumentError prepare_maxcut_eval(0, Tuple{Int,Int}[])
        @test_throws ArgumentError prepare_maxcut_eval(true, Tuple{Int,Int}[])
        @test_throws ArgumentError prepare_maxcut_eval(3, [(0, 2)])
        @test_throws ArgumentError prepare_maxcut_eval(3, [(1, 4)])
        @test_throws ArgumentError prepare_maxcut_eval(3, [(1, 1)])
        @test_throws ArgumentError prepare_maxcut_eval(3, [(1, 2), (2, 1)])
        @test_throws ArgumentError prepare_maxcut_eval(3, [[1]])
        @test_throws ArgumentError prepare_maxcut_eval(3, [(1.0, 2.0)])
        @test_throws ArgumentError prepare_maxcut_eval(3, [(1, 2)]; weights=[1.0, 2.0])
        @test_throws ArgumentError prepare_maxcut_eval(3, [(1, 2)]; weights=[-1.0])
        @test_throws ArgumentError prepare_maxcut_eval(3, [(1, 2)]; weights=[Inf])
        @test_throws ArgumentError prepare_maxcut_eval(3, [(1, 2)]; weights=["one"])

        ev = prepare_maxcut_eval(3, Tuple{Int,Int}[])
        angles = QAOAAngles([0.2], [0.3])
        @test maxcut_value_max(ev) == 0.0
        @test maxcut_expectation(ev, angles) == 0.0
        @test maxcut_success_probability(ev, angles) ≈ 1.0 atol=1e-12
        @test_throws ArgumentError maxcut_expectation(
            ev, QAOAAngles([NaN], [0.1]))
        @test_throws ArgumentError maxcut_expectation_and_gradient!(
            zeros(1), ev, angles)
    end

    @testset "repeated optimization calls reuse buffers" begin
        ev = prepare_maxcut_eval(5, [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)])
        angles = QAOAAngles([0.2, -0.3], [0.4, 0.1])
        gradient = zeros(4)

        maxcut_expectation(ev, angles)
        maxcut_expectation_and_gradient!(gradient, ev, angles)
        maxcut_expectation_and_gradient(ev, angles)

        @test @allocated(maxcut_expectation(ev, angles)) ≤ 1024
        @test @allocated(maxcut_expectation_and_gradient!(gradient, ev, angles)) ≤ 1024
        @test @allocated(maxcut_expectation_and_gradient(ev, angles)) ≤ 1024
    end
end
