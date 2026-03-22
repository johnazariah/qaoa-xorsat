using QaoaXorsat
using Test

function build_gamma_full_vector(angles::QAOAAngles)
    p = depth(angles)
    gamma_vector = zeros(Float64, QaoaXorsat.basso_bit_count(p))
    phase_positions = QaoaXorsat.basso_phase_bit_positions(p)
    mirrored = QaoaXorsat.build_gamma_vector(angles)

    for (gamma_index, bit_index) in pairs(phase_positions)
        gamma_vector[bit_index] = mirrored[gamma_index]
    end

    gamma_vector
end

function configuration_spins(configuration::Integer, bit_count::Int)
    [QaoaXorsat.z_eigenvalue((Int(configuration) >> (index - 1)) & 1) for index in 1:bit_count]
end

function build_branch_message(angles::QAOAAngles)
    configuration_count = QaoaXorsat.basso_configuration_count(depth(angles))
    ComplexF64[QaoaXorsat.f_function(angles, configuration) for configuration in 0:configuration_count-1]
end

function build_constraint_kernel(angles::QAOAAngles, branch_degree::Int)
    p = depth(angles)
    bit_count = QaoaXorsat.basso_bit_count(p)
    configuration_count = QaoaXorsat.basso_configuration_count(p)
    gamma_vector = build_gamma_full_vector(angles)
    scale = inv(sqrt(float(branch_degree)))

    ComplexF64[
        cos(scale * sum(gamma_vector .* configuration_spins(configuration, bit_count)))
        for configuration in 0:configuration_count-1
    ]
end

function naive_constraint_fold(angles::QAOAAngles, branch_degree::Int; child_arity::Int=2)
    configuration_count = QaoaXorsat.basso_configuration_count(depth(angles))
    branch_message = build_branch_message(angles)
    kernel = build_constraint_kernel(angles, branch_degree)
    folded = zeros(ComplexF64, configuration_count)

    function recurse(target::Int, accumulated_xor::Int, weight::ComplexF64, remaining::Int)
        if iszero(remaining)
            folded[target + 1] += kernel[xor(target, accumulated_xor) + 1] * weight
            return
        end

        for configuration in 0:configuration_count-1
            recurse(
                target,
                xor(accumulated_xor, configuration),
                weight * branch_message[configuration + 1],
                remaining - 1,
            )
        end
    end

    for target in 0:configuration_count-1
        recurse(target, 0, 1.0 + 0.0im, child_arity)
    end

    folded
end

function wht_constraint_fold(angles::QAOAAngles, branch_degree::Int; child_arity::Int=2)
    branch_message = build_branch_message(angles)
    kernel = build_constraint_kernel(angles, branch_degree)
    kernel_hat = QaoaXorsat.wht(kernel)
    branch_hat = QaoaXorsat.wht(branch_message)

    QaoaXorsat.iwht(kernel_hat .* (branch_hat .^ child_arity))
end

@testset "WHT factorisation" begin
    @testset "round trip" begin
        @testset "n=$n" for n in (3, 5)
            values = randn(ComplexF64, 1 << n)
            @test QaoaXorsat.iwht(QaoaXorsat.wht(values)) ≈ values atol = 1e-12
        end
    end

    @testset "xor convolution theorem" begin
        left = randn(ComplexF64, 8)
        right = randn(ComplexF64, 8)
        direct = zeros(ComplexF64, 8)

        for target in 0:7
            for source in 0:7
                direct[target + 1] += left[source + 1] * right[xor(target, source) + 1]
            end
        end

        @test QaoaXorsat.xor_convolution(left, right) ≈ direct atol = 1e-10
    end

    @testset "xor autoconvolution theorem" begin
        values = randn(ComplexF64, 8)
        direct = zeros(ComplexF64, 8)

        for delta in 0:7
            for source in 0:7
                direct[delta + 1] += values[source + 1] * values[xor(source, delta) + 1]
            end
        end

        @test QaoaXorsat.xor_autoconvolution(values) ≈ direct atol = 1e-10
    end

    @testset "naive vs WHT constraint fold — p=$p, D=$D" for (p, D, trials) in [
        (1, 3, 100),
        (1, 4, 50),
        (2, 3, 20),
        (2, 4, 10),
        (3, 3, 5),
        (3, 4, 3),
    ]
        branch_degree = D - 1
        max_error = 0.0

        @testset "trial $trial" for trial in 1:trials
            angles = QAOAAngles(π .* rand(p), (π / 2) .* rand(p))
            naive = naive_constraint_fold(angles, branch_degree)
            transformed = wht_constraint_fold(angles, branch_degree)
            max_error = max(max_error, maximum(abs.(naive .- transformed)))

            @test transformed ≈ naive atol = 1e-10
        end

        @info "p=$p, D=$D: max |S_naive - S_wht| = $max_error over $trials trials"
    end
end