using QaoaXorsat
using Test

@testset "Spectral analysis" begin
    @testset "spectral_snapshot" begin
        @testset "all-ones vector" begin
            # WHT of all-ones vector of length 2^n has a single nonzero coefficient
            N = 8
            ones_vec = ones(ComplexF64, N)
            snap = spectral_snapshot(ones_vec, 0)

            @test snap.step == 0
            @test length(snap.magnitudes) == N
            @test snap.magnitudes[1] ≈ N atol = 1e-12  # WHT(1) = N at index 0
            @test all(snap.magnitudes[2:end] .< 1e-12)   # all others zero
            # Effective rank should be 1 at any tolerance
            @test snap.effective_ranks[1e-10] == 1
            @test snap.effective_ranks[1e-4] == 1
        end

        @testset "delta function" begin
            N = 8
            delta = zeros(ComplexF64, N)
            delta[1] = 1.0
            snap = spectral_snapshot(delta, 1)

            # WHT of delta is the all-ones vector (scaled)
            @test snap.magnitudes[1] ≈ 1.0 atol = 1e-12
            @test snap.magnitudes[end] ≈ 1.0 atol = 1e-12  # all equal magnitude
            # Effective rank should be N (fully dense)
            @test snap.effective_ranks[1e-4] == N
        end

        @testset "sparse signal" begin
            N = 16
            # Construct a signal that is 2-sparse in WHT domain
            signal = zeros(ComplexF64, N)
            signal[1] = 3.0 + 0im
            signal[2] = 1.0 + 0im
            time_domain = QaoaXorsat.iwht(signal)

            snap = spectral_snapshot(time_domain, 0)
            @test snap.magnitudes[1] ≈ 3.0 atol = 1e-12
            @test snap.magnitudes[2] ≈ 1.0 atol = 1e-12
            @test all(snap.magnitudes[3:end] .< 1e-12)
            @test snap.effective_ranks[1e-10] == 2
        end
    end

    @testset "spectral_decay_rate" begin
        @testset "flat spectrum has zero decay" begin
            mags = ones(Float64, 100)
            snap = SpectralSnapshot(0, mags, Dict{Float64,Int}())
            rate, _ = spectral_decay_rate(snap)
            @test rate ≈ 0.0 atol = 0.01
        end

        @testset "power-law decay" begin
            # |c_r| = r^{-2} → log|c_r| = -2 log(r) → decay rate ≈ 2
            mags = [1.0 / r^2 for r in 1:1000]
            snap = SpectralSnapshot(0, mags, Dict{Float64,Int}())
            rate, r² = spectral_decay_rate(snap)
            @test rate ≈ 2.0 atol = 0.01
            @test r² > 0.99
        end
    end

    @testset "basso_branch_tensor_instrumented" begin
        @testset "produces correct branch tensor" begin
            @testset "k=$k, D=$D, p=$p" for (k, D, p) in [(2, 3, 1), (3, 4, 1), (2, 3, 2), (3, 4, 2)]
                params = TreeParams(k, D, p)
                angles = QAOAAngles(rand(p), rand(p))

                # Reference: uninstrumented branch tensor
                f_table = QaoaXorsat.basso_f_table(angles)
                reference = QaoaXorsat.basso_branch_tensor(params, angles; f_table)

                # Instrumented version
                result, profile = basso_branch_tensor_instrumented(params, angles; f_table)

                @test result ≈ reference atol = 1e-12
                @test length(profile.snapshots) == p + 1  # steps 0 through p
                @test profile.snapshots[1].step == 0
                @test profile.snapshots[end].step == p
            end
        end

        @testset "initial snapshot is 1-sparse" begin
            params = TreeParams(3, 4, 2)
            angles = QAOAAngles(rand(2), rand(2))
            _, profile = basso_branch_tensor_instrumented(params, angles)

            # B^(0) = all-ones → 1-sparse in WHT domain
            @test profile.snapshots[1].effective_ranks[1e-10] == 1
        end
    end

    @testset "format_spectral_report" begin
        params = TreeParams(2, 3, 1)
        angles = QAOAAngles([0.5], [0.3])
        _, profile = basso_branch_tensor_instrumented(params, angles)
        report = format_spectral_report(profile)

        @test contains(report, "k=2")
        @test contains(report, "D=3")
        @test contains(report, "p=1")
        @test contains(report, "t=0")
        @test contains(report, "t=1")
    end

    @testset "CSV output" begin
        params = TreeParams(2, 3, 1)
        angles = QAOAAngles([0.5], [0.3])
        _, profile = basso_branch_tensor_instrumented(params, angles)

        @testset "spectrum CSV" begin
            buf = IOBuffer()
            write_spectral_csv(buf, profile)
            csv = String(take!(buf))
            lines = split(csv, "\n"; keepempty=false)
            @test startswith(lines[1], "step,rank,magnitude,relative_magnitude")
            # 2 steps × 2^3=8 entries = 16 data lines + 1 header
            @test length(lines) == 17
        end

        @testset "ranks CSV" begin
            buf = IOBuffer()
            write_effective_ranks_csv(buf, profile)
            csv = String(take!(buf))
            lines = split(csv, "\n"; keepempty=false)
            @test startswith(lines[1], "step,N")
            # 2 steps (t=0, t=1) + 1 header
            @test length(lines) == 3
        end
    end
end
