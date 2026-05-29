using QaoaXorsat
using DoubleFloats
using Random
using Test

@testset "Optimization" begin
    @testset "canonicalize_angles" begin
        angles = QAOAAngles([-0.5, 2π + 0.25], [-0.1, π + 0.2])
        canonical = canonicalize_angles(angles)

        @test canonical.γ[1] ≈ 2π - 0.5 atol = 1e-12
        @test canonical.γ[2] ≈ 0.25 atol = 1e-12
        @test canonical.β[1] ≈ π - 0.1 atol = 1e-12
        @test canonical.β[2] ≈ 0.2 atol = 1e-12
    end

    @testset "random_angles" begin
        rng = MersenneTwister(1234)
        angles = random_angles(3; rng)

        @test depth(angles) == 3
        @test all(0.0 ≤ γ < 2π for γ in angles.γ)
        @test all(0.0 ≤ β < π for β in angles.β)
    end

    @testset "extend_angles" begin
        base = QAOAAngles([0.2, 0.4], [0.1, 0.3])
        extended = extend_angles(base, 4)

        @test extended.γ == [0.2, 0.4, 0.4, 0.4]
        @test extended.β == [0.1, 0.3, 0.3, 0.3]
    end

    @testset "depth_optimization_budget" begin
        budget_p3 = QaoaXorsat.depth_optimization_budget(3, 8, 200)
        budget_p4 = QaoaXorsat.depth_optimization_budget(4, 8, 200)
        budget_p5 = QaoaXorsat.depth_optimization_budget(5, 8, 200)

        @test budget_p3.restarts == 8
        @test budget_p3.maxiters == 200
        @test budget_p4.restarts == 4
        @test budget_p4.maxiters == 400
        @test budget_p5.restarts == 2
        @test budget_p5.maxiters == 800
        @test QaoaXorsat.retry_optimization_budget(200) == 200
    end

    @testset "optimize_angles MaxCut p=1" begin
        params = TreeParams(2, 3, 1)
        optimum = 0.5 + sqrt(3) / 9
        seed = QAOAAngles([0.7], [0.3])

        result = optimize_angles(
            params;
            clause_sign=-1,
            restarts=0,
            maxiters=100,
            initial_guesses=[seed],
            rng=MersenneTwister(7),
        )

        @test result.value ≈ optimum atol = 1e-6
        @test depth(result.angles) == 1
        @test result.starts == 1
        @test result.evaluations ≥ 1
        @test result.wall_time_seconds ≥ 0.0
        @test result.best_start_wall_time_seconds ≥ 0.0
        @test result.best_start_wall_time_seconds ≤ result.wall_time_seconds
        @test result.restarts == 0
        @test result.maxiters == 100
        @test result.retry_count == 0
        @test result.best_start_kind == :seeded
        @test length(result.start_results) == 1
    end

    @testset "optimize_angles checkpointed Double64" begin
        mktempdir() do dir
            result = optimize_angles(
                TreeParams(2, 3, 1);
                clause_sign=-1,
                restarts=0,
                maxiters=5,
                initial_guesses=[QAOAAngles([0.7], [0.3])],
                rng=MersenneTwister(31),
                eval_eltype=Double64,
                checkpointed=true,
                checkpoint_disk_dir=dir,
                checkpoint_max_ram_checkpoints=1,
            )

            @test QaoaXorsat.is_valid_qaoa_value(result.value)
            @test depth(result.angles) == 1
            @test isempty(readdir(dir))
        end
    end

    @testset "optimize_angles start telemetry" begin
        result = optimize_angles(
            TreeParams(2, 3, 1);
            clause_sign=-1,
            restarts=1,
            maxiters=5,
            initial_guesses=[QAOAAngles([0.7], [0.3])],
            rng=MersenneTwister(19),
        )

        @test result.starts == 2
        @test result.restarts == 1
        @test result.maxiters == 5
        @test length(result.start_results) == 2
        @test [start.kind for start in result.start_results] == [:seeded, :random]
        @test all(start -> start.evaluations ≥ 1, result.start_results)
        @test all(start -> start.wall_time_seconds ≥ 0.0, result.start_results)
    end

    @testset "angle snapshots and result stores" begin
        angles = QAOAAngles([0.1, 0.2], [0.3, 0.4])
        snapshot = AngleSnapshot(
            2, 3, 2, 1, 7, 3, 12.5, 0.75, 1.0e-3, 2.5e-4, 8.0e-3,
            angles, :best_seen, "2026-05-29T00:00:00Z",
        )

        mktempdir() do dir
            snapshot_path = joinpath(dir, "snapshots.csv")
            write_angle_snapshot!(snapshot_path, snapshot)
            lines = readlines(snapshot_path)
            @test lines[1] == snapshot_csv_header()
            @test length(lines) == 2
            @test occursin("best_seen", lines[2])
            @test occursin("0.1;0.2", lines[2])
            @test occursin("0.3;0.4", lines[2])

            store = CsvResultStore(joinpath(dir, "sweep.csv"), :sweep)
            append_record!(store, AngleRecord(2, 3, 1, 0.7, 1.5,
                QAOAAngles([0.5], [0.6]), Dict{Symbol,Any}()))
            append_record!(store, AngleRecord(2, 3, 1, 0.8, 2.5,
                QAOAAngles([0.7], [0.8]), Dict{Symbol,Any}()))

            best = read_best_record(store, 2, 3, 1)
            @test best !== nothing
            @test best.value == 0.8
            @test best.angles.γ == [0.7]
            @test resolve_warm_start(PreviousDepthWarmStart(store, 2, 3, 1), 3) |> depth == 3
        end
    end

    @testset "core best-angle snapshots and plateau policy" begin
        snapshots = AngleSnapshot[]
        policy = PlateauPolicy(
            enabled=true,
            min_evaluations=1,
            value_window=1,
            gradient_window=1,
            value_range_tol=1.0,
            gradient_ceiling=1.0,
            gradient_value_range_tol=1.0,
        )

        result = optimize_angles(
            TreeParams(2, 3, 1);
            clause_sign=-1,
            restarts=0,
            maxiters=5,
            autodiff=:charge_adjoint,
            initial_guesses=[QAOAAngles([0.7], [0.3])],
            rng=MersenneTwister(29),
            plateau_policy=policy,
            on_angle_snapshot=snapshot -> push!(snapshots, snapshot),
        )

        @test !isempty(snapshots)
        @test all(snapshot -> snapshot.k == 2 && snapshot.D == 3 && snapshot.p == 1, snapshots)
        @test all(snapshot -> snapshot.start_index == 1, snapshots)
        @test all(snapshot -> depth(snapshot.angles) == 1, snapshots)
        @test all(snapshot -> QaoaXorsat.is_valid_qaoa_value(snapshot.value), snapshots)
        @test QaoaXorsat.is_valid_qaoa_value(result.value)
        @test result.converged
        @test result.termination_reason in (:value_plateau, :gradient_plateau, :optim_converged)
        @test result.best_snapshot !== nothing
    end

    @testset "optimize_angles canonicalizes stored result angles" begin
        result = optimize_angles(
            TreeParams(2, 3, 1);
            clause_sign=-1,
            restarts=0,
            maxiters=5,
            initial_guesses=[QAOAAngles([-0.7], [π + 0.3])],
            rng=MersenneTwister(23),
        )

        @test all(0.0 ≤ γ < 2π for γ in result.angles.γ)
        @test all(0.0 ≤ β < π for β in result.angles.β)
    end

    @testset "optimize_depth_sequence warm starts" begin
        callback_results = QaoaXorsat.AngleOptimizationResult[]
        results = optimize_depth_sequence(
            2,
            3,
            [1, 2];
            clause_sign=-1,
            restarts=0,
            maxiters=5,
            rng=MersenneTwister(11),
            on_result=result -> push!(callback_results, result),
        )

        @test length(results) == 2
        @test length(callback_results) == 2
        @test depth.(getfield.(callback_results, :angles)) == [1, 2]
        @test depth(results[1].angles) == 1
        @test depth(results[2].angles) == 2
        @test all(result -> isfinite(result.value), results)
        @test all(result -> result.wall_time_seconds ≥ 0.0, results)
        @test all(result -> result.best_start_wall_time_seconds ≥ 0.0, results)
        @test all(result -> result.best_start_wall_time_seconds ≤ result.wall_time_seconds, results)
        @test results[1].best_start_kind in (:random, :seeded)
        @test results[2].best_start_kind in (:warm, :random, :retry)
        @test results[2].restarts == 0
        @test results[2].maxiters in (5, 10)
        @test all(result -> !isempty(result.start_results), results)
    end

    @testset "merge_optimization_results" begin
        starts = [QaoaXorsat.AngleOptimizationStartResult(:random, 0.5, 1.0, 10, 5, true, QaoaXorsat.OptimizationTraceEntry[])]

        primary = AngleOptimizationResult(
            QAOAAngles([0.3], [0.5]), 0.7, 1.0, 0.8, 20, 1, 5, false,
            1, 100, 0, :warm, 1.0e-6, starts,
        )
        secondary_better_value = AngleOptimizationResult(
            QAOAAngles([0.4], [0.6]), 0.8, 2.0, 1.5, 30, 1, 8, true,
            0, 200, 0, :retry, 1.0e-6, starts,
        )

        merged = QaoaXorsat.merge_optimization_results(primary, secondary_better_value)
        @test merged.value == 0.8
        @test merged.angles.γ == [0.4]
        @test merged.angles.β == [0.6]
        @test merged.evaluations == 50
        @test merged.starts == 2
        @test merged.retry_count == 1

        @testset "tie-break: converged secondary wins" begin
            primary_unconverged = AngleOptimizationResult(
                QAOAAngles([0.3], [0.5]), 0.7, 1.0, 0.8, 20, 1, 5, false,
                1, 100, 0, :warm, 1.0e-6, starts,
            )
            secondary_converged = AngleOptimizationResult(
                QAOAAngles([0.31], [0.51]), 0.7, 2.0, 1.5, 30, 1, 8, true,
                0, 200, 0, :retry, 1.0e-6, starts,
            )

            merged_tie = QaoaXorsat.merge_optimization_results(primary_unconverged, secondary_converged)
            @test merged_tie.converged == true
            @test merged_tie.angles.γ == [0.31]
            @test merged_tie.angles.β == [0.51]
        end

        @testset "primary wins when equal and both converged" begin
            primary_conv = AngleOptimizationResult(
                QAOAAngles([0.3], [0.5]), 0.7, 1.0, 0.8, 20, 1, 5, true,
                1, 100, 0, :warm, 1.0e-6, starts,
            )
            secondary_conv = AngleOptimizationResult(
                QAOAAngles([0.31], [0.51]), 0.7, 2.0, 1.5, 30, 1, 8, true,
                0, 200, 0, :retry, 1.0e-6, starts,
            )

            merged_both = QaoaXorsat.merge_optimization_results(primary_conv, secondary_conv)
            @test merged_both.angles.γ == [0.3]
            @test merged_both.angles.β == [0.5]
        end
    end

    @testset "optimization policy constructors" begin
        prod_p10 = QaoaXorsat.optimization_policy(TreeParams(2, 3, 10))
        @test prod_p10.autodiff == :adjoint
        @test !prod_p10.checkpointed
        @test !prod_p10.plateau.enabled

        prod_p13 = QaoaXorsat.optimization_policy(TreeParams(2, 3, 13))
        @test prod_p13.autodiff == :charge_adjoint
        @test prod_p13.checkpointed
        @test prod_p13.plateau.enabled

        low_mem = QaoaXorsat.optimization_policy(TreeParams(2, 3, 13); memory_profile=:low)
        @test low_mem.autodiff == :charge_adjoint
        @test low_mem.checkpointed
        @test low_mem.checkpoint_max_ram_checkpoints == 1

        high_mem = QaoaXorsat.optimization_policy(TreeParams(2, 3, 13); memory_profile=:high)
        @test high_mem.autodiff == :adjoint
        @test !high_mem.checkpointed

        non_prod = QaoaXorsat.optimization_policy(TreeParams(2, 3, 13); mode=:debug)
        @test non_prod.restarts == 2
        @test non_prod.maxiters == 100
        @test non_prod.autodiff == :charge_adjoint
        @test !non_prod.plateau.enabled

        from_ints = QaoaXorsat.optimization_policy(2, 3, 13)
        @test from_ints.restarts == prod_p13.restarts
        @test from_ints.maxiters == prod_p13.maxiters
        @test from_ints.autodiff == prod_p13.autodiff
        @test from_ints.checkpointed == prod_p13.checkpointed
    end

    @testset "result store edge cases" begin
        mktempdir() do dir
            sweep_path = joinpath(dir, "mixed-sweep.csv")
            open(sweep_path, "w") do io
                println(io, "k,D,p,ctilde,wall_seconds,gamma,beta")
                println(io, "2,3,2,1.5,1.0,0.1;0.2,0.3;0.4") # invalid value (>1)
                println(io, "2,3,2,0.77,1.2,0.11;0.22,0.33;0.44")
                println(io, "garbage,row")
            end

            sweep_store = CsvResultStore(sweep_path, :sweep)
            records = read_records(sweep_store)
            @test length(records) == 2

            best = read_best_record(sweep_store, 2, 3, 2)
            @test best !== nothing
            @test best.value == 0.77

            @test resolve_warm_start(PreviousDepthWarmStart(sweep_store, 9, 9, 1), 3) === nothing

            swarm_path = joinpath(dir, "swarm.csv")
            open(swarm_path, "w") do io
                println(io, "k,D,p,ctilde,evals,wall_seconds,gamma,beta")
                println(io, "2,3,2,0.72,50,1.0,0.1;0.2,0.3;0.4")
            end
            swarm_store = CsvResultStore(swarm_path, :swarm)
            swarm_records = read_records(swarm_store)
            @test length(swarm_records) == 1
            @test swarm_records[1].metadata[:evaluations] == 50

            @test_throws ArgumentError append_record!(swarm_store,
                AngleRecord(2, 3, 2, 0.5, 1.0, QAOAAngles([0.1, 0.2], [0.3, 0.4]), Dict{Symbol,Any}()))
        end
    end

    @testset "run_optimization structured callbacks" begin
        policy = OptimizationPolicy(
            restarts=0,
            maxiters=10,
            g_abstol=1.0e-6,
            autodiff=:charge_adjoint,
            eval_eltype=Float64,
            checkpointed=false,
            checkpoint_max_ram_checkpoints=typemax(Int),
            plateau=PlateauPolicy(
                enabled=true,
                min_evaluations=1,
                value_window=1,
                gradient_window=1,
                value_range_tol=1.0,
                gradient_ceiling=1.0,
                gradient_value_range_tol=1.0,
            ),
        )

        spec = OptimizationRunSpec(
            TreeParams(2, 3, 1),
            -1,
            [QAOAAngles([0.7], [0.3])],
            :seeded,
            policy,
            MersenneTwister(77),
        )

        events = OptimizationEvent[]
        snapshot_events = AngleSnapshot[]
        depth_events = DepthResultEvent[]
        callbacks = OptimizationCallbacks(
            on_event = e -> push!(events, e),
            on_evaluation = _ -> nothing,
            on_angle_snapshot = s -> push!(snapshot_events, s),
            on_depth_result = e -> push!(depth_events, e),
        )

        result = run_optimization(spec; callbacks)
        @test QaoaXorsat.is_valid_qaoa_value(result.value)
        @test result.restarts == policy.restarts
        @test result.maxiters == policy.maxiters
        @test result.g_abstol == policy.g_abstol
        @test !isempty(events)
        @test any(e -> e isa AngleSnapshotEvent, events)
        @test any(e -> e isa PlateauEvent, events)
        @test any(e -> e isa DepthResultEvent, events)
        @test !isempty(snapshot_events)
        @test length(depth_events) == 1
        @test depth_events[1].result === result
    end

    @testset "structured event dispatcher routes callbacks" begin
        seen_events = OptimizationEvent[]
        eval_events = EvaluationEvent[]
        snapshots = AngleSnapshot[]
        depth_events = DepthResultEvent[]

        callbacks = OptimizationCallbacks(
            on_event = e -> push!(seen_events, e),
            on_evaluation = e -> push!(eval_events, e),
            on_angle_snapshot = s -> push!(snapshots, s),
            on_depth_result = e -> push!(depth_events, e),
        )

        eval_event = EvaluationEvent(1, 2, 3.0, 0.7, 1.0e-3, 0.71)
        QaoaXorsat._emit_optimization_event!(callbacks, eval_event)
        @test seen_events[end] === eval_event
        @test eval_events[end] === eval_event

        snapshot = AngleSnapshot(
            2, 3, 1, 1, 2, 3, 4.0, 0.72, 1.0e-3, 2.0e-4, 3.0e-3,
            QAOAAngles([0.1], [0.2]), :best_seen, "2026-05-29T00:00:00Z",
        )
        snapshot_event = AngleSnapshotEvent(snapshot)
        QaoaXorsat._emit_optimization_event!(callbacks, snapshot_event)
        @test seen_events[end] === snapshot_event
        @test snapshots[end] === snapshot

        depth_event = DepthResultEvent(:ok)
        QaoaXorsat._emit_optimization_event!(callbacks, depth_event)
        @test seen_events[end] === depth_event
        @test depth_events[end] === depth_event
    end

    @testset "optimize_angles event callbacks without legacy hooks" begin
        events = OptimizationEvent[]
        callbacks = OptimizationCallbacks(on_event = e -> push!(events, e))
        policy = PlateauPolicy(
            enabled=true,
            min_evaluations=1,
            value_window=1,
            gradient_window=1,
            value_range_tol=1.0,
            gradient_ceiling=1.0,
            gradient_value_range_tol=1.0,
        )

        result = optimize_angles(
            TreeParams(2, 3, 1);
            clause_sign=-1,
            restarts=0,
            maxiters=8,
            autodiff=:charge_adjoint,
            initial_guesses=[QAOAAngles([0.7], [0.3])],
            rng=MersenneTwister(91),
            plateau_policy=policy,
            callbacks=callbacks,
            on_angle_snapshot=nothing,
        )

        @test QaoaXorsat.is_valid_qaoa_value(result.value)
        @test any(e -> e isa AngleSnapshotEvent, events)
        @test any(e -> e isa PlateauEvent, events)
    end

    @testset "optimize_angles emits evaluation events with zero throttle" begin
        policy = PlateauPolicy(
            enabled=true,
            min_evaluations=1,
            value_window=1,
            gradient_window=1,
            value_range_tol=1.0,
            gradient_ceiling=1.0,
            gradient_value_range_tol=1.0,
        )

        events = OptimizationEvent[]
        callbacks = OptimizationCallbacks(on_event = e -> push!(events, e))

        result = optimize_angles(
            TreeParams(2, 3, 1);
            clause_sign=-1,
            restarts=0,
            maxiters=8,
            autodiff=:charge_adjoint,
            initial_guesses=[QAOAAngles([0.7], [0.3])],
            rng=MersenneTwister(131),
            plateau_policy=PlateauPolicy(enabled=false),
            callbacks=callbacks,
            on_evaluation=nothing,
            evaluation_report_seconds=0.0,
        )

        @test QaoaXorsat.is_valid_qaoa_value(result.value)
        eval_events = filter(e -> e isa EvaluationEvent, events)
        @test !isempty(eval_events)
    end

    @testset "optimize_angles rejects negative evaluation throttle" begin
        @test_throws ArgumentError optimize_angles(
            TreeParams(2, 3, 1);
            clause_sign=-1,
            restarts=0,
            maxiters=3,
            initial_guesses=[QAOAAngles([0.7], [0.3])],
            rng=MersenneTwister(41),
            evaluation_report_seconds=-1.0,
        )
    end
end
