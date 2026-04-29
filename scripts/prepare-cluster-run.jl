#!/usr/bin/env julia

println("prepare-cluster-run.jl is retired.")
println()
println("Current startup workflows:")
println("  Local MaxCut:          scripts/start-maxcut-local.sh or .ps1")
println("  Stephen SLURM XORSAT:  bash scripts/start-xorsat-slurm.sh")
println()
println("The SLURM XORSAT path now submits scripts/qaoa_cluster_p16.sh, which reads")
println("existing warm-start CSVs directly and no longer needs generated TOML configs.")
