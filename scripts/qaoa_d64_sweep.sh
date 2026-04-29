#!/usr/bin/env bash
# Legacy SLURM entry point. Kept so old sbatch commands submit the current
# Double64 + CPU-checkpointed Max-k-XORSAT cluster sweep instead of the old
# all-pairs-from-p=1 D64 job.
#
# Prefer:
#   bash scripts/start-xorsat-slurm.sh
#
#SBATCH --job-name=qaoa-p16
#SBATCH --array=1-9
#SBATCH --partition=c3dssd
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1400G
#SBATCH --requeue
#SBATCH --comment="maintain_node"
#SBATCH --output=qaoa-p16_%A-%a.out
#SBATCH --error=qaoa-p16_%A-%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/qaoa_cluster_p16.sh"
