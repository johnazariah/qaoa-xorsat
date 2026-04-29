# Compatibility wrapper for the Windows P710 box.
#
# The only supported local workflow is MaxCut. Max-k-XORSAT is run on
# Stephen's SLURM cluster via scripts/start-xorsat-slurm.sh.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $ScriptDir "start-maxcut-local.ps1") @args
