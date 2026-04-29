# Compatibility wrapper. Canonical local MaxCut startup lives in:
#   scripts\start-maxcut-local.ps1

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $ScriptDir "start-maxcut-local.ps1") @args
