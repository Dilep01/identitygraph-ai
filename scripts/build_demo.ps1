param([string]$Run = 'demo', [int]$Seed = 41, [int]$Accounts = 240, [int]$Epochs = 12)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\project_env.ps1"
Set-Location $ProjectRoot
& '.\.venv\Scripts\python.exe' -m identitygraph.cli build --run $Run --seed $Seed --accounts $Accounts --gnn-epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw 'Build failed. Partial run is retained for diagnosis; use a new run name.' }
