param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\project_env.ps1"
Set-Location $ProjectRoot
& '.\.venv\Scripts\python.exe' -m identitygraph.cli serve --port $Port
