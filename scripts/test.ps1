$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\project_env.ps1"
Set-Location $ProjectRoot
& '.\.venv\Scripts\python.exe' -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& '.\.venv\Scripts\python.exe' -m ruff check src tests
if ($LASTEXITCODE -ne 0) { throw 'Lint failed.' }
