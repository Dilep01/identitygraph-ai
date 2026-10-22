param([string]$PythonExe = '')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\project_env.ps1"
Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (-not $PythonExe) {
        $Candidates = @(
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
        )
        $PythonExe = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    }
    if (-not $PythonExe) { throw 'Pass -PythonExe with an existing Python 3.11 or 3.12 executable.' }
    & $PythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
if (Test-Path -LiteralPath 'requirements-lock.txt') {
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & '.\.venv\Scripts\python.exe' -m pip install --no-deps -e .
} else {
    & '.\.venv\Scripts\python.exe' -m pip install -e '.[dev]'
}
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
