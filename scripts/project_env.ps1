$ProjectRoot = Split-Path -Parent $PSScriptRoot
foreach ($sub in @('.tmp', '.cache\pip', '.cache\torch', '.cache\matplotlib', '.cache\huggingface')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $sub) | Out-Null
}
$env:TEMP = Join-Path $ProjectRoot '.tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
$env:TORCH_HOME = Join-Path $ProjectRoot '.cache\torch'
$env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache\matplotlib'
$env:HF_HOME = Join-Path $ProjectRoot '.cache\huggingface'
$env:PYTHONPYCACHEPREFIX = Join-Path $ProjectRoot '.cache\pycache'
$env:PYTEST_ADDOPTS = '--basetemp="' + (Join-Path $ProjectRoot '.tmp\pytest') + '"'
