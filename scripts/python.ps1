$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv first.' }
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -X utf8 @args
    if ($LASTEXITCODE -ne 0) { throw 'Python failed.' }
} finally { Pop-Location }
