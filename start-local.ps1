param([switch]$Evaluate, [switch]$RebuildIndex, [switch]$CpuEmbeddings)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if ($CpuEmbeddings) { $env:EMBEDDING_NUM_GPU = '0' }
function Check-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)." }
}
function Wait-Endpoint([string]$Url) {
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try { return Invoke-RestMethod -Uri $Url -TimeoutSec 2 }
        catch { Start-Sleep -Seconds 1 }
    }
    throw "Service did not become ready: $Url. Check the log files."
}
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    Check-Exit 'Create virtual environment'
}
$Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
# Repair an incomplete venv as well as handling a fresh checkout.
& $Python -m ensurepip --upgrade
Check-Exit 'Bootstrap pip'
& $Python -m pip install -r backend/requirements.txt
Check-Exit 'Install Python dependencies'
if (-not $env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434' }
try { $null = Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags" -TimeoutSec 3 }
catch {
    if ($env:OLLAMA_BASE_URL -notin @('http://127.0.0.1:11434', 'http://localhost:11434')) { throw }
    Start-Process -FilePath (Get-Command ollama).Source -ArgumentList 'serve' -WindowStyle Hidden `
        -RedirectStandardOutput 'ollama.log' -RedirectStandardError 'ollama-error.log'
    $null = Wait-Endpoint "$env:OLLAMA_BASE_URL/api/tags"
}
if (-not $env:EMBEDDING_MODEL) { $env:EMBEDDING_MODEL = 'bge-m3' }
if (-not $env:LLM_MODEL) { $env:LLM_MODEL = 'llama3.1:8b' }
$tags = Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags"
foreach ($model in @($env:EMBEDDING_MODEL, $env:LLM_MODEL)) {
    $qualified = if ($model.Contains(':')) { $model } else { "${model}:latest" }
    if ($qualified -notin $tags.models.name) {
        # The CLI uses OLLAMA_HOST rather than the application's OLLAMA_BASE_URL.
        $oldHost = $env:OLLAMA_HOST
        try { $env:OLLAMA_HOST = $env:OLLAMA_BASE_URL; ollama pull $model; Check-Exit "Pull $model" }
        finally { $env:OLLAMA_HOST = $oldHost }
    }
}
$indexArgs = @('preprocessjson.py')
if ($RebuildIndex) { $indexArgs += '--force' }
& $Python @indexArgs
Check-Exit 'Build/validate embedding index'
& $Python evaluate.py stats
Check-Exit 'Extract corpus statistics'
$corpus = Get-Content 'evaluation/corpus_stats.json' -Raw | ConvertFrom-Json
Push-Location frontend
try { npm ci; Check-Exit 'Install frontend'; npm run build; Check-Exit 'Build frontend' }
finally { Pop-Location }
$health = $null
try { $health = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2 } catch {}
if ($health) {
    if ($health.corpus_sha256 -ne $corpus.corpus_sha256) {
        throw 'Port 8000 is occupied by another or stale backend. Stop it and rerun.'
    }
    $expectedPlacement = if ($env:EMBEDDING_NUM_GPU) { $env:EMBEDDING_NUM_GPU } else { 'automatic' }
    if ($health.api_version -ne '1.2.0' -or $health.embedding_num_gpu -ne $expectedPlacement) {
        throw 'Existing backend has old code or different embedding placement. Stop it and rerun, or use .\run-evaluation.ps1 to validate and restart the project backend.'
    }
} else {
    Start-Process -FilePath $Python -ArgumentList '-m uvicorn backend.main:app --host 127.0.0.1 --port 8000' `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -RedirectStandardOutput 'backend.log' -RedirectStandardError 'backend-error.log'
    $null = Wait-Endpoint 'http://127.0.0.1:8000/health'
}
# Smoke-test actual embeddings, retrieval and generation before declaring success.
$body = @{ question = 'What is a random variable?' } | ConvertTo-Json
$result = Invoke-RestMethod 'http://127.0.0.1:8000/query' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
$result | ConvertTo-Json -Depth 10 | Set-Content 'evaluation/smoke_response.json' -Encoding utf8
if (-not $result.answer -or $result.contexts.Count -eq 0) { throw 'End-to-end smoke test failed.' }
if ($Evaluate) {
    & $Python evaluate.py benchmark --repeats 3
    Check-Exit 'Retrieval benchmark'
}
try { $null = Invoke-WebRequest 'http://127.0.0.1:5173' -TimeoutSec 2 }
catch {
    $node = (Get-Command node).Source
    Start-Process -FilePath $node -ArgumentList 'frontend/node_modules/vite/bin/vite.js frontend --host 127.0.0.1 --port 5173 --strictPort' `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -RedirectStandardOutput 'frontend.log' -RedirectStandardError 'frontend-error.log'
    $null = Wait-Endpoint 'http://127.0.0.1:5173'
}
Write-Host 'Running: http://127.0.0.1:5173 (API docs: http://127.0.0.1:8000/docs)'
