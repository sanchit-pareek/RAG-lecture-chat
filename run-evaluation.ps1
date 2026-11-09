param([switch]$Full, [ValidateRange(1,10)][int]$Repeats = 1, [string]$ResumeRun)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run start-local.ps1 to install the project first.' }
function Check-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed. Completed artifacts are preserved; backend placement has not been assumed." }
}
if ($ResumeRun) {
    $run = (Resolve-Path -LiteralPath $ResumeRun).Path
    $evaluationRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'evaluation')) + [IO.Path]::DirectorySeparatorChar
    if (-not $run.StartsWith($evaluationRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'ResumeRun must be an existing run directory under this project evaluation directory.'
    }
    if (Test-Path -LiteralPath (Join-Path $run 'paired')) { throw 'Paired results already exist; refusing to overwrite them.' }
} else {
    $run = Join-Path $PSScriptRoot ('evaluation/run-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    $null = New-Item -ItemType Directory -Path $run
}
Write-Host "Results: $run"
# Preserve a separate source-only review packet before exposing new system answers.
if (-not (Test-Path -LiteralPath (Join-Path $run 'independent-label-review.json'))) {
    & $Python prepare_label_review.py --output (Join-Path $run 'independent-label-review.json')
    Check-Exit 'Prepare independent label review'
}
Write-Host 'Stage 1/3: CPU retrieval across all 30 original questions (no generation).'
$cpuResults = Join-Path $run 'cpu-retrieval.json'
if ($ResumeRun) {
    & $Python -c 'import json,sys; from backend.rag import corpus,model_digest; r=json.load(open(sys.argv[1])); c=r["configuration"]; assert str(c["embedding_num_gpu"])=="0", "Not a CPU run"; assert c["metadata"]["corpus_sha256"]==corpus()[1]["corpus_sha256"], "Corpus changed"; assert c["metadata"]["model_digest"]==model_digest(), "Model changed"; print("Reusing verified saved CPU retrieval results")' $cpuResults
    Check-Exit 'Validate saved CPU run'
} else {
    & $Python evaluate.py benchmark --embedding-num-gpu 0 --repeats 1 --output $cpuResults
    Check-Exit 'CPU retrieval'
}
& $Python compare_rankings.py --candidate $cpuResults --output (Join-Path $run 'placement-comparison.json') --require-exact
Check-Exit 'CPU ranking equivalence'
Write-Host 'Stage 2/3: apply CPU embeddings to this project backend.'
$health = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 5
$codeHash = (& $Python -c 'from evaluation_suite import code_hash; print(code_hash())').Trim()
Check-Exit 'Compute runtime fingerprint'
if ($health.code_sha256 -ne $codeHash -or [string]$health.embedding_num_gpu -ne '0') {
    # Never terminate an arbitrary port owner: verify PID, executable and project launch.
    $listeners = @(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction Stop)
    $owners = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -ne 1) { throw 'Cannot identify a unique listener on port 8000. No process was stopped.' }
    $backendPid = [int]$owners[0]
    if ($health.pid -and [int]$health.pid -ne $backendPid) { throw 'Health PID differs from port owner. No process was stopped.' }
    $backendProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $backendPid"
    # Windows venv Python can redirect to the base interpreter: ExecutablePath
    # then differs even though the command explicitly launches this project's venv.
    $projectPythonLaunch = '^\s*"?' + [regex]::Escape($Python) + '"?\s+'
    $usesProjectPython = $backendProcess -and (
        $backendProcess.ExecutablePath -eq $Python -or
        $backendProcess.CommandLine -match $projectPythonLaunch)
    if (-not $usesProjectPython -or
        $backendProcess.CommandLine -notmatch '(?:^|\s)uvicorn\s+backend\.main:app(?:\s|$)') {
        $details = $backendProcess | Select-Object ProcessId, ExecutablePath, CommandLine | Format-List | Out-String
        throw "Cannot establish ownership of the running backend. No process was stopped.`nExpected project Python: $Python`nObserved process:`n$details`nIf this is your backend, stop it in its original terminal and start it there using:`n`$env:EMBEDDING_NUM_GPU = '0'`n& '$Python' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`nThen resume this run from another terminal."
    }
    Stop-Process -Id $backendPid
    Wait-Process -Id $backendPid -Timeout 10 -ErrorAction SilentlyContinue
    $env:EMBEDDING_NUM_GPU = '0'
    Start-Process -FilePath $Python -ArgumentList '-m uvicorn backend.main:app --host 127.0.0.1 --port 8000' `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $run 'backend.log') -RedirectStandardError (Join-Path $run 'backend-error.log')
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            $health = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2
            if ($health.code_sha256 -eq $codeHash -and [string]$health.embedding_num_gpu -eq '0') { $ready = $true; break }
        } catch {}
    }
    if (-not $ready) { throw "CPU backend did not become ready. See $run/backend-error.log" }
}
$health | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $run 'applied-health.json') -Encoding utf8
Write-Host 'Stage 3/3: paired full-response HTTP measurements; every completed answer is saved.'
$suiteArgs = @('evaluation_suite.py', '--repeats', "$Repeats", '--timeout', '180', '--output-dir', (Join-Path $run 'paired'))
if (-not $Full) {
    # Fixed before this run, spanning definitions, numerics, derivations and known retrieval misses.
    $suiteArgs += @('--query-ids','q01','q02','q07','q11','q15','q20','q23','q30')
    Write-Host 'Pilot: 8 varied questions, 16 paired answers plus one warmup. Use -Full for all 29.'
}
& $Python @suiteArgs
Check-Exit 'Paired API evaluation'
Write-Host "Finished measurements: $run. Independent labels and blinded answer ratings still require review."
