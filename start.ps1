param([switch]$RulesOnly, [switch]$Stop)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$statePath = Join-Path $PSScriptRoot 'data/runtime/services.json'
$runtime = Split-Path -Parent $statePath

if ($Stop) {
    if (Test-Path -LiteralPath $statePath) {
        $services = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $failed = @()
        foreach ($service in $services) {
            $process = Get-Process -Id $service.id -ErrorAction SilentlyContinue
            if ($process -and $process.StartTime.ToUniversalTime().Ticks.ToString() -eq $service.started) {
                # The Windows venv launcher and its Python server are separate processes.
                taskkill.exe /PID $service.id /T /F | Out-Null
                if ($LASTEXITCODE -ne 0) { $failed += $service }
            }
        }
        if ($failed.Count) {
            ConvertTo-Json -InputObject @($failed) | Set-Content -LiteralPath $statePath -Encoding utf8
            throw 'Could not stop a project process. Run Stop from the shell that owns the services.'
        }
        Remove-Item -LiteralPath $statePath
    }
    return
}
if (Test-Path -LiteralPath $statePath) { throw 'Run .\start.ps1 -Stop before restarting.' }
if (!(Test-Path 'data/runtime/filings.sqlite3')) { throw 'Build the index with .\setup.ps1 first.' }
foreach ($port in $(if ($RulesOnly) { @(8010, 8501) } else { @(8010, 8012, 8501) })) {
    $probe = [System.Net.Sockets.TcpClient]::new()
    try {
        $probe.Connect('127.0.0.1', $port)
        throw "Loopback port $port is already in use."
    } catch [System.Net.Sockets.SocketException] {
        # A connection refused by the local host leaves the port free.
    } finally {
        $probe.Dispose()
    }
}

$env:HF_HOME = Join-Path $PSScriptRoot '.cache/huggingface'
$env:FINANCE_PLANNER = if ($RulesOnly) { 'rules' } else { 'model' }
$env:FINANCE_MODEL_URL = 'http://127.0.0.1:8012'
$env:FINANCE_API_URL = 'http://127.0.0.1:8010'
$env:PYTHONUTF8 = '1'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
$started = [System.Collections.Generic.List[object]]::new()

function Start-ServiceProcess($name, $python, [string[]]$arguments) {
    if (!(Test-Path -LiteralPath $python)) { throw "Missing runtime $python. Run .\setup.ps1." }
    $logs = @{
        RedirectStandardOutput = (Join-Path $runtime "$name.out.log")
        RedirectStandardError = (Join-Path $runtime "$name.err.log")
    }
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru @logs
    $started.Add(@{ id = $process.Id; started = $process.StartTime.ToUniversalTime().Ticks.ToString(); name = $name })
    ConvertTo-Json -InputObject @($started.ToArray()) | Set-Content -LiteralPath $statePath -Encoding utf8
    return $process
}

function Wait-Health($url, $process, $seconds) {
    $until = [DateTime]::UtcNow.AddSeconds($seconds)
    do {
        if ($process.HasExited) { throw "Service exited. See logs in $runtime." }
        try {
            $response = Invoke-WebRequest -Uri $url -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return }
        } catch {
            Start-Sleep -Milliseconds 400
        }
    } while ([DateTime]::UtcNow -lt $until)
    throw "Service did not become healthy at $url. See logs in $runtime."
}

if (!$RulesOnly) {
    $model = Start-ServiceProcess 'model' (Join-Path $PSScriptRoot '.model-venv/Scripts/python.exe') @(
        '-m', 'uvicorn', 'finance_agent.model_service:app', '--host', '127.0.0.1', '--port', '8012'
    )
    Wait-Health 'http://127.0.0.1:8012/health' $model 120
}
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$api = Start-ServiceProcess 'api' $python @('-m', 'uvicorn', 'finance_agent.api:app', '--host', '127.0.0.1', '--port', '8010')
Wait-Health 'http://127.0.0.1:8010/health' $api 20
$ui = Start-ServiceProcess 'ui' $python @(
    '-m', 'streamlit', 'run', 'demo.py', '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true'
)
Wait-Health 'http://127.0.0.1:8501/_stcore/health' $ui 30
Write-Output "Demo http://127.0.0.1:8501  API http://127.0.0.1:8010/docs  Planner $env:FINANCE_PLANNER"
