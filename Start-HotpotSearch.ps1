param([switch]$Stop, [switch]$Foreground, [switch]$Training, [int]$Port = 8020)
$ErrorActionPreference = 'Stop'
if ($Training -and !$PSBoundParameters.ContainsKey('Port')) { $Port = 8021 }
$serverModule = if ($Training) { 'hotpot_search.training_api' } else { 'hotpot_search.api' }
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.model-venv\Scripts\python.exe'
$runtimePath = Join-Path $projectRoot 'data\hotpotqa\runtime'
$pidPath = Join-Path $runtimePath "server-$Port.json"
if ($Stop) {
    if (!(Test-Path -LiteralPath $pidPath)) { Write-Output 'No recorded HotpotSearch server.'; exit 0 }
    $saved = Get-Content -LiteralPath $pidPath -Raw | ConvertFrom-Json
    $running = Get-CimInstance Win32_Process -Filter "ProcessId = $($saved.pid)"
    if ($running) {
        $ownedModule = if ($saved.module) { $saved.module } else { 'hotpot_search.api' }
        if ($ownedModule -notin @('hotpot_search.api','hotpot_search.training_api') -or $running.ExecutablePath -ne $pythonPath -or $running.CommandLine -notlike "*$ownedModule*") {
            throw 'PID ownership mismatch; refusing to stop another process.'
        }
        $process = Get-Process -Id $saved.pid
        $recordedStart = ([DateTime]$saved.startTime).ToUniversalTime()
        if ($process.StartTime.ToUniversalTime().Ticks -ne $recordedStart.Ticks) { throw 'PID was reused; refusing to stop.' }
        # uv's Windows venv redirector can own an actual interpreter child.
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($saved.pid)"
        foreach ($child in $children) {
            if ($child.CommandLine -like "*$ownedModule*") { Stop-Process -Id $child.ProcessId }
        }
        Stop-Process -Id $saved.pid -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $pidPath
    Write-Output 'Stopped HotpotSearch.'
    exit 0
}
if (!(Test-Path -LiteralPath $pythonPath)) { throw 'Missing .model-venv; install requirements-hotpot.txt in a Python environment first.' }
if ($Training) {
    $searchReady = $false
    try { $searchReady = (Invoke-RestMethod -Uri 'http://127.0.0.1:8020/health' -TimeoutSec 2).ready } catch { }
    if (!$searchReady) { & $PSCommandPath }
}
if ($Foreground) {
    Push-Location $projectRoot
    try { & $pythonPath -X utf8 -m $serverModule --port $Port } finally { Pop-Location }
    exit $LASTEXITCODE
}
$listener = [System.Net.Sockets.TcpClient]::new()
try {
    $connected = $listener.ConnectAsync('127.0.0.1', $Port).Wait(500)
    if ($connected -and $listener.Connected) { throw "Port $Port is already in use; check http://127.0.0.1:$Port/health" }
} catch [System.AggregateException] {
    # Connection refused: expected when starting a fresh local server.
} finally { $listener.Dispose() }
New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-X','utf8','-m',$serverModule,'--port',"$Port") -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimePath "server-$Port.out.log") -RedirectStandardError (Join-Path $runtimePath "server-$Port.err.log")
@{ pid = $process.Id; startTime = $process.StartTime.ToUniversalTime().ToString('o'); port = $Port; module = $serverModule } | ConvertTo-Json | Set-Content -LiteralPath $pidPath -Encoding utf8
$deadline = [DateTime]::UtcNow.AddSeconds(30)
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        if ($health.ready) { Write-Output "Ready, PID $($process.Id). API docs: http://127.0.0.1:$Port/docs"; exit 0 }
    } catch { }
    if ($process.HasExited) { throw "Server exited. Check $runtimePath/server-$Port.err.log" }
    Start-Sleep -Milliseconds 300
}
throw "Server did not become ready within 30 seconds. Check $runtimePath/server-$Port.err.log"
