param(
    [ValidateSet('Status','Pause','Resume')][string]$Action = 'Status',
    [string]$RunDirectory = 'data/runtime/sft-teacher-epoch1-20261113',
    [ValidateRange(0,2147483647)][int]$SegmentUpdates = 0
)
$ErrorActionPreference = 'Stop'
$runPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $RunDirectory))
$workspacePrefix = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
if (-not $runPath.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'RunDirectory must be within this workspace.'
}
$statusPath = Join-Path $runPath 'status.json'
$status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
if ($Action -eq 'Status') {
    $status | ConvertTo-Json -Depth 8
    $pointer = Join-Path $runPath 'latest-checkpoint.json'
    if (Test-Path -LiteralPath $pointer) { Get-Content -LiteralPath $pointer }
    return
}
if ($Action -eq 'Pause') {
    if ($status.state -eq 'complete' -or $status.state -eq 'paused') {
        Write-Host "Training is already $($status.state)."
        return
    }
    if ($status.state -notin @('training','preparing')) { throw 'This run is not actively training.' }
    $null = New-Item -ItemType File -Path (Join-Path $runPath 'pause.requested') -Force
    Write-Host 'Pause requested. The current optimizer update will finish, then a checkpoint will be saved and the process will exit.'
    return
}
if (Test-Path -LiteralPath (Join-Path $runPath 'run.lock')) {
    throw 'Run lock exists. Do not resume while another training process may still be running.'
}
if (-not (Test-Path -LiteralPath (Join-Path $runPath 'latest-checkpoint.json'))) { throw 'No checkpoint is available.' }
if ($status.state -eq 'complete') { throw 'This training run is already complete.' }
$config = Get-Content -LiteralPath (Join-Path $runPath 'config.json') -Raw | ConvertFrom-Json
$pausePath = Join-Path $runPath 'pause.requested'
if (Test-Path -LiteralPath $pausePath) { Remove-Item -LiteralPath $pausePath }
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$arguments = @('-u', '-X', 'utf8', '-m', 'finance_agent.sft_teacher', '--resume',
    '--output', ('"' + $runPath + '"'), '--release', ('"' + $config.release + '"'),
    '--snapshot', ('"' + $config.snapshot + '"'), '--steps', $config.steps.ToString(),
    '--accumulation', $config.gradient_accumulation.ToString(), '--max-length', $config.max_length.ToString(),
    '--learning-rate', $config.learning_rate.ToString('R', [Globalization.CultureInfo]::InvariantCulture),
    '--seed', $config.seed.ToString(), '--selection', $config.selection,
    '--warmup-steps', $config.warmup_steps_excluded_from_speed.ToString(),
    '--checkpoint-every', $config.checkpoint_every.ToString())
$arguments += @('--rank', $config.rank.ToString(), '--alpha', $config.alpha.ToString(),
    '--dropout', $config.dropout.ToString('R', [Globalization.CultureInfo]::InvariantCulture),
    '--target-modules') + @($config.target_modules)
if ($SegmentUpdates -gt 0) { $arguments += @('--segment-updates', $SegmentUpdates.ToString()) }
$suffix = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')
$process = Start-Process -FilePath (Join-Path $PSScriptRoot '.model-venv/Scripts/python.exe') `
    -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $runPath "resume-$suffix.stdout.log") `
    -RedirectStandardError (Join-Path $runPath "resume-$suffix.stderr.log") -PassThru
Write-Host "Resume process started, PID $($process.Id). It will verify checkpoint hashes and restore the optimizer and RNG state."
