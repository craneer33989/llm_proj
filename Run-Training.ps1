param(
    [ValidateSet('Plan','Prepare','Train','Resume','Pause','Status','Evaluate','Report','All')]
    [string]$Action = 'Status',
    [string]$Config = 'configs/teacher-sft.example.json',
    [ValidateRange(0,2147483647)][int]$SegmentUpdates = 0
)
$ErrorActionPreference = 'Stop'
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $PSScriptRoot $Config }
$arguments = @('-u','-X','utf8','-m','finance_agent.training_pipeline',$Action.ToLowerInvariant(),'--config',$configPath)
if ($SegmentUpdates -gt 0) { $arguments += @('--segment-updates',$SegmentUpdates.ToString()) }
Push-Location -LiteralPath $PSScriptRoot
try {
    & (Join-Path $PSScriptRoot '.venv/Scripts/python.exe') @arguments
    if ($LASTEXITCODE -ne 0) { throw "Training pipeline exited with code $LASTEXITCODE" }
} finally { Pop-Location }
