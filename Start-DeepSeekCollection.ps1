param(
    [int]$Minutes = 105,
    [double]$MaxUsd = 10,
    [string]$OutputDirectory = 'data/runtime/deepseek-auto-20261113',
    [string]$ApiKeyFile
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY) -and [string]::IsNullOrWhiteSpace($ApiKeyFile)) {
    throw 'DEEPSEEK_API_KEY is unavailable. Run this script in the same PowerShell window where you set the key.'
}
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
$runDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $OutputDirectory))
$workspacePrefix = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
if (-not $runDirectory.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputDirectory must be inside this project.'
}
if (Test-Path -LiteralPath (Join-Path $runDirectory 'run.lock')) { throw 'A collection process may already be running.' }
if (Test-Path -LiteralPath (Join-Path $runDirectory '01-pro-pilot/manifest.json')) {
    throw 'This output directory already contains a teacher run. Choose a new OutputDirectory.'
}
$null = New-Item -ItemType Directory -Path $runDirectory -Force
$arguments = @('-X', 'utf8', '-m', 'finance_agent.collect_sft', '--output', ('"' + $runDirectory + '"'),
               '--minutes', $Minutes.ToString(), '--max-usd', $MaxUsd.ToString([Globalization.CultureInfo]::InvariantCulture))
if (-not [string]::IsNullOrWhiteSpace($ApiKeyFile)) {
    $resolvedKeyFile = (Resolve-Path -LiteralPath $ApiKeyFile).Path
    $arguments += @('--api-key-file', ('"' + $resolvedKeyFile + '"'))
}
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $runDirectory 'stdout.log') -RedirectStandardError (Join-Path $runDirectory 'stderr.log') -PassThru
$launch = @{ pid = $process.Id; output = $runDirectory; started_at_utc = [DateTime]::UtcNow.ToString('o'); maximum_estimated_usd = $MaxUsd; minutes = $Minutes }
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDirectory 'launch.json') -Encoding UTF8
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'data/runtime/deepseek-collection-latest.json') -Encoding UTF8
Write-Host "Collection started, PID $($process.Id). Results: $runDirectory"
Write-Host 'The key is used in process memory and was not copied into collection output files.'
