<#
.SYNOPSIS
    Start the local Terminal visualization console.

.DESCRIPTION
    Uses the repository virtual-environment Python when it exists and falls
    back to the Python executable available on PATH.  The server remains in
    the foreground so its URL, errors, and Ctrl+C shutdown are visible.

.PARAMETER Port
    Loopback HTTP port. Defaults to 8787.

.PARAMETER Replay
    Optional rotating_slots ReplayBundle directory or its replay.json,
    console JSON, legacy HTML, or single-file console replay to open.

.PARAMETER NoBrowser
    Do not open the default browser after the server binds.
#>
[CmdletBinding()]
param(
    [int]$Port = 8787,
    [string]$Replay,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repo "venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $python = $pythonCommand.Source
}

Set-Location -LiteralPath $repo
$arguments = @(
    "-m",
    "terminal_marl.visualization.console_server",
    "--port",
    [string]$Port
)
if ($Replay) {
    $arguments += @("--replay", $Replay)
}
if ($NoBrowser) {
    $arguments += "--no-browser"
}

& $python @arguments
exit $LASTEXITCODE
