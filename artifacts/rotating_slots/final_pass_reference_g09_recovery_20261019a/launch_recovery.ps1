param([switch]$LocalCheckOnly, [switch]$Supervise)
$ErrorActionPreference = 'Stop'
if ($LocalCheckOnly -and $Supervise) { throw 'Choose one mode.' }
$taskBase = 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_reference_g09_recovery_20261019a'
$taskPython = 'D:\7_codespace\terminal\venv\Scripts\python.exe'
$taskDriver = Join-Path $taskBase 'recover_g09.py'
$taskOutput = Join-Path $taskBase 'run_20261019a'
$taskPrefix = Join-Path $taskBase 'launch_20261019a'
foreach ($taskRequired in @($taskPython, $taskDriver, 'D:\7_codespace\terminal\artifacts\rotating_slots\performance_reference_r0\.reference_manifest.json', 'D:\7_codespace\terminal\artifacts\rotating_slots\final_pass_reference_regression_local_20261019a\G09\benchmark')) {
    if (-not (Test-Path -LiteralPath $taskRequired)) { throw "Missing fixed input: $taskRequired" }
}
foreach ($taskNew in @($taskOutput, "$taskPrefix.stdout.txt", "$taskPrefix.stderr.txt", "$taskPrefix.supervisor.json", "$taskPrefix.completion.json")) {
    if (Test-Path -LiteralPath $taskNew) { throw "Refuse existing output/evidence: $taskNew" }
}
$taskSelf = $MyInvocation.MyCommand.Path
$taskHashes = @{ driver = (Get-FileHash -LiteralPath $taskDriver -Algorithm SHA256).Hash.ToLower(); launcher = (Get-FileHash -LiteralPath $taskSelf -Algorithm SHA256).Hash.ToLower() }
$taskLogicalArgv = @($taskPython, '-I', '-B', $taskDriver, '--run-missing', '--workers', '2', '--output', $taskOutput)
if ($LocalCheckOnly) {
    $taskTokens = $null; $taskErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($taskSelf, [ref]$taskTokens, [ref]$taskErrors) | Out-Null
    if ($taskErrors.Count) { throw ($taskErrors | Out-String) }
    @{ status = 'LOCAL_CHECK_ONLY'; hashes = $taskHashes; output = $taskOutput; argv = $taskLogicalArgv; AST_errors = 0; output_exists = $false; process_started = $false; server_executed = $false } | ConvertTo-Json -Depth 5
    exit 0
}
if (-not $Supervise) {
    if (Test-Path -LiteralPath "$taskPrefix.request.json") { throw 'Existing launch request; do not repeat.' }
    $taskShell = (Get-Process -Id $PID).Path
    $taskSupervisor = Start-Process -FilePath $taskShell -ArgumentList @('-NoProfile', '-File', ('"{0}"' -f $taskSelf), '-Supervise') -WindowStyle Hidden -PassThru
    $taskRequest = @{ status = 'SUPERVISOR_STARTED_NOT_COMPLETION'; request_PID = $PID; supervisor_PID = $taskSupervisor.Id; UTC = [DateTime]::UtcNow.ToString('o'); hashes = $taskHashes; argv = $taskLogicalArgv }
    $taskRequest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$taskPrefix.request.json" -Encoding UTF8
    $taskRequest | ConvertTo-Json -Depth 5
    exit 0
}
$taskStart = [DateTime]::UtcNow
$taskWatch = [Diagnostics.Stopwatch]::StartNew()
$taskChild = $null; $taskChildExit = $null; $taskException = $null
$taskRecord = @{ supervisor_PID = $PID; started_UTC = $taskStart.ToString('o'); argv = $taskLogicalArgv; hashes = $taskHashes; output = $taskOutput }
try {
    $taskChildArgs = @('-I', '-B', ('"{0}"' -f $taskDriver), '--run-missing', '--workers', '2', '--output', ('"{0}"' -f $taskOutput))
    $taskChild = Start-Process -FilePath $taskPython -ArgumentList $taskChildArgs -WorkingDirectory 'D:\7_codespace\terminal' -WindowStyle Hidden -RedirectStandardOutput "$taskPrefix.stdout.txt" -RedirectStandardError "$taskPrefix.stderr.txt" -PassThru
    $taskRecord.child_PID = $taskChild.Id
    $taskRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$taskPrefix.supervisor.json" -Encoding UTF8
    $null = $taskChild.Handle
    $taskChild.WaitForExit()
    $taskChild.Refresh()
    if ($null -eq $taskChild.ExitCode) { throw 'Child exited without an available numeric ExitCode.' }
    $taskChildExit = [int]$taskChild.ExitCode
}
catch { $taskException = $_ | Out-String }
finally {
    $taskWatch.Stop()
    $taskSupervisorExit = if ($null -ne $taskChildExit) { $taskChildExit } else { 1 }
    $taskRecord.ended_UTC = [DateTime]::UtcNow.ToString('o')
    $taskRecord.elapsed_wall_s = $taskWatch.Elapsed.TotalSeconds
    $taskRecord.child_numeric_exit = $taskChildExit
    $taskRecord.supervisor_numeric_exit = $taskSupervisorExit
    $taskRecord.exception = $taskException
    $taskRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$taskPrefix.completion.json" -Encoding UTF8
    if ($null -ne $taskChildExit) { "$taskChildExit" | Set-Content -LiteralPath "$taskPrefix.child_numeric_exit.txt" -Encoding ASCII }
    "$taskSupervisorExit" | Set-Content -LiteralPath "$taskPrefix.supervisor_numeric_exit.txt" -Encoding ASCII
}
exit $taskSupervisorExit
