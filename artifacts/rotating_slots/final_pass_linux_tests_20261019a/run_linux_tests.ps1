param([switch]$LocalCheckOnly)
$ErrorActionPreference = 'Stop'
$taskProject = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$taskDriver = 'artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_tests.py'
$taskLauncher = 'artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_linux_tests.ps1'
$taskTests = @(
    'tests/test_rotating_slots_performance_bernstein_precompute.py',
    'tests/test_rotating_slots_performance_root_reuse.py',
    'tests/test_rotating_slots_performance_peak_reuse.py',
    'tests/test_rotating_slots_performance_energy_queries.py',
    'tests/test_rotating_slots_performance_p2.py',
    'tests/test_rotating_slots_performance_cartesian_component.py',
    'tests/test_rotating_slots_performance_execution_reuse.py',
    'tests/test_rotating_slots_performance_windows.py',
    'tests/test_rotating_slots_performance_paired.py',
    'tests/test_rotating_slots_performance_parallel.py',
    'tests/test_rotating_slots_performance_worker_lifecycle.py',
    'tests/test_rotating_slots_performance_deadline.py',
    'tests/test_rotating_slots_performance_runner.py',
    'tests/test_rotating_slots_physics.py'
)
$taskCore = Join-Path $taskProject 'terminal_marl/core/rotating_slots'
$taskFiles = @('terminal_marl/__init__.py','terminal_marl/core/__init__.py')
foreach ($taskDirectory in @($taskCore,(Join-Path $taskCore 'performance'))) {
    foreach ($taskFile in Get-ChildItem -LiteralPath $taskDirectory -File -Filter '*.py') {
        $taskFiles += [IO.Path]::GetRelativePath($taskProject,$taskFile.FullName).Replace('\','/')
    }
}
$taskFiles += @('scripts/server/run_rotating_slots_performance.py',
    'scripts/server/run_rotating_slots_performance_windows.py',
    'scripts/server/run_rotating_slots_performance_paired.py', $taskDriver, $taskLauncher)
$taskFiles += $taskTests
$taskFiles = @($taskFiles | Sort-Object -Unique)
$taskCommit = (& git -C $taskProject rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $taskCommit -notmatch '^[a-f0-9]{40}$') { throw 'Cannot read exact local HEAD label' }
$taskExpected = [ordered]@{}
foreach ($taskRelative in $taskFiles) {
    $taskPath = Join-Path $taskProject $taskRelative
    if (-not (Test-Path -LiteralPath $taskPath -PathType Leaf)) { throw "Required file not found: $taskRelative" }
    $taskHash = (Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskHash -notmatch '^[a-f0-9]{64}$') { throw "Invalid computed hash: $taskRelative" }
    $taskExpected[$taskRelative] = $taskHash
}
$taskRequest = [ordered]@{schema='final_pass_linux_tests_source_v1'; implementation_commit=$taskCommit;
    driver_relative=$taskDriver; driver_sha256=$taskExpected[$taskDriver]; tests=$taskTests; files=$taskExpected}
$taskJson = $taskRequest | ConvertTo-Json -Depth 6 -Compress
$taskRequestBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($taskJson))
$taskRoundTrip = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($taskRequestBase64)) | ConvertFrom-Json
foreach ($taskRelative in $taskFiles) {
    if ($taskRoundTrip.files.$taskRelative -ne $taskExpected[$taskRelative]) { throw "Base64 roundtrip mismatch: $taskRelative" }
}
if (@($taskRoundTrip.tests).Count -ne 14 -or $taskRoundTrip.driver_sha256 -ne $taskExpected[$taskDriver]) { throw 'Test/driver roundtrip mismatch' }
# Tiny gate only: verify the synced driver bytes, then run that actual file.
$taskGate = @'
import sys
sys.dont_write_bytecode=True
import base64,hashlib,json,pathlib,runpy
request=json.loads(base64.b64decode(sys.argv[2],validate=True))
assert sys.platform=='linux' and sys.flags.isolated and sys.dont_write_bytecode
assert request['schema']=='final_pass_linux_tests_source_v1'
relative='artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_tests.py'
assert request['driver_relative']==relative
driver=pathlib.Path('/home/user/projects/terminal')/relative
assert hashlib.sha256(driver.read_bytes()).hexdigest()==request['driver_sha256']==request['files'][relative]
sys.argv=[str(driver),'--request-base64',sys.argv[2]]
runpy.run_path(str(driver),run_name='__main__')
'@
$taskGateBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($taskGate))
$taskBootstrap = 'import sys;sys.dont_write_bytecode=True;import base64;exec(base64.b64decode(sys.argv[1]))'
if ($taskBootstrap.Contains("'")) { throw 'Bootstrap cannot contain shell single quotes' }
if ($taskGateBase64 -notmatch '^[A-Za-z0-9+/=]+$' -or $taskRequestBase64 -notmatch '^[A-Za-z0-9+/=]+$') { throw 'Unexpected base64 characters' }
$taskCommand = "/home/user/envs/terminal_env/bin/python -I -B -c '$taskBootstrap' '$taskGateBase64' '$taskRequestBase64'"
if ($taskCommand.Length -ge 30000) { throw "Final command must be shorter than 30000 characters: $($taskCommand.Length)" }
if ($LocalCheckOnly) {
    $taskParseTokens = $null; $taskParseErrors = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile($PSCommandPath,[ref]$taskParseTokens,[ref]$taskParseErrors)
    if ($taskParseErrors.Count -ne 0) { throw ($taskParseErrors | Out-String) }
    $taskLocalPython = Join-Path $taskProject 'venv/Scripts/python.exe'
    $taskAst = 'import sys;sys.dont_write_bytecode=True;import ast,base64;ast.parse(base64.b64decode(sys.argv[1]).decode());print("REMOTE_GATE_AST_OK")'
    & $taskLocalPython -I -B -c $taskAst $taskGateBase64
    if ($LASTEXITCODE -ne 0) { throw 'Tiny gate AST check failed' }
    & $taskLocalPython -I -B (Join-Path $taskProject $taskDriver) --local-check --request-base64 $taskRequestBase64
    if ($LASTEXITCODE -ne 0) { throw 'Driver local check failed' }
    [ordered]@{status='LOCAL_CHECK_ONLY_NOT_EXECUTED'; implementation_commit=$taskCommit;
        source_hash_count=$taskExpected.Count; driver_sha256=$taskExpected[$taskDriver]; test_file_count=14;
        final_command_characters=$taskCommand.Length; command_character_limit_exclusive=30000;
        powershell_ast_errors=$taskParseErrors.Count; base64_roundtrip=$true; shell_arguments_single_quoted=$true;
        server_output='/home/user/training_outputs/terminal/rotating_slots_perf_v1/final_pass_linux_tests_20261019b';
        remote_output_existence_checked=$false; server_executed=$false; pytest_executed=$false} | ConvertTo-Json
    exit 0
}
# Formal execution is authorized separately after primary review/source freeze.
$taskAttempt = Join-Path $PSScriptRoot ('local_launch_'+[Guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $taskAttempt
$taskStarted = [DateTime]::UtcNow.ToString('o')
$taskRequest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskAttempt 'source_request.json') -Encoding utf8
$taskArgv = @('ssh','lab-server',$taskCommand)
$taskArgv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskAttempt 'argv.json') -Encoding utf8
& ssh lab-server $taskCommand 1> (Join-Path $taskAttempt 'stdout.log') 2> (Join-Path $taskAttempt 'stderr.log')
$taskExit = $LASTEXITCODE
[ordered]@{argv=$taskArgv; start_utc=$taskStarted; end_utc=[DateTime]::UtcNow.ToString('o');
    numeric_exit_code=$taskExit; implementation_commit=$taskCommit; no_hard_timeout=$true} |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskAttempt 'attempt.json') -Encoding utf8
if ($null -ne $taskExit) { Set-Content -LiteralPath (Join-Path $taskAttempt 'numericexitcode.txt') -Value $taskExit }
Write-Output "Local SSH evidence: $taskAttempt"
Get-Content -LiteralPath (Join-Path $taskAttempt 'stdout.log')
Get-Content -LiteralPath (Join-Path $taskAttempt 'stderr.log')
exit $taskExit
