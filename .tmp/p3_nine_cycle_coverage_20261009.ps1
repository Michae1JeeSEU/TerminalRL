$ErrorActionPreference = 'Stop'

$items = @(
    [ordered]@{ group = 'old3'; path = 'D:\7_codespace\training_outputs\terminal\p3_v25_neutral_course_v2_20261008\train_flow6_mixed' },
    [ordered]@{ group = 'old3'; path = 'D:\7_codespace\training_outputs\terminal\p3_v25_neutral_course_v2_20261008\train_dense24_fixed' },
    [ordered]@{ group = 'old3'; path = 'D:\7_codespace\training_outputs\terminal\p3_v25_neutral_course_v2_20261008\train_dense24_mixed' },
    [ordered]@{ group = 'mid3'; path = 'D:\7_codespace\training_outputs\terminal\p3_v26_resources_20261009\train_01' },
    [ordered]@{ group = 'mid3'; path = 'D:\7_codespace\training_outputs\terminal\p3_v26_resources_20261009\train_02' },
    [ordered]@{ group = 'mid3'; path = 'D:\7_codespace\training_outputs\terminal\p3_v26_resources_20261009\train_03' },
    [ordered]@{ group = 'new3'; path = 'D:\7_codespace\training_outputs\terminal\p3_v26_supplemental_20261009\train_01' },
    [ordered]@{ group = 'new3'; path = 'D:\7_codespace\training_outputs\terminal\p3_v26_supplemental_20261009\train_02' },
    [ordered]@{ group = 'new3'; path = 'D:\7_codespace\training_outputs\terminal\p3_v26_supplemental_20261009\train_03' }
)

$records = @()
foreach ($item in $items) {
    $root = $item.path
    $resultPath = Join-Path $root 'result.json'
    $d = Get-Content -Raw $resultPath | ConvertFrom-Json
    $ds = $d.reconstruction.dataset_check

    $proposals = [ordered]@{}
    $proposalSum = 0
    foreach ($p in $ds.proposal_counts.PSObject.Properties) {
        $proposals[[string]$p.Name] = [int]$p.Value
        $proposalSum += [int]$p.Value
    }

    $interfaces = [ordered]@{}
    $interfaceR = 0
    $interfaceW = 0
    foreach ($p in $d.coverage.interface_gates.PSObject.Properties) {
        $r = 0
        $w = 0
        foreach ($q in $p.Value.PSObject.Properties) {
            if ([string]$q.Name -eq '1') { $r = [int]$q.Value }
            if ([string]$q.Name -eq '0') { $w = [int]$q.Value }
        }
        $interfaces[[string]$p.Name] = [ordered]@{ R = $r; W = $w }
        $interfaceR += $r
        $interfaceW += $w
    }

    $labelContext = [ordered]@{}
    foreach ($name in @('low_soc', 'busy_lz', 'after_retry', 'fast_inner')) {
        $prop = $d.coverage.contexts.label.PSObject.Properties | Where-Object { $_.Name -eq $name }
        $labelContext[$name] = if ($null -eq $prop) { $null } else { [int]$prop.Value }
    }

    $outcomes = [ordered]@{ N = 0; E = 0; F = 0; D = 0; U = [int]$d.rollout.unresolved }
    foreach ($p in $d.rollout.outcome_counts.PSObject.Properties) {
        switch ([string]$p.Name) {
            'completed' { $outcomes.N = [int]$p.Value }
            'emergency_landed' { $outcomes.E = [int]$p.Value }
            'failed_energy' { $outcomes.F = [int]$p.Value }
            'diverted_low_soc' { $outcomes.D = [int]$p.Value }
        }
    }

    $seedRoot = Join-Path $root ('seed_' + [string]$d.seed)
    $contextPath = Join-Path $seedRoot 'context_observations.jsonl'
    $neighborDistribution = [ordered]@{}
    $contextRows = 0
    $maskMismatches = 0
    if (Test-Path $contextPath) {
        foreach ($line in Get-Content -LiteralPath $contextPath) {
            $row = $line | ConvertFrom-Json
            $n = @($row.traffic).Count
            $maskN = @($row.traffic_mask).Where({ [bool]$_ }).Count
            if ($n -ne $maskN) { $maskMismatches++ }
            $contextRows++
            if (-not $neighborDistribution.Contains([string]$n)) { $neighborDistribution[[string]$n] = 0 }
            $neighborDistribution[[string]$n]++
        }
    }
    $neighborMax = if ($neighborDistribution.Count) {
        [int](($neighborDistribution.Keys | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum)
    } else { $null }

    $resourcePath = Join-Path $seedRoot 'resource_context_observations.jsonl'
    $resource = [ordered]@{
        status = 'MISSING'
        path = $resourcePath
        rows = $null
        observation_schema = $null
        original_labels_preserved = $null
        note = $null
    }
    if ($item.group -eq 'old3') {
        $resource.note = 'expected: old protocol stores105D; official loader rebuilds159 in memory from verified snapshot; nine-loader full check pending'
    }
    if ($null -ne $d.resource_reconstruction) {
        $resource.status = if (([int]$d.resource_reconstruction.rows -eq [int]$ds.valid_labels) -and [bool]$d.resource_reconstruction.original_labels_preserved) { 'PASS' } else { 'FAIL' }
        $resource.rows = [int]$d.resource_reconstruction.rows
        $resource.observation_schema = [string]$d.resource_reconstruction.observation_schema
        $resource.original_labels_preserved = [bool]$d.resource_reconstruction.original_labels_preserved
    } elseif (Test-Path $resourcePath) {
        $resource.status = 'PRESENT_UNPARSED'
        $resource.rows = (Get-Content -LiteralPath $resourcePath).Count
    }

    $rawRows = [int]$ds.raw_rows
    $actorRows = [int]$ds.actor_eligible_rows
    $forcedRows = [int]$ds.forced_wait_rows
    $labels = [int]$ds.valid_labels
    $okLabels = [int]$ds.annotation_status_counts.OK
    $releaseRows = [int]$ds.teacher_release_rows
    $waitRows = [int]$ds.teacher_wait_rows
    $outcomeTotal = ($outcomes.N + $outcomes.E + $outcomes.F + $outcomes.D + $outcomes.U)
    $coreChecks = [ordered]@{
        raw_rows = ($rawRows -eq ($actorRows + $forcedRows))
        labels = (($actorRows -eq $labels) -and ($labels -eq $okLabels))
        release_wait_actor = (($releaseRows + $waitRows) -eq $actorRows)
        proposal_release = ($proposalSum -eq $releaseRows)
        interface_release_wait = (($interfaceR -eq $releaseRows) -and ($interfaceW -eq $waitRows))
        context_rows = ($contextRows -eq $labels)
        outcomes = (($outcomeTotal -eq [int]$d.rollout.requested) -and [bool]$d.rollout.terminal -and (-not [bool]$d.rollout.censored) -and ($outcomes.U -eq 0))
        traffic_mask = ($maskMismatches -eq 0)
    }
    $coreIntegrity = if (($coreChecks.Values -contains $false)) { 'FAIL' } else { 'PASS' }

    $records += [ordered]@{
        seed = [int]$d.seed
        group = $item.group
        path = $root
        result_path = $resultPath
        protocol = [string]$d.protocol
        partition = [string]$d.partition
        schema = [string]$ds.schema
        status = [string]$d.status
        error = if ($null -eq $d.error) { $null } else { [string]$d.error }
        raw_rows = $rawRows
        actor_rows = $actorRows
        forced_wait_rows = $forcedRows
        valid_labels = $labels
        ok_labels = $okLabels
        resource = $resource
        release_R = $releaseRows
        wait_W = $waitRows
        proposal_counts = $proposals
        interface_gates_R_W = $interfaces
        label_context_counts = $labelContext
        actual_N_E_F_D_U = $outcomes
        rollout_requested = [int]$d.rollout.requested
        rollout_terminal = [bool]$d.rollout.terminal
        rollout_censored = [bool]$d.rollout.censored
        context_path = $contextPath
        context_rows = $contextRows
        neighbor_N_distribution = $neighborDistribution
        neighbor_N_max = $neighborMax
        traffic_mask_mismatches = $maskMismatches
        integrity = [ordered]@{ core = $coreIntegrity; checks = $coreChecks; resource = $resource.status }
    }
}

$aggregate = [ordered]@{}
foreach ($groupName in @('old6', 'new3', 'nine')) {
    $selected = switch ($groupName) {
        'old6' { @($records | Where-Object { $_.group -in @('old3', 'mid3') }) }
        'new3' { @($records | Where-Object { $_.group -eq 'new3' }) }
        'nine' { @($records) }
    }
    $proposal = [ordered]@{}
    $actor = 0; $labels = 0; $r = 0; $w = 0
    foreach ($record in $selected) {
        $actor += $record.actor_rows
        $labels += $record.valid_labels
        $r += $record.release_R
        $w += $record.wait_W
        foreach ($p in $record.proposal_counts.GetEnumerator()) {
            if (-not $proposal.Contains([string]$p.Key)) { $proposal[[string]$p.Key] = 0 }
            $proposal[[string]$p.Key] += [int]$p.Value
        }
    }
    $aggregate[$groupName] = [ordered]@{ seeds = @($selected.seed); actor_rows = $actor; valid_labels = $labels; R = $r; W = $w; R_plus_W = ($r + $w); proposal_counts = $proposal }
}

$leaveOneOut = @()
for ($i = 0; $i -lt $records.Count; $i++) {
    $held = $records[$i]
    $train = @{}
    for ($j = 0; $j -lt $records.Count; $j++) {
        if ($i -eq $j) { continue }
        foreach ($p in $records[$j].proposal_counts.GetEnumerator()) {
            if (-not $train.ContainsKey([string]$p.Key)) { $train[[string]$p.Key] = 0 }
            $train[[string]$p.Key] += [int]$p.Value
        }
    }
    $absent = @($held.proposal_counts.Keys | Where-Object { -not $train.ContainsKey($_) } | Sort-Object { [int]$_ })
    $rare = @($train.Keys | Where-Object { $train[$_] -ge 1 -and $train[$_] -le 5 } | Sort-Object { [int]$_ })
    $absentRows = 0; $rareRows = 0
    foreach ($k in $absent) { $absentRows += [int]$held.proposal_counts[$k] }
    foreach ($k in $rare) { if ($held.proposal_counts.Contains($k)) { $rareRows += [int]$held.proposal_counts[$k] } }
    $leaveOneOut += [ordered]@{
        held_seed = $held.seed
        train_seed_count = 8
        held_release_R = $held.release_R
        absent_classes_0 = @($absent)
        absent_held_release_rows = $absentRows
        rare_classes_1_to_5 = @($rare)
        rare_held_release_rows = $rareRows
    }
}

$witnessPath = 'D:\7_codespace\terminal\.tmp\p3_neighbor_witness_train_20261009\witnesses.jsonl'
$witnessStates = @{}
$witnessQueries = @{}
if (Test-Path $witnessPath) {
    foreach ($line in Get-Content -LiteralPath $witnessPath) {
        $row = $line | ConvertFrom-Json
        $key = [string]$row.seed
        if (-not $witnessStates.ContainsKey($key)) { $witnessStates[$key] = 0; $witnessQueries[$key] = 0 }
        $witnessStates[$key]++
        $witnessQueries[$key] += [int]$row.plan_calls
    }
}
$witnessCoverage = [ordered]@{
    path = $witnessPath
    states_by_seed = $witnessStates
    query_plan_calls_by_seed = $witnessQueries
    total_states = [int](($witnessStates.Values | Measure-Object -Sum).Sum)
    total_query_plan_calls = [int](($witnessQueries.Values | Measure-Object -Sum).Sum)
    covered_seeds = @($witnessStates.Keys | Sort-Object { [int]$_ } | ForEach-Object { [int]$_ })
    uncovered_new_seeds = @(893604, 893605, 893606)
}
foreach ($seed in @(893604, 893605, 893606)) {
    if (-not $witnessStates.ContainsKey([string]$seed)) { $witnessStates[[string]$seed] = 0 }
    if (-not $witnessQueries.ContainsKey([string]$seed)) { $witnessQueries[[string]$seed] = 0 }
}

$oldSixSeeds = @(893511, 893512, 893513, 893601, 893602, 893603)
$oldSixOriginalAbsent = 0
$oldSixOriginalRare = 0
$expandedSixAbsent = 0
$expandedSixRare = 0
foreach ($loo in $leaveOneOut) {
    if ($oldSixSeeds -contains [int]$loo.held_seed) {
        $expandedSixAbsent += [int]$loo.absent_held_release_rows
        $expandedSixRare += [int]$loo.rare_held_release_rows
    }
}
for ($i = 0; $i -lt $records.Count; $i++) {
    if (-not ($oldSixSeeds -contains [int]$records[$i].seed)) { continue }
    $held = $records[$i]
    $train = @{}
    for ($j = 0; $j -lt $records.Count; $j++) {
        if ($j -eq $i -or -not ($oldSixSeeds -contains [int]$records[$j].seed)) { continue }
        foreach ($p in $records[$j].proposal_counts.GetEnumerator()) {
            if (-not $train.ContainsKey([string]$p.Key)) { $train[[string]$p.Key] = 0 }
            $train[[string]$p.Key] += [int]$p.Value
        }
    }
    $absent = @($held.proposal_counts.Keys | Where-Object { -not $train.ContainsKey($_) })
    $rare = @($train.Keys | Where-Object { $train[$_] -ge 1 -and $train[$_] -le 5 })
    foreach ($k in $absent) { $oldSixOriginalAbsent += [int]$held.proposal_counts[$k] }
    foreach ($k in $rare) { if ($held.proposal_counts.Contains($k)) { $oldSixOriginalRare += [int]$held.proposal_counts[$k] } }
}
$allNineR = [int]$aggregate.nine.R
$oldSixR = [int]$aggregate.old6.R
$looComparison = [ordered]@{
    original_six = [ordered]@{
        held_seed_set = $oldSixSeeds
        denominator_R = $oldSixR
        absent_rows = $oldSixOriginalAbsent
        absent_percent = [math]::Round(100.0 * $oldSixOriginalAbsent / $oldSixR, 6)
        rare_rows = $oldSixOriginalRare
        rare_percent = [math]::Round(100.0 * $oldSixOriginalRare / $oldSixR, 6)
    }
    expanded_nine_train_on_original_six_held = [ordered]@{
        held_seed_set = $oldSixSeeds
        denominator_R = $oldSixR
        absent_rows = $expandedSixAbsent
        absent_percent = [math]::Round(100.0 * $expandedSixAbsent / $oldSixR, 6)
        rare_rows = $expandedSixRare
        rare_percent = [math]::Round(100.0 * $expandedSixRare / $oldSixR, 6)
    }
    all_nine = [ordered]@{
        held_seed_set = $records.seed
        denominator_R = $allNineR
        absent_rows = [int](($leaveOneOut.absent_held_release_rows | Measure-Object -Sum).Sum)
        absent_percent = [math]::Round(100.0 * (($leaveOneOut.absent_held_release_rows | Measure-Object -Sum).Sum) / $allNineR, 6)
        rare_rows = [int](($leaveOneOut.rare_held_release_rows | Measure-Object -Sum).Sum)
        rare_percent = [math]::Round(100.0 * (($leaveOneOut.rare_held_release_rows | Measure-Object -Sum).Sum) / $allNineR, 6)
    }
}
$witnessCoverage.states_by_seed = $witnessStates
$witnessCoverage.query_plan_calls_by_seed = $witnessQueries

$artifact = [ordered]@{
    generated_by = 'read-only aggregation; no training/query/reset'
    source_groups = [ordered]@{ old3 = 'p3_v25_neutral_course_v2_20261008'; mid3 = 'p3_v26_resources_20261009'; new3 = 'p3_v26_supplemental_20261009' }
    field_sources = [ordered]@{
        result = 'result.json: protocol, partition, status, reconstruction.dataset_check, coverage, rollout'
        resource = 'result.json: resource_reconstruction; old3 stores105D context and official loader rebuilds159 in memory from verified snapshots; resource JSONL path is explicitly missing'
        interfaces = 'result.json: coverage.interface_gates'
        label_context = 'result.json: coverage.contexts.label'
        neighbor = 'seed_<seed>/context_observations.jsonl: traffic row count; traffic_mask cross-check'
        leave_one_cycle_out = 'result.json: reconstruction.dataset_check.proposal_counts and teacher_release_rows'
        witness = '.tmp/p3_neighbor_witness_train_20261009/witnesses.jsonl: seed, one line per state, plan_calls'
    }
    records = $records
    aggregate = $aggregate
    leave_one_cycle_out = $leaveOneOut
    leave_one_cycle_out_comparison = $looComparison
    witness_coverage = $witnessCoverage
}

$outputPath = 'D:\7_codespace\terminal\.tmp\p3_nine_cycle_coverage_20261009.json'
$artifact | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $outputPath -Encoding utf8
Write-Output $outputPath
