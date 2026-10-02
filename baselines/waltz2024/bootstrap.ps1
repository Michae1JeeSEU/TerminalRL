[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$BaselineRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$RepositoryUrl = "https://github.com/MarWaltz/TUD_RL"

$Snapshots = @(
    [pscustomobject]@{
        Name = "upstream"
        Commit = "92a34ba703a5ec8c8bcad2503653dc12b971854b"
    },
    [pscustomobject]@{
        Name = "upstream_dev"
        Commit = "2395f35438571e274dcb233b4cdf4f7d692ff798"
    }
)

function Remove-OwnBootstrapTemp {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $resolved = [System.IO.Path]::GetFullPath($Path)
    $baseline = [System.IO.Path]::GetFullPath($BaselineRoot).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
    $requiredPrefix = $baseline + [System.IO.Path]::DirectorySeparatorChar
    $leaf = [System.IO.Path]::GetFileName($resolved)

    if ($resolved -eq $baseline -or
        -not $resolved.StartsWith($requiredPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $leaf.StartsWith(".bootstrap-tmp-", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove an unverified bootstrap path: $resolved"
    }

    Remove-Item -LiteralPath $resolved -Recurse -Force
}

foreach ($snapshot in $Snapshots) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $BaselineRoot $snapshot.Name))
    if (Test-Path -LiteralPath $target) {
        Write-Output "SKIP existing snapshot: $target"
        continue
    }

    $temp = [System.IO.Path]::GetFullPath((Join-Path $BaselineRoot (".bootstrap-tmp-{0}-{1}" -f $snapshot.Name, [guid]::NewGuid().ToString("N"))))
    New-Item -ItemType Directory -Path $temp | Out-Null

    try {
        $archive = Join-Path $temp "TUD_RL-$($snapshot.Commit).zip"
        $expanded = Join-Path $temp "expanded"
        $staging = Join-Path $temp "snapshot"
        $url = "$RepositoryUrl/archive/$($snapshot.Commit).zip"

        Write-Output "DOWNLOAD $snapshot.Name from $url"
        Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
        Expand-Archive -LiteralPath $archive -DestinationPath $expanded

        $roots = @(Get-ChildItem -LiteralPath $expanded -Directory -Force)
        if ($roots.Count -ne 1) {
            throw "Expected exactly one extracted archive root, found $($roots.Count)"
        }

        New-Item -ItemType Directory -Path $staging | Out-Null
        foreach ($item in @(Get-ChildItem -LiteralPath $roots[0].FullName -Force)) {
            Copy-Item -LiteralPath $item.FullName -Destination $staging -Recurse
        }

        $nestedGit = @(Get-ChildItem -LiteralPath $staging -Directory -Force -Filter ".git" -Recurse -ErrorAction SilentlyContinue)
        if ($nestedGit.Count -gt 0) {
            throw "Refusing a snapshot containing nested .git directories"
        }

        if (Test-Path -LiteralPath $target) {
            Write-Output "SKIP snapshot created concurrently: $target"
            continue
        }

        New-Item -ItemType Directory -Path $target | Out-Null
        foreach ($item in @(Get-ChildItem -LiteralPath $staging -Force)) {
            Copy-Item -LiteralPath $item.FullName -Destination $target -Recurse
        }
        Write-Output "READY $target at commit $($snapshot.Commit)"
    }
    finally {
        Remove-OwnBootstrapTemp -Path $temp
    }
}

$BlueSkyRepository = "TUDelft-CNS-ATM/bluesky"
$BlueSkyCommit = "f1626f99623c1ac20eadce0acaf930cb66842848"
$BlueSkyTreePrefix = "bluesky/resources/performance/OpenAP/"
$AssetRoot = [System.IO.Path]::GetFullPath((Join-Path $BaselineRoot "assets\openap_legacy"))
$AssetExpectedBytes = 165964
$AssetExpectedFiles = 23
$AssetRequiredPaths = @(
    "LICENSE",
    "bluesky/resources/performance/OpenAP/fixwing/engines.csv",
    "bluesky/resources/performance/OpenAP/fixwing/aircraft.json",
    "bluesky/resources/performance/OpenAP/fixwing/dragpolar.csv",
    "bluesky/resources/performance/OpenAP/rotor/aircraft.json",
    "bluesky/resources/performance/OpenAP/synonym.dat"
)

$AssetHeaders = @{
    "Accept" = "application/vnd.github+json"
    "User-Agent" = "waltz2024-bootstrap"
}
$treeUrl = "https://api.github.com/repos/$BlueSkyRepository/git/trees/$BlueSkyCommit`?recursive=1"
Write-Output "CHECK BlueSky asset tree $BlueSkyCommit"
$treeResponse = Invoke-RestMethod -Uri $treeUrl -Headers $AssetHeaders
if ($treeResponse.truncated) {
    throw "GitHub asset tree is truncated; refusing incomplete OpenAP acquisition"
}

$openapEntries = @($treeResponse.tree | Where-Object {
    $_.type -eq "blob" -and $_.path.StartsWith($BlueSkyTreePrefix, [System.StringComparison]::Ordinal)
})
$licenseEntries = @($treeResponse.tree | Where-Object {
    $_.type -eq "blob" -and $_.path -eq "LICENSE"
})
if ($openapEntries.Count -ne 22 -or $licenseEntries.Count -ne 1) {
    throw "Unexpected BlueSky asset tree: OpenAP blobs=$($openapEntries.Count), LICENSE blobs=$($licenseEntries.Count)"
}
$assetEntries = @($licenseEntries + $openapEntries)

$missingAssets = @()
foreach ($entry in $assetEntries) {
    $destination = [System.IO.Path]::GetFullPath((Join-Path $AssetRoot $entry.path))
    $assetPrefix = $AssetRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $destination.StartsWith($assetPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing an asset path outside the asset root: $($entry.path)"
    }
    if (Test-Path -LiteralPath $destination) {
        Write-Output "SKIP existing asset: $destination"
    }
    else {
        $missingAssets += [pscustomobject]@{ Entry = $entry; Destination = $destination }
    }
}

if ($missingAssets.Count -gt 0) {
    $assetTemp = [System.IO.Path]::GetFullPath((Join-Path $BaselineRoot (".bootstrap-tmp-assets-{0}" -f [guid]::NewGuid().ToString("N"))))
    New-Item -ItemType Directory -Path $assetTemp | Out-Null
    try {
        foreach ($missing in $missingAssets) {
            $entry = $missing.Entry
            $staged = [System.IO.Path]::GetFullPath((Join-Path $assetTemp $entry.path))
            $stagedParent = Split-Path -Parent $staged
            New-Item -ItemType Directory -Path $stagedParent -Force | Out-Null
            $rawUrl = "https://raw.githubusercontent.com/$BlueSkyRepository/$BlueSkyCommit/$($entry.path)"
            Write-Output "DOWNLOAD asset $($entry.path)"
            Invoke-WebRequest -Uri $rawUrl -Headers $AssetHeaders -OutFile $staged -UseBasicParsing
            if (-not (Test-Path -LiteralPath $staged)) {
                throw "Missing downloaded asset: $($entry.path)"
            }
        }

        foreach ($missing in $missingAssets) {
            $destination = $missing.Destination
            if (Test-Path -LiteralPath $destination) {
                Write-Output "SKIP asset created concurrently: $destination"
                continue
            }
            $destinationParent = Split-Path -Parent $destination
            New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
            $staged = Join-Path $assetTemp $missing.Entry.path
            Copy-Item -LiteralPath $staged -Destination $destination
            Write-Output "READY asset $destination"
        }
    }
    finally {
        Remove-OwnBootstrapTemp -Path $assetTemp
    }
}

if (-not (Test-Path -LiteralPath $AssetRoot)) {
    throw "OpenAP asset root was not created: $AssetRoot"
}
$assetFiles = @(Get-ChildItem -LiteralPath $AssetRoot -Recurse -File)
$assetBytes = [int64](($assetFiles | Measure-Object -Property Length -Sum).Sum)
if ($assetFiles.Count -ne $AssetExpectedFiles -or $assetBytes -ne $AssetExpectedBytes) {
    throw "Unexpected local OpenAP asset set: files=$($assetFiles.Count), bytes=$assetBytes"
}
foreach ($requiredPath in $AssetRequiredPaths) {
    $requiredFile = Join-Path $AssetRoot $requiredPath
    if (-not (Test-Path -LiteralPath $requiredFile)) {
        throw "Missing required OpenAP asset: $requiredPath"
    }
}
Write-Output "VALID OpenAP legacy assets: files=$($assetFiles.Count), bytes=$assetBytes, commit=$BlueSkyCommit"
Write-Output "Legacy asset LICENSE is copied verbatim; no license is inferred by this script."
Write-Output "No Python environment installation or runtime test is performed by this script."
