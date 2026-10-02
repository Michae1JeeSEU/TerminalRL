$ErrorActionPreference = 'Stop'

$Root = [System.IO.Path]::GetFullPath((Split-Path -Parent $MyInvocation.MyCommand.Path)).TrimEnd('\', '/')
$Upstream = [System.IO.Path]::GetFullPath((Join-Path $Root 'upstream'))
$Commit = '87a8e5ba7393d622768a7695770a4fa4e09c26bd'
$ArchiveUrl = "https://github.com/rrsaxena92/Vertiport_schedule/archive/$Commit.zip"

function Assert-ContainedPath([string]$Candidate, [string]$Parent, [string]$Label) {
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate).TrimEnd('\', '/')
    $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $prefix = $parentFull + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidateFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must be contained under $parentFull; received $candidateFull"
    }
    return $candidateFull
}

$Upstream = Assert-ContainedPath $Upstream $Root 'upstream destination'

if (Test-Path -LiteralPath $Upstream) {
    Write-Host "upstream already exists; preserving the existing snapshot: $Upstream"
    exit 0
}

$TempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$TempName = "saxena2024_schedule_" + [guid]::NewGuid().ToString('N')
$TempRoot = Join-Path $TempBase $TempName
$Archive = Join-Path $TempRoot 'source.zip'
$Extracted = Join-Path $TempRoot 'extract'
try {
    New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
    Invoke-WebRequest -Uri $ArchiveUrl -OutFile $Archive
    Expand-Archive -LiteralPath $Archive -DestinationPath $Extracted
    $Top = Get-ChildItem -LiteralPath $Extracted -Directory | Select-Object -First 1
    if ($null -eq $Top) {
        throw 'GitHub archive did not contain a top-level directory.'
    }
    $ExtractedFull = [System.IO.Path]::GetFullPath($Extracted)
    $SourceFull = Assert-ContainedPath $Top.FullName $ExtractedFull 'archive source'
    $DestinationFull = Assert-ContainedPath $Upstream $Root 'upstream destination'
    if (-not (Test-Path -LiteralPath $SourceFull -PathType Container)) {
        throw "Archive source is not a directory: $SourceFull"
    }
    if (Test-Path -LiteralPath $DestinationFull) {
        throw "Upstream destination appeared during bootstrap: $DestinationFull"
    }
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    Move-Item -LiteralPath $SourceFull -Destination $DestinationFull
    Write-Host "Downloaded official snapshot $Commit to $Upstream"
}
finally {
    if (Test-Path -LiteralPath $TempRoot) {
        $resolvedBase = ([System.IO.Path]::GetFullPath($TempBase)).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
        $resolvedTarget = [System.IO.Path]::GetFullPath($TempRoot)
        $insideTemp = $resolvedTarget.StartsWith($resolvedBase, [System.StringComparison]::OrdinalIgnoreCase)
        $expectedName = [System.IO.Path]::GetFileName($resolvedTarget) -eq $TempName
        if ($insideTemp -and $expectedName -and $resolvedTarget -ne $resolvedBase) {
            Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
        }
        else {
            Write-Warning "Refusing to remove unexpected temporary path: $resolvedTarget"
        }
    }
}
