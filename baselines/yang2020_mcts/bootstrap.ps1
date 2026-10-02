[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot 'upstream'),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Repository = 'https://github.com/xuxiyang1993/Multi_MCTS_Guidance_Separation_Assurance'
$Commit = '2c83aeab807daee6bab0561199116210d44839da'
$Archive = "$Repository/archive/$Commit.zip"
$PackageRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
$TempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\', '/')

function Resolve-AbsolutePath([string]$InputPath, [string]$BasePath) {
    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        return [System.IO.Path]::GetFullPath($InputPath)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $InputPath))
}

function Assert-ContainedPath([string]$Candidate, [string]$Parent, [string]$Label) {
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate).TrimEnd('\', '/')
    $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $prefix = $parentFull + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidateFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must be contained under $parentFull; received $candidateFull"
    }
    return $candidateFull
}

$Destination = Assert-ContainedPath (Resolve-AbsolutePath $Destination $PackageRoot) $PackageRoot 'Destination'
$TempRoot = Assert-ContainedPath (Join-Path $TempBase ("yang2020_mcts_" + [guid]::NewGuid().ToString('N'))) $TempBase 'TempRoot'
$ZipPath = Join-Path $TempRoot 'source.zip'
$ExtractRoot = Join-Path $TempRoot 'extract'

try {
    if ((Test-Path -LiteralPath $Destination) -and -not $Force) {
        Write-Output "upstream already exists; preserving the existing snapshot: $Destination"
        return
    }
    New-Item -ItemType Directory -Force -Path $TempRoot, $ExtractRoot | Out-Null
    Invoke-WebRequest -Uri $Archive -OutFile $ZipPath
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $ExtractRoot -Force
    $unpacked = Get-ChildItem -LiteralPath $ExtractRoot -Directory | Select-Object -First 1
    if (-not $unpacked) { throw 'GitHub archive contained no top-level directory.' }

    if (Test-Path -LiteralPath $Destination) {
        if ($Force) { Remove-Item -LiteralPath $Destination -Recurse -Force }
        else { throw "Destination exists: $Destination" }
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($name in @('MCTS', 'Simulators', 'README.md', 'requirements.txt', 'LICENSE', '.gitignore', '.gitattributes')) {
        $source = Join-Path $unpacked.FullName $name
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination $Destination -Recurse -Force
        }
    }
    @{
        repository = $Repository
        commit = $Commit
        archive = $Archive
        extracted = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Destination 'SOURCE_ARCHIVE.json') -Encoding UTF8
    Write-Output "Extracted $Commit to $Destination without nested Git metadata."
}
finally {
    if (Test-Path -LiteralPath $TempRoot) {
        $resolvedTemp = (Resolve-Path -LiteralPath $TempRoot).Path
        Assert-ContainedPath $resolvedTemp $TempBase 'TempRoot cleanup' | Out-Null
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
