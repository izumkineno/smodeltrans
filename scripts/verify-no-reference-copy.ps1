param(
    [string]$TargetRoot = "$PSScriptRoot/../src-tauri/src",
    [string]$ReferenceRoot = "$PSScriptRoot/../../PROJECT-TRNS/src/models",
    [string]$OutputPath = "$PSScriptRoot/../../.omc/state/smodeltrans-backend/ppocrv5/iteration-3/no-copy.json"
)

$ErrorActionPreference = "Stop"

function Get-NormalizedContentHash {
    param([string]$Path)

    $text = Get-Content -LiteralPath $Path -Raw
    $text = ($text -replace "`r`n", "`n") -replace "`r", "`n"
    $normalized = $text -split "`n" |
        ForEach-Object { ($_ -replace "//.*$", "").Trim() } |
        Where-Object { $_.Length -gt 0 } |
        ForEach-Object { $_ -replace "\s+", " " }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($normalized -join "`n"))
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return [Convert]::ToHexString($sha.ComputeHash($bytes)).ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Get-NormalizedBlocks {
    param([string]$Path)

    $text = Get-Content -LiteralPath $Path -Raw
    $text = ($text -replace "`r`n", "`n") -replace "`r", "`n"
    $lines = $text -split "`n" |
        ForEach-Object { ($_ -replace "//.*$", "").Trim() } |
        Where-Object { $_.Length -gt 0 } |
        ForEach-Object { $_ -replace "\s+", " " }
    $blocks = @{}
    for ($index = 0; $index -le ($lines.Count - 20); $index++) {
        $block = ($lines[$index..($index + 19)] -join "`n")
        $blocks[$block] = $true
    }
    return $blocks
}

$targetFiles = Get-ChildItem -LiteralPath $TargetRoot -Recurse -Filter *.rs -File
$referenceFiles = @()
if (Test-Path -LiteralPath "$ReferenceRoot/hy") {
    $referenceFiles += Get-ChildItem -LiteralPath "$ReferenceRoot/hy" -Recurse -Filter *.rs -File
}
if (Test-Path -LiteralPath "$ReferenceRoot/ppocrv5") {
    $referenceFiles += Get-ChildItem -LiteralPath "$ReferenceRoot/ppocrv5" -Recurse -Filter *.rs -File
}

$referenceHashes = @{}
$referenceBlocks = @{}
foreach ($file in $referenceFiles) {
    $referenceHashes[(Get-NormalizedContentHash $file.FullName)] = $file.FullName
    foreach ($block in (Get-NormalizedBlocks $file.FullName).Keys) {
        if (-not $referenceBlocks.ContainsKey($block)) {
            $referenceBlocks[$block] = $file.FullName
        }
    }
}

$findings = @()
foreach ($file in $targetFiles) {
    $hash = Get-NormalizedContentHash $file.FullName
    if ($referenceHashes.ContainsKey($hash)) {
        $findings += [pscustomobject]@{
            kind = "full_file_hash_match"
            target = $file.FullName
            reference = $referenceHashes[$hash]
        }
    }
    foreach ($block in (Get-NormalizedBlocks $file.FullName).Keys) {
        if ($referenceBlocks.ContainsKey($block)) {
            $findings += [pscustomobject]@{
                kind = "normalized_20_line_block_match"
                target = $file.FullName
                reference = $referenceBlocks[$block]
            }
        }
    }
}

$result = [pscustomobject]@{
    status = if ($findings.Count -eq 0) { "pass" } else { "fail" }
    targetRoot = (Resolve-Path -LiteralPath $TargetRoot).Path
    referenceRoot = if (Test-Path -LiteralPath $ReferenceRoot) { (Resolve-Path -LiteralPath $ReferenceRoot).Path } else { $ReferenceRoot }
    targetFileCount = $targetFiles.Count
    referenceFileCount = $referenceFiles.Count
    findings = $findings
}

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
if ($findings.Count -gt 0) {
    throw "Reference-copy findings detected; see $OutputPath"
}

Write-Output "no-copy verification passed: $($targetFiles.Count) target files checked against $($referenceFiles.Count) reference files"
