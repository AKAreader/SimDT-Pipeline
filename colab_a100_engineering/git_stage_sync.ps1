<#
.SYNOPSIS
    SimDT Pipeline - Master Control One-Click GitHub Stage Sync Script (git_stage_sync.ps1)
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$WorkTree = "",

    [Parameter(Mandatory = $false)]
    [string]$GitDir = "C:\Users\Administrator\.gemini\antigravity\brain\611a8caf-a6f3-40d9-8d8a-4173bfcaca85\scratch\simdt_repo.git",

    [Parameter(Mandatory = $false)]
    [string]$CommitMessage = "chore(stage): sync SimDT pipeline stage deliverables",

    [Parameter(Mandatory = $false)]
    [string]$PhaseTag = "",

    [Parameter(Mandatory = $false)]
    [string]$RemoteUrl = "https://github.com/AKAreader/SimDT-Pipeline.git",

    [Parameter(Mandatory = $false)]
    [string]$Branch = "main",

    [Parameter(Mandatory = $false)]
    [int]$MaxFileSizeMB = 50,

    [Parameter(Mandatory = $false)]
    [switch]$Push
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($WorkTree)) {
    $WorkTree = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}

function Invoke-SimDTGit {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    & git --git-dir="$GitDir" --work-tree="$WorkTree" @GitArgs
}

# Remove broken .git folder on Z:\ if it is a directory so git uses --git-dir cleanly
$dotGitPath = Join-Path $WorkTree ".git"
if ((Test-Path -LiteralPath $dotGitPath) -and (Get-Item -LiteralPath $dotGitPath -Force).PSIsContainer) {
    Remove-Item -LiteralPath $dotGitPath -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath (Join-Path $GitDir "HEAD"))) {
    New-Item -ItemType Directory -Path $GitDir -Force | Out-Null
    & git init --bare --initial-branch=$Branch "$GitDir" | Out-Null
}
& git --git-dir="$GitDir" config core.bare false
& git --git-dir="$GitDir" config core.worktree "$WorkTree"
& git --git-dir="$GitDir" config core.autocrlf false

# Gatekeeper: block large files or model weights before staging
$blockedExts = @(".pth", ".pt", ".tar", ".ckpt", ".safetensors", ".bin", ".onnx", ".engine", ".zip", ".tgz", ".7z", ".rar")
Invoke-SimDTGit add -A

$stagedFiles = Invoke-SimDTGit diff --cached --name-only
foreach ($relPath in $stagedFiles) {
    if ([string]::IsNullOrWhiteSpace($relPath)) { continue }
    $ext = [System.IO.Path]::GetExtension($relPath).ToLowerInvariant()
    if ($blockedExts -contains $ext) {
        Invoke-SimDTGit reset HEAD -- "$relPath" | Out-Null
        throw "Blocked weight/archive file from staging: $relPath"
    }
    $fullPath = Join-Path $WorkTree $relPath
    if (Test-Path -LiteralPath $fullPath) {
        $sizeMB = (Get-Item -LiteralPath $fullPath).Length / 1MB
        if ($sizeMB -gt $MaxFileSizeMB) {
            Invoke-SimDTGit reset HEAD -- "$relPath" | Out-Null
            throw ("Blocked oversized file ({0:N2} MB > {1} MB): {2}" -f $sizeMB, $MaxFileSizeMB, $relPath)
        }
    }
}

$stagedAfterCheck = Invoke-SimDTGit diff --cached --name-only
if ($stagedAfterCheck) {
    Write-Host "[Git Sync] Committing staged files: $CommitMessage"
    Invoke-SimDTGit commit -m "$CommitMessage"
} else {
    Write-Host "[Git Sync] Working tree clean, no new commit needed."
}

Invoke-SimDTGit branch -M $Branch
$headSha = (Invoke-SimDTGit rev-parse HEAD).Trim()
Write-Host "[Git Sync] Current HEAD SHA: $headSha"

if (-not [string]::IsNullOrWhiteSpace($PhaseTag)) {
    Write-Host "[Git Sync] Tagging $PhaseTag -> $headSha"
    Invoke-SimDTGit tag -f -a "$PhaseTag" -m "Stage Tag: $PhaseTag ($CommitMessage)"
}

if (-not [string]::IsNullOrWhiteSpace($RemoteUrl)) {
    $existingRemotes = Invoke-SimDTGit remote
    if ($existingRemotes -contains "origin") {
        Invoke-SimDTGit remote set-url origin "$RemoteUrl"
    } else {
        Invoke-SimDTGit remote add origin "$RemoteUrl"
    }
}

if ($Push) {
    Write-Host "[Git Sync] Pushing branch $Branch and tags to $RemoteUrl ..."
    Invoke-SimDTGit push -u origin $Branch --force
    Invoke-SimDTGit push origin --tags --force
    Write-Host "[Git Sync] Push completed successfully!"
}
