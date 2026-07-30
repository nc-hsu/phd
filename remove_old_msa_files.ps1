<#
.SYNOPSIS
    Remove the MSA files orphaned by the run_batch_msa_buildings refactor from the
    per-site analysis folders.

.DESCRIPTION
    The MSA pipeline was simplified from
        run_batch_msa_sites -> run_msa_site -> run_batch_msa_per_stripe_record
    to
        run_batch_msa_buildings -> run_batch_msa_per_record -> run_msa_per_record

    This leaves two now-unused run scripts behind in every site folder:
        run_msa_site.py
        run_batch_msa_per_stripe_record.py

    The still-current files (run_batch_msa_per_record.py, run_msa_per_record.py,
    injection_functions.py, msa_process_recorders.py, config_msa_*.py) are NOT
    touched -- re-running notebook 050 overwrites them.

    Optionally (-IncludeOldStripeFolders) it also removes old positionally-indexed
    stripe result folders (stripe_1, stripe_2, ...). Stripe folders are now keyed by
    intensity tag (stripe_00pt360, ...), so the integer-named ones are stale RESULT
    folders from a pre-refactor run. This is destructive to those results and is
    off by default.

.PARAMETER Root
    The parent folder whose subtree holds the per-site mdof/ analysis folders.
    Default: D:\07_wp1_casestudy_sites

.PARAMETER Execute
    Actually delete. Without this switch the script runs as a DRY RUN and only
    prints what it would delete.

.PARAMETER IncludeOldStripeFolders
    Also target old integer-named stripe result folders (stripe_<int>). Destructive
    to any results in them. Still honours -Execute / dry run.

.EXAMPLE
    # dry run (default) -- prints every folder and the files it would delete
    .\remove_old_msa_files.ps1

.EXAMPLE
    # actually delete the stale run scripts
    .\remove_old_msa_files.ps1 -Execute

.EXAMPLE
    # also remove old positional stripe result folders, dry run first
    .\remove_old_msa_files.ps1 -IncludeOldStripeFolders
    .\remove_old_msa_files.ps1 -IncludeOldStripeFolders -Execute
#>
[CmdletBinding()]
param(
    [string] $Root = 'D:\07_wp1_casestudy_sites',
    [switch] $Execute,
    [switch] $IncludeOldStripeFolders
)

# Files the refactor made superfluous. Add names here if more turn up.
$StaleFileNames = @(
    'run_msa_site.py',
    'run_batch_msa_per_stripe_record.py'
)

# Old positional stripe RESULT folders look like "stripe_1"; the new scheme is
# "stripe_00pt360" (always contains a non-digit), so this regex never matches new ones.
$OldStripeFolderPattern = '^stripe_\d+$'

if (-not (Test-Path -LiteralPath $Root)) {
    Write-Error "Root folder not found: $Root"
    exit 1
}

$mode = if ($Execute) { 'EXECUTE' } else { 'DRY RUN' }
Write-Host ''
Write-Host "==== Remove old MSA files [$mode] ====" -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host "Targeting files: $($StaleFileNames -join ', ')"
if ($IncludeOldStripeFolders) {
    Write-Host "Also targeting old stripe folders matching /$OldStripeFolderPattern/" -ForegroundColor Yellow
}
Write-Host ''

# --- collect the stale files, grouped by their containing folder -------------
$staleFiles = Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $StaleFileNames -contains $_.Name }

# --- collect old stripe folders (optional) -----------------------------------
$staleDirs = @()
if ($IncludeOldStripeFolders) {
    $staleDirs = Get-ChildItem -LiteralPath $Root -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match $OldStripeFolderPattern }
}

$fileCount = 0
$dirCount  = 0

# --- report + (optionally) delete files, grouped by folder -------------------
$byFolder = $staleFiles | Group-Object DirectoryName | Sort-Object Name
foreach ($group in $byFolder) {
    Write-Host $group.Name -ForegroundColor Green
    foreach ($file in ($group.Group | Sort-Object Name)) {
        Write-Host "    - $($file.Name)"
        $fileCount++
        if ($Execute) {
            try {
                Remove-Item -LiteralPath $file.FullName -Force -ErrorAction Stop
            } catch {
                Write-Warning "      failed to delete $($file.FullName): $($_.Exception.Message)"
            }
        }
    }
}

# --- report + (optionally) delete old stripe folders -------------------------
if ($IncludeOldStripeFolders -and $staleDirs) {
    Write-Host ''
    Write-Host '---- old stripe result folders ----' -ForegroundColor Yellow
    foreach ($dir in ($staleDirs | Sort-Object FullName)) {
        Write-Host "    - $($dir.FullName)"
        $dirCount++
        if ($Execute) {
            try {
                Remove-Item -LiteralPath $dir.FullName -Recurse -Force -ErrorAction Stop
            } catch {
                Write-Warning "      failed to delete $($dir.FullName): $($_.Exception.Message)"
            }
        }
    }
}

# --- summary -----------------------------------------------------------------
Write-Host ''
$verb = if ($Execute) { 'Deleted' } else { 'Would delete' }
Write-Host "$verb $fileCount file(s)$(if ($IncludeOldStripeFolders) { " and $dirCount old stripe folder(s)" })." -ForegroundColor Cyan
if (-not $Execute) {
    Write-Host 'Dry run only -- re-run with -Execute to actually delete.' -ForegroundColor Yellow
}
Write-Host ''
