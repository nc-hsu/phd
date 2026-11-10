<#
.SYNOPSIS
    Delete the AvgSA(0.3) MSA result folders from every structure analysis folder,
    leaving the structure folders themselves (models, run scripts, configs) intact.

.DESCRIPTION
    The disaggregation was redone, so the MSA runs conditioned on AvgSA_03 need to be
    regenerated. This script removes ONLY the result folders whose name is exactly one of

        msa_AvgSA_03__opt_ks_mu_sigma
        msa_AvgSA_03__ks
        AvgSA_03__opt_ks_mu_sigma_stripes
        AvgSA_03__ks_stripes

    together with everything inside them (the stripe_* folders and their recorder
    output). Nothing else in a structure folder is touched.

    How the targets are found:
      * The whole tree under -Root is searched for DIRECTORIES (never files) whose
        name matches one of the target names exactly, case-insensitively. Exact
        matching means similarly named folders (e.g. msa_AvgSA_03__ks_old, or a
        different IM such as msa_AvgSA_05__ks) are never caught.
      * If a target folder sits inside another target folder, only the outer one is
        listed: deleting the outer folder removes the inner one anyway, and listing
        both would make the inner delete fail with "path not found".

    Safety:
      * DRY RUN by default. Without -Execute the script only prints what it would
        delete, with the number of stripe_* subfolders and the size of each target,
        so you can check the list before anything is removed.
      * With -Execute you are asked to type YES before deletion starts, unless
        -Force is also given.

    Long paths:
      Stripe output folders can be deeply nested and exceed the classic 260-character
      Windows path limit, which makes a plain Remove-Item fail. Each delete is first
      tried normally; on failure it is retried through the extended-length path form
      (\\?\D:\...), which bypasses that limit.

.PARAMETER Root
    Parent folder whose subtree holds the per-site / per-structure analysis folders.
    Default: D:\07_wp1_casestudy_sites

.PARAMETER Execute
    Actually delete. Without this switch nothing is deleted (dry run).

.PARAMETER Force
    Skip the "type YES" confirmation when -Execute is given.

.EXAMPLE
    # dry run (default): lists every folder that would be deleted
    .\remove_avgsa03_msa_results.ps1

.EXAMPLE
    # delete, with a confirmation prompt
    .\remove_avgsa03_msa_results.ps1 -Execute

.EXAMPLE
    # a different root folder, dry run
    .\remove_avgsa03_msa_results.ps1 -Root 'E:\some_other_copy'
#>
[CmdletBinding()]
param(
    [string] $Root = 'D:\07_wp1_casestudy_sites',
    [switch] $Execute,
    [switch] $Force
)

# Exact folder names to remove. Add names here if other result folders need the
# same treatment; matching is exact, so each entry must be the full folder name.
$TargetFolderNames = @(
    'msa_AvgSA_03__opt_ks_mu_sigma',
    'msa_AvgSA_03__ks',
    'AvgSA_03__opt_ks_mu_sigma_stripes',
    'AvgSA_03__ks_stripes'
)

# --- check the root exists before doing anything -----------------------------
if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    Write-Error "Root folder not found: $Root"
    exit 1
}

$mode = if ($Execute) { 'EXECUTE' } else { 'DRY RUN' }
Write-Host ''
Write-Host "==== Remove AvgSA_03 MSA result folders [$mode] ====" -ForegroundColor Cyan
Write-Host "Root:      $Root"
Write-Host "Targeting: $($TargetFolderNames -join ', ')"
Write-Host ''
Write-Host 'Searching (this can take a while on a large tree)...'

# --- collect candidate folders -----------------------------------------------
# -Directory restricts the search to folders, so a FILE that happens to share a
# target name can never be selected. -contains on a string array is an exact,
# case-insensitive comparison, which is what we want for Windows folder names.
$candidates = Get-ChildItem -LiteralPath $Root -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $TargetFolderNames -contains $_.Name } |
    Sort-Object FullName

# --- drop targets nested inside another target --------------------------------
# After sorting by full path, a nested folder always comes after its parent. Keep a
# folder only if its path does not start with an already-kept folder's path plus a
# separator (the separator stops "...\msa_AvgSA_03__ks" from matching the unrelated
# sibling "...\msa_AvgSA_03__ks_old").
$targets = New-Object System.Collections.Generic.List[System.IO.DirectoryInfo]
foreach ($dir in $candidates) {
    $isNested = $false
    foreach ($kept in $targets) {
        if ($dir.FullName.StartsWith($kept.FullName + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            $isNested = $true
            break
        }
    }
    if (-not $isNested) { $targets.Add($dir) }
}

if ($targets.Count -eq 0) {
    Write-Host ''
    Write-Host 'No matching folders found. Nothing to do.' -ForegroundColor Yellow
    exit 0
}

# --- report what will be deleted ---------------------------------------------
# For each target show the parent (structure) folder, the target name, how many
# stripe_* subfolders it holds, and its total size. The stripe count is a quick
# sanity check that this really is an MSA result folder.
$totalBytes = 0
Write-Host ''
$byParent = $targets | Group-Object { $_.Parent.FullName } | Sort-Object Name
foreach ($group in $byParent) {
    Write-Host $group.Name -ForegroundColor Green
    foreach ($dir in $group.Group) {
        $stripeCount = @(Get-ChildItem -LiteralPath $dir.FullName -Directory -Filter 'stripe_*' -ErrorAction SilentlyContinue).Count
        $bytes = (Get-ChildItem -LiteralPath $dir.FullName -Recurse -File -Force -ErrorAction SilentlyContinue |
                  Measure-Object -Property Length -Sum).Sum
        if ($null -eq $bytes) { $bytes = 0 }
        $totalBytes += $bytes
        Write-Host ("    - {0}   ({1} stripe folders, {2:N1} MB)" -f $dir.Name, $stripeCount, ($bytes / 1MB))
    }
}

Write-Host ''
Write-Host ("Found {0} folder(s) in {1} structure folder(s), {2:N2} GB in total." -f `
    $targets.Count, $byParent.Count, ($totalBytes / 1GB)) -ForegroundColor Cyan

# --- dry run stops here --------------------------------------------------------
if (-not $Execute) {
    Write-Host 'Dry run only -- re-run with -Execute to actually delete.' -ForegroundColor Yellow
    Write-Host ''
    exit 0
}

# --- confirmation before an irreversible delete --------------------------------
# Remove-Item bypasses the Recycle Bin, so there is no undo. Require an explicit
# YES unless the caller passed -Force.
if (-not $Force) {
    Write-Host ''
    $answer = Read-Host "Permanently delete these $($targets.Count) folder(s)? Type YES to continue"
    if ($answer -cne 'YES') {
        Write-Host 'Aborted. Nothing was deleted.' -ForegroundColor Yellow
        exit 0
    }
}

# --- delete --------------------------------------------------------------------
# Try a normal recursive delete first. If it fails (typically a path over 260
# characters), retry through the \\?\ extended-length prefix, which tells Windows
# to skip its path-length check.
$deleted = 0
$failed  = New-Object System.Collections.Generic.List[string]

foreach ($dir in $targets) {
    try {
        Remove-Item -LiteralPath $dir.FullName -Recurse -Force -ErrorAction Stop
        $deleted++
        Write-Host "deleted  $($dir.FullName)"
    } catch {
        $longPath = '\\?\' + $dir.FullName
        try {
            Remove-Item -LiteralPath $longPath -Recurse -Force -ErrorAction Stop
            $deleted++
            Write-Host "deleted  $($dir.FullName)  (long-path retry)"
        } catch {
            $failed.Add($dir.FullName)
            Write-Warning "FAILED   $($dir.FullName): $($_.Exception.Message)"
        }
    }
}

# --- summary -------------------------------------------------------------------
Write-Host ''
Write-Host "Deleted $deleted of $($targets.Count) folder(s)." -ForegroundColor Cyan
if ($failed.Count -gt 0) {
    Write-Host "$($failed.Count) folder(s) could not be deleted (often a file open in another program):" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    exit 2
}
Write-Host ''
