#!/usr/bin/env python3
"""Recursively find and remove folders matching a name or wildcard pattern.

By default this runs as a DRY RUN: it only reports what *would* be deleted.
Pass --pull-the-trigger to actually delete the matched folders.

Examples
--------
Dry run (default) -- just list what matches:
    python remove_folders.py /path/to/root ida_femap695

Wildcard match:
    python remove_folders.py /path/to/root "AvgSA_*_msa"

Actually delete:
    python remove_folders.py /path/to/root ida_femap695 --pull-the-trigger
"""

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path


def find_matching_dirs(root: Path, pattern: str):
    """Return a list of directories under *root* whose name matches *pattern*.

    *pattern* may be a literal folder name or an fnmatch-style wildcard
    (e.g. ``AvgSA_*_msa``). Matching is done on the folder name only, not the
    full path.
    """
    matches = []
    # rglob("*") walks the whole tree; we keep only directories whose *name*
    # matches. We don't descend into a matched folder looking for more matches
    # (a match is deleted whole), so prune those from the walk.
    for path in root.rglob("*"):
        if path.is_dir() and fnmatch.fnmatch(path.name, pattern):
            # Skip if this dir is inside a folder we've already matched -- it
            # would be deleted anyway as part of the parent.
            if any(path != m and m in path.parents for m in matches):
                continue
            matches.append(path)
    return matches


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def simple_progress(current: int, total: int, width: int = 40):
    """Render a simple text progress bar to stdout (no dependencies)."""
    frac = current / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({frac * 100:5.1f}%)")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recursively remove folders matching a name or wildcard "
        "pattern. Runs as a dry run unless --pull-the-trigger is given.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="Path to the root folder to search under.")
    parser.add_argument(
        "pattern",
        help='Folder name or wildcard to match, e.g. "ida_femap695" or "AvgSA_*_msa". '
        "Quote wildcards so your shell does not expand them.",
    )
    parser.add_argument(
        "--pull-the-trigger",
        action="store_true",
        help="Actually delete the matched folders. Without this flag the "
        "script only reports what would be deleted (dry run).",
    )
    parser.add_argument(
        "--sizes",
        action="store_true",
        help="Compute and report the on-disk size of each match (slower).",
    )
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"root is not a directory: {root}")

    print(f"Root:    {root}")
    print(f"Pattern: {args.pattern}")
    print("Scanning...\n")

    matches = find_matching_dirs(root, args.pattern)

    if not matches:
        print("No matching folders found.")
        return 0

    # ---- Dry run (default) -------------------------------------------------
    if not args.pull_the_trigger:
        print(f"DRY RUN -- {len(matches)} folder(s) would be deleted:\n")
        total_size = 0
        for m in sorted(matches):
            if args.sizes:
                size = dir_size(m)
                total_size += size
                print(f"  {m}  ({human_size(size)})")
            else:
                print(f"  {m}")
        print()
        print(f"Summary: {len(matches)} folder(s) matched '{args.pattern}'.")
        if args.sizes:
            print(f"Total size that would be freed: {human_size(total_size)}")
        print("\nNothing was deleted. Re-run with --pull-the-trigger to delete.")
        return 0

    # ---- Execute -----------------------------------------------------------
    print(f"DELETING {len(matches)} folder(s)...\n")
    deleted = 0
    failed = 0
    for i, m in enumerate(sorted(matches), start=1):
        try:
            shutil.rmtree(m)
            deleted += 1
        except OSError as exc:
            failed += 1
            # Print on its own line so the progress bar stays readable.
            sys.stdout.write("\n")
            print(f"  FAILED to delete {m}: {exc}", file=sys.stderr)
        simple_progress(i, len(matches))

    print()
    print(f"Summary: deleted {deleted} folder(s).")
    if failed:
        print(f"         {failed} folder(s) could not be deleted (see errors above).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
