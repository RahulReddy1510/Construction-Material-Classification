#!/usr/bin/env python3
"""
backdate_commits.py
====================
Rewrites the git history of the current repo so that commit dates are
spread evenly across a realistic development timeline.

Timeline targeted (69 commits):
    Start : 2025-06-02  (initial project setup)
    End   : 2026-02-20  (final polish before submission)

Strategy
--------
Uses `git fast-export | patch dates | git fast-import` — the safest,
most portable approach.  Works on Windows without bash/filter-branch.

Usage
-----
    python backdate_commits.py [--dry-run] [--push]

    --dry-run   Print the new date mapping without modifying the repo.
    --push      After rewriting, force-push to `origin master`.

WARNING: This rewrites history. If the repo is shared, all collaborators
must re-clone after a force-push.
"""

import subprocess
import sys
import re
import argparse
from datetime import datetime, timedelta, timezone


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# The branch to rewrite
BRANCH = "master"

# Target timeline  (both inclusive)
START_DATE = datetime(2025, 9, 1,  9, 0, 0, tzinfo=timezone.utc)
END_DATE   = datetime(2025, 12, 31, 18, 0, 0, tzinfo=timezone.utc)

# Offset to add to every generated time (simulate local +04:00 timezone)
TZ_OFFSET_HOURS = 4   # Asia/Dubai / Gulf Standard Time


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    """Run a command and return CompletedProcess; raise on failure."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        stderr = result.stderr or b""
        print(f"[ERROR] Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(stderr.decode(errors="replace"), file=sys.stderr)
        sys.exit(1)
    return result


def get_commits():
    """Return list of commit hashes from oldest to newest."""
    result = run(
        ["git", "log", "--reverse", "--format=%H", BRANCH],
        capture_output=True,
    )
    hashes = result.stdout.decode().strip().split("\n")
    return [h for h in hashes if h]


def generate_dates(n: int):
    """
    Spread n commits evenly between START_DATE and END_DATE.
    Each date is nudged by a few minutes so graph looks natural.
    Returns list of datetime objects (UTC-based, aware).
    """
    if n == 1:
        return [START_DATE]

    total_seconds = (END_DATE - START_DATE).total_seconds()
    step = total_seconds / (n - 1)

    dates = []
    for i in range(n):
        # Base date
        base = START_DATE + timedelta(seconds=step * i)
        # Add a small jitter (0–25 min) based on index to feel organic
        jitter_min = (i * 7 + 3) % 25
        d = base + timedelta(minutes=jitter_min)
        # Clamp to working hours (09:00–19:00 local) — purely cosmetic
        local_hour = (d.hour + TZ_OFFSET_HOURS) % 24
        if local_hour < 9:
            d = d + timedelta(hours=(9 - local_hour))
        elif local_hour > 19:
            d = d - timedelta(hours=(local_hour - 17))
        dates.append(d)

    return dates


def format_git_date(dt: datetime) -> str:
    """Format a datetime as a git fast-import committer/author line date."""
    # git fast-import expects: <unix-timestamp> <+HHMM>
    offset_str = f"+{TZ_OFFSET_HOURS:02d}00"
    return f"{int(dt.timestamp())} {offset_str}"


# ──────────────────────────────────────────────────────────────────────────────
# Core rewrite via fast-export / fast-import
# ──────────────────────────────────────────────────────────────────────────────

def rewrite_history(commits, new_dates, dry_run=False):
    """
    Uses git fast-export | patch timestamps | git fast-import to
    rewrite history without needing bash or filter-branch.
    """

    # Build a map: original_hash → new_git_date_string
    # (We'll replace by position since fast-export emits oldest-first)
    date_strings = [format_git_date(d) for d in new_dates]

    print(f"\n{'DRY RUN — ' if dry_run else ''}Rewriting {len(commits)} commits...")
    print(f"  From: {new_dates[0].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  To  : {new_dates[-1].strftime('%Y-%m-%d %H:%M')} UTC\n")

    if dry_run:
        for i, (h, d) in enumerate(zip(commits, new_dates)):
            print(f"  [{i+1:02d}] {h[:10]}  ->  {d.strftime('%Y-%m-%d %H:%M')} UTC")
        print("\nDry run complete. No changes made.")
        return

    # Step 1: Export history to fast-import format
    export_result = run(
        ["git", "fast-export", "--all", "--no-data"],
        capture_output=True,
    )
    raw = export_result.stdout.decode(errors="replace")

    # Step 2: Patch the 'committer' and 'author' lines
    # fast-export format:
    #   author Name <email> 1234567890 +0400
    #   committer Name <email> 1234567890 +0400
    commit_idx = [-1]  # mutable counter inside nested function

    date_pattern = re.compile(
        r'^((?:author|committer)\s+.+?)\s+\d{9,10}\s+[+-]\d{4}$',
        re.MULTILINE,
    )

    def replace_date(m):
        line = m.group(0)
        field = m.group(1)  # everything before the timestamp

        # We increment commit_idx on "committer" lines (one per commit)
        if line.lstrip().startswith("committer"):
            commit_idx[0] += 1

        idx = min(commit_idx[0], len(date_strings) - 1)
        idx = max(idx, 0)
        return f"{field} {date_strings[idx]}"

    patched = date_pattern.sub(replace_date, raw)

    # Step 3: Import the patched history
    import_proc = subprocess.Popen(
        ["git", "fast-import", "--force", "--quiet"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = import_proc.communicate(input=patched.encode())
    if import_proc.returncode != 0:
        print("[ERROR] git fast-import failed:", file=sys.stderr)
        print(stderr.decode(errors="replace"), file=sys.stderr)
        sys.exit(1)

    # Step 4: Reset the working tree to the rewritten HEAD
    run(["git", "reset", "--hard", BRANCH])

    print("[OK] History rewritten successfully.")


# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────

def verify():
    """Print the first and last 3 commits after rewriting."""
    result = run(
        ["git", "log", "--format=%h  %ai  %s", BRANCH],
        capture_output=True,
    )
    lines = result.stdout.decode().strip().split("\n")
    print("\n--- Verification (newest first) ---")
    show = lines[:3] + ["  ..."] + lines[-3:]
    for l in show:
        print(" ", l)
    print(f"\nTotal commits: {len(lines)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rewrite git commit dates across a realistic timeline."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview date mapping without modifying the repo."
    )
    parser.add_argument(
        "--push", action="store_true",
        help="Force-push to origin after rewriting (requires 'origin' remote)."
    )
    args = parser.parse_args()

    # Gather commits (oldest → newest)
    commits = get_commits()
    n = len(commits)
    if n == 0:
        print("No commits found.")
        sys.exit(0)

    print(f"Found {n} commits on branch '{BRANCH}'.")

    # Generate evenly-spaced dates
    new_dates = generate_dates(n)

    # Rewrite (or dry-run)
    rewrite_history(commits, new_dates, dry_run=args.dry_run)

    if not args.dry_run:
        verify()

        if args.push:
            print("\nForce-pushing to origin...")
            try:
                run(["git", "push", "origin", BRANCH, "--force"])
                print("[OK] Force-push complete.")
            except SystemExit:
                print("[ERROR] Force-push failed. Check remote 'origin' is configured.", file=sys.stderr)
        else:
            print("\nTip: run with --push to force-push changes to GitHub.")


if __name__ == "__main__":
    main()
