#!/usr/bin/env python3
"""
Publish analysis/outputs/ to the disposable `results` branch.

analysis/outputs/ is gitignored on main so that re-running a sweep does not
rewrite multi-MB evaluation_data.csv files into main's permanent history. The
results instead live on an orphan `results` branch, checked out in its own
worktree (default: ../bm4tc-results). This script mirrors the current contents
of analysis/outputs/ into that worktree and commits the snapshot.

The worktree MUST be separate from the main one. If `results` were checked out
in place, `git checkout main` would see analysis/outputs/** tracked in HEAD and
absent in the target, and delete your analysis results from disk.

Snapshot semantics: the sync uses `rsync --delete`, so each commit is a snapshot
of analysis/outputs/ as it stands right now, not an incremental log. Results
deleted locally also leave the branch tip (older snapshots stay reachable via
the branch's history until the branch itself is deleted).

Usage:
    python tools/publish_results.py [options]

Options:
    --push          Push to origin after committing (sets upstream on first push)
    --dry-run       Show what rsync would transfer, and the files that would be
                    committed, without writing anything
    --message TEXT  Commit message (default: "results: snapshot analysis outputs")
    --worktree PATH Path to the results worktree (default: ../bm4tc-results)
    --setup         Create the results branch + worktree if missing, seeding it
                    with a .gitignore and README

Examples:
    python tools/publish_results.py --dry-run
    python tools/publish_results.py
    python tools/publish_results.py --push -m "results: mnist r12 gibbs sweep"
    python tools/publish_results.py --setup
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SOURCE = PROJECT_ROOT / "analysis" / "outputs"
DEFAULT_WORKTREE = PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}-results"
BRANCH = "results"
DEFAULT_MESSAGE = "results: snapshot analysis outputs"

GITIGNORE = """\
# Results branch: generated analysis outputs only, mirroring analysis/outputs/
# in the main working tree. Disposable -- see README.md.

# HPO run outputs: the studies themselves are not worth keeping; the winning
# hyperparameters live in the seed_sweep configs on main.
analysis/outputs/**/hpo/

__pycache__/
*.pyc
.ipynb_checkpoints
"""


def run(cmd, cwd=None, check=True, capture=False):
    """Run a command, echoing it so the user can see what happened."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(
        cmd, cwd=cwd, check=check, text=True,
        capture_output=capture,
    )


def git_out(args, cwd=PROJECT_ROOT):
    """Capture stdout of a git command; empty string on failure."""
    res = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    return res.stdout.strip() if res.returncode == 0 else ""


def branch_exists():
    return bool(git_out(["rev-parse", "--verify", "--quiet", BRANCH]))


def worktree_for_branch():
    """Path of the worktree that has `results` checked out, or None."""
    listing = git_out(["worktree", "list", "--porcelain"])
    path = None
    for line in listing.splitlines():
        if line.startswith("worktree "):
            path = Path(line.split(" ", 1)[1])
        elif line == f"branch refs/heads/{BRANCH}":
            return path
    return None


def setup(worktree: Path):
    """Create the orphan branch + worktree and seed it with .gitignore/README."""
    if branch_exists():
        print(f"Branch '{BRANCH}' already exists.")
        existing = worktree_for_branch()
        if existing:
            print(f"Checked out at: {existing}")
            return existing
        print(f"Not checked out anywhere; adding worktree at {worktree}")
        run(["git", "worktree", "add", str(worktree), BRANCH], cwd=PROJECT_ROOT)
        return worktree

    print(f"Creating orphan branch '{BRANCH}' at {worktree}")
    run(
        ["git", "worktree", "add", "--orphan", "-b", BRANCH, str(worktree)],
        cwd=PROJECT_ROOT,
    )
    (worktree / ".gitignore").write_text(GITIGNORE)
    readme = worktree / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# bm4tc — `{BRANCH}` branch\n\n"
            "Generated analysis outputs, kept off `main` so that re-running a "
            "sweep does not rewrite multi-MB CSVs into permanent history.\n\n"
            "**This branch is disposable.** It is an orphan branch — it shares "
            "no history with `main` and holds no source code. Delete and "
            "recreate it whenever its history gets heavy; nothing on `main` "
            "depends on it.\n\n"
            "Refresh with `python tools/publish_results.py` from the main "
            "worktree.\n"
        )
    return worktree


def main():
    parser = argparse.ArgumentParser(
        description="Publish analysis/outputs/ to the disposable `results` branch.",
    )
    parser.add_argument("--push", action="store_true",
                        help="Push to origin after committing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be synced and committed, write nothing")
    parser.add_argument("-m", "--message", default=DEFAULT_MESSAGE,
                        help=f"Commit message (default: {DEFAULT_MESSAGE!r})")
    parser.add_argument("--worktree", type=Path, default=None,
                        help=f"Results worktree path (default: {DEFAULT_WORKTREE})")
    parser.add_argument("--setup", action="store_true",
                        help="Create the results branch + worktree if missing")
    args = parser.parse_args()

    if not SOURCE.is_dir():
        sys.exit(f"error: {SOURCE} does not exist — nothing to publish.")

    worktree = args.worktree or worktree_for_branch() or DEFAULT_WORKTREE

    if args.setup:
        worktree = setup(worktree)
    elif not (worktree / ".git").exists():
        sys.exit(
            f"error: no results worktree at {worktree}.\n"
            f"       Run with --setup to create it."
        )

    if worktree.resolve() == PROJECT_ROOT.resolve():
        sys.exit(
            "error: the results worktree must be separate from the main one.\n"
            "       Checking `results` out in place would make `git checkout main`\n"
            "       delete analysis/outputs/ from disk."
        )

    dest = worktree / "analysis" / "outputs"
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    print(f"Syncing {SOURCE}/ → {dest}/")
    rsync = ["rsync", "-a", "--delete", "--exclude", "hpo/"]
    if args.dry_run:
        # -n needs the destination to exist to report anything useful; if it
        # doesn't, rsync would list the whole tree as new, which is still the
        # honest answer for a first publish.
        rsync += ["-n", "-v"]
    rsync += [f"{SOURCE}/", f"{dest}/"]
    run(rsync)

    if args.dry_run:
        print("\n--dry-run: would then run, in the results worktree:")
        print(f"  $ git add -A")
        print(f"  $ git commit -m {args.message!r}")
        if args.push:
            print(f"  $ git push -u origin {BRANCH}")
        return

    run(["git", "add", "-A"], cwd=worktree)
    staged = git_out(["diff", "--cached", "--stat"], cwd=worktree)
    if not staged:
        print("Nothing changed since the last snapshot; no commit made.")
        return

    print(staged.splitlines()[-1])
    run(["git", "commit", "-m", args.message], cwd=worktree)

    if args.push:
        upstream = git_out(
            ["rev-parse", "--abbrev-ref", f"{BRANCH}@{{upstream}}"], cwd=worktree
        )
        push = ["git", "push"] if upstream else ["git", "push", "-u", "origin", BRANCH]
        run(push, cwd=worktree)
    else:
        print(f"\nNot pushed. Run with --push, or from {worktree}: git push -u origin {BRANCH}")


if __name__ == "__main__":
    main()
