#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
REMOTE="${REMOTE:-origin}"
MESSAGE="${1:-"chore: sync local changes"}"

echo "Repository: $REPO_ROOT"
echo "Branch:     $BRANCH"
echo "Remote:     $REMOTE"
echo "Message:    $MESSAGE"
echo

git status --short
read -rp "Stage, commit, and push these changes? [y/N] " answer
[[ "\$answer" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

git add -A
git commit -m "$MESSAGE"
git push "$REMOTE" "$BRANCH"
