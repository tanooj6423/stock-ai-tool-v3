#!/bin/bash
# Deploy the stock app to the Hugging Face Space.
# Takes a snapshot of the CURRENT folder contents (no git
# history, no crypto/ binaries) and force-pushes it to the
# Space. Your main branch and GitHub are never touched.
# Usage:  ./deploy_hf.sh
set -e

cd "$(dirname "$0")"

# Clean up any stale lock from a crashed/interrupted git
rm -f .git/index.lock

CURRENT_BRANCH=$(git branch --show-current)

# Remove leftover temp branch from any previous failed run
git branch -D hf-deploy 2>/dev/null || true

git checkout --orphan hf-deploy
git reset -q

# Stage everything, then strip what the Space must NOT get
git add -A
git rm -r --cached -q crypto 2>/dev/null || true
git rm -r --cached -q .claude 2>/dev/null || true
git rm --cached -q deploy_hf.sh 2>/dev/null || true
git rm --cached -q .env 2>/dev/null || true
git rm -r --cached -q data 2>/dev/null || true

git commit -q -m "Deploy stock app snapshot $(date +%Y-%m-%d_%H:%M)"
git push hf hf-deploy:main --force

# Return to your normal branch exactly as it was
git checkout -f "$CURRENT_BRANCH"
git branch -D hf-deploy

echo ""
echo "✅ Pushed. The Space is rebuilding now (2-5 min):"
echo "   https://huggingface.co/spaces/Tanoo0233/stock-ai-tool-v3"
