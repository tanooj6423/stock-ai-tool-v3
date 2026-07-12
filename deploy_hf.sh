#!/bin/bash
# Deploy the stock app to the Hugging Face Space.
# Pushes a clean single-commit snapshot (no history,
# no crypto/ binaries — HF rejects binary files in git).
# Usage:  ./deploy_hf.sh
set -e

cd "$(dirname "$0")"

# Safety: refuse to run with uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "You have uncommitted changes. Commit them first."
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)

git checkout --orphan hf-deploy
git reset -q

# Stage everything the Space needs
git add -A

# Strip what the Space must NOT get
git rm -r --cached -q crypto 2>/dev/null || true
git rm -r --cached -q .claude 2>/dev/null || true
git rm --cached -q deploy_hf.sh 2>/dev/null || true
git rm --cached -q .env 2>/dev/null || true

git commit -q -m "Deploy stock app snapshot $(date +%Y-%m-%d_%H:%M)"
git push hf hf-deploy:main --force

# Return to your normal branch and clean up
git checkout -f "$CURRENT_BRANCH"
git branch -D hf-deploy

echo ""
echo "✅ Pushed to the Space. It will rebuild in a few minutes:"
echo "   https://huggingface.co/spaces/Tanoo0233/stock-ai-tool-v3"
