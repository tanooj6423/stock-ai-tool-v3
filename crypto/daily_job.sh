#!/bin/zsh
# Daily crypto pipeline + deploy. Run from the hf-space worktree by launchd
# (com.tanooj.crypto-daily). Logs: ~/Library/Logs/crypto-daily.log
set -e
cd "$(dirname "$0")/.."          # repo root of whichever worktree holds this
PY=/opt/anaconda3/bin/python3

echo "=== crypto daily $(date -u '+%F %T') UTC ==="
$PY -m crypto.pipeline daily

git add -A crypto
if git diff --cached --quiet; then
    echo "no data changes to commit"
else
    git commit -m "daily refresh $(date -u +%F)"
fi
# push unconditionally: retries any commit whose push failed on a prior run
git push hf-crypto HEAD:main     # redeploys the Hugging Face Space
git push origin HEAD             # keep GitHub copy of this branch current
echo "=== done $(date -u '+%F %T') UTC ==="
