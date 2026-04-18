#!/bin/bash

# ========================================================================
# -- bomb_repo_texas_justice_with_backup.sh --
# ========================================================================
# Forcefully overwrites a remote GitHub repository with the current
# working directory contents — BUT FIRST archives the overwritten
# state to a timestamped backup branch so you can always revert or
# inspect it later.
#
# Backup lives at: backup-pre-bomb-YYYYMMDD-HHMMSS
# ========================================================================

set -e  # exit immediately on any error

# ------------------------------------------------------------------------
# 1. dependency check
# ------------------------------------------------------------------------
if ! command -v gh &> /dev/null; then
    echo "error: github cli (gh) is not installed."
    echo "install it first. i can't interface with the api without it."
    exit 1
fi

# ------------------------------------------------------------------------
# 2. authentication
# ------------------------------------------------------------------------
if ! gh auth status &> /dev/null; then
    echo "authentication missing. logging you in now..."
    gh auth login
fi

# ------------------------------------------------------------------------
# 3. target acquisition
# ------------------------------------------------------------------------
target_repo=""
if [ -n "$1" ]; then
    target_repo="$1"
else
    echo "fetching your repository list..."
    repo_list=$(gh repo list --limit 50 --json name --jq '.[].name')
    if [ -z "$repo_list" ]; then
        echo "you don't have any repositories. create one first."
        exit 1
    fi
    echo "select the repository to bomb:"
    select repo in $repo_list; do
        if [ -n "$repo" ]; then
            username=$(gh api user --jq ".login")
            target_repo="https://github.com/$username/$repo.git"
            break
        else
            echo "invalid selection. try again."
        fi
    done
fi
echo "target acquired: $target_repo"

# ------------------------------------------------------------------------
# 4. the safety catch (texas justice — now with a safety net)
# ------------------------------------------------------------------------
echo ""
echo "WARNING: you are about to completely bomb:"
echo " -> $target_repo"
echo "with the contents of:"
echo " -> $(pwd)"
echo ""
echo "this is a destructive action on main. remote history WILL be erased from main."
echo "BUT the old state will be preserved in a backup branch."
read -p "are you sure you want to proceed? (yes/no): " confirmation
if [[ "$confirmation" != "yes" ]]; then
    echo "aborting. the prisoner lives another day."
    exit 0
fi

# ------------------------------------------------------------------------
# 5. execution — backup first, then bomb
# ------------------------------------------------------------------------
echo "executing bomb with backup..."

# initialize or re-initialize local git state
git init -q

# set up remote
git remote remove origin 2>/dev/null || true
git remote add origin "$target_repo"

# fetch whatever is currently on the remote so we can archive it
echo "fetching remote state for backup..."
git fetch origin 2>/dev/null || true

# determine remote branch
if git ls-remote --heads origin main | grep -q "refs/heads/main"; then
    REMOTE_BRANCH="main"
elif git ls-remote --heads origin master | grep -q "refs/heads/master"; then
    REMOTE_BRANCH="master"
else
    REMOTE_BRANCH=""
fi

# create timestamped backup branch
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
BACKUP_BRANCH="backup-pre-bomb-${TIMESTAMP}"

if [ -n "$REMOTE_BRANCH" ]; then
    echo "creating backup branch ${BACKUP_BRANCH} from remote ${REMOTE_BRANCH}..."
    git branch -f "$BACKUP_BRANCH" "origin/${REMOTE_BRANCH}"
    git push -u origin "$BACKUP_BRANCH" --force-with-lease
    echo "backup stored: ${BACKUP_BRANCH}"
else
    echo "no existing remote branch found (new/empty repo). no backup needed."
fi

# now perform the actual bomb
echo "staging current directory contents..."
git add .

echo "committing..."
git commit -m "force overwrite bomb: $(date)" || echo "nothing new to commit — proceeding anyway"

echo "standardizing branch to main..."
git branch -M main

echo "detonating the bomb: force push to main..."
git push -u origin main --force

echo ""
echo "done. repository bombed."
echo ""
echo "old state preserved at branch: ${BACKUP_BRANCH}"
echo "revert anytime with:"
echo "  git checkout ${BACKUP_BRANCH}"
echo "  git branch -M main"
echo "  git push -u origin main --force"
echo "or just browse the backup branch on GitHub." 
