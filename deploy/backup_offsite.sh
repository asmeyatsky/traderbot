#!/usr/bin/env bash
# ============================================================================
# Off-site backup sync
#
# Pushes the latest pg_dump from deploy/backups/ to s3://traderbotapp-backups/.
# Intended to run from cron after `deploy.sh backup`. See RUNBOOK.md § 5.
#
# Retention is governed by the bucket's lifecycle policy (30 days), NOT by
# this script — don't add local deletions here; `deploy.sh backup` already
# prunes local copies to the last 7.
#
# Exits non-zero on any failure so cron's mailer / log aggregator picks it up.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/backups"
BUCKET="${TRADERBOT_BACKUP_BUCKET:-traderbotapp-backups}"
REGION="${AWS_REGION:-eu-west-2}"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: awscli not installed on this host — install it or remove the cron entry." >&2
  exit 1
fi

if [ ! -d "$BACKUP_DIR" ]; then
  echo "ERROR: $BACKUP_DIR does not exist. Run 'bash deploy.sh backup' first." >&2
  exit 1
fi

LATEST=$(ls -t "${BACKUP_DIR}"/traderbot_*.sql.gz 2>/dev/null | head -1 || true)
if [ -z "$LATEST" ]; then
  echo "ERROR: no backup files found in ${BACKUP_DIR}." >&2
  exit 1
fi

FILENAME=$(basename "$LATEST")
DEST="s3://${BUCKET}/${FILENAME}"

echo "==> Uploading ${FILENAME} to ${DEST} (region=${REGION})"
aws s3 cp --region "$REGION" --storage-class STANDARD_IA "$LATEST" "$DEST"

# Smoke-test the object exists with the expected size.
LOCAL_SIZE=$(stat -c '%s' "$LATEST" 2>/dev/null || stat -f '%z' "$LATEST")
REMOTE_SIZE=$(aws s3api head-object --region "$REGION" --bucket "$BUCKET" --key "$FILENAME" --query 'ContentLength' --output text)

if [ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]; then
  echo "ERROR: size mismatch — local=${LOCAL_SIZE}, remote=${REMOTE_SIZE}" >&2
  exit 1
fi

echo "==> Off-site backup OK (${FILENAME}, ${LOCAL_SIZE} bytes)"
