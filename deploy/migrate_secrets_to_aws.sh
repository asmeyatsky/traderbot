#!/usr/bin/env bash
#
# Migrate production secrets from .env.prod on disk to AWS Secrets Manager.
# Documented in deploy/README.md §"Secrets Management — Migrating Off .env.prod".
#
# Usage:
#   ./migrate_secrets_to_aws.sh                        # dry-run (default; prints what would happen)
#   ./migrate_secrets_to_aws.sh --apply                # actually creates the secret
#   ./migrate_secrets_to_aws.sh --apply --force        # overwrite if secret already exists
#
# Dry-run is intentional: this script operates on live production state
# (AWS account + EC2 .env file). Require explicit --apply to mutate anything.
#
# Prereqs:
#   - aws CLI configured with IAM permissions: secretsmanager:CreateSecret,
#     secretsmanager:PutSecretValue, secretsmanager:DescribeSecret
#   - Run ON the EC2 host (or copy .env.prod locally and run there)
#
set -euo pipefail

ENV_FILE="${ENV_FILE:-/opt/traderbot/deploy/.env.prod}"
SECRET_NAME="${SECRET_NAME:-traderbot/prod}"
AWS_REGION="${AWS_REGION:-eu-west-2}"
APPLY=false
FORCE=false

# Keys that are NOT secrets and should stay in .env.prod (config, not creds).
NON_SECRET_KEYS=(
    "ENVIRONMENT"
    "ALLOWED_ORIGINS"
    "ALLOWED_HOSTS"
    "AWS_SECRETS_NAME"
    "AWS_REGION"
    "OTEL_EXPORTER_OTLP_ENDPOINT"
    "OTEL_SERVICE_NAME"
    "APP_VERSION"
    "ENABLE_LIVE_TRADING"
    "EMERGENCY_HALT"
    "CIRCUIT_BREAKER_VOLATILITY_THRESHOLD"
    "CIRCUIT_BREAKER_RESET_MINUTES"
    "CHAT_MODEL"
)

# ---------- parse args --------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=true ;;
        --force) FORCE=true ;;
        -h|--help)
            head -n 20 "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

# ---------- sanity ------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found." >&2
    exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: aws CLI not installed." >&2
    exit 1
fi

is_non_secret() {
    local key=$1
    for nk in "${NON_SECRET_KEYS[@]}"; do
        [[ "$key" == "$nk" ]] && return 0
    done
    return 1
}

# ---------- build JSON payload ------------------------------------------
echo "Reading $ENV_FILE..."

# We build JSON with jq so quoting survives special characters. Fall back to
# a hand-rolled builder if jq isn't installed (unlikely on EC2).
if command -v jq >/dev/null 2>&1; then
    JSON_PAYLOAD='{}'
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip blank lines and comments.
        [[ -z "${line// }" || "$line" == \#* ]] && continue
        # Parse KEY=VALUE, allowing = in VALUE.
        if [[ "$line" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            # Strip matching surrounding quotes if present.
            value="${value%\"}"; value="${value#\"}"
            value="${value%\'}"; value="${value#\'}"
            if is_non_secret "$key"; then
                continue
            fi
            JSON_PAYLOAD=$(jq --arg k "$key" --arg v "$value" '. + {($k): $v}' <<< "$JSON_PAYLOAD")
        fi
    done < "$ENV_FILE"
else
    echo "ERROR: jq not installed — install it or export JSON manually." >&2
    exit 1
fi

SECRET_COUNT=$(jq 'length' <<< "$JSON_PAYLOAD")
echo "Found $SECRET_COUNT secret key(s) to migrate."
echo

# Show key names only (never values) so the operator can eyeball them.
echo "Keys to be migrated to Secrets Manager (values hidden):"
jq -r 'keys[] | "  - " + .' <<< "$JSON_PAYLOAD"
echo

# ---------- dry-run branch ----------------------------------------------
if [[ "$APPLY" != "true" ]]; then
    echo "DRY RUN — no changes made."
    echo "Re-run with --apply to create/update the AWS Secrets Manager secret."
    exit 0
fi

# ---------- apply branch ------------------------------------------------
echo "Checking whether secret '$SECRET_NAME' already exists..."
if aws secretsmanager describe-secret \
       --region "$AWS_REGION" --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
    if [[ "$FORCE" != "true" ]]; then
        echo "ERROR: Secret '$SECRET_NAME' already exists. Re-run with --force to overwrite." >&2
        exit 1
    fi
    echo "Secret exists — writing a new version (force mode)..."
    aws secretsmanager put-secret-value \
        --region "$AWS_REGION" \
        --secret-id "$SECRET_NAME" \
        --secret-string "$JSON_PAYLOAD" >/dev/null
else
    echo "Secret does not exist — creating..."
    aws secretsmanager create-secret \
        --region "$AWS_REGION" \
        --name "$SECRET_NAME" \
        --description "TraderBot production secrets — migrated from .env.prod" \
        --secret-string "$JSON_PAYLOAD" >/dev/null
fi

echo
echo "✓ Secret '$SECRET_NAME' written in region '$AWS_REGION'."
echo
echo "Next steps (MANUAL — this script intentionally does not touch .env.prod):"
echo "  1. Attach an IAM policy to the EC2 instance role allowing"
echo "     secretsmanager:GetSecretValue on this secret's ARN."
echo "  2. Edit $ENV_FILE:"
echo "       - Add:  AWS_SECRETS_NAME=$SECRET_NAME"
echo "       - Remove the keys listed above (they're now in AWS SM)."
echo "       - Remove ALLOW_ENV_SECRETS=true if present."
echo "  3. Restart the stack:   cd \$(dirname \"$ENV_FILE\") && bash deploy.sh restart"
echo "  4. Verify:              docker logs traderbot-api | grep -i 'secrets manager' | tail"
