#!/usr/bin/env bash
# ============================================================================
# Deploy / Update TraderBot on EC2
#
# Usage (from the deploy/ directory):
#   bash deploy.sh          # Full rebuild and deploy
#   bash deploy.sh pull     # Pull latest images only (no rebuild)
#   bash deploy.sh migrate  # Run Alembic migrations only
#   bash deploy.sh logs     # Tail logs
#   bash deploy.sh down     # Stop everything
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="${SCRIPT_DIR}/.env.prod"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.prod.yml"
COMPOSE="docker compose -f ${COMPOSE_FILE} --env-file ${ENV_FILE}"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: ${ENV_FILE} not found. Copy .env.prod.example and fill in values."
  exit 1
fi

case "${1:-deploy}" in
  deploy)
    echo "==> Pulling latest code"
    cd "$SCRIPT_DIR/.." && git pull --ff-only
    cd "$SCRIPT_DIR"

    # Prune BEFORE the build. On a t4g.medium the pip install of the
    # full ML stack (~2GB of packages) needs ~6GB of working space;
    # accumulated layers from past deploys eat that fast. A single
    # disk-full build error wedges prod until an operator SSHes in.
    #
    # When the root filesystem is over 80% full we run an AGGRESSIVE
    # prune that removes every image not in use by a running container
    # and the full build cache — no age filter. Deploys run ~30s longer
    # because everything rebuilds from base, but the alternative is a
    # failed deploy that needs manual SSH intervention.
    echo "==> Pre-build cleanup"
    USED_PCT=$(df --output=pcent / | tail -1 | tr -d ' %')
    echo "    Disk usage before prune: ${USED_PCT}%"
    if [ "${USED_PCT:-0}" -gt 80 ]; then
      echo "    Disk over 80% — running AGGRESSIVE prune (no age filter)"
      docker image prune -af || true
      docker builder prune -af || true
    else
      echo "    Disk under 80% — running routine prune (7-day filter)"
      docker image prune -f
      docker image prune -af --filter "until=168h" || true
      docker builder prune -f --filter "until=168h" || true
    fi
    df -h / | tail -1

    echo "==> Building and starting containers"
    $COMPOSE build --pull
    $COMPOSE up -d --remove-orphans

    echo "==> Running database migrations"
    $COMPOSE exec -T api python -m alembic upgrade head

    echo "==> Post-build cleanup (dangling layers from this build)"
    docker image prune -f

    echo "==> Deployment complete. Checking health..."
    sleep 5
    $COMPOSE ps
    ;;

  migrate)
    echo "==> Running database migrations"
    $COMPOSE exec -T api python -m alembic upgrade head
    ;;

  logs)
    $COMPOSE logs -f --tail=100
    ;;

  down)
    echo "==> Stopping all containers"
    $COMPOSE down
    ;;

  restart)
    echo "==> Restarting api and frontend"
    $COMPOSE restart api frontend caddy
    ;;

  status)
    $COMPOSE ps
    echo ""
    echo "==> Disk usage:"
    df -h / | tail -1
    echo ""
    echo "==> Memory:"
    free -h
    ;;

  backup)
    echo "==> Backing up PostgreSQL database"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="${SCRIPT_DIR}/backups"
    mkdir -p "$BACKUP_DIR"
    $COMPOSE exec -T db pg_dump -U trading traderbot | gzip > "${BACKUP_DIR}/traderbot_${TIMESTAMP}.sql.gz"
    echo "==> Backup saved to ${BACKUP_DIR}/traderbot_${TIMESTAMP}.sql.gz"
    # Keep only last 7 backups
    ls -t "${BACKUP_DIR}"/traderbot_*.sql.gz 2>/dev/null | tail -n +8 | xargs -r rm
    ;;

  *)
    echo "Usage: $0 {deploy|migrate|logs|down|restart|status|backup}"
    exit 1
    ;;
esac
