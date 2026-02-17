"""
Trading Scheduler

Runs autonomous trading cycle and order polling on a schedule using APScheduler.

Architectural Intent:
- Infrastructure concern only — no business logic
- Delegates all work to AutonomousTradingService
- Cron-based schedule: trading cycle during market hours, polling throughout the day
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

from src.application.services.autonomous_trading_service import AutonomousTradingService

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None
_lock = threading.Lock()


def start_scheduler(service: AutonomousTradingService, cycle_minutes: int = 15) -> None:
    """Start the background scheduler with trading and polling jobs."""
    global _scheduler

    with _lock:
        if _scheduler is not None:
            logger.warning("Scheduler already running")
            return

        _scheduler = BackgroundScheduler(timezone="US/Eastern")

        # Trading cycle: every N minutes, Mon-Fri 9:30-15:45 ET
        _scheduler.add_job(
            service.run_trading_cycle,
            "cron",
            day_of_week="mon-fri",
            hour="9-15",
            minute=f"*/{cycle_minutes}",
            id="autonomous_trading_cycle",
            name="Autonomous Trading Cycle",
            misfire_grace_time=300,
        )

        # Order status polling: every 5 minutes, all day Mon-Fri
        _scheduler.add_job(
            service.poll_pending_orders,
            "cron",
            day_of_week="mon-fri",
            minute="*/5",
            id="order_status_poll",
            name="Order Status Poll",
            misfire_grace_time=120,
        )

        _scheduler.start()
        logger.info(
            f"Trading scheduler started (cycle={cycle_minutes}min, poll=5min)"
        )


def stop_scheduler() -> None:
    """Shut down the scheduler gracefully."""
    global _scheduler
    with _lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            _scheduler = None
            logger.info("Trading scheduler stopped")
