"""
Broker Account Management API Router

Endpoints for linking, updating, and deleting per-user brokerage accounts.
API keys are encrypted at rest; responses never expose raw secret keys.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import logging

from src.infrastructure.security import get_current_user
from src.application.use_cases.broker_account import (
    LinkBrokerAccountUseCase,
    GetBrokerAccountsUseCase,
    UpdateBrokerSettingsUseCase,
    DeleteBrokerAccountUseCase,
)
from src.presentation.api.dependencies import (
    get_link_broker_account_use_case,
    get_get_broker_accounts_use_case,
    get_update_broker_settings_use_case,
    get_delete_broker_account_use_case,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/broker-accounts", tags=["broker-accounts"])


# ── DTOs ────────────────────────────────────────────────────────────────────

class LinkBrokerRequest(BaseModel):
    broker_type: str = Field(..., description="Broker type, e.g. 'alpaca'")
    api_key: str = Field(..., min_length=1, description="Broker API key")
    secret_key: str = Field(..., min_length=1, description="Broker secret key")
    paper_trading: bool = Field(True, description="Use paper trading endpoint")
    label: Optional[str] = Field(None, description="Optional display label")


class UpdateBrokerRequest(BaseModel):
    paper_trading: Optional[bool] = None
    is_active: Optional[bool] = None


class BrokerAccountResponse(BaseModel):
    id: str
    broker_type: str
    paper_trading: bool
    label: Optional[str]
    is_active: bool
    api_key_hint: str  # last 4 chars only
    created_at: str
    updated_at: str


def _to_response(account) -> BrokerAccountResponse:
    """Convert domain entity to response DTO. Never expose full keys."""
    return BrokerAccountResponse(
        id=account.id,
        broker_type=account.broker_type.value,
        paper_trading=account.paper_trading,
        label=account.label,
        is_active=account.is_active,
        api_key_hint=f"****{account.api_key[-4:]}" if len(account.api_key) >= 4 else "****",
        created_at=account.created_at.isoformat() if account.created_at else "",
        updated_at=account.updated_at.isoformat() if account.updated_at else "",
    )


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def link_broker_account(
    body: LinkBrokerRequest,
    current_user_id: str = Depends(get_current_user),
    use_case: LinkBrokerAccountUseCase = Depends(get_link_broker_account_use_case),
) -> BrokerAccountResponse:
    """Link or update a brokerage account with encrypted API keys."""
    try:
        account = use_case.execute(
            user_id=current_user_id,
            broker_type=body.broker_type,
            api_key=body.api_key,
            secret_key=body.secret_key,
            paper_trading=body.paper_trading,
            label=body.label,
        )
        return _to_response(account)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error linking broker account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to link broker account",
        )


@router.get("")
async def list_broker_accounts(
    current_user_id: str = Depends(get_current_user),
    use_case: GetBrokerAccountsUseCase = Depends(get_get_broker_accounts_use_case),
) -> List[BrokerAccountResponse]:
    """List all linked broker accounts for the current user."""
    accounts = use_case.execute(current_user_id)
    return [_to_response(a) for a in accounts]


@router.patch("/{account_id}")
async def update_broker_settings(
    account_id: str,
    body: UpdateBrokerRequest,
    current_user_id: str = Depends(get_current_user),
    use_case: UpdateBrokerSettingsUseCase = Depends(get_update_broker_settings_use_case),
) -> BrokerAccountResponse:
    """Update paper/live trading mode or deactivate a broker account."""
    account = use_case.execute(
        account_id=account_id,
        user_id=current_user_id,
        paper_trading=body.paper_trading,
        is_active=body.is_active,
    )
    if not account:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Broker account not found")
    return _to_response(account)


@router.delete("/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_broker_account(
    account_id: str,
    current_user_id: str = Depends(get_current_user),
    use_case: DeleteBrokerAccountUseCase = Depends(get_delete_broker_account_use_case),
):
    """Unlink a broker account and delete stored credentials."""
    deleted = use_case.execute(account_id=account_id, user_id=current_user_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Broker account not found")
