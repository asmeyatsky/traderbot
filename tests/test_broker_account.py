"""
Tests for the Broker Account Feature

Covers:
- Domain: BrokerAccount entity tests (immutability, validation, state transitions)
- Application: Use case tests with mocked repository
- Presentation: Router DTO serialization tests
"""
import sys
import types
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

# Mock requests module before importing broker_integration
if 'requests' not in sys.modules:
    requests_mock = types.ModuleType('requests')
    requests_mock.Session = MagicMock  # type: ignore[attr-defined]
    requests_mock.exceptions = types.ModuleType('requests.exceptions')  # type: ignore[assignment]
    requests_mock.exceptions.HTTPError = Exception  # type: ignore[attr-defined]
    requests_mock.exceptions.RequestException = Exception  # type: ignore[attr-defined]
    sys.modules['requests'] = requests_mock
    sys.modules['requests.exceptions'] = requests_mock.exceptions  # type: ignore[assignment]
    adapters_mod = types.ModuleType('requests.adapters')
    adapters_mod.HTTPAdapter = MagicMock  # type: ignore[attr-defined]
    sys.modules['requests.adapters'] = adapters_mod
    urllib3_mod = types.ModuleType('urllib3')
    urllib3_util = types.ModuleType('urllib3.util')
    urllib3_retry = types.ModuleType('urllib3.util.retry')
    urllib3_retry.Retry = MagicMock  # type: ignore[attr-defined]
    sys.modules['urllib3'] = urllib3_mod
    sys.modules['urllib3.util'] = urllib3_util
    sys.modules['urllib3.util.retry'] = urllib3_retry

from src.domain.entities.broker_account import BrokerAccount, BrokerType
from src.domain.ports.broker_account_repository_port import BrokerAccountRepositoryPort
from src.application.use_cases.broker_account import (
    LinkBrokerAccountUseCase,
    GetBrokerAccountsUseCase,
    UpdateBrokerSettingsUseCase,
    DeleteBrokerAccountUseCase,
)


# ============================================================================
# Domain Tests
# ============================================================================


class TestBrokerAccount:
    """Test BrokerAccount entity."""

    def _make_account(self, **overrides):
        defaults = dict(
            id="ba-1",
            user_id="user-1",
            broker_type=BrokerType.ALPACA,
            api_key="PK_TEST_1234",
            secret_key="SK_TEST_5678",
            paper_trading=True,
            label="My Alpaca",
            is_active=True,
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        defaults.update(overrides)
        return BrokerAccount(**defaults)

    def test_create_valid_account(self):
        account = self._make_account()
        assert account.id == "ba-1"
        assert account.broker_type == BrokerType.ALPACA
        assert account.paper_trading is True

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="API key must not be empty"):
            self._make_account(api_key="")

    def test_empty_secret_key_raises(self):
        with pytest.raises(ValueError, match="Secret key must not be empty"):
            self._make_account(secret_key="   ")

    def test_immutable(self):
        account = self._make_account()
        with pytest.raises(AttributeError):
            account.api_key = "new-key"  # type: ignore[misc]

    def test_switch_to_live(self):
        account = self._make_account(paper_trading=True)
        live = account.switch_to_live()
        assert live.paper_trading is False
        assert account.paper_trading is True  # original unchanged

    def test_switch_to_paper(self):
        account = self._make_account(paper_trading=False)
        paper = account.switch_to_paper()
        assert paper.paper_trading is True

    def test_deactivate(self):
        account = self._make_account(is_active=True)
        deactivated = account.deactivate()
        assert deactivated.is_active is False
        assert account.is_active is True

    def test_update_keys(self):
        account = self._make_account()
        updated = account.update_keys("NEW_KEY", "NEW_SECRET")
        assert updated.api_key == "NEW_KEY"
        assert updated.secret_key == "NEW_SECRET"
        assert account.api_key == "PK_TEST_1234"  # original unchanged


class TestBrokerType:
    """Test BrokerType enum."""

    def test_alpaca_value(self):
        assert BrokerType.ALPACA.value == "alpaca"

    def test_from_string(self):
        assert BrokerType("alpaca") == BrokerType.ALPACA

    def test_invalid_broker_raises(self):
        with pytest.raises(ValueError):
            BrokerType("invalid_broker")


# ============================================================================
# Use Case Tests
# ============================================================================


class TestLinkBrokerAccountUseCase:
    """Test LinkBrokerAccountUseCase."""

    def _make_repo(self):
        repo = Mock(spec=BrokerAccountRepositoryPort)
        repo.save = Mock(side_effect=lambda a: a)
        repo.get_by_user_and_broker = Mock(return_value=None)
        return repo

    def test_link_new_account(self):
        repo = self._make_repo()
        use_case = LinkBrokerAccountUseCase(broker_account_repository=repo)

        result = use_case.execute(
            user_id="user-1",
            broker_type="alpaca",
            api_key="PK123",
            secret_key="SK456",
            paper_trading=True,
            label="Test",
        )

        assert result.user_id == "user-1"
        assert result.broker_type == BrokerType.ALPACA
        assert result.api_key == "PK123"
        assert result.paper_trading is True
        repo.save.assert_called_once()

    def test_update_existing_account(self):
        repo = self._make_repo()
        existing = BrokerAccount(
            id="ba-existing",
            user_id="user-1",
            broker_type=BrokerType.ALPACA,
            api_key="OLD_KEY",
            secret_key="OLD_SECRET",
            paper_trading=True,
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        repo.get_by_user_and_broker.return_value = existing

        use_case = LinkBrokerAccountUseCase(broker_account_repository=repo)
        result = use_case.execute(
            user_id="user-1",
            broker_type="alpaca",
            api_key="NEW_KEY",
            secret_key="NEW_SECRET",
            paper_trading=False,
        )

        assert result.id == "ba-existing"
        assert result.api_key == "NEW_KEY"
        assert result.paper_trading is False
        repo.save.assert_called_once()

    def test_invalid_broker_type_raises(self):
        repo = self._make_repo()
        use_case = LinkBrokerAccountUseCase(broker_account_repository=repo)

        with pytest.raises(ValueError):
            use_case.execute(
                user_id="user-1",
                broker_type="invalid",
                api_key="key",
                secret_key="secret",
            )


class TestGetBrokerAccountsUseCase:
    """Test GetBrokerAccountsUseCase."""

    def test_returns_user_accounts(self):
        repo = Mock(spec=BrokerAccountRepositoryPort)
        account = BrokerAccount(
            id="ba-1",
            user_id="user-1",
            broker_type=BrokerType.ALPACA,
            api_key="key",
            secret_key="secret",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        repo.get_by_user.return_value = [account]

        use_case = GetBrokerAccountsUseCase(broker_account_repository=repo)
        result = use_case.execute("user-1")

        assert len(result) == 1
        assert result[0].id == "ba-1"


class TestUpdateBrokerSettingsUseCase:
    """Test UpdateBrokerSettingsUseCase."""

    def test_toggle_paper_trading(self):
        repo = Mock(spec=BrokerAccountRepositoryPort)
        account = BrokerAccount(
            id="ba-1",
            user_id="user-1",
            broker_type=BrokerType.ALPACA,
            api_key="key",
            secret_key="secret",
            paper_trading=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        repo.get_by_id.return_value = account
        repo.save = Mock(side_effect=lambda a: a)

        use_case = UpdateBrokerSettingsUseCase(broker_account_repository=repo)
        result = use_case.execute(account_id="ba-1", user_id="user-1", paper_trading=False)

        assert result is not None
        assert result.paper_trading is False
        repo.save.assert_called_once()

    def test_wrong_user_returns_none(self):
        repo = Mock(spec=BrokerAccountRepositoryPort)
        account = BrokerAccount(
            id="ba-1",
            user_id="user-1",
            broker_type=BrokerType.ALPACA,
            api_key="key",
            secret_key="secret",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        repo.get_by_id.return_value = account

        use_case = UpdateBrokerSettingsUseCase(broker_account_repository=repo)
        result = use_case.execute(account_id="ba-1", user_id="wrong-user")

        assert result is None

    def test_not_found_returns_none(self):
        repo = Mock(spec=BrokerAccountRepositoryPort)
        repo.get_by_id.return_value = None

        use_case = UpdateBrokerSettingsUseCase(broker_account_repository=repo)
        result = use_case.execute(account_id="nonexistent", user_id="user-1")

        assert result is None


class TestDeleteBrokerAccountUseCase:
    """Test DeleteBrokerAccountUseCase."""

    def test_delete_own_account(self):
        repo = Mock(spec=BrokerAccountRepositoryPort)
        account = BrokerAccount(
            id="ba-1",
            user_id="user-1",
            broker_type=BrokerType.ALPACA,
            api_key="key",
            secret_key="secret",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        repo.get_by_id.return_value = account
        repo.delete.return_value = True

        use_case = DeleteBrokerAccountUseCase(broker_account_repository=repo)
        result = use_case.execute(account_id="ba-1", user_id="user-1")

        assert result is True
        repo.delete.assert_called_once_with("ba-1")

    def test_delete_other_user_returns_false(self):
        repo = Mock(spec=BrokerAccountRepositoryPort)
        account = BrokerAccount(
            id="ba-1",
            user_id="user-1",
            broker_type=BrokerType.ALPACA,
            api_key="key",
            secret_key="secret",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        repo.get_by_id.return_value = account

        use_case = DeleteBrokerAccountUseCase(broker_account_repository=repo)
        result = use_case.execute(account_id="ba-1", user_id="other-user")

        assert result is False


# TestGetUserBrokerServiceUseCase removed — GetUserBrokerServiceUseCase deleted
# 2026-04 (Phase 4 burn-down). If a per-user broker factory returns, test it in
# tests/test_broker_factory.py alongside BrokerServiceFactory.


# ============================================================================
# Router DTO Tests
# ============================================================================


class TestBrokerAccountApiKeyMasking:
    """Test API key masking logic (inline, no router import needed)."""

    @staticmethod
    def _mask_key(api_key: str) -> str:
        """Replicate the masking logic from the router."""
        return f"****{api_key[-4:]}" if len(api_key) >= 4 else "****"

    def test_hides_full_api_key(self):
        hint = self._mask_key("PK_ABCDEFGHIJ1234")
        assert hint == "****1234"
        assert "PK_ABCDEFGHIJ" not in hint

    def test_short_key_masked(self):
        assert self._mask_key("PK") == "****"

    def test_exactly_4_chars(self):
        assert self._mask_key("ABCD") == "****ABCD"
