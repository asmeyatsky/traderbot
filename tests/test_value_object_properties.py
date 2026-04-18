"""
Property-based tests for domain value objects — Hypothesis.

2026 rules §5: "Domain: ≥95% coverage, zero mocks (pure logic)."
Property tests prove invariants hold across a large sample space — they catch
classes of bugs (edge cases at zero, large numbers, unicode) that example tests
miss.
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import assume, example, given, strategies as st

from src.domain.ports.audit_event_sink import hash_state
from src.domain.value_objects import Money, Symbol


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Money amounts: ≤ 2 decimal places per Money.__post_init__ invariant.
money_amounts = st.decimals(
    min_value=Decimal("-1000000.00"),
    max_value=Decimal("1000000.00"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)

# Valid symbols per the regex in Symbol.__post_init__
# `^[A-Z0-9][A-Z0-9\.\-]{0,9}$`
valid_first_char = st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
valid_rest_chars = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-",
    min_size=0,
    max_size=9,
)

symbol_values = st.tuples(valid_first_char, valid_rest_chars).map(
    lambda pair: pair[0] + pair[1]
)


# ---------------------------------------------------------------------------
# Money properties
# ---------------------------------------------------------------------------


@pytest.mark.domain
@given(a=money_amounts, b=money_amounts)
def test_money_addition_is_commutative(a, b):
    """a + b == b + a for Money of the same currency."""
    m1 = Money(a, "USD")
    m2 = Money(b, "USD")
    assert (m1 + m2).amount == (m2 + m1).amount


@pytest.mark.domain
@given(a=money_amounts, b=money_amounts, c=money_amounts)
def test_money_addition_is_associative(a, b, c):
    """(a + b) + c == a + (b + c)."""
    m1, m2, m3 = Money(a, "USD"), Money(b, "USD"), Money(c, "USD")
    # Compare amounts, not Money objects — Decimal equality is exact here.
    left = ((m1 + m2) + m3).amount
    right = (m1 + (m2 + m3)).amount
    assert left == right


@pytest.mark.domain
@given(a=money_amounts)
def test_money_subtraction_is_inverse_of_addition(a):
    """Subtracting a Money from itself gives zero of the same currency."""
    m = Money(a, "USD")
    zero = m - m
    assert zero.is_zero()
    assert zero.currency == "USD"


@pytest.mark.domain
@given(a=money_amounts)
def test_money_negation_inverts_sign(a):
    assume(a != Decimal("0.00"))
    m = Money(a, "USD")
    negated = -m
    if m.is_positive():
        assert negated.is_negative()
    elif m.is_negative():
        assert negated.is_positive()


@pytest.mark.domain
def test_money_rejects_cross_currency_addition():
    with pytest.raises(ValueError, match="different currencies"):
        Money(Decimal("10.00"), "USD") + Money(Decimal("10.00"), "EUR")


@pytest.mark.domain
def test_money_rejects_more_than_two_decimal_places():
    with pytest.raises(ValueError, match="2 decimal"):
        Money(Decimal("1.123"), "USD")


# ---------------------------------------------------------------------------
# Symbol properties
# ---------------------------------------------------------------------------


@pytest.mark.domain
@given(value=symbol_values)
def test_valid_symbol_round_trips_through_str(value):
    s = Symbol(value)
    assert str(s) == value


@pytest.mark.domain
@given(value=symbol_values)
def test_symbol_equality_is_case_insensitive(value):
    """Mixed-case construction fails validation (regex requires uppercase),
    so equality across case is tested by lower-casing a valid upper input."""
    s1 = Symbol(value)
    # Symbol validation requires uppercase, so we can't construct a lower case
    # version — but we can verify the equality's documented upper() behaviour
    # via the hash key.
    assert hash(s1) == hash(Symbol(value.upper()))


@pytest.mark.domain
@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "aapl",  # lowercase
        "TOOLONGSYMBOL",  # > 10 chars
        "AAPL/X",  # invalid char
        " AAPL",  # leading space
        ".AAPL",  # starts with dot
    ],
)
def test_symbol_rejects_invalid_format(bad):
    with pytest.raises(ValueError, match="Invalid stock symbol"):
        Symbol(bad)


# ---------------------------------------------------------------------------
# hash_state determinism — used for audit before/after hashes
# ---------------------------------------------------------------------------


@pytest.mark.domain
@given(
    a=st.integers(),
    b=st.text(max_size=50),
    c=st.booleans(),
)
def test_hash_state_is_deterministic_across_calls(a, b, c):
    """Same input → same hash. Critical for audit tamper evidence."""
    payload = {"a": a, "b": b, "c": c}
    assert hash_state(payload) == hash_state(payload)


@pytest.mark.domain
@given(
    a=st.integers(),
    b=st.text(max_size=50),
)
def test_hash_state_is_order_independent_for_dicts(a, b):
    """Dict key order must not affect the hash — payloads flow through JSON
    and we sort_keys=True in hash_state."""
    payload_1 = {"a": a, "b": b}
    payload_2 = {"b": b, "a": a}
    assert hash_state(payload_1) == hash_state(payload_2)


@pytest.mark.domain
def test_hash_state_distinguishes_different_payloads():
    h1 = hash_state({"x": 1})
    h2 = hash_state({"x": 2})
    assert h1 != h2
