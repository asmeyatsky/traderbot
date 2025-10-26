"""
API Documentation Enhancements

Provides comprehensive OpenAPI/Swagger documentation for the trading platform API.
Includes detailed descriptions, examples, and error responses for all endpoints.
"""
from typing import Dict, Any

# Order endpoints documentation
ORDER_ENDPOINTS_DOCS: Dict[str, Dict[str, Any]] = {
    "create_order": {
        "summary": "Create a new trading order",
        "description": """
        Creates a new trading order for the authenticated user. Validates order parameters
        against user risk limits and portfolio constraints before placement.

        The order is validated for:
        - Sufficient cash balance for the position
        - Position size constraints
        - Risk tolerance limits
        - Trading pause status

        Once validated, the order is either immediately executed (market orders) or
        placed for execution (limit orders, stop losses).
        """,
        "tags": ["Orders"],
        "responses": {
            201: {
                "description": "Order created successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "order_id": "order-123",
                            "status": "PENDING",
                            "symbol": "AAPL",
                            "order_type": "MARKET",
                            "position_type": "LONG",
                            "quantity": 100,
                            "price": 150.50,
                            "placed_at": "2025-10-26T10:30:00Z",
                        }
                    }
                }
            },
            400: {
                "description": "Invalid order parameters",
                "content": {
                    "application/json": {
                        "example": {
                            "error": "Validation failed",
                            "details": ["Insufficient cash balance", "Position size exceeds limit"]
                        }
                    }
                }
            },
            429: {
                "description": "Rate limit exceeded",
                "content": {
                    "application/json": {
                        "example": {
                            "error": "Too Many Requests",
                            "detail": "Rate limit exceeded: 100/minute"
                        }
                    }
                }
            },
        }
    },
    "get_order": {
        "summary": "Get order details",
        "description": """
        Retrieves detailed information about a specific order.

        Returns:
        - Order status and execution details
        - Fill information for partial fills
        - Commission and fees
        - Associated position (if applicable)
        """,
        "tags": ["Orders"],
        "responses": {
            200: {"description": "Order details retrieved successfully"},
            404: {"description": "Order not found"},
            401: {"description": "Unauthorized - invalid or missing token"},
        }
    },
    "get_user_orders": {
        "summary": "Get all orders for authenticated user",
        "description": """
        Retrieves all orders for the authenticated user with optional filtering.

        Supports filtering by:
        - Status (PENDING, EXECUTED, CANCELLED, PARTIALLY_FILLED)
        - Symbol
        - Date range
        - Order type

        Results are paginated and sorted by most recent first.
        """,
        "tags": ["Orders"],
        "parameters": [
            {
                "name": "status",
                "in": "query",
                "description": "Filter by order status",
                "schema": {
                    "type": "string",
                    "enum": ["PENDING", "EXECUTED", "CANCELLED", "PARTIALLY_FILLED"]
                }
            },
            {
                "name": "symbol",
                "in": "query",
                "description": "Filter by stock symbol",
                "schema": {"type": "string", "example": "AAPL"}
            },
            {
                "name": "skip",
                "in": "query",
                "description": "Number of items to skip for pagination",
                "schema": {"type": "integer", "default": 0}
            },
            {
                "name": "limit",
                "in": "query",
                "description": "Maximum number of items to return",
                "schema": {"type": "integer", "default": 50, "maximum": 500}
            },
        ]
    },
}

# Portfolio endpoints documentation
PORTFOLIO_ENDPOINTS_DOCS: Dict[str, Dict[str, Any]] = {
    "get_portfolio": {
        "summary": "Get user's portfolio",
        "description": """
        Retrieves comprehensive portfolio information including:
        - Total portfolio value and allocation
        - Cash balance and invested amount
        - Current positions with unrealized P&L
        - Portfolio performance metrics
        - Risk analytics

        Data is current as of the latest market data refresh.
        """,
        "tags": ["Portfolio"],
        "responses": {
            200: {
                "description": "Portfolio retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "portfolio_id": "portfolio-123",
                            "total_value": 50000.00,
                            "cash_balance": 10000.00,
                            "invested_value": 40000.00,
                            "positions": [
                                {
                                    "symbol": "AAPL",
                                    "quantity": 100,
                                    "average_entry_price": 150.00,
                                    "current_price": 155.00,
                                    "unrealized_gain": 500.00,
                                    "unrealized_percentage": 0.33
                                }
                            ],
                            "performance": {
                                "total_return": 5000.00,
                                "total_return_percentage": 11.1,
                                "ytd_return": 2000.00,
                                "ytd_return_percentage": 4.3
                            }
                        }
                    }
                }
            }
        }
    },
    "get_portfolio_summary": {
        "summary": "Get portfolio summary with key metrics",
        "description": """
        Lightweight endpoint for getting portfolio summary without detailed holdings.
        Useful for dashboard displays and frequent polling.

        Returns only key metrics for better performance.
        """,
        "tags": ["Portfolio"],
        "responses": {
            200: {"description": "Portfolio summary retrieved successfully"}
        }
    },
    "rebalance_portfolio": {
        "summary": "Rebalance portfolio to target allocation",
        "description": """
        Generates rebalancing orders to align portfolio with target allocation.

        Takes into account:
        - Risk tolerance settings
        - Investment goals
        - Current positions
        - Market conditions
        - Tax implications (if applicable)

        Returns suggested orders but does NOT automatically execute them.
        """,
        "tags": ["Portfolio"],
        "responses": {
            200: {"description": "Rebalancing orders generated"},
            400: {"description": "Cannot rebalance - portfolio conditions not met"}
        }
    }
}

# User endpoints documentation
USER_ENDPOINTS_DOCS: Dict[str, Dict[str, Any]] = {
    "get_profile": {
        "summary": "Get user profile",
        "description": """
        Retrieves the authenticated user's profile information including:
        - Basic account information
        - Risk tolerance and investment goals
        - Trading preferences and restrictions
        - Account status
        """,
        "tags": ["Users"],
        "responses": {
            200: {"description": "User profile retrieved successfully"}
        }
    },
    "update_profile": {
        "summary": "Update user profile",
        "description": """
        Updates user profile information. Some fields may have restrictions:
        - Risk tolerance cannot be changed more than once per month
        - Investment goal changes may trigger portfolio review

        Email changes require verification before taking effect.
        """,
        "tags": ["Users"],
        "responses": {
            200: {"description": "Profile updated successfully"},
            400: {"description": "Invalid update parameters"},
            429: {"description": "Too many profile changes - retry later"}
        }
    },
    "get_preferences": {
        "summary": "Get user trading preferences",
        "description": """
        Retrieves detailed trading preferences including:
        - Sector preferences and exclusions
        - Position size limits
        - Loss limits (daily, weekly, monthly)
        - Notification settings
        - Trading approval requirements
        """,
        "tags": ["Users"],
        "responses": {
            200: {"description": "Preferences retrieved successfully"}
        }
    },
    "update_preferences": {
        "summary": "Update trading preferences",
        "description": """
        Updates trading preferences. Changes take effect immediately and may affect:
        - Order validation rules
        - Risk monitoring
        - Notification behavior

        Some preferences require trading to be paused during update.
        """,
        "tags": ["Users"],
        "responses": {
            200: {"description": "Preferences updated successfully"},
            409: {"description": "Cannot update - trading must be paused"}
        }
    }
}

# Risk Management endpoints documentation
RISK_ENDPOINTS_DOCS: Dict[str, Dict[str, Any]] = {
    "get_risk_analysis": {
        "summary": "Get comprehensive risk analysis",
        "description": """
        Returns detailed risk analysis of the portfolio including:
        - Value at Risk (VaR) calculations
        - Maximum Drawdown analysis
        - Concentration risk metrics
        - Correlation analysis
        - Stress test results

        Analysis is based on current holdings and market conditions.
        """,
        "tags": ["Risk Management"],
        "responses": {
            200: {"description": "Risk analysis retrieved successfully"}
        }
    },
    "check_risk_status": {
        "summary": "Check current risk status",
        "description": """
        Quick endpoint to check if any risk limits are breached:
        - Daily loss limit
        - Weekly loss limit
        - Monthly loss limit
        - Maximum drawdown
        - Position concentration limits

        Returns status and recommendations if limits are approached.
        """,
        "tags": ["Risk Management"],
        "responses": {
            200: {
                "description": "Risk status retrieved",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "WARNING",
                            "breached_limits": ["Daily loss limit at 80%"],
                            "recommendations": ["Consider pausing trading"],
                            "time_to_reset": "2025-10-27T00:00:00Z"
                        }
                    }
                }
            }
        }
    }
}

# Authentication documentation
AUTH_DOCS = {
    "login": {
        "summary": "Authenticate and receive JWT token",
        "description": """
        Authenticates user credentials and returns JWT token for subsequent requests.

        Token includes:
        - User ID
        - Token type (access/refresh)
        - Expiration time
        - Scopes/permissions

        Use returned access_token in Authorization header for all requests.
        """,
        "tags": ["Authentication"],
        "security": [],  # No security for this endpoint
        "responses": {
            200: {
                "description": "Authentication successful",
                "content": {
                    "application/json": {
                        "example": {
                            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
                            "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
                            "token_type": "bearer",
                            "expires_in": 1800
                        }
                    }
                }
            },
            401: {"description": "Invalid credentials"},
            429: {"description": "Too many login attempts - try again later"}
        }
    },
    "refresh": {
        "summary": "Refresh access token",
        "description": """
        Refreshes expired access token using a refresh token.

        Refresh tokens are longer-lived and can be used to obtain new access tokens
        without re-authenticating.
        """,
        "tags": ["Authentication"],
        "responses": {
            200: {"description": "Token refreshed successfully"},
            401: {"description": "Invalid or expired refresh token"}
        }
    }
}

def add_documentation_to_router(router):
    """Add detailed documentation to router endpoints."""
    # This function would be called during router initialization
    # to add comprehensive OpenAPI documentation
    pass
