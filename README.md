# AI-Powered Autonomous Trading Platform

An enterprise-grade, AI-powered autonomous trading platform that intelligently aggregates real-time market data, company news, and financial fundamentals to enable intelligent trading decisions. Built with clean architecture, domain-driven design, and comprehensive security measures.

## 🎯 Key Features

- **Real-time Market Data**: Integration with multiple data providers (Polygon, Alpha Vantage, Finnhub)
- **AI-Powered Sentiment Analysis**: NLP-based analysis of financial news and market sentiment
- **Automated Trading**: Risk-aware order execution with multiple broker integrations (Alpaca, Interactive Brokers)
- **Advanced Risk Management**: Position limits, loss limits, drawdown controls, sector constraints
- **Portfolio Optimization**: Intelligent portfolio rebalancing and allocation strategies
- **Backtesting Framework**: Validate trading strategies against historical data
- **RESTful API**: Comprehensive API with OpenAPI documentation
- **Enterprise Security**: JWT authentication, rate limiting, CORS, input validation

## 🏗️ Architecture

Built on **Clean Architecture** and **Domain-Driven Design** principles:

```
┌─────────────────────────────────────────────────────────────┐
│                  Presentation Layer (API)                   │
│              (FastAPI Routers, OpenAPI Docs)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Application Layer (Use Cases)                   │
│        (Business Process Orchestration, DTOs)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          Domain Layer (Business Logic, Entities)            │
│    (Orders, Positions, Portfolio, Users, Value Objects)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Infrastructure Layer (Adapters)                  │
│    (Database, Cache, APIs, External Services)              │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

- **Domain Layer**: Pure business logic, independent of frameworks
- **Application Layer**: Orchestrates domain objects, implements use cases
- **Infrastructure Layer**: External service integration, data persistence
- **Presentation Layer**: HTTP API, request/response handling

## 📋 Project Structure

```
traderbot/
├── src/
│   ├── domain/                          # Domain Layer
│   │   ├── entities/
│   │   │   ├── trading.py              # Order, Position, Portfolio entities
│   │   │   └── user.py                 # User entity with preferences
│   │   ├── value_objects/
│   │   │   └── __init__.py             # Money, Symbol, Price, Sentiment
│   │   ├── services/
│   │   │   ├── trading.py              # Trading domain services
│   │   │   └── risk_management.py      # Risk management logic
│   │   ├── ports/                      # Interface definitions
│   │   ├── events.py                   # Domain events
│   │   └── exceptions.py               # Custom exceptions
│   │
│   ├── application/                     # Application Layer
│   │   ├── use_cases/
│   │   │   └── trading.py              # Trading use cases
│   │   └── dtos/                       # Data transfer objects
│   │       ├── order_dtos.py
│   │       ├── portfolio_dtos.py
│   │       └── user_dtos.py
│   │
│   ├── infrastructure/                  # Infrastructure Layer
│   │   ├── config/
│   │   │   └── settings.py             # Environment configuration
│   │   ├── database.py                 # Database management
│   │   ├── cache.py                    # Redis caching
│   │   ├── security.py                 # JWT authentication
│   │   ├── logging.py                  # Structured logging
│   │   ├── di_container.py             # Dependency injection
│   │   ├── api_clients/                # External API clients
│   │   ├── repositories/               # Data persistence
│   │   └── adapters/                   # External service adapters
│   │
│   ├── presentation/                    # Presentation Layer
│   │   └── api/
│   │       ├── main.py                 # FastAPI app initialization
│   │       └── routers/
│   │           ├── orders.py           # Order endpoints
│   │           ├── portfolio.py        # Portfolio endpoints
│   │           └── users.py            # User endpoints
│   │
│   └── tests/                          # Test Suite
│       └── domain_tests.py             # Domain layer tests
│
├── Dockerfile                          # Container image
├── docker-compose.yml                  # Development environment
├── .env.example                        # Environment template
├── setup.py                            # Package configuration
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- PostgreSQL 14+ or SQLite (for development)
- Redis 7+ (for caching)

### Local Development Setup

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd traderbot
   ```

2. **Create Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Run application**
   ```bash
   uvicorn src.presentation.api.main:app --reload
   ```

   Access API documentation: http://localhost:8000/api/docs

### Docker Development Setup

```bash
# Start all services (API, Database, Cache)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 🔐 Security & Authentication

### JWT Authentication

All protected endpoints require JWT bearer tokens:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/orders
```

### Security Features

- Password hashing with bcrypt
- JWT token expiration (configurable)
- Rate limiting (slowapi)
- CORS configuration
- Input validation with Pydantic
- Sensitive data masking in logs
- Environment variable validation

## 📚 API Documentation

### Interactive API Docs

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redocs
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json

### Endpoint Groups

#### Orders API (`/api/v1/orders`)
- `POST /create` - Create new order
- `GET /{order_id}` - Get order details
- `GET /` - List user orders
- `PUT /{order_id}` - Update order
- `DELETE /{order_id}` - Cancel order

#### Portfolio API (`/api/v1/portfolio`)
- `GET /` - Get portfolio details
- `GET /performance` - Portfolio performance metrics
- `GET /allocation` - Portfolio allocation breakdown
- `POST /cash-deposit` - Deposit cash
- `POST /cash-withdraw` - Withdraw cash

#### Users API (`/api/v1/users`)
- `POST /register` - Register new user
- `POST /login` - Login user
- `GET /me` - Get current user profile
- `PUT /me` - Update profile
- `PUT /me/risk-settings` - Update risk settings
- `PUT /me/sector-preferences` - Update sector preferences

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest src/tests/

# Specific test file
pytest src/tests/domain_tests.py

# With coverage
pytest --cov=src src/tests/
```

### Test Structure

- **Domain Tests**: Entity immutability, value objects, business logic
- **Application Tests**: Use case orchestration, service integration
- **Infrastructure Tests**: Repository implementations, adapter mocking
- **API Tests**: Endpoint validation, authentication, response formats

## 🔧 Configuration

### Environment Variables

See `.env.example` for comprehensive configuration:

```env
# Market Data APIs
POLYGON_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
FINNHUB_API_KEY=...

# Broker APIs
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/traderbot
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=... # Must be 32+ characters
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## 📊 Database Schema

### Core Entities

- **Users**: Trading platform users with preferences
- **Orders**: Trading orders with lifecycle tracking
- **Positions**: Active trading positions
- **Portfolios**: User portfolio aggregations
- **PortfolioHistory**: Historical portfolio snapshots for backtesting

## 🔄 Data Flow

```
User Request
    ↓
API Router (Input Validation)
    ↓
Use Case (Orchestration)
    ↓
Domain Service (Business Logic)
    ↓
Domain Entity (State Management)
    ↓
Repository (Persistence)
    ↓
Database / Cache
```

## 🛠️ Development Workflow

### Adding New Features

1. **Define Domain Model** (`src/domain/entities/`)
2. **Create Value Objects** (`src/domain/value_objects/`)
3. **Implement Domain Services** (`src/domain/services/`)
4. **Define Repository Ports** (`src/domain/ports/`)
5. **Implement Use Cases** (`src/application/use_cases/`)
6. **Create DTOs** (`src/application/dtos/`)
7. **Implement Repositories** (`src/infrastructure/repositories/`)
8. **Add API Router** (`src/presentation/api/routers/`)
9. **Write Tests** (`src/tests/`)

### Code Quality Standards

- Clean Architecture principles
- Domain-Driven Design patterns
- SOLID principles
- Comprehensive type hints
- Full test coverage (80%+)
- Clear docstrings and comments

## 📈 Performance Considerations

- **Database**: Connection pooling with QueuePool
- **Caching**: Redis caching layer for frequent queries
- **Async Operations**: FastAPI async endpoints
- **Rate Limiting**: slowapi rate limiting per IP
- **Batch Operations**: Efficient bulk database operations

## 🚨 Error Handling

Comprehensive exception hierarchy:

```
DomainException (base)
├── OrderException
│   ├── OrderValidationException
│   ├── InsufficientFundsException
│   ├── PositionSizeViolationException
│   └── OrderNotFound
├── PortfolioException
├── UserException
├── RiskManagementException
└── TradingException
```

## 📝 Logging

Structured logging with automatic sensitive data filtering:

```python
logger.info("Order placed", extra={
    "user_id": user_id,
    "symbol": symbol,
    "quantity": quantity,
})
```

Sensitive fields automatically masked in logs:
- API keys
- Passwords
- Tokens
- Secret keys

## 🤝 Contributing

1. Create feature branch from `main`
2. Implement feature following architecture
3. Write comprehensive tests
4. Update documentation
5. Submit pull request with detailed description

## 📄 License

See LICENSE file for details

## 🔗 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)

## 📞 Support

For issues and questions, please open an issue on GitHub or contact the development team.

---

**Last Updated**: 2025-10-26
**Version**: 1.0.0