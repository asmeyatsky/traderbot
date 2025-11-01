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
- **Advanced Risk Analytics**: Value at Risk (VaR), Expected Shortfall (ES), stress testing, correlation analysis
- **Enhanced User Experience**: Comprehensive dashboard with technical indicators, performance charts, allocation breakdowns
- **Market Data Enhancement**: Multi-source integration, sentiment analysis, economic calendar
- **Performance Optimization**: Advanced caching strategies, monitoring, response time optimization
- **Multi-Broker Integration**: Support for Alpaca, Interactive Brokers, and other providers
- **Alternative Data Sources**: Satellite imagery, credit card data, supply chain, social media sentiment, ESG scoring
- **Advanced AI/ML Models**: Price prediction, regime detection, volatility forecasting, portfolio optimization
- **Reinforcement Learning Trading Agents**: DQN, PPO, A2C algorithms with ensemble methods

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
│   │   │   ├── risk_management.py      # Risk management logic
│   │   │   ├── advanced_risk_management.py # Advanced risk analytics
│   │   │   ├── dashboard_analytics.py  # Enhanced dashboard analytics
│   │   │   ├── market_data_enhancement.py # Market data enhancement
│   │   │   ├── ml_model_service.py     # ML/AI model services
│   │   │   └── rl_trading_agents.py    # Reinforcement learning agents
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
│   │   ├── cache_layer.py              # Multi-tier caching
│   │   ├── performance_optimization.py # Performance optimization
│   │   ├── broker_integration.py       # Broker integration
│   │   ├── alternative_data_integration.py # Alternative data integration
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
│   │           ├── users.py            # User endpoints
│   │           ├── risk.py             # Risk analytics endpoints
│   │           ├── dashboard.py        # Dashboard endpoints
│   │           ├── market_data.py      # Market data endpoints
│   │           ├── performance.py      # Performance monitoring endpoints
│   │           ├── brokers.py          # Broker integration endpoints
│   │           ├── alternative_data.py # Alternative data endpoints
│   │           ├── ml.py               # Machine learning endpoints
│   │           └── rl.py               # Reinforcement learning endpoints
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

### Enhanced Endpoint Groups

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

#### Risk Analytics API (`/api/v1/risk`)
- `GET /portfolio/{user_id}` - Get portfolio risk metrics (VaR, ES, volatility, correlations)
- `POST /stress-test/{user_id}` - Perform stress testing under market scenarios
- `GET /correlation-matrix/{user_id}` - Get portfolio correlation matrix

#### Enhanced Dashboard API (`/api/v1/dashboard`)
- `GET /overview/{user_id}` - Get comprehensive portfolio dashboard
- `GET /allocation/{user_id}` - Get portfolio allocation breakdown
- `GET /technical-indicators/{symbol}` - Get technical indicators for a symbol

#### Market Data Enhancement API (`/api/v1/market-data`)
- `GET /enhanced/{symbol}` - Get enhanced market data with news sentiment
- `GET /sentiment/{symbol}` - Get news sentiment for a symbol
- `GET /economic-calendar` - Get economic calendar events
- `GET /volatility-forecast/{symbol}` - Get volatility forecast

#### Performance Monitoring API (`/api/v1/performance`)
- `GET /metrics` - Get system performance metrics
- `GET /cache-stats` - Get cache performance statistics
- `POST /cache/warm/{user_id}` - Warm up user's cache

#### Broker Integration API (`/api/v1/brokers`)
- `GET /available` - Get available broker integrations
- `POST /{broker_type}/place-order` - Place an order with a specific broker
- `GET /{broker_type}/positions` - Get positions from a specific broker
- `GET /{broker_type}/account-info` - Get account information from a broker

#### Alternative Data API (`/api/v1/alternative-data`)
- `GET /satellite/{symbol}` - Get satellite imagery data
- `GET /credit-card/{symbol}` - Get credit card transaction trends
- `GET /supply-chain/{symbol}` - Get supply chain events
- `GET /social-sentiment/{symbol}` - Get social media sentiment
- `GET /esg/{symbol}` - Get ESG scores
- `GET /insights/{symbol}` - Get alternative data insights

#### Machine Learning API (`/api/v1/ml`)
- `GET /predict/{symbol}` - Get price prediction
- `GET /regime/{symbol}` - Get market regime detection
- `GET /signal/{symbol}/{user_id}` - Get trading signal
- `POST /optimize-portfolio/{user_id}` - Optimize portfolio allocation
- `GET /volatility-forecast/{symbol}` - Get volatility forecast
- `GET /model-performance/{model_type}` - Get model performance metrics

#### Reinforcement Learning API (`/api/v1/rl`)
- `GET /algorithms` - Get available RL algorithms
- `POST /agents/train/{symbol}` - Train RL trading agent
- `POST /agents/evaluate/{symbol}` - Evaluate RL trading agent
- `GET /agents/ensemble-performance` - Get ensemble performance
- `POST /agents/get-action/{symbol}/{user_id}` - Get action from RL agent

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
- **Caching**: Multi-tier caching (L1 Memory, L2 Redis) for frequent queries
- **Async Operations**: FastAPI async endpoints
- **Rate Limiting**: slowapi rate limiting per IP
- **Batch Operations**: Efficient bulk database operations
- **Performance Monitoring**: Cache hit rates, response time tracking
- **Query Optimization**: Optimized database queries and indexing

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
├── AdvancedRiskManagementException
├── MLModelException
├── RLAgentException
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

**Last Updated**: 2025-11-01
**Version**: 1.0.0