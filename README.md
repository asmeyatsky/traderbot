# AI-Powered Autonomous Trading Platform

This is an AI-powered autonomous trading platform that intelligently aggregates real-time company news, market data, and financial fundamentals to enable users to buy and sell shares autonomously. The platform leverages cutting-edge artificial intelligence, machine learning, and natural language processing to maximize returns while managing risk effectively.

## Architecture

This project follows clean architecture principles with clear separation of concerns:

- **Domain Layer**: Contains business logic and domain models
- **Application Layer**: Contains use cases and DTOs
- **Infrastructure Layer**: Contains external service adapters and data repositories
- **Presentation Layer**: Contains API controllers and CLI commands

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables: `cp .env.example .env` and fill in your API keys
4. Run the application: `uvicorn src.presentation.api.main:app --reload`

## Configuration

The application uses environment variables for configuration. See `.env.example` for required variables.

## Running Tests

```bash
pytest src/tests/
```

## Project Structure

```
├── src/
│   ├── domain/          # Domain entities, services, and ports
│   ├── application/     # Use cases and DTOs
│   ├── infrastructure/  # External adapters and repositories
│   ├── presentation/    # API and CLI interfaces
│   └── tests/           # Unit and integration tests
├── data/                # Model data and processed datasets
├── config/              # Configuration files
└── requirements.txt     # Python dependencies
```

## Features

- Real-time market data aggregation
- AI-powered sentiment analysis
- Automated trading execution
- Risk management system
- Portfolio optimization
- Backtesting capabilities