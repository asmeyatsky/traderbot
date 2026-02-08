# Prerequisites for AI Trading Platform

This document outlines all the required API keys and configuration values needed to run the AI Trading Platform application.

## Required API Keys

### Financial Data APIs
1. **Polygon API Key**
   - Provider: [Polygon.io](https://polygon.io/)
   - Purpose: Real-time and historical stock, forex, and crypto market data
   - How to get: Sign up for an account and get a free API key
   - Rate Limits: Free tier offers 500 requests per day
   
   dTd71Bs_JRm9eTn6wqefbiTdpDyKwOnB

2. **Alpha Vantage API Key**
   - Provider: [Alpha Vantage](https://www.alphavantage.co/)
   - Purpose: Stock market data and technical indicators
   - How to get: Sign up for a free account to get an API key
   - Rate Limits: Free tier offers 5 requests per minute
   
   G5PVUBXD9RQ9FUG9

3. **MarketAxess API Key**
   - Provider: [MarketAxess](https://www.marketaux.com/)
   - Purpose: Financial news and sentiment analysis
   - How to get: Register for an account and get API access

pHuJyqFsr6HCjo1lK2Js7LjGUxQV5PjyYIe67CsR

4. **Finnhub API Key**
   - Provider: [Finnhub](https://finnhub.io/)
   - Purpose: Real-time stock market data and webhooks
   - How to get: Sign up for a free account
   
   d4v1l71r01qnm7pov0e0d4v1l71r01qnm7pov0eg
   

### Broker API Keys
5. **Alpaca API Keys**
   - Provider: [Alpaca](https://alpaca.markets/)
   - Purpose: Brokerage account for executing trades
   - Keys required: 
     - `ALPACA_API_KEY` - Alpaca API key
     - `ALPACA_SECRET_KEY` - Alpaca secret key
   - How to get: Create an account with Alpaca (paper trading available for testing)
   - Note: For live trading, you need to fund your account


### Authentication
6. **JWT Secret Key**
   - Provider: Self-generated
   - Purpose: JWT token signing for authentication
   - How to get: Generate a secure random string
   - Command: `openssl rand -hex 32` (generates 64-character hex string)
   - Requirement: Must be at least 32 characters long
   
   9338575bb4d9c34cbf5b9437151b19bfe95e1d23bb2c24a30f37b19955b61eac

### Database and Cache
7. **Database URL**
   - Provider: Self-provisioned (PostgreSQL recommended)
   - Purpose: Application data storage
   - Format: `postgresql://username:password@host:port/database_name`

8. **Redis URL**
   - Provider: Self-provisioned
   - Purpose: Caching and session storage
   - Format: `redis://host:port/database_number`

## GitHub Secrets Setup

For deploying with GitHub Actions, create these secrets in your repository:

1. `GCP_SA_KEY` - Google Cloud Platform service account key JSON
2. `DATABASE_URL` - Database connection string
3. `REDIS_URL` - Redis connection string
4. `POLYGON_API_KEY` - Polygon API key
5. `ALPHA_VANTAGE_API_KEY` - Alpha Vantage API key
6. `MARKETAUX_API_KEY` - MarketAxess API key
7. `FINNHUB_API_KEY` - Finnhub API key
8. `ALPACA_API_KEY` - Alpaca API key
9. `ALPACA_SECRET_KEY` - Alpaca secret key
10. `JWT_SECRET_KEY` - JWT secret key (at least 32 characters)

## Setting Up GitHub Secrets

1. Go to your GitHub repository
2. Click on the "Settings" tab
3. In the left sidebar, go to "Secrets and variables" → "Actions"
4. Click "New repository secret" for each required secret
5. Enter the name and value for each secret

## Optional API Keys (For Enhanced Features)

These APIs provide additional functionality but aren't required for basic operation:

- **Interactive Brokers API Key**: For additional broker integration
- **AWS Credentials**: For S3 storage of models and backtest results (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

## Testing Configuration

For initial testing without all API keys:
1. Some providers offer sandbox/paper trading environments
2. You can temporarily modify the application to allow optional API keys with graceful degradation
3. Consider starting with basic API keys and adding more as needed

## Security Recommendations

- Never commit API keys to version control
- Use environment variables or secrets management systems
- Regularly rotate API keys
- Use the principle of least privilege when setting up API access
- Monitor API usage to avoid unexpected charges
