from setuptools import setup, find_packages

setup(
    name="ai-trading-platform",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "python-dotenv==1.0.0",
        "requests==2.31.0",

        # Data processing
        "pandas==2.1.4",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",

        # ML/DL
        "tensorflow==2.15.0",
        "torch==2.1.2",
        "transformers==4.36.0",

        # API and validation
        "pydantic[email]==2.5.0",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",

        # Database
        "sqlalchemy==2.0.23",
        "asyncpg==0.29.0",

        # Caching and message queue
        "redis==5.0.1",
        "celery==5.3.4",
        "kafka-python==2.0.2",

        # Real-time communication
        "websockets==12.0",

        # Security
        "pyjwt==2.8.0",
        "passlib==1.7.4",
        "bcrypt==4.1.2",

        # Dependency injection
        "dependency-injector==4.41.0",

        # Rate limiting
        "slowapi==0.1.9",

        # NLP/Sentiment analysis
        "textblob==0.17.1",
        "vaderSentiment==3.3.2",
        "nltk==3.8.1",
        "spacy==3.7.2",

        # Market data APIs
        "yfinance==0.2.18",
        "alpha-vantage==2.1.3",  # Changed to available version
        "polygon-api-client==1.13.0",
        "finnhub-python==2.4.14",

        # Testing
        "pytest==7.4.3",
        "pytest-asyncio==0.21.1",
        "pytest-mock==3.12.0",
        "pytest-cov==4.1.0",
        "factory-boy==3.3.0",

        # Utilities
        "python-slugify==8.0.1",
        "Pillow==10.1.0",
        "python-multipart==0.0.6",
        "cryptography==41.0.7",  # Changed to available version
    ],
    author="AI Trading Platform Team",
    author_email="dev@aitradingplatform.com",
    description="An AI-powered autonomous trading platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aitradingplatform/ai-trading-platform",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
)