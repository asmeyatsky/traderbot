"""
Unified Model Training Pipeline

Orchestrates training of all ML models (LSTM, XGBoost, RL agents)
for a configurable set of symbols. Can be run as a CLI script or
imported and called programmatically.

Usage:
    python -m src.infrastructure.data_processing.training_pipeline
    python -m src.infrastructure.data_processing.training_pipeline --symbols AAPL MSFT GOOGL
    python -m src.infrastructure.data_processing.training_pipeline --models lstm xgboost
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import List, Optional

from src.domain.value_objects import Symbol

logger = logging.getLogger(__name__)

# Default symbols for training — top liquid US equities
DEFAULT_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'JPM', 'V', 'JNJ',
]


def train_lstm(symbols: List[str], lookback_years: int = 3) -> dict:
    """Train LSTM models for all symbols."""
    from src.infrastructure.data_processing.ml_model_service import LSTMPricePredictionService

    service = LSTMPricePredictionService()
    results = {}

    for sym in symbols:
        logger.info(f"[LSTM] Training for {sym}...")
        start = time.time()
        try:
            symbol = Symbol(value=sym)
            success = service.retrain_model(symbol, lookback_years=lookback_years)
            elapsed = time.time() - start
            results[sym] = {
                'success': success,
                'elapsed_seconds': round(elapsed, 1),
                'performance': None,
            }
            if success:
                perf = service.get_model_performance(sym)
                results[sym]['performance'] = {
                    'accuracy': perf.accuracy,
                    'sharpe_ratio': perf.sharpe_ratio,
                }
                logger.info(f"[LSTM] {sym}: accuracy={perf.accuracy:.3f} in {elapsed:.1f}s")
            else:
                logger.warning(f"[LSTM] {sym}: training failed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[LSTM] {sym}: error - {e}")
            results[sym] = {'success': False, 'elapsed_seconds': round(elapsed, 1), 'error': str(e)}

    return results


def train_xgboost(symbols: List[str], lookback_years: int = 3) -> dict:
    """Train XGBoost models for all symbols."""
    from src.infrastructure.data_processing.ml_model_service import XGBoostPredictionService

    service = XGBoostPredictionService()
    results = {}

    for sym in symbols:
        logger.info(f"[XGBoost] Training for {sym}...")
        start = time.time()
        try:
            symbol = Symbol(value=sym)
            success = service.retrain_model(symbol, lookback_years=lookback_years)
            elapsed = time.time() - start
            results[sym] = {
                'success': success,
                'elapsed_seconds': round(elapsed, 1),
            }
            if success:
                perf = service.get_model_performance(sym)
                importance = service.get_feature_importance(sym)
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                results[sym]['performance'] = {'accuracy': perf.accuracy}
                results[sym]['top_features'] = top_features
                logger.info(f"[XGBoost] {sym}: accuracy={perf.accuracy:.3f} in {elapsed:.1f}s")
            else:
                logger.warning(f"[XGBoost] {sym}: training failed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[XGBoost] {sym}: error - {e}")
            results[sym] = {'success': False, 'elapsed_seconds': round(elapsed, 1), 'error': str(e)}

    return results


def train_rl_agents(symbols: List[str], episodes: int = 200) -> dict:
    """Train DQN RL agents for all symbols."""
    from src.infrastructure.data_processing.ml_model_service import RLTradingAgentService

    service = RLTradingAgentService()
    results = {}

    training_data = [{'symbol': sym, 'episodes': episodes} for sym in symbols]

    for item in training_data:
        sym = item['symbol']
        logger.info(f"[RL-DQN] Training for {sym} ({episodes} episodes)...")
        start = time.time()
        try:
            success = service.train([item])
            elapsed = time.time() - start
            results[sym] = {
                'success': success,
                'elapsed_seconds': round(elapsed, 1),
            }
            if success:
                logger.info(f"[RL-DQN] {sym}: trained in {elapsed:.1f}s")
            else:
                logger.warning(f"[RL-DQN] {sym}: training failed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[RL-DQN] {sym}: error - {e}")
            results[sym] = {'success': False, 'elapsed_seconds': round(elapsed, 1), 'error': str(e)}

    return results


def verify_sentiment() -> dict:
    """Verify FinBERT sentiment analysis is working."""
    from src.infrastructure.data_processing.ml_model_service import TransformerSentimentAnalysisService

    service = TransformerSentimentAnalysisService()
    test_texts = [
        "Apple reports record quarterly revenue beating all analyst expectations",
        "Company announces massive layoffs amid declining sales and revenue miss",
        "Markets closed mixed today with no significant movement",
    ]

    results = {}
    for text in test_texts:
        sentiment = service.analyze_sentiment(text)
        results[text[:50]] = {
            'score': float(sentiment.score),
            'confidence': float(sentiment.confidence),
            'source': sentiment.source,
        }
        logger.info(f"[Sentiment] '{text[:50]}...' → score={sentiment.score}, source={sentiment.source}")

    return results


def run_pipeline(
    symbols: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    lookback_years: int = 3,
    rl_episodes: int = 200,
) -> dict:
    """
    Run the full training pipeline.

    Args:
        symbols: List of stock symbols to train on (default: DEFAULT_SYMBOLS)
        models: Which models to train ('lstm', 'xgboost', 'rl', 'sentiment', 'all')
        lookback_years: Years of historical data for training
        rl_episodes: Number of RL training episodes per symbol
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    if models is None:
        models = ['all']

    train_all = 'all' in models
    pipeline_results = {
        'started_at': datetime.now().isoformat(),
        'symbols': symbols,
        'models_requested': models,
    }

    logger.info(f"Starting training pipeline for {len(symbols)} symbols: {symbols}")
    pipeline_start = time.time()

    # 1. Verify sentiment
    if train_all or 'sentiment' in models:
        logger.info("=" * 60)
        logger.info("Phase 1: Verifying FinBERT Sentiment Analysis")
        logger.info("=" * 60)
        pipeline_results['sentiment'] = verify_sentiment()

    # 2. Train LSTM
    if train_all or 'lstm' in models:
        logger.info("=" * 60)
        logger.info("Phase 2: Training LSTM Models")
        logger.info("=" * 60)
        pipeline_results['lstm'] = train_lstm(symbols, lookback_years)

    # 3. Train XGBoost
    if train_all or 'xgboost' in models:
        logger.info("=" * 60)
        logger.info("Phase 3: Training XGBoost Models")
        logger.info("=" * 60)
        pipeline_results['xgboost'] = train_xgboost(symbols, lookback_years)

    # 4. Train RL agents
    if train_all or 'rl' in models:
        logger.info("=" * 60)
        logger.info("Phase 4: Training RL (DQN) Agents")
        logger.info("=" * 60)
        pipeline_results['rl'] = train_rl_agents(symbols, rl_episodes)

    total_elapsed = time.time() - pipeline_start
    pipeline_results['completed_at'] = datetime.now().isoformat()
    pipeline_results['total_elapsed_seconds'] = round(total_elapsed, 1)

    logger.info("=" * 60)
    logger.info(f"Training pipeline completed in {total_elapsed:.1f}s")
    logger.info("=" * 60)

    # Summary
    for model_type in ['lstm', 'xgboost', 'rl']:
        if model_type in pipeline_results:
            model_results = pipeline_results[model_type]
            successes = sum(1 for r in model_results.values() if r.get('success'))
            logger.info(f"  {model_type.upper()}: {successes}/{len(model_results)} symbols trained successfully")

    return pipeline_results


def main():
    """CLI entry point for the training pipeline."""
    parser = argparse.ArgumentParser(description='TraderBot Model Training Pipeline')
    parser.add_argument(
        '--symbols', nargs='+', default=None,
        help=f'Stock symbols to train on (default: {DEFAULT_SYMBOLS[:5]}...)'
    )
    parser.add_argument(
        '--models', nargs='+', default=['all'],
        choices=['all', 'lstm', 'xgboost', 'rl', 'sentiment'],
        help='Which models to train (default: all)'
    )
    parser.add_argument(
        '--lookback-years', type=int, default=3,
        help='Years of historical data for training (default: 3)'
    )
    parser.add_argument(
        '--rl-episodes', type=int, default=200,
        help='Number of RL training episodes per symbol (default: 200)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    results = run_pipeline(
        symbols=args.symbols,
        models=args.models,
        lookback_years=args.lookback_years,
        rl_episodes=args.rl_episodes,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total time: {results['total_elapsed_seconds']}s")
    for model_type in ['lstm', 'xgboost', 'rl']:
        if model_type in results:
            model_results = results[model_type]
            for sym, r in model_results.items():
                status = "OK" if r.get('success') else "FAIL"
                perf = r.get('performance', {})
                acc = perf.get('accuracy', 'N/A')
                print(f"  {model_type.upper():8s} | {sym:6s} | {status:4s} | acc={acc}")


if __name__ == '__main__':
    main()
