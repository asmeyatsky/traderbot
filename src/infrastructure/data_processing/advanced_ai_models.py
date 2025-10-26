"""
Advanced AI Models for Trading Predictions

This module implements sophisticated AI models using TensorFlow
for price prediction, pattern recognition, and trading signal generation.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import pickle
import os

from src.domain.value_objects import Symbol, Price
from src.domain.entities.trading import Order
from src.domain.ports import MarketDataPort
from src.infrastructure.config.settings import settings


class LSTMPricePredictor:
    """
    LSTM-based model for predicting stock price movements.
    
    Implements the Predictive Analytics Models requirement from PRD:
    - Price prediction: 1-day, 5-day, 30-day horizons
    - Feature engineering: 200+ technical, fundamental, sentiment features
    """
    
    def __init__(self, sequence_length: int = 60, features_count: int = 20):
        self.sequence_length = sequence_length
        self.features_count = features_count
        self.model = None
        self.scaler = None
        
    def build_model(self):
        """
        Build the LSTM model architecture.
        """
        model = keras.Sequential([
            layers.LSTM(100, return_sequences=True, input_shape=(self.sequence_length, self.features_count)),
            layers.Dropout(0.2),
            layers.LSTM(100, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(50),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)  # Predict next price
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_features(self, historical_data: List[Dict[str, Any]]) -> np.array:
        """
        Prepare features from historical data for the model.
        """
        # This is a simplified feature preparation
        # In a real implementation, we'd have 200+ features as required by PRD
        features = []
        
        for i in range(len(historical_data)):
            # Basic features: price, volume, technical indicators
            item = historical_data[i]
            feature_row = [
                item.get('open', 0),
                item.get('high', 0),
                item.get('low', 0),
                item.get('close', 0),
                item.get('volume', 0),
                # Add more features here
            ]
            
            # Normalize features (simplified)
            features.append(feature_row)
        
        # Ensure we have enough features
        while len(features) < self.sequence_length:
            features.insert(0, features[0])  # Pad with first value
        
        # Take the last sequence_length values
        features = features[-self.sequence_length:]
        
        return np.array(features)
    
    def train(self, training_data: List[Dict[str, Any]], epochs: int = 50):
        """
        Train the LSTM model on historical data.
        """
        if self.model is None:
            self.build_model()
        
        # Prepare training data
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(training_data)):
            seq = self.prepare_features(training_data[i-self.sequence_length:i])
            target = training_data[i]['close']  # Predict next closing price
            sequences.append(seq)
            targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def predict(self, input_sequence: List[Dict[str, Any]]) -> float:
        """
        Predict the next price value.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X = self.prepare_features(input_sequence).reshape(1, self.sequence_length, -1)
        prediction = self.model.predict(X)
        return float(prediction[0][0])
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        """
        if self.model:
            self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        """
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)


class TransformerSentimentModel:
    """
    Transformer-based model for sentiment analysis of financial news.
    
    Implements advanced NLP capabilities beyond the basic sentiment analyzer.
    """
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
    
    def build_model(self):
        """
        Build a transformer-based model for sentiment analysis.
        """
        # Input layer
        input_ids = keras.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = keras.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        # Embedding layer
        embedding = layers.Embedding(self.vocab_size, 128)(input_ids)
        
        # Transformer block
        transformer_block = layers.MultiHeadAttention(
            num_heads=8, key_dim=128
        )(embedding, embedding, attention_mask=tf.cast(attention_mask, tf.float32))
        
        # Global average pooling
        pooling = layers.GlobalAveragePooling1D()(transformer_block)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu')(pooling)
        dropout1 = layers.Dropout(0.5)(dense1)
        output = layers.Dense(3, activation='softmax', name='sentiment_output')(dropout1)  # Positive, Neutral, Negative
        
        model = keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_text(self, text: str) -> Tuple[np.array, np.array]:
        """
        Preprocess text for the transformer model.
        """
        # This is a simplified tokenization
        # In real implementation, we'd use proper tokenizers like BERT
        tokens = text.lower().split()
        # Convert to indices (simplified)
        token_ids = [hash(token) % self.vocab_size for token in tokens]
        
        # Pad or truncate to max_length
        if len(token_ids) < self.max_length:
            token_ids.extend([0] * (self.max_length - len(token_ids)))
        else:
            token_ids = token_ids[:self.max_length]
        
        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token_id != 0 else 0 for token_id in token_ids]
        
        return np.array(token_ids), np.array(attention_mask)
    
    def train(self, texts: List[str], labels: List[int], epochs: int = 10):
        """
        Train the transformer model.
        """
        if self.model is None:
            self.build_model()
        
        # Preprocess texts
        input_ids_list = []
        attention_masks_list = []
        
        for text in texts:
            input_ids, attention_mask = self.preprocess_text(text)
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
        
        X_input_ids = np.array(input_ids_list)
        X_attention_masks = np.array(attention_masks_list)
        
        # Convert labels to categorical
        y = keras.utils.to_categorical(labels, num_classes=3)
        
        # Train the model
        history = self.model.fit(
            [X_input_ids, X_attention_masks], y,
            epochs=epochs,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def predict_sentiment(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment for a given text.
        Returns probabilities for positive, neutral, negative.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        input_ids, attention_mask = self.preprocess_text(text)
        input_ids = input_ids.reshape(1, -1)
        attention_mask = attention_mask.reshape(1, -1)
        
        prediction = self.model.predict([input_ids, attention_mask])
        
        return {
            'positive': float(prediction[0][2]),  # Index 2 for positive
            'neutral': float(prediction[0][1]),   # Index 1 for neutral
            'negative': float(prediction[0][0])   # Index 0 for negative
        }


class TechnicalIndicatorComputer:
    """
    Compute technical indicators for use as features in ML models.
    """
    
    @staticmethod
    def calculate_sma(prices: List[float], window: int) -> List[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < window:
            return [0.0] * len(prices)
        
        sma = []
        for i in range(len(prices)):
            if i < window - 1:
                sma.append(0.0)
            else:
                sma.append(sum(prices[i - window + 1:i + 1]) / window)
        return sma
    
    @staticmethod
    def calculate_ema(prices: List[float], window: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < window:
            return [0.0] * len(prices)
        
        ema = []
        multiplier = 2 / (window + 1)
        
        for i, price in enumerate(prices):
            if i == 0:
                ema.append(price)
            else:
                ema_value = (price * multiplier) + (ema[i-1] * (1 - multiplier))
                ema.append(ema_value)
        
        return ema
    
    @staticmethod
    def calculate_rsi(prices: List[float], window: int = 14) -> List[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < window + 1:
            return [0.0] * len(prices)
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        rsis = [0.0] * (window + 1)  # Initial values
        
        # Calculate first RSI
        avg_gain = sum(gains[:window]) / window
        avg_loss = sum(losses[:window]) / window
        
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1 + rs))
            rsis.append(rsi)
        
        # Calculate subsequent RSI values
        for i in range(window + 1, len(prices)):
            avg_gain = ((avg_gain * (window - 1)) + gains[i-1]) / window
            avg_loss = ((avg_loss * (window - 1)) + losses[i-1]) / window
            
            if avg_loss == 0:
                rsis.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1 + rs))
                rsis.append(rsi)
        
        return rsis
    
    @staticmethod
    def calculate_macd(prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD (12, 26, 9)."""
        ema12 = TechnicalIndicatorComputer.calculate_ema(prices, 12)
        ema26 = TechnicalIndicatorComputer.calculate_ema(prices, 26)
        
        macd_line = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
        signal_line = TechnicalIndicatorComputer.calculate_ema(macd_line[25:], 9)  # Start after ema26 is valid
        histogram = [m - s for m, s in zip(macd_line[25:], signal_line)]
        
        # Pad the beginning with zeros
        signal_line = [0.0] * 25 + signal_line
        histogram = [0.0] * 25 + histogram
        
        return macd_line, signal_line, histogram


class EnsemblePredictor:
    """
    Ensemble model combining multiple AI approaches for better predictions.
    """
    
    def __init__(self):
        self.lstm_predictor = None
        self.transformer_sentiment = None
        self.technical_computer = TechnicalIndicatorComputer()
        self.weights = {'lstm': 0.4, 'sentiment': 0.3, 'technical': 0.3}
    
    def prepare_features_with_technicals(self, historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance historical data with technical indicators.
        """
        if not historical_data:
            return []
        
        closes = [item['close'] for item in historical_data if 'close' in item]
        
        # Calculate various technical indicators
        sma_20 = self.technical_computer.calculate_sma(closes, 20)
        sma_50 = self.technical_computer.calculate_sma(closes, 50)
        rsi = self.technical_computer.calculate_rsi(closes)
        
        # Add technical indicators to original data
        enhanced_data = []
        for i, item in enumerate(historical_data):
            enhanced_item = item.copy()
            if i < len(sma_20):
                enhanced_item['sma_20'] = sma_20[i]
            if i < len(sma_50):
                enhanced_item['sma_50'] = sma_50[i]
            if i < len(rsi):
                enhanced_item['rsi'] = rsi[i]
            enhanced_data.append(enhanced_item)
        
        return enhanced_data
    
    def predict_price_direction(self, 
                                 symbol: Symbol, 
                                 historical_data: List[Dict[str, Any]], 
                                 news_sentiment: float) -> Dict[str, Any]:
        """
        Predict price direction using ensemble approach.
        """
        predictions = {}
        
        # LSTM price prediction
        if self.lstm_predictor:
            try:
                current_sequence = historical_data[-self.lstm_predictor.sequence_length:]
                lstm_pred = self.lstm_predictor.predict(current_sequence)
                current_price = historical_data[-1]['close']
                lstm_direction = (lstm_pred - current_price) / current_price
                predictions['lstm'] = lstm_direction
            except:
                predictions['lstm'] = 0.0
        
        # Sentiment influence
        predictions['sentiment'] = news_sentiment / 100.0  # Scale to -1 to 1
        
        # Technical analysis
        if len(historical_data) > 50:
            prices = [item['close'] for item in historical_data if 'close' in item]
            if len(prices) > 50:
                sma_20 = self.technical_computer.calculate_sma(prices, 20)[-1]
                sma_50 = self.technical_computer.calculate_sma(prices, 50)[-1]
                
                # If short-term SMA > long-term SMA, bullish
                technical_signal = 1.0 if sma_20 > sma_50 else -1.0
                predictions['technical'] = technical_signal
        
        # Combine predictions using weights
        weighted_prediction = 0
        total_weight = 0
        
        for model, weight in self.weights.items():
            if model in predictions:
                weighted_prediction += predictions[model] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            final_prediction = weighted_prediction / total_weight
        else:
            final_prediction = 0.0
        
        return {
            'direction': final_prediction,
            'confidence': abs(final_prediction),  # Simple confidence measure
            'model_predictions': predictions,
            'recommendation': self._get_recommendation(final_prediction)
        }
    
    def _get_recommendation(self, direction: float) -> str:
        """
        Convert prediction direction to trading recommendation.
        """
        if direction > 0.05:
            return 'STRONG_BUY'
        elif direction > 0.02:
            return 'BUY'
        elif direction > -0.02:
            return 'HOLD'
        elif direction > -0.05:
            return 'SELL'
        else:
            return 'STRONG_SELL'


# Initialize the advanced AI models
advanced_lstm_model = LSTMPricePredictor()
advanced_transformer_model = TransformerSentimentModel()
ensemble_predictor = EnsemblePredictor()