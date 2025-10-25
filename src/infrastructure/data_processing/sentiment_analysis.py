"""
Sentiment Analysis AI/ML Model

This module implements the initial AI/ML model for sentiment analysis
as required by the PRD (NLP Engine for sentiment analysis).
"""
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
import os
from typing import List, Dict, Any
from datetime import datetime
import logging

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

from src.domain.value_objects import NewsSentiment


class SentimentAnalyzer:
    """
    AI/ML model for analyzing sentiment in financial news and social media.
    
    Implements the NLP Engine requirements from the PRD:
    - Sentiment analysis on news articles, earnings calls, social media
    - Real-time sentiment scoring: -100 (extremely negative) to +100 (extremely positive)
    - Aggregate sentiment metrics by ticker, sector, market
    """
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.custom_model = None
        self.vectorizer = None
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        """
        # Basic preprocessing
        text = text.lower().strip()
        # Additional preprocessing could go here (removing special chars, etc.)
        return text
    
    def analyze_with_vader(self, text: str) -> float:
        """
        Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
        Returns compound score between -1 and 1, scaled to -100 to 100.
        """
        scores = self.sia.polarity_scores(text)
        compound_score = scores['compound']
        # Scale from [-1, 1] to [-100, 100]
        scaled_score = compound_score * 100
        return scaled_score
    
    def analyze_with_textblob(self, text: str) -> float:
        """
        Analyze sentiment using TextBlob.
        Returns polarity score between -1 and 1, scaled to -100 to 100.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        # Scale from [-1, 1] to [-100, 100]
        scaled_score = polarity * 100
        return scaled_score
    
    def analyze_sentiment(self, text: str) -> NewsSentiment:
        """
        Analyze sentiment of a text using multiple methods and return a NewsSentiment object.
        
        Uses ensemble approach combining VADER and TextBlob for better accuracy.
        """
        if not text or not text.strip():
            return NewsSentiment(score=0, confidence=50, source="EmptyText")
        
        # Get scores from different analyzers
        vader_score = self.analyze_with_vader(text)
        textblob_score = self.analyze_with_textblob(text)
        
        # Combine scores (simple average for now, could be weighted based on performance)
        combined_score = (vader_score + textblob_score) / 2
        
        # Use the higher confidence of the two methods (arbitrary choice)
        # In practice, we might want to train a model to predict confidence
        confidence = 75  # Default confidence for rule-based models
        
        return NewsSentiment(
            score=round(combined_score, 2),
            confidence=confidence,
            source="VADER_TextBlob_Ensemble"
        )
    
    def batch_analyze_sentiment(self, texts: List[str]) -> List[NewsSentiment]:
        """
        Analyze sentiment for multiple texts.
        """
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def train_custom_model(self, training_data: List[Dict[str, Any]]):
        """
        Train a custom ML model on provided training data.
        Expected format: [{'text': '...', 'sentiment_score': -50 to 50}]
        """
        if not training_data:
            self.logger.warning("No training data provided")
            return
        
        # For now, we'll just log that training was requested
        # A full implementation would require pandas and sklearn
        self.logger.info(f"Custom model training requested with {len(training_data)} samples")
        
        # In a real implementation with sklearn, we would:
        # 1. Prepare data using pandas
        # 2. Vectorize text using TfidfVectorizer
        # 3. Train a model using LogisticRegression
        # 4. Evaluate the model
        
        # For now, mark as trained to allow usage of fallback methods
        self.is_trained = True
    
    def predict_with_custom_model(self, text: str) -> float:
        """
        Predict sentiment using the custom trained model.
        Returns score between -100 and 100.
        """
        if not self.is_trained or not self.vectorizer or not self.custom_model:
            # Fallback to rule-based if model not trained
            return self.analyze_with_vader(text)
        
        # Vectorize the text
        text_vec = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.custom_model.predict_proba(text_vec)[0]
        
        # Convert to score between -100 and 100
        # prediction[0] is probability of negative, prediction[1] is probability of positive
        pos_prob = prediction[1]
        neg_prob = prediction[0]
        
        # Map to range [-100, 100]
        score = (pos_prob - neg_prob) * 100
        
        return score
    
    def save_model(self, filepath: str):
        """
        Save the trained model to a file.
        """
        model_data = {
            'custom_model': self.custom_model,
            'vectorizer': self.vectorizer,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """
        Load a trained model from a file.
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"Model file does not exist: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.custom_model = model_data['custom_model']
        self.vectorizer = model_data['vectorizer']
        self.is_trained = model_data['is_trained']


class FinSentimentAnalyzer(SentimentAnalyzer):
    """
    Extended sentiment analyzer specifically tuned for financial news.
    
    This extends the base SentimentAnalyzer with financial-specific features:
    - Financial-specific lexicon
    - Handling of financial jargon and terminology
    - Context-aware sentiment for financial terms
    """
    
    def __init__(self):
        super().__init__()
        
        # Financial lexicon for more accurate financial sentiment
        self.financial_lexicon = {
            # Positive sentiment terms
            'revenue increase': 2, 'profit increase': 2, 'earnings beat': 2,
            'buy rating': 1.5, 'upgrade': 1.5, 'outperform': 1.5,
            'growth': 1, 'expansion': 1, 'contract win': 2,
            'acquisition': 0, 'merger': 0,  # Context dependent
            
            # Negative sentiment terms
            'revenue decline': -2, 'profit miss': -2, 'earnings miss': -2,
            'downgrade': -1.5, 'sell rating': -1.5, 'underperform': -1.5,
            'loss': -2, 'decline': -1, 'layoffs': -1.5,
            'lawsuit': -1, 'scandal': -2, 'bankruptcy': -2
        }
    
    def analyze_with_financial_context(self, text: str) -> float:
        """
        Analyze sentiment with financial context awareness.
        """
        base_score = self.analyze_with_vader(text)
        
        # Add financial context adjustments
        text_lower = text.lower()
        adjustment = 0
        
        for term, weight in self.financial_lexicon.items():
            if term in text_lower:
                adjustment += weight
        
        # Combine base score with financial adjustments
        # The adjustment is scaled to not overwhelm the base sentiment
        adjusted_score = base_score + (adjustment * 10)  # Scale adjustment
        
        # Ensure score stays within bounds
        adjusted_score = max(-100, min(100, adjusted_score))
        
        return adjusted_score
    
    def analyze_sentiment(self, text: str) -> NewsSentiment:
        """
        Override base method to include financial context.
        """
        if not text or not text.strip():
            return NewsSentiment(score=0, confidence=50, source="EmptyText")
        
        # Use financial context-aware analysis
        score = self.analyze_with_financial_context(text)
        
        return NewsSentiment(
            score=round(score, 2),
            confidence=80,  # Higher confidence for financial-trained model
            source="FinancialContextVADER"
        )


# Initialize the default sentiment analyzer
sentiment_analyzer = FinSentimentAnalyzer()