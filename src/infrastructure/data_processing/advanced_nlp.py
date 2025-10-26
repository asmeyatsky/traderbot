"""
Advanced Natural Language Processing for Financial Text Analysis

This module implements state-of-the-art NLP techniques for processing
financial documents, earnings calls, news, and social media content.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import re
import logging
from dataclasses import dataclass

# Try to import advanced NLP libraries
try:
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
    import torch
    from torch import nn
    HAS_TRANSFORMERS = True
except ImportError:
    print("Transformers library not available. Using basic NLP.")
    HAS_TRANSFORMERS = False

try:
    import spacy
    from spacy import displacy
    HAS_SPACY = True
except ImportError:
    print("spaCy library not available.")
    HAS_SPACY = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    print("NLTK library not available.")
    HAS_NLTK = False

from src.domain.value_objects import NewsSentiment, Symbol
from src.infrastructure.data_processing.sentiment_analysis import FinSentimentAnalyzer


@dataclass
class NERResult:
    """Result of Named Entity Recognition."""
    entities: List[Tuple[str, str]]  # (entity, label)
    symbols: List[Symbol]
    companies: List[str]
    people: List[str]
    dates: List[str]
    numbers: List[str]


@dataclass
class FinancialDocument:
    """Structured representation of a financial document."""
    title: str
    content: str
    date: datetime
    source: str
    entities: NERResult
    sentiment: NewsSentiment
    key_phrases: List[str]
    summary: str
    relevance_score: float


class AdvancedNLPProcessor:
    """
    Advanced NLP processor for financial text analysis.
    """
    
    def __init__(self):
        self.fin_sentiment_analyzer = FinSentimentAnalyzer()
        
        # Initialize components based on available libraries
        self.sentiment_analyzer = None
        self.ner_model = None
        self.summarizer = None
        self.tokenizer = None
        
        if HAS_TRANSFORMERS:
            try:
                # Initialize transformer models
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                
                # Initialize NER model
                self.ner_model = pipeline(
                    "ner",
                    aggregation_strategy="simple",
                    model="Davlan/bert-base-multilingual-cased-ner-hrl"
                )
                
                # Initialize summarization model
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn"
                )
            except:
                print("Could not initialize transformer models, using fallback methods")
        
        if HAS_SPACY:
            try:
                # Load spaCy model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Could not load spaCy model, using fallback methods")
                self.nlp = None
        else:
            self.nlp = None
        
        if HAS_NLTK:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                self.sia = SentimentIntensityAnalyzer()
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except:
                print("Could not initialize NLTK components")
                self.sia = None
                self.lemmatizer = None
                self.stop_words = set()
        else:
            self.sia = None
            self.lemmatizer = None
            self.stop_words = set()
        
        # Financial domain specific patterns
        self.financial_patterns = {
            'symbols': r'\b[A-Z]{1,5}\b',  # Stock symbols (1-5 uppercase letters)
            'earnings': r'earnings|profit|revenue|eps|q\d',  # Earnings related terms
            'metrics': r'p/e|p/b|ro[ae]|debt.*ratio|market.*cap',  # Financial metrics
            'dates': r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',  # Dates
            'numbers': r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+%',  # Numbers with formatting
        }
    
    def extract_financial_entities(self, text: str) -> NERResult:
        """
        Extract financial entities from text using multiple approaches.
        """
        entities = []
        symbols = []
        companies = []
        people = []
        dates = []
        numbers = []
        
        # Extract using regex patterns first
        for pattern_name, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if pattern_name == 'symbols':
                # Filter potential symbols (validate if they're likely real symbols)
                potential_symbols = [match for match in matches if self._is_likely_symbol(match)]
                symbols.extend([Symbol(s) for s in potential_symbols])
                entities.extend([(s, 'SYMBOL') for s in potential_symbols])
            elif pattern_name == 'dates':
                dates.extend(matches)
            elif pattern_name == 'numbers':
                numbers.extend(matches)
        
        # Use NER if available
        if self.ner_model:
            try:
                ner_results = self.ner_model(text)
                for entity in ner_results:
                    entity_text = entity['word']
                    entity_label = entity['entity_group']
                    entities.append((entity_text, entity_label))
                    
                    if entity_label in ['ORG', 'CORP']:
                        companies.append(entity_text)
                    elif entity_label == 'PER':
                        people.append(entity_text)
            except:
                pass  # Fallback if NER fails
        
        # Use spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append((ent.text, ent.label_))
                    
                    if ent.label_ in ['ORG', 'COMPANY']:
                        companies.append(ent.text)
                    elif ent.label_ == 'PERSON':
                        people.append(ent.text)
                    elif ent.label_ in ['DATE', 'TIME']:
                        dates.append(ent.text)
            except:
                pass  # Fallback if spaCy fails
        
        # Use NLTK named entity recognition
        if HAS_NLTK:
            try:
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    tokens = word_tokenize(sentence)
                    pos_tags = nltk.pos_tag(tokens)
                    chunks = nltk.ne_chunk(pos_tags)
                    
                    for chunk in chunks:
                        if hasattr(chunk, 'label'):
                            entity = " ".join([token for token, pos in chunk.leaves()])
                            label = chunk.label()
                            entities.append((entity, label))
                            
                            if label in ['ORGANIZATION', 'COMPANY']:
                                companies.append(entity)
                            elif label == 'PERSON':
                                people.append(entity)
            except:
                pass  # Fallback if NLTK NER fails
        
        # Remove duplicates while preserving order
        entities = list(dict.fromkeys(entities))  # Remove duplicate tuples
        symbols = list(dict.fromkeys(symbols))  # Remove duplicate symbols
        companies = list(dict.fromkeys(companies))
        people = list(dict.fromkeys(people))
        dates = list(dict.fromkeys(dates))
        numbers = list(dict.fromkeys(numbers))
        
        return NERResult(
            entities=entities,
            symbols=symbols,
            companies=companies,
            people=people,
            dates=dates,
            numbers=numbers
        )
    
    def _is_likely_symbol(self, symbol: str) -> bool:
        """
        Check if a string is likely a stock symbol.
        """
        # Basic validation: 1-5 uppercase letters, not a common word
        if not re.match(r'^[A-Z]{1,5}$', symbol):
            return False
        
        # Exclude common non-stock words
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'HAS', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        return symbol not in common_words
    
    def analyze_sentiment_advanced(self, text: str) -> NewsSentiment:
        """
        Perform advanced sentiment analysis using multiple methods.
        """
        if HAS_TRANSFORMERS and self.sentiment_analyzer:
            try:
                # Use transformer model for sentiment analysis
                results = self.sentiment_analyzer(text)
                
                # Process results to get a single sentiment score
                # Assuming the model returns scores for different sentiment classes
                positive_score = 0
                negative_score = 0
                neutral_score = 0
                
                for result in results[0]:  # Results for first (and usually only) input
                    label = result['label']
                    score = result['score']
                    
                    if label.lower() in ['positive', 'pos', '1']:
                        positive_score = score
                    elif label.lower() in ['negative', 'neg', '0']:
                        negative_score = score
                    elif label.lower() in ['neutral', 'neu', '2']:
                        neutral_score = score
                
                # Convert to our -100 to 100 scale
                # Positive score - Negative score, scaled to [-100, 100]
                sentiment_score = (positive_score - negative_score) * 100
                
                return NewsSentiment(
                    score=sentiment_score,
                    confidence=max(positive_score, negative_score, neutral_score) * 100,
                    source="TransformerModel"
                )
            except:
                pass  # Fallback if transformer analysis fails
        
        # Fallback to our financial sentiment analyzer
        return self.fin_sentiment_analyzer.analyze_sentiment(text)
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from the text.
        """
        key_phrases = set()
        
        # Use regex for common financial phrases
        financial_phrases = [
            r'earnings.*beat',
            r'revenue.*growth',
            r'profit.*marg',  # profit margin
            r'market.*cap',   # market cap
            r'dividend.*yield',
            r'price.*earnings',  # P/E ratio
            r'buy.*rating',
            r'sell.*rating',
            r'earnings.*call',
            r'guidance.*updat',
            r'acquisitio',
            r'merger',
            r'partnership',
            r'contract.*win',
            r'FDA.*approv',
            r'clinical.*trial',
            r'regulatory.*clear',
        ]
        
        for phrase_pattern in financial_phrases:
            matches = re.findall(phrase_pattern, text, re.IGNORECASE)
            key_phrases.update(matches)
        
        # Use basic NLP for phrase extraction
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 5:  # Limit to short phrases
                        key_phrases.add(chunk.text.lower())
                
                # Extract important entities and their relationships
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'MONEY', 'PERCENT', 'DATE']:
                        key_phrases.add(ent.text.lower())
            except:
                pass
        
        # Get the most relevant phrases (by occurrence or importance)
        return list(key_phrases)[:20]  # Return top 20 phrases
    
    def summarize_document(self, text: str, max_length: int = 150) -> str:
        """
        Summarize a financial document.
        """
        if HAS_TRANSFORMERS and self.summarizer:
            try:
                # Use transformer model for summarization
                max_input_length = 1024  # BART has max input length
                
                # Truncate if too long
                if len(text) > max_input_length:
                    text = text[:max_input_length]
                
                summary_result = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                
                return summary_result[0]['summary_text']
            except:
                pass  # Fallback if summarization fails
        
        # Fallback to simple extractive summarization
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text
        
        # Simple approach: return first 3 sentences
        return " ".join(sentences[:3])
    
    def calculate_document_relevance(self, text: str, target_symbols: List[Symbol]) -> float:
        """
        Calculate how relevant a document is to specific symbols.
        """
        relevance_score = 0.0
        
        # Check for symbol mentions
        text_upper = text.upper()
        for symbol in target_symbols:
            symbol_str = str(symbol).upper()
            # Count occurrences of the symbol
            occurrences = len(re.findall(r'\b' + re.escape(symbol_str) + r'\b', text_upper))
            relevance_score += occurrences * 0.5  # Weight for symbol mentions
        
        # Check for company name mentions (if available in metadata)
        # This would require a mapping of symbols to company names
        
        # Check for financial terms
        financial_terms = ['earnings', 'revenue', 'profit', 'loss', 'growth', 'decline', 
                          'beat', 'miss', 'guidance', 'forecast', 'outlook']
        for term in financial_terms:
            if term.lower() in text.lower():
                relevance_score += 0.1
        
        # Normalize by document length to avoid bias toward longer documents
        doc_length = len(text.split())
        if doc_length > 0:
            relevance_score = min(1.0, relevance_score * 100 / doc_length)
        
        return min(relevance_score, 1.0)  # Cap at 1.0
    
    def process_financial_document(self, 
                                 title: str, 
                                 content: str, 
                                 date: datetime, 
                                 source: str,
                                 target_symbols: List[Symbol]) -> FinancialDocument:
        """
        Process a complete financial document end-to-end.
        """
        # Extract entities
        entities = self.extract_financial_entities(content)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment_advanced(content)
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(content)
        
        # Summarize document
        summary = self.summarize_document(content)
        
        # Calculate relevance
        relevance_score = self.calculate_document_relevance(content, target_symbols)
        
        return FinancialDocument(
            title=title,
            content=content,
            date=date,
            source=source,
            entities=entities,
            sentiment=sentiment,
            key_phrases=key_phrases,
            summary=summary,
            relevance_score=relevance_score
        )


class EarningsCallAnalyzer:
    """
    Specialized analyzer for earnings call transcripts.
    """
    
    def __init__(self):
        self.nlp_processor = AdvancedNLPProcessor()
        self.q_keywords = ['question', 'q:', 'question:', 'operator']
        self.a_keywords = ['answer', 'a:', 'answer:', 'management']
    
    def analyze_earnings_call(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze an earnings call transcript.
        """
        results = {
            'management_sentiment': [],
            'q_and_a_sentiment': [],
            'key_metrics': self._extract_metrics(transcript),
            'forward_looking_statements': self._find_forward_looking_statements(transcript),
            'risks_mentioned': self._find_risks(transcript),
            'opportunities_mentioned': self._find_opportunities(transcript),
            'executive_tone_analysis': self._analyze_executive_tone(transcript)
        }
        
        # Analyze sentiment in different sections
        sections = self._split_transcript_into_sections(transcript)
        
        for section_type, section_text in sections.items():
            sentiment = self.nlp_processor.analyze_sentiment_advanced(section_text)
            results[f'{section_type}_sentiment'] = sentiment
        
        return results
    
    def _split_transcript_into_sections(self, transcript: str) -> Dict[str, str]:
        """
        Split earnings call transcript into sections.
        """
        sections = {
            'management_presentation': '',
            'q_and_a': '',
            'forward_looking_guidance': ''
        }
        
        lines = transcript.split('\n')
        
        # Simple heuristic to identify sections
        in_qa_section = False
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in self.q_keywords):
                in_qa_section = True
                sections['q_and_a'] += line + '\n'
            elif in_qa_section:
                sections['q_and_a'] += line + '\n'
            else:
                sections['management_presentation'] += line + '\n'
        
        return sections
    
    def _extract_metrics(self, transcript: str) -> List[Dict[str, Any]]:
        """
        Extract financial metrics mentioned in the transcript.
        """
        metrics = []
        
        # Pattern for finding metrics like "revenue was $X billion"
        metric_patterns = [
            r'(revenue|sales|income|profit|earnings|eps|ebitda|ebit|margin|growth).*?(\$\d+\.?\d*\s*(?:million|billion|k|thousand)?)',
            r'(revenue|sales|income|profit|earnings|eps|ebitda|ebit|margin|growth).*?(\d+\.?\d*\s*%)',
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            for metric_name, value in matches:
                metrics.append({
                    'metric': metric_name,
                    'value': value,
                    'context': self._get_context(transcript, metric_name, 50)
                })
        
        return metrics
    
    def _get_context(self, text: str, keyword: str, context_length: int = 50) -> str:
        """
        Get context around a keyword.
        """
        match = re.search(r'.{0,' + str(context_length) + r'}' + re.escape(keyword) + r'.{0,' + str(context_length) + r'}', text, re.IGNORECASE)
        return match.group(0) if match else ""
    
    def _find_forward_looking_statements(self, transcript: str) -> List[str]:
        """
        Identify forward-looking statements in the transcript.
        """
        forward_looking_indicators = [
            r'expect',
            r'anticipate',
            r'believe',
            r'project',
            r'forecast',
            r'plan',
            r'intend',
            r'outlook',
            r'guidance',
            r'target',
            r'goal',
            r'objective'
        ]
        
        statements = []
        for indicator in forward_looking_indicators:
            pattern = r'.*?' + indicator + r'.*?[\.\n]'
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            statements.extend([m.strip() for m in matches])
        
        return list(set(statements))  # Remove duplicates
    
    def _find_risks(self, transcript: str) -> List[str]:
        """
        Identify risks mentioned in the transcript.
        """
        risk_keywords = [
            'risk', 'uncertainty', 'challenge', 'threat', 'concern', 
            'volatile', 'decline', 'decrease', 'loss', 'negative'
        ]
        
        sentences = re.split(r'[.\n]', transcript)
        risks = []
        
        for sentence in sentences:
            for keyword in risk_keywords:
                if keyword.lower() in sentence.lower():
                    risks.append(sentence.strip())
                    break
        
        return list(set(risks))
    
    def _find_opportunities(self, transcript: str) -> List[str]:
        """
        Identify opportunities mentioned in the transcript.
        """
        opportunity_keywords = [
            'opportunity', 'growth', 'expansion', 'market', 'potential', 
            'increase', 'improve', 'upgrade', 'innovate', 'develop'
        ]
        
        sentences = re.split(r'[.\n]', transcript)
        opportunities = []
        
        for sentence in sentences:
            for keyword in opportunity_keywords:
                if keyword.lower() in sentence.lower():
                    opportunities.append(sentence.strip())
                    break
        
        return list(set(opportunities))
    
    def _analyze_executive_tone(self, transcript: str) -> Dict[str, float]:
        """
        Analyze the tone of executives in the call.
        """
        # This would analyze language patterns, confidence indicators, etc.
        # For now, a simple implementation
        positive_indicators = ['strong', 'excellent', 'outperform', 'beaten', 'raised', 'increased', 'growing', 'momentum']
        negative_indicators = ['decline', 'decrease', 'challenges', 'concerns', 'difficult', 'declining', 'decreased', 'falling']
        
        transcript_lower = transcript.lower()
        
        positive_count = sum(1 for word in positive_indicators if word in transcript_lower)
        negative_count = sum(1 for word in negative_indicators if word in transcript_lower)
        
        total_indicators = positive_count + negative_count
        if total_indicators > 0:
            positive_ratio = positive_count / total_indicators
            negative_ratio = negative_count / total_indicators
        else:
            positive_ratio = 0.5
            negative_ratio = 0.5
        
        return {
            'positive_tone_ratio': positive_ratio,
            'negative_tone_ratio': negative_ratio,
            'confidence_level': 1.0 - abs(positive_ratio - negative_ratio)  # More balanced = less confident
        }


class NewsClassifier:
    """
    Classify financial news into categories.
    """
    
    def __init__(self):
        self.categories = {
            'earnings': ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'eps', 'revenue', 'profit'],
            'merger_acquisition': ['acquisition', 'merger', 'buyout', 'takeover', 'purchase'],
            'regulatory': ['regulatory', 'fda', 'approval', 'clearance', 'compliance', 'audit'],
            'product': ['product', 'launch', 'release', 'development', 'trial', 'research'],
            'market': ['market', 'index', 'trading', 'volume', 'trend', 'bull', 'bear'],
            'dividend': ['dividend', 'yield', 'payout', 'distribution'],
            'management': ['ceo', 'executive', 'leadership', 'appointment', 'resignation'],
            'legal': ['lawsuit', 'litigation', 'legal', 'settlement', 'dispute']
        }
    
    def classify_news(self, title: str, content: str) -> List[str]:
        """
        Classify news into relevant categories.
        """
        text = (title + ' ' + content).lower()
        assigned_categories = []
        
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in text:
                    if category not in assigned_categories:
                        assigned_categories.append(category)
                    break  # Found one keyword, move to next category
        
        return assigned_categories


# Initialize the advanced NLP services
advanced_nlp_processor = AdvancedNLPProcessor()
earnings_call_analyzer = EarningsCallAnalyzer()
news_classifier = NewsClassifier()