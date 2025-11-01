"""
Alternative Data Integration Service

Implements integration with alternative data sources including:
- Satellite imagery data
- Credit card transaction data
- Supply chain data
- Social media sentiment
- ESG scoring
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from decimal import Decimal
import requests
import numpy as np
from enum import Enum

from src.domain.value_objects import Symbol


class AlternativeDataSource(Enum):
    SATELLITE_IMAGERY = "satellite_imagery"
    CREDIT_CARD_DATA = "credit_card_data"
    SUPPLY_CHAIN = "supply_chain"
    SOCIAL_MEDIA = "social_media"
    ESG_DATA = "esg_data"
    WEB_TRAFFIC = "web_traffic"
    JOB_POSTINGS = "job_postings"
    SHIPPING_DATA = "shipping_data"


@dataclass
class SatelliteDataPoint:
    """Data class for satellite imagery data"""
    asset_id: str  # e.g., store location, facility ID
    latitude: float
    longitude: float
    measurement_type: str  # parking_lots, building_activity, etc.
    value: Decimal  # normalized value
    date: datetime
    source: str
    confidence: Decimal  # 0-100


@dataclass
class CreditCardTrend:
    """Data class for credit card transaction trends"""
    merchant_category: str
    geographic_region: str
    trend_value: Decimal  # percentage change
    base_period: str  # e.g., "2019-12" for pre-pandemic baseline
    current_period: str
    data_point_count: int
    confidence: Decimal  # 0-100


@dataclass
class SupplyChainEvent:
    """Data class for supply chain events"""
    event_id: str
    company: str
    supplier: str
    event_type: str  # disruption, delay, quality_issue
    severity: str  # low, medium, high, critical
    estimated_impact: Decimal  # percentage revenue impact
    start_date: datetime
    estimated_resolution: Optional[datetime] = None
    source: str = ""


@dataclass
class SocialMediaSentiment:
    """Data class for social media sentiment"""
    platform: str  # twitter, reddit, etc.
    mention_count: int
    sentiment_score: Decimal  # -100 to 100
    topic_category: str  # earnings, product_launch, etc.
    date: datetime
    sample_size: int
    confidence: Decimal  # 0-100


@dataclass
class ESGScore:
    """Data class for ESG scoring"""
    company: str
    environmental_score: Decimal  # 0-100
    social_score: Decimal  # 0-100
    governance_score: Decimal  # 0-100
    overall_esg_score: Decimal  # 0-100
    data_date: datetime
    source: str
    trend_direction: str  # positive, negative, stable


@dataclass
class WebTrafficData:
    """Data class for website traffic data"""
    domain: str
    unique_visitors: int
    page_views: int
    avg_session_duration: int  # in seconds
    bounce_rate: Decimal  # percentage
    traffic_source: str
    date: datetime
    comparison_to_prev_period: Decimal  # percentage change


@dataclass
class AlternativeDataInsight:
    """Data class for insights derived from alternative data"""
    symbol: Symbol
    insight_type: str  # demand_trend, supply_risk, consumer_sentiment, etc.
    confidence_level: Decimal  # 0-100
    direction: str  # positive, negative, neutral
    impact_score: Decimal  # -100 to 100, how much it might affect stock
    supporting_data: List[Dict[str, Any]]  # raw data points that support the insight
    generated_at: datetime
    data_sources: List[AlternativeDataSource]


class AlternativeDataIntegrationService(ABC):
    """
    Abstract base class for alternative data integration services.
    """
    
    @abstractmethod
    def get_satellite_data(self, symbol: Symbol, days: int = 30) -> List[SatelliteDataPoint]:
        """Get satellite imagery data for a symbol"""
        pass
    
    @abstractmethod
    def get_credit_card_trends(self, symbol: Symbol, category: str = None) -> List[CreditCardTrend]:
        """Get credit card transaction trends for a symbol"""
        pass
    
    @abstractmethod
    def get_supply_chain_events(self, symbol: Symbol, days: int = 30) -> List[SupplyChainEvent]:
        """Get supply chain events for a symbol"""
        pass
    
    @abstractmethod
    def get_social_media_sentiment(self, symbol: Symbol, days: int = 7) -> List[SocialMediaSentiment]:
        """Get social media sentiment for a symbol"""
        pass
    
    @abstractmethod
    def get_esg_scores(self, symbol: Symbol) -> Optional[ESGScore]:
        """Get ESG scores for a symbol"""
        pass
    
    @abstractmethod
    def get_web_traffic_data(self, company_domain: str, days: int = 30) -> List[WebTrafficData]:
        """Get web traffic data for a company domain"""
        pass
    
    @abstractmethod
    def generate_alternative_data_insights(self, symbol: Symbol) -> List[AlternativeDataInsight]:
        """Generate insights from alternative data sources"""
        pass


class DefaultAlternativeDataIntegrationService(AlternativeDataIntegrationService):
    """
    Default implementation of alternative data integration services.
    Note: This is a simplified implementation using mock data - in production,
    this would connect to real alternative data APIs
    """
    
    def __init__(self):
        self._satellite_data = self._generate_mock_satellite_data()
        self._credit_card_trends = self._generate_mock_credit_card_data()
        self._supply_chain_events = self._generate_mock_supply_chain_events()
        self._social_media_data = self._generate_mock_social_media_data()
        self._esg_scores = self._generate_mock_esg_scores()
        self._web_traffic_data = self._generate_mock_web_traffic_data()
    
    def _generate_mock_satellite_data(self) -> Dict[str, List[SatelliteDataPoint]]:
        """Generate mock satellite data"""
        # For demonstration, we'll use a few well-known companies
        symbols = ['AAPL', 'TSLA', 'AMZN', 'WMT']
        data = {}
        
        for symbol in symbols:
            data[symbol] = []
            for i in range(10):  # 10 data points per symbol
                data_point = SatelliteDataPoint(
                    asset_id=f"{symbol}_store_{i}",
                    latitude=np.random.uniform(30, 50),
                    longitude=np.random.uniform(-120, -70),
                    measurement_type="parking_lots",
                    value=Decimal(str(round(np.random.uniform(50, 100), 2))),
                    date=datetime.now() - timedelta(days=i),
                    source="Mock Satellite Provider",
                    confidence=Decimal(str(round(np.random.uniform(70, 95), 2)))
                )
                data[symbol].append(data_point)
        
        return data
    
    def _generate_mock_credit_card_data(self) -> Dict[str, List[CreditCardTrend]]:
        """Generate mock credit card trend data"""
        symbols = ['AAPL', 'TSLA', 'AMZN', 'WMT']
        data = {}
        
        for symbol in symbols:
            data[symbol] = [
                CreditCardTrend(
                    merchant_category="Electronics" if symbol == "AAPL" else "Automotive" if symbol == "TSLA" else "General Retail",
                    geographic_region="US",
                    trend_value=Decimal(str(round(np.random.uniform(-10, 20), 2))),
                    base_period="2019-12",
                    current_period=datetime.now().strftime("%Y-%m"),
                    data_point_count=10000,
                    confidence=Decimal(str(round(np.random.uniform(75, 90), 2)))
                )
            ]
        
        return data
    
    def _generate_mock_supply_chain_events(self) -> Dict[str, List[SupplyChainEvent]]:
        """Generate mock supply chain events"""
        symbols = ['AAPL', 'TSLA', 'NVDA']
        data = {}
        
        for symbol in symbols:
            data[symbol] = [
                SupplyChainEvent(
                    event_id=f"event_{symbol}_{i}",
                    company=symbol,
                    supplier=f"Supplier_{np.random.randint(100, 999)}",
                    event_type=np.random.choice(["disruption", "delay", "quality_issue"]),
                    severity=np.random.choice(["low", "medium", "high"]),
                    estimated_impact=Decimal(str(round(np.random.uniform(1, 15), 2))),
                    start_date=datetime.now() - timedelta(days=np.random.randint(1, 30)),
                    source="Mock Supply Chain Monitor"
                )
                for i in range(np.random.randint(1, 4))  # 1-3 events per company
            ]
        
        return data
    
    def _generate_mock_social_media_data(self) -> Dict[str, List[SocialMediaSentiment]]:
        """Generate mock social media data"""
        symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']
        data = {}
        
        for symbol in symbols:
            data[symbol] = [
                SocialMediaSentiment(
                    platform=np.random.choice(["Twitter", "Reddit", "StockTwits"]),
                    mention_count=np.random.randint(1000, 10000),
                    sentiment_score=Decimal(str(round(np.random.uniform(-30, 50), 2))),
                    topic_category=np.random.choice(["Earnings", "Product Launch", "CEO News", "Market Conditions"]),
                    date=datetime.now() - timedelta(days=i),
                    sample_size=np.random.randint(1000, 10000),
                    confidence=Decimal(str(round(np.random.uniform(70, 90), 2)))
                )
                for i in range(7)  # 7 days of data
            ]
        
        return data
    
    def _generate_mock_esg_scores(self) -> Dict[str, ESGScore]:
        """Generate mock ESG scores"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        data = {}
        
        for symbol in symbols:
            data[symbol] = ESGScore(
                company=symbol,
                environmental_score=Decimal(str(round(np.random.uniform(60, 95), 2))),
                social_score=Decimal(str(round(np.random.uniform(50, 90), 2))),
                governance_score=Decimal(str(round(np.random.uniform(65, 95), 2))),
                overall_esg_score=Decimal(str(round(np.random.uniform(60, 90), 2))),
                data_date=datetime.now() - timedelta(days=np.random.randint(1, 30)),
                source="Mock ESG Provider",
                trend_direction=np.random.choice(["positive", "negative", "stable"])
            )
        
        return data
    
    def _generate_mock_web_traffic_data(self) -> Dict[str, List[WebTrafficData]]:
        """Generate mock web traffic data"""
        domains = ['apple.com', 'tesla.com', 'amazon.com', 'microsoft.com']
        data = {}
        
        for domain in domains:
            data[domain] = [
                WebTrafficData(
                    domain=domain,
                    unique_visitors=np.random.randint(1000000, 100000000),
                    page_views=np.random.randint(5000000, 500000000),
                    avg_session_duration=np.random.randint(60, 300),
                    bounce_rate=Decimal(str(round(np.random.uniform(30, 70), 2))),
                    traffic_source=np.random.choice(["Direct", "Search", "Social", "Referral"]),
                    date=datetime.now() - timedelta(days=i),
                    comparison_to_prev_period=Decimal(str(round(np.random.uniform(-15, 25), 2)))
                )
                for i in range(30)  # 30 days of data
            ]
        
        return data
    
    def get_satellite_data(self, symbol: Symbol, days: int = 30) -> List[SatelliteDataPoint]:
        """Get satellite imagery data for a symbol"""
        symbol_str = str(symbol)
        all_data = self._satellite_data.get(symbol_str, [])
        
        # Filter by days
        cutoff_date = datetime.now() - timedelta(days=days)
        return [d for d in all_data if d.date >= cutoff_date]
    
    def get_credit_card_trends(self, symbol: Symbol, category: str = None) -> List[CreditCardTrend]:
        """Get credit card transaction trends for a symbol"""
        symbol_str = str(symbol)
        trends = self._credit_card_trends.get(symbol_str, [])
        
        if category:
            trends = [t for t in trends if category.lower() in t.merchant_category.lower()]
        
        return trends
    
    def get_supply_chain_events(self, symbol: Symbol, days: int = 30) -> List[SupplyChainEvent]:
        """Get supply chain events for a symbol"""
        symbol_str = str(symbol)
        all_events = self._supply_chain_events.get(symbol_str, [])
        
        # Filter by days
        cutoff_date = datetime.now() - timedelta(days=days)
        return [e for e in all_events if e.start_date >= cutoff_date]
    
    def get_social_media_sentiment(self, symbol: Symbol, days: int = 7) -> List[SocialMediaSentiment]:
        """Get social media sentiment for a symbol"""
        symbol_str = str(symbol)
        all_data = self._social_media_data.get(symbol_str, [])
        
        # Filter by days
        cutoff_date = datetime.now() - timedelta(days=days)
        return [d for d in all_data if d.date >= cutoff_date]
    
    def get_esg_scores(self, symbol: Symbol) -> Optional[ESGScore]:
        """Get ESG scores for a symbol"""
        symbol_str = str(symbol)
        return self._esg_scores.get(symbol_str)
    
    def get_web_traffic_data(self, company_domain: str, days: int = 30) -> List[WebTrafficData]:
        """Get web traffic data for a company domain"""
        all_data = self._web_traffic_data.get(company_domain, [])
        
        # Filter by days
        cutoff_date = datetime.now() - timedelta(days=days)
        return [d for d in all_data if d.date >= cutoff_date]
    
    def generate_alternative_data_insights(self, symbol: Symbol) -> List[AlternativeDataInsight]:
        """Generate insights from alternative data sources"""
        insights = []
        
        # Generate insights based on the available data
        symbol_str = str(symbol)
        
        # Insight 1: Consumer demand trend based on satellite and credit card data
        sat_data = self.get_satellite_data(symbol, 14)
        if sat_data:
            avg_activity = sum(d.value for d in sat_data) / len(sat_data)
            direction = "positive" if avg_activity > 75 else "negative" if avg_activity < 25 else "neutral"
            
            insight = AlternativeDataInsight(
                symbol=symbol,
                insight_type="consumer_demand_trend",
                confidence_level=Decimal('85.0'),
                direction=direction,
                impact_score=Decimal(str(round(avg_activity - 50, 2))),
                supporting_data=[{"type": "satellite", "avg_activity": float(avg_activity)}],
                generated_at=datetime.now(),
                data_sources=[AlternativeDataSource.SATELLITE_IMAGERY]
            )
            insights.append(insight)
        
        # Insight 2: Supply chain risk based on supply chain events
        supply_events = self.get_supply_chain_events(symbol, 30)
        if supply_events:
            high_impact_events = [e for e in supply_events if float(e.estimated_impact) > 5.0]
            direction = "negative" if high_impact_events else "positive"
            impact_score = Decimal(str(-sum(float(e.estimated_impact) for e in high_impact_events)))
            
            insight = AlternativeDataInsight(
                symbol=symbol,
                insight_type="supply_chain_risk",
                confidence_level=Decimal('78.0'),
                direction=direction,
                impact_score=impact_score,
                supporting_data=[{"type": "supply_chain", "event_count": len(high_impact_events)}],
                generated_at=datetime.now(),
                data_sources=[AlternativeDataSource.SUPPLY_CHAIN]
            )
            insights.append(insight)
        
        # Insight 3: Social sentiment trend
        social_data = self.get_social_media_sentiment(symbol, 7)
        if social_data:
            avg_sentiment = sum(d.sentiment_score for d in social_data) / len(social_data)
            direction = "positive" if avg_sentiment > 10 else "negative" if avg_sentiment < -10 else "neutral"
            
            insight = AlternativeDataInsight(
                symbol=symbol,
                insight_type="social_sentiment_trend",
                confidence_level=Decimal('82.0'),
                direction=direction,
                impact_score=avg_sentiment,
                supporting_data=[{"type": "social_media", "avg_sentiment": float(avg_sentiment)}],
                generated_at=datetime.now(),
                data_sources=[AlternativeDataSource.SOCIAL_MEDIA]
            )
            insights.append(insight)
        
        # Insight 4: ESG factor
        esg_data = self.get_esg_scores(symbol)
        if esg_data:
            overall_score = esg_data.overall_esg_score
            direction = "positive" if overall_score > 70 else "negative" if overall_score < 40 else "neutral"
            
            insight = AlternativeDataInsight(
                symbol=symbol,
                insight_type="esg_factor",
                confidence_level=Decimal('90.0'),
                direction=direction,
                impact_score=overall_score - 50,  # Center around 0
                supporting_data=[{"type": "esg", "score": float(overall_score)}],
                generated_at=datetime.now(),
                data_sources=[AlternativeDataSource.ESG_DATA]
            )
            insights.append(insight)
        
        return insights