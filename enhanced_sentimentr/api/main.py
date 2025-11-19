"""
FastAPI web service for sentiment analysis
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import asyncio
import logging
import time
from datetime import datetime

from ..core.analyzer import HybridSentimentAnalyzer
from ..core.models import AnalysisConfig, SentimentMethod


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced SentimentR API",
    description="Advanced sentiment analysis with Gemini integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = HybridSentimentAnalyzer()


# Pydantic models for API
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")


class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    
    @validator('texts')
    def validate_texts(cls, v):
        if len(v) < 1:
            raise ValueError('At least one text is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 texts allowed')
        return v


class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    method: SentimentMethod = Field(SentimentMethod.HYBRID, description="Analysis method")
    include_emotions: bool = Field(False, description="Include emotion analysis")
    include_aspects: bool = Field(False, description="Include aspect analysis")
    include_explanation: bool = Field(False, description="Include explanation")
    gemini_api_key: Optional[str] = Field(None, description="Gemini API key (optional)")


class BatchAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    method: SentimentMethod = Field(SentimentMethod.HYBRID, description="Analysis method")
    include_emotions: bool = Field(False, description="Include emotion analysis")
    include_aspects: bool = Field(False, description="Include aspect analysis")
    include_explanation: bool = Field(False, description="Include explanation")
    batch_size: int = Field(10, ge=1, le=50, description="Batch processing size")
    gemini_api_key: Optional[str] = Field(None, description="Gemini API key (optional)")
    
    @validator('texts')
    def validate_texts(cls, v):
        if len(v) < 1:
            raise ValueError('At least one text is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 texts allowed')
        return v


class ConfigUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = Field(None, description="Gemini API key")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced SentimentR API",
        "version": "2.0.0",
        "description": "Advanced sentiment analysis with Gemini integration",
        "endpoints": {
            "analyze": "/analyze",
            "batch_analyze": "/batch_analyze",
            "quick_analyze": "/quick/{text}",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_available": analyzer.is_gemini_available(),
        "cache_stats": analyzer.get_cache_stats()
    }


@app.post("/analyze")
async def analyze_sentiment(request: AnalysisRequest):
    """
    Analyze sentiment of a single text
    """
    try:
        # Set API key if provided
        if request.gemini_api_key:
            analyzer.set_gemini_api_key(request.gemini_api_key)
        
        # Create config
        config = AnalysisConfig(
            method=request.method,
            include_emotions=request.include_emotions,
            include_aspects=request.include_aspects,
            include_explanation=request.include_explanation
        )
        
        # Analyze sentiment
        result = await analyzer.analyze(request.text, config)
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_analyze")
async def batch_analyze_sentiment(request: BatchAnalysisRequest):
    """
    Analyze sentiment of multiple texts
    """
    try:
        # Set API key if provided
        if request.gemini_api_key:
            analyzer.set_gemini_api_key(request.gemini_api_key)
        
        # Create config
        config = AnalysisConfig(
            method=request.method,
            include_emotions=request.include_emotions,
            include_aspects=request.include_aspects,
            include_explanation=request.include_explanation
        )
        
        # Analyze sentiments
        result = await analyzer.batch_analyze(request.texts, config, request.batch_size)
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quick/{text}")
async def quick_analyze(text: str):
    """
    Quick sentiment analysis for a single text (GET request)
    """
    try:
        if len(text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long for quick analysis (max 1000 chars)")
        
        # Use default config for quick analysis
        config = AnalysisConfig(method=SentimentMethod.RULE_BASED)  # Faster for quick analysis
        result = await analyzer.analyze(text, config)
        
        return {
            "text": text,
            "polarity": result.polarity,
            "confidence": result.confidence,
            "sentiment": "positive" if result.polarity > 0.1 else "negative" if result.polarity < -0.1 else "neutral",
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        logger.error(f"Quick analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    """
    Update API configuration
    """
    try:
        if request.gemini_api_key:
            analyzer.set_gemini_api_key(request.gemini_api_key)
            return {"message": "Gemini API key updated successfully"}
        
        return {"message": "No configuration changes made"}
        
    except Exception as e:
        logger.error(f"Config update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/clear_cache")
async def clear_cache():
    """
    Clear all caches
    """
    try:
        analyzer.clear_caches()
        return {"message": "All caches cleared successfully"}
        
    except Exception as e:
        logger.error(f"Cache clear failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/cache")
async def get_cache_stats():
    """
    Get cache statistics
    """
    try:
        return analyzer.get_cache_stats()
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/methods")
async def get_available_methods():
    """
    Get available analysis methods
    """
    return {
        "methods": [method.value for method in SentimentMethod],
        "descriptions": {
            "rule_based": "Fast rule-based analysis using lexicons and linguistic rules",
            "gemini": "AI-powered analysis using Google Gemini (requires API key)",
            "hybrid": "Combination of rule-based and Gemini (recommended)",
            "ensemble": "Advanced ensemble approach (same as hybrid currently)"
        },
        "gemini_available": analyzer.is_gemini_available()
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Enhanced SentimentR API starting up...")
    logger.info(f"Gemini available: {analyzer.is_gemini_available()}")


# Shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Enhanced SentimentR API shutting down...")
    analyzer.clear_caches()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
