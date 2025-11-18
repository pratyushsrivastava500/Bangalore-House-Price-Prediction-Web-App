"""
Configuration Module
Centralized configuration for the application
"""

import os
from pathlib import Path


class Config:
    """Configuration class for the House Price Prediction application"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    SRC_DIR = BASE_DIR / 'src'
    
    # Data files
    RAW_DATA_FILE = DATA_DIR / 'Bengaluru_House_Data.csv'
    PROCESSED_DATA_FILE = DATA_DIR / 'processed_data.csv'
    
    # Model files
    MODEL_FILE = MODELS_DIR / 'model_pickel'
    MODEL_FILE_NEW = MODELS_DIR / 'house_price_model.pkl'
    
    # Model parameters
    MODEL_TYPE = 'linear_regression'  # Options: 'linear_regression', 'lasso', 'decision_tree'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Preprocessing parameters
    MIN_LOCATION_THRESHOLD = 10  # Minimum occurrences for a location to be kept separate
    MIN_SQFT_PER_BHK = 300  # Minimum square footage per bedroom
    
    # App configuration
    APP_TITLE = "Bangalore House Rate Prediction"
    APP_DESCRIPTION = "ML-powered house price prediction for Bangalore"
    
    # UI Configuration
    THEME_COLOR = "black"
    SUCCESS_MESSAGE_TEMPLATE = "Predicted Price: ₹{:.2f} Lakhs"
    
    # Validation limits
    MIN_SQFT = 300
    MAX_SQFT = 50000
    MIN_BATH = 1
    MAX_BATH = 10
    MIN_BHK = 1
    MAX_BHK = 10
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_model_path(cls) -> Path:
        """Get the path to the model file"""
        # Check if new model exists, otherwise use old path
        if cls.MODEL_FILE_NEW.exists():
            return cls.MODEL_FILE_NEW
        return cls.MODEL_FILE
    
    @classmethod
    def get_data_path(cls) -> Path:
        """Get the path to the data file"""
        return cls.RAW_DATA_FILE


# Development configuration
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOGGING_LEVEL = 'DEBUG'


# Production configuration
class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOGGING_LEVEL = 'INFO'


# Default configuration
config = Config()
