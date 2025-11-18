"""
Utility Functions
Helper functions and common utilities
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any


def format_price(price: float) -> str:
    """
    Format price for display
    
    Args:
        price: Price in lakhs
        
    Returns:
        Formatted price string
    """
    if price >= 100:
        return f"₹{price:.2f} Lakhs (₹{price/100:.2f} Crores)"
    else:
        return f"₹{price:.2f} Lakhs"


def validate_inputs(sqft: Any, bath: Any, bhk: Any) -> tuple:
    """
    Validate and convert user inputs
    
    Args:
        sqft: Square footage input
        bath: Number of bathrooms input
        bhk: Number of BHK input
        
    Returns:
        Tuple of (is_valid, error_message, converted_values)
    """
    try:
        sqft = float(sqft)
        bath = int(bath)
        bhk = int(bhk)
        
        # Validation checks
        if sqft <= 0:
            return False, "Square footage must be positive", None
        if bath <= 0:
            return False, "Number of bathrooms must be positive", None
        if bhk <= 0:
            return False, "Number of BHK must be positive", None
        if sqft < 300 * bhk:
            return False, f"Square footage seems too low for {bhk} BHK (minimum ~{300*bhk} sqft recommended)", None
        if bath > bhk + 2:
            return False, "Number of bathrooms seems unusually high for the given BHK", None
            
        return True, "", (sqft, bath, bhk)
        
    except ValueError:
        return False, "Please enter valid numeric values", None


def get_unique_locations(df: pd.DataFrame) -> List[str]:
    """
    Get list of unique locations from DataFrame
    
    Args:
        df: DataFrame containing location column
        
    Returns:
        Sorted list of unique locations
    """
    if 'location' in df.columns:
        return sorted(df['location'].unique().tolist())
    return []


def save_json(data: Dict[Any, Any], filepath: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[Any, Any]:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing the loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for the dataset
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'total_records': len(df),
        'num_locations': df['location'].nunique() if 'location' in df.columns else 0,
    }
    
    if 'price' in df.columns:
        stats['avg_price'] = df['price'].mean()
        stats['min_price'] = df['price'].min()
        stats['max_price'] = df['price'].max()
        stats['median_price'] = df['price'].median()
    
    if 'total_sqft' in df.columns:
        stats['avg_sqft'] = df['total_sqft'].mean()
        stats['min_sqft'] = df['total_sqft'].min()
        stats['max_sqft'] = df['total_sqft'].max()
    
    if 'bhk' in df.columns:
        stats['bhk_distribution'] = df['bhk'].value_counts().to_dict()
    
    return stats


def create_feature_vector(sqft: float, bath: int, bhk: int, 
                         location: str, feature_columns: List[str]) -> np.ndarray:
    """
    Create feature vector for prediction
    
    Args:
        sqft: Square footage
        bath: Number of bathrooms
        bhk: Number of BHK
        location: Location name
        feature_columns: List of feature column names
        
    Returns:
        Feature vector as numpy array
    """
    x = np.zeros(len(feature_columns))
    
    # Set numeric features
    if 'total_sqft' in feature_columns:
        x[feature_columns.index('total_sqft')] = sqft
    if 'bath' in feature_columns:
        x[feature_columns.index('bath')] = bath
    if 'bhk' in feature_columns:
        x[feature_columns.index('bhk')] = bhk
    
    # Set location feature
    location_feature = f'location_{location}'
    if location_feature in feature_columns:
        x[feature_columns.index(location_feature)] = 1
    
    return x


def get_model_info() -> Dict[str, str]:
    """
    Get information about the model and application
    
    Returns:
        Dictionary containing model information
    """
    return {
        'name': 'Bangalore House Price Predictor',
        'version': '2.0',
        'model_type': 'Machine Learning Regression',
        'framework': 'Streamlit',
        'features': ['Location', 'Square Footage', 'Bathrooms', 'BHK'],
        'output': 'House price in lakhs (₹)'
    }
