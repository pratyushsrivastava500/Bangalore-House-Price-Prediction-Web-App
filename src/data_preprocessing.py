"""
Data Preprocessing Module
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Tuple


class DataPreprocessor:
    """Class to handle all data preprocessing operations"""
    
    def __init__(self, data_path: str):
        """
        Initialize the DataPreprocessor
        
        Args:
            data_path: Path to the CSV file containing house data
        """
        self.data_path = data_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            DataFrame containing the raw data
        """
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def remove_unnecessary_columns(self) -> pd.DataFrame:
        """
        Remove columns that are not needed for prediction
        
        Returns:
            DataFrame with unnecessary columns removed
        """
        columns_to_drop = ['area_type', 'balcony', 'availability', 'society']
        existing_columns = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=existing_columns, errors='ignore')
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values by dropping rows with NaN
        
        Returns:
            DataFrame with no missing values
        """
        self.df = self.df.dropna()
        return self.df
    
    def extract_bhk(self) -> pd.DataFrame:
        """
        Extract BHK (number of bedrooms) from size column
        
        Returns:
            DataFrame with BHK column added
        """
        if 'size' in self.df.columns:
            self.df['bhk'] = self.df['size'].apply(lambda x: int(x.split(' ')[0]))
        return self.df
    
    @staticmethod
    def is_float(x):
        """
        Check if a value can be converted to float
        
        Args:
            x: Value to check
            
        Returns:
            Boolean indicating if conversion is possible
        """
        try:
            float(x)
            return True
        except:
            return False
    
    def convert_sqft_to_num(self, x):
        """
        Convert total_sqft to numeric value
        Handles ranges (e.g., '1000-1200') by taking average
        
        Args:
            x: Square footage value (string or number)
            
        Returns:
            Numeric square footage value
        """
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None
    
    def process_total_sqft(self) -> pd.DataFrame:
        """
        Process total_sqft column to ensure all values are numeric
        
        Returns:
            DataFrame with processed total_sqft column
        """
        if 'total_sqft' in self.df.columns:
            self.df['total_sqft'] = self.df['total_sqft'].apply(self.convert_sqft_to_num)
            self.df = self.df.dropna(subset=['total_sqft'])
        return self.df
    
    def add_price_per_sqft(self) -> pd.DataFrame:
        """
        Add a new feature: price per square foot
        
        Returns:
            DataFrame with price_per_sqft column
        """
        if 'price' in self.df.columns and 'total_sqft' in self.df.columns:
            # Price is in lakhs, convert to actual price and divide by sqft
            self.df['price_per_sqft'] = (self.df['price'] * 100000) / self.df['total_sqft']
        return self.df
    
    def remove_outliers_sqft_per_bhk(self) -> pd.DataFrame:
        """
        Remove outliers based on square footage per BHK
        Typical size per BHK should be around 300 sqft minimum
        
        Returns:
            DataFrame with outliers removed
        """
        if 'total_sqft' in self.df.columns and 'bhk' in self.df.columns:
            self.df = self.df[~(self.df['total_sqft'] / self.df['bhk'] < 300)]
        return self.df
    
    def remove_price_per_sqft_outliers(self) -> pd.DataFrame:
        """
        Remove outliers in price per square foot using standard deviation method
        
        Returns:
            DataFrame with price outliers removed
        """
        if 'price_per_sqft' not in self.df.columns:
            return self.df
            
        df_out = pd.DataFrame()
        for key, subdf in self.df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        self.df = df_out
        return self.df
    
    def remove_bhk_outliers(self) -> pd.DataFrame:
        """
        Remove outliers where price of lower BHK is higher than higher BHK
        in the same location
        
        Returns:
            DataFrame with BHK outliers removed
        """
        exclude_indices = []
        for location, location_df in self.df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = np.mean(bhk_df.price_per_sqft)
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats > bhk_stats[bhk]:
                    for idx, row in bhk_df.iterrows():
                        exclude_indices.append(idx)
        self.df = self.df.drop(exclude_indices)
        return self.df
    
    def reduce_location_cardinality(self, threshold: int = 10) -> pd.DataFrame:
        """
        Reduce the number of locations by grouping less frequent ones as 'other'
        
        Args:
            threshold: Minimum number of data points for a location to be kept
            
        Returns:
            DataFrame with reduced location categories
        """
        if 'location' not in self.df.columns:
            return self.df
            
        location_stats = self.df.groupby('location')['location'].agg('count').sort_values(ascending=False)
        locations_less_than_threshold = location_stats[location_stats <= threshold]
        self.df['location'] = self.df['location'].apply(
            lambda x: 'other' if x in locations_less_than_threshold else x
        )
        return self.df
    
    def preprocess_full_pipeline(self) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline
        
        Returns:
            Fully preprocessed DataFrame ready for model training
        """
        self.load_data()
        self.remove_unnecessary_columns()
        self.handle_missing_values()
        self.extract_bhk()
        self.process_total_sqft()
        self.add_price_per_sqft()
        self.remove_outliers_sqft_per_bhk()
        self.reduce_location_cardinality()
        self.remove_price_per_sqft_outliers()
        self.remove_bhk_outliers()
        
        # Drop helper columns
        if 'price_per_sqft' in self.df.columns:
            self.df = self.df.drop(['price_per_sqft'], axis=1)
        if 'size' in self.df.columns:
            self.df = self.df.drop(['size'], axis=1)
            
        self.cleaned_df = self.df
        return self.cleaned_df
    
    def get_location_dummies(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create dummy variables for location and prepare data for modeling
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.cleaned_df is None:
            raise ValueError("Data must be preprocessed first using preprocess_full_pipeline()")
        
        dummies = pd.get_dummies(self.cleaned_df['location'], drop_first=True)
        df_with_dummies = pd.concat([self.cleaned_df, dummies], axis=1)
        df_with_dummies = df_with_dummies.drop('location', axis=1)
        
        X = df_with_dummies.drop('price', axis=1)
        y = df_with_dummies['price']
        
        return X, y
