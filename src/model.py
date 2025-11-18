"""
Model Training and Prediction Module
Handles model creation, training, evaluation, and predictions
"""

import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from typing import Dict, Any, Tuple, Optional

# Fix for older sklearn versions - add compatibility imports
try:
    import sklearn.linear_model.base
except (ImportError, AttributeError):
    # For newer sklearn versions, create the module path for backward compatibility
    import sklearn.linear_model._base
    sys.modules['sklearn.linear_model.base'] = sklearn.linear_model._base


class HousePriceModel:
    """Class to handle model training, evaluation, and prediction"""
    
    def __init__(self, model_type: str = 'linear_regression'):
        """
        Initialize the model
        
        Args:
            model_type: Type of model to use ('linear_regression', 'lasso', 'decision_tree')
        """
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        self.locations = None
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                   random_state: int = 42) -> None:
        """
        Split data into training and testing sets
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.feature_columns = X.columns.tolist()
        
    def train_linear_regression(self) -> LinearRegression:
        """
        Train a linear regression model
        
        Returns:
            Trained LinearRegression model
        """
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        return lr_model
    
    def train_lasso(self, alpha: float = 1.0) -> Lasso:
        """
        Train a Lasso regression model
        
        Args:
            alpha: Regularization strength
            
        Returns:
            Trained Lasso model
        """
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(self.X_train, self.y_train)
        return lasso_model
    
    def train_decision_tree(self, max_depth: int = None) -> DecisionTreeRegressor:
        """
        Train a decision tree model
        
        Args:
            max_depth: Maximum depth of the tree
            
        Returns:
            Trained DecisionTreeRegressor model
        """
        dt_model = DecisionTreeRegressor(max_depth=max_depth)
        dt_model.fit(self.X_train, self.y_train)
        return dt_model
    
    def train_with_grid_search(self) -> Dict[str, Any]:
        """
        Train multiple models with grid search and return the best one
        
        Returns:
            Dictionary containing best model and its score
        """
        # Define algorithms to try
        algos = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'lasso': {
                'model': Lasso(),
                'params': {
                    'alpha': [1, 2, 3, 5, 10],
                    'selection': ['random', 'cyclic']
                }
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(),
                'params': {
                    'criterion': ['mse', 'friedman_mse'],
                    'splitter': ['best', 'random']
                }
            }
        }
        
        scores = []
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        
        for algo_name, config in algos.items():
            gs = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=cv, 
                return_train_score=False
            )
            gs.fit(self.X_train, self.y_train)
            scores.append({
                'model': algo_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_,
                'grid_search': gs
            })
        
        # Return best performing model
        best_result = max(scores, key=lambda x: x['best_score'])
        self.model = best_result['grid_search'].best_estimator_
        
        return best_result
    
    def train(self) -> None:
        """Train the model based on the specified model type"""
        if self.model_type == 'linear_regression':
            self.model = self.train_linear_regression()
        elif self.model_type == 'lasso':
            self.model = self.train_lasso()
        elif self.model_type == 'decision_tree':
            self.model = self.train_decision_tree()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def evaluate(self) -> float:
        """
        Evaluate the model on test data
        
        Returns:
            R² score on test set
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.score(self.X_test, self.y_test)
    
    def predict_price(self, location: str, sqft: float, bath: int, bhk: int) -> float:
        """
        Predict house price based on input features
        
        Args:
            location: Location of the house
            sqft: Total square footage
            bath: Number of bathrooms
            bhk: Number of bedrooms
            
        Returns:
            Predicted price in lakhs
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded first")
        
        # Create feature array with zeros
        x = np.zeros(len(self.feature_columns))
        
        # Set the numeric features
        if 'total_sqft' in self.feature_columns:
            x[self.feature_columns.index('total_sqft')] = sqft
        if 'bath' in self.feature_columns:
            x[self.feature_columns.index('bath')] = bath
        if 'bhk' in self.feature_columns:
            x[self.feature_columns.index('bhk')] = bhk
        
        # Set the location feature if it exists
        location_feature = f'location_{location}'
        if location_feature in self.feature_columns:
            x[self.feature_columns.index(location_feature)] = 1
        
        return self.model.predict([x])[0]
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file
        
        Args:
            filepath: Path where the model should be saved
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HousePriceModel':
        """
        Load a trained model from a file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            HousePriceModel instance with loaded model
        """
        # Add sklearn compatibility modules before unpickling
        import sklearn.linear_model._base
        sys.modules['sklearn.linear_model.base'] = sklearn.linear_model._base
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Error loading model from {filepath}: {str(e)}")
        
        instance = cls()
        
        # Handle both old and new model formats
        if isinstance(model_data, dict):
            instance.model = model_data.get('model')
            instance.feature_columns = model_data.get('feature_columns', [])
        else:
            # Old format - just the model
            instance.model = model_data
            # Create default feature columns for backward compatibility
            instance.feature_columns = ['total_sqft', 'bath', 'bhk'] + [f'col_{i}' for i in range(240)]
        
        return instance
    
    def get_locations(self, X: pd.DataFrame) -> list:
        """
        Extract unique locations from the feature set
        
        Args:
            X: Features DataFrame with location dummy variables
            
        Returns:
            List of location names
        """
        location_cols = [col for col in X.columns if col.startswith('location_')]
        locations = [col.replace('location_', '') for col in location_cols]
        self.locations = ['other'] + locations  # 'other' is the dropped first category
        return self.locations
