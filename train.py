"""
Training Script
Script to train the house price prediction model from scratch
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import DataPreprocessor
from src.model import HousePriceModel
from config.config import Config


def main():
    """Main training function"""
    print("=" * 60)
    print("House Price Prediction Model Training")
    print("=" * 60)
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Step 1: Load and preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    data_path = Config.get_data_path()
    
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        return
    
    preprocessor = DataPreprocessor(str(data_path))
    
    try:
        cleaned_df = preprocessor.preprocess_full_pipeline()
        print(f"✓ Data preprocessed successfully. Shape: {cleaned_df.shape}")
        
        # Save processed data
        processed_data_path = Config.PROCESSED_DATA_FILE
        cleaned_df.to_csv(processed_data_path, index=False)
        print(f"✓ Processed data saved to {processed_data_path}")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return
    
    # Step 2: Prepare features and target
    print("\n[2/4] Preparing features and target variables...")
    try:
        X, y = preprocessor.get_location_dummies()
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"✓ Number of locations: {preprocessor.cleaned_df['location'].nunique()}")
    except Exception as e:
        print(f"❌ Error preparing features: {e}")
        return
    
    # Step 3: Train model
    print("\n[3/4] Training model...")
    model = HousePriceModel(model_type=Config.MODEL_TYPE)
    
    try:
        # Split data
        model.split_data(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
        print(f"✓ Data split - Train: {len(model.X_train)}, Test: {len(model.X_test)}")
        
        # Train model
        model.train()
        print(f"✓ Model trained successfully")
        
        # Evaluate
        score = model.evaluate()
        print(f"✓ Model R² Score: {score:.4f}")
        
        # Get locations
        locations = model.get_locations(X)
        print(f"✓ Total locations: {len(locations)}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Step 4: Save model
    print("\n[4/4] Saving model...")
    try:
        model_path = Config.MODEL_FILE_NEW
        model.save_model(str(model_path))
        print(f"✓ Model saved to {model_path}")
        
        # Also save to old location for backward compatibility
        model.save_model(str(Config.MODEL_FILE))
        print(f"✓ Model also saved to {Config.MODEL_FILE} (backward compatibility)")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Training completed successfully!")
    print("=" * 60)
    print(f"\nModel Performance:")
    print(f"  - R² Score: {score:.4f}")
    print(f"  - Training samples: {len(model.X_train)}")
    print(f"  - Test samples: {len(model.X_test)}")
    print(f"  - Features: {len(model.feature_columns)}")
    print(f"  - Locations: {len(locations)}")
    print(f"\nYou can now run the Streamlit app using:")
    print(f"  streamlit run app.py")


if __name__ == "__main__":
    main()
