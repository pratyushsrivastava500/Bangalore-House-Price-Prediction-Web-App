"""
Streamlit Web Application for Bangalore House Price Prediction
Modular version with clean architecture
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model import HousePriceModel
from src.utils import format_price, validate_inputs
from config.config import Config


# Page configuration
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    try:
        model_path = Config.get_model_path()
        model = HousePriceModel.load_model(str(model_path))
        return model, None
    except Exception as e:
        return None, str(e)


def main():
    """Main application function"""
    
    # Title and header
    st.title("🏠 Bangalore House Rate Prediction")
    
    html_temp = """
    <div style="background-color:#f0f2f6;padding:10px;border-radius:10px">
    <h2 style="color:#1f77b4;text-align:center;">ML-Powered House Price Prediction</h2>
    <p style="text-align:center;color:#666;">Get instant price predictions for properties in Bangalore</p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load model
    model, error = load_model()
    
    if error:
        st.error(f"⚠️ Error loading model: {error}")
        st.info("Please ensure the model file exists in the correct location.")
        return
    
    # Create two columns for input layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Property Details")
        location = st.text_input(
            "Location",
            placeholder="e.g., Whitefield, Electronic City, Rajaji Nagar",
            help="Enter the location/area in Bangalore"
        )
        
        sqft = st.text_input(
            "Total Area (Square Feet)",
            placeholder="e.g., 1200",
            help="Enter the total built-up area in square feet"
        )
    
    with col2:
        st.subheader("🏡 Property Configuration")
        bhk = st.text_input(
            "Number of BHK",
            placeholder="e.g., 2, 3, 4",
            help="Number of Bedrooms, Hall, and Kitchen"
        )
        
        bath = st.text_input(
            "Number of Bathrooms",
            placeholder="e.g., 2, 3",
            help="Total number of bathrooms"
        )
    
    st.markdown("---")
    
    # Prediction button
    col_button, col_empty = st.columns([1, 3])
    
    with col_button:
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    # Perform prediction
    if predict_button:
        if not all([location, sqft, bath, bhk]):
            st.warning("⚠️ Please fill in all fields before predicting.")
        else:
            # Validate inputs
            is_valid, error_msg, converted_values = validate_inputs(sqft, bath, bhk)
            
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                sqft_val, bath_val, bhk_val = converted_values
                
                try:
                    # Make prediction
                    with st.spinner("Calculating price..."):
                        predicted_price = model.predict_price(
                            location=location,
                            sqft=sqft_val,
                            bath=bath_val,
                            bhk=bhk_val
                        )
                    
                    # Display result
                    st.success("✅ Prediction Complete!")
                    
                    # Create result display
                    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                    
                    with result_col2:
                        st.markdown(
                            f"""
                            <div style="background-color:#d4edda;padding:20px;border-radius:10px;border:2px solid #28a745">
                            <h2 style="color:#155724;text-align:center;margin:0">
                            {format_price(predicted_price)}
                            </h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Show input summary
                    st.markdown("### 📊 Input Summary")
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    with summary_col1:
                        st.metric("Location", location)
                    with summary_col2:
                        st.metric("Area", f"{sqft_val} sqft")
                    with summary_col3:
                        st.metric("BHK", bhk_val)
                    with summary_col4:
                        st.metric("Bathrooms", bath_val)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;color:#666;padding:20px">
        <p>💡 <i>This prediction is based on machine learning models trained on Bangalore housing data.</i></p>
        <p><i>Actual prices may vary based on market conditions and property specifics.</i></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            """
            This application uses machine learning to predict house prices 
            in Bangalore based on:
            - Location
            - Total area (sq ft)
            - Number of bedrooms (BHK)
            - Number of bathrooms
            """
        )
        
        st.header("📖 Usage Tips")
        st.markdown(
            """
            - Enter accurate property details
            - Location names should match Bangalore areas
            - Typical area ranges: 500-5000 sqft
            - Price shown is in Lakhs (₹)
            """
        )
        
        st.header("🔧 Model Info")
        st.markdown(
            f"""
            - **Model Type:** Linear Regression
            - **Features:** 243
            - **Framework:** Scikit-learn
            - **Interface:** Streamlit
            """
        )


if __name__ == '__main__':
    main()
