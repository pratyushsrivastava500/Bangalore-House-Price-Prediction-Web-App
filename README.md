# 🏠 Bangalore House Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> A machine learning web application that predicts house prices in Bangalore using Linear Regression. Built with clean architecture and modular design for production deployment.

## 📋 Overview

The Bangalore House Price Prediction app enables users to:

- **Predict Prices** for properties based on location, area, BHK, and bathrooms
- **Get Instant Results** with real-time ML predictions
- **Analyze Market Trends** across 240+ Bangalore localities
- **Make Informed Decisions** for buying or selling properties

## ✨ Features

### 🎯 ML-Powered Predictions
- Accurate price estimation using Linear Regression
- Real-time predictions with sub-second response
- Supports 240+ Bangalore localities
- Handles 1-10 BHK configurations

### 🏗️ Clean Architecture
- Modular design with separation of concerns
- Type hints and comprehensive docstrings
- Centralized configuration management
- Production-ready error handling

### 💻 User Experience
- Clean, intuitive Streamlit interface
- Input validation with helpful error messages
- Dual currency display (Lakhs/Crores)
- Responsive design for all devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pratyushsrivastava500/House-Price-Prediction-Web-App--Using-Streamlit.git
   cd House-Price-Prediction-Web-App--Using-Streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - Navigate to `http://localhost:8501`

### Training a New Model

```bash
python train.py
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      Streamlit Web Interface        │
│  • User inputs (location, sqft)    │
│  • Display predictions              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Utility Layer                │
│  • Input validation                 │
│  • Price formatting                 │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Model Layer                  │
│  • Load trained model               │
│  • Make predictions                 │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    Data Preprocessing Layer         │
│  • Feature engineering              │
│  • Location encoding                │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Configuration Layer            │
│  • Paths & parameters               │
└─────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.28+ |
| **ML Model** | Scikit-learn (Linear Regression) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib (EDA) |
| **Model Persistence** | Pickle |

## 📁 Project Structure

```
House-Price-Prediction-Web-App--Using-Streamlit/
├── app.py                      # Main Streamlit application
├── train.py                    # Model training script
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore patterns
├── config/
│   └── config.py              # Configuration settings
├── src/
│   ├── data_preprocessing.py  # Data pipeline
│   ├── model.py               # ML model management
│   └── utils.py               # Utility functions
├── data/
│   └── Bengaluru_House_Data.csv  # Training dataset
├── models/
│   └── model_pickel           # Trained model
└── notebooks/
    └── House price Prediction.ipynb  # EDA notebook
```

## 📊 Dataset Information

**Source:** Bangalore House Price Data

**Statistics:**

| Attribute | Details |
|-----------|---------|
| **Records** | 13,320 properties |
| **Features** | 9 columns |
| **Target Variable** | Price (in Lakhs ₹) |

**Key Features:**

| Feature | Description | Type | Example |
|---------|-------------|------|---------|
| `location` | Property locality | Categorical | Whitefield, Electronic City |
| `size` | Number of BHK | Categorical | 2 BHK, 3 BHK |
| `total_sqft` | Total area | Numeric | 1200, 1500 |
| `bath` | Number of bathrooms | Numeric | 2, 3 |
| `balcony` | Number of balconies | Numeric | 1, 2 |
| `price` | Price in Lakhs | Numeric | 75.5, 120.0 |

**Preprocessing Steps:**
- Removed duplicates and missing values
- Outlier detection using domain knowledge
- Feature engineering (price per sqft, BHK extraction)
- Location encoding with one-hot encoding
- Final features: 243 (3 numeric + 240 location dummies)

## 📖 Usage Guide

### Making Predictions

1. **Enter Property Details:**
   - Location (e.g., "Whitefield", "Electronic City")
   - Total area in square feet
   - Number of bedrooms (BHK)
   - Number of bathrooms

2. **Get Prediction:**
   - Click "Predict Price"
   - View estimated price in Lakhs and Crores

3. **Analyze Results:**
   - Review input summary
   - Check price reasonability

### Example Queries

**2 BHK in Whitefield:**
```
Location: Whitefield
Area: 1200 sqft
BHK: 2
Bathrooms: 2
Result: ₹75.50 Lakhs
```

**3 BHK in Electronic City:**
```
Location: Electronic City
Area: 1500 sqft
BHK: 3
Bathrooms: 3
Result: ₹65.20 Lakhs
```

## 🤖 Model Performance

**Algorithm:** Linear Regression

| Metric | Value |
|--------|-------|
| **R² Score (Test)** | 0.84 |
| **MAE** | 12.3 Lakhs |
| **RMSE** | 18.7 Lakhs |
| **Features** | 243 (3 numeric + 240 location dummies) |
| **Training Time** | < 1 second |

**Top Predictive Features:**

1. Total Square Feet (40%)
2. Location (35%)
3. BHK (15%)
4. Bathrooms (10%)

## 🔮 Future Enhancements

- [ ] Add more ML models (Random Forest, XGBoost)
- [ ] Implement hyperparameter tuning
- [ ] Add property age and amenities features
- [ ] Create interactive location map
- [ ] Deploy to cloud (Streamlit Cloud/AWS)
- [ ] Add user authentication
- [ ] REST API development
- [ ] Mobile app version

## 🔧 Troubleshooting

**Issue: Streamlit not found**
```bash
pip install streamlit
```

**Issue: Module import errors**
```bash
pip install -r requirements.txt
```

**Issue: Model file not found**
```bash
python train.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Bangalore House Price Dataset** contributors
- **Scikit-learn** for ML algorithms
- **Streamlit** for the UI framework
- **Pandas & NumPy** for data processing

## 📧 Contact

For questions or support, please open an issue on GitHub.

⚠️ **Disclaimer:** This tool is for informational purposes only and should not replace professional real estate advice or property valuation services.

---

<div align="center">

**Made with ❤️ and Python | © 2025 Pratyush Srivastava**

</div>
