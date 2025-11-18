# 🏠 Bangalore House Price Prediction Web App

A **modular** machine learning web application built with **Streamlit** that predicts house prices in Bangalore based on location, area, bathrooms, and BHK configuration.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## 📑 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Module Documentation](#module-documentation)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## ✨ Features

- **🎨 Modern Web Interface**: Clean, intuitive UI built with Streamlit
- **⚡ Real-time Predictions**: Get instant house price predictions in lakhs (₹)
- **📊 Input Validation**: Smart validation of user inputs
- **🤖 Machine Learning Model**: Trained on Bangalore house price dataset
- **🔧 Modular Architecture**: Easy to maintain and extend
- **📱 Responsive Design**: Works seamlessly across devices

## 📁 Project Structure

```
House-Price-Prediction-Web-App--Using-Streamlit/
│
├── 📂 src/                          # Source code modules
│   ├── data_preprocessing.py        # Data loading and preprocessing
│   ├── model.py                     # Model training and prediction
│   └── utils.py                     # Helper functions
│
├── 📂 config/                       # Configuration files
│   └── config.py                    # Centralized configuration
│
├── 📂 data/                         # Data storage
│   ├── Bengaluru_House_Data.csv    # Raw dataset
│   └── processed_data.csv          # Generated after training
│
├── 📂 models/                       # Trained models
│   ├── model_pickel                # Pre-trained model
│   └── house_price_model.pkl       # Generated after training
│
├── 📂 notebooks/                    # Jupyter notebooks
│   └── House price Prediction.ipynb # EDA and exploration
│
├── 📄 app.py                        # Main Streamlit application
├── 📄 train.py                      # Model training script
├── 📄 requirements.txt              # Dependencies
└── 📄 README.md                     # Documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/pratyushsrivastava500/House-Price-Prediction-Web-App--Using-Streamlit.git
   cd House-Price-Prediction-Web-App--Using-Streamlit
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Quick Start (Use Pre-trained Model)

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Train Your Own Model

```bash
# Train the model
python train.py

# Run the application
streamlit run app.py
```

## 📚 Module Documentation

### `src/data_preprocessing.py`
Handles all data preprocessing operations including loading, cleaning, feature engineering, and outlier removal.

**Key Methods:**
- `preprocess_full_pipeline()`: Complete preprocessing pipeline
- `get_location_dummies()`: Create features for modeling

### `src/model.py`
Manages model training, evaluation, and predictions.

**Key Methods:**
- `train()`: Train the model
- `predict_price()`: Make price predictions
- `save_model()` / `load_model()`: Model persistence

### `src/utils.py`
Utility functions for validation, formatting, and helper operations.

**Key Functions:**
- `validate_inputs()`: Validate user inputs
- `format_price()`: Format price display

### `config/config.py`
Centralized configuration for paths, parameters, and settings.

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Programming language |
| **Streamlit** | Web application framework |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning |
| **Matplotlib** | Data visualization |

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">

**Made with ❤️ using Python and Streamlit**

If you found this project helpful, please give it a ⭐!

</div>
