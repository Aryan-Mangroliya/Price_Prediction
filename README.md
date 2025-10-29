🏠 Price Prediction using Machine Learning

This project predicts housing prices based on various features using Scikit-Learn pipelines and a Random Forest Regressor.
It includes both training and inference functionality within a single script.


PRICE_PREDICTION/
│
├── housing.csv          # Dataset used for training/testing
├── main.py              # Main Python script (training + prediction)
├── model.pkl            # Trained model (auto-generated)
├── pipeline.pkl         # Data preprocessing pipeline (auto-generated)
└── output.csv           # Predictions on unseen data (auto-generated)

⚙️ How It Works
If no model exists, the script:
Loads the dataset (housing.csv)
Splits data using StratifiedShuffleSplit
Builds preprocessing pipelines for numeric and categorical columns
Trains a Random Forest Regressor
Saves the model and preprocessing pipeline as .pkl files
If a model already exists, the script:
Loads the saved model and pipeline
Transforms new input data (input.csv)
Generates predictions
Saves results to output.csv

🧠 Tech Stack
Python 3.13+
NumPy – numerical operations
Pandas – data manipulation
Scikit-learn – model training, evaluation, and pipelines
Joblib – model persistence


🧩 Key Features

Handles missing values using SimpleImputer
Scales numerical features with StandardScaler
Encodes categorical features with OneHotEncoder
Performs stratified sampling to preserve income distribution
Saves reusable model and pipeline files

✨ Author

Aryan Mangroliya
🎓 BCA Student | 📊 Data Science Learner | 💡 Python Developer


