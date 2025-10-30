import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_column, cat_column):
    """Build preprocessing pipeline for numeric and categorical columns."""
    num_pipeline = Pipeline([
        ("Imputer", SimpleImputer(strategy="median")),
        ("Scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("OneHot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("Num", num_pipeline, num_column),
        ("Cat", cat_pipeline, cat_column)
    ])
    return full_pipeline


if not os.path.exists(MODEL_FILE):
    # 1️⃣ Load dataset
    housing = pd.read_csv("housing.csv")

    # 2️⃣ Stratified split based on income category
    housing['income_cat'] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_column = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_column = ["ocean_proximity"]

    # 3️⃣ Build and apply pipeline
    pipeline = build_pipeline(num_column, cat_column)
    housing_prepared = pipeline.fit_transform(housing_features)

    # 4️⃣ Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # 5️⃣ Cross-validation performance
    scores = cross_val_score(model, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-scores)
    print("\n📊 Cross-validation results:")
    print(f"Mean RMSE: {rmse_scores.mean():.2f}")
    print(f"Standard Deviation: {rmse_scores.std():.2f}")

    # 6️⃣ Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("\n✅ Model trained and saved successfully!")

    # 7️⃣ Visualizations
    plt.figure(figsize=(6, 4))
    sns.histplot(rmse_scores, kde=True, color="orange")
    plt.title("Cross-Validation RMSE Distribution")
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.show()

    # 8️⃣ Feature correlation heatmap
    corr_matrix = housing.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

else:
    # 🔍 Inference Section
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data["median_house_value"] = predictions

    # Save predictions
    input_data.to_csv("output.csv", index=False)
    print("\n📁 Inference complete. Results saved to output.csv")

    # Evaluation metrics if ground truth exists
    if "median_house_value" in input_data.columns:
        true_values = input_data["median_house_value"]
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)

        print("\n📈 Model Evaluation Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.2f}")
