import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_column,cat_column):

    #for numerical columns
    num_pipeline=Pipeline([
        ("Imputer",SimpleImputer(strategy="median")),
        ("Scler",StandardScaler())
    ])

    #for categorical columns

    cat_pipeline=Pipeline([
        ("OneHot",OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct The Full Pipeline

    full_pipeline=ColumnTransformer([
        ("Num",num_pipeline,num_column),
        ("Cat",cat_pipeline,cat_column)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    #Let's Train The Model  
    # 1. Load The Dataset

    housing= pd.read_csv("housing.csv")

    # 2. Create a stratified test set

    housing['income_cat']= pd.cut(housing["median_income"],
                                bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                                labels=[1,2,3,4,5])

    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index,test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv",index=False)
        housing=housing.loc[train_index].drop("income_cat",axis=1)
        
    
    housing_labels=housing['median_house_value'].copy()
    housing_features=housing.drop("median_house_value",axis=1)

    num_column=housing_features.drop("ocean_proximity",axis=1).columns.tolist() 
    cat_column=["ocean_proximity"]

    pipeline=build_pipeline(num_column,cat_column)
    housing_prepared=pipeline.fit_transform(housing_features)
    
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print("Model is Trained Congrats!")

else:
    # Let's Do Inference

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data=pd.read_csv("input.csv")
    transformed_data=pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data["median_house_value"] = predictions
 
    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")
    
