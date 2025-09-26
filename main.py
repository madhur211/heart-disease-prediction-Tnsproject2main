from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import json
import pandas as pd

# Load model, scaler, and feature columns
with open('Heart_disease_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

app = FastAPI()

# Input schema with raw categorical features (not one-hot encoded)
class HeartDiseaseInput(BaseModel):
    age: float
    sex: int
    resting_blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: int
    max_heart_rate: float
    exercise_induced_angina: int
    st_depression: float
    num_major_vessels: int
    chest_pain_type: int
    resting_ecg: int
    st_slope: int
    thalassemia: int

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    try:
        input_dict = data.dict()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical variables to match training features
        cat_cols = ['chest_pain_type', 'resting_ecg', 'st_slope', 'thalassemia']

        # Use pandas get_dummies to create one-hot encoding projecting on train categories
        input_df = pd.get_dummies(input_df, columns=cat_cols)

        # Ensure all training columns are in dataframe (add missing with 0)
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns per model training
        input_df = input_df[feature_columns]

        # Scale numeric features (assuming scaler was fit on these columns)
        numeric_cols = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'st_depression']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {"prediction": int(prediction), "probability_of_disease": round(float(probability), 4)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "Model loaded and API is healthy."}



















# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # import pickle
# # import json
# # import pandas as pd

# # # Load model, scaler, and feature columns
# # with open('../Heart_disease_rf_model.pkl', 'rb') as f:
# #     model = pickle.load(f)

# # with open('../scaler.pkl', 'rb') as f:
# #     scaler = pickle.load(f)

# # with open('../feature_columns.json', 'r') as f:
# #     feature_columns = json.load(f)

# # # Define input data schema based on features in heart disease dataset
# # # class HeartDiseaseInput(BaseModel):
# # #     age: int
# # #     sex: int
# # #     chestpaintype: int
# # #     restingbloodpressure: int
# # #     cholesterol: int
# # #     fastingbloodsugar: int
# # #     restingecg: int
# # #     maxheartrate: int
# # #     exerciseinducedangina: int
# # #     stdepression: float
# # #     slope: int
# # #     majorvessels: int
# # #     thalassemia: int

# # class HeartDiseaseInput(BaseModel):
# #     age: int
# #     sex: int
# #     chest_pain_type: int
# #     resting_blood_pressure: int
# #     cholesterol: int
# #     fasting_blood_sugar: int
# #     resting_ecg: int
# #     max_heart_rate: int
# #     exercise_induced_angina: int
# #     st_depression: float
# #     st_slope: int
# #     num_major_vessels: int
# #     thalassemia: int


# # app = FastAPI()

# # @app.get("/")
# # def root():
# #     return {"message": "Heart Disease Prediction API is running."}

# # @app.post("/predict")
# # def predict(data: HeartDiseaseInput):
# #     try:
# #         # Convert input to dict and dataframe
# #         input_dict = data.dict()
# #         input_df = pd.DataFrame([input_dict])

# #         # Reorder columns as per training
# #         input_df = input_df[feature_columns]

# #         # Scale features
# #         scaled_features = scaler.transform(input_df)

# #         # Predict
# #         prediction = model.predict(scaled_features)[0]

# #         # Optionally get prediction probability if needed
# #         probability = model.predict_proba(scaled_features)[0][1]

# #         return {"prediction": int(prediction), "probability_of_disease": round(float(probability), 4)}

# #     except Exception as e:
# #         raise HTTPException(status_code=400, detail=str(e))

# # @app.get("/health")
# # def health_check():
# #     return {"status": "Model loaded and API is healthy."}




# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pickle
# import json
# import pandas as pd

# # Load model, scaler, and feature columns
# with open('../Heart_disease_rf_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('../scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# with open('../feature_columns.json', 'r') as f:
#     feature_columns = json.load(f)

# # Define input data schema with underscore attribute names matching JSON keys
# class HeartDiseaseInput(BaseModel):
#     age: int
#     sex: int
#     chest_pain_type: int
#     resting_blood_pressure: int
#     cholesterol: int
#     fasting_blood_sugar: int
#     resting_ecg: int
#     max_heart_rate: int
#     exercise_induced_angina: int
#     st_depression: float
#     st_slope: int
#     num_major_vessels: int
#     thalassemia: int

# app = FastAPI()

# @app.get("/")
# def root():
#     return {"message": "Heart Disease Prediction API is running."}

# @app.post("/predict")
# def predict(data: HeartDiseaseInput):
#     try:
#         # Convert input to dict and dataframe
#         input_dict = data.dict()
#         input_df = pd.DataFrame([input_dict])
        
#         # Rename columns to match model's expected feature names (no underscores)
#         # rename_map = {
#         #     "chest_pain_type": "chestpaintype",
#         #     "resting_blood_pressure": "restingbloodpressure",
#         #     "fasting_blood_sugar": "fastingbloodsugar",
#         #     "max_heart_rate": "maxheartrate",
#         #     "exercise_induced_angina": "exerciseinducedangina",
#         #     "st_depression": "stdepression",
#         #     "st_slope": "slope",
#         #     "num_major_vessels": "majorvessels",
#         #     "resting_ecg": "restingecg",
#         #     "age": "age",
#         #     "sex": "sex",
#         #     "cholesterol": "cholesterol",
#         #     "thalassemia": "thalassemia"
#         # }
#         # input_df = input_df.rename(columns=rename_map)
#         # print("Columns in input_df after rename:", input_df.columns.tolist())
#         # print("Feature columns expected by model:", feature_columns)

#         # Reorder columns as per training
#         input_df = input_df[feature_columns]

#         # Scale features
#         scaled_features = scaler.transform(input_df)

#         # Predict
#         prediction = model.predict(scaled_features)[0]

#         # Optionally get prediction probability
#         probability = model.predict_proba(scaled_features)[0][1]

#         return {"prediction": int(prediction), "probability_of_disease": round(float(probability), 4)}

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.get("/health")
# def health_check():
#     return {"status": "Model loaded and API is healthy."}
