#for independent running of app.py
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load model, scaler, feature columns (same as backend)
with open("Heart_disease_rf_model.pkl", "rb") as f_model:
    model = pickle.load(f_model)

with open("scaler.pkl", "rb") as f_scaler:
    scaler = pickle.load(f_scaler)

with open("feature_columns.json", "r") as f_feat:
    feature_columns = json.load(f_feat)

st.title("Heart Disease Prediction")



# User Input fields
age = st.number_input('Age', min_value=0, max_value=120, value=63, step=1, format="%d")
sex = st.radio('Sex', options=[0, 1], index=1, format_func=lambda x: 'Female' if x == 0 else 'Male')
resting_blood_pressure = st.number_input('Resting Blood Pressure', min_value=0, max_value=250, value=145, step=1, format="%d")
cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=233, step=1, format="%d")
fasting_blood_sugar = st.radio('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
max_heart_rate = st.number_input('Max Heart Rate Achieved',min_value=0, max_value=250, value=150, step=1, format="%d")
exercise_induced_angina = st.radio('Exercise Induced Angina', options=[0, 1],
                                   format_func=lambda x: 'No' if x == 0 else 'Yes')
st_depression = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=2.3, format="%.2f")
num_major_vessels = st.selectbox('Number of major vessels colored by fluoroscopy', options=[0, 1, 2, 3], format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
chest_pain_type = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3],format_func=lambda x: {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}[x])
resting_ecg = st.selectbox('Resting ECG', options=[0, 1, 2], format_func=lambda x: {0: 'Normal', 1: 'Having ST-T wave abnormality', 2: 'Showing probable or definite left ventricular hypertrophy'}[x])
st_slope = st.selectbox('Slope of the peak exercise', options=[0, 1, 2], format_func=lambda x: {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}[x])
thalassemia = st.selectbox('Thalassemia', options=[0, 1, 2, 3], format_func=lambda x: {0: 'Unknown', 1: 'Normal', 2: 'Fixed Defect',3: 'Reversible Defect'}[x])

if st.button('Predict'):
    try:
        # Your existing input processing and prediction code here
        input_dict = {
            "age": age,
            "sex": sex,
            "resting_blood_pressure": resting_blood_pressure,
            "cholesterol": cholesterol,
            "fasting_blood_sugar": fasting_blood_sugar,
            "max_heart_rate": max_heart_rate,
            "exercise_induced_angina": exercise_induced_angina,
            "st_depression": st_depression,
            "num_major_vessels": num_major_vessels,
            "chest_pain_type": chest_pain_type,
            "resting_ecg": resting_ecg,
            "st_slope": st_slope,
            "thalassemia": thalassemia
        }

        input_df = pd.DataFrame([input_dict])
        cat_cols = ['chest_pain_type', 'resting_ecg', 'st_slope', 'thalassemia']
        input_df = pd.get_dummies(input_df, columns=cat_cols)
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]
        numeric_cols = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'st_depression']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("Prediction: Heart Disease Present")
        else:
            st.success("Prediction: No Heart Disease")

        st.info(f"Prediction Probability: {prediction_proba * 100:.2f}%")

        # Display Input Summary
        chest_pain_type_labels = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
        resting_ecg_labels = {0: 'Normal', 1: 'Having ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}
        slope_labels = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
        thalassemia_labels = {0: 'Unknown', 1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}
        st.subheader("Input Summary")
        
        input_summary = {
            "Age": age,
            "Sex": "Male" if sex == 1 else "Female",
            "Resting BP": resting_blood_pressure,
            "Cholesterol": cholesterol,
            "Fasting Blood Sugar": "Yes" if fasting_blood_sugar == 1 else "No",
            "Max Heart Rate": max_heart_rate,
            "Exercise-Induced Angina": "Yes" if exercise_induced_angina == 1 else "No",
            "ST Depression": st_depression,
            "Num Major Vessels": num_major_vessels,
            "Chest Pain Type": chest_pain_type_labels[chest_pain_type],
            "Resting ECG": resting_ecg_labels[resting_ecg],
            "Slope": slope_labels[st_slope],
            "Thalassemia": thalassemia_labels[thalassemia]
        }
        st.table(pd.DataFrame(list(input_summary.items()), columns=["Feature", "Value"]))

        # Feature importance visualization (Random Forest example)
        st.subheader("Top Features Influencing Prediction")
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(5)
        fig, ax = plt.subplots()
        ax.barh(feature_imp_df["Feature"], feature_imp_df["Importance"], color="skyblue")
        ax.set_xlabel("Importance")
        ax.set_title("Top 5 Feature Importances")
        st.pyplot(fig)

        # Load saved model metrics from JSON file
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)

        # Display model performance metrics
        st.subheader("Model Performance on Test Data")
        st.table(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]))

    except Exception as e:
        st.error(f"Prediction error: {e}")













#for dependent of app.py and main.py
#docker using below code





# import streamlit as st
# import requests

# st.title("Heart Disease Prediction")



# age = st.number_input('Age', min_value=0.0, max_value=120.0, value=63.0, format="%.2f")
# sex = st.radio('Sex', options=[0, 1], index=1, format_func=lambda x: 'Female' if x == 0 else 'Male')
# resting_blood_pressure = st.number_input('Resting Blood Pressure', min_value=0.0, max_value=250.0, value=145.0, format="%.2f")
# cholesterol = st.number_input('Cholesterol', min_value=0.0, max_value=600.0, value=233.0, format="%.2f")
# fasting_blood_sugar = st.radio('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
# max_heart_rate = st.number_input('Max Heart Rate Achieved', min_value=0.0, max_value=250.0, value=150.0, format="%.2f")
# exercise_induced_angina = st.radio('Exercise Induced Angina', options=[0, 1])
# st_depression = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=2.3, format="%.2f")
# num_major_vessels = st.selectbox('Number of major vessels colored by fluoroscopy', options=[0, 1, 2, 3])
# chest_pain_type = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
# resting_ecg = st.selectbox('Resting ECG', options=[0, 1, 2])
# st_slope = st.selectbox('Slope of the peak exercise', options=[0, 1, 2])
# thalassemia = st.selectbox('Thalassemia', options=[0, 1, 2, 3])


# if st.button('Predict'):
#     data = {
#         "age": age,
#         "sex": sex,
#         "resting_blood_pressure": resting_blood_pressure,
#         "cholesterol": cholesterol,
#         "fasting_blood_sugar": fasting_blood_sugar,
#         "max_heart_rate": max_heart_rate,
#         "exercise_induced_angina": exercise_induced_angina,
#         "st_depression": st_depression,
#         "num_major_vessels": num_major_vessels,
#         "chest_pain_type": chest_pain_type,
#         "resting_ecg": resting_ecg,
#         "st_slope": st_slope,
#         "thalassemia": thalassemia
#     }

#     response = requests.post('http://127.0.0.1:8000/predict', json=data)

#     if response.status_code == 200:
#         result = response.json()
#         pred = result.get('prediction')
#         prob = result.get('probability_of_disease')
#         st.success(f"Prediction: {'Heart Disease Present' if pred == 1 else 'No Heart Disease'}")
#         st.info(f"Prediction Probability: {prob*100:.2f}%")
#     else:
#         st.error(f"Error: {response.text}")


