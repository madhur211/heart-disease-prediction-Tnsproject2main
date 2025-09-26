TNS_PROJECT 2: Heart Disease Detection

This project implements heart disease detection using various machine learning models, packaged with both backend and deployment support using FastAPI and Docker.




Repository Contents
app.py – Contains both frontend and backend logic for deployment purposes.

main.py – Contains backend logic only, for model training and API setup.

Dockerfile – Used to containerize the application for easy deployment.

feature_columns.json – Lists the feature columns used by the model.

heart_disease_dataset.csv – Dataset containing patient details for model training and testing.

Heart_disease_rf_model.pkl – Random Forest model file for heart disease prediction.

models.ipynb – Jupyter notebook to experiment, train, and compare multiple classification models.

requirements.txt – Lists Python dependencies required to run the project.

scaler.pkl – Preprocessing scaler object used to normalize data before prediction.

start.sh – Shell script to automate starting the backend server and other setup tasks.

.streamlit/ – Directory for Streamlit configuration files as needed.





Prerequisites
Python 3.8 or above
(Optional) Docker if intended to use containerization



Steps to Download/Clone & Run

Clone the repository:
git clone https://github.com/your-username/TNS_PROJECT_2.git
cd TNS_PROJECT_2

Install dependencies:
pip install -r requirements.txt


Run the application (choose one method):

A) Direct Python execution
python app.py

B) Using Docker (recommended for deployment)
docker build -t heart-disease-app .
docker run -p 8000:8000 heart-disease-app

Access the app:

For FastAPI: Go to http://localhost:8000 in your browser.



Additional Notes
The models, scalers, and JSON files must remain in the repo root for code compatibility.

To retrain models or compare new ones, use models.ipynb.

Deployment config and scripts are directly provided (see Dockerfile and start.sh).


Used Render.com Cloud service for Deployment
Deployed on: https://tns-project-2.onrender.com

It will go in sleep mode after 15 minutes of activation, since no traffic
Mail/message me to start the web service 