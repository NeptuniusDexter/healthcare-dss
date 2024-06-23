import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
import streamlit.components.v1 as components

# Load the model and preprocessor
model_path = 'logistic_regression_model.joblib'
preprocessor_path = 'preprocessor.joblib'

# Load the model and preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Ensure preprocessor is a ColumnTransformer
if not isinstance(preprocessor, ColumnTransformer):
    st.error("Loaded preprocessor is not a ColumnTransformer. Check the preprocessor.joblib file.")
    st.stop()

# Streamlit app
st.title("Heart Disease Prediction")

# Define session state to manage pages
if 'page' not in st.session_state:
    st.session_state.page = 'main'

# Functions to switch pages
def show_main():
    st.session_state.page = 'main'

def show_data_cleaning():
    st.session_state.page = 'data_cleaning'

def show_data_visualisation():
    st.session_state.page = 'data_visualisation'

def show_model_training():
    st.session_state.page = 'model_training'

# Define layout based on the current page
if st.session_state.page == 'main':
    st.write("Please note this project is for educational purposes and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
    
    st.write("The following buttons will display the Jupyter Notebooks which were used to clean, visualise and preprocess the heart disease data set and train the machine learning models used in this application")

    st.button("Data Cleaning Notebook", on_click=show_data_cleaning)
    st.button("Data Visualisation Notebook", on_click=show_data_visualisation)
    st.button("Model Training Notebook", on_click=show_model_training)

    st.write("This is the Healthcare DSS. This app was designed to use trained learning models to assist health care professionals to make informed decisions regarding patient diagnosis")
    st.write("Please fill in the patient information below, click the diagnose button and the app will inform you whether the patient likely suffers from heart disease or not")

    # Input fields for patient information
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (in mg/dl)", min_value=100, max_value=600, value=180)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], format_func=lambda x: ["Normal", "Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy"][x])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=175)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

    # Diagnose button
    if st.button("Diagnose"):
        # Create a dataframe from the input
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })

        # Apply preprocessing
        try:
            input_data_preprocessed = preprocessor.transform(input_data)
        except Exception as e:
            st.error(f"Error in preprocessing the input data: {e}")
            st.stop()

        # Make prediction
        try:
            prediction = model.predict(input_data_preprocessed)
            prediction_proba = model.predict_proba(input_data_preprocessed)
        except Exception as e:
            st.error(f"Error in making prediction: {e}")
            st.stop()

        # Display the result with different colors
        if prediction[0] == 1:
            st.markdown(
                '<p style="color:white;background-color:rgba(128, 0, 0, 0.2);border:solid red 1px;padding: 10px;border-radius: 15px">The patient is likely to have heart disease.</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p style="color:white;background-color:rgba(12, 140, 0, 0.2);border:solid green 1px;padding: 10px;border-radius: 15px">The patient is unlikely to have heart disease.</p>',
                unsafe_allow_html=True
            )

elif st.session_state.page == 'data_cleaning':
    st.button("Back to Main", on_click=show_main)
    components.html(open("HealthcareDataProcessing.html").read(), height=800, scrolling=True)

elif st.session_state.page == 'data_visualisation':
    st.button("Back to Main", on_click=show_main)
    components.html(open("DataVisualisation.html").read(), height=800, scrolling=True)

elif st.session_state.page == 'model_training':
    st.button("Back to Main", on_click=show_main)
    components.html(open("ModelTraining.html").read(), height=800, scrolling=True)
