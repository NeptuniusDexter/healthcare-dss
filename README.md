# Healthcare Decision Support System (DSS)

This is a Healthcare Decision Support System designed to assist healthcare professionals in diagnosing heart disease. The app uses machine learning models to predict whether a patient is likely to have heart disease based on their medical information.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Models](#models)
- [Contact](#contact)

## Features

- Predict heart disease likelihood based on patient data
- Data preprocessing and model training notebooks
- Data visualization for insights
- Easy-to-use Streamlit interface

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/NeptuniusDexter/healthcare-dss.git
    cd healthcare-dss
    ```

2. Open the project in Visual Studio Code.

3. Install the Dev Containers extension in Visual Studio Code if you haven't already:
    - Go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window or by pressing `Ctrl+Shift+X`.
    - Search for "Dev Containers" and install the extension by Microsoft.

4. Reopen the project in a Dev Container:
    - Press `F1` to open the Command Palette.
    - Type `Dev Containers: Reopen in Container` and select it.
    - VS Code will build the container and reopen the project inside it.

5. Make sure you have the following files in the project directory:
    - `logistic_regression_model.joblib`
    - `preprocessor.joblib`
    - `HealthcareDataProcessing.html`
    - `DataVisualisation.html`
    - `ModelTraining.html`

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false
    ```

2. Open your browser and go to `http://localhost:8501`

3. Fill in the patient information and click the "Diagnose" button to get a prediction.

4. Navigate to different notebooks by clicking the respective buttons:
    - Data Cleaning Notebook
    - Data Visualisation Notebook
    - Model Training Notebook


## Notebooks

- **Data Cleaning Notebook**: Contains the steps and code for cleaning and preprocessing the data.
- **Data Visualisation Notebook**: Provides visualizations for better understanding and insights of the data.
- **Model Training Notebook**: Details the training process for the machine learning models used in the prediction.

## Models

The project uses three machine learning models for heart disease prediction:

1. **Logistic Regression**:
    - Suitable for binary classification problems.
    - Provides probabilities for the predictions.

2. **Random Forest**:
    - Handles a variety of outcomes.
    - Robust and less likely to overfit.

3. **Support Vector Machine (SVM)**:
    - Effective for high-dimensional spaces.
    - Works well for binary classification tasks.

## Contact

For any questions or issues, please open an issue on this repository or contact:

- Heinrich Basson
- bmdnf2dl7@vossie.net

---

*This project is for educational purposes and should not be used as a substitute for professional medical advice, diagnosis, or treatment.*
