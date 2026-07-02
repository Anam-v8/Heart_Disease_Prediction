# ❤️ Heart Disease Prediction System

## Overview

The **Heart Disease Prediction System** is a Machine Learning web application that predicts whether a patient is at risk of heart disease based on clinical parameters. The application uses a trained **Random Forest Classifier** and provides an intuitive web interface built with **Streamlit** for real-time predictions.

---

## Features

* Interactive and user-friendly Streamlit interface
* Heart disease risk prediction using Machine Learning
* Random Forest Classifier with hyperparameter tuning
* Patient information summary
* Prediction confidence score
* Health recommendations based on prediction
* Feature importance visualization
* Responsive and modern UI

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Pickle

---

## Project Structure

```text
Heart_Disease_Prediction/
│
├── app/
│   ├── app.py
│   ├── utils.py
│   └── style.css
│
├── data/
│   └── heart.csv
│
├── models/
│   ├── heart_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── Heart_Disease.ipynb
│
├── images/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Scaling
6. Model Training
7. Hyperparameter Tuning using GridSearchCV
8. Model Evaluation
9. Model Serialization
10. Streamlit Application Development

---

## Model Performance

| Model                    | Test Accuracy |
| ------------------------ | ------------: |
| Random Forest Classifier |     **83.6%** |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Heart_Disease_Prediction.git
```

Navigate to the project directory:

```bash
cd Heart_Disease_Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

---

## Future Improvements

* Deploy the application to Streamlit Community Cloud
* Add support for multiple machine learning models
* Improve prediction explanations using SHAP values
* Store prediction history in a database
* Add user authentication for personalized dashboards

---

## Disclaimer

This application is intended for educational purposes only and should not be used as a substitute for professional medical diagnosis or treatment.

---

## Developer

**Anamika**

Aspiring Software Developer | Machine Learning Enthusiast
