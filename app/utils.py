import pandas as pd

def preprocess_input(
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
):
    """
    Convert user-friendly values into model-ready format.
    """

    sex = 1 if sex == "Male" else 0

    cp = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }[cp]

    fbs = 1 if fbs == "Yes" else 0

    restecg = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }[restecg]

    exang = 1 if exang == "Yes" else 0

    slope = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }[slope]

    thal = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }[thal]

    return pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])