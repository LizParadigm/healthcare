import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

# Configuración de CORS
origins = ['*']

app = FastAPI(title='Stroke Prediction API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Cargar el modelo
model = load(pathlib.Path('model/stroke_model.joblib'))

# Definir el modelo de entrada
class InputData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

# Definir el modelo de salida
class OutputData(BaseModel):
    stroke_prediction: int

@app.post('/predict', response_model=OutputData)
def predict(data: InputData):
    # Procesar la entrada y convertirla en un formato adecuado para el modelo
    input_data = pd.DataFrame({
        'gender': [data.gender],
        'age': [data.age],
        'hypertension': [data.hypertension],
        'heart_disease': [data.heart_disease],
        'ever_married': [data.ever_married],
        'work_type': [data.work_type],
        'Residence_type': [data.Residence_type],
        'avg_glucose_level': [data.avg_glucose_level],
        'bmi': [data.bmi],
        'smoking_status': [data.smoking_status]
    })

    # Realizar codificación de variables categóricas
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Alinear las columnas con el modelo
    model_input = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Realizar la predicción
    prediction = model.predict(model_input)[0]

    return OutputData(stroke_prediction=int(prediction))
