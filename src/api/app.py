from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("models/xgboost_model.pkl")

class PatientData(BaseModel):
    age: float
    bmi: float
    glucose: float
    pollution_level: float

@app.post("/predict")
async def predict_risk(patient: PatientData):
    data = [[patient.age, patient.bmi, patient.glucose, patient.pollution_level]]
    risk = model.predict_proba(data)[0][1]
    return {"risk_score": float(risk)}
