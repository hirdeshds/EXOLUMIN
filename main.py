from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/Random_forest_exo.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

class InputData(BaseModel):
    koi_score: float
    koi_fpflag_nt: int
    koi_fpflag_ss: int
    koi_fpflag_co: int
    koi_fpflag_ec: int
    koi_period: float
    koi_time0bk: float
    koi_impact: float
    koi_duration: float
    koi_depth: float
    koi_prad: float
    koi_teq: float
    koi_insol: float
    koi_model_snr: float
    koi_steff: float
    koi_slogg: float
    koi_srad: float

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    
    missing_cols = [col for col in columns if col not in input_dict]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing feature(s): {missing_cols}")
    
    features = np.array([input_dict[col] for col in columns]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    pred = model.predict(features_scaled)[0]
    pred_proba = model.predict_proba(features_scaled)[0][1]
    
    return {"prediction": int(pred), "probability": float(pred_proba)}
