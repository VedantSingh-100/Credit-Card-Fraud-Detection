from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

app = FastAPI()

class TransactionData(BaseModel):
    features: list

input_dim = 31
model = torch.jit.load("model_scripted.pt", map_location=torch.device("cpu"))
model.eval()

@app.post("/predict")
def predict(transaction: TransactionData):
    try:
        features = np.array(transaction.features, dtype=np.float32)
        features = torch.tensor(features).unsqueeze(0)
        with torch.no_grad():
            output = model(features)
        prediction = output.item()
        return {"fraud_probability": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))