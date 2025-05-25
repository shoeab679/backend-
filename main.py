from fastapi import FastAPI
from adaptive_learning_system import AdaptiveLearningSystem

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "EduSaarthi Backend is running"}

@app.get("/predict")
def predict_dummy():
    # You can later integrate with your class methods
    return {"result": "Prediction or Recommendation logic goes here"}
