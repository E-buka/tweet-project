from fastapi import FastAPI
from pydantic import BaseModel 
from tweet_analysis.inference import predict

app = FastAPI(title="Tweet Sentiment API")
    
class Userinput(BaseModel):
    text: str 
    
@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /predict"}
   
    
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predictor(text_input:Userinput):
    results = predict(text_input.text)
    return {
        'Tweet-sentiment': results.sentiment,
        'Label': results.label,
        'Positive-probability': results.positive_probability
    }
