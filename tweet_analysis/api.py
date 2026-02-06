from fastapi import FastAPI
from pydantic import BaseModel 
from fastapi import HTTPException
from contextlib import asynccontextmanager 
from tweet_analysis.inference import predict, get_spark

    
class Userinput(BaseModel):
    text: str 
    
@asynccontextmanager
async def warmup(app: FastAPI):
    try:
        _ = get_spark()
        print("warmup complete")
    except Exception as e:
        print("Warmup failed", repr(e))
    
app = FastAPI(title="Tweet Sentiment API", lifespan=warmup)
  
@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /predict"}
   
    
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predictor(text_input:Userinput):
    try:
        results = predict(text_input.text)
        return {
        'Tweet-sentiment': results.sentiment,
        'Label': results.label,
        'Positive-probability': results.positive_probability
        }
    except Exception as e:
        print('Prediction error', repr(e))
        raise HTTPException(status_code=500, details=str(e))
