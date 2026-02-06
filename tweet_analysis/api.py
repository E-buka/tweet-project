from fastapi import FastAPI
from pydantic import BaseModel 
from contextlib import asynccontextmanager
from tweet_analysis import (start_spark, load_pipeline_model,
                            predict, PredictionResult)
from tweet_analysis import TEXT_COL, DATE_COL
from datetime import datetime, timezone
from pathlib import Path 


spark = None
model = None

BASE_DIR = Path(__file__).resolve().parents[1]
model_dir = BASE_DIR / "models" / "tweet_model"
 

@asynccontextmanager
async def lifespan(app: FastAPI):
    #start spark and load model
    global spark, model
    spark = start_spark()
    model = load_pipeline_model(str(model_dir))
    yield
    model = None
    spark = None
    
app = FastAPI(title="Tweet Sentiment API", lifespan=lifespan)
    
class Userinput(BaseModel):
    text: str 
    
@app.get("/")
def status_check():
    return {"status": "connected successfully"}


@app.post("/predict")
async def predictor(text_input:Userinput):
    tweet = text_input.text
    date_str = datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S %Y")
    
    df = spark.createDataFrame([{TEXT_COL:tweet, 
                                 DATE_COL:date_str}
                               ])
    results = predict(df=df, model=model)
    
    return {
        'Tweet-sentiment': results.sentiment,
        'Label': results.label,
        'Positive-probability': results.positive_probability
    }
