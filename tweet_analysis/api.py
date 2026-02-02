from fastapi import FastAPI
from pydantic import BaseModel 
from contextlib import asynccontextmanager

from tweet_analysis import (start_spark, load_pipeline_model,
                            predict, PredictionResult)
from tweet_analysis import TEXT_COL, DATE_COL
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (PROJECT_ROOT / "models" / "tweet_model").as_posix()


@asynccontextmanager
async def lifespan(app: FastAPI):
    #start spark and load model
    spark = start_spark()
    model = load_pipeline_model(MODEL_DIR)
    yield
    #model = None
    
app = FastAPI(title="weet Sentiment API", lifespan=lifespan)
    
class PredictRequest(BaseModel):
    model_config = {"extra":"forbid"}
    text: str
    
@app.get("/")
def status_check():
    return {"status": "connected successfully"}

@app.post("/tweet")
async def tweet(tweet: PredictRequest):
    tweet = tweet.text
    date_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    
    df = spark.createDataFrame([{TEXT_COL:tweet, 
                                 DATE_COL:date_str}
                               ])
    result = predict(df=df, model=model)
    return result.json_result()