from fastapi import FastAPI
from pydantic import BaseModel 
from contextlib import asynccontextmanager
from tweet_analysis.inference import load_pipeline_model, predict
from tweet_analysis.schema import TEXT_COL, DATE_COL
from tweet_analysis.config import start_spark 

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
def root():
    return {"status": "ok", "message": "Use POST /predict"}
   
    
@app.get("/health")
def health():
    return {"status": "healthy"}


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
