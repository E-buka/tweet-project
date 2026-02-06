from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml.pipeline import PipelineModel
from dataclasses import dataclass
from typing import Optional, Union
from pyspark.sql import DataFrame
from datetime import datetime
import json
from functools import lru_cache

from tweet_analysis.config import start_spark 
from tweet_analysis.schema import TEXT_COL, DATE_COL, LABEL_COL
from tweet_analysis.preprocessing import text_cleaner, date_cleaner


@dataclass
class PredictionResult:
    label : Union[int, float] = None 
    positive_probability: float = None
    sentiment: str = None

    
    
    def json_result(self):
        result = {
            "label": self.label,
            "positive_probability": self.positive_probability,
            "sentiment": self.sentiment
        }
        return json.dumps(result)


@lru_cache(maxsize=1)    
def get_spark():
    return start_spark()
        
@lru_cache(maxsize=1)    
def load_pipeline_model(model_dir:str) -> PipelineModel:
   # _ = get_spark()
    return PipelineModel.load(model_dir)


def get_tweet(tweet: str, date_str: Optional[str]=None) -> DataFrame:
    """start spark session and take a user input
    capture timestamp of input and return a dataframe"""
    spark = get_spark()
    if date_str is None:
        date_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    
    return spark.createDataFrame([{TEXT_COL:tweet, 
                                 DATE_COL:date_str}
                               ])


def predict(df:DataFrame, model: PipelineModel, keep_cols: Optional[list[str]]=None) -> PredictionResult:
    """Take a raw impute dataframe as df
    and predict using a loaded spark pipelinemodel
    """
    df = date_cleaner(text_cleaner(df))
           
    prediction = model.transform(df)
    #getting positive probability
    prediction = prediction.withColumn("p_positive", vector_to_array(F.col("probability"))[1])      
    

    cols = []
    if keep_cols:
        cols.extend([c for c in keep_cols if c in prediction.columns])
    
    for c in ['prediction', "p_positive", 'probability', 'rawPrediction']:
        if c in prediction.columns:
            cols.append(c)
        
    if LABEL_COL in prediction.columns:
        cols.append(LABEL_COL)
        
    if cols:
        prediction = prediction.select(*cols)
    
    label = prediction.select(F.col('prediction').cast('int')).collect()[0][0]
    positive_probability = round(prediction.select(vector_to_array(F.col("probability"))[1]).collect()[0][0], 5)
    sentiment = "Positive" if label == 1 else "Negative"

    
    
        
    return PredictionResult(
                            label=label,
                            positive_probability = positive_probability,
                            sentiment = sentiment
                           )
                    

if __name__ == "__main__":
    pass # comment out to run
    # tweet = get_tweet(tweet="example tweet")
    # model = load_pipeline_model('models/tweet_model')
    # prediction_result = predict(df=tweet, model=model)
    # print(prediction_result.sentiment)
    # print(prediction_result.label)
    # print(prediction_result.positive_probability)
    