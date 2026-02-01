from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml.pipeline import PipelineModel
from dataclasses import dataclass
from typing import Optional
from pyspark.sql import DataFrame
from datetime import datetime
import json

from tweet_analysis import start_spark 
from tweet_analysis import TEXT_COL, DATE_COL, LABEL_COL
from tweet_analysis import text_cleaner, date_cleaner


@dataclass
class PredictionResult:
    predictions_df: DataFrame 
    
    def first_row(self) -> dict:
        row = self.predictions_df.limit(1).collect()
        return row[0].asDict() if row else {}
    
    def show(self, n: int=5, truncate:bool = False) -> None:
        self.predictions_df.show(n=n, truncate=truncate)
        
    def json_result(self):
        predictions_ = self.predictions_df.select(
            F.col('prediction').cast('int').alias('label'),
            F.col('p_positive').alias('positive_probability'),
            vector_to_array('probability').alias('probability'), 
            vector_to_array('rawPrediction').alias('rawProbability')
        )
        row = predictions_.limit(1).collect()[0]
        result = row.asDict()
        json_result = json.dumps(result)
        print(json_result)

    
def load_pipeline_model(model_dir:str) -> PipelineModel:
    return PipelineModel.load(model_dir)


def get_tweet() -> DataFrame:
    """start spark session and take a user input
    capture timestamp of input and return a dataframe"""
    
    spark = start_spark()
    tweet = input("Please enter a tweet: ")
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
        
    return PredictionResult(predictions_df=prediction)
                    

if __name__ == "__main__":
    tweet = get_tweet()
    model = load_pipeline_model('models/tweet_model')
    prediction_result = predict(df=tweet, model=model)
    prediction_result.json_result()