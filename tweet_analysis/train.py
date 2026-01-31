
from tweet_analysis import start_spark, AssembleFeatures, add_weights
from tweet_analysis import TEXT_COL, LABEL_COL, NUMERIC_COLS
from pyspark.ml.classification import LogisticRegression

def set_estimator(estimator, **params):
    return estimator.setParams(**params)
    

def train():
    """ train model on training data and save the pipeline
    """
    spark = start_spark()
    
    train = spark.read.parquet('data/splits/train.parquet')
    
    train_w, weight_col = add_weights(train, LABEL_COL)
    
    estimator = set_estimator(LogisticRegression(),
                              featuresCol='features',
                              labelCol=LABEL_COL,
                              weightCol=weight_col,
                              maxIter=100,
                              regParam=0.1,
                              elasticNetParam=0.0
                              )
    feature_assembler = AssembleFeatures(NUMERIC_COLS=NUMERIC_COLS, 
                                         TEXT_COL=TEXT_COL, 
                                         LABEL_COL=LABEL_COL,
                                         use_numeric = True, 
                                         estimator=estimator
                                         )
    pipeline = feature_assembler.build_pipeline()
    
    model = pipeline.fit(train_w)
    
    model.write().overwrite().save("models/tweet_model")
    
if __name__ == "__main__":
    train()
