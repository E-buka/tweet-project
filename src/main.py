# main.py
from src import start_spark, df_schema, TEXT_COL, LABEL_COL, DATE_COL, NUMERIC_COLS
from pyspark.storagelevel import StorageLevel
from src import text_cleaner, date_cleaner, target_cleaner
from src import AssembleFeatures, build_pipeline, set_estimator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.pipeline import PipelineModel
from src import auc_, f1_, accuracy_, precision_recall, save_to_json, confusion_matrix

def main():
    """Load, prepare, train adn save model 
    Then reload and predict data, and finally evaluate results 
    """
    # starting spark session and loading the data 
    spark = start_spark()
    df = (spark.read
          .format('csv')
          .option('encoding', 'ISO-8859-1')
          .option('header', 'true')
          .schema(df_schema)
          .load('tweets.csv')
         )
    
    #cleaning the dataframe
    df = text_cleaner(df)
    df = date_cleaner(df)
    df = target_cleaner(df)
    
    #selecting columns to keep
    clean_df = df.select(NUMERIC_COLS + [TEXT_COL, LABEL_COL])
    
    clean_df.persist(StorageLevel.MEMORY_AND_DISK) #persisting to memory
    _ = clean_df.count() #materialising
    
    #train test split
    train, test = clean_df.randomSplit(weights=[0.8, 0.2], seed=44)
    
    #feature preparations
    feature_assembler = AssembleFeatures(NUMERIC_COLS=NUMERIC_COLS, TEXT_COL=TEXT_COL,
                     LABEL_COL=LABEL_COL, persist_level=StorageLevel.MEMORY_AND_DISK)
    
    df_idf = feature_assembler.prepare_text(train)
    df_idf, weight_col = feature_assembler.add_weights(df_idf)
    
    #setting the estimator
    parameters = {'featuresCol': 'features', 
                'labelCol': LABEL_COL, 
                'weightCol': weight_col, 
                'maxIter': 100, 
                'regParam': 0.1,
                'elasticNetParam': 0.0
    }
    estimator = set_estimator(estimator=LogisticRegression(), **parameters)
    
    # building pipeline and fitting the model
    pipeline = build_pipeline(feature_assembler=feature_assembler, use_numeric=True, estimator = estimator)
    model = pipeline.fit(df_idf)
    
    ## saving the model
    model.write().overwrite().save("/tweet_project/models/full_model")
    
    # freeing up the memory
    feature_assembler.unpersist()
    clean_df.unpersist()
    
    #reloadingmodel
    trained_model = PipelineModel.load("/tweet_project/models/full_model")
    
    #prepare the test data and predict
    test_idf = feature_assembler.prepare_text(test)
    prediction = trained_model.transform(test_idf)

    # evaluate prediction results
    precision, recall = precision_recall(prediction, LABEL_COL)

    prediction_results = {
                       'AUC' : auc_(prediction, LABEL_COL),
                       'f1_score': f1_(prediction, LABEL_COL),
                       'accuracy': accuracy_(prediction, LABEL_COL),
                       'precision': precision,
                       'recall': recall
                       }
    
    save_to_json(path="/tweet_project/reports", **prediction_results)

    confusion_matrix = confusion_matrix(prediction, LABEL_COL)
    confusion_matrix.write.mode("overwrite").csv("/tweet_project/reports/confusion_matrix")


