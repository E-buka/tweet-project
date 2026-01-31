
from tweet_analysis import start_spark, LABEL_COL
from pyspark.ml.pipeline import PipelineModel
from tweet_analysis import (auc_, f1_, accuracy_, precision_recall, 
                            save_to_json, confusion_matrix)

def main():
    """Load test data and predict, then evaluate results 
    """
    spark = start_spark()
    test = spark.read.parquet("data/splits/test.parquet")

    tweet_model = PipelineModel.load("models/tweet_model")
    
    prediction = tweet_model.transform(test)

    # evaluate prediction results
    precision, recall = precision_recall(prediction, LABEL_COL)

    results = {'AUC' : auc_(prediction, LABEL_COL),
               'f1_score': f1_(prediction, LABEL_COL),
               'accuracy': accuracy_(prediction, LABEL_COL),
               'precision': precision,
                'recall': recall
                }
    
    save_to_json(path="reports/", **results)

    cm = confusion_matrix(prediction, LABEL_COL)
    cm.write.mode("overwrite").parquet("reports/confusion_matrix")

if __name__ == "__main__":
    main()