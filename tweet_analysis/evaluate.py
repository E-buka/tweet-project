from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import functions as F
import json
from .schema import LABEL_COL
from pyspark.sql import DataFrame

def auc_(prediction: DataFrame, LABEL_COL):
    """ auc evaluation """
    auc_evaluator = BinaryClassificationEvaluator(
                        rawPredictionCol='rawPrediction',
                        labelCol=LABEL_COL,
                        metricName='areaUnderROC')
    return auc_evaluator.evaluate(prediction)
    
def f1_(prediction: DataFrame, LABEL_COL):
    """ F1-score evaluation """
    f1_evaluator = MulticlassClassificationEvaluator(
                            predictionCol='prediction', 
                            labelCol=LABEL_COL,
                            metricName='f1')
    return f1_evaluator.evaluate(prediction)
    
def accuracy_(prediction: DataFrame, LABEL_COL):
    """ accuracy evaluation """
    acc_evaluator = MulticlassClassificationEvaluator(
                            predictionCol='prediction', 
                            labelCol=LABEL_COL,
                            metricName='accuracy')
    return acc_evaluator.evaluate(prediction)

def precision_recall(prediction:DataFrame, LABEL_COL):
    """ precision recall evaluator"""
    tp_ = prediction.filter((F.col(LABEL_COL) == 1) & (F.col('prediction') == 1)).count()
    tn_ = prediction.filter((F.col(LABEL_COL) == 0) & (F.col('prediction') == 0)).count()
    fp_ = prediction.filter((F.col(LABEL_COL) == 0) & (F.col('prediction') == 1)).count()
    fn_ = prediction.filter((F.col(LABEL_COL) == 1) & (F.col('prediction') == 0)).count()

    precision_= tp_ / (tp_ + fp_) if (tp_ + fp_) != 0 else 0  

    recall_ = tp_ / (tp_ + fn_) if (tp_ + fn_) != 0 else 0.0  
    return precision_, recall_

def save_to_json(path:str=None, **kwargs):
    if not path:
        raise ValueError('Provide a path')
    path = path+'.json' 
    with open(path, "w") as f:
        json.dump(kwargs, f)
    

def confusion_matrix(prediction:DataFrame, LABEL_COL): 
    cm = (prediction
          .select(F.col(LABEL_COL).cast('int').alias('label'), 
                  F.col('prediction').cast('int').alias('predicted'))
          .groupBy('label', 'predicted')
          .count()
         )
    confusion_matrx = (cm
                        .groupBy('label')
                        .pivot('predicted')
                        .sum('count')
                        .na.fill(0)
                        .orderBy('label')
                       )
    return confusion_matrx
