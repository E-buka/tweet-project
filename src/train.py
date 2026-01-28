
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.ml.pipeline import Pipeline, PipelineModel
from .features import AssembleFeatures


def set_estimator(estimator, **params):
    estimator = estimator.setParams(params)
    return estimator

def build_pipeline(feature_assembler:AssembleFeatures, use_numeric=False, estimator=None):
    """ builds a model pipeline"""
    if use_numeric:
            numeric_vec, scaled_num = feature_assembler.numeric_assembler()
            assembler = VectorAssembler(inputCols=['tfidf_token', 'scaled_num_vector'], outputCol='features')
            pipeline = Pipeline(stages = [numeric_vec, scaled_num, assembler, estimator])
                
    else:
        assembler = VectorAssembler(inputCols=['tfidf_token'], outputCol='features')
        pipeline = Pipeline(stages= [assembler, estimator])
            
    return pipeline
    

def pipeline_fit(pipeline:Pipeline, df:DataFrame) -> PipelineModel:
    """ fits the model"""
    model = pipeline.fit(df)
    return model
    