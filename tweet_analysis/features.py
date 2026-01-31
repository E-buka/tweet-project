from pyspark.ml.feature import (RegexTokenizer,
                                StopWordsRemover,
                                HashingTF,
                                IDF, 
                                VectorAssembler,
                                StandardScaler
                                )

from pyspark.ml import Pipeline
from pyspark.sql import functions as F


class AssembleFeatures:
   
    def __init__(self, NUMERIC_COLS=None, TEXT_COL=None, LABEL_COL=None,
                use_numeric = False, estimator=None):
        
        self.NUMERIC_COLS = NUMERIC_COLS
        self.LABEL_COL = LABEL_COL
        self.TEXT_COL = TEXT_COL

        self.use_numeric = use_numeric
        self.estimator = estimator
        
     

    def numeric_assembler(self):
        """assemble and scale numeric features
        """
        numeric_vec = VectorAssembler(inputCols=self.NUMERIC_COLS,
                                      outputCol='num_features', handleInvalid='keep')
        scaled_num = StandardScaler(inputCol='num_features', outputCol='scaled_num_vector', withMean=False)
        return numeric_vec, scaled_num

    def prepare_text(self):
        """ tokenize, filter, hash and transform text
        """
        tokenizer = RegexTokenizer(inputCol = self.TEXT_COL, 
                               outputCol='word_token', 
                              pattern='\\W+', 
                              minTokenLength=2)
    
        token_filter = StopWordsRemover(inputCol='word_token', 
                                            outputCol='token_nostops')
            
        hashed = HashingTF(inputCol='token_nostops', 
                               outputCol='hashed_token', 
                               numFeatures=2**16)
            
        tfidf = IDF(minDocFreq=5, 
                        inputCol='hashed_token', 
                        outputCol='tfidf_token')
        
        return tokenizer, token_filter, hashed, tfidf

    def build_pipeline(self):
        """ Build pipeline for numeric and/or text only model
        """
        tokenizer, token_filter, hashed, tfidf = self.prepare_text()

        if self.use_numeric and not self.NUMERIC_COLS:
            raise ValueError("use_numeric=True but NUMERIC_COLS is empty.")
        
        elif self.use_numeric and self.NUMERIC_COLS:
            numeric_vec, scaled_num = self.numeric_assembler()
            final_assembler = VectorAssembler(inputCols = ['tfidf_token',
                                                     'scaled_num_vector'],
                                              outputCol = 'features'
                                       )
            self.pipeline = Pipeline(stages=[tokenizer, token_filter, hashed,
                                             tfidf, numeric_vec, scaled_num,
                                             final_assembler, 
                                             self.estimator])
            return self.pipeline

        final_assembler = VectorAssembler(inputCols=['tfidf_token'],
                                          outputCol='features')
        self.pipeline = Pipeline(stages=[tokenizer, token_filter, hashed, 
                                         tfidf, final_assembler,
                                         self.estimator])
        return self.pipeline
        

def add_weights(df, LABEL_COL, weight_col='class_weight'):
    """adds weight features to the data
    """
    label_count = df.groupBy(LABEL_COL).count().collect()
    label_map = {row[LABEL_COL]: row['count'] for row in label_count}
    total_count = sum(label_map.values())
    n_classes = len(label_map)

    weights = {label : total_count / count * n_classes for label, count in label_map.items()}

    expr = None
    for k, weight in weights.items():
        cond = (F.col(LABEL_COL) == F.lit(k))
        expr = F.when(cond, F.lit(float(weight))) if expr is None else expr.when(cond, F.lit(float(weight)))

    return df.withColumn(weight_col, expr.otherwise(F.lit(1.0))), weight_col

