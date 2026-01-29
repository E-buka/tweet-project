
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
   
    def __init__(self, NUMERIC_COLS=None, TEXT_COL=None, LABEL_COL=None, persist_level=None):
        self.persist_level = persist_level
        self.NUMERIC_COLS = NUMERIC_COLS
        self.LABEL_COL = LABEL_COL
        self.TEXT_COL = TEXT_COL
        
        self.text_model = None
        self.df_idf = None

    def numeric_assembler(self):
        """assemble and scale numeric features
        """
        numeric_vec = VectorAssembler(inputCols=self.NUMERIC_COLS,
                                      outputCol='num_features', handleInvalid='keep')
        scaled_num = StandardScaler(inputCol='num_features', outputCol='scaled_num_vector', withMean=False)
        return numeric_vec, scaled_num

    def prepare_text(self, df, fit=False):
        """ tokenize text features and persist to memory to avoid recomputing
        """
        if fit:
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
            
            idf_pipe = Pipeline(stages= [tokenizer, token_filter, hashed, tfidf])
            self.text_model = idf_pipe.fit(df)
        
        if self.text_model is None:
            raise RuntimeError("Text model has not been fitted")
            
        transformed = self.text_model.transform(df)

        cols_keep = [self.LABEL_COL, 'tfidf_token'] + self.NUMERIC_COLS
        transformed = transformed.select(*cols_keep)

        if fit:
            self.df_idf = transformed.persist(self.persist_level)
            _ = self.df_idf.count()
            return self.df_idf
        
        return transformed


    def add_weights(self, df, weight_col='class_weight'):
        """adds weight features to the vectorised data
        """
        label_count = df.groupBy(self.LABEL_COL).count().collect()
        label_map = {row[self.LABEL_COL]: row['count'] for row in label_count}
        total_count = sum(label_map.values())
        n_classes = len(label_map)

        weights = {label : total_count / count * n_classes for label, count in label_map.items()}

        expr = None
        for k, weight in weights.items():
            cond = (F.col(self.LABEL_COL) == F.lit(k))
            expr = F.when(cond, F.lit(float(weight))) if expr is None else expr.when(cond, F.lit(float(weight)))

        df = df.withColumn(weight_col, expr.otherwise(F.lit(1.0)))
        
        return df, weight_col

    def unpersist(self):
        if self.df_idf is not None:
            self.df_idf.unpersist()
            self.df_idf = None
        return self
