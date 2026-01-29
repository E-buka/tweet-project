# preprocessing file
from .schema import TEXT_COL, LABEL_COL, DATE_COL
from pyspark.sql import functions as F 
from pyspark.sql import DataFrame


def text_cleaner(df: DataFrame) -> DataFrame:
    """ Clean the text data and generate numeric features based on text data
    """
    # get the number of exclaims before cleaning the text
    df = df.withColumn('num_exclaim', F.length(F.col(TEXT_COL)) - F.length(F.regexp_replace(F.col(TEXT_COL), '!', '')))
    ##clean the text
    df = (df.withColumn(TEXT_COL, F.coalesce(F.lower(F.col(TEXT_COL)), F.lit("")))
        .withColumn(TEXT_COL, F.coalesce(F.regexp_replace(F.col(TEXT_COL), r'http\S+|www\.\S+', ' '), F.lit("")))
        .withColumn(TEXT_COL, F.coalesce(F.regexp_replace(F.col(TEXT_COL), r'@\w+', ' '), F.lit("")))
        .withColumn(TEXT_COL, F.coalesce(F.regexp_replace(F.col(TEXT_COL), r'[^a-z0-9\s]', ' '), F.lit("")))
        .withColumn(TEXT_COL, F.coalesce(F.regexp_replace(F.col(TEXT_COL), r'\s+', ' '), F.lit("")))
        .withColumn(TEXT_COL, F.trim(TEXT_COL))
         ) 
    #generate numeric features from the text
    df = (df.withColumn('char_count', F.length(F.col(TEXT_COL)))
          .withColumn('total_words', F.size(F.split(F.col(TEXT_COL), ' ')))
         )

    return df
    

def date_cleaner(df: DataFrame) -> DataFrame:
    """clean date column and generate numeric features from the date
    Date is expected to contain timezone and format MMM dd HH:mm:ss yyyy
    """
    df = (df.withColumn(DATE_COL, F.regexp_replace(F.col(DATE_COL), r'(?<=:\d{2}) (?:[A-Za-z])* (?=\d){2,4}', ' '))
        .withColumn(DATE_COL, F.regexp_replace(F.col(DATE_COL), r'\s+', ' '))
         )

    df = (df.withColumn(DATE_COL, F.to_timestamp(F.col(DATE_COL).substr(5, 24), 'MMM dd HH:mm:ss yyyy'))
        .withColumn('hour', F.hour(F.col(DATE_COL)))
        .withColumn('dayofweek', F.dayofweek(F.col(DATE_COL)))
         )

    return df

def target_cleaner(df: DataFrame) -> DataFrame:
    """maps the target 0 to 0 and anyother value to 1"""
    df = df.withColumn(LABEL_COL, F.when(F.col(LABEL_COL) == 0, 0).otherwise(1))
    return df

