from pyspark.sql.types import *

df_schema = StructType([
    StructField('Target', IntegerType(), True),
    StructField('ID', LongType(), True),
    StructField('Date', StringType(), True),
    StructField('flag', StringType(), True),
    StructField('User', StringType(), True),
    StructField('Text', StringType(), True)
    ])    

# final expected schema
TEXT_COL = 'Text'
LABEL_COL  = 'Target'
DATE_COL = 'Date'

NUMERIC_COLS = ['hour',
               'dayofweek',
               'char_count', 
               'total_words', 
               'num_exclaim'
              ]
