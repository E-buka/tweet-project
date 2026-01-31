from tweet_analysis import start_spark, df_schema 
from tweet_analysis import TEXT_COL, LABEL_COL, NUMERIC_COLS
from tweet_analysis import text_cleaner, date_cleaner, target_cleaner

def run():
    """clean data frame, and split, then save
    """
    spark = start_spark()
    
    df = (spark.read
          .format('csv')
          .option('encoding', 'ISO-8859-1')
          .option('header', 'true')
          .schema(df_schema)
          .load('/home/ebuka/tweets.csv')
        )
    
    df = target_cleaner(date_cleaner(text_cleaner(df)))
    
    clean_df = df.select(NUMERIC_COLS + [TEXT_COL, LABEL_COL])
    
    train, test = clean_df.randomSplit(weights=[0.8, 0.2], seed=44)
    train.coalesce(1).write.mode('overwrite').parquet('data/splits/train.parquet')
    test.coalesce(1).write.mode('overwrite').parquet('data/splits/test.parquet')
    

if __name__ == "__main__":
    run()