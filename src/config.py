from pyspark.sql import SparkSession

def start_spark():
    """ starting spark session
    """
    #from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("SentimentAnalysis")
        .config("spark.driver.memory", "5g")
        .config("spark.executor.memory", "3g")
        .config("spark.executor.cores", "2")
        .config("spark.sql.shuffle.partitions", "24")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
        )
    return spark

