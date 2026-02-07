# for running locally
# from pyspark.sql import SparkSession

# def start_spark():
#     return  (
    #     SparkSession.builder
    #     .appName("SentimentAnalysis")
    #     .config("spark.driver.memory", "5g")
    #     .config("spark.executor.memory", "3g")
    #     .config("spark.executor.cores", "2")
    #     .config("spark.sql.shuffle.partitions", "24")
    #     .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    #     .getOrCreate()
    # )

# for deployment
import os
from pyspark.sql import SparkSession

def start_spark():
    return (
        SparkSession.builder
        .appName("SentimentAnalysis")
        .master("local[1]")  # 1 core reduces overhead
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "512m"))
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
