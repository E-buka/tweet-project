# import os
# from pyspark.sql import SparkSession

# def start_spark():
#     return (
#         SparkSession.builder
#         .appName("SentimentAnalysis")
#         .master("local[*]")
#         # Make Spark bind only inside the container
#         .config("spark.driver.bindAddress", "127.0.0.1")
#         .config("spark.driver.host", "127.0.0.1")
#         # Disable Spark UI (prevents port confusion + reduces noise)
#         .config("spark.ui.enabled", "false")
#         # Reduce network surprises
#         .config("spark.network.timeout", "300s")
#         .config("spark.executor.heartbeatInterval", "60s")
#         .getOrCreate()
#     )

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
