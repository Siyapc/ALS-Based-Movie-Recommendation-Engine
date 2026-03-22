# Databricks notebook source
# MAGIC %md
# MAGIC # ALS-Based Movie Recommendation Engine
# MAGIC
# MAGIC ### - Project Objective
# MAGIC
# MAGIC The goal of this project is to build a personalized movie recommendation system using collaborative filtering techniques on Databricks Free Edition.
# MAGIC
# MAGIC The system predicts:
# MAGIC
# MAGIC “Which movies is a user most likely to rate highly?”
# MAGIC
# MAGIC This is achieved using the ALS (Alternating Least Squares) algorithm from Apache Spark MLlib.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full Pipeline Architecture
# MAGIC
# MAGIC Bronze Layer → Raw CSV data
# MAGIC
# MAGIC Silver Layer → Cleaned & structured data
# MAGIC
# MAGIC ML Layer → ALS model training
# MAGIC
# MAGIC Evaluation Layer → RMSE performance validation
# MAGIC
# MAGIC Recommendation Layer → Manual generation of top-N movies
# MAGIC
# MAGIC Presentation Layer → Join with movie titles
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #🥉 Bronze Layer (Data Ingestion)
# MAGIC Load MovieLens datasets:
# MAGIC Ratings (userId, movieId, rating)
# MAGIC Movies (movieId, title, genres)
# MAGIC Store raw data as-is
# MAGIC No cleaning or transformation
# MAGIC Schema inferred automatically
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Load data (RAW)
# Load ratings data
ratings_df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("/Volumes/workspace/default/movie_volume/ratings.csv")

# Load movies data
movies_df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("/Volumes/workspace/default/movie_volume/movies.csv")

ratings_df.show(5)
movies_df.show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC #🥈 Silver Layer (Data Cleaning)
# MAGIC Select required columns: userId, movieId, rating
# MAGIC Convert data types:
# MAGIC userId, movieId → int
# MAGIC rating → float
# MAGIC Remove null values
# MAGIC
# MAGIC ✔ Output: Clean, structured data ready for ML

# COMMAND ----------

# DBTITLE 1,Data Cleaning (SILVER)
from pyspark.sql.functions import col

ratings_clean = ratings_df \
    .select(
        col("userId").cast("int"),
        col("movieId").cast("int"),
        col("rating").cast("float")
    ) \
    .dropna()

movies_clean = movies_df \
    .select(
        col("movieId").cast("int"),
        col("title"),
        col("genres")
    ) \
    .dropna()

ratings_clean.printSchema()
movies_clean.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #🤖 Model Building (ALS)
# MAGIC Use ALS (collaborative filtering)
# MAGIC
# MAGIC Learns:
# MAGIC User preferences
# MAGIC Movie features
# MAGIC
# MAGIC Key configs:
# MAGIC userCol, itemCol, ratingCol
# MAGIC coldStartStrategy = "drop"
# MAGIC nonnegative = True
# MAGIC
# MAGIC 🧪 Train-Test Split
# MAGIC 80% training
# MAGIC 20% testing
# MAGIC
# MAGIC ✔ Ensures proper evaluation on unseen data

# COMMAND ----------

# DBTITLE 1,Build Recommendation Model (ALS)
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Split data
train, test = ratings_clean.randomSplit([0.8, 0.2])

# Define ALS model
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)

# Train
model = als.fit(train)

# Predict
predictions = model.transform(test)


# COMMAND ----------

# MAGIC %md
# MAGIC #📊 Model Evaluation
# MAGIC Metric: RMSE
# MAGIC
# MAGIC Lower RMSE = better accuracy

# COMMAND ----------

# DBTITLE 1,Evaluate Model
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print("RMSE:", rmse)


# COMMAND ----------

# MAGIC %md
# MAGIC #🎯 Recommendation Generation (Manual)
# MAGIC (Used due to Free Edition limitations)
# MAGIC
# MAGIC - Get all users
# MAGIC - Get all movies
# MAGIC - Create all user–movie pairs
# MAGIC - Predict ratings using ALS
# MAGIC - Rank movies per user
# MAGIC - Select Top 5 recommendations per user

# COMMAND ----------

# DBTITLE 1,Recommend movies for all users:
users = ratings_clean.select("userId").distinct()
movies_df = ratings_clean.select("movieId").distinct()

from pyspark.sql.functions import col

user_movie = users.crossJoin(movies_df)



# COMMAND ----------

# DBTITLE 1,Predict Ratings
predictions = model.transform(user_movie)


# COMMAND ----------

# DBTITLE 1,Get Top 5 Per User
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

window = Window.partitionBy("userId").orderBy(col("prediction").desc())

top_recommendations = predictions.withColumn(
    "rank",
    row_number().over(window)
).filter(col("rank") <= 5)

top_recommendations.show()


# COMMAND ----------

final = top_recommendations.join(movies_clean, "movieId")

final.select("userId", "title", "prediction").show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC #🚀 Final Result
# MAGIC
# MAGIC A complete pipeline that:
# MAGIC
# MAGIC - Ingests raw data
# MAGIC - Cleans & prepares it
# MAGIC - Trains ML model
# MAGIC - Evaluates accuracy
# MAGIC - Generates personalized recommendations