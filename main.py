from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, min, sum, corr

import pandas as pd
import matplotlib.pyplot as plt

import findspark

findspark.init()

spark = (SparkSession.builder
         .appName("The top most common words in The Master and Margarita, by Bulgakov Mikhail")
         .master("local[*]")
         .getOrCreate()
         )
# Load the dataset into a PySpark DataFrame
df = spark.read.csv("NCHS_-_Death_rates_and_life_expectancy_at_birth.csv", header=True)

# Group the data by Race
grouped_df = df.groupBy("Race")

# Compute the maximum and minimum values of Age-adjusted Death Rate for each group
result_df = grouped_df.agg(max("Age-adjusted Death Rate").alias("Max"),
                           min("Age-adjusted Death Rate").alias("Min"))


# Display the results
result_df.show()

grouped_df = df.groupBy("Year")

# Compute the average life expectancy for each group
result_df = grouped_df.agg(avg("Average Life Expectancy (Years)").alias("Avg Life Expectancy"))

# Display the results
result_df.show()

pandas_df = result_df.toPandas()

# Create line chart
plt.plot(pandas_df["Year"], pandas_df["Avg Life Expectancy"])
plt.xlabel("Year")
plt.ylabel("Avg Life Expectancy")
plt.title("Life Rates by Year")
plt.show()

grouped_df = df.groupBy("Year")

# Compute the sum of Age-adjusted Death Rate for each group
result_df = grouped_df.agg(sum("Age-adjusted Death Rate").alias("Total AADR"))

# Sort the results in descending order and select the top 5 years
result_df = result_df.orderBy(result_df["Total AADR"].desc()).limit(5)

# Display the results
result_df.show()

pandas_df = result_df.toPandas()

# Create line chart
plt.plot(pandas_df["Year"], pandas_df["Total AADR"])
plt.xlabel("Year")
plt.ylabel("Total AADR")
plt.title("Sum of Age-adjusted Death Rate for each group")
plt.show()

# Group the data by Race and Sex
grouped_df = df.groupBy("Race", "Sex")

# Compute the average Age-adjusted Death Rate for each group
result_df = grouped_df.agg(avg("Age-adjusted Death Rate").alias("Avg AADR"))

# Display the results
result_df.show()


# Group the data by Year
grouped_df = df.groupBy("Year")

# Compute the correlation between Average Life Expectancy and Age-adjusted Death Rate for each group
result_df = grouped_df.agg(corr("Average Life Expectancy (Years)", "Age-adjusted Death Rate").alias("Correlation"))

pandas_df = result_df.toPandas()

# Create line chart
plt.plot(pandas_df["Year"], pandas_df["Correlation"])
plt.xlabel("Year")
plt.ylabel("Correlation")
plt.title("Correlation between Average Life Expectancy and Age-adjusted Death Rate for each group")
plt.show()

# Display the results
result_df.show()
