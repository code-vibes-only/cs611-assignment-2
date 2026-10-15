from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print("loaded from:", filepath, "row count:", df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd) + "dpd_" + str(mob) + "mob").cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "loan_start_date", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    return df


# fixed category lists from the EDA (data quality §4); a missing category is 0 in every column
CATEGORIES = {
    "Occupation": ["Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur", "Journalist",
                   "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer"],
    "Credit_Mix": ["Bad", "Standard", "Good"],
    "Payment_of_Min_Amount": ["Yes", "No", "NM"],
    "Payment_Behaviour": ["Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
                          "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
                          "High_spent_Medium_value_payments", "High_spent_Large_value_payments"],
}

NUMERIC_FEATURES = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
                    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
                    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
                    "Credit_History_Months", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]


def process_features_gold_table(snapshot_date_str, silver_attributes_directory, silver_financials_directory,
                                silver_clickstream_directory, gold_feature_store_directory, spark):
    # define arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d") # the application month being processed
    partition_date = snapshot_date_str.replace("-", "_")

    # applicants this month: attributes and financials are taken in the application month
    attributes = spark.read.parquet(silver_attributes_directory + "silver_attributes_" + partition_date + ".parquet")
    financials = spark.read.parquet(silver_financials_directory + "silver_financials_" + partition_date + ".parquet")
    df = attributes.select("Customer_ID", "snapshot_date", "Age", "Occupation").join(financials.drop("snapshot_date"), "Customer_ID", "inner")
    print(snapshot_date_str, "applicants:", attributes.count(), "with financials:", df.count())

    # engineered ratios using Monthly_Inhand_Salary as the measure of income
    df = df.withColumn("Debt_to_Annual_Salary", col("Outstanding_Debt") / (12 * col("Monthly_Inhand_Salary")))
    df = df.withColumn("EMI_to_Monthly_Salary", col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))

    # one-hot encoding for categories
    one_hot_columns = []
    for c, values in CATEGORIES.items():
        for v in values:
            df = df.withColumn(c + "_" + v, F.when(col(c) == v, 1).otherwise(0))
            one_hot_columns.append(c + "_" + v)

    # clickstream: average of each feature over the months up to and including the application month
    clickstream = spark.read.parquet(silver_clickstream_directory + "*").filter(col("snapshot_date") <= snapshot_date) # later months are future information at time of application so they are removed
    fe_columns = [f"fe_{i}" for i in range(1, 21)]
    clickstream_history = clickstream.groupBy("Customer_ID").agg(*[F.avg(c).alias(c + "_avg") for c in fe_columns])
    df = df.join(clickstream_history, "Customer_ID", "left")

    # select columns to save: keys, then features (no identifiers, no raw text)
    df = df.select(["Customer_ID", "snapshot_date"] + NUMERIC_FEATURES + ["Debt_to_Annual_Salary", "EMI_to_Monthly_Salary"]+ one_hot_columns + [c + "_avg" for c in fe_columns])

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + partition_date + ".parquet"
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    return df



