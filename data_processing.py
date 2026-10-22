import argparse
import os
import pyspark

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# to call this script: python data_processing.py --snapshotdate "2023-01-01" --layer bronze --table attributes

# each source's raw file and the months it has snapshots for (from A1 notebooks/01_source_inventory.ipynb)
SOURCES = {
    "attributes": ("data/features_attributes.csv", "2023-01-01", "2025-01-01"),
    "financials": ("data/features_financials.csv", "2023-01-01", "2025-01-01"),
    "clickstream": ("data/feature_clickstream.csv", "2023-01-01", "2024-12-01"),
    "loan_daily": ("data/lms_loan_daily.csv", "2023-01-01", "2025-11-01"),
}

SILVER_FUNCTIONS = {
    "attributes": utils.data_processing_silver_table.process_silver_attributes_table,
    "financials": utils.data_processing_silver_table.process_silver_financials_table,
    "clickstream": utils.data_processing_silver_table.process_silver_clickstream_table,
    "loan_daily": utils.data_processing_silver_table.process_silver_loan_table,
}

# the source whose months each gold table follows
GOLD_SOURCE = {"label_store": "loan_daily", "feature_store": "attributes"}


def main(snapshotdate, layer, table):
    # skip months the source has no data for (e.g. no applications after 2025-01)
    _, first_date, last_date = SOURCES[GOLD_SOURCE.get(table, table)]
    if not first_date <= snapshotdate <= last_date:
        print(f"no {table} data for {snapshotdate}, skipping")
        return

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    if layer == "bronze":
        bronze_directory = f"datamart/bronze/{table}/"
        os.makedirs(bronze_directory, exist_ok=True)
        utils.data_processing_bronze_table.process_bronze_table(snapshotdate, SOURCES[table][0], table, bronze_directory, spark)

    elif layer == "silver":
        silver_directory = f"datamart/silver/{table}/"
        os.makedirs(silver_directory, exist_ok=True)
        SILVER_FUNCTIONS[table](snapshotdate, f"datamart/bronze/{table}/", silver_directory, spark)

    elif layer == "gold" and table == "label_store":
        os.makedirs("datamart/gold/label_store/", exist_ok=True)
        utils.data_processing_gold_table.process_labels_gold_table(
            snapshotdate, "datamart/silver/loan_daily/", "datamart/gold/label_store/", spark, dpd=30, mob=6)

    elif layer == "gold" and table == "feature_store":
        os.makedirs("datamart/gold/feature_store/", exist_ok=True)
        utils.data_processing_gold_table.process_features_gold_table(
            snapshotdate, "datamart/silver/attributes/", "datamart/silver/financials/", "datamart/silver/clickstream/",
            "datamart/gold/feature_store/", spark)

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run one month of one datamart table")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--layer", type=str, required=True, choices=["bronze", "silver", "gold"])
    parser.add_argument("--table", type=str, required=True, help="source name, or label_store / feature_store for gold")
    args = parser.parse_args()

    main(args.snapshotdate, args.layer, args.table)
