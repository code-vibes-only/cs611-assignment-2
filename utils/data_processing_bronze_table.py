from pyspark.sql.functions import col

def process_bronze_table(snapshot_date_str, source_csv, table_name, bronze_directory, spark):
    # load data - IRL ingest from back end source system
    # no inferSchema=True: keep raw values and leave type definition to silver layer
    # escape='"' formats quoted fields ("ODonnell""d" -> ODonnell"d) 
    df = spark.read.csv(source_csv, header=True, escape='"').filter(col("snapshot_date") == snapshot_date_str)
    print(snapshot_date_str, table_name, "row count:", df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_" + table_name + "_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return df
