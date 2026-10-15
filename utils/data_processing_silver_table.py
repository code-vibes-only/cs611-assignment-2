import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def read_bronze(table_name, snapshot_date_str, bronze_directory, spark):
    """Read the bronze CSV partition for a source for a given snapshot date, all columns read as strings"""
    partition_name = "bronze_" + table_name + "_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_directory + partition_name
    df = spark.read.csv(filepath, header=True, escape='"')
    print("loaded from:", filepath, "row count:", df.count())
    return df


def write_silver(df, table_name, snapshot_date_str, silver_directory):
    """Write the cleaned silver parquet partition for a source for a given snapshot date"""
    partition_name = "silver_" + table_name + "_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)


def cast_columns(df, column_type_map):
    """Cast column to target type. null if unable to parse."""
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    return df


def process_silver_attributes_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Clean attributes table"""
    
    df = read_bronze("attributes", snapshot_date_str, bronze_directory, spark)

    # Age: remove the stray trailing underscore ("23_"), then enforce the type
    df = df.withColumn("Age", F.regexp_replace(col("Age"), "_$", ""))

    # clean data: enforce schema / data type
    df = cast_columns(df, {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    })

    # -500 and ages above 100 are not valid ages; keep only feasible values
    df = df.withColumn("Age", F.when(col("Age").between(0, 100), col("Age")))

    # placeholders become missing values
    df = df.withColumn("Occupation", F.when(col("Occupation") != "_______", col("Occupation")))
    df = df.withColumn("SSN", F.when(col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), col("SSN")))

    write_silver(df, "attributes", snapshot_date_str, silver_directory)
    return df


def process_silver_financials_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Clean financials table"""
    df = read_bronze("financials", snapshot_date_str, bronze_directory, spark)

    # numbers stored as text: treat whole-value placeholders (__10000__, __-333...__) as missing
    # remove trailing underscore ("52312.68_" -> "52312.68", "_" -> "") and treat as valid value
    text_numbers = ["Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
                    "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance"]
    for c in text_numbers:
        df = df.withColumn(c, F.when(~col(c).rlike("^__.*__$"), F.regexp_replace(col(c), "_$", "")))

    # clean data: enforce schema / data type
    df = cast_columns(df, {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    })
    
    # cast as integer
    df = df.withColumn("Num_Credit_Inquiries", col("Num_Credit_Inquiries").cast(IntegerType()))

    # treat values outside the dense body of each count column as missing (ranges from notebooks/02_data_quality.ipynb)
    valid_ranges = {
        "Num_Bank_Accounts": (0, 11),
        "Num_Credit_Card": (0, 11),
        "Interest_Rate": (0, 34),
        "Num_of_Loan": (0, 9),
        "Num_of_Delayed_Payment": (0, 28),
        "Num_Credit_Inquiries": (0, 17),
    }
    for c, (low, high) in valid_ranges.items():
        df = df.withColumn(c, F.when(col(c).between(low, high), col(c)))

    # cross-checks against Monthly_Inhand_Salary, which has no defects
    df = df.withColumn("Annual_Income",F.when(col("Annual_Income") <= 2 * 12 * col("Monthly_Inhand_Salary"), col("Annual_Income")))
    df = df.withColumn("Total_EMI_per_month",F.when(col("Total_EMI_per_month") <= col("Monthly_Inhand_Salary"), col("Total_EMI_per_month")))

    # categorical placeholders become missing; NM in Payment_of_Min_Amount is kept as a category
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix") != "_", col("Credit_Mix")))
    df = df.withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour") != "!@9#%8", col("Payment_Behaviour")))

    # augment data: credit history in months ("10 Years and 9 Months" -> 129)
    years = F.regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast(IntegerType())
    months = F.regexp_extract(col("Credit_History_Age"), r"(\d+) Months", 1).cast(IntegerType())
    df = df.withColumn("Credit_History_Months", years * 12 + months)

    write_silver(df, "financials", snapshot_date_str, silver_directory)
    return df


def process_silver_clickstream_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    df = read_bronze("clickstream", snapshot_date_str, bronze_directory, spark)

    # enforce schema / data type
    column_type_map = {f"fe_{i}": IntegerType() for i in range(1, 21)}
    column_type_map["Customer_ID"] = StringType()
    column_type_map["snapshot_date"] = DateType()
    df = cast_columns(df, column_type_map)

    write_silver(df, "clickstream", snapshot_date_str, silver_directory)
    return df


def process_silver_loan_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    df = read_bronze("loan_daily", snapshot_date_str, bronze_directory, spark)

    # enforce schema / data type
    df = cast_columns(df, {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    })

    # augment data: add month on book (Lab 2)
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due (Lab 2)
    # fill the column where 0/0 gives null
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0, subset=["installments_missed"])
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0,
                    F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0,
                    F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    write_silver(df, "loan_daily", snapshot_date_str, silver_directory)
    return df


