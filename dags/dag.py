from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def run(script_args):
    """Bash command that runs one project script for the run's month ({{ ds }})"""
    return f'cd /app && python3 {script_args} --snapshotdate "{{{{ ds }}}}"'


with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 7, 1),
    catchup=True,
    max_active_runs=1,  # months run in order: month T reads tables built for earlier months
) as dag:

    # --- label store ---
    dep_check_source_label_data = EmptyOperator(task_id="dep_check_source_label_data")

    bronze_label_store = BashOperator(task_id="bronze_label_store", bash_command=run("data_processing.py --layer bronze --table loan_daily"))
    silver_label_store = BashOperator(task_id="silver_label_store", bash_command=run("data_processing.py --layer silver --table loan_daily"))
    gold_label_store = BashOperator(task_id="gold_label_store", bash_command=run("data_processing.py --layer gold --table label_store"))

    label_store_completed = EmptyOperator(task_id="label_store_completed")

    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # --- feature store ---
    gold_feature_store = BashOperator(task_id="gold_feature_store", bash_command=run("data_processing.py --layer gold --table feature_store"))

    feature_store_completed = EmptyOperator(task_id="feature_store_completed")

    for table in ["attributes", "financials", "clickstream"]:
        dep_check = EmptyOperator(task_id=f"dep_check_source_{table}")
        bronze = BashOperator(task_id=f"bronze_{table}", bash_command=run(f"data_processing.py --layer bronze --table {table}"))
        silver = BashOperator(task_id=f"silver_{table}", bash_command=run(f"data_processing.py --layer silver --table {table}"))
        dep_check >> bronze >> silver >> gold_feature_store

    gold_feature_store >> feature_store_completed
