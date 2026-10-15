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

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 3, 1),
    catchup=True,
    max_active_runs=1,  # one month at a time, in order
) as dag:

    start = EmptyOperator(task_id="start")

    # prints the month this run is for, and lists the data the scripts will read
    check_month = BashOperator(
        task_id="check_month",
        bash_command='cd /app && echo "run for month {{ ds }}" && ls data',
    )

    start >> check_month
