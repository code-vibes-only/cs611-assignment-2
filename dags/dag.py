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

    # --- model auto training: only does work in training months (TRAIN_DATES in model_train.py) ---
    model_automl_start = EmptyOperator(task_id="model_automl_start")

    model_1_automl = BashOperator(task_id="model_1_automl", bash_command=run("model_train.py --modeltype logreg"))
    model_2_automl = BashOperator(task_id="model_2_automl", bash_command=run("model_train.py --modeltype xgb"))
    model_select = BashOperator(task_id="model_select", bash_command=run("model_select.py"))

    model_automl_completed = EmptyOperator(task_id="model_automl_completed")

    feature_store_completed >> model_automl_start
    label_store_completed >> model_automl_start
    model_automl_start >> [model_1_automl, model_2_automl] >> model_select >> model_automl_completed

    # --- model inference: the selected model scores this month's applications ---
    model_inference_start = EmptyOperator(task_id="model_inference_start")

    model_inference = BashOperator(task_id="model_inference", bash_command=run("model_inference.py"))

    model_inference_completed = EmptyOperator(task_id="model_inference_completed")

    model_automl_completed >> model_inference_start  # in a training month, score with the model trained this month
    model_inference_start >> model_inference >> model_inference_completed
