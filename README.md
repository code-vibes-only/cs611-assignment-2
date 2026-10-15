# CS611 Assignment 2: Loan default ML pipeline

## Context
A bank lends cash loans and wants a model that predicts, at the point of loan application, whether a customer will default. This repository builds the production ML pipeline for that model, orchestrated by Airflow in Docker: it trains and selects a model into a model bank, uses it to score each month's applications into a gold table, and monitors the model's performance and stability over time into another gold table.

The datamart code in `utils/` starts from my Assignment 1 pipeline: https://github.com/code-vibes-only/cs611-assignment-1

## Data
The four raw CSV files are not stored in Git. Copy them into `data/`:

    data/
    ├── feature_clickstream.csv
    ├── features_attributes.csv
    ├── features_financials.csv
    └── lms_loan_daily.csv

## How to run
1. Put the four CSV files in `data/` (see above)
2. Build and start the environment:
   ```sh
   docker-compose build
   docker-compose up
   ```
3. Open Airflow at http://localhost:8080 and log in with `admin` / `admin`
4. Unpause the DAG `dag`. It catches up one run per month from its start date
5. JupyterLab, for the notebooks, is at http://localhost:8888
