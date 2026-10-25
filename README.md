# CS611 Assignment 2: Loan default ML pipeline

## Context
A bank lends cash loans and wants a model that predicts, at the point of loan application, whether a customer will default. This repository builds the production ML pipeline for that model, orchestrated by Airflow in Docker: it trains and selects a model into a model bank, uses it to score each month's applications into a gold table, and monitors the model's performance and stability over time into another gold table.

The datamart code in `utils/` starts from my Assignment 1 pipeline: https://github.com/code-vibes-only/cs611-assignment-1

## What the DAG does each month
One DAG (`dags/dag.py`) runs once per month from 2023-01 to 2025-07 and is backfilled with `catchup=True`, one month at a time. Each run, for its month T (`{{ ds }}`):

| Stage | Tasks | Script |
|---|---|---|
| Label store | `bronze_label_store` → `silver_label_store` → `gold_label_store` | `data_processing.py` |
| Feature store | `bronze_*` → `silver_*` for attributes, financials, clickstream → `gold_feature_store` | `data_processing.py` |
| Training (only in training months) | `model_1_automl` (logistic regression), `model_2_automl` (XGBoost) → `model_select` | `model_train.py`, `model_select.py` |
| Inference | `model_inference`: the selected model scores month T's applications | `model_inference.py` |
| Monitoring | `model_monitor`: stability of month T, performance of month T − 6 | `model_monitor.py` |
| Visualisation | `model_monitor_charts`: redraws the monitoring charts | `model_monitor_charts.py` |

## Repository layout
- `dags/dag.py`: the Airflow DAG
- `data_processing.py`: runs one month of one datamart table, using the bronze/silver/gold functions in `utils/`
- `model_train.py`, `model_select.py`, `model_inference.py`, `model_monitor.py`, `model_monitor_charts.py`: the ML pipeline steps, one script per task
- `notebooks/01_datamart_check.ipynb`: the DAG's datamart against Assignment 1's
- `notebooks/02_model_training.ipynb`: label availability, feature stability, candidate models, leakage checks
- `notebooks/03_feature_borrowing.ipynb`: features from the sample datamarts tested as extra columns, under a rule set in advance
- `notebooks/04_feature_pruning.ipynb`: the features neither candidate uses tested for removal, under a rule set in advance
- `notebooks/05_monitoring_check.ipynb`: predictions and monitoring checked, one month recomputed by hand
- `Dockerfile`, `docker-compose.yaml`, `requirements.txt`: the environment (Airflow 2.11, Python 3.12, Java for Spark)

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
4. The DAG `dag` starts its backfill by itself: 31 monthly runs, about an hour in total. The model is trained in the 2024-03 run (about half an hour in); predictions follow from then on, and performance monitoring from the 2024-09 run
5. JupyterLab, for the notebooks, is at http://localhost:8888

## Outputs
| Output | Path |
|---|---|
| Feature store (one row per application) | `datamart/gold/feature_store/` |
| Label store (`30dpd_6mob`) | `datamart/gold/label_store/` |
| Model bank: candidate artefacts and the selection record | `model_bank/` |
| Predictions (gold) | `datamart/gold/model_predictions/<model>/` |
| Monitoring results (gold) | `datamart/gold/model_monitoring/` |
| Monitoring charts | `datamart/monitoring_charts/` |

## Design rules
- **Prediction point:** the application month. Features use information up to that month only; the clickstream features average the 3 months up to and including it.
- **Labels mature 6 months after application.** A model trained at month T uses only labels dated before T, so applications up to T − 7. The model is trained at 2024-03-01 on applications 2023-01 to 2023-06 (80/20 train/test) and 2023-07 to 2023-08 (out-of-time, OOT). Imputation and scaling are fitted on the training split only.
- **Features:** the model trains on 59 of the feature store's 66 features (`FEATURE_COLUMNS` in `model_train.py`). Notebook 04 dropped the 7 that neither candidate used on the training split; the selected model's OOT AUC stayed within 0.005 at three train dates. The feature store keeps all 66.
- **Selection:** the candidate with the higher OOT AUC goes into the selection record that inference reads.
- **Monitoring:** stability is measured in the month scored (PSI of the scores and of each feature against the training data); performance (AUC, Gini) six months later, when the labels arrive. Status bands: PSI < 0.1 green, 0.1–0.25 amber, > 0.25 red; AUC within 0.05 of the OOT AUC at training green, within 0.10 amber, otherwise red.

## Monitoring charts
In every run, `model_monitor_charts` redraws three charts in `datamart/monitoring_charts/` from every monitoring partition up to that month:
- `performance.png`: the AUC of each scored month against the OOT AUC at training, coloured by status; below it, the actual bad rate against the predicted one (the mean score).
- `score_stability.png`: the PSI of each month's scores against the training scores, with the 0.1 and 0.25 bands.
- `feature_stability.png`: the CSI status of the 20 most drifted features, month by month.

What they show over the backfill (applications scored from 2024-03 to 2025-01):
- Until 2024-06 everything is green: score PSI 0.015 to 0.046, AUC 0.814 to 0.844.
- The clickstream feed is empty for applications from 2024-07. In that month's run the score PSI jumps to 0.417 and all 16 clickstream features the model uses turn red; the other 43 features stay green.
- Performance confirms it six months later: the 2024-07 applications score an AUC of 0.801 (amber) and 2024-08's 0.751 (red); 6 of the 7 months from 2024-07 are amber or red.
- The model also under-predicts from 2024-07: its mean score is 2 to 9 points below the actual bad rate, against at most 4 points before.

## Model governance and refresh SOP
Roles: the **model owner** (data scientist) reads the charts each month and runs this SOP; the **data engineer** owns the source feeds and the pipeline; the **model approver** (credit risk) signs off any change of model.

A red status is first traced to its cause. A data or pipeline fault is fixed and the affected months rerun; a model is retrained only when the fault cannot be fixed or the applicants have really changed.

| Trigger (in the monitoring table) | Action | Owner | When |
|---|---|---|---|
| All green | None; review the charts | Model owner | Monthly |
| Score PSI or a feature's CSI amber | Find the cause: a feed or pipeline change, or a real change in applicants. Record it | Model owner | Within the month |
| A feature's CSI red | Check the feed with the data engineer. A pipeline fault: fix it and rerun the affected months by clearing their tasks in Airflow. A feed that cannot be restored: train a challenger without the feature | Model owner, data engineer | Within a week |
| Score PSI red | As for a red feature, and tell the approver: a fixed score cut-off now approves a different share of applicants | Model owner, model approver | Within a week |
| AUC red, or AUC amber 3 months in a row | Train a challenger on the latest matured labels | Model owner | The next month |
| Mean score and actual bad rate more than 5 points apart, 2 months in a row | Tell the approver: losses expected from the score are off. Recalibrate, or train a challenger | Model owner, model approver | The next month |
| Every 6 months, even when green | Train a challenger, to test for a better model | Model owner | Scheduled |

How a challenger replaces the model:
1. **Train.** On a branch, add the refresh month to `TRAIN_DATES` in `model_train.py`, and change `FEATURE_COLUMNS` if a feature is dropped (keeping the earlier list for earlier train dates, so they rebuild unchanged). Running that month's `model_` tasks trains both candidates and selects the better on OOT AUC.
2. **Compare.** The challenger must beat the current model on the same OOT months, scored on the data the current model now receives (for a dead feed, with that feature blanked). Otherwise the comparison favours the model that relies on the dead feed.
3. **Approve and deploy.** The approver signs off the comparison; merging the branch deploys it, since from the refresh month inference reads the newest selection record. A cautious switch scores both models for a month or two first (shadow), and the monitoring table holds rows for each `model_name`.
4. **Roll back.** Earlier artefacts and selection records stay in `model_bank/`. Removing the new selection record and clearing `model_inference` for the months since the refresh puts the previous model back.

Applied to this backfill:
- **2024-07 run:** score PSI and 16 feature CSIs red. The cause is the clickstream feed: every value is missing from 2024-07, so the model scores these applicants with the training medians. The feed cannot be restored from the data supplied, so the SOP calls for a challenger without clickstream.
- **A plain refresh would not fix it.** A model trained at month T learns from applications T − 14 to T − 9 (plus T − 8 and T − 7 as OOT), so through the 2025-08 run its training split still has months with clickstream: a refreshed model would learn to rely on a feed that no longer arrives.
- **2025-01 and 2025-02 runs:** performance confirms the alert: 2024-07 amber, 2024-08 red, and in both months the mean score is nearly 7 points below the actual bad rate. This triggers the challenger and the approver's review.
- **What to expect from the challenger:** in notebook 02, looking back with every label, XGBoost without clickstream scored an AUC of 0.804 on the 2024-07 to 2025-01 applications against 0.797 with it, despite a lower OOT AUC while the feed worked (0.806 against 0.854; fixed settings, all 66 features). The challenger itself is not trained in this pipeline.

## Deployment options
- **Batch, monthly (this pipeline):** Airflow scores each month's applications after the month closes and writes them to the gold table. It suits backfills, monitoring and portfolio reporting; with labels 6 months behind, monthly monitoring is frequent enough.
- **Real-time, for the lending decision:** a customer is approved or declined when they apply, so in production the same artefact would sit behind an endpoint that scores one application at a time. The features must be computed the same way at that moment (the 3-month clickstream average needs a feature store that is updated as events arrive). The batch pipeline would stay, to log the predictions and run the monitoring.
- **Rolling out a new model:** shadow first (scored alongside, decisions unchanged), then canary (a small share of applications), then the full switch, keeping the previous model ready (blue/green) for a rollback.
