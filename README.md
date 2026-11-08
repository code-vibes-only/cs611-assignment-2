# CS611 Assignment 2: Loan default ML pipeline

## Context
A bank lends cash loans and wants a model that predicts, at the point of loan application, whether a customer will default. This repository builds the production ML pipeline for that model, orchestrated by Airflow in Docker: it trains and selects a model into a model bank, uses it to score each month's applications into a gold table, and monitors the model's performance and stability over time into another gold table.

The score supports the approve-or-decline decision. On the loan ledger every default costs the bank 60–100% of the $10,000 principal ($7,531 on average) and every other loan is repaid in full, so a missed default costs about $7,500 and a good customer turned away costs the margin on one loan. Lending to every applicant loses about 22 cents per dollar before interest; the model's value is in choosing whom not to lend to (notebook 06).

The datamart code in `utils/` starts from my Assignment 1 pipeline: https://github.com/code-vibes-only/cs611-assignment-1

## What the DAG does each month
One DAG (`dags/dag.py`) runs once per month from 2023-01 to 2025-07 and is backfilled with `catchup=True`, one month at a time. Each run, for its month T (`{{ ds }}`):

| Stage | Tasks | Script |
|---|---|---|
| Label store | `bronze_label_store` → `silver_label_store` → `gold_label_store` | `data_processing.py` |
| Feature store | `bronze_*` → `silver_*` for attributes, financials, clickstream → `gold_feature_store` | `data_processing.py` |
| Training (only in training months) | `model_train_logreg` (logistic regression), `model_train_xgb` (XGBoost): a random search over each one's settings, scored by 3-fold CV AUC → `model_select`: the higher test AUC | `model_train.py`, `model_select.py` |
| Inference | `model_inference`: the selected model scores month T's applications | `model_inference.py` |
| Monitoring | `model_monitor`: stability of month T, performance of month T − 6 | `model_monitor.py` |
| Visualisation | `model_monitor_charts`: redraws the monitoring charts | `model_monitor_charts.py` |

Every task waits for its own run of the month before (`depends_on_past` and `wait_for_downstream`), so a failed month holds the backfill instead of being skipped over: no month is built on a half-built earlier one, and once a model is due, scoring a month without one is an error, not a skip.

## Repository layout
- `dags/dag.py`: the Airflow DAG
- `data_processing.py`: runs one month of one datamart table, using the bronze/silver/gold functions in `utils/`
- `model_train.py`, `model_select.py`, `model_inference.py`, `model_monitor.py`, `model_monitor_charts.py`: the ML pipeline steps, one script per task
- `notebooks/01_datamart_check.ipynb`: the DAG's datamart against Assignment 1's
- `notebooks/02_model_training.ipynb`: label availability, feature stability, candidate models, leakage checks
- `notebooks/03_feature_borrowing.ipynb`: features from the sample datamarts tested as extra columns, under a rule set in advance
- `notebooks/04_feature_pruning.ipynb`: the features neither candidate uses tested for removal, under a rule set in advance
- `notebooks/05_monitoring_check.ipynb`: predictions and monitoring checked, one month recomputed by hand
- `notebooks/06_model_choice.ipynb`: the two candidates weighed on train, validation, test and OOT AUC, with a bootstrap interval on the margin; precision and recall at each score cut-off, and whether the score reads as a probability; what the model is worth in loans and dollars from the loan ledger at typical lending rates, and the cut-off the cost of errors implies; what drives the score
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

The login, the Fernet key and the token-less JupyterLab are fixed in `docker-compose.yaml` so the pipeline runs with no setup; in production they would come from a secrets store.

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
- **Prediction point:** the application month. Features use information up to that month only; the clickstream features average the 3 months up to and including it. For the first two application months (2023-01 and 2023-02) the window reaches before the data starts, so their averages cover one and two months.
- **Labels mature 6 months after application.** A label is 30 or more days past due at the sixth month of the loan. On this ledger that is a loss, not a late payment: every such loan ended its ten installments with $6,000 to $10,000 unpaid, and every other loan was repaid in full (notebook 06). A model trained at month T uses only labels dated before T, so applications up to T − 7. The model is trained at 2024-03-01 on applications 2023-01 to 2023-06 (80/20 train/test) and 2023-07 to 2023-08 (out-of-time, OOT). Imputation and scaling are steps of the pipeline the random search tunes, so in every validation fold, as on test and OOT, they are fitted on the training part only.
- **Features:** the model trains on 59 of the feature store's 66 features (`FEATURE_COLUMNS` in `model_train.py`, listed per train date so a refresh can use its own list while earlier dates rebuild unchanged). Notebook 04 dropped the 7 that neither candidate used on the training split; the selected model's test AUC stayed within 0.005 at three train dates. Age is among the 7, which also keeps a protected attribute out of the decision; names and SSNs never enter the feature store. The feature store keeps all 66.
- **Selection:** each candidate is tuned on its validation folds (3-fold cross-validation on the training split; XGBoost's random search draws 60 of its 216 settings, logistic regression tries all 20 values of C); the candidate with the higher test AUC goes into the selection record that inference reads. The record keeps every candidate's train, validation, test and OOT AUC, so the approver can see a candidate that drops sharply from train to test, and a field for the approver's sign-off. The OOT months are not used to choose the model or its features (notebooks 03 and 04 decide on test AUC too): they stay the untouched check of the chosen model, and its OOT AUC is the baseline for performance monitoring. At 2024-03-01 XGBoost leads on validation, test and OOT by 0.011 to 0.015 each; the test margin on its own is within sampling noise (95% bootstrap interval −0.009 to +0.032), so the case is that all three agree. Its train − test gap (0.072, against 0.027 for logistic regression) means its training score overstates it, so logistic regression stays in the model bank as the stable alternative, and the explainable one: its coefficients give a declined applicant a reason (notebook 06).
- **Cut-off:** the pipeline outputs a probability of default, not a decision. The score reads as a probability on average (mean 0.273 against a bad rate of 0.282 on test) and within 6 points in the safer half of applicants, where the cut-offs below fall; in the second-riskiest tenth it predicts 57% where 77% default (notebook 06). Declining or referring the applications above a cut-off trades a missed default (about $7,500) against a good customer turned away: the margin on one loan. The ledger records no interest, so the margin is the interest a good loan would earn at a typical rate, before funding costs: APR ÷ 12 × $55,000, the balances owed over its ten months. With a margin *m*, flagging pays above a score of m / (7,531 + m): 0.07 at a 12% APR (the US banks' average on personal loans, $550), 0.15 at 30% (the average offer to credit scores under 580, $1,375) and 0.18 at 36% (the US cap on loans to service members, $1,650). At 30%, the base case for a book where 29% of applicants default, the 0.15 cut-off declines 58% of test applicants, keeps out 86% of the unpaid balances and turns away 47% of the good customers; over a year of 6,000 applications it turns the $7.1M loss of approving everyone into a $1.4M profit, where 0.5 still loses $0.56M (notebook 06). Credit risk sets the margin and the cut-off, and whether "flag" means decline, refer or price.
- **Monitoring:** stability is measured in the month scored (PSI of the scores and of each feature against the training data); performance (AUC, Gini) six months later, when the labels arrive. Status bands: PSI < 0.1 green, 0.1–0.25 amber, > 0.25 red; AUC within 0.05 of the OOT AUC at training green, within 0.10 amber, otherwise red. The OOT AUC is the baseline because those months are the model's untouched check on later data; it is also the highest of the three estimates, so the bands are strict: against the test AUC (0.822) only 2024-08 and 2025-01 would be amber. The score PSI and the bad-rate comparison do not depend on the baseline. The PSI bands are the usual industry ones; the AUC bands, and the SOP's limits below (3 months in a row, 5 points, every 6 months), are our starting settings, for the model owner and the approver to tune once the monitoring has a longer history.

## Monitoring charts
In every run, `model_monitor_charts` redraws three charts in `datamart/monitoring_charts/` from every monitoring partition up to that month:
- `performance.png`: the AUC of each scored month against the OOT AUC at training, coloured by status; below it, the actual bad rate against the predicted one (the mean score).
- `score_stability.png`: the PSI of each month's scores against the training scores, with the 0.1 and 0.25 bands.
- `feature_stability.png`: the CSI (characteristic stability index: the PSI of one feature) status of the 20 most drifted features, month by month.

The charts mark the first month whose score PSI turned red, once the run has reached it. What they show over the backfill (applications scored from 2024-03 to 2025-01):
- Until 2024-06 everything is green: score PSI 0.015 to 0.046, AUC 0.814 to 0.844.
- Applicants from 2024-07 have no clickstream: the clickstream covers only the customers who applied up to 2024-06 (for them it runs to 2024-12), and no later applicant was added to it. In that month's run the score PSI jumps to 0.417 and all 16 clickstream features the model uses turn red; the other 43 features stay green.
- Performance confirms it six months later: the 2024-07 applications score an AUC of 0.801 (amber) and 2024-08's 0.751 (red); 6 of the 7 months from 2024-07 are amber or red against the OOT baseline.
- The model also under-predicts from 2024-07: its mean score is 2 to 9 points below the actual bad rate, against at most 4 points before. In money, July 2024's applicants defaulted 34 loans more than the scores implied, about $254k of unpaid balances the expected loss missed (notebook 06).

## Model governance and refresh SOP
Roles: the **model owner** (data scientist) reads the charts each month and runs this SOP; the **data engineer** owns the source feeds and the pipeline; the **model approver** (credit risk) signs off any change of model and writes their name into the selection record (`approved_by`) when it is deployed.

A red status is first traced to its cause. A data or pipeline fault is fixed and the affected months rerun; a model is retrained only when the fault cannot be fixed or the applicants have really changed.

| Trigger (in the monitoring table) | Action | Owner | When |
|---|---|---|---|
| All green | None; review the charts | Model owner | Monthly |
| Score PSI or a feature's CSI amber | Find the cause: a feed or pipeline change, or a real change in applicants. Record it | Model owner | Within the month |
| A feature's CSI red | Check the feed with the data engineer. A feed or pipeline fault: fix it and rerun from the first affected month (below). A feature that cannot be restored for new applicants: flag the scores made without it, decide with the approver whether to pause inference until a model without the feature is approved, and train that challenger | Model owner, data engineer, model approver | Within a week |
| Score PSI red | As for a red feature, and tell the approver: a fixed score cut-off now approves a different share of applicants | Model owner, model approver | Within a week |
| AUC red, or AUC amber 3 months in a row | Train a challenger on the latest matured labels | Model owner | The next month |
| Mean score and actual bad rate more than 5 points apart, 2 months in a row | Tell the approver: losses expected from the score are off. Recalibrate, or train a challenger | Model owner, model approver | The next month |
| Every 6 months, even when green | Train a challenger, to test for a better model | Model owner | Scheduled |

How to rerun after a fix: clear the failed task and everything downstream of it, in that month and every later month, so the 3-month clickstream window, the training data, the predictions, the monitoring and the charts are all rebuilt from the fix:

```sh
docker-compose exec airflow-scheduler airflow tasks clear dag -t <task_id> -s <first affected month> -e 2025-07-01 --downstream -y
```

A model is never retrained in place: a retrain is a new train date in `TRAIN_DATES`, so the artefacts and selection record of a date that has already scored months stay as they were.

How a challenger replaces the model:
1. **Train.** On a branch, add the refresh month to `TRAIN_DATES` in `model_train.py` with its own feature list in `FEATURE_COLUMNS` (the earlier date keeps its list, so it rebuilds unchanged). Running that month's `model_` tasks trains both candidates and selects the better on test AUC.
2. **Compare.** The challenger must beat the current model on the same recent months, which neither model was tuned or selected on, scored on the data the current model now receives (for a feature that stopped arriving, with it blanked). Otherwise the comparison favours the model that relies on the missing feature.
3. **Approve and deploy.** The approver signs off the comparison and writes their name into the new selection record; merging the branch deploys it, since from the refresh month inference reads the newest selection record. A cautious switch scores both models for a month or two first (shadow): monitoring measures every model that scored a month, and the monitoring table holds rows for each `model_name`.
4. **Roll back.** Earlier artefacts and selection records stay in `model_bank/`. Removing the new selection record and the new model's folder under `datamart/gold/model_predictions/`, then clearing `model_inference` from the refresh month to the last run (the command above), puts the previous model back.

Applied to this backfill:
- **2024-07 run:** score PSI and 16 feature CSIs red. The cause is the clickstream: customers who applied from 2024-07 were never added to it, so for them every value is missing and the model scores them with the training medians (for earlier customers the clickstream runs to 2024-12). The data engineer's question is why new applicants are not captured; it cannot be fixed from the data supplied, so the SOP calls for a challenger without clickstream. At a fixed 0.5 cut-off the share of applicants flagged falls from 18–23% to 11.5–16.5%, which shows in the month scored; once the labels arrive, the defaulters caught fall from 44–54% to 28–41%, and the loans approved in July to September leave 18 to 19 cents per dollar unpaid against 14.5 to 15 before (notebook 06).
- **A plain refresh would not fix it.** A model trained at month T learns from applications T − 14 to T − 9 (plus T − 8 and T − 7 as OOT), so through the 2025-08 run its training split still has months with clickstream: a refreshed model would learn to rely on a feature its new applicants do not have.
- **2025-01 and 2025-02 runs:** performance confirms the alert: 2024-07 amber, 2024-08 red, and in both months the mean score is nearly 7 points below the actual bad rate. This triggers the challenger and the approver's review.
- **What to expect from the challenger:** in notebook 02, looking back with every label, XGBoost without clickstream scored an AUC of 0.804 on the 2024-07 to 2025-01 applications against 0.797 with it, despite a lower OOT AUC while the clickstream was there (0.806 against 0.854; fixed settings, all 66 features). The challenger itself is not trained in this pipeline.

## Deployment options
- **Batch, monthly (this pipeline):** Airflow scores each month's applications after the month closes and writes them to the gold table. It suits backfills, monitoring and portfolio reporting; with labels 6 months behind, monthly monitoring is frequent enough.
- **Real-time, for the lending decision:** a customer is approved or declined when they apply, so in production the same artefact would sit behind an endpoint that scores one application at a time. The features must be computed the same way at that moment (the 3-month clickstream average needs a feature store that is updated as events arrive). The batch pipeline would stay, to log the predictions and run the monitoring.
- **Rolling out a new model:** shadow first (scored alongside, decisions unchanged), then canary (a small share of applications), then the full switch, keeping the previous model ready (blue/green) for a rollback.
