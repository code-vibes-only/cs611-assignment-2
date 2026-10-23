import argparse
import os
import pickle
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# to call this script: python model_train.py --snapshotdate "2024-03-01" --modeltype xgb

# months in which a model is trained (the first month with 8 months of matured labels, notebooks/02_model_training.ipynb)
TRAIN_DATES = ["2024-03-01"]

# the features the model uses: the feature store's 66 minus the 7 that neither candidate used on the 2024-03-01
# training split (notebooks/04_feature_pruning.ipynb); the feature store itself keeps all 66
FEATURE_COLUMNS = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Months", "Total_EMI_per_month",
    "Amount_invested_monthly", "Debt_to_Annual_Salary", "EMI_to_Monthly_Salary", "Occupation_Accountant",
    "Occupation_Architect", "Occupation_Developer", "Occupation_Doctor", "Occupation_Engineer",
    "Occupation_Entrepreneur", "Occupation_Journalist", "Occupation_Lawyer", "Occupation_Manager",
    "Occupation_Mechanic", "Occupation_Media_Manager", "Occupation_Musician", "Occupation_Scientist",
    "Occupation_Teacher", "Occupation_Writer", "Credit_Mix_Bad", "Credit_Mix_Standard", "Credit_Mix_Good",
    "Payment_of_Min_Amount_Yes", "Payment_of_Min_Amount_No", "Payment_of_Min_Amount_NM",
    "Payment_Behaviour_Low_spent_Small_value_payments", "Payment_Behaviour_Low_spent_Medium_value_payments",
    "Payment_Behaviour_Low_spent_Large_value_payments", "Payment_Behaviour_High_spent_Small_value_payments",
    "Payment_Behaviour_High_spent_Medium_value_payments", "Payment_Behaviour_High_spent_Large_value_payments",
    "fe_1_avg_3m", "fe_2_avg_3m", "fe_3_avg_3m", "fe_4_avg_3m", "fe_5_avg_3m", "fe_7_avg_3m", "fe_8_avg_3m",
    "fe_9_avg_3m", "fe_10_avg_3m", "fe_11_avg_3m", "fe_12_avg_3m", "fe_13_avg_3m", "fe_14_avg_3m",
    "fe_15_avg_3m", "fe_18_avg_3m", "fe_19_avg_3m",
]

# hyperparameter search space for each candidate
PARAM_DIST = {
    "logreg": {
        "C": np.logspace(-3, 2, 20),  # 20 regularisation strengths from 0.001 to 100
    },
    "xgb": {
        "n_estimators": [25, 50, 100],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "min_child_weight": [1, 3, 5],
    },
}


def main(snapshotdate, modeltype):
    if snapshotdate not in TRAIN_DATES:
        print(f"{snapshotdate} is not a training month, skipping")
        return
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    # windows are on the application month; an application's label is known 6 months later,
    # so a model trained at T may only use applications up to T - 7 months (label date < T)
    config = {}
    config["model_train_date_str"] = snapshotdate
    config["model_train_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["label_maturity_months"] = 6
    config["train_test_period_months"] = 6
    config["oot_period_months"] = 2
    config["train_test_ratio"] = 0.8
    config["oot_end_date"] = config["model_train_date"] - relativedelta(months=config["label_maturity_months"] + 1)
    config["oot_start_date"] = config["oot_end_date"] - relativedelta(months=config["oot_period_months"] - 1)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=config["train_test_period_months"])
    pprint.pprint(config)

    # --- get labels: only those whose label date is before the train date ---
    labels_pdf = pd.read_parquet("datamart/gold/label_store/")
    labels_pdf = labels_pdf[pd.to_datetime(labels_pdf["snapshot_date"]) < config["model_train_date"]]
    print("labels available at train date:", labels_pdf.shape[0])

    # --- get features at the application month ---
    features_pdf = pd.read_parquet("datamart/gold/feature_store/")
    features_pdf["snapshot_date"] = pd.to_datetime(features_pdf["snapshot_date"])
    features_pdf = features_pdf[(features_pdf["snapshot_date"] >= config["train_test_start_date"]) & (features_pdf["snapshot_date"] <= config["oot_end_date"])]
    print("applications in the train/test and oot windows:", features_pdf.shape[0])

    # join on customer + application month (label_store.loan_start_date = feature_store.snapshot_date)
    labels_pdf = labels_pdf[["Customer_ID", "loan_start_date", "label"]].rename(columns={"loan_start_date": "snapshot_date"})
    labels_pdf["snapshot_date"] = pd.to_datetime(labels_pdf["snapshot_date"])
    data_pdf = features_pdf.merge(labels_pdf, on=["Customer_ID", "snapshot_date"], how="inner")
    # Spark writes rows in no fixed order; sort so the random split and the model are the same on every rebuild
    data_pdf = data_pdf.sort_values(["snapshot_date", "Customer_ID"]).reset_index(drop=True)
    print("joined rows", data_pdf.shape[0])

    # --- split data into train - test - oot ---
    feature_cols = FEATURE_COLUMNS
    oot_pdf = data_pdf[data_pdf["snapshot_date"] >= config["oot_start_date"]]
    train_test_pdf = data_pdf[data_pdf["snapshot_date"] <= config["train_test_end_date"]]

    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"],
        test_size=1 - config["train_test_ratio"],
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"]
    )
    print('X_train', X_train.shape[0], 'X_test', X_test.shape[0], 'X_oot', X_oot.shape[0])

    # --- preprocessing fitted on the training split only (no train-test contamination) ---
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))

    def preprocess(X):
        return scaler.transform(imputer.transform(X))

    X_train_processed = preprocess(X_train)
    X_test_processed = preprocess(X_test)
    X_oot_processed = preprocess(X_oot)

    # --- train model: random search with cross-validation on AUC ---
    if modeltype == "logreg":
        estimator = LogisticRegression(max_iter=1000, random_state=88)
    else:
        estimator = xgb.XGBClassifier(eval_metric="logloss", random_state=88)

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_DIST[modeltype],
        scoring="roc_auc",
        n_iter=20,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train_processed, y_train)
    best_model = random_search.best_estimator_
    print("Best parameters found: ", random_search.best_params_)

    # --- evaluate on train, test and oot ---
    results = {}
    for split, X, y in [("train", X_train_processed, y_train), ("test", X_test_processed, y_test), ("oot", X_oot_processed, y_oot)]:
        auc = roc_auc_score(y, best_model.predict_proba(X)[:, 1])
        results["auc_" + split] = auc
        results["gini_" + split] = round(2 * auc - 1, 3)
    pprint.pprint(results)

    # --- reference distributions for stability monitoring: training scores and features ---
    train_scores = best_model.predict_proba(X_train_processed)[:, 1]
    reference = {
        "score_bin_edges": np.quantile(train_scores, np.linspace(0, 1, 11)),  # deciles of the training scores
        "train_scores": train_scores,
        "train_features": X_train.reset_index(drop=True),  # raw values, before imputation
    }

    # --- prepare model artefact to save ---
    model_artefact = {}
    model_artefact['model'] = best_model
    model_artefact['model_version'] = f"credit_model_{modeltype}_" + snapshotdate.replace('-', '_')
    model_artefact['preprocessing_transformers'] = {'imputer': imputer, 'stdscaler': scaler}
    model_artefact['feature_columns'] = feature_cols
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {
        'X_train': X_train.shape[0], 'X_test': X_test.shape[0], 'X_oot': X_oot.shape[0],
        'y_train': round(y_train.mean(), 3), 'y_test': round(y_test.mean(), 3), 'y_oot': round(y_oot.mean(), 3),
    }
    model_artefact['results'] = results
    model_artefact['hp_params'] = random_search.best_params_
    model_artefact['reference'] = reference

    # --- save artefact to model bank ---
    os.makedirs("model_bank/", exist_ok=True)
    file_path = os.path.join("model_bank/", model_artefact['model_version'] + '.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)
    print(f"Model saved to {file_path}")

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modeltype", type=str, required=True, choices=["logreg", "xgb"])
    args = parser.parse_args()

    main(args.snapshotdate, args.modeltype)
