import argparse
import os
import pickle
import platform
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# to call this script: python model_train.py --snapshotdate "2024-03-01" --modeltype xgb

# months in which a model is trained (the first month with 8 months of matured labels, notebooks/02_model_training.ipynb)
TRAIN_DATES = ["2024-03-01"]

# the features each train date's model uses. A refresh adds its own date with its own list, so earlier dates
# rebuild unchanged. 2024-03-01: the feature store's 66 minus the 7 that neither candidate used on its training
# split (notebooks/04_feature_pruning.ipynb); the feature store itself keeps all 66
FEATURE_COLUMNS = {
    "2024-03-01": [
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
    ],
}

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

# draws of the random search; the whole grid when it is smaller (logistic regression's 20 values of C)
N_ITER = 60


def training_config(train_date_str):
    """The windows for a train date. Windows are on the application month; an application's label is known
    6 months later, so a model trained at T may only use applications up to T - 7 months (label date < T)."""
    config = {}
    config["model_train_date_str"] = train_date_str
    config["model_train_date"] = datetime.strptime(train_date_str, "%Y-%m-%d")
    config["label_maturity_months"] = 6
    config["train_test_period_months"] = 6
    config["oot_period_months"] = 2
    config["train_test_ratio"] = 0.8
    config["oot_end_date"] = config["model_train_date"] - relativedelta(months=config["label_maturity_months"] + 1)
    config["oot_start_date"] = config["oot_end_date"] - relativedelta(months=config["oot_period_months"] - 1)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=config["train_test_period_months"])
    return config


def load_training_data(train_date_str, root="."):
    """The labelled applications a model trained at this date may learn from, split into the train/test window and
    the out-of-time (OOT) window. Each frame keeps the keys (Customer_ID, snapshot_date), the features and the label.
    Shared with the notebooks (root = the repository folder), so an analysis rebuilds exactly what the model saw."""
    config = training_config(train_date_str)

    # labels: only those whose label date is before the train date
    labels_pdf = pd.read_parquet(os.path.join(root, "datamart/gold/label_store/"))
    labels_pdf = labels_pdf[pd.to_datetime(labels_pdf["snapshot_date"]) < config["model_train_date"]]

    # features at the application month, inside the windows
    features_pdf = pd.read_parquet(os.path.join(root, "datamart/gold/feature_store/"))
    features_pdf["snapshot_date"] = pd.to_datetime(features_pdf["snapshot_date"])
    features_pdf = features_pdf[(features_pdf["snapshot_date"] >= config["train_test_start_date"]) & (features_pdf["snapshot_date"] <= config["oot_end_date"])]

    # join on customer + application month (label_store.loan_start_date = feature_store.snapshot_date)
    labels_pdf = labels_pdf[["Customer_ID", "loan_start_date", "label"]].rename(columns={"loan_start_date": "snapshot_date"})
    labels_pdf["snapshot_date"] = pd.to_datetime(labels_pdf["snapshot_date"])
    data_pdf = features_pdf.merge(labels_pdf, on=["Customer_ID", "snapshot_date"], how="inner")
    # Spark writes rows in no fixed order; sort so the random split and the model are the same on every rebuild
    data_pdf = data_pdf.sort_values(["snapshot_date", "Customer_ID"]).reset_index(drop=True)

    oot_pdf = data_pdf[data_pdf["snapshot_date"] >= config["oot_start_date"]]
    train_test_pdf = data_pdf[data_pdf["snapshot_date"] <= config["train_test_end_date"]]
    return config, train_test_pdf, oot_pdf


def split_train_test(train_test_pdf, config):
    """The stratified 80/20 split of the train/test window, the same on every call (fixed seed, sorted input)"""
    return train_test_split(train_test_pdf, test_size=1 - config["train_test_ratio"], random_state=88, shuffle=True,
                            stratify=train_test_pdf["label"])


def main(snapshotdate, modeltype):
    if snapshotdate not in TRAIN_DATES:
        print(f"{snapshotdate} is not a training month, skipping")
        return
    print('\n\n---starting job---\n\n')

    # --- get the labelled applications this train date may learn from ---
    config, train_test_pdf, oot_pdf = load_training_data(snapshotdate)
    pprint.pprint(config)
    print("applications in the train/test and oot windows:", train_test_pdf.shape[0] + oot_pdf.shape[0])

    # --- split data into train - test - oot ---
    feature_cols = FEATURE_COLUMNS[snapshotdate]
    train_pdf, test_pdf = split_train_test(train_test_pdf, config)
    X_train, y_train = train_pdf[feature_cols], train_pdf["label"]
    X_test, y_test = test_pdf[feature_cols], test_pdf["label"]
    X_oot, y_oot = oot_pdf[feature_cols], oot_pdf["label"]
    print('X_train', X_train.shape[0], 'X_test', X_test.shape[0], 'X_oot', X_oot.shape[0])

    # --- train model: random search with cross-validation on AUC ---
    # the imputer and scaler are steps of the searched pipeline, so in every fold they are fitted on that fold's
    # training part only (no train-test contamination, in the validation folds as well as on test and oot)
    if modeltype == "logreg":
        estimator = LogisticRegression(max_iter=1000, random_state=88)
    else:
        estimator = xgb.XGBClassifier(eval_metric="logloss", random_state=88)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    param_distributions = {"model__" + name: values for name, values in PARAM_DIST[modeltype].items()}
    grid_size = int(np.prod([len(values) for values in PARAM_DIST[modeltype].values()]))

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        scoring="roc_auc",
        n_iter=min(N_ITER, grid_size),
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    best_pipeline = random_search.best_estimator_  # refitted on the whole training split with the best settings
    imputer, scaler, best_model = (best_pipeline.named_steps[step] for step in ["imputer", "scaler", "model"])
    best_params = {name.replace("model__", ""): value for name, value in random_search.best_params_.items()}
    print("Best parameters found: ", best_params)

    def preprocess(X):
        return scaler.transform(imputer.transform(X))

    X_train_processed = preprocess(X_train)
    X_test_processed = preprocess(X_test)
    X_oot_processed = preprocess(X_oot)

    # --- evaluate on validation (the cross-validation folds the search tuned on), train, test and oot ---
    results = {}
    results["auc_cv"] = random_search.best_score_  # mean AUC of the best settings over the 3 validation folds
    results["gini_cv"] = round(2 * random_search.best_score_ - 1, 3)
    for split, X, y in [("train", X_train_processed, y_train), ("test", X_test_processed, y_test), ("oot", X_oot_processed, y_oot)]:
        auc = roc_auc_score(y, best_model.predict_proba(X)[:, 1])
        results["auc_" + split] = auc
        results["gini_" + split] = round(2 * auc - 1, 3)
    pprint.pprint(results)

    # --- reference distributions for stability monitoring: training scores and features ---
    reference = {
        "train_scores": best_model.predict_proba(X_train_processed)[:, 1],
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
    # applications per month in the windows, and the library versions the model was fitted with: what a registry
    # needs to rebuild or audit the model
    model_artefact['rows_by_month'] = {str(month.date()): int(n) for month, n in
                                       pd.concat([train_test_pdf, oot_pdf]).groupby("snapshot_date").size().items()}
    model_artefact['library_versions'] = {
        'python': platform.python_version(), 'scikit-learn': sklearn.__version__, 'xgboost': xgb.__version__,
        'pandas': pd.__version__, 'numpy': np.__version__,
    }
    model_artefact['results'] = results
    model_artefact['hp_params'] = best_params
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
