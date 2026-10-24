import argparse
import glob
import os
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# to call this script: python model_monitor.py --snapshotdate "2024-09-01"

LABEL_MATURITY_MONTHS = 6  # 30dpd_6mob: an application's label is known 6 months after it is scored


def psi(reference, current):
    """Population Stability Index of current values against reference values.
    Bins are the reference deciles; missing values are their own bin, so a feed that stops shows up."""
    reference, current = pd.Series(reference, dtype=float), pd.Series(current, dtype=float)
    edges = np.unique(np.nanquantile(reference, np.linspace(0.1, 0.9, 9)))

    def proportions(values):
        counts = np.bincount(np.digitize(values.dropna(), edges, right=True), minlength=len(edges) + 1)
        counts = np.append(counts, values.isna().sum())
        return np.clip(counts / len(values), 1e-4, None)  # an empty bin would make log() infinite

    ref, cur = proportions(reference), proportions(current)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_status(value):
    """PSI bands: < 0.1 little or no drift, 0.1-0.25 moderate drift, > 0.25 significant drift"""
    return "green" if value < 0.1 else "amber" if value <= 0.25 else "red"


def auc_status(value, auc_at_training):
    """AUC drop from the model's out-of-time AUC at training"""
    drop = auc_at_training - value
    return "green" if drop <= 0.05 else "amber" if drop <= 0.10 else "red"


def load_predictions(month_str):
    """Predictions for one application month, from whichever model scored it (None if not scored)"""
    files = glob.glob("datamart/gold/model_predictions/*/*_predictions_" + month_str.replace('-', '_') + ".parquet")
    return pd.read_parquet(files[0]) if files else None


def load_model_artefact(model_version):
    with open(f"model_bank/{model_version}.pkl", 'rb') as file:
        return pickle.load(file)


def main(snapshotdate):
    rows = []

    # --- stability: this month's scores and features against the training reference ---
    predictions = load_predictions(snapshotdate)
    if predictions is not None:
        model_version = predictions["model_name"].iloc[0]
        model_artefact = load_model_artefact(model_version)
        reference = model_artefact["reference"]
        features = pd.read_parquet("datamart/gold/feature_store/gold_feature_store_" + snapshotdate.replace('-', '_') + ".parquet")

        def add(metric, value, n, status):
            rows.append([model_version, snapshotdate, snapshotdate, "stability", metric, value, n, status])

        value = psi(reference["train_scores"], predictions["model_predictions"])
        add("psi_score", value, len(predictions), psi_status(value))
        add("mean_score", predictions["model_predictions"].mean(), len(predictions), "")
        for feature in model_artefact["feature_columns"]:
            value = psi(reference["train_features"][feature], features[feature])
            add("csi_" + feature, value, len(features), psi_status(value))

    # --- performance: the month whose labels mature now ---
    scored_month = (datetime.strptime(snapshotdate, "%Y-%m-%d") - relativedelta(months=LABEL_MATURITY_MONTHS)).strftime("%Y-%m-%d")
    scored = load_predictions(scored_month)
    label_path = "datamart/gold/label_store/gold_label_store_" + snapshotdate.replace('-', '_') + ".parquet"
    if scored is not None and os.path.exists(label_path):
        labels = pd.read_parquet(label_path)[["Customer_ID", "loan_start_date", "label"]]
        labels = labels.rename(columns={"loan_start_date": "snapshot_date"})
        joined = scored.merge(labels, on=["Customer_ID", "snapshot_date"], how="inner")
        model_version = scored["model_name"].iloc[0]
        auc_at_training = load_model_artefact(model_version)["results"]["auc_oot"]

        def add(metric, value, status):
            rows.append([model_version, snapshotdate, scored_month, "performance", metric, value, len(joined), status])

        auc = roc_auc_score(joined["label"], joined["model_predictions"])
        add("auc", auc, auc_status(auc, auc_at_training))
        add("gini", 2 * auc - 1, auc_status(auc, auc_at_training))
        add("actual_bad_rate", joined["label"].mean(), "")
        add("predicted_bad_rate", joined["model_predictions"].mean(), "")

    if not rows:
        print(f"nothing to monitor for {snapshotdate}, skipping")
        return

    # --- save monitoring results to datamart gold table ---
    monitoring_pdf = pd.DataFrame(rows, columns=["model_name", "run_date", "prediction_month", "metric_group", "metric", "value", "n", "status"])
    for c in ["run_date", "prediction_month"]:
        monitoring_pdf[c] = pd.to_datetime(monitoring_pdf[c]).dt.date
    gold_directory = "datamart/gold/model_monitoring/"
    os.makedirs(gold_directory, exist_ok=True)
    filepath = gold_directory + "model_monitoring_" + snapshotdate.replace('-', '_') + ".parquet"
    monitoring_pdf.to_parquet(filepath, index=False)
    print(monitoring_pdf[~monitoring_pdf["metric"].str.startswith("csi_")].to_string(index=False))
    print("features with red CSI:", (monitoring_pdf["metric"].str.startswith("csi_") & (monitoring_pdf["status"] == "red")).sum())
    print("saved to:", filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    main(args.snapshotdate)
