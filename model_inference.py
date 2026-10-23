import argparse
import glob
import json
import os
import pickle

import pandas as pd

# to call this script: python model_inference.py --snapshotdate "2024-03-01"


def current_model_version(snapshotdate):
    """The model selected at the latest train date on or before this month, or None if none is trained yet"""
    selections = []
    for file_path in glob.glob("model_bank/model_selection_*.json"):
        with open(file_path) as file:
            selections.append(json.load(file))
    selections = [s for s in selections if s["train_date"] <= snapshotdate]
    if not selections:
        return None
    return max(selections, key=lambda s: s["train_date"])["selected_model"]


def main(snapshotdate):
    # --- find the best model available at this month ---
    model_version = current_model_version(snapshotdate)
    if model_version is None:
        print(f"no model trained on or before {snapshotdate}, skipping")
        return

    # --- load feature store partition for the month (the applications to score) ---
    feature_path = "datamart/gold/feature_store/gold_feature_store_" + snapshotdate.replace('-', '_') + ".parquet"
    if not os.path.exists(feature_path):
        print(f"no applications for {snapshotdate}, skipping")
        return
    features_pdf = pd.read_parquet(feature_path)
    print("loaded from:", feature_path, "row count:", features_pdf.shape[0])

    # --- load model artefact from model bank ---
    with open(f"model_bank/{model_version}.pkl", 'rb') as file:
        model_artefact = pickle.load(file)
    print("Model loaded successfully! " + model_version)

    # --- preprocess with the transformers fitted at training ---
    X_inference = features_pdf[model_artefact["feature_columns"]]
    X_inference = model_artefact["preprocessing_transformers"]["imputer"].transform(X_inference)
    X_inference = model_artefact["preprocessing_transformers"]["stdscaler"].transform(X_inference)

    # --- model prediction inference ---
    y_inference = model_artefact["model"].predict_proba(X_inference)[:, 1]

    # prepare output
    y_inference_pdf = features_pdf[["Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = model_version
    y_inference_pdf["model_predictions"] = y_inference

    # --- save model inference to datamart gold table ---
    gold_directory = f"datamart/gold/model_predictions/{model_version}/"
    os.makedirs(gold_directory, exist_ok=True)
    filepath = gold_directory + model_version + "_predictions_" + snapshotdate.replace('-', '_') + '.parquet'
    y_inference_pdf.to_parquet(filepath, index=False)
    print('saved to:', filepath, "mean prediction:", round(y_inference.mean(), 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    main(args.snapshotdate)
