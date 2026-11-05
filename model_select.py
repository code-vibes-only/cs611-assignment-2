import argparse
import glob
import json
import pickle

from model_train import TRAIN_DATES

# to call this script: python model_select.py --snapshotdate "2024-03-01"

# each candidate was tuned on the validation folds, so the final choice between them is made on the test split;
# the out-of-time months are kept as the untouched check of the chosen model: recorded, never compared
SELECTION_METRIC = "auc_test"


def main(snapshotdate):
    if snapshotdate not in TRAIN_DATES:
        print(f"{snapshotdate} is not a training month, skipping")
        return

    # compare every candidate trained this month
    date_suffix = snapshotdate.replace('-', '_')
    candidates = {}
    metrics = {}
    for file_path in sorted(glob.glob(f"model_bank/credit_model_*_{date_suffix}.pkl")):
        with open(file_path, 'rb') as file:
            model_artefact = pickle.load(file)
        version = model_artefact['model_version']
        results = model_artefact['results']
        candidates[version] = results[SELECTION_METRIC]
        # the whole picture for the approver: a large drop from train to test is a sign of over-tuning
        metrics[version] = {
            "auc_train": results["auc_train"],
            "auc_cv": results["auc_cv"],
            "auc_test": results["auc_test"],
            "auc_oot": results["auc_oot"],
            "train_test_gap": results["auc_train"] - results["auc_test"],
        }
        print(version, {name: round(value, 4) for name, value in metrics[version].items()})

    best_model_version = max(candidates, key=candidates.get)

    # selection record: inference reads this to find the model to use
    selection = {
        "train_date": snapshotdate,
        "selected_model": best_model_version,
        "selection_metric": SELECTION_METRIC,
        "candidates": candidates,
        "metrics": metrics,
    }
    file_path = f"model_bank/model_selection_{date_suffix}.json"
    with open(file_path, 'w') as file:
        json.dump(selection, file, indent=2)
    print("selected:", best_model_version, "saved to:", file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    main(args.snapshotdate)
