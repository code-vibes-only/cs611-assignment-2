import argparse
import glob
import json
import pickle

from model_train import TRAIN_DATES

# to call this script: python model_select.py --snapshotdate "2024-03-01"

SELECTION_METRIC = "auc_oot"  # out-of-time AUC: performance on the most recent months the model never saw


def main(snapshotdate):
    if snapshotdate not in TRAIN_DATES:
        print(f"{snapshotdate} is not a training month, skipping")
        return

    # compare every candidate trained this month
    date_suffix = snapshotdate.replace('-', '_')
    candidates = {}
    for file_path in sorted(glob.glob(f"model_bank/credit_model_*_{date_suffix}.pkl")):
        with open(file_path, 'rb') as file:
            model_artefact = pickle.load(file)
        candidates[model_artefact['model_version']] = model_artefact['results'][SELECTION_METRIC]
        print(model_artefact['model_version'], SELECTION_METRIC, round(candidates[model_artefact['model_version']], 4))

    best_model_version = max(candidates, key=candidates.get)

    # selection record: inference reads this to find the model to use
    selection = {
        "train_date": snapshotdate,
        "selected_model": best_model_version,
        "selection_metric": SELECTION_METRIC,
        "candidates": candidates,
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
