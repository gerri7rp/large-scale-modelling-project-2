from pathlib import Path
import pandas as pd
import os
import argparse

"""
Call the file from main dirctory: python utils/merge_csvs.py 2026-10-11_1125
Using the experiment id you want to process.
"""


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(ROOT / "results")


def merge_csvs(experiment_id):
    input_dir  = Path(RESULTS_DIR / "raw")
    output_dir = Path(RESULTS_DIR / "processed" / experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    directories  = os.listdir(input_dir)

    for run_type in directories:
        # Get csv files

        if experiment_id not in os.listdir(Path(input_dir / run_type)):
            continue

        dir = Path(input_dir / run_type / experiment_id)
        csv_files = list(dir.glob("*.csv"))

        dfs = []

        # Read CSV files one by one
        for f in csv_files:
            dfs.append(pd.read_csv(f))

        # Concatenate and save once
        result = pd.concat(dfs, ignore_index=True)
        saving_path = Path(output_dir / f"{run_type}.csv")
        result.to_csv(saving_path, index=False)

        print(f"csv file saved to {saving_path}")

        # # Delete original CSV files
        # for f in csv_files:
        #     f.unlink()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_id",
        help="ID of the experiment to process"
    )
    args = parser.parse_args()

    merge_csvs(args.experiment_id)

