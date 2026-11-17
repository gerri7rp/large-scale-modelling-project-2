import csv
import os
from pathlib import Path

from utils import check_solution


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(
    os.environ.get("RESULTS_DIR", ROOT / "results/raw")
)


def save_and_validate_results(matrix, csv_path, run_type, run_id, n_ranks, chunk_size, N, max_iter, tolerance, init_value,
                              time_compute, time_comm, time_wait):
    is_valid = validate_result(matrix)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "run_id", "type", "is_valid", "n_ranks", "rank", "chunk_size","N", "max_iter", 
        "tolerance", "init_value", "t_compute", "t_communication", "t_wait_done", "t_total",
    ]

    # Append rows for this experiment.
    with csv_path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if csvfile.tell() == 0:
            writer.writerow(columns)

        for rank in range(n_ranks):
            compute = float(time_compute[rank])
            communication = float(time_comm[rank])
            wait = float(time_wait[rank])

            writer.writerow([
                run_id, run_type, is_valid, n_ranks, rank, chunk_size,
                int(N), int(max_iter), float(tolerance), int(init_value),
                compute, communication, wait,
                compute + communication + wait,
            ])

    return is_valid



def validate_result(matrix):
    solution_tolerance = 10e-3

    is_valid = check_solution.is_solution_correct(matrix, solution_tolerance)
    print(f"is valid: {is_valid}")
    return is_valid


        
    # if is_valid:
    #     print("\n\nResults are correct.\n\n")

    # elif is_valid == None:
    #     print("\n\nThere is no reference image available for this size.\n\n")

    # else:
    #     raise Exception("\n\nTHE RESULTING IMAGE DOESN'T MATCH THE REFERENCE IMAGE!!\n\n")
    #     # print("\n\nTHE RESULTING IMAGE DOESN'T MATCH THE REFERENCE IMAGE!!\n\n")

    # return is_valid

