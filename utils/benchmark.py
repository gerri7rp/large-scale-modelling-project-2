import csv
import os
from pathlib import Path

from utils import check_solution


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(
    os.environ.get("RESULTS_DIR", ROOT / "results/raw")
)


def save_and_validate_results(csv_path, run_type, run_id, n_ranks, N, max_iter, n_iterations, tolerance,
                               init_value, jacobi_time, comm_time, flops, lups):
    # is_valid = validate_result(matrix)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "run_id", "type", "n_ranks", "rank", "N", "max_iter", "n_iterations",
        "tolerance", "init_value", "jacobi_time", "comm_time", "flops", "lups"
    ]

    # Append rows for this experiment.
    with csv_path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if csvfile.tell() == 0:
            writer.writerow(columns)

        for rank in range(n_ranks):
            writer.writerow([
                run_id, run_type, n_ranks, rank,
                int(N), int(max_iter), int(n_iterations), float(tolerance), int(init_value), 
                float(jacobi_time[rank]),
                float(comm_time[rank]),
                float(flops[rank]) if flops is not None else "",
                float(lups[rank]) if lups is not None else "",
            ])

    return f"Results saved to CSV file: {csv_path}.", True



def validate_result(matrix):
    solution_tolerance = 10e-3

    # Implement this
    # is_valid = check_solution.is_solution_correct(matrix, solution_tolerance)
    # print(f"is valid: {is_valid}")


    return None


        
    # if is_valid:
    #     print("\n\nResults are correct.\n\n")

    # elif is_valid == None:
    #     print("\n\nThere is no reference image available for this size.\n\n")

    # else:
    #     raise Exception("\n\nTHE RESULTING IMAGE DOESN'T MATCH THE REFERENCE IMAGE!!\n\n")
    #     # print("\n\nTHE RESULTING IMAGE DOESN'T MATCH THE REFERENCE IMAGE!!\n\n")

    # return is_valid

