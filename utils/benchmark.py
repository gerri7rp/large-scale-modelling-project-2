import csv
import os
from pathlib import Path

from utils import check_solution


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(
    os.environ.get("RESULTS_DIR", ROOT / "results/raw")
)


def get_next_csv_path(run_type):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    copy_number = 1
    while True:
        csv_path = RESULTS_DIR / f"{run_type}_{copy_number}.csv"
        if not csv_path.exists():
            return csv_path
        copy_number += 1



def save_and_validate_results(image, csv_path, run_type, run_id, n_ranks, chunk_size, time_compute, time_comm, time_wait):
    is_valid = validate_result(image)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "run_id", "type", "is_valid", "n_ranks", "rank", "chunk_size",
        "width", "height", "xmin", "xmax", "ymin", "ymax",
        "t_compute", "t_communication", "t_wait_done", "t_total",
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
                int(size[0]), int(size[1]),
                *map(float, xlim), *map(float, ylim),
                compute, communication, wait,
                compute + communication + wait,
            ])




def validate_result(u, tol):
    """Validate the computed image and show a log."""
    is_valid = check_solution.is_resulting_image_correct(image, xlim, ylim)
        
    if is_valid:
        print("\n\nResults are correct.\n\n")

    elif is_valid == None:
        print("\n\nThere is no reference image available for this size.\n\n")

    else:
        raise Exception("\n\nTHE RESULTING IMAGE DOESN'T MATCH THE REFERENCE IMAGE!!\n\n")
        # print("\n\nTHE RESULTING IMAGE DOESN'T MATCH THE REFERENCE IMAGE!!\n\n")

    return is_valid






# def validate_and_save_image(image, xlim, ylim, run_id):
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#     image = np.asarray(image)

#     # Keep the expected orientation used by your existing checker.
#     is_correct = check_solution.is_resulting_image_correct(image)

#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots()
#     ax.imshow(image, extent=np.concatenate([xlim, ylim]))
#     ax.set_xlabel(r"x / Re(p_0)")
#     ax.set_ylabel(r"y / Im(p_0)")
#     fig.savefig(
#         RESULTS_DIR / f"figure_{run_id}.png",
#         bbox_inches="tight",
#         pad_inches=0,
#     )
#     plt.close(fig)

#     return is_correct
