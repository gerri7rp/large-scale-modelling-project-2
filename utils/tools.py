import sys
from pathlib import Path

def get_run_parameters(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT, csv_path=None):
    argv = sys.argv[1:]
    if argv:
        N = int(argv.pop(0))
    if argv:
        MAX_ITER = int(argv.pop(0))
    if argv:
        TOLERANCE = float(argv.pop(0))
    if argv:
        INIT_VALUE = int(argv.pop(0))
    if argv:
        PLOT = argv.pop(0).lower() in ("true", "1", "yes")  # bool("False") would be True
    if argv:
        csv_path = Path(argv.pop(0))
    if argv:
        solution_path = Path(argv.pop(0))
    if argv:
        DECOMB_DIM = int(argv.pop(0))
    if argv:
        DECOMB_AXES = [int(axis) for axis in argv.pop(0).split(",")]

    return N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT, csv_path, solution_path, DECOMB_DIM, DECOMB_AXES
