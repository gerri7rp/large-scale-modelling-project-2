import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so utils is importable
from utils.plotting import plot_convergence, plot_slice_vs_analytical
from utils.benchmark import save_and_validate_results, validate_result


N = 300
MAX_ITER = 10000
TOLERANCE = 10e-6
INIT_VALUE = 10
PLOT = False

def get_run_parameters(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT):
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
        PLOT = bool(argv.pop(0))
    if argv:
        experiment_id = argv.pop(0)
    if argv:
        csv_path = Path(argv.pop(0))

    return N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT



def source(N):
    # Interior points of [-1, 1] with N+2 points including the boundary
    h = 2 / (N + 1)
    dim1_points = np.linspace(-1, 1, N + 2)[1:-1]
    dim1_sin = np.sin(np.pi * dim1_points)
    return h ** 2 * 3 * np.pi ** 2 * dim1_sin[:, None, None] * dim1_sin[None, :, None] * dim1_sin[None, None, :]

def jacobi_step(u, d2f, N):
    return 1 / 6  * (u[0: N,1: N + 1, 1 : N + 1] + u[2: N+2,1: N + 1, 1 : N + 1] + u[1: N + 1, 0: N, 1 : N + 1] + u[1: N + 1, 2: N + 2, 1 : N + 1] + u[1: N + 1, 1: N + 1, 0 : N] + u[1: N + 1, 1: N + 1, 2: N + 2] + d2f)

def set_bc(u):
    print(f"shape of u in set_bc = {u.shape}", flush=True)
    u[0, :, :] = 0
    u[-1, :, :] = 0
    u[:, 0, :] = 0
    u[:, -1, :] = 0
    u[:, :, 0] = 0
    u[:, :, -1] = 0

def init_matrix(N, INIT_VALUE):
    # Inits all matrix values to a certain value
    u_curr = np.full((N + 2, N + 2, N + 2), INIT_VALUE, dtype=float)

    # Sets matrix boundaries to 0
    set_bc(u_curr)
    
    return u_curr
    

def norm(u_curr, u_new):
    return np.max(np.abs(u_new - u_curr[1: -1, 1: -1, 1: -1]))

def solve(N, max_iter, tolerance, init_value):
    d2f = source(N)
    u_curr = init_matrix(N, init_value)

    convergence_hist = []

    for i in range(max_iter):
        u_new = jacobi_step(u_curr, d2f, N)
        diff = norm(u_curr, u_new)
        u_curr[1: N + 1, 1: N + 1, 1: N + 1] = u_new
        convergence_hist.append(diff)

        if diff < tolerance:
            break

    return u_curr, convergence_hist, i


if __name__ == "__main__":
    N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT = get_run_parameters(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT)
    u, convergence_hist, i = solve(N, MAX_ITER, TOLERANCE, INIT_VALUE)

    if i + 1== MAX_ITER:
        print(f"Max iterations reached: diff:{convergence_hist[-1]} vs tolerance of :{TOLERANCE} \n")    
    else:
        print(f"Reached tolerance after {i} iterations of max {MAX_ITER} \n ")  
    

    if PLOT == True:
        plot_convergence(convergence_hist, TOLERANCE, title=f"Jacobi convergence, N = {N}")
        # Slice of the solution vs the analytical solution
        pts = np.linspace(-1, 1, N + 2)[1:-1]
        plot_slice_vs_analytical(u[1:-1, 1:-1, 1:-1], pts, z=0.5, title=f"Jacobi, N = {N}, {len(convergence_hist)} iterations")


    # is_valid = validate_result(image_collector.T)
        
    # save_and_validate_results(
    #     csv_path, RUN_TYPE, RUN_ID, is_valid, n_ranks, chunk_size, size,
    #     xlim, ylim, time_compute, time_comm, time_wait
    # )
