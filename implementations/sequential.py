import sys
from pathlib import Path

import numpy as np
import os
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so utils is importable
from utils.plotting import plot_convergence, plot_slice_vs_analytical
from utils.benchmark import save_and_validate_results, validate_result
from utils.tools import get_run_parameters

N = 300
MAX_ITER = 10000
TOLERANCE = 10e-6
INIT_VALUE = 10
PLOT = False

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
        # diff = norm(u_curr, u_new)
        u_curr[1: N + 1, 1: N + 1, 1: N + 1] = u_new
        # convergence_hist.append(diff)

        # if diff < tolerance:
            # break

    return u_curr, convergence_hist, i


if __name__ == "__main__":
    N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT, csv_path, solution_path = get_run_parameters(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT)
    total_time_array = time.time()
    u, convergence_hist, i = solve(N, MAX_ITER, TOLERANCE, INIT_VALUE)
    total_time_array = np.array([time.time() - total_time_array])
    # if i + 1== MAX_ITER:
    #     print(f"Max iterations reached: diff:{convergence_hist[-1]} vs tolerance of :{TOLERANCE} \n")    
    # else:
    #     print(f"Reached tolerance after {i} iterations of max {MAX_ITER} \n ")  
    

    if PLOT == True:
        plot_convergence(convergence_hist, TOLERANCE, title=f"Jacobi convergence, N = {N}")
        # Slice of the solution vs the analytical solution
        pts = np.linspace(-1, 1, N + 2)[1:-1]
        plot_slice_vs_analytical(u[1:-1, 1:-1, 1:-1], pts, z=0.5, title=f"Jacobi, N = {N}, {len(convergence_hist)} iterations")



    RUN_TYPE = Path(__file__).stem
    RUN_ID   = os.environ.get("RUN_ID", "manual-run")
    n_ranks = 1
    chunk_size = 0
    
    _, _ =  save_and_validate_results(
        matrix=u, csv_path=csv_path, run_type=RUN_TYPE, run_id=RUN_ID, n_ranks=n_ranks, 
        chunk_size=chunk_size, N=N, max_iter=MAX_ITER, tolerance=TOLERANCE, init_value=INIT_VALUE, time_compute=[0], time_comm=[0], time_wait=[0]
    )

    # if not is_valid:
    #     print("\n\nTHE RESULTS OF THE RUN ARE NOT CORRECT!!\n\n")
