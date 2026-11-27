import sys
from time import time, perf_counter
import numpy as np
import os
from mpi4py import MPI
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so utils is importable
from utils.datatypes import get_datatypes
from utils.sendrecv_sliceedges import sendrecv_sliceedges, init_slice, solve_slice
from utils.plotting import plot_convergence, plot_slice_vs_analytical
from utils.benchmark import save_and_validate_results
from utils.tools import get_run_parameters




def slice_the_box(N, size, rank):
    if N%size == 0:
        N_slice = N//size
        N_slice_distributions = np.full(size, N // size)
    else:
        N_slice_distributions = np.full(size, N // size) # makes an array with the "min" number of N_slice
        N_slice_distributions[:(N % size)] += 1 # adds the missing N_slice numbers
        N_slice = N_slice_distributions[rank] # allocates the correcte slice size to rank
    return N_slice, N_slice_distributions


def handle_plots(convergence_hist, tolerance, max_iter, n_ranks, file, N):
    n_iter = len(convergence_hist)
    
    if n_iter == max_iter and convergence_hist[-1] >= tolerance:
        print(f"Max iterations reached: diff:{convergence_hist[-1]} vs tolerance of :{tolerance} \n")
    else:
        print(f"Reached tolerance after {n_iter} iterations of max {max_iter} \n")

    u = np.fromfile(file, dtype=np.float64).reshape(N, N, N)

    plot_convergence(convergence_hist, tolerance, title=f"parallel/Jacobi sendrecv convergence, N = {N}, size = {n_ranks}")
    pts = np.linspace(-1, 1, N + 2)[1:-1]
    plot_slice_vs_analytical(u, pts, z=0.5, title=f"parallel/Jacobi sendrecv, N = {N}, size = {n_ranks}, {n_iter} iterations")


def write_results_to_file(comm, u_curr, SOLUTION_PATH):
    u_write = np.ascontiguousarray(u_curr[1:-1, 1:-1, 1:-1])

    amode = MPI.MODE_CREATE + MPI.MODE_WRONLY

    fh = MPI.File.Open(comm, SOLUTION_PATH, amode = amode)
    fh.Write_ordered(u_write)
    fh.Close()


def main(N, MAX_ITER, TOLERANCE, INIT_VALUE):
    ### 1. INITIALIZATION
    initialization_time = perf_counter()

    # Set up comm 
    comm = MPI.COMM_WORLD.Clone() # always clone :)
    size = comm.size
    rank = comm.Get_rank()

    # Create cartisian map
    dims = MPI.Compute_dims(size, 1)
    cart_comm = comm.Create_cart(dims, periods=[False])

    # Calculate box indexes per rank
    N_slice, N_slice_distributions = slice_the_box(N, size, rank)
    slice_idx = np.array([N_slice_distributions[:rank].sum(), N_slice_distributions[:rank + 1].sum()])

    # Init box per rank
    u_initial_slice = init_slice(N_slice, N, INIT_VALUE, rank, size)

    # Set up data type
    LHS_dt, RHS_dt = get_datatypes(u_initial_slice) # datatype taking the LHS and RHS of the slice #should take 2 and second to last slice instead? Shouldnt send ghost slice
    LHS_dt.Commit()
    RHS_dt.Commit()



    ### 2. JACOBI COMPUTATION 
    jacobi_time = perf_counter()

    u_curr, convergence_hist, i, flops, lops, t_comm = solve_slice(N=N, N_slice=N_slice, slice_idx = slice_idx, max_iter=MAX_ITER, tolerance=TOLERANCE, init_value=INIT_VALUE, rank=rank, size=size, u_curr=u_initial_slice, LHS_dt=LHS_dt, RHS_dt=RHS_dt, comm=cart_comm)
    LHS_dt.Free()
    RHS_dt.Free()

    jacobi_time = np.array([perf_counter() - jacobi_time])

 
    # ### 3. WRITING RESULTS TO FILE
    # fileWriting_time = perf_counter()

    # write_results_to_file(comm, u_curr, SOLUTION_PATH)
    # comm.Barrier()

    # fileWriting_time = np.array([perf_counter() - fileWriting_time])


    return rank, comm, size, u_curr, i, flops, lops, convergence_hist, jacobi_time, t_comm



if __name__ == "__main__":
    # Parameters by default
    N = 100
    MAX_ITER = 10000
    TOLERANCE = 10e-5
    INIT_VALUE = 10
    PLOT = True

    # Get run parameters from the command line
    N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT, CSV_PATH, DECOMB_DIM, DECOMB_AXES = get_run_parameters()

    # Main
    t0 = perf_counter()
    rank, comm, n_ranks, u, n_iterations, flops, lops, convergence_hist, total_time, comm_time = main(N, MAX_ITER, TOLERANCE, INIT_VALUE)



    # Rank 0 gathers data from all ranks
    total_time_array = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None
    all_niter = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None
    flops_array = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None
    lops_array = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None
    comm_time_array = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None

    comm.Gather(np.array([total_time], dtype=np.float64), total_time_array, root=0)
    comm.Gather(np.array([n_iterations], dtype=np.float64), all_niter, root=0)
    comm.Gather(np.array([flops], dtype=np.float64), flops_array, root=0)
    comm.Gather(np.array([lops], dtype=np.float64), lops_array, root=0)
    comm.Gather(np.array([comm_time], dtype=np.float64), comm_time_array, root=0)
    

    # Rank 0 saves all the data in a csv
    if rank == 0:
        jacobi_time_array = total_time_array - comm_time_array
        RUN_TYPE = Path(__file__).stem
        RUN_ID   = os.environ.get("RUN_ID", "manual-run")

        #if PLOT:
        #    handle_plots(convergence_hist, TOLERANCE, MAX_ITER, n_ranks, N)

        is_valid =  save_and_validate_results( 
            csv_path = CSV_PATH, 
            run_type = RUN_TYPE, 
            run_id = RUN_ID, 
            n_ranks = n_ranks, 
            N = N, 
            max_iter = MAX_ITER, 
            n_iterations = n_iterations, 
            tolerance = TOLERANCE, 
            init_value = INIT_VALUE, 
            jacobi_time = jacobi_time_array,
            comm_time = comm_time_array,
            flops = flops_array,
            lups = lops_array
        )
        
