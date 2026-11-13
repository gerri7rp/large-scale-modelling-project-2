import sys
from time import time
import numpy as np
from mpi4py import MPI
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so utils is importable
from utils.datatypes import get_datatypes
from utils.sendrecv_sliceedges import sendrecv_sliceedges, init_slice, solve_slice
from utils.plotting import plot_convergence, plot_slice_vs_analytical
from utils.benchmark import save_and_validate_results, validate_result
from implementations.sequential import get_run_parameters




def slice_the_box(N, size, rank):
    if N%size == 0:
        N_slice = N//size
        N_slice_distributions = np.full(size, N // size)
    else:
        N_slice_distributions = np.full(size, N // size) # makes an array with the "min" number of N_slice
        N_slice_distributions[:(N % size)] += 1 # adds the missing N_slice numbers
        N_slice = N_slice_distributions[rank] # allocates the correcte slice size to rank
    return N_slice, N_slice_distributions

def main(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT):
    ############# SET UP COMM+DATA+ETC ####################################################
    #######################################################################################

    # Set up comm 
    comm = MPI.COMM_WORLD.Clone() # always clone :)
    size = comm.size # total slices/ranks
    rank = comm.Get_rank()

    N_slice, N_slice_distributions = slice_the_box(N, size, rank)
    slice_idx = np.array([N_slice_distributions[:rank].sum(), N_slice_distributions[:rank + 1].sum()])

    u_initial_slice = init_slice(N_slice, N, INIT_VALUE, rank, size)

    # Set up data type
    LHS_dt, RHS_dt = get_datatypes(u_initial_slice) # datatype taking the LHS and RHS of the slice #should take 2 and second to last slice instead? Shouldnt send ghost slice

    LHS_dt.Commit()
    RHS_dt.Commit()

    ############# JACOBI COMPUTATION #########################################################
    #######################################################################################

    u_curr, convergence_hist, i = solve_slice(N=N, N_slice=N_slice, slice_idx = slice_idx, max_iter=MAX_ITER, tolerance=TOLERANCE, init_value=INIT_VALUE, rank=rank, size=size, u_curr=u_initial_slice, LHS_dt=LHS_dt, RHS_dt=RHS_dt, comm=comm)

    LHS_dt.Free()
    RHS_dt.Free()

    #write to file
    amode = MPI.MODE_CREATE + MPI.MODE_WRONLY
    out_dir = Path("results/solutions/parallel")
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier() # folder exists before anyone opens the file
    file = str(out_dir / f"N={N},size={size},IV={INIT_VALUE}.bin")
    fh = MPI.File.Open(comm, file, amode = amode)

    fh.Set_size(0) # del old content
    comm.Barrier()

    u_write = np.ascontiguousarray(u_curr[1:-1, 1:-1, 1:-1])
    offset = 0 # offset used to seek in the file
    for r in range(comm.size):
        if r != comm.rank:
            offset = comm.bcast(offset, root=r)
            continue

        fh.Seek(offset)
        fh.Write(u_write)
        comm.bcast(fh.Get_position(), root=r)
    fh.Close()



    comm.Barrier()

    if rank == 0:
        n_iter = len(convergence_hist)
        if n_iter == MAX_ITER and convergence_hist[-1] >= TOLERANCE:
            print(f"Max iterations reached: diff:{convergence_hist[-1]} vs tolerance of :{TOLERANCE} \n")
        else:
            print(f"Reached tolerance after {n_iter} iterations of max {MAX_ITER} \n")

        u = np.fromfile(file, dtype=np.float64).reshape(N, N, N)

        if PLOT:
            plot_convergence(convergence_hist, TOLERANCE, title=f"parallel/Jacobi sendrecv convergence, N = {N}, size = {size}")
            pts = np.linspace(-1, 1, N + 2)[1:-1]
            plot_slice_vs_analytical(u, pts, z=0.5, title=f"parallel/Jacobi sendrecv, N = {N}, size = {size}, {n_iter} iterations")

if __name__ == "__main__":
    N = 300
    MAX_ITER = 10000
    TOLERANCE = 10e-5
    INIT_VALUE = 10
    PLOT = True

    # Set up data
    N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT = get_run_parameters(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT)
    main(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT)

