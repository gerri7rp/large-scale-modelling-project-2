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

N = 300
MAX_ITER = 10000
TOLERANCE = 10e-6
INIT_VALUE = 10
PLOT = False

############# SET UP COMM+DATA+ETC ####################################################
#######################################################################################
# Set up comm 
comm = MPI.COMM_WORLD.Clone() # always clone :)
size = comm.size # total slices/ranks
rank = comm.Get_rank()

# Set up data
N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT = get_run_parameters(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT)
if N%size == 0:
    N_slice = N//size
else:
    N_slice_distributions = np.full(size, N // size) # makes an array with the "min" number of N_slice
    N_slice_distributions[:(N % size)] += 1 # adds the missing N_slice numbers
    N_slice = N_slice_distributions[rank] # allocates the correcte slice size to rank

u_initial_slice = init_slice(N_slice, N, INIT_VALUE)

# Set up data type
LHS_dt, RHS_dt = get_datatypes(u_initial_slice) # datatype taking the LHS and RHS of the slice
LHS_dt.Commit()
RHS_dt.Commit()

############# JACOBI COMPUTATION #########################################################
#######################################################################################

u_curr, convergence_hist, i = solve_slice(N=N, N_slice=N_slice, max_iter=MAX_ITER, tolerance=TOLERANCE, init_value=INIT_VALUE, rank=rank, size=size, u_curr=u_initial_slice, LHS_dt=LHS_dt, RHS_dt=RHS_dt, comm=comm)

LHS_dt.Free()
RHS_dt.Free()