from implementations.sendrecv_implementation import CSV_PATH
import mpi4py.MPI as MPI
import numpy as np
import sys
import csv
from sys import argv
from math import prod
from pathlib import Path
from time import perf_counter
from numba import jit
import os

from utils.benchmark import save_and_validate_results

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so utils is importable
from implementations.sequential import RUN_TYPE, get_run_parameters
from utils.datatypes import get_datatypes_all
from implementations.sequential import plot_convergence
from utils.tools import get_run_parameters
def calculate_slice_shape(N, dims, rank):
    shape = []
    for d in range(3):
        try:
            coordinate = (rank // prod(dims[d + 1 :])) % dims[d]
            base, remainder = divmod(N, dims[d])
            shape.append(base + (coordinate < remainder))
        except:
            shape.append(N)

    return tuple(shape)

def get_global_index(N, dims, rank):
    global_idx = []
    for d in range(3):
        try:
            coordinate = (rank // prod(dims[d + 1 :])) % dims[d]
            base, remainder = divmod(N, dims[d])
            start = coordinate * base + min(coordinate, remainder)
            stop = start + base + (coordinate < remainder)
            global_idx.append((start, stop))
        except:
            global_idx.append((0,N))
    return tuple(global_idx)

def calculate_source_term(global_idx, N):
    (x_start, x_stop), (y_start, y_stop), (z_start, z_stop) = global_idx

    # Map index to domain [-1,1]
    h = 2 / (N + 1)
    x_eval = -1 + h + h * np.arange(x_start, x_stop)
    y_eval = -1 + h + h * np.arange(y_start, y_stop)
    z_eval = -1 + h + h * np.arange(z_start, z_stop)

    #x_grid, y_grid, z_grid = np.meshgrid(x_eval, y_eval, z_eval)

    return h**2 * 3 * np.pi**2 * np.sin(np.pi * x_eval)[:, None, None] * np.sin(np.pi * y_eval)[None, :, None] * np.sin(np.pi * z_eval)[None, None, :]

def send(to_rank, datatype, send_buf, comm):
    if to_rank == MPI.PROC_NULL:
        pass
    else:
        comm.Send([send_buf, 1, datatype], to_rank)

def recv(from_rank, datatype, recv_buf, comm):
    if from_rank == MPI.PROC_NULL:
        pass
    else:
        comm.Recv([recv_buf, 1, datatype], from_rank)

def sendrecv_sliceedges(u_curr_slice, neighbors, dim2datatype, dims, comm):
    rank = comm.Get_rank()
    coords = comm.Get_coords(rank)
    # Send recv for each dimension
    for i in range(len(dims)):
        # Skip if size of dim is 1
        if dims[i] == 1:
            continue

        # Handle from/to ranks
        left_rank, right_rank = neighbors[i]
        l_send, r_send, l_recv, r_recv = dim2datatype[i]

        # Get the position of this rank in the dimension currently being exchanged.
        coordinate = coords[i]

        # Send and recieve data
        # even
        if coordinate % 2 == 0:
            send(right_rank, r_send, u_curr_slice, comm)
            send(left_rank ,l_send, u_curr_slice, comm)
            recv(right_rank, r_recv, u_curr_slice, comm)
            recv(left_rank, l_recv, u_curr_slice, comm)

        # odd
        else:
            recv(left_rank, l_recv, u_curr_slice, comm)
            recv(right_rank, r_recv, u_curr_slice, comm)
            send(left_rank, l_send, u_curr_slice, comm)
            send(right_rank, r_send, u_curr_slice, comm)


@jit
def jacobi_iteration(u, u_new, source):
    # Get dimensions
    Nx,Ny,Nz = u.shape
    max_val = 0

    for x in range(1, Nx - 1):  # Optimal for-loop arangement to minimize chache misses
        for y in range(1, Ny - 1):
            for z in range(1, Nz - 1):
                # Calculate new value
                new_val = 1/6 * (
                    u[x-1,y,z] + u[x+1,y,z] +
                    u[x,y-1,z] + u[x,y+1,z] +
                    u[x,y,z-1] + u[x,y,z+1] +
                    source[x-1,y-1,z-1])
                u_new[x,y,z] = new_val
                # max_val = max(max_val,abs(new_val - u[x,y,z]))
    
    return max_val


def main(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT, DECOMB_DIM,
         DECOMB_AXES, solution_path):
    # Setup communication
    comm = MPI.COMM_WORLD.Clone()
    size = comm.size
    rank = comm.Get_rank()

    # Create cartisian map
    dims = [1, 1, 1]
    for axis in DECOMB_AXES:
        dims[axis] = 0
    dims = MPI.Compute_dims(size, dims)
    cart_comm = comm.Create_cart(dims, periods=[False, False, False])
    neighbors = [cart_comm.Shift(axis, 1) for axis in range(3)]
    #print(f"Cartisian map dim: {dims}")

    # Calculate domain decomposition
    slice_shape = calculate_slice_shape(N, dims, rank)
    global_idx = get_global_index(N, dims, rank)
    #print(f"Rank {rank} has global index {global_idx}")

    # Initial array
    u_curr = np.full([d+2 for d in slice_shape], INIT_VALUE, dtype=float)
    u_curr[[0,-1], :, :] = 0    # Init ghost_cells as zero
    u_curr[:, [0,-1], :] = 0
    u_curr[:, :, [0,-1]] = 0
    
    # Set up data types
    dtypes = get_datatypes_all(u_curr)
    for dtype in dtypes:
        dtype.Commit()

    # Create dimension 2 datatype table 
    (left_send, right_send, front_send, back_send, buttom_send, top_send,
                  left_recv, right_recv, front_recv, back_recv, buttom_recv, top_recv) = dtypes
    dim2datatype = {
        0: (left_send,right_send,left_recv,right_recv),
        1: (front_send, back_send,front_recv,back_recv),
        2: (buttom_send, top_send,buttom_recv,top_recv)
    }


    # calculate source term
    source = calculate_source_term(global_idx, N)

    convergence_hist = []
    u_new = np.zeros_like(u_curr)

    t0 = perf_counter()
    comm_time = 0
    for i in range(MAX_ITER):
        t0_comm = perf_counter()
        sendrecv_sliceedges(u_curr, neighbors, dim2datatype, dims, cart_comm)
        cart_comm.Barrier()
        comm_time += perf_counter() - t0_comm

        # Do a Jacobi iteration
        diff = jacobi_iteration(u_curr, u_new, source)
        # t0_comm = perf_counter()
        # max_diff = cart_comm.allreduce(diff, op = MPI.MAX)
        # comm_time += perf_counter() - t0_comm
        #print(f"Max diff: {max_diff}")
        u_curr, u_new = u_new, u_curr

        # # Convergence and stuff
        # convergence_hist.append(max_diff)
        # if max_diff < TOLERANCE:
        #     break
    time = perf_counter() - t0

    # Free datatypes
    for dtype in dtypes:
        dtype.Free()

    # Write to file
    # write_results_to_file(cart_comm, u_curr, global_idx, N, solution_path)

    # Calculate flops and lops
    Nx, Ny, Nz = u_curr.shape
    local_points = (Nx - 2) * (Ny - 2) * (Nz - 2)
    n_iter = i + 1
    flops = (7 * local_points * n_iter) / time
    lops = (local_points * n_iter) / time

    # Plot 
    if (PLOT == True) and (rank == 0):
        print(f"Total time: {time}")
        plot_convergence(convergence_hist, TOLERANCE, title=f"Jacobi convergence, N = {N}")
        
    return rank, size, time, n_iter, flops, lops, comm_time
    

if __name__ == "__main__":
    N = 100
    MAX_ITER = 10000
    TOLERANCE = 10e-5
    INIT_VALUE = 10
    PLOT = True
    DECOMB_DIM = 3
    DECOMB_AXES = [0,1,2]
    solution_path = "results/solutions/my.bin"
    CSV_PATH = None

    argv = sys.argv[1:]
    try:
        N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT, CSV_PATH, SOLUTION_PATH, DECOMB_DIM, DECOMB_AXES = get_run_parameters(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT)
    except:
        print("Error parsing command line arguments. Using default values.")

    RUN_ID = os.environ.get("RUN_ID", "manual-run")
    if MPI.COMM_WORLD.rank == 0:
        solution_path.parent.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    rank, n_ranks, time, n_iterations, flops, lops, comm_time = main(
        N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT,
        DECOMB_DIM, DECOMB_AXES, solution_path,
    )

    comm = MPI.COMM_WORLD
    total_time_array = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None
    all_niter = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None
    flops_array = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None
    lops_array = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None
    comm_time_array = np.zeros(n_ranks, dtype=np.float64) if rank == 0 else None

    comm.Gather(np.array([time], dtype=np.float64), total_time_array, root=0)
    comm.Gather(np.array([n_iterations], dtype=np.float64), all_niter, root=0)
    comm.Gather(np.array([flops], dtype=np.float64), flops_array, root=0)
    comm.Gather(np.array([lops], dtype=np.float64), lops_array, root=0)
    comm.Gather(np.array([comm_time], dtype=np.float64), comm_time_array, root=0)

    jacobi_time_array = total_time_array - comm_time_array
    is_valid =  save_and_validate_results(
        csv_path = CSV_PATH, 
        run_type = f"{RUN_TYPE}_DIM{DECOMB_DIM}_AXES{'-'.join(map(str, DECOMB_AXES))}", 
        run_id = RUN_ID, 
        n_ranks = n_ranks, 
        chunk_size = 0, 
        N = N, 
        max_iter = MAX_ITER, 
        n_iterations = n_iterations, 
        tolerance = TOLERANCE, 
        init_value = INIT_VALUE, 
        jacobi_time = jacobi_time_array,
        comm_time = comm_time_array,
        flops = flops_array,
        lops = lops_array
        )
    