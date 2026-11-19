import mpi4py.MPI as MPI
import numpy as np
import sys
from math import prod
from pathlib import Path
from time import perf_counter
from numba import jit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so utils is importable
from implementations.sequential import get_run_parameters
from utils.datatypes import get_datatypes_all
from implementations.sequential import plot_convergence

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

def sendrecv_sliceedges(u_curr_slice, neighbors, dim2datatype, dims, comm):

    # Send recv for each dimension
    for i in range(len(dims)):
        # Skip if size of dim is 1
        if dims[i] == 1:
            continue

        # Handle from/to ranks
        left_rank, right_rank = neighbors[i]
        l_send, r_send, l_recv, r_recv = dim2datatype[i]
        
        # Create recv buffers
        ghost_left = [slice(None)] * u_curr_slice.ndim
        ghost_right = [slice(None)] * u_curr_slice.ndim
        ghost_left[i] = 0
        ghost_right[i] = -1
        ghost_left = tuple(ghost_left)
        ghost_right = tuple(ghost_right)

        # Send and recieve data
        comm.Sendrecv(
            [u_curr_slice, 1, r_send], dest=right_rank,
            recvbuf=[u_curr_slice, 1, l_recv], source=left_rank
        )
        comm.Sendrecv(
            [u_curr_slice, 1, l_send], dest=left_rank,
            recvbuf=[u_curr_slice, 1, r_recv], source=right_rank
        )

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
                max_val = max(max_val,abs(new_val - u[x,y,z]))
    
    return max_val

def write_results_to_file(comm, u_curr, global_idx, N, solution_path):
    local_data = np.ascontiguousarray(u_curr[1:-1, 1:-1, 1:-1])
    starts = tuple(start for start, stop in global_idx)

    filetype = MPI.DOUBLE.Create_subarray(
        sizes=(N, N, N),
        subsizes=local_data.shape,
        starts=starts,
        order=MPI.ORDER_C,
    )
    filetype.Commit()

    fh = MPI.File.Open(
        comm,
        str(solution_path),
        MPI.MODE_CREATE | MPI.MODE_WRONLY,
    )
    try:
        fh.Set_size(0)
        fh.Set_view(0, MPI.DOUBLE, filetype, "native")
        fh.Write_all(local_data)
    finally:
        fh.Close()
        filetype.Free()

def main(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT, DECOMB_DIM):
    # Setup communication
    comm = MPI.COMM_WORLD.Clone()
    size = comm.size
    rank = comm.Get_rank()

    # Create cartisian map
    dims = MPI.Compute_dims(size, DECOMB_DIM)
    cart_comm = comm.Create_cart(dims, periods=[False] * DECOMB_DIM)
    neighbors = [cart_comm.Shift(axis, 1) for axis in range(DECOMB_DIM)]
    #print(f"Cartisian map dim: {dims}")

    # Calculate domain decomposition
    slice_shape = calculate_slice_shape(N, dims, rank)
    global_idx = get_global_index(N, dims, rank)
    #print(f"Rank {rank} has global index {global_idx}")

    # Initial array
    u_curr = np.full([d+2 for d in slice_shape], INIT_VALUE, dtype=float) # No need to enforce boundary condtion here as the ghost cells get overwritten later
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
    for i in range(MAX_ITER):
        sendrecv_sliceedges(u_curr, neighbors, dim2datatype, dims, cart_comm)
        cart_comm.Barrier()

        # Do a Jacobi iteration
        diff = jacobi_iteration(u_curr, u_new, source)
        max_diff = cart_comm.allreduce(diff, op = MPI.MAX)
        #print(f"Max diff: {max_diff}")
        u_curr, u_new = u_new, u_curr

        # Convergence and stuff
        convergence_hist.append(max_diff)
        if max_diff < TOLERANCE:
            break
    time = perf_counter() - t0

    # Free datatypes
    for dtype in dtypes:
        dtype.Free()

    # Write to file
    write_results_to_file(cart_comm, u_curr, global_idx, N, solution_path)

    if (PLOT == True) and (rank == 0):
        print(f"Total time: {time}")
        plot_convergence(convergence_hist, TOLERANCE, title=f"Jacobi convergence, N = {N}")
        



    
    

if __name__ == "__main__":
    N = 200
    MAX_ITER = 10000
    TOLERANCE = 10e-5
    INIT_VALUE = 10
    PLOT = True
    DECOMB_DIM = 1
    solution_path = "results/solutions/my.bin"

    # Get parameters
    # N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT, csv_path, solution_path = get_run_parameters(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT)
    main(N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT, DECOMB_DIM)
    