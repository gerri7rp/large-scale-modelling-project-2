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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so utils is importable
from implementations.sequential import get_run_parameters
from utils.datatypes import get_datatypes_all
from implementations.sequential import plot_convergence
from utils.benchmark import save_and_validate_results, validate_result


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
        # ghost_left = [slice(None)] * u_curr_slice.ndim
        # ghost_right = [slice(None)] * u_curr_slice.ndim
        # ghost_left[i] = 0
        # ghost_right[i] = -1
        # ghost_left = tuple(ghost_left)
        # ghost_right = tuple(ghost_right)

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

def append_results(csv_path, run_id, N, size, decomb_dim, decomb_axes,
                   tolerance, max_iter, init_value, times, n_iter, flops, lops, comm_times):
    if csv_path is None:
        csv_path = Path("results/raw/decomb_implementation") / run_id

    csv_path = Path(csv_path)
    if csv_path.suffix != ".csv":
        csv_path = csv_path / (
            f"np{size}_N{N}_dim{decomb_dim}_axes{'-'.join(map(str, decomb_axes))}"
            f"_iter{max_iter}_tol{tolerance}_init{init_value}.csv"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id", "N", "size", "decomb_dim", "decomb_axes",
        "tolerance", "max_iter", "init_value",
        "rank", "time","n_iter", "flops", "lops","comm_time"
    ]

    with csv_path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(columns)

        axes_text = str(list(decomb_axes))
        for rank in range(size):
            writer.writerow([
                run_id,
                int(N),
                int(size),
                int(decomb_dim),
                axes_text,
                float(tolerance),
                int(max_iter),
                int(init_value),
                int(rank),
                float(times[rank]),
                int(n_iter[rank]),
                float(flops[rank]),
                float(lops[rank]),
                float(comm_times[rank])
            ])

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
        t0_comm = perf_counter()
        max_diff = cart_comm.allreduce(diff, op = MPI.MAX)
        comm_time += perf_counter() - t0_comm
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
    # write_results_to_file(cart_comm, u_curr, global_idx, N, solution_path)

    # Calculate flops and lups
    Nx, Ny, Nz = u_curr.shape
    local_points = (Nx - 2) * (Ny - 2) * (Nz - 2)
    n_iter = i + 1
    flops = (7 * local_points * n_iter) / time
    lups = (local_points * n_iter) / time

    # Plot 
    if (PLOT == True) and (rank == 0):
        print(f"Total time: {time}")
        plot_convergence(convergence_hist, TOLERANCE, title=f"Jacobi convergence, N = {N}")
        
    return rank, size, time, n_iter, flops, lups, comm_time




    
    

if __name__ == "__main__":
    N = 100
    MAX_ITER = 10000
    TOLERANCE = 10e-5
    INIT_VALUE = 0
    PLOT = True
    DECOMB_DIM = 3
    DECOMB_AXES = [0,1,2]
    solution_path = "results/solutions/my.bin"
    csv_path = None

    argv = sys.argv[1:]
    if len(argv) == 9:
        N = int(argv[0])
        MAX_ITER = int(argv[1])
        TOLERANCE = float(argv[2])
        INIT_VALUE = int(argv[3])
        PLOT = argv[4].lower() == "true"
        solution_path = Path(argv[5])
        csv_path = Path(argv[6])
        DECOMB_DIM = int(argv[7])
        DECOMB_AXES = [int(axis) for axis in argv[8].split(",")]


    RUN_ID = os.environ.get("RUN_ID", "manual-run")
    if MPI.COMM_WORLD.rank == 0:
        solution_path.parent.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()
    rank, size, time, n_iter, flops, lops, comm_time = main(
        N, MAX_ITER, TOLERANCE, INIT_VALUE, PLOT,
        DECOMB_DIM, DECOMB_AXES, solution_path,
    )

    comm = MPI.COMM_WORLD
    times = np.zeros(size, dtype=np.float64) if rank == 0 else None
    all_niter = np.zeros(size, dtype=np.float64) if rank == 0 else None
    all_flops = np.zeros(size, dtype=np.float64) if rank == 0 else None
    all_lops = np.zeros(size, dtype=np.float64) if rank == 0 else None
    all_comm_times = np.zeros(size, dtype=np.float64) if rank == 0 else None

    comm.Gather(np.array([time], dtype=np.float64), times, root=0)
    comm.Gather(np.array([n_iter], dtype=np.float64), all_niter, root=0)
    comm.Gather(np.array([flops], dtype=np.float64), all_flops, root=0)
    comm.Gather(np.array([lops], dtype=np.float64), all_lops, root=0)
    comm.Gather(np.array([comm_time], dtype=np.float64), all_comm_times, root=0)

    if rank == 0:
        append_results(
            csv_path,
            RUN_ID,
            N,
            size,
            DECOMB_DIM,
            DECOMB_AXES,
            TOLERANCE,
            MAX_ITER,
            INIT_VALUE,
            times,
            all_niter,
            all_flops,
            all_lops,
            all_comm_times
        )
    