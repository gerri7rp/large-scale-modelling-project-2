import sys
from time import time, perf_counter
import numpy as np
from mpi4py import MPI
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so utils is importable
from implementations.sequential import norm

def source(N, N_slice, slice_idx, rank, size):
    # Interior points of [-1, 1] with N+2 points including the boundary

    h = 2 / (N + 1)
    start, stop = slice_idx
    x = -1 + h * (np.arange(start, stop) + 1) 
    sin_x = np.sin(np.pi * x)
    
    yz = np.linspace(-1, 1, N + 2)[1: -1]
    sin_yz = np.sin(np.pi * yz)
    return h ** 2 * 3 * np.pi ** 2 * sin_x[:, None, None] * sin_yz[None, :, None] * sin_yz[None, None, :]

def send(to_rank, datatype, send_buf, comm):
    if to_rank == MPI.PROC_NULL:
        pass
    else:
        comm.Send([send_buf, 1, datatype], to_rank)

def recv(from_rank, recv_buf, comm):
    if from_rank == MPI.PROC_NULL:
        pass
    else:
        comm.Recv(recv_buf, from_rank)

def sendrecv_sliceedges(rank, size, u_curr_slice, LHS_dt, RHS_dt, comm):
    # Create recv buffer
    LHS_recv_buff, RHS_recv_buff = np.zeros_like(u_curr_slice[0,:,:]).astype(np.float64), np.zeros_like(u_curr_slice[-1,:,:]).astype(np.float64) # we need one for each side to save the data for the computation
    left_rank, right_rank = comm.Shift(0,1)

    # even ranks
    if rank%2 == 0:
        send(right_rank, RHS_dt, u_curr_slice, comm)
        send(left_rank, LHS_dt, u_curr_slice, comm)
        recv(right_rank, RHS_recv_buff, comm)
        recv(left_rank, LHS_recv_buff, comm)

    # odd ranks
    else:
        recv(left_rank, LHS_recv_buff, comm)
        recv(right_rank, RHS_recv_buff, comm)
        send(left_rank, LHS_dt, u_curr_slice, comm)
        send(right_rank, RHS_dt, u_curr_slice, comm)

    return LHS_recv_buff, RHS_recv_buff

def init_slice(N_slice, N, INIT_VALUE, rank, size):
    # Inits all matrix values to a certain value
    u_curr = np.full((N_slice + 2, N + 2, N + 2), INIT_VALUE, dtype=float)

    # Sets matrix boundaries to 0
    set_bc(u_curr, rank, size)
    
    return u_curr

def set_bc(u, rank, size):
    u[:, 0, :] = 0
    u[:, -1, :] = 0
    u[:, :, 0] = 0
    u[:, :, -1] = 0

    if rank == 0:
        u[0, :, :] = 0
    if rank == size-1:
        u[-1, :, :] = 0


def jacobi_step_slice(u, d2f, N, N_slice):
    return 1 / 6  * (u[0: N_slice, 1: N + 1, 1 : N + 1] + 
                     u[2: N_slice + 2, 1: N + 1, 1 : N + 1] + 
                     u[1: N_slice + 1, 0: N, 1 : N + 1] + 
                     u[1: N_slice + 1, 2: N + 2, 1 : N + 1] + 
                     u[1: N_slice + 1, 1: N + 1, 0 : N] + 
                     u[1: N_slice + 1, 1: N + 1, 2: N + 2] + d2f)


def solve_slice(N, N_slice, slice_idx, max_iter, tolerance, init_value, rank, size, u_curr, LHS_dt, RHS_dt, comm):
    # NOT SURE THIS WORKS
    #### Need to only compute the source for the releavant interval
    d2f = source(N, N_slice, slice_idx, rank, size)

    convergence_hist = []

    t0 = perf_counter() # Time
    for i in range(max_iter):
        # Get the edges
        u_curr[0,:,:], u_curr[-1,:,:] = sendrecv_sliceedges(rank=rank, size=size, u_curr_slice=u_curr, LHS_dt=LHS_dt, RHS_dt=RHS_dt, comm=comm)
        comm.Barrier() # Don't know if this is needed, but to ensure that all edges has been recved before computing new values
        # Do jacobi
        u_new = jacobi_step_slice(u_curr, d2f, N, N_slice)


        if i % 10 == 0:
            local_diff = norm(u_curr, u_new)
            diff = comm.allreduce(local_diff, op=MPI.MAX) # to ensure entire cube, and not just current slice, has converges
            convergence_hist.append(diff)
            if diff < tolerance:
                u_curr[1:-1, 1:-1, 1:-1] = u_new
                break
        
        u_curr[1:-1, 1:-1, 1:-1] = u_new
        
    time = perf_counter() - t0
    n_iter = i + 1  # number of iterations
    M = np.prod(u_new.shape)
    n_flops_norm = M # this doesnt count abs as a flop
    n_flops_jacobi = 7
    flops = ((n_flops_jacobi * M) * n_iter + n_flops_norm * len(convergence_hist)) / time
    lups = (M * n_iter) / time

    return u_curr, convergence_hist, i, flops, lups


    