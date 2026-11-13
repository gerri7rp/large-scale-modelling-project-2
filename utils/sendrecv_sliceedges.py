import sys
from time import time
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


def sendrecv_sliceedges(rank, size, u_curr_slice, LHS_dt, RHS_dt, comm):
    # Create recv buffer
    LHS_recv_buff, RHS_recv_buff = np.zeros_like(u_curr_slice[0,:,:]).astype(np.float64), np.zeros_like(u_curr_slice[-1,:,:]).astype(np.float64) # we need one for each side to save the data for the computation

    # Start with even ranks
    if rank%2 == 0: 


        # For the rightmost slice
        if rank == size-1:
            # Send to the right (no rank to send to)

            # Send to the left
            comm.Send([u_curr_slice, 1, LHS_dt], rank-1)

            # Recv from the right (No rank to the right, so nothing to recieve. Border cond = 0, which recv_buff already is)

            # Recv from the left
            comm.Recv(LHS_recv_buff, rank-1) 
        # For the leftmost slice
        elif rank == 0: 
            # Send to the right
            comm.Send([u_curr_slice, 1, RHS_dt], rank+1)

            # Send to the left(no rank to send to)

            # Recv from the right
            comm.Recv(RHS_recv_buff, rank+1) 

            # Recv from the left (No rank to the left, so nothing to recieve. Border cond = 0, which recv_buff already is)
        # For the rest
        else: 
            # Send to the right 
            comm.Send([u_curr_slice, 1, RHS_dt], rank+1)

            # Send to the left
            comm.Send([u_curr_slice, 1, LHS_dt], rank-1)

            # Recv from the right
            comm.Recv(RHS_recv_buff, rank+1) 

            # Recv from the left 
            comm.Recv(LHS_recv_buff, rank-1) 
    else:
        # For the rightmost slice
        if rank == size-1:

            # Recv from the left
            comm.Recv(LHS_recv_buff, rank-1) 

            # Recv from the right

            # Send to the left
            comm.Send([u_curr_slice, 1, LHS_dt], rank-1)

            # Send to the right 
        # For the leftmost slice
        elif rank == 0: 
            # Recv from the left

            # Recv from the right
            comm.Recv(RHS_recv_buff, rank+1) 

            # Send to the left

            # Send to the right 
            comm.Send([u_curr_slice, 1, RHS_dt], rank+1)
        # For the rest
        else: 
            # Recv from the left
            comm.Recv(LHS_recv_buff, rank-1) 

            # Recv from the right
            comm.Recv(RHS_recv_buff, rank+1) 

            # Send to the left
            comm.Send([u_curr_slice, 1, LHS_dt], rank-1)

            # Send to the right 
            comm.Send([u_curr_slice, 1, RHS_dt], rank+1)
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

    for i in range(max_iter):
        # Get the edges
        u_curr[0,:,:], u_curr[-1,:,:] = sendrecv_sliceedges(rank=rank, size=size, u_curr_slice=u_curr, LHS_dt=LHS_dt, RHS_dt=RHS_dt, comm=comm)
        comm.Barrier() # Don't know if this is needed, but to ensure that all edges has been recved before computing new values
        # Do jacobi
        u_new = jacobi_step_slice(u_curr, d2f, N, N_slice)
        local_diff = norm(u_curr, u_new)
        diff = comm.allreduce(local_diff, op=MPI.MAX) # to ensure entire cube, and not just current slice, has converges
        u_curr[1:-1, 1:-1, 1:-1] = u_new
        convergence_hist.append(diff)

        if diff < tolerance:
            break

    return u_curr, convergence_hist, i


    