import sys
from time import time
import numpy as np
from mpi4py import MPI

# Set up comm and cartesian map
comm = MPI.COMM_WORLD.Clone()
size = comm.size
N = size//3
dims = MPI.Compute_dims(N*3,3)
cart_comm = comm.Create_cart(dims, periods=[False, False, False])

# Datatype
dt = None #custom datatype

# Data
block = np.random.rand(3,3,3)

# Finding your location in the cartesian map
rank = comm.Get_rank()
coords = cart_comm.Get_coords(rank)

# Start by even ranks send/recv in x-direction
if rank%2 == 0:
    # Send in x-direction
    source, dest = cart_comm.Shift(0, 1)
    block_recved_bounds = np.empty_like(block)

    # Send to dest/right first
    if dest == MPI.COMM_NULL: # Handle cube boundary dest/right
        
        pass
    elif source == MPI.COMM_NULL: # Handle cube boundary source/left

    else:
        comm.Send((block, 1, dt), dest)
        comm.Recv()
