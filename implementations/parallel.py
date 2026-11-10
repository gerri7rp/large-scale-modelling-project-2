import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

def get_datatypes(arr: np.ndarray):
    """
    Creates custom datatypes for getting the left,right,top,bottom,front and back boundary
    of an arbitrary 3d array size
    """
    # Get array information
    Nx,Ny,Nz = arr.shape

    # Left boundary
    left = MPI.DOUBLE.Create_subarray(
        arr.shape, (1, Ny, Nz), (0, 0, 0), order=MPI.ORDER_C
    )
    right = MPI.DOUBLE.Create_subarray(
        arr.shape, (1, Ny, Nz), (Nx - 1, 0, 0), order=MPI.ORDER_C
    )
    # front = MPI.DOUBLE.Create_subarray(
    #     arr.shape, (Nx, 1, Nz), (0, 0, 0), order=MPI.ORDER_C
    # )
    # back = MPI.DOUBLE.Create_subarray(
    #     arr.shape, (Nx, 1, Nz), (0, Ny-1, 0), order=MPI.ORDER_C
    # )
    # buttom = MPI.DOUBLE.Create_subarray(
    #     arr.shape, (Nx, Ny, 1), (0, 0, 0), order=MPI.ORDER_C
    # )
    # top = MPI.DOUBLE.Create_subarray(
    #     arr.shape, (Nx, Ny, 1), (0, 0, Nz-1), order=MPI.ORDER_C
    # )


    return left, right