from PIL import Image
from pathlib import Path
import numpy as np

def analytical_solution(x: np.ndarray,
           y: np.ndarray,
           z: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)



def is_solution_correct(matrix: np.array, atol) -> bool:
    """
    Compares if resulting matrix and the analytical solution are the same. (Checks if the results are correct)
    Returns:
        True:  if the results match.
        False: if the results don't match.
    """

    print("HELLO IS SOLUTION CORRECT")

    if matrix.ndim != 3:
        print("The dimension of the results is incorrect!")
        return False

    # Obtains all the matrix points
    n = matrix.shape[0]
    points = np.linspace(-1.0, 1.0, n + 2)[1:-1]

    # Calculates analytical solution for all the points of the matrix
    reference = analytical_solution(
        points[:, None, None],
        points[None, :, None],
        points[None, None, :],
    )

    # Checks that the analytical solution and the result are close enough
    is_valid = bool(np.allclose(matrix, reference, rtol=0.0, atol=atol))

    return is_valid

