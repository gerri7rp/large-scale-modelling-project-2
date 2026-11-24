from PIL import Image
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def plot_saved_result_vs_analytical(
    solution_path: str | Path,
    z: float = 0.5,
    atol: float | None = None,
    show: bool = True,
):
    """Load a binary 3D result and plot a slice against the analytical solution.

    The file is assumed to contain only the interior points, in C/NumPy order,
    with shape (N, N, N).
    """
    import matplotlib.pyplot as plt

    solution_path = Path(solution_path)
    flat_result = np.fromfile(solution_path, dtype=np.float64)

    if flat_result.size == 0:
        raise ValueError(f"Solution file is empty: {solution_path}")

    side_length = round(flat_result.size ** (1 / 3))
    if side_length ** 3 != flat_result.size:
        raise ValueError(
            f"Solution file contains {flat_result.size} values, which cannot form "
            "a cubic (N, N, N) result"
        )

    result = flat_result.reshape((side_length, side_length, side_length))
    points = np.linspace(-1.0, 1.0, side_length + 2)[1:-1]

    title = f"solution_check/{solution_path.stem}_N{side_length}"

    from utils.plotting import plot_slice_vs_analytical

    plot_slice_vs_analytical(result, points, z=z, title=title)

    if atol is not None:
        is_correct = is_solution_correct(result, atol=atol)
        print(
            f"{solution_path}: result {'matches' if is_correct else 'does not match'} "
            f"the analytical solution at atol={atol:g}"
        )

    if show:
        plt.show()

    return result


if __name__ == "__main__":
    solution_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("results/my.bin")
    )
    plot_saved_result_vs_analytical(solution_path, atol=1e-4)

