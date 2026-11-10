import numpy as np
import matplotlib.pyplot as plt

from check_solution import analytical_solution


def plot_planes(grid: tuple[np.ndarray, np.ndarray, np.ndarray], 
                arr: np.ndarray, 
                x_idx: int = 10, 
                y_idx: int = 10, 
                z_idx: int = 10,
                cmap = plt.cm.viridis,
                **kwargs) -> None:
    # Extract grid
    N = len(arr)
    X,Y,Z = grid

    # Define plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Normalize values for better vizualisation
    norm = plt.Normalize(vmin=-1, vmax=1)

    # Plot xyz planes
    ax.plot_surface(X[x_idx, :, :],Y[x_idx, :, :],Z[x_idx, :, :], facecolors=cmap(norm(arr[x_idx, :, :])) )
    ax.plot_surface(X[:, y_idx, :],Y[:, y_idx, :],Z[:, y_idx, :], facecolors=cmap(norm(arr[:, y_idx, :])) )
    ax.plot_surface(X[:, :, z_idx],Y[:, :, z_idx],Z[:, :, z_idx], facecolors=cmap(norm(arr[:, :, z_idx])) )

    # Add kwargs
    if kwargs:
        ax.set(**kwargs)

    plt.show()

if __name__ == "__main__":
    N = 100

    # Create meshgrid
    x_eval = np.linspace(-1, 1, N)
    y_eval = np.linspace(-1, 1, N)
    z_eval = np.linspace(-1, 1, N)
    X,Y,Z = np.meshgrid(x_eval,y_eval,z_eval)

    # Evaluate true solution
    sol_true = analytical_solution(X,Y,Z)

    plot_planes((X,Y,Z), sol_true,  xlabel= "hej")
