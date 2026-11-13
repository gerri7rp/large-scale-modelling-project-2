import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path


def _savefig(title: str, kind: str) -> None:
    """Save to results/plots/<prefix>/<kind>/<name>, where title = "<prefix>/<name>".
    E.g. title="parallel/Jacobi N=300", kind="convergence"
    -> results/plots/parallel/convergence/Jacobi N=300.png"""
    title = Path(title)
    path = Path("results/plots") / title.parent / kind / title.name
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)

try:
    from utils.check_solution import analytical_solution  # imported from repo root
except ImportError:
    from check_solution import analytical_solution  # run directly from utils/


def plot_convergence(convergence_hist: list[float],
                     tolerance: float,
                     title: str, **kwargs) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(1, len(convergence_hist) + 1), convergence_hist, linewidth=2)

    # Mark the stopping tolerance
    if tolerance is not None:
        ax.axhline(tolerance, color="gray", linestyle="--", linewidth=1, label=f"tolerance = {tolerance:g}")
        ax.legend(frameon=False)

    ax.set(xlabel="Iteration", ylabel=r"$\max|u^{k+1} - u^k|$")
    ax.grid(True, alpha=0.3)

    # Add kwargs
    if kwargs:
        ax.set(**kwargs)

    fig.tight_layout()
    _savefig(title, "convergence")



def plot_slice_vs_analytical(u: np.ndarray,
                             pts: np.ndarray,
                             z: float = 0.5,
                             **kwargs) -> None:
    """Surface of the numerical solution on the slice closest to z, with the
    analytical solution as a wireframe on top, next to the pointwise error."""
    z_idx = int(np.argmin(np.abs(pts - z)))
    X, Y = np.meshgrid(pts, pts, indexing="ij")
    u_slice = u[:, :, z_idx]
    u_true = analytical_solution(X, Y, pts[z_idx])

    fig = plt.figure(figsize=(12, 5.5))

    # Numerical solution as colored surface, analytical as wireframe
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    lim = np.abs(u_true).max()
    surf = ax.plot_surface(X, Y, u_slice, cmap="RdBu_r", vmin=-lim, vmax=lim,
                           linewidth=0, antialiased=True, alpha=0.9)
    stride = max(1, len(pts) // 15)
    ax.plot_wireframe(X, Y, u_true, rstride=stride, cstride=stride,
                      color="black", linewidth=0.6, label="analytical")
    ax.set(xlabel="x", ylabel="y", zlim=(-lim, lim),
           xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[-1, 0, 1],
           title=f"Solution at z = {pts[z_idx]:.2f}")
    ax.legend(handles=[Patch(facecolor=plt.cm.RdBu_r(0.85), label="numerical"),
                       Line2D([], [], color="black", linewidth=0.6, label="analytical")],
              loc="upper left", frameon=False)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="u")

    # Pointwise error on the same slice
    ax2 = fig.add_subplot(1, 2, 2)
    err = np.abs(u_slice - u_true)
    im = ax2.imshow(err.T, origin="lower", extent=(pts[0], pts[-1], pts[0], pts[-1]), cmap="viridis")
    ax2.set(xlabel="x", ylabel="y", xticks=[-1, -0.5, 0, 0.5, 1], yticks=[-1, -0.5, 0, 0.5, 1],
            title=f"Error |u - u_exact|, max = {err.max():.2e}")
    ax2.set_aspect("equal")
    fig.colorbar(im, ax=ax2, shrink=0.8, label="error")

    title = kwargs.pop("title", "slice_vs_analytical")
    fig.suptitle(Path(title).name)

    fig.tight_layout()
    _savefig(title, "solution")



def plot_planes(grid: tuple[np.ndarray, np.ndarray, np.ndarray], 
                arr: np.ndarray, 
                x_idx: int = 10, 
                y_idx: int = 10, 
                z_idx: int = 10,
                cmap = plt.cm.viridis, title: str = "planes", **kwargs) -> None:
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

    _savefig(title, "planes")


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
