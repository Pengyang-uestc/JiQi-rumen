"""Simple computation graph: NumPy manual backward vs. PyTorch autograd.

Run: python main.py --x 2 --y 3
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save the figure without opening a GUI window.
import matplotlib.pyplot as plt
import numpy as np
import torch


def numpy_forward_backward(x, y):
    """Compute scalar values and gradients, using only NumPy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Forward: split the expression into simple operations.
    a = x * y
    b = x ** 2
    c = a + b
    f = c ** 2

    # Backward: start at the output, then apply the chain rule.
    df = np.ones_like(f)  # df/df = 1
    dc = df * 2 * c      # df/dc = df/df * d(c^2)/dc
    da = dc * 1          # c = a + b, so dc/da = 1
    db = dc * 1          # c = a + b, so dc/db = 1

    # x reaches the output through TWO branches; add both contributions.
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dx += da * y         # a = x*y, so da/dx = y
    dy += da * x         # a = x*y, so da/dy = x
    dx += db * 2 * x     # b = x^2, so db/dx = 2*x

    values = {"x": x, "y": y, "a": a, "b": b, "c": c, "f": f}
    gradients = {"df": df, "dc": dc, "da": da, "db": db,
                 "dx": dx, "dy": dy}
    return values, gradients


def torch_forward_backward(x, y):
    """Let PyTorch compute gradients for the original expression."""
    x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y_tensor = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    f_tensor = (x_tensor * y_tensor + x_tensor ** 2) ** 2
    f_tensor.backward()
    return f_tensor.item(), x_tensor.grad.item(), y_tensor.grad.item()


def draw_graph(values, output_path):
    """Draw forward dependencies; backward travels along the reverse edges."""
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = {"x": (0.8, 3.0), "y": (0.8, 0.8),
                 "a": (3.6, 1.0), "b": (3.6, 3.0),
                 "c": (6.5, 2.0), "f": (9.4, 2.0)}
    labels = {"x": f"x = {values['x']:g}", "y": f"y = {values['y']:g}",
              "a": f"a = x * y\n= {values['a']:g}",
              "b": f"b = x^2\n= {values['b']:g}",
              "c": f"c = a + b\n= {values['c']:g}",
              "f": f"f = c^2\n= {values['f']:g}"}
    for source, target in [("x", "a"), ("y", "a"), ("x", "b"),
                           ("a", "c"), ("b", "c"), ("c", "f")]:
        ax.annotate("", xy=positions[target], xytext=positions[source],
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#526477",
                                    shrinkA=43, shrinkB=47))
    for name, position in positions.items():
        ax.text(*position, labels[name], ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.65", fc="#e7f1fb", ec="#3673a6"))
    ax.set(xlim=(-0.3, 10.7), ylim=(-0.1, 4.1))
    ax.set_title("f(x, y) = (xy + x^2)^2", fontsize=17)
    ax.text(5.2, 0.0, "Forward: follow arrows. Backward: reverse arrows; sum branches.",
            ha="center", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=2.0)
    parser.add_argument("--y", type=float, default=3.0)
    args = parser.parse_args()

    values, gradients = numpy_forward_backward(args.x, args.y)
    print("NumPy forward:")
    for name, value in values.items():
        print(f"  {name} = {value:g}")
    print("NumPy backward:")
    for name, value in gradients.items():
        print(f"  {name} = {value:g}")

    torch_f, torch_dx, torch_dy = torch_forward_backward(args.x, args.y)
    print(f"PyTorch: f = {torch_f:g}, dx = {torch_dx:g}, dy = {torch_dy:g}")
    np.testing.assert_allclose(
        [values["f"], gradients["dx"], gradients["dy"]],
        [torch_f, torch_dx, torch_dy], rtol=1e-12, atol=1e-12,
    )
    print("PASS: NumPy manual gradients match PyTorch autograd.")

    output_path = Path(__file__).resolve().parent / "computation_graph.png"
    draw_graph(values, output_path)
    print(f"Saved computation graph: {output_path}")


if __name__ == "__main__":
    main()
