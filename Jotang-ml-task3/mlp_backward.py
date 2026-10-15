"""Task 3: one forward/backward pass, first N=1, then N=4.

Run: python mlp_backward.py
NumPy implements every backward operation manually. PyTorch only checks it.
All calculations use float64 on CPU; no optimizer or parameter update is used.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
RTOL, ATOL = 1e-10, 1e-12
PARAMETERS = ("W1", "b1", "W2", "b2")


def initialize_parameters():
    """Fixed, small weights: input dimension D=2, hidden width H=3."""
    return {
        "W1": np.array([[0.4, -0.3, 0.2], [-0.5, 0.6, 0.1]], dtype=np.float64),
        "b1": np.array([[0.1, -0.2, 0.05]], dtype=np.float64),
        "W2": np.array([[0.7], [-0.4], [0.3]], dtype=np.float64),
        "b2": np.array([[-0.1]], dtype=np.float64),
    }  # W1:(2,3), b1:(1,3), W2:(3,1), b2:(1,1)


def numpy_forward(X, y, parameters):
    """X:(N,2), y:(N,1), loss: scalar shape ()."""
    W1, b1, W2, b2 = (parameters[name] for name in PARAMETERS)
    z1 = X @ W1 + b1                     # (N,2)@(2,3)+(1,3) -> (N,3)
    a1 = np.maximum(z1, 0.0)             # ReLU: (N,3)
    z2 = a1 @ W2 + b2                    # (N,3)@(3,1)+(1,1) -> (N,1)
    p = 1.0 / (1.0 + np.exp(-z2))        # Sigmoid: (N,1), P(y=1)
    # Our small inputs/weights keep p away from 0 and 1; no clipping needed.
    sample_loss = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))  # (N,1)
    loss = np.mean(sample_loss)           # mean over N scalar losses -> ()
    return dict(X=X, y=y, **parameters, z1=z1, a1=a1, z2=z2, p=p,
                sample_loss=sample_loss, loss=loss)


def numpy_backward(values):
    """Every entry is d(loss)/d(node), with the SAME shape as that node."""
    X, y = values["X"], values["y"]       # (N,2), (N,1)
    W1, W2 = values["W1"], values["W2"]  # (2,3), (3,1)
    z1, a1, p = values["z1"], values["a1"], values["p"]
    N = X.shape[0]
    g = {"loss": np.array(1.0, dtype=np.float64)}  # ()
    g["sample_loss"] = np.ones_like(p) / N         # (N,1): mean backward
    # BCE backward, then Sigmoid backward: keep both steps explicit.
    g["p"] = g["sample_loss"] * (-y / p + (1.0 - y) / (1.0 - p))  # (N,1)
    g["z2"] = g["p"] * p * (1.0 - p)             # (N,1), = (p-y)/N
    np.testing.assert_allclose(g["z2"], (p - y) / N, rtol=RTOL, atol=ATOL)
    g["W2"] = a1.T @ g["z2"]                     # (3,N)@(N,1) -> (3,1)
    g["b2"] = g["z2"].sum(axis=0, keepdims=True)  # (1,1): undo broadcast
    g["a1"] = g["z2"] @ W2.T                     # (N,1)@(1,3) -> (N,3)
    g["z1"] = g["a1"] * (z1 > 0.0)               # (N,3): ReLU mask
    # At z1=0 we choose derivative 0, consistent with PyTorch.
    g["W1"] = X.T @ g["z1"]                      # (2,N)@(N,3) -> (2,3)
    g["b1"] = g["z1"].sum(axis=0, keepdims=True)  # (1,3): undo broadcast
    g["X"] = g["z1"] @ W1.T                      # (N,3)@(3,2) -> (N,2)
    return g


def torch_forward_backward(X, y, parameters):
    """Copy the EXACT input/parameters; use Linear, ReLU, Sigmoid and BCE."""
    nodes = {name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
             for name, value in {"X": X, **parameters}.items()}
    target = torch.tensor(y, dtype=torch.float64)  # labels are constants
    # F.linear uses weight shape (out,in), so transpose our (in,out) matrices.
    # The leaf W1/W2 themselves retain the same shapes used by NumPy.
    nodes["z1"] = F.linear(nodes["X"], nodes["W1"].T, nodes["b1"].squeeze(0))
    nodes["a1"] = torch.relu(nodes["z1"])
    nodes["z2"] = F.linear(nodes["a1"], nodes["W2"].T, nodes["b2"].squeeze(0))
    nodes["p"] = torch.sigmoid(nodes["z2"])
    nodes["sample_loss"] = F.binary_cross_entropy(nodes["p"], target, reduction="none")
    nodes["loss"] = nodes["sample_loss"].mean()
    # Intermediate (non-leaf) gradients are not saved by default.
    for node in nodes.values():
        node.retain_grad()
    nodes["loss"].backward()
    values = {name: node.detach().numpy().copy() for name, node in nodes.items()}
    gradients = {name: node.grad.detach().numpy().copy() for name, node in nodes.items()}
    return values, gradients


def array_text(array):
    return np.array2string(np.asarray(array), precision=12, suppress_small=False,
                           floatmode="fixed")


def compare_case(X, y, parameters):
    """Print/save every parameter and intermediate gradient, not only True/False."""
    values = numpy_forward(X, y, parameters)
    gradients = numpy_backward(values)
    torch_values, torch_gradients = torch_forward_backward(X, y, parameters)
    lines = [f"## Batch N={X.shape[0]}", "", "### Forward values", ""]
    for name, value in values.items():
        lines.extend([f"{name}: shape={np.shape(value)}", "```text", array_text(value), "```", ""])
        if name in torch_values:
            np.testing.assert_allclose(value, torch_values[name], rtol=RTOL, atol=ATOL)
    lines.extend(["### Gradient comparison", "",
                  "All gradients below are derivatives of the mean loss.", ""])
    for name in (*PARAMETERS, "X", "z1", "a1", "z2", "p", "sample_loss", "loss"):
        manual, automatic = gradients[name], torch_gradients[name]
        assert np.shape(manual) == np.shape(values[name]) == np.shape(automatic)
        error = float(np.max(np.abs(manual - automatic)))
        passed = np.allclose(manual, automatic, rtol=RTOL, atol=ATOL)
        lines.extend([f"dL/d{name}: NumPy shape={np.shape(manual)}, "
                      f"PyTorch shape={np.shape(automatic)}", "```text",
                      "NumPy:\n" + array_text(manual),
                      "PyTorch:\n" + array_text(automatic), "```",
                      f"max absolute error={error:.6e}; {'PASS' if passed else 'FAIL'}", ""])
        np.testing.assert_allclose(manual, automatic, rtol=RTOL, atol=ATOL)
    lines.extend(["Forward and ALL gradient checks: PASS", ""])
    return lines, values, gradients


def draw_graph(output_path):
    """Forward graph, with parameter/target dependencies and node shapes."""
    fig, ax = plt.subplots(figsize=(18, 5.5))
    positions = {
        "X": (0, 1), "z1": (2.5, 1), "a1": (5, 1), "z2": (7.5, 1),
        "p": (10, 1), "ell": (12.5, 1), "loss": (15, 1),
        "param1": (2.5, 3.0), "param2": (7.5, 3.0), "y": (12.5, 3.0),
    }
    labels = {
        "X": "X\n(N, 2)", "z1": "z1 = X @ W1 + b1\n(N, 3)",
        "a1": "a1 = ReLU(z1)\n(N, 3)", "z2": "z2 = a1 @ W2 + b2\n(N, 1)",
        "p": "p = Sigmoid(z2)\n(N, 1)", "ell": "ell = BCE(p, y)\n(N, 1)",
        "loss": "L = mean(ell)\nscalar ()",
        "param1": "W1 (2, 3)\nb1 (1, 3)",
        "param2": "W2 (3, 1)\nb2 (1, 1)", "y": "y (constant)\n(N, 1)",
    }
    order = ["X", "z1", "a1", "z2", "p", "ell", "loss"]
    edges = list(zip(order[:-1], order[1:]))
    edges += [("param1", "z1"), ("param2", "z2"), ("y", "ell")]
    for source, target in edges:
        ax.annotate("", xy=positions[target], xytext=positions[source],
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#526477",
                                    shrinkA=44, shrinkB=54))
    for name, point in positions.items():
        ax.text(*point, labels[name], ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.6", fc="#e7f1fb", ec="#3673a6"))
    ax.text(7.5, -0.55, "Backward: reverse the arrows. Each gradient has its node's shape.\n"
            "Mean loss contributes 1/N; shared bias gradients sum over samples.",
            ha="center", fontsize=12)
    ax.set(xlim=(-1.2, 16.3), ylim=(-1.0, 4.0))
    ax.set_title("Two-layer binary classifier: hidden width H=3; batch N=1 or N=4", fontsize=16)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    # EXACT same generator settings as Task 1's baseline. No standardization there.
    X_all, y_all = make_moons(n_samples=1000, noise=0.2, random_state=42)
    # The first four raw labels are all 1. Include both classes to check BCE.
    indices = np.array([0, 1, 4, 9])  # fixed labels: [1, 1, 0, 0]
    X = X_all[indices].astype(np.float64)                 # (4,2)
    y = y_all[indices].astype(np.float64).reshape(-1, 1)  # (4,1), avoid broadcasting
    parameters = initialize_parameters()
    original_parameters = {name: value.copy() for name, value in parameters.items()}
    lines = ["# NumPy vs. PyTorch: measured results", "",
             "Data: make_moons(n_samples=1000, noise=0.2, random_state=42).",
             "Single sample: index 0. Small batch: indices 0, 1, 4, 9 (labels 1, 1, 0, 0).",
             "CPU float64; fixed parameters; mean BCE; no parameter updates.",
             f"Tolerance: rtol={RTOL}, atol={ATOL}.", ""]
    for N in (1, 4):
        result, _, _ = compare_case(X[:N], y[:N], parameters)
        lines.extend(result)

    # Independent batch-mean check: shared parameter gradients are sample means.
    batch_gradients = numpy_backward(numpy_forward(X, y, parameters))
    singles = [numpy_backward(numpy_forward(X[i:i+1], y[i:i+1], parameters))
               for i in range(4)]
    for name in PARAMETERS:
        expected = np.mean([g[name] for g in singles], axis=0)
        np.testing.assert_allclose(batch_gradients[name], expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_array_equal(parameters[name], original_parameters[name])
    lines.extend(["Batch parameter gradients equal the mean of four single-sample gradients: PASS.",
                  "All parameters are unchanged: PASS.", ""])
    report = "\n".join(lines)
    print(report)
    (ROOT / "mlp_gradient_results.md").write_text(report, encoding="utf-8")
    draw_graph(ROOT / "mlp_computation_graph.png")
    print("Saved: mlp_gradient_results.md and mlp_computation_graph.png")


if __name__ == "__main__":
    main()
