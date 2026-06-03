"""Convert GR00T ONNX models (balance + walk) to TorchScript for GPU inference.

Usage:
    conda activate isaaclab
    python scripts/convert_gr00t_to_torchscript.py

Model architecture (both balance and walk are identical):
    obs(516) -> estimator(516->256->256->35)
        latent = est[:, :3]           (3D)
        code = L2_normalize(est[:, 3:])  (32D)
    actor_input = concat(obs[:, -86:], latent, code) = 86+3+32 = 121
    actor_input -> actor(121->512->256->256->15)
    output = actor_output  (no final clip needed, raw range ~[-1,1])
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn


class Gr00tPolicy(nn.Module):
    def __init__(self, onnx_path: str):
        super().__init__()
        model = onnx.load(onnx_path)
        weights = {}
        for init in model.graph.initializer:
            weights[init.name] = onnx.numpy_helper.to_array(init)

        # Estimator: 516 -> 256 -> 256 -> 35
        self.estimator = nn.Sequential(
            nn.Linear(516, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 35),
        )

        # Actor: 121 -> 512 -> 256 -> 256 -> 15
        self.actor = nn.Sequential(
            nn.Linear(121, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 15),
        )

        # Load weights from ONNX
        estimator_pairs = [
            (0, "estimator.0.weight", "estimator.0.bias"),
            (2, "estimator.2.weight", "estimator.2.bias"),
            (4, "estimator.4.weight", "estimator.4.bias"),
        ]
        for idx, w_name, b_name in estimator_pairs:
            self.estimator[idx].weight.data = torch.from_numpy(weights[w_name].copy())
            self.estimator[idx].bias.data = torch.from_numpy(weights[b_name].copy())

        actor_pairs = [
            (0, "actor.0.weight", "actor.0.bias"),
            (2, "actor.2.weight", "actor.2.bias"),
            (4, "actor.4.weight", "actor.4.bias"),
            (6, "actor.6.weight", "actor.6.bias"),
        ]
        for idx, w_name, b_name in actor_pairs:
            self.actor[idx].weight.data = torch.from_numpy(weights[w_name].copy())
            self.actor[idx].bias.data = torch.from_numpy(weights[b_name].copy())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (N, 516)
        est = self.estimator(obs)  # (N, 35)
        latent = est[:, :3]  # (N, 3)
        code_raw = est[:, 3:]  # (N, 32)
        # L2 normalize with epsilon guard
        code_norm = torch.norm(code_raw, dim=-1, keepdim=True).clamp(min=1e-12)
        code = code_raw / code_norm
        # Actor input: current frame (last 86) + latent + normalized code
        current_frame = obs[:, -86:]  # (N, 86)
        actor_input = torch.cat([current_frame, latent, code], dim=-1)  # (N, 121)
        return self.actor(actor_input)  # (N, 15)


def verify_conversion(onnx_path: str, ts_model: nn.Module, atol: float = 1e-5) -> bool:
    """Compare ONNX runtime output vs TorchScript output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not available, skipping verification")
        return True

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    test_input = np.random.randn(4, 516).astype(np.float32)

    # ONNX output
    onnx_output = session.run(None, {input_name: test_input})[0]

    # TorchScript output
    with torch.no_grad():
        ts_output = ts_model(torch.from_numpy(test_input)).numpy()

    max_diff = np.max(np.abs(onnx_output - ts_output))
    mean_diff = np.mean(np.abs(onnx_output - ts_output))
    print(f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

    if max_diff > atol:
        print(f"  FAILED: max diff {max_diff:.2e} > {atol:.2e}")
        return False
    print(f"  PASSED: max diff {max_diff:.2e} <= {atol:.2e}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert GR00T ONNX to TorchScript")
    parser.add_argument("--demo-dir", default="demo", help="Directory containing ONNX models")
    parser.add_argument("--output-dir", default="demo", help="Output directory for TorchScript models")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for verification")
    args = parser.parse_args()

    models = {
        "walk": "GR00T-WholeBodyControl-Walk.onnx",
        "balance": "GR00T-WholeBodyControl-Balance.onnx",
    }

    all_ok = True
    for name, onnx_filename in models.items():
        onnx_path = os.path.join(args.demo_dir, onnx_filename)
        output_path = os.path.join(args.output_dir, f"gr00t_{name}.pt")

        if not os.path.exists(onnx_path):
            print(f"SKIP: {onnx_path} not found")
            continue

        print(f"Converting {name}: {onnx_path} -> {output_path}")

        # Build model from ONNX weights
        model = Gr00tPolicy(onnx_path)
        model.eval()

        # Trace with example input
        dummy_input = torch.randn(1, 516)
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy_input)
        traced.save(output_path)
        print(f"  Saved: {output_path}")

        # Verify
        ok = verify_conversion(onnx_path, traced, atol=args.atol)
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll conversions successful!")
    else:
        print("\nSome conversions FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
