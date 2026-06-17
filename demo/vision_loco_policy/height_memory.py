from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_HEIGHT_FILE = "vision_height_memory.npz"


@dataclass
class HeightMemoryConfig:
    record_path: Path
    record_start_s: float = 9.2
    record_end_s: float = 13.0
    flat_after_end: bool = False


class HeightMemory:
    def __init__(self, config: HeightMemoryConfig):
        self.config = config
        self.record_path = Path(config.record_path)
        self._loaded = False
        self._times_s: torch.Tensor | None = None
        self._scans: torch.Tensor | None = None

    def record(self, times_s: list[float], scans: list[np.ndarray | torch.Tensor]) -> None:
        if len(times_s) != len(scans):
            raise ValueError("times_s and scans must have the same length")
        if len(times_s) == 0:
            raise ValueError("No height scan samples to record")

        scan_arrays = []
        for scan in scans:
            if isinstance(scan, torch.Tensor):
                scan = scan.detach().cpu().numpy()
            scan = np.asarray(scan, dtype=np.float32).reshape(-1)
            if scan.shape[0] != 2079:
                raise ValueError(f"Expected height scan with 2079 values, got {scan.shape[0]}")
            scan_arrays.append(scan)

        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.record_path,
            times_s=np.asarray(times_s, dtype=np.float32),
            scans=np.asarray(scan_arrays, dtype=np.float32),
            record_start_s=np.asarray(self.config.record_start_s, dtype=np.float32),
            record_end_s=np.asarray(self.config.record_end_s, dtype=np.float32),
        )
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not self.record_path.exists():
            raise FileNotFoundError(f"Missing recorded height memory file: {self.record_path}")
        data = np.load(self.record_path, allow_pickle=False)
        self._times_s = torch.tensor(data["times_s"], dtype=torch.float32)
        self._scans = torch.tensor(data["scans"], dtype=torch.float32)
        self._loaded = True

    def sample(self, elapsed_time_s: float, num_envs: int, device: torch.device) -> torch.Tensor:
        self.load()
        assert self._times_s is not None and self._scans is not None

        if elapsed_time_s <= self.config.record_start_s:
            return torch.zeros(num_envs, 2079, device=device)
        if elapsed_time_s >= self.config.record_end_s:
            if self.config.flat_after_end:
                return torch.zeros(num_envs, 2079, device=device)
            scan = self._scans[-1].to(device=device)
            if scan.ndim == 1:
                scan = scan.unsqueeze(0)
            if scan.shape[0] != num_envs:
                scan = scan.repeat(num_envs, 1)
            return scan

        idx = torch.searchsorted(self._times_s, torch.tensor([elapsed_time_s], dtype=torch.float32)).item()
        idx = max(0, min(int(idx), int(self._times_s.shape[0] - 1)))
        scan = self._scans[idx].to(device=device)
        if scan.ndim == 1:
            scan = scan.unsqueeze(0)
        if scan.shape[0] != num_envs:
            scan = scan.repeat(num_envs, 1)
        return scan
