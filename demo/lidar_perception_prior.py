from dataclasses import dataclass, field


@dataclass
class BoxPrior:
    length: float = 1.0
    width: float = 0.8
    height: float = 0.6


@dataclass
class DitchPrior:
    length: float = 3.0
    width: float = 0.95
    depth: float = 0.6


@dataclass
class LidarPerceptionPrior:
    box: BoxPrior = field(default_factory=BoxPrior)
    ditch: DitchPrior = field(default_factory=DitchPrior)
