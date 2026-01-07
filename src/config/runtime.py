from dataclasses import dataclass
from src.config.policy import RouteSimilarityPolicy, DeviationPolicy, ImprovementPolicy, BusDistancePolicy, SeverityScorePolicy
import torch


@dataclass(frozen=True)
class RuntimeContext:
    device: str
    similarity: RouteSimilarityPolicy
    deviation: DeviationPolicy
    improvement: ImprovementPolicy
    distance: BusDistancePolicy
    severity: SeverityScorePolicy


def create_runtime_context(verbose=False) -> RuntimeContext:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        if device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

    return RuntimeContext(
        device=device,
        similarity=RouteSimilarityPolicy(),
        deviation=DeviationPolicy(),
        improvement=ImprovementPolicy(),
        distance=BusDistancePolicy(),
        severity=SeverityScorePolicy()
        )