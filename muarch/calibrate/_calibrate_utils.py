import numpy as np

__all__ = ["validate_target_mean", "validate_target_sd"]


def validate_target_mean(data: np.ndarray, mean: np.ndarray):
    assert len(mean) == data.shape[2], "length of target returns (mean) must equal number of assets"
    assert np.isfinite(np.asarray(mean)).all(), "all target returns (mean) must be finite"


def validate_target_sd(data: np.ndarray, sd: np.ndarray):
    _, _, num_assets = data.shape
    assert num_assets == len(sd), "length of target vols (sd) must equal number of assets"
    assert np.isfinite(np.asarray(sd)).all(), "all target vols (sd) must be finite"
    assert (np.asarray(sd) >= 0).all(), "target vol must all be >= 0"
