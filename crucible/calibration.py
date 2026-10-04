"""Crucible scoring core — pure Python, zero dependencies.

Answers the question MiroFish never asks: did the simulation KNOW anything?

Brier score decomposition (Murphy 1973), exact form:
    Brier = reliability - resolution + uncertainty + residual
    - reliability: "when we said 70%, did it happen 70% of the time?" (0 is perfect)
    - resolution:  "do our probabilities separate outcomes?" (higher is better)
    - uncertainty: irreducible variance of the outcome itself (p(1-p) of base rate)
    - residual:    everything the binned view cannot see: within-bin forecast
      variance MINUS twice the within-bin forecast/outcome covariance. The
      classical three-term identity assumes forecasts are constant within bins;
      with real-valued forecasts and coarse bins it lies by exactly this
      residual. We carry it so the identity holds for ANY binning. When it is
      large, your bins are too coarse to interpret separately — use finer bins
      or trust brier + reliability alone.

An ensemble that only re-derives its seed priors scores NO better than its
inputs. That test lives in tests/test_calibration.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def brier(prob: float, outcome: int) -> float:
    """Mean squared error of one probabilistic call. 0 = perfect, 1 = worst."""
    return (prob - outcome) ** 2


def log_score(prob: float, outcome: int, eps: float = 1e-12) -> float:
    """Negative log-likelihood of one call. 0 = perfect, grows without bound."""
    if outcome == 1 and prob == 1.0:
        return 0.0
    if outcome == 0 and prob == 0.0:
        return 0.0
    p = min(max(prob, eps), 1.0 - eps)
    return -(math.log(p) if outcome == 1 else math.log(1.0 - p))


def ensemble_average(probs: list[float]) -> float:
    """The one-line reason ensembles beat single runs (on average)."""
    if not probs:
        raise ValueError("ensemble needs at least one probability")
    return sum(probs) / len(probs)


@dataclass
class CalibrationReport:
    """Scorecard for a set of resolved predictions."""

    brier: float
    log_score: float
    reliability: float
    resolution: float
    uncertainty: float
    n: int
    bins: list[dict]
    residual: float = 0.0

    def summary(self) -> str:
        return (
            f"n={self.n}  brier={self.brier:.3f}  log={self.log_score:.3f}\n"
            f"reliability={self.reliability:.3f} (lower=better)  "
            f"resolution={self.resolution:.3f} (higher=better)  "
            f"uncertainty={self.uncertainty:.3f} (irreducible)  "
            f"residual={self.residual:.4f}\n"
            f"identity check: rel - res + unc + residual = "
            f"{self.reliability - self.resolution + self.uncertainty + self.residual:.3f}"
        )


def calibration_report(
    probs: list[float], outcomes: list[int], n_bins: int = 5
) -> CalibrationReport:
    """Score a batch of resolved predictions and decompose the Brier score.

    `probs[i]` is the forecast for `outcomes[i]` (1 = happened, 0 = did not).
    """
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must align")
    if not probs:
        raise ValueError("need at least one resolved prediction")
    for p in probs:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability out of range: {p}")

    n = len(probs)
    base_rate = sum(outcomes) / n
    brier_mean = sum(brier(p, o) for p, o in zip(probs, outcomes)) / n
    log_mean = sum(log_score(p, o) for p, o in zip(probs, outcomes)) / n
    uncertainty = base_rate * (1.0 - base_rate)

    bins: list[dict] = []
    reliability = 0.0
    resolution = 0.0
    residual = 0.0
    edges = [i / n_bins for i in range(n_bins + 1)]
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        members = [
            (p, o)
            for p, o in zip(probs, outcomes)
            if (lo <= p < hi) or (i == n_bins - 1 and p == hi)
        ]
        if not members:
            bins.append({"range": (lo, hi), "n": 0})
            continue
        conf = sum(p for p, _ in members) / len(members)
        freq = sum(o for _, o in members) / len(members)
        bins.append(
            {
                "range": (lo, hi),
                "n": len(members),
                "mean_confidence": conf,
                "observed_frequency": freq,
            }
        )
        weight = len(members) / n
        reliability += weight * (conf - freq) ** 2
        resolution += weight * (freq - base_rate) ** 2
        if len(members) > 1:
            var_p = sum((p - conf) ** 2 for p, _ in members) / len(members)
            cov_po = (
                sum((p - conf) * (o - freq) for p, o in members) / len(members)
            )
            residual += weight * (var_p - 2.0 * cov_po)

    return CalibrationReport(
        brier=brier_mean,
        log_score=log_mean,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        n=n,
        bins=bins,
        residual=residual,
    )
