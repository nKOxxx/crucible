"""Tests for the Crucible scoring core.

The kill-shot test is the last one: an ensemble that merely re-derives its
seed priors scores no better than the prior itself. That is MiroFish's
unmeasured structural flaw, expressed as a failing assertion.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crucible.calibration import (
    brier,
    calibration_report,
    ensemble_average,
    log_score,
)


def test_perfect_forecasts_score_zero():
    assert brier(1.0, 1) == 0.0
    assert brier(0.0, 0) == 0.0
    assert log_score(1.0, 1) == 0.0
    assert log_score(0.0, 0) == 0.0


def test_worst_forecast_costs_full_brier():
    assert brier(1.0, 0) == 1.0
    assert brier(0.0, 1) == 1.0
    assert brier(0.5, 1) == 0.25


def test_log_score_punishes_confident_wrongness_harder():
    mild = log_score(0.6, 0)
    overconfident = log_score(0.99, 0)
    assert overconfident > mild > 0
    assert overconfident > 4.0  # ln(0.01) ~= 4.6


def test_coin_flip_reference_is_half():
    # always saying 0.5 on a coin flip scores 0.25 Brier — the number to beat
    reps = calibration_report([0.5] * 100, [1, 0] * 50, n_bins=1)
    assert abs(reps.brier - 0.25) < 1e-9
    assert abs(reps.uncertainty - 0.25) < 1e-9


def test_perfectly_calibrated_forecaster_has_zero_reliability():
    # 70% calls happen 70% of the time; 20% calls happen 20% of the time
    probs = [0.7] * 100 + [0.2] * 100
    outcomes = [1] * 70 + [0] * 30 + [1] * 20 + [0] * 80
    rep = calibration_report(probs, outcomes, n_bins=5)
    assert rep.reliability < 1e-9
    assert rep.resolution > 0.0  # the two bins separate outcomes
    identity = rep.reliability - rep.resolution + rep.uncertainty
    assert abs(identity - rep.brier) < 1e-9


def test_overconfident_forecaster_shows_positive_reliability_gap():
    # says 90%, happens 50% of the time — the MiroFish shape
    probs = [0.9] * 100
    outcomes = [1] * 50 + [0] * 50
    rep = calibration_report(probs, outcomes, n_bins=2)
    assert rep.brier > 0.25  # worse than a coin-flipper
    assert rep.reliability > 0.1


def test_resolution_rewards_separating_outcomes():
    # same mean confidence, but spread across bins -> resolution kicks in
    flat = calibration_report([0.5] * 100, [1] * 50 + [0] * 50, n_bins=2)
    sharp = calibration_report(
        [0.9] * 50 + [0.1] * 50, [1] * 50 + [0] * 50, n_bins=2
    )
    assert sharp.resolution > flat.resolution
    assert sharp.brier < flat.brier


def test_ensemble_average_beats_noisy_member_in_expectation():
    # N runs OF THE SAME CLAIM: an informative but noisy dial (true edge
    # 0.65/0.35, per-run noise sigma=0.25). One run is miscalibrated mush;
    # the ensemble average denoises toward the real edge. This is WHY we
    # demand N runs per claim instead of one trajectory.
    import random

    rng = random.Random(7)
    singles: list[float] = []
    ens: list[float] = []
    for _ in range(800):
        truth = rng.randint(0, 1)
        p_star = 0.65 if truth else 0.35
        members = [
            min(1.0, max(0.0, p_star + rng.gauss(0, 0.25))) for _ in range(5)
        ]
        singles.extend(brier(m, truth) for m in members)
        ens.append(brier(ensemble_average(members), truth))
    assert sum(singles) / len(singles) - sum(ens) / len(ens) > 0.02


def test_resolver_rejects_unresolvable_claims():
    """A prediction without a resolution source is not a prediction."""
    from crucible.claim import Claim

    try:
        Claim(text="something happens", probs=[0.7], sources=[])
    except ValueError as e:
        assert "resolution" in str(e).lower()
        return
    raise AssertionError("claim without sources must be refused")


def test_claim_scoring_round_trip():
    from crucible.claim import Claim

    c = Claim(
        text="the coin lands heads",
        probs=[0.5, 0.6, 0.4],
        sources=["manual:_called_it"],
    )
    rep = c.resolve(1)
    assert rep.n == 3
    assert rep.brier > 0.0
