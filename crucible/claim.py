"""Claims — falsifiable statements with a resolution path."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Claim:
    """A resolvable claim. No resolution source, no claim — that is the law.

    `probs` is the per-run forecast distribution from the simulation ensemble
    (one probability per OASIS run). `sources` names where the outcome comes
    from: "market:polymarket:<slug>", "metaculus:<id>", or "manual:<who>".
    """

    text: str
    probs: list[float]
    sources: list[str]
    resolve_by: date | None = None
    id: str | None = None
    _report: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("claim text is required")
        if not self.probs:
            raise ValueError("an unensemble'd claim is a vibe, not a forecast")
        for p in self.probs:
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"probability out of range: {p}")
        if not self.sources or any(not s.strip() for s in self.sources):
            raise ValueError(
                "a prediction without a resolution source is not a prediction: "
                "name at least one source (market:*, metaculus:*, manual:*)"
            )

    def resolve(self, outcome: int, scorer=None) -> object:
        """Settle the claim with ground truth and return its scorecard."""
        if outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1")
        from crucible.calibration import calibration_report

        self._report = calibration_report(self.probs, [outcome] * len(self.probs))
        return self._report
