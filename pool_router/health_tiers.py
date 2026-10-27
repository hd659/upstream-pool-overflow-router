"""4-tier health overflow selection with daily-request ceiling.

An upstream's tier combines two signals:
  1. Daily request ceiling: once daily_reqs >= daily_cap it is EXHAUSTED and
     never selected (hard gate, independent of usage).
  2. Usage ratio (in_use / capacity) buckets into 4 tiers:
       T1_FRESH     ratio <  0.50
       T2_WARM      ratio <  0.81   (0.81 = historical starvation threshold)
       T3_HOT       ratio <  0.95
       T4_CRITICAL  ratio <  1.00   (last resort)
     ratio >= 1.0 means no free slot -> excluded.
  Error signals demote by one tier (capped at T4) instead of excluding,
  so a noisy-but-alive upstream still beats starvation.

Selection order: lowest tier, then lowest ratio, then most daily headroom,
then uid for determinism. Enterprise callers scan enterprise then shared;
shared callers scan shared only (one-way overflow).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, List, Optional, Sequence, Tuple

USAGE_BOUNDS: Tuple[float, float, float] = (0.50, 0.81, 0.95)
ERROR_DEMOTE_RATE = 0.10


class Tier(IntEnum):
    T1_FRESH = 1
    T2_WARM = 2
    T3_HOT = 3
    T4_CRITICAL = 4
    EXHAUSTED = 99


@dataclass(frozen=True)
class Candidate:
    uid: str
    pool: str
    capacity: int
    in_use: int = 0
    daily_reqs: int = 0
    daily_cap: Optional[int] = None  # None = unlimited
    error_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity < 0 or self.in_use < 0 or self.daily_reqs < 0:
            raise ValueError("counters must be >= 0")
        if self.daily_cap is not None and self.daily_cap < 0:
            raise ValueError("daily_cap must be >= 0")
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError("error_rate must be in [0, 1]")

    @property
    def usage_ratio(self) -> float:
        return self.in_use / self.capacity if self.capacity else 1.0

    @property
    def daily_headroom(self) -> float:
        if self.daily_cap is None:
            return float("inf")
        return float(self.daily_cap - self.daily_reqs)


def usage_tier(ratio: float, bounds: Sequence[float] = USAGE_BOUNDS) -> Tier:
    if ratio >= 1.0:
        return Tier.EXHAUSTED
    for limit, tier in zip(bounds, (Tier.T1_FRESH, Tier.T2_WARM, Tier.T3_HOT)):
        if ratio < limit:
            return tier
    return Tier.T4_CRITICAL


def classify(c: Candidate, bounds: Sequence[float] = USAGE_BOUNDS) -> Tier:
    if c.daily_cap is not None and c.daily_reqs >= c.daily_cap:
        return Tier.EXHAUSTED
    tier = usage_tier(c.usage_ratio, bounds)
    if tier is Tier.EXHAUSTED:
        return tier
    if c.error_rate >= ERROR_DEMOTE_RATE:
        tier = Tier(min(int(tier) + 1, int(Tier.T4_CRITICAL)))
    return tier


def _key(c: Candidate, bounds: Sequence[float]):
    return (classify(c, bounds), c.usage_ratio, -c.daily_headroom, c.uid)


def rank(cands: Iterable[Candidate],
         bounds: Sequence[float] = USAGE_BOUNDS) -> List[Candidate]:
    """Selectable candidates sorted best-first (EXHAUSTED dropped)."""
    live = [c for c in cands if classify(c, bounds) is not Tier.EXHAUSTED]
    return sorted(live, key=lambda c: _key(c, bounds))


def overflow_order(caller_pool: str) -> List[str]:
    if caller_pool == "enterprise":
        return ["enterprise", "shared"]
    if caller_pool == "shared":
        return ["shared"]
    raise ValueError(f"unknown caller pool: {caller_pool}")


@dataclass(frozen=True)
class Selection:
    candidate: Candidate
    tier: Tier
    overflowed: bool


def select(caller_pool: str, cands: Sequence[Candidate],
           bounds: Sequence[float] = USAGE_BOUNDS) -> Optional[Selection]:
    """Pick the best slot honouring one-way overflow.

    The home pool is exhausted entirely before overflowing: a T4 enterprise
    upstream is still preferred over a T1 shared one, so shared capacity is
    only consumed when enterprise truly has nothing left.
    """
    for idx, pool in enumerate(overflow_order(caller_pool)):
        ranked = rank((c for c in cands if c.pool == pool), bounds)
        if ranked:
            best = ranked[0]
            return Selection(best, classify(best, bounds), overflowed=idx > 0)
    return None
