"""Directional pool isolation: enterprise -> shared overflow, never reverse.

Fixes the bug where enterprise pool was coded as two-way hard isolation,
starving enterprise callers once their pool saturated (81%+ usage).

Rules:
  - Enterprise callers try the enterprise pool first, then overflow into shared.
  - Shared callers may ONLY use the shared pool (never borrow enterprise slots).
  - Within a pool, upstreams are chosen by 4 health tiers:
      HEALTHY > DEGRADED > PROBATION > UNHEALTHY(never selected)
    and inside a tier by lowest utilisation, then by id for determinism.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from threading import Lock
from typing import Dict, List, Optional


class PoolKind:
    ENTERPRISE = "enterprise"
    SHARED = "shared"


class HealthTier(IntEnum):
    HEALTHY = 0
    DEGRADED = 1
    PROBATION = 2
    UNHEALTHY = 3


def classify_health(error_rate: float, consecutive_failures: int) -> HealthTier:
    """Map raw signals to a tier. Thresholds are deliberately simple."""
    if consecutive_failures >= 5 or error_rate >= 0.5:
        return HealthTier.UNHEALTHY
    if consecutive_failures >= 2 or error_rate >= 0.2:
        return HealthTier.PROBATION
    if error_rate >= 0.05:
        return HealthTier.DEGRADED
    return HealthTier.HEALTHY


@dataclass
class Upstream:
    uid: str
    pool: str
    capacity: int
    in_use: int = 0
    error_rate: float = 0.0
    consecutive_failures: int = 0

    @property
    def tier(self) -> HealthTier:
        return classify_health(self.error_rate, self.consecutive_failures)

    @property
    def utilisation(self) -> float:
        return self.in_use / self.capacity if self.capacity else 1.0

    def has_room(self) -> bool:
        return self.in_use < self.capacity


@dataclass(frozen=True)
class Slot:
    upstream_id: str
    pool: str
    overflowed: bool
    tier: HealthTier


class NoSlotAvailable(RuntimeError):
    pass


@dataclass
class PoolRouter:
    upstreams: Dict[str, Upstream] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add(self, up: Upstream) -> None:
        if up.pool not in (PoolKind.ENTERPRISE, PoolKind.SHARED):
            raise ValueError(f"unknown pool: {up.pool}")
        if up.capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.upstreams[up.uid] = up

    @staticmethod
    def allowed_pools(caller_pool: str) -> List[str]:
        """The one-way overflow rule, as data."""
        if caller_pool == PoolKind.ENTERPRISE:
            return [PoolKind.ENTERPRISE, PoolKind.SHARED]
        if caller_pool == PoolKind.SHARED:
            return [PoolKind.SHARED]
        raise ValueError(f"unknown caller pool: {caller_pool}")

    def _pick(self, pool: str) -> Optional[Upstream]:
        cands = [
            u for u in self.upstreams.values()
            if u.pool == pool and u.has_room() and u.tier != HealthTier.UNHEALTHY
        ]
        if not cands:
            return None
        return min(cands, key=lambda u: (u.tier, u.utilisation, u.uid))

    def acquire_upstream_slot(self, caller_pool: str) -> Slot:
        pools = self.allowed_pools(caller_pool)
        with self._lock:
            for idx, pool in enumerate(pools):
                up = self._pick(pool)
                if up is not None:
                    up.in_use += 1
                    return Slot(up.uid, pool, overflowed=idx > 0, tier=up.tier)
        raise NoSlotAvailable(f"no slot for caller pool {caller_pool!r}")

    def release(self, slot: Slot, success: bool = True) -> None:
        with self._lock:
            up = self.upstreams[slot.upstream_id]
            if up.in_use <= 0:
                raise RuntimeError(f"double release on {up.uid}")
            up.in_use -= 1
            if success:
                up.consecutive_failures = 0
            else:
                up.consecutive_failures += 1

    def pool_usage(self, pool: str) -> float:
        ups = [u for u in self.upstreams.values() if u.pool == pool]
        cap = sum(u.capacity for u in ups)
        return sum(u.in_use for u in ups) / cap if cap else 0.0
