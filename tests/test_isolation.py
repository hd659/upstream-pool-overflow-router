import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_router.isolation import (  # noqa: E402
    HealthTier, NoSlotAvailable, PoolKind, PoolRouter, Upstream, classify_health,
)

E, S = PoolKind.ENTERPRISE, PoolKind.SHARED


def make(ent_cap=2, shared_cap=2):
    r = PoolRouter()
    r.add(Upstream("e1", E, ent_cap))
    r.add(Upstream("s1", S, shared_cap))
    return r


def test_allowed_pools_is_one_way():
    assert PoolRouter.allowed_pools(E) == [E, S]
    assert PoolRouter.allowed_pools(S) == [S]
    with pytest.raises(ValueError):
        PoolRouter.allowed_pools("vip")


def test_enterprise_prefers_own_pool():
    slot = make().acquire_upstream_slot(E)
    assert slot.pool == E and not slot.overflowed


def test_enterprise_overflows_to_shared_when_saturated():
    r = make(ent_cap=1)
    r.acquire_upstream_slot(E)
    slot = r.acquire_upstream_slot(E)
    assert slot.pool == S and slot.overflowed


def test_shared_never_borrows_enterprise():
    r = make(ent_cap=5, shared_cap=1)
    r.acquire_upstream_slot(S)
    with pytest.raises(NoSlotAvailable):
        r.acquire_upstream_slot(S)
    assert r.upstreams["e1"].in_use == 0


def test_regression_high_usage_enterprise_not_starved():
    # Bug: at 81%+ enterprise usage, later enterprise callers got rejected.
    r = PoolRouter()
    r.add(Upstream("e1", E, 10, in_use=9))
    r.add(Upstream("s1", S, 10))
    slots = [r.acquire_upstream_slot(E) for _ in range(3)]
    assert [s.pool for s in slots] == [E, S, S]


def test_total_exhaustion_raises():
    r = make(1, 1)
    r.acquire_upstream_slot(E)
    r.acquire_upstream_slot(E)
    with pytest.raises(NoSlotAvailable):
        r.acquire_upstream_slot(E)


@pytest.mark.parametrize("err,fails,tier", [
    (0.0, 0, HealthTier.HEALTHY),
    (0.1, 0, HealthTier.DEGRADED),
    (0.0, 2, HealthTier.PROBATION),
    (0.3, 0, HealthTier.PROBATION),
    (0.0, 5, HealthTier.UNHEALTHY),
    (0.6, 0, HealthTier.UNHEALTHY),
])
def test_classify_health(err, fails, tier):
    assert classify_health(err, fails) == tier


def test_tier_order_beats_utilisation():
    r = PoolRouter()
    r.add(Upstream("deg", E, 10, error_rate=0.1))
    r.add(Upstream("ok", E, 10, in_use=8))
    assert r.acquire_upstream_slot(E).upstream_id == "ok"


def test_utilisation_breaks_ties_in_tier():
    r = PoolRouter()
    r.add(Upstream("a", E, 10, in_use=5))
    r.add(Upstream("b", E, 10, in_use=1))
    assert r.acquire_upstream_slot(E).upstream_id == "b"


def test_unhealthy_skipped_then_overflow():
    r = PoolRouter()
    r.add(Upstream("e1", E, 10, consecutive_failures=5))
    r.add(Upstream("s1", S, 10))
    slot = r.acquire_upstream_slot(E)
    assert slot.upstream_id == "s1" and slot.overflowed


def test_release_and_failure_tracking():
    r = make()
    slot = r.acquire_upstream_slot(E)
    r.release(slot, success=False)
    assert r.upstreams["e1"].in_use == 0
    assert r.upstreams["e1"].consecutive_failures == 1
    with pytest.raises(RuntimeError):
        r.release(slot)


def test_repeated_failures_demote_and_reroute():
    r = make(ent_cap=10, shared_cap=10)
    for _ in range(5):
        r.release(r.acquire_upstream_slot(E), success=False)
    assert r.upstreams["e1"].tier == HealthTier.UNHEALTHY
    assert r.acquire_upstream_slot(E).pool == S


def test_add_validation():
    r = PoolRouter()
    with pytest.raises(ValueError):
        r.add(Upstream("x", "other", 1))
    with pytest.raises(ValueError):
        r.add(Upstream("x", E, -1))


def test_concurrent_acquire_never_exceeds_capacity():
    r = make(ent_cap=20, shared_cap=20)
    got, errs = [], []

    def worker():
        try:
            got.append(r.acquire_upstream_slot(E))
        except NoSlotAvailable:
            errs.append(1)

    ts = [threading.Thread(target=worker) for _ in range(60)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(got) == 40 and len(errs) == 20
    assert r.pool_usage(E) == 1.0 and r.pool_usage(S) == 1.0
