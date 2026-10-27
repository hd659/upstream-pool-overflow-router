import pytest

from pool_router.health_tiers import (
    Candidate, Tier, classify, overflow_order, rank, select, usage_tier,
)


def C(uid, pool="enterprise", cap=100, used=0, **kw):
    return Candidate(uid=uid, pool=pool, capacity=cap, in_use=used, **kw)


@pytest.mark.parametrize("ratio,tier", [
    (0.0, Tier.T1_FRESH), (0.49, Tier.T1_FRESH), (0.50, Tier.T2_WARM),
    (0.80, Tier.T2_WARM), (0.81, Tier.T3_HOT), (0.94, Tier.T3_HOT),
    (0.95, Tier.T4_CRITICAL), (0.99, Tier.T4_CRITICAL), (1.0, Tier.EXHAUSTED),
])
def test_usage_tier_boundaries(ratio, tier):
    assert usage_tier(ratio) is tier


def test_daily_ceiling_is_hard_gate():
    assert classify(C("a", daily_reqs=10, daily_cap=10)) is Tier.EXHAUSTED
    assert classify(C("a", daily_reqs=9, daily_cap=10)) is Tier.T1_FRESH
    assert classify(C("a", daily_cap=0)) is Tier.EXHAUSTED


def test_error_demotes_one_tier_capped():
    assert classify(C("a", used=10, error_rate=0.2)) is Tier.T2_WARM
    assert classify(C("a", used=97, error_rate=0.9)) is Tier.T4_CRITICAL


def test_full_capacity_and_zero_capacity_excluded():
    assert classify(C("a", used=100)) is Tier.EXHAUSTED
    assert classify(C("a", cap=0)) is Tier.EXHAUSTED


def test_rank_orders_by_tier_ratio_headroom_uid():
    cs = [
        C("hot", used=90),
        C("fresh_b", used=10, daily_reqs=50, daily_cap=100),
        C("fresh_a", used=10, daily_reqs=10, daily_cap=100),
        C("fresh_c", used=10),
        C("dead", used=100),
    ]
    assert [c.uid for c in rank(cs)] == ["fresh_c", "fresh_a", "fresh_b", "hot"]


def test_overflow_order_is_one_way():
    assert overflow_order("enterprise") == ["enterprise", "shared"]
    assert overflow_order("shared") == ["shared"]
    with pytest.raises(ValueError):
        overflow_order("vip")


def test_enterprise_at_81pct_not_starved():
    # Regression: C#6/C#7/C#8 at 81%+ were starved by two-way isolation.
    cs = [C(f"C#{i}", used=81 + i) for i in (6, 7, 8)]
    sel = select("enterprise", cs)
    assert sel is not None and sel.candidate.uid == "C#6"
    assert sel.tier is Tier.T3_HOT and not sel.overflowed


def test_enterprise_prefers_home_t4_over_shared_t1():
    cs = [C("e", used=98), C("s", pool="shared", used=0)]
    sel = select("enterprise", cs)
    assert sel.candidate.uid == "e" and not sel.overflowed


def test_enterprise_overflows_when_home_exhausted():
    cs = [C("e1", used=100), C("e2", daily_reqs=5, daily_cap=5),
          C("s", pool="shared", used=40)]
    sel = select("enterprise", cs)
    assert sel.candidate.uid == "s" and sel.overflowed


def test_shared_never_borrows_enterprise():
    cs = [C("e", used=0), C("s", pool="shared", used=100)]
    assert select("shared", cs) is None


def test_none_when_everything_exhausted():
    cs = [C("e", used=100), C("s", pool="shared", daily_reqs=1, daily_cap=1)]
    assert select("enterprise", cs) is None


def test_custom_bounds():
    assert classify(C("a", used=60), bounds=(0.7, 0.8, 0.9)) is Tier.T1_FRESH


@pytest.mark.parametrize("kw", [
    {"cap": -1}, {"used": -1}, {"daily_reqs": -1},
    {"daily_cap": -1}, {"error_rate": 1.5},
])
def test_validation(kw):
    with pytest.raises(ValueError):
        C("x", **kw)
