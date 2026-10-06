import pytest
from analysis.utils.resolve import (
    resolve_regime_from_path,
    resolve_embedding_from_path,
    embedding_range_size,
)


# ---- resolve_regime_from_path — new vocabulary ----

def test_resolve_regime_nat():
    result = resolve_regime_from_path("outputs/circles/nat/fourier/d4r3/seed_sweep_a0_0102")
    assert result == "nat"


def test_resolve_regime_at():
    result = resolve_regime_from_path("outputs/circles/at/legendre/d10r6/seed_sweep_2804")
    assert result == "at"


def test_resolve_regime_nat_multirun():
    result = resolve_regime_from_path("outputs/moons/nat/legendre/d10r6/hpo_a1_2804/0")
    assert result == "nat"


def test_resolve_regime_at_with_time():
    result = resolve_regime_from_path("outputs/circles/at/fourier/d10r6/seed_sweep_2804_1234")
    assert result == "at"


def test_resolve_regime_legacy_tokens_are_not_regimes():
    """D10/D12: dis/gen/adv/cls are no longer regime names."""
    for token in ("dis", "gen", "adv", "cls"):
        assert resolve_regime_from_path(f"outputs/seed_sweep/{token}/fourier/d4r3/moons_0102") is None


def test_resolve_regime_none():
    result = resolve_regime_from_path("outputs/some_other/path/here")
    assert result is None


def test_resolve_regime_unknown_token():
    result = resolve_regime_from_path("outputs/seed_sweep/gan/fourier/d4r3/moons_0102")
    assert result is None


# ---- embedding_range_size ----

def test_embedding_range_size_fourier():
    assert embedding_range_size("fourier") == pytest.approx(1.0)


def test_embedding_range_size_legendre():
    assert embedding_range_size("legendre") == pytest.approx(2.0)


def test_embedding_range_size_hermite():
    assert embedding_range_size("hermite") == pytest.approx(8.0)


def test_embedding_range_size_chebychev1():
    assert embedding_range_size("chebychev1") == pytest.approx(1.98)


def test_embedding_range_size_unknown_fallback():
    assert embedding_range_size("unknown_emb") == pytest.approx(1.0)


# ---- resolve_embedding_from_path ----

def test_resolve_embedding_from_path():
    assert resolve_embedding_from_path("outputs/moons/nat/legendre/d10r6/hpo_a1_2804") == "legendre"
    assert resolve_embedding_from_path("outputs/moons/nat/d10r6/hpo_a1_2804") is None
