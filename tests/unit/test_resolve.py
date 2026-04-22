import pytest
from analysis.utils.resolve import (
    resolve_regime_from_path,
    embedding_range_size,
    normalize_param,
    resolve_params,
)


# ---- resolve_regime_from_path ----

def test_resolve_regime_cls():
    result = resolve_regime_from_path("outputs/seed_sweep/cls/fourier/d4D3/moons_0102")
    assert result == "pre"


def test_resolve_regime_gen():
    result = resolve_regime_from_path("outputs/seed_sweep/gen/legendre/d10D6/moons_0102")
    assert result == "gen"


def test_resolve_regime_adv():
    result = resolve_regime_from_path("outputs/seed_sweep/adv/fourier/d30D18/moons_0102")
    assert result == "adv"


def test_resolve_regime_gan():
    result = resolve_regime_from_path("outputs/seed_sweep/gan/fourier/d4D3/moons_0102")
    assert result == "gan"


def test_resolve_regime_clsadv_priority():
    result = resolve_regime_from_path("outputs/seed_sweep/clsadv/fourier/d30D18/moons_0102")
    assert result == "adv"


def test_resolve_regime_clsgen_priority():
    result = resolve_regime_from_path("outputs/seed_sweep/clsgen/legendre/d30D18/moons_0102")
    assert result == "gen"


def test_resolve_regime_none():
    result = resolve_regime_from_path("outputs/some_other/path/here")
    assert result is None


def test_resolve_regime_old_style_underscore():
    result = resolve_regime_from_path("outputs/seed_sweep_adv_d30D18fourier_moons_4k_1202")
    assert result == "adv"


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


# ---- normalize_param ----

def test_normalize_param_aliases():
    assert normalize_param("wd") == "weight-decay"
    assert normalize_param("bs") == "batch-size"


def test_normalize_param_passthrough():
    assert normalize_param("lr") == "lr"


# ---- resolve_params ----

def test_resolve_params_pre_returns_lr_path():
    result = resolve_params("pre", ["lr"])
    assert "lr" in result
    assert "classification" in result["lr"]


def test_resolve_params_unknown_regime_raises():
    with pytest.raises(ValueError):
        resolve_params("nonexistent_regime", ["lr"])


def test_resolve_params_weight_decay_alias():
    result = resolve_params("pre", ["wd"])
    assert "weight-decay" in result
