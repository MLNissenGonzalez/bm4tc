import pytest
from analysis.utils.resolve import (
    resolve_regime_from_path,
    embedding_range_size,
    normalize_param,
    resolve_params,
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


# ---- resolve_regime_from_path — legacy backward compat ----

def test_resolve_regime_cls_legacy():
    result = resolve_regime_from_path("outputs/seed_sweep/cls/fourier/d4r3/moons_0102")
    assert result == "dis"


def test_resolve_regime_dis_legacy():
    result = resolve_regime_from_path("outputs/seed_sweep/dis/fourier/d4r3/circles_0102")
    assert result == "dis"


def test_resolve_regime_gen_legacy():
    result = resolve_regime_from_path("outputs/seed_sweep/gen/legendre/d10r6/moons_0102")
    assert result == "gen"


def test_resolve_regime_adv_legacy():
    result = resolve_regime_from_path("outputs/seed_sweep/adv/fourier/d10r6/moons_0102")
    assert result == "adv"


def test_resolve_regime_old_flat_path():
    result = resolve_regime_from_path("outputs/seed_sweep_adv_d30r18fourier_moons_4k_1202")
    assert result == "adv"


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


# ---- normalize_param ----

def test_normalize_param_aliases():
    assert normalize_param("wd") == "weight-decay"
    assert normalize_param("bs") == "batch-size"


def test_normalize_param_passthrough():
    assert normalize_param("lr") == "lr"


# ---- resolve_params — new vocabulary ----

def test_resolve_params_nat_returns_nll_lr_path():
    result = resolve_params("nat", ["lr"])
    assert "lr" in result
    assert "trainer.nll" in result["lr"]


def test_resolve_params_at_returns_adversarial_lr_path():
    result = resolve_params("at", ["lr"])
    assert "lr" in result
    assert "trainer.adversarial" in result["lr"]


def test_resolve_params_at_has_clean_weight():
    result = resolve_params("at", ["clean-weight"])
    assert "clean-weight" in result
    assert "clean_weight" in result["clean-weight"]


# ---- resolve_params — legacy aliases ----

def test_resolve_params_dis_returns_nll_lr_path():
    result = resolve_params("dis", ["lr"])
    assert "lr" in result
    assert "trainer.nll" in result["lr"]


def test_resolve_params_gen_returns_nll_lr_path():
    result = resolve_params("gen", ["lr"])
    assert "lr" in result
    assert "trainer.nll" in result["lr"]


def test_resolve_params_adv_returns_adversarial_lr_path():
    result = resolve_params("adv", ["lr"])
    assert "lr" in result
    assert "trainer.adversarial" in result["lr"]


def test_resolve_params_weight_decay_alias():
    result = resolve_params("nat", ["wd"])
    assert "weight-decay" in result


def test_resolve_params_unknown_regime_raises():
    with pytest.raises(ValueError):
        resolve_params("nonexistent_regime", ["lr"])
