"""JEM in bm4tc.core.jem: model, sampler and trainer.

The pinned numbers come from the pre-Phase-6 baseline (baselines/jem at commit
2f0ce4b4, scratch script in the Phase 6.2 commit message): the moved model and
sampler reproduce them exactly; the new trainer reproduces the old natural
trainer's first epoch (see EPOCHS for why only the first).
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bm4tc.core.interface import class_probabilities, log_px, nll
from bm4tc.core.jem.model import (JEMMLP, JEMModelConfig, matched_width,
                                  mlp_parameter_count, mps_parameter_count)
from bm4tc.core.jem.sampler import ReplayBuffer, SGLDConfig, SGLDSampler
from bm4tc.core.jem.train import JEMConfig, JEMTrainer, ValidSamplerConfig
from bm4tc.core.objective import OptimizerConfig
from bm4tc.core.train import TrainConfig

CPU = torch.device("cpu")


def small(seed=0):
    torch.manual_seed(seed)
    return JEMMLP(JEMModelConfig(hidden_dims=[8, 8]), data_dim=4, num_classes=3)


def test_matching_r20_real_degrees_of_freedom():
    target = 2 * mps_parameter_count(144, 3, 20, 10)
    assert target == 349_040
    width = matched_width(target, 144, 10)
    assert width == 518
    model = JEMMLP(JEMModelConfig(match_in_dim=3, match_bond_dim=20), 144, 10)
    assert model.hidden_dims == [518, 518]
    assert model.count_parameters() == mlp_parameter_count(144, [518, 518], 10)
    assert abs(model.count_parameters() - target) < abs(
        mlp_parameter_count(144, [519, 519], 10) - target)


def test_interface():
    model = small()
    x, y = torch.rand(5, 4) * 2 - 1, torch.tensor([0, 1, 2, 0, 1])
    f = model(x)
    assert torch.equal(model.log_joint(x), f)
    torch.testing.assert_close(log_px(model, x), torch.logsumexp(f, -1))
    torch.testing.assert_close(class_probabilities(model, x), f.softmax(-1))
    torch.testing.assert_close(nll(model, x, y), torch.nn.functional.cross_entropy(f, y))


def test_pinned_logits_and_sgld():
    model = small()
    x = torch.linspace(-1, 1, 12).reshape(3, 4)
    torch.testing.assert_close(model(x).flatten(), torch.tensor([
        0.058306559920310974, -0.3149015009403229, 0.09663770347833633,
        0.02477501705288887, -0.00039057619869709015, 0.028788279742002487,
        -0.0054853446781635284, 0.08444949984550476, -0.03414660692214966]), rtol=0, atol=1e-7)
    sampler = SGLDSampler(SGLDConfig(num_steps=5, step_size=0.1, noise_std=0.01, buffer_size=16),
                          ReplayBuffer(16, 4, model.input_range, seed=0))
    torch.manual_seed(1)
    torch.testing.assert_close(sampler.sample_training(model, 4, "cpu").flatten(), torch.tensor([
        0.38916242122650146, 0.8097428679466248, -0.19148622453212738, 0.7598147988319397,
        -0.7465163469314575, -0.5588380098342896, 0.6308173537254333, 0.567107081413269,
        -0.45225560665130615, 0.001027795486152172, 0.6123890280723572, 0.9865741729736328,
        -0.6065918803215027, -0.5022276043891907, -0.699862003326416, -0.9197969436645508]),
        rtol=0, atol=1e-7)


def _loaders():
    torch.manual_seed(0)
    X = torch.rand(64, 4) * 2 - 1
    Y = (X.sum(1) > 0).long() + (X[:, 0] > 0.5).long()
    train = DataLoader(TensorDataset(X[:48], Y[:48]), batch_size=16, shuffle=True)
    valid = DataLoader(TensorDataset(X[48:], Y[48:]), batch_size=16)
    return train, valid


def _trainer(alpha, evasion=None, clean_weight=0.0, max_epoch=3, save=False):
    train, valid = _loaders()
    model = small()
    cfg = TrainConfig(alpha=alpha, max_epoch=max_epoch, batch_size=16, patience=10,
                      evasion=evasion, clean_weight=clean_weight, save=save,
                      optimizer=OptimizerConfig(name="adam", kwargs={"lr": 1e-2, "weight_decay": 0.0}))
    jem = JEMConfig(
        sampler=SGLDConfig(num_steps=5, step_size=0.1, noise_std=0.01, buffer_size=32),
        valid_sampler=ValidSamplerConfig(num_steps=5, step_size=0.1, buffer_size=32,
                                         batch_size=8, num_batches=2, seed=7),
        energy_l2=1e-2, grad_clip=10.0, input_noise_std=0.01)
    return JEMTrainer(model, cfg, jem, train, valid, CPU, seed=0)


# alpha 0.5, 3 epochs: (objective + penalty on train, loss_dis/valid,
# objective/valid, acc/valid). Epoch 1 is the old NaturalTrainer's (nll/train,
# dis_loss/valid, mixed_loss/valid, acc/valid). Later epochs differ from it on
# purpose: validation is now the shared evaluate(), whose disabled tqdm around the
# loader costs one torch RNG draw that the old JEM validation did not, so the
# training stream shifts from epoch 2 on.
EPOCHS = [
    (1.07615065574646, 0.9986305236816406, 1.0344629883766174, 0.6875),
    (0.9945387803018093, 0.9299278259277344, 0.9544612765312195, 0.625),
    (0.9379352734734615, 0.867594301700592, 0.8835703432559967, 0.625),
]


def test_natural_epochs_are_pinned():
    log = []
    _trainer(0.5).train(on_epoch_end=lambda e, r: log.append(r))
    assert len(log) == 3
    for record, (train, dis, objective, acc) in zip(log, EPOCHS):
        t, v = record["train"], record["valid"]
        assert t["objective"] + t["penalty"] == pytest.approx(train, rel=1e-5)
        assert v["loss_dis"] == pytest.approx(dis, rel=1e-5)
        assert v["objective"] == pytest.approx(objective, rel=1e-5)
        assert v["acc"] == acc


def test_alpha_zero_has_no_generative_term():
    log = []
    _trainer(0.0, max_epoch=2).train(on_epoch_end=lambda e, r: log.append(r))
    v = log[-1]["valid"]
    assert v["loss_gen"] != v["loss_gen"]          # nan: no SGLD at alpha 0
    assert v["objective"] == v["loss_dis"]
    assert log[-1]["train"]["penalty"] == 0.0


def test_at_selects_on_objective_and_saves(tmp_path):
    evasion = {"method": "PGD", "eps_rel": [0.1], "num_steps": 3}
    trainer = _trainer(0.5, evasion=evasion, clean_weight=0.5, max_epoch=2, save=True)
    log = []
    trainer.train(on_epoch_end=lambda e, r: log.append(r), output_dir=tmp_path)
    v = log[-1]["valid"]
    assert set(v["rob"]) == {0.1} and v["n_rob"] == 8
    assert "eps_rel" in log[-1]["train"]
    loaded, extra = JEMMLP.load(str(tmp_path / "model"))
    for a, b in zip(loaded.state_dict().values(), trainer.model.state_dict().values()):
        assert torch.equal(a, b.cpu())
    assert torch.equal(extra["replay_buffer"]["data"], trainer.sampler.buffer.data)


def test_sgld_purification_moves_within_each_sweep():
    from bm4tc.core.jem.purification import SGLDPurification
    model = small()
    x = torch.rand(10, 4) * 1.6 - 0.8
    purifier = SGLDPurification(step_rel=0.05, steps=10, step_size=0.5, noise_std=0.01,
                                batch_size=4)
    snaps = purifier.purify_snapshots(model, x, [1, 3], "cpu")
    x1, s1 = snaps[1]
    x3, s3 = snaps[3]
    assert (x1 - x).abs().max() <= 0.1 + 1e-6          # radius 0.05 · width 2
    assert (x3 - x).abs().max() <= 0.3 + 1e-6          # re-centred every sweep
    assert (x1 - x).abs().max() > 0.01 and not torch.equal(x1, x3)
    torch.testing.assert_close(s1, log_px(model, x1).detach())
