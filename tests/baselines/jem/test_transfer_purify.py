import numpy as np

from baselines.jem.transfer_purify import (
    FIG_NAME,
    SAVE_DIR,
    _parse_models,
    _select_rows,
    _transfer_mask,
)


def test_jem_transfer_purify_uses_the_jem_pdf_output_path():
    assert SAVE_DIR == "figures/jem_mnist/transfer_purify"
    assert FIG_NAME == "transfer_purify_grid.pdf"


def test_jem_transfer_mask_requires_every_model_to_transfer():
    labels = np.array([0, 1, 2])
    clean = [np.array([0, 1, 2]), np.array([0, 1, 2])]
    adversarial = [np.array([8, 1, 7]), np.array([8, 9, 2])]
    assert _transfer_mask(clean, adversarial, labels).tolist() == [True, False, False]


def test_jem_transfer_grid_row_selection_and_model_parser():
    assert _select_rows(np.array([5, 0, 3, 0]), [0, 3, 9]) == {
        0: 1,
        3: 2,
        9: None,
    }
    assert _parse_models([r"a001=purified $\alpha=0.01$=outputs/a001/0"]) == [
        ("a001", r"purified $\alpha=0.01$", "outputs/a001/0")
    ]
