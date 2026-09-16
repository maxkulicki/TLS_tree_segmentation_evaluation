"""Tests for the comparison against the published methods.

Expected values are derived in the comments, not copied from a run. The point
of the paired test is that it can disagree with the ranking of the means, so
the fixtures below are built to make it do exactly that.
"""

import numpy as np
import pandas as pd
import pytest

from tlseval.report import PRIMARY, compare_published, holm


# ── Holm-Bonferroni ──────────────────────────────────────────────────────────

def test_holm_step_down():
    # Sorted ascending: 0.01, 0.02, 0.04. m = 3.
    #   0.01 * 3 = 0.03
    #   0.02 * 2 = 0.04
    #   0.04 * 1 = 0.04, but the running maximum from the previous step is
    #              already 0.04, so it stays 0.04 (adjusted p is monotone).
    # Input order is 0.04, 0.01, 0.02, and the output must follow it.
    assert np.allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])


def test_holm_caps_at_one():
    # 0.5 * 2 = 1.0; 0.6 * 1 = 0.6, raised to 1.0 by the running maximum.
    assert np.allclose(holm([0.5, 0.6]), [1.0, 1.0])


def test_holm_ignores_nan_in_family_size():
    # One real p-value beside a NaN: family size is 1, so it is unadjusted.
    out = holm([0.03, np.nan])
    assert np.isclose(out[0], 0.03) and np.isnan(out[1])


# ── comparison against the published table ───────────────────────────────────

@pytest.fixture
def published(tmp_path):
    """Two rival methods over 40 plots, built so the means and the paired test
    disagree.

    `steady` beats `erratic` by 0.02 on every single plot. `erratic` is handed
    one plot where it wins by 1.0, which is enough to pull its mean above
    `steady`'s (0.02 * 39 = 0.78 < 1.0) while leaving 39 of 40 paired
    differences pointing the other way.
    """
    n = 40
    base = np.linspace(0.30, 0.70, n)
    steady = base + 0.02
    erratic = base.copy()
    erratic[0] = base[0] + 1.0
    df = pd.DataFrame({
        "source_file": [f"Plot_{i:02d}.laz" for i in range(n)],
        "steady_mean_iou": steady,
        "erratic_mean_iou": erratic,
    })
    path = tmp_path / "published.csv"
    df.to_csv(path, index=False)
    return path, base


def test_comparison_reports_means_plot_wins_and_adjusted_p_values(published):
    path, base = published
    # "Your method" is `steady` replayed through the comparison.
    mine = pd.DataFrame({"plot": [f"Plot_{i:02d}.laz" for i in range(len(base))],
                         PRIMARY: base + 0.02})
    out = compare_published(mine, path).set_index("method")

    # erratic's mean is higher because of one large outlier ...
    assert out.loc["erratic", "mean_iou"] > out.loc["your method", "mean_iou"]
    # ... although your method is higher on 39 of the 40 shared plots.
    assert out.loc["erratic", "you_win"] == "39/40"
    assert out.loc["erratic", "p_holm"] < 0.05
    assert "verdict" not in out.columns


def test_identical_method_has_undefined_p_value(published):
    path, base = published
    mine = pd.DataFrame({"plot": [f"Plot_{i:02d}.laz" for i in range(len(base))],
                         PRIMARY: base + 0.02})
    out = compare_published(mine, path).set_index("method")
    # Every paired difference against `steady` is exactly zero. The signed-rank
    # test is undefined there.
    assert np.isnan(out.loc["steady", "p_holm"])


def test_only_shared_plots_are_compared(published):
    path, base = published
    # Half the plots, and one name that is in neither table.
    keep = [f"Plot_{i:02d}.laz" for i in range(0, 40, 2)] + ["Plot_99.laz"]
    mine = pd.DataFrame({"plot": keep, PRIMARY: np.full(len(keep), 0.5)})
    out = compare_published(mine, path).set_index("method")
    assert out.loc["steady", "n_plots"] == 20
    assert out.loc["your method", "n_plots"] == 20
