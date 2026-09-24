import numpy as np
import pandas as pd
import pytest

from medsignal.signals.disproportionality import compute_signals


def one_pair(a, n_drug, n_reaction, n_total):
    return compute_signals(pd.DataFrame([{
        "drug_group": "drug x", "reaction_pt": "REACTION Y",
        "a": a, "n_drug": n_drug, "n_reaction": n_reaction, "n_total": n_total,
    }])).iloc[0]


def test_hand_calculated_example():
    # a=20, b=80, c=100, d=9800
    row = one_pair(a=20, n_drug=100, n_reaction=120, n_total=10_000)
    assert (row.b, row.c, row.d) == (80, 100, 9800)
    assert row.prr == pytest.approx(19.8)        # (20/100) / (100/9900)
    assert row.ror == pytest.approx(24.5)        # (20*9800) / (80*100)
    assert row.chi2 == pytest.approx(285.3, abs=0.1)
    assert row.is_signal


def test_equal_reporting_rates_are_not_a_signal():
    # 10% of drug X reports and 10% of other reports mention the reaction.
    row = one_pair(a=10, n_drug=100, n_reaction=1000, n_total=10_000)
    assert row.prr == pytest.approx(1.0)
    assert not row.is_signal


def test_zero_cell_stays_finite():
    row = one_pair(a=5, n_drug=5, n_reaction=5, n_total=100)   # b = 0 and c = 0
    assert np.isfinite([row.prr, row.ror, row.ror_ci_low, row.chi2]).all()


def test_fewer_than_three_reports_is_never_a_signal():
    row = one_pair(a=2, n_drug=10, n_reaction=2, n_total=10_000)
    assert not row.is_signal