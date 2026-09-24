import numpy as np
import pandas as pd

from medsignal.monitoring.drift import drift_table


def test_detects_shifted_column_and_ignores_stable_one():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"age_years": rng.normal(50, 10, 3000),
                              "is_us": rng.integers(0, 2, 3000)})
    current = pd.DataFrame({"age_years": rng.normal(65, 10, 3000),      # patients got older
                            "is_us": rng.integers(0, 2, 3000)})         # unchanged
    table, _ = drift_table(reference, current)
    drifted = dict(zip(table["column"], table["drifted"]))
    assert drifted["age_years"]
    assert not drifted["is_us"]
