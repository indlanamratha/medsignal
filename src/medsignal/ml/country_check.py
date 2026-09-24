"""Check how much the model depends on the report's country.

FDA rules mean reports from outside the US mostly reach FAERS only when they are
serious. So "is the report from the US?" is partly a reporting rule, not medicine.
This script measures performance with and without the country features, and on
US-only reports, where that shortcut is not available.
"""
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import duckdb
from medsignal.ml.features import build_features, feature_columns
from medsignal.ml.train import DATABASE, TEST_YEAR, TRAIN_END

COUNTRY_FEATURES = {"is_us", "country_missing"}


def lightgbm_model():
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
        random_state=42, verbose=-1)


def main() -> None:
    con = duckdb.connect(str(DATABASE), read_only=True)
    df = build_features(con, TRAIN_END)
    con.close()

    year = df["received_date"].dt.year
    train, test = df[year <= TRAIN_END], df[year == TEST_YEAR]
    test_us = test[test["is_us"] == 1]
    all_features = feature_columns(df)
    no_country = [f for f in all_features if f not in COUNTRY_FEATURES]

    print(f"Test 2025: {len(test):,} reports ({100 * test['is_serious'].mean():.1f}% serious); "
          f"US only: {len(test_us):,} ({100 * test_us['is_serious'].mean():.1f}% serious)")
    print(f"Non-US test reports that are serious: {100 * test.loc[test['is_us'] == 0, 'is_serious'].mean():.1f}%\n")

    rows = []
    for label, features in [("with country features", all_features), ("without country features", no_country)]:
        print(f"Training LightGBM {label}...")
        model = lightgbm_model().fit(train[features], train["is_serious"])
        for subset_name, subset in [("all 2025 reports", test), ("US-only 2025 reports", test_us)]:
            p = model.predict_proba(subset[features])[:, 1]
            rows.append({"model": label, "evaluated_on": subset_name,
                         "baseline_pr_auc": subset["is_serious"].mean(),
                         "pr_auc": average_precision_score(subset["is_serious"], p),
                         "roc_auc": roc_auc_score(subset["is_serious"], p)})

    print("\n===== Country check =====")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()