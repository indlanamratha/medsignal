"""Train and compare models that predict whether an adverse event report is serious.

Use case: safety teams receive thousands of reports. Flagging likely-serious
reports lets them review the most urgent ones first, so recall matters most.

Split by time, not randomly: train on 2020-2023, validate on 2024, test on 2025.
This mimics real use, where a model trained on past reports scores future ones.
"""
import duckdb
import joblib
import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medsignal.config import DATA_DIR, PROJECT_ROOT
from medsignal.ml.features import build_features, feature_columns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATABASE = DATA_DIR / "medsignal.duckdb"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "analytics" / "results"
TRAIN_END, VALID_YEAR, TEST_YEAR = 2023, 2024, 2025
TARGET_RECALL = 0.80
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT = "serious-report-triage"
REGISTERED_MODEL = "serious-report-triage"

MODEL_PARAMS = {
    "logistic_regression": {"max_iter": 2000, "class_weight": "balanced"},
    "xgboost": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "aucpr",
                "random_state": 42},
    "lightgbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8,
                 "subsample_freq": 1, "colsample_bytree": 0.8, "class_weight": "balanced",
                 "random_state": 42},
}


def make_model(name: str, pos_weight: float):
    params = MODEL_PARAMS[name]
    if name == "logistic_regression":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(**params))
    if name == "xgboost":
        return xgb.XGBClassifier(**params, scale_pos_weight=pos_weight, n_jobs=-1)
    return lgb.LGBMClassifier(**params, verbose=-1)


def threshold_for_recall(y_true, scores, target=TARGET_RECALL) -> float:
    """Highest threshold that still catches at least `target` of serious reports."""
    for t in np.sort(np.unique(np.round(scores, 3)))[::-1]:
        if recall_score(y_true, scores >= t) >= target:
            return float(t)
    return 0.0


def evaluate(name, y_valid, p_valid, y_test, p_test) -> dict:
    t = threshold_for_recall(y_valid, p_valid)
    pred = p_test >= t
    return {
        "model": name,
        "valid_pr_auc": average_precision_score(y_valid, p_valid),
        "test_roc_auc": roc_auc_score(y_test, p_test),
        "test_pr_auc": average_precision_score(y_test, p_test),
        "threshold": t,
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
    }


def tree_shap(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values from the libraries' built-in TreeSHAP (last column is the bias term)."""
    if isinstance(model, xgb.XGBClassifier):
        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    return model.predict(X, pred_contrib=True)[:, :-1]


def plot_importance(shap_values: np.ndarray, features: list[str], path) -> pd.Series:
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values()
    top = importance.tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#2a78d6", height=0.7)
    ax.set_xlabel("Mean |SHAP value| (impact on predicted log-odds)")
    ax.set_title("What drives the serious-report prediction (top 15 features)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return importance.sort_values(ascending=False)


def main() -> None:
    con = duckdb.connect(str(DATABASE), read_only=True)
    df = build_features(con, TRAIN_END)
    con.close()

    features = feature_columns(df)
    year = df["received_date"].dt.year
    train, valid, test = df[year <= TRAIN_END], df[year == VALID_YEAR], df[year == TEST_YEAR]
    X_tr, y_tr = train[features], train["is_serious"]
    X_va, y_va = valid[features], valid["is_serious"]
    X_te, y_te = test[features], test["is_serious"]

    print(f"Features: {len(features)}")
    for name, part in [("train 2020-2023", train), ("valid 2024", valid), ("test 2025", test)]:
        print(f"  {name:<16} {len(part):>8,} reports, {100 * part['is_serious'].mean():.1f}% serious")

    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    split_info = {"train_years": f"2020-{TRAIN_END}", "valid_year": VALID_YEAR, "test_year": TEST_YEAR,
                  "n_features": len(features), "n_train": len(train), "target_recall": TARGET_RECALL}

    base_rate = y_tr.mean()
    rows = [evaluate("baseline (always predicts training rate)",
                     y_va, np.full(len(y_va), base_rate), y_te, np.full(len(y_te), base_rate))]
    models, run_ids = {}, {}
    for name in MODEL_PARAMS:
        print(f"Training {name}...")
        with mlflow.start_run(run_name=name) as run:
            model = make_model(name, pos_weight).fit(X_tr, y_tr)
            row = evaluate(name, y_va, model.predict_proba(X_va)[:, 1],
                           y_te, model.predict_proba(X_te)[:, 1])
            mlflow.log_params({**MODEL_PARAMS[name], **split_info, "model_type": name})
            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "model"})
        models[name], run_ids[name] = model, run.info.run_id
        rows.append(row)

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print("\n===== Model comparison (threshold chosen on 2024 for 80% recall) =====")
    print(results.round(3).to_string(index=False))

    best_name = results.iloc[1:].sort_values("valid_pr_auc", ascending=False).iloc[0]["model"]
    best = models[best_name]
    threshold = float(results.set_index("model").loc[best_name, "threshold"])
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": best, "features": features, "threshold": threshold,
                 "model_name": best_name}, MODELS_DIR / "serious_model.joblib")
    print(f"\nBest model on validation PR-AUC: {best_name} (saved to models/serious_model.joblib)")

    with mlflow.start_run(run_id=run_ids[best_name]):
        info = mlflow.sklearn.log_model(best, name="model", input_example=X_te.head(5),
                                        serialization_format="cloudpickle",
                                        registered_model_name=REGISTERED_MODEL)
    mlflow.MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL, "champion", info.registered_model_version)
    print(f"Registered {REGISTERED_MODEL} version {info.registered_model_version} as 'champion' in MLflow")

    if best_name in ("xgboost", "lightgbm"):
        sample = X_te.sample(min(5000, len(X_te)), random_state=42)
        REPORTS_DIR.mkdir(exist_ok=True)
        importance = plot_importance(tree_shap(best, sample), features, REPORTS_DIR / "shap_importance.png")
        print("\n===== Top 15 features by mean |SHAP| (saved chart to reports/shap_importance.png) =====")
        print(importance.head(15).round(3).to_string())
        mlflow.log_artifact(str(REPORTS_DIR / "shap_importance.png"), run_id=run_ids[best_name])


if __name__ == "__main__":
    main()
