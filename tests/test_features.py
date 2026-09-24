"""Training-serving skew test: the API's feature builder must match the training one."""
import duckdb
import numpy as np
import pandas as pd

from medsignal.ml.features import build_features, feature_columns, features_from_report


def tiny_warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("""
        create table stg_reports as select * from (values
            ('r1', date '2021-03-01', true,  64.0, 90.0, 'F', 'Physician', 'US'),
            ('r2', date '2022-06-01', false, null, null, 'M', 'Consumer',  null),
            ('r3', date '2025-01-15', true,  45.0, null, 'U', 'Other health professional', 'GB')
        ) t(report_id, received_date, is_serious, age_years, weight_kg, sex, reporter_type_label, country)
    """)
    con.execute("""
        create table fct_report_drug as select * from (values
            ('r1', 'semaglutide'), ('r1', 'metformin'), ('r2', 'tirzepatide'), ('r3', 'semaglutide')
        ) t(report_id, drug_group)
    """)
    con.execute("""
        create table stg_report_drugs as select * from (values
            ('r1', 'Suspect'), ('r1', 'Suspect'), ('r1', 'Concomitant'),
            ('r2', 'Suspect'), ('r3', 'Suspect'), ('r3', 'Concomitant'), ('r3', 'Concomitant')
        ) t(report_id, drug_role)
    """)
    con.execute("""
        create table stg_report_reactions as select * from (values
            ('r1', 'PANCREATITIS'), ('r1', 'NAUSEA'), ('r1', 'DEATH'),
            ('r2', 'NAUSEA'), ('r2', 'INJECTION SITE PAIN'),
            ('r3', 'PANCREATITIS'), ('r3', 'HOSPITALISATION')
        ) t(report_id, reaction_pt)
    """)
    return con


def test_api_features_match_training_features():
    df = build_features(tiny_warehouse(), train_end_year=2023)
    features = feature_columns(df)
    requests = {
        "r1": {"age_years": 64, "weight_kg": 90, "sex": "F", "reporter_type": "Physician",
               "country": "US", "suspect_drugs": ["semaglutide", "metformin"],
               "n_suspect_drugs": 2, "n_concomitant_drugs": 1,
               "reactions": ["PANCREATITIS", "nausea", "DEATH"]},
        "r2": {"age_years": None, "weight_kg": None, "sex": "M", "reporter_type": "Consumer",
               "country": None, "suspect_drugs": ["tirzepatide"], "n_concomitant_drugs": 0,
               "reactions": ["NAUSEA", "INJECTION SITE PAIN"]},
    }
    for report_id, request in requests.items():
        trained = df.loc[df["report_id"] == report_id, features].astype(float).reset_index(drop=True)
        served = features_from_report(request, features)
        pd.testing.assert_frame_equal(served, trained, check_dtype=False)


def test_outcome_terms_never_become_features():
    df = build_features(tiny_warehouse(), train_end_year=2023)
    assert not any("death" in c or "hospital" in c for c in feature_columns(df))
    row = df.loc[df["report_id"] == "r1"].iloc[0]
    assert row["n_reactions"] == 2          # DEATH was dropped
    assert np.isnan(df.loc[df["report_id"] == "r2", "age_years"].iloc[0])
