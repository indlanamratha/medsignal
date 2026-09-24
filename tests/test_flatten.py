from datetime import date

from medsignal.ingestion.flatten import age_in_years, extract, parse_date


def test_age_in_months_converts_to_years():
    assert age_in_years("18", "802") == 1.5


def test_impossible_age_is_dropped():
    assert age_in_years("500", "801") is None


def test_parse_date():
    assert parse_date("20250131") == date(2025, 1, 31)
    assert parse_date("not a date") is None


def test_extract_splits_one_report_into_three_parts():
    raw = {
        "safetyreportid": "1",
        "safetyreportversion": "2",
        "receivedate": "20250101",
        "serious": "1",
        "seriousnessdeath": "1",
        "patient": {
            "patientonsetage": "60",
            "patientonsetageunit": "801",
            "patientsex": "2",
            "drug": [{"medicinalproduct": " ozempic ", "drugcharacterization": "1",
                      "openfda": {"generic_name": ["SEMAGLUTIDE"]}}],
            "reaction": [{"reactionmeddrapt": "Nausea"},
                         {"reactionmeddrapt": "Vomiting"}],
        },
    }
    report, drugs, reactions = extract(raw)
    assert report["is_serious"] and report["died"]
    assert report["age_years"] == 60
    assert drugs[0]["medicinal_product"] == "OZEMPIC"
    assert drugs[0]["generic_name"] == "SEMAGLUTIDE"
    assert [r["reaction_pt"] for r in reactions] == ["NAUSEA", "VOMITING"]