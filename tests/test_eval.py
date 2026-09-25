from medsignal.eval.run_eval import fact_recall, hit_rank, matches_truth, normalize, numbers_in


def test_normalize_handles_unicode_dashes():
    assert normalize("Thyroid C‑cell tumors") == "thyroid c-cell tumors"


def test_fact_recall_with_alternatives():
    answer = "Common reactions include nausea and diarrhoea."
    assert fact_recall(answer, ["nausea", "diarrhea|diarrhoea"]) == 1.0
    assert fact_recall(answer, ["nausea", "constipation"]) == 0.5


def test_numbers_and_truth_matching():
    assert numbers_in("There were 116,333 reports (20.5% serious).") == [116333.0, 20.5]
    assert matches_truth("116,333 reports", 116333)
    assert matches_truth("about 20.5% were serious", 20.5)
    assert not matches_truth("116,000 reports", 116333)
    assert matches_truth("Tirzepatide had the most reports.", "tirzepatide")


def test_hit_rank_needs_right_drug_and_section():
    chunks = [{"drug_group": "tirzepatide", "section": "Boxed Warning"},
              {"drug_group": "semaglutide", "section": "Adverse Reactions"},
              {"drug_group": "semaglutide", "section": "Boxed Warning"}]
    assert hit_rank(chunks, "semaglutide", ["Boxed Warning"]) == 3
    assert hit_rank(chunks, "semaglutide", []) == 2
    assert hit_rank(chunks, "metformin", ["Boxed Warning"]) is None
