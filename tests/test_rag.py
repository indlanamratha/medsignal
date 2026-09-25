import pandas as pd

from medsignal.rag.labels import chunk_label, select_labels, split_words
from medsignal.rag.retrieve import detect_drug, drug_aliases, reciprocal_rank_fusion

LABEL = {
    "set_id": "abc",
    "effective_time": "20250101",
    "openfda": {"brand_name": ["OZEMPIC"], "generic_name": ["SEMAGLUTIDE"]},
    "boxed_warning": ["WARNING: RISK OF THYROID C-CELL TUMORS ..."],
    "adverse_reactions": [" ".join(f"word{i}" for i in range(700))],
}


def test_split_words_overlaps_and_covers_all_text():
    pieces = split_words(" ".join(f"w{i}" for i in range(700)), size=300, overlap=50)
    assert len(pieces) == 3
    assert pieces[0].split()[-50:] == pieces[1].split()[:50]      # 50-word overlap
    assert pieces[-1].split()[-1] == "w699"                        # nothing lost at the end


def test_chunks_keep_citation_metadata_and_never_mix_sections():
    rows = chunk_label(LABEL, "semaglutide")
    assert {r["section"] for r in rows} == {"Boxed Warning", "Adverse Reactions"}
    boxed = next(r for r in rows if r["section"] == "Boxed Warning")
    assert boxed["brand_name"] == "Ozempic"
    assert boxed["chunk_id"] == "abc:boxed_warning:0"
    assert boxed["context_text"].startswith("Ozempic (Semaglutide) - Boxed Warning:")


def test_select_labels_keeps_newest_per_brand_and_skips_combinations():
    labels = [
        {"effective_time": "20240101", "openfda": {"brand_name": ["GLUCOPHAGE"], "generic_name": ["METFORMIN"]}},
        {"effective_time": "20250101", "openfda": {"brand_name": ["GLUCOPHAGE"], "generic_name": ["METFORMIN"]}},
        {"effective_time": "20250101",
         "openfda": {"brand_name": ["JANUMET"], "generic_name": ["SITAGLIPTIN AND METFORMIN"]}},
    ]
    chosen = select_labels(labels, allow_combinations=False)
    assert len(chosen) == 1 and chosen[0]["effective_time"] == "20250101"


def test_reciprocal_rank_fusion_rewards_items_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])
    assert fused[0] == "b"          # 2nd and 1st beats 1st and absent
    assert set(fused) == {"a", "b", "c", "d"}


def test_detect_drug_by_brand_or_generic_name():
    chunks = pd.DataFrame({"drug_group": ["semaglutide", "tirzepatide"],
                           "brand_name": ["Wegovy", "Zepbound"],
                           "generic_name": ["Semaglutide", "Tirzepatide"]})
    aliases = drug_aliases(chunks)
    assert detect_drug("What is the boxed warning for Wegovy?", aliases) == "semaglutide"
    assert detect_drug("Does tirzepatide cause nausea?", aliases) == "tirzepatide"
    assert detect_drug("What is aspirin?", aliases) is None


def test_select_labels_detects_combinations_by_substance_count():
    labels = [{"effective_time": "20250101",
               "openfda": {"brand_name": ["SYNJARDY XR"], "generic_name": ["EMPAGLIFLOZIN"],
                           "substance_name": ["EMPAGLIFLOZIN", "METFORMIN HYDROCHLORIDE"]}}]
    assert select_labels(labels, allow_combinations=False) == []


def test_expand_query_adds_the_matching_label_section():
    from medsignal.rag.retrieve import expand_query
    assert expand_query("Who should not take Xenical?").endswith("contraindications")
    assert "dosage and administration" in expand_query("How often is Trulicity dosed?")
    assert expand_query("What is the boxed warning for Wegovy?") == "What is the boxed warning for Wegovy?"


def test_near_duplicate_passages_from_different_labels_are_dropped():
    from medsignal.rag.retrieve import drop_near_duplicates
    text = "Nursing mothers discontinue drug or nursing taking into consideration importance of drug"
    chunks = [{"brand_name": "Phentermine", "text": text},
              {"brand_name": "Lomaira", "text": text.replace("mothers", "mother")},
              {"brand_name": "Phentermine", "text": "Contraindications history of cardiovascular disease"}]
    kept = drop_near_duplicates(chunks)
    assert [c["brand_name"] for c in kept] == ["Phentermine", "Phentermine"]
    assert kept[1]["text"].startswith("Contraindications")
