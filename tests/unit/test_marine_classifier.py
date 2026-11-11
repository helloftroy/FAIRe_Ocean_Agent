"""scripts/marine_audit.classify_sample: decides which samples curation deletes.

Cases from the 2026-11-11 dry run, where the first version would have
deleted marine samples ("coral metagenome" matched "oral metagenome",
"Acropora" matched "crop", marine hosts behind generic organism names).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from marine_audit import classify_sample  # noqa: E402

KEEP = [
    {"organism": "coral metagenome", "host": "Paramuricea sp."},
    {"organism": "Acropora cervicornis", "env_broad_scale": "coral reef"},
    {"organism": "plant metagenome", "host": "Zostera marina"},
    {"organism": "gut metagenome", "host": "Sardinella"},
    {"organism": "invertebrate metagenome", "host": "Aurelia aurita"},
    {"organism": "Vibrio harveyi", "isolation_source": "liver of diseased deep sea cage-cultured grouper"},
    {"organism": "metagenome", "host": "Acropora humilis", "env_broad_scale": "coral reef"},
    {"organism": "marine metagenome"},
    {"organism": "freshwater metagenome"},
]
DROP = [
    ({"organism": "Homo sapiens", "isolation_source": "sea water swimmers"}, "medical"),
    ({"organism": "human gut metagenome"}, "medical"),
    ({"organism": "human oral metagenome"}, "medical"),
    ({"organism": "Severe acute respiratory syndrome coronavirus 2"}, "medical"),
    ({"organism": "Danio rerio"}, "medical"),
    ({"organism": "soil metagenome"}, "terrestrial"),
    ({"organism": "Oryza sativa"}, "terrestrial"),
    ({"organism": "wastewater metagenome", "env_broad_scale": "sewage"}, "engineered"),
]


@pytest.mark.parametrize("fields", KEEP)
def test_marine_or_freshwater_evidence_is_kept(fields):
    assert classify_sample(fields) in ("marine", "freshwater")


@pytest.mark.parametrize("fields,category", DROP)
def test_obviously_unwanted_is_non_marine(fields, category):
    assert classify_sample(fields) == category


def test_no_evidence_is_unclassified():
    assert classify_sample({"organism": "metagenome"}) is None


def test_generic_host_associated_label_defers_to_the_host():
    from marine_audit import classify_sample

    assert classify_sample({"organism": "fish gut metagenome", "host": "Sparisoma viride"}) is None
    assert classify_sample({"organism": "gut metagenome", "host": "Lates calcarifer"}) is None
    assert classify_sample({"organism": "invertebrate metagenome", "host": "Exaiptasia diaphana"}) is None
    assert classify_sample({"organism": "gut metagenome", "host": "Homo sapiens"}) == "medical"
    assert classify_sample({"organism": "gut metagenome", "host": "Equus caballus"}) is None
    assert classify_sample({"organism": "gut metagenome", "host": "Gallus gallus"}) == "terrestrial"
    assert classify_sample({"organism": "Cyprinus carpio"}) == "other_host"   # named host still decides
