#!/usr/bin/env python3
"""Gap-2 audit: candidate mapping rules for unmapped sample-level raw names.

Reads mapping_inventory.py's inventory.csv. For every sample-level raw
field name with no rule (and >= MIN_ENTITIES samples), proposes a target:

  same_name   normalizes (case, spaces, _ - / parentheses, unit suffix) to a
              raw name that already has a rule -> copy that rule (same
              transform). Safest kind.
  ena_alias   a curated ENA / INSDC display spelling (CURATED below) -> copy
              the rule of the named raw name.
  mixs        normalizes to a MIxS slot name / title / alias whose MIXS id is
              a FAIRe term's exact mapping -> new rule to that FAIRe term
              (transform must be chosen when it is added).
  faire       normalizes to a FAIRe sampleMetadata term name -> same.

and the gain: samples with a present value under this name that have no
fact under any name already mapped to the same FAIRe target(s).

Usage: python scripts/faire_strict/mapping_candidates.py DB INVENTORY_CSV OUT_CSV
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MIN_ENTITIES = 100
csv.field_size_limit(10**8)

# ENA checklist / INSDC display spellings -> an existing raw name whose rule
# is copied. Only names whose meaning is unambiguous.
CURATED = {
    "geographic location (latitude)": "latitude",
    "geographic location (longitude)": "longitude",
    "geographic location (country and/or sea)": "geo_loc_name",
    "geographic location (country and/or sea,region)": "geo_loc_name",
    "geographic location (country and/or sea, region)": "geo_loc_name",
    "geographic location (countryand/orsea,region)": "geo_loc_name",
    "country": "geo_loc_name",
    "geographic location (depth)": "depth",
    "lat lon": "lat_lon",
    "latitude and longitude": "lat_lon",
    "collection_timestamp": "collection_date",
    "collection_datetime": "collection_date",
    "sampling event, date-time, start": "collection_date",
    "sampling event, latitude, start": "latitude",
    "sampling event, longitude, start": "longitude",
    "depth_m": "depth",
    "water depth": "tot_depth_water_col",
    "total depth of water column": "tot_depth_water_col",
}
_UNIT = re.compile(r"\s*[\(\[]\s*(m|meters?|metres?|°?c|deg ?c|celsius|psu|ppt|‰|µm|um|mm|cm|ml|l|mg/l|µmol/l|umol/l|%)\s*[\)\]]\s*$", re.I)
_MISSING = re.compile(r"^\s*(missing|not applicable|not collected|not provided|restricted access|na|n/a|none|unknown|null|-)\b", re.I)


def norm(name: str) -> str:
    name = _UNIT.sub("", name or "")
    name = re.sub(r"_(m|meters|c|psu)$", "", name.lower())
    return re.sub(r"[^a-z0-9]", "", name)


def main() -> None:
    db_path, inv_path, out_path = sys.argv[1:4]
    inv = [r for r in csv.DictReader(open(inv_path)) if r["entity_level"] == "sample"]
    mapped = {r["fact_type"]: set(r["targets"].split()) for r in inv if r["targets"]}
    by_norm = collections.defaultdict(set)
    for name in mapped:
        by_norm[norm(name)].add(name)

    terms = json.load(open(REPO / "standards/compiled/faire_registry.json"))
    faire = {t["upstream_field_name"] for t in terms if "sampleMetadata" in (t.get("data_type") or [])}
    mixs_to_faire = {}
    for t in terms:
        if "sampleMetadata" in (t.get("data_type") or []):
            for m in t.get("exact_mappings") or []:
                if m.lower().startswith("mixs:"):
                    mixs_to_faire[m.upper()] = t["upstream_field_name"]
    slots = yaml.safe_load(open(REPO / "reference/mixs/mixs.yaml"))["slots"]
    mixs_norm = {}
    for slot, d in slots.items():
        target = mixs_to_faire.get((d.get("slot_uri") or "").upper())
        if not target:
            continue
        for label in [slot, d.get("title") or "", *(d.get("aliases") or [])]:
            if label:
                mixs_norm.setdefault(norm(label), target)
    faire_norm = {norm(t): t for t in faire}

    cands = []
    for r in inv:
        name = r["fact_type"]
        if r["targets"] or int(r["entities"]) < MIN_ENTITIES:
            continue
        n = norm(name)
        kind = clone = target = None
        if by_norm.get(n):
            kind, clone = "same_name", sorted(by_norm[n])[0]
        elif name.lower() in CURATED and CURATED[name.lower()] in mapped:
            kind, clone = "ena_alias", CURATED[name.lower()]
        elif n in mixs_norm:
            kind, target = "mixs", f"sampleMetadata.{mixs_norm[n]}"
        elif n in faire_norm:
            kind, target = "faire", f"sampleMetadata.{faire_norm[n]}"
        if not kind:
            continue
        targets = mapped[clone] if clone else {target}
        cands.append({"name": name, "kind": kind, "clone_rules_of": clone or "", "targets": targets,
                      "entities": int(r["entities"]), "studies": int(r["studies"]), "examples": r["examples"]})
    print(f"{len(cands)} candidates", flush=True)

    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    def entity_set(names, require_value=True):
        out = set()
        names = sorted(names)
        for i in range(0, len(names), 200):
            chunk = names[i : i + 200]
            q = (f"select entity_id, raw_value from raw_facts indexed by ix_raw_facts_fact_type_candidate "
                 f"where fact_type_candidate in ({','.join('?' * len(chunk))}) and entity_level='sample' and entity_id is not null")
            for e, v in db.execute(q, chunk):
                if not require_value or (v and not _MISSING.match(v)):
                    out.add(e)
        return out

    covered: dict[str, set] = {}
    for c in cands:
        for t in c["targets"]:
            if t not in covered:
                covered[t] = entity_set([m for m, ts in mapped.items() if t in ts])
    rows = []
    for c in cands:
        have = entity_set([c["name"]])
        already = set.union(*(covered[t] for t in c["targets"]))
        rows.append({**{k: v for k, v in c.items() if k != "targets"}, "targets": " ".join(sorted(c["targets"])),
                     "with_value": len(have), "gain": len(have - already)})
    rows.sort(key=lambda r: -r["gain"])
    with open(out_path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["name", "kind", "clone_rules_of", "targets", "gain", "with_value",
                                               "entities", "studies", "examples"])
        w.writeheader()
        w.writerows(rows)
    for r in rows[:40]:
        print(f"{r['gain']:>8,} {r['kind']:9s} {r['name'][:45]:45s} -> {r['clone_rules_of'] or r['targets']}")


if __name__ == "__main__":
    main()
