#!/usr/bin/env python3
"""Marine-relevance audit of every study (2026-11-10).

studies.marine_relevance_status is 'unknown' for all studies: no marine
screen was ever run. This classifies each study from evidence already in the
database; it changes nothing.

Per sample entity, one category from (first hit wins):
  organism (BioSample taxonomy name, e.g. "marine metagenome",
  "human gut metagenome", "Homo sapiens"), else host, else
  isolation_source / env_broad_scale text.
Categories: medical (human / lab animal incl. zebrafish / clinical pathogen), marine,
freshwater, terrestrial, engineered, other_host; "ambiguous" values
("metagenome", "sediment metagenome", "aquatic metagenome", ...) are not
counted.

Per study:
  marine        >= 50 % of classified samples marine
  mixed         10-50 % marine (keep; review)
  non_marine:X  < 10 % marine, X = the largest other category
  title_marine / title_non_marine:X / unknown
                no classified samples: decided by title + description
                keywords; unknown = no signal (review)
A study whose samples say non-marine but whose title has a marine keyword
is reported as review_conflict instead of non_marine.

Usage: python scripts/marine_audit.py DB OUT_DIR
"""
from __future__ import annotations

import collections
import csv
import re
import sqlite3
import sys
from pathlib import Path

ORDER = ("medical", "marine", "freshwater", "terrestrial", "engineered", "other_host")
PATTERNS = {
    "medical": r"homo sapiens|\bhuman|patient|clinical|hospital|sars-cov|coronavirus|influenza|hiv\b|"
    r"plasmodium|tuberculosis|streptococcus|staphylococcus|klebsiella|acinetobacter|clostridioides|"
    r"campylobacter|listeria|salmonella|escherichia coli|pseudomonas aeruginosa|enterococcus|neisseria|"
    r"helicobacter|mus musculus|\bmouse\b|\bmice\b|murine|rattus|\brats?\b|danio rerio|zebrafish|vaginal|oral metagenome|"
    r"skin metagenome|lung metagenome|\bfeces|fecal|faecal|stool|saliva|sputum|urine|blood|"
    r"cancer|tumou?r|infant|child|covid|disease|infection|obes|diabet",
    "marine": r"marine|sea ?water|ocean|coral|sponge|porifera|estuar|salt ?marsh|mangrove|hydrothermal|"
    r"seagrass|kelp|intertidal|coastal|plankton|symbiodin|whale|dolphin|cetacean|brackish|\breef|"
    r"deep[- ]sea|cold seep|beach|tidal|seaweed|macroalga|oyster|mussel|shrimp|\bcrab|tunicate|ascidian|"
    r"cnidaria|jellyfish|anemone|bivalve|cephalopod|squid|octopus|polychaete|seabird|penguin|\btuna|"
    r"\bsea\b|\bseas\b|\bbay\b|gulf|fjord|lagoon|atlantic|pacific|arctic|antarctic|southern ocean|"
    r"mediterranean|baltic|benthic|pelagic|abyss|hadal|trench|sea ice|diatom|dinoflagellate|"
    r"copepod|krill|salmon(?!ella)|\bcod\b|herring|sardine|anchov|seal\b|sea lion|turtle|shark|ray\b",
    "freshwater": r"freshwater|fresh water|\blake|\brivers?\b|stream|\bpond|groundwater|aquifer|"
    r"drinking water|tap water|wetland|hot spring|glacier|snow|reservoir|daphnia",
    "terrestrial": r"\bsoil|rhizosphere|\broots?\b|plant|phyllosphere|arabidopsis|oryza|zea mays|"
    r"triticum|hordeum|glycine max|solanum|compost|terrestrial|insect|mosquito|drosophila|bovine|"
    r"bos taurus|sus scrofa|\bpigs?\b|porcine|chicken|gallus|canis|cattle|\bcow|sheep|ovis|goat|"
    r"honey ?bee|apis mellifera|termite|\bleaf|leaves|forest|desert|permafrost|caenorhabditis|"
    r"saccharomyces|grassland|agricultur|crop|maize|wheat|rice\b|cave",
    "engineered": r"wastewater|sewage|activated sludge|bioreactor|anaerobic digest|digester|\bfood|"
    r"indoor|air metagenome|built environment|dairy|cheese|fermented|brewery|wine|kimchi|landfill|"
    r"mine drainage|oil reservoir|hydrocarbon reservoir",
    "other_host": r"gut metagenome|insect metagenome|invertebrate metagenome|animal|"
    r"cyprinus|carassius|\bcarp\b|tilapia|oreochromis|oncorhynchus mykiss|rainbow trout",
}
RX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
AMBIGUOUS = re.compile(
    r"^(metagenome|sediment metagenome|aquatic metagenome|water metagenome|viral metagenome|"
    r"bacterium|uncultured.*|.*bacterium|biofilm metagenome|microbial mat metagenome|"
    r"subsurface metagenome|eukaryote metagenome|fungus metagenome|synthetic metagenome|"
    r"mixed sample|unidentified|environmental samples?|not applicable|missing.*|unknown)$",
    re.I,
)


def categorize(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip()
    if not t or AMBIGUOUS.match(t):
        return None
    for cat in ORDER:
        if RX[cat].search(t):
            return cat
    return None


def text_category(text: str) -> tuple[str | None, dict[str, int]]:
    hits = {cat: len(RX[cat].findall(text)) for cat in ORDER}
    if hits["marine"]:
        return "marine", hits
    best = max((c for c in ORDER if c != "marine"), key=lambda c: hits[c])
    return (best if hits[best] else None), hits


def main() -> None:
    db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    # entity -> (study, {field: value}) for the three evidence fields
    fields = ("organism", "host", "isolation_source", "env_broad_scale")
    ev: dict[str, dict[str, str]] = collections.defaultdict(dict)
    ent_study: dict[str, str] = {}
    q = (
        "select entity_id, study_id, fact_type_candidate, raw_value from raw_facts "
        "indexed by ix_raw_facts_fact_type_candidate "
        f"where fact_type_candidate in ({','.join('?' * len(fields))}) and entity_level='sample' and entity_id is not null"
    )
    for eid, sid, field, value in db.execute(q, fields):
        if field not in ev[eid]:
            ev[eid][field] = value or ""
        ent_study.setdefault(eid, sid)
    print(f"{len(ev):,} sample entities with evidence", flush=True)

    per_study: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    organisms: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for eid, f in ev.items():
        cat = None
        for field in fields:
            cat = categorize(f.get(field))
            if cat:
                break
        sid = ent_study[eid]
        per_study[sid][cat or "unclassified"] += 1
        if f.get("organism"):
            organisms[sid][f["organism"].lower()] += 1

    n_samples = collections.Counter(
        sid for (sid,) in db.execute("select study_id from entities where entity_level='sample'")
    )

    rows = []
    for sid, title, desc in db.execute("select study_id, title, description from studies"):
        c = per_study.get(sid, collections.Counter())
        classified = sum(v for k, v in c.items() if k != "unclassified")
        tcat, hits = text_category(f"{title or ''} {desc or ''}")
        title_marine = bool(RX["marine"].search(title or ""))
        if classified:
            m = c["marine"] / classified
            if m >= 0.5:
                cls = "marine"
            elif m >= 0.1:
                cls = "mixed"
            else:
                other = max((k for k in ORDER if k != "marine"), key=lambda k: c[k])
                cls = "review_conflict" if title_marine else f"non_marine:{other}"
        else:
            m = None
            cls = "title_marine" if tcat == "marine" else (f"title_non_marine:{tcat}" if tcat else "unknown")
        rows.append({
            "study_id": sid,
            "class": cls,
            "pct_marine_samples": "" if m is None else round(100 * m, 1),
            "classified_samples": classified,
            "sample_entities": n_samples.get(sid, 0),
            **{f"n_{k}": c[k] for k in ORDER},
            "top_organisms": " | ".join(f"{o} ({n})" for o, n in organisms[sid].most_common(3)),
            "title": (title or "")[:200],
        })

    with (out / "marine_audit.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["class"], -r["sample_entities"])))

    summary = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        key = r["class"].split(":")[0] + (":" + r["class"].split(":")[1] if ":" in r["class"] else "")
        summary[key][0] += 1
        summary[key][1] += r["sample_entities"]
    with (out / "marine_audit_summary.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["class", "studies", "sample_entities"])
        for k, (s, n) in sorted(summary.items(), key=lambda kv: -kv[1][0]):
            w.writerow([k, s, n])
            print(f"{k:32s} {s:>6,} studies {n:>10,} samples")


if __name__ == "__main__":
    main()
