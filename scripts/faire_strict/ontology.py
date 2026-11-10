"""ENVO and OBI lookups for the strict-FAIRe correction (rules R19, R20).

Both ontologies are read from the pinned, gzipped copies in
reference/ontologies/ (versions and checksums in VERSIONS.md there).
Nothing is fetched at run time.
"""

from __future__ import annotations

import gzip
import csv
import difflib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ONTOLOGY_DIR = REPO / "reference/ontologies"
ENVO_FILE = ONTOLOGY_DIR / "envo_2026-10-23.json.gz"
OBI_FILE = ONTOLOGY_DIR / "obi_2026-07-27.owl.gz"

_OBO = "http://purl.obolibrary.org/obo/"

# FAIRe's recommended branch for each env_* term (checklist "vocabulary").
ENVO_BRANCH = {
    "env_broad_scale": "ENVO:00000428",  # biome
    "env_local_scale": "ENVO:01000281",  # layer
    "env_medium": "ENVO:00010483",  # environmental material
}
OBI_DNA_SEQUENCER = "OBI_0400103"


def normalise_label(text: str) -> str:
    """Lower case, underscores as spaces, single spaces, no edge punctuation."""
    text = text.replace("_", " ").replace(" ", " ")
    text = re.sub(r"\s+", " ", text).strip().strip(".,;:").strip()
    return text.lower()


@dataclass
class EnvoTerm:
    curie: str  # ENVO:00000447
    label: str
    deprecated: bool
    exact_synonyms: tuple[str, ...]
    parents: tuple[str, ...]
    replaced_by: str | None = None  # IAO:0100001 "term replaced by", obsolete classes only
    other_synonyms: tuple[str, ...] = ()  # narrow / broad / related: only to confirm a label next to its ID


@dataclass
class Envo:
    version: str
    terms: dict[str, EnvoTerm]
    by_label: dict[str, list[str]] = field(default_factory=dict)
    by_synonym: dict[str, list[str]] = field(default_factory=dict)
    _ancestors: dict[str, frozenset[str]] = field(default_factory=dict)

    def ancestors(self, curie: str) -> frozenset[str]:
        if curie not in self._ancestors:
            seen: set[str] = set()
            stack = list(self.terms[curie].parents) if curie in self.terms else []
            while stack:
                parent = stack.pop()
                if parent in seen:
                    continue
                seen.add(parent)
                if parent in self.terms:
                    stack.extend(self.terms[parent].parents)
            self._ancestors[curie] = frozenset(seen)
        return self._ancestors[curie]

    def in_branch(self, curie: str, root: str) -> bool:
        return curie == root or root in self.ancestors(curie)

    def lookup_id(self, digits: str) -> EnvoTerm | None:
        """Exact ID first; then the same number zero-padded to 8 digits."""
        for candidate in (digits, digits.zfill(8)):
            term = self.terms.get(f"ENVO:{candidate}")
            if term:
                return term
        return None


def _curie(iri: str) -> str | None:
    if not iri.startswith(_OBO):
        return None
    local = iri[len(_OBO):]
    prefix, _, number = local.partition("_")
    return f"{prefix}:{number}" if number else None


def load_envo(path: Path = ENVO_FILE) -> Envo:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        graph = json.load(handle)["graphs"][0]
    terms: dict[str, EnvoTerm] = {}
    parents: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        if edge.get("pred") != "is_a":
            continue
        sub, obj = _curie(edge["sub"]), _curie(edge["obj"])
        if sub and obj:
            parents.setdefault(sub, []).append(obj)
    for node in graph["nodes"]:
        curie = _curie(node.get("id", ""))
        if not curie or not curie.startswith("ENVO:") or node.get("type") != "CLASS" or not node.get("lbl"):
            continue
        meta = node.get("meta", {})
        synonyms = tuple(
            s["val"] for s in meta.get("synonyms", []) if s.get("pred") == "hasExactSynonym" and s.get("val")
        )
        others = tuple(
            s["val"] for s in meta.get("synonyms", []) if s.get("pred") != "hasExactSynonym" and s.get("val")
        )
        replaced = [
            b["val"] for b in meta.get("basicPropertyValues", []) if b.get("pred") == _OBO + "IAO_0100001"
        ]
        replaced_by = None
        if len(replaced) == 1:
            value = replaced[0]
            replaced_by = _curie(value) if value.startswith("http") else value.replace("_", ":", 1)
        terms[curie] = EnvoTerm(
            curie=curie,
            label=node["lbl"],
            deprecated=bool(meta.get("deprecated")),
            exact_synonyms=synonyms,
            parents=tuple(parents.get(curie, ())),
            replaced_by=replaced_by,
            other_synonyms=others,
        )
    envo = Envo(version=graph["meta"].get("version", ""), terms=terms)
    for term in terms.values():
        if term.deprecated:
            continue  # obsolete classes are never chosen from a label
        envo.by_label.setdefault(normalise_label(term.label), []).append(term.curie)
        for synonym in term.exact_synonyms:
            envo.by_synonym.setdefault(normalise_label(synonym), []).append(term.curie)
    return envo


@dataclass
class ObiSequencer:
    obi_id: str  # OBI_0002630 (FAIRe writes OBI ids with an underscore)
    label: str


def load_obi_sequencers(path: Path = OBI_FILE) -> dict[str, ObiSequencer]:
    """All named subclasses of DNA sequencer (OBI_0400103), by OBI id."""
    ns = {
        "owl": "http://www.w3.org/2002/07/owl#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    }
    about = f"{{{ns['rdf']}}}about"
    resource = f"{{{ns['rdf']}}}resource"
    with gzip.open(path, "rb") as handle:
        root = ET.parse(handle).getroot()
    labels: dict[str, str] = {}
    parents: dict[str, list[str]] = {}
    deprecated: set[str] = set()
    for cls in root.findall("owl:Class", ns):
        iri = cls.get(about, "")
        if not iri.startswith(_OBO):
            continue
        local = iri[len(_OBO):]
        label = cls.find("rdfs:label", ns)
        if label is not None and label.text:
            labels[local] = label.text.strip()
        for sub in cls.findall("rdfs:subClassOf", ns):
            if sub.get(resource, "").startswith(_OBO):
                parents.setdefault(local, []).append(sub.get(resource)[len(_OBO):])
        dep = cls.find("owl:deprecated", ns)
        if dep is not None and (dep.text or "").strip().lower() == "true":
            deprecated.add(local)
    children: dict[str, list[str]] = {}
    for child, ps in parents.items():
        for parent in ps:
            children.setdefault(parent, []).append(child)
    found: dict[str, ObiSequencer] = {}
    stack = list(children.get(OBI_DNA_SEQUENCER, ()))
    while stack:
        local = stack.pop()
        if local in found:
            continue
        if local in labels and local not in deprecated:
            found[local] = ObiSequencer(local, labels[local])
        stack.extend(children.get(local, ()))
    return found


# ---------------------------------------------------------------------------
# R19: env_broad_scale / env_local_scale / env_medium -> "label [ENVO:id]"

_ENVO_ID = r"(?:ENVO|ENV)\s*[:_\-–]?\s*(\d{6,8})"
_ID_TOKEN = re.compile(rf"^{_ENVO_ID}$", re.IGNORECASE)
_LABEL_WITH_ID = re.compile(rf"^(.+?)\s*[\[(]\s*{_ENVO_ID}\s*[\])]$", re.IGNORECASE)
_FOREIGN_ID = re.compile(r"^(?:.+?\s*[\[(]\s*)?([A-Za-z]+)\s*[:_]\s*\d{4,}\s*[\])]?$")
_ENVO_PREFIXED_LABEL = re.compile(r"^envo\s*:\s*(?=\D)", re.IGNORECASE)


def _faire_envo(term: EnvoTerm) -> str:
    return f"{term.label} [{term.curie}]"


def _names(term: EnvoTerm) -> set[str]:
    names = {term.label, *term.exact_synonyms, *term.other_synonyms}
    if term.deprecated and term.label.startswith("obsolete "):
        names.add(term.label[len("obsolete "):])
    return {normalise_label(n) for n in names}


def _label_matches(label: str, term: EnvoTerm) -> bool:
    """The label next to an ID is the ID's label or any of its synonyms."""
    return normalise_label(label) in _names(term)


_WORD = re.compile(r"[a-z0-9]+")


def _label_close(label: str, term: EnvoTerm) -> bool:
    """A typo (similarity >= 0.9) or one name's words contain the other's.

    Only used when the cell also gives the ID, so the ID decides the term
    and the label is just a cross-check against swapped IDs.
    """
    norm = normalise_label(label)
    words = set(_WORD.findall(norm))
    for name in _names(term):
        if difflib.SequenceMatcher(None, norm, name).ratio() >= 0.9:
            return True
        other = set(_WORD.findall(name))
        if words and other and (words <= other or other <= words):
            return True
    return False


def _resolve_id(envo: Envo, term: EnvoTerm) -> tuple[EnvoTerm | None, str]:
    """A live ENVO term for an ID; obsolete IDs follow an ENVO replacement."""
    if not term.deprecated:
        return term, ""
    if term.replaced_by and term.replaced_by.startswith("ENVO:"):
        new = envo.terms.get(term.replaced_by)
        if new and not new.deprecated:
            return new, "R19e_envo_obsolete_replaced"
    return None, "R19f_envo_obsolete_no_replacement"


# Exact ENVO synonyms that mean something else in our data (checked by
# hand in the dry run, correct_to_faire.md R19d).
_SYNONYM_BLOCKLIST = {
    "tissue": "tissue paper",  # biological tissue in our data
    "gas": "gasoline",
    "mountains": "seamount",
    "human biome": "anthropogenic terrestrial biome",
    "subway": "subway train",  # station / tunnel environments
}

_MANUAL_MAP_FILE = Path(__file__).resolve().parent / "maps/envo_manual.csv"


def _load_manual_map() -> dict[tuple[str, str], str]:
    with _MANUAL_MAP_FILE.open(newline="") as handle:
        return {(r["column"], normalise_label(r["label"])): r["envo_id"] for r in csv.DictReader(handle)}


_MANUAL_MAP = _load_manual_map()


def resolve_envo_label(envo: Envo, label: str, column: str = "") -> tuple[EnvoTerm, str] | None:
    """Exact ENVO label, else a single exact synonym, else the reviewed
    manual map (maps/envo_manual.csv). Never fuzzy."""
    norm = normalise_label(_ENVO_PREFIXED_LABEL.sub("", label))
    hits = envo.by_label.get(norm, [])
    if len(hits) == 1:
        return envo.terms[hits[0]], "R19c_envo_label_only"
    if not hits and norm not in _SYNONYM_BLOCKLIST:
        syn = envo.by_synonym.get(norm, [])
        if len(set(syn)) == 1:
            return envo.terms[syn[0]], "R19d_envo_exact_synonym"
    manual = _MANUAL_MAP.get((column, norm))
    if manual and manual in envo.terms and not envo.terms[manual].deprecated:
        return envo.terms[manual], "R19m_envo_manual_map"
    return None


def resolve_envo_tokens(envo: Envo, tokens: list[str], column: str = "") -> list[tuple[str, str, str]]:
    """[(original text, new item, rule)] for one cell's ' | ' tokens.

    A label token directly followed by an ID token is one pair (our export
    wrote label and ID as separate list items). Items that cannot be
    resolved exactly become "other: <original>" (decision D0).
    """
    out: list[tuple[str, str, str]] = []
    # Repeated adjacent tokens ("marine biome | marine biome | ENVO:...")
    collapsed = [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1]]
    i = 0
    while i < len(collapsed):
        token = collapsed[i]
        label, digits = None, None
        pair_text = None
        combined = _LABEL_WITH_ID.match(token)
        id_only = _ID_TOKEN.match(token)
        if combined:
            label, digits = combined.group(1), combined.group(2)
            original = token
        elif id_only:
            digits = id_only.group(1)
            original = token
        else:
            label = token
            original = token
            nxt = collapsed[i + 1] if i + 1 < len(collapsed) else ""
            nxt_id = _ID_TOKEN.match(nxt)
            if nxt_id:
                digits = nxt_id.group(1)
                original = f"{token} | {nxt}"
                # one list item in the output: "other: label [ENVO:id]"
                pair_text = f"{token} [{nxt}]"
                i += 1
        i += 1
        if label is not None and (label.lstrip().startswith("=") or '"' in label):
            # Spreadsheet formula text, e.g. =IF(F62="Tissue","coral reef
            out.append((original, f"other: {pair_text or original}", "R19j_spreadsheet_formula"))
            continue
        if digits is None:
            if _FOREIGN_ID.match(label):
                out.append((original, f"other: {original}", "R19g_not_envo_id"))
                continue
            hit = resolve_envo_label(envo, label, column)
            if hit:
                out.append((original, _faire_envo(hit[0]), hit[1]))
            else:
                out.append((original, f"other: {original}", "R07_unmapped_other"))
            continue
        term = envo.lookup_id(digits)
        if term is None:
            out.append((original, f"other: {pair_text or original}", "R19h_envo_id_unknown"))
            continue
        live, obsolete_rule = _resolve_id(envo, term)
        if live is None:
            out.append((original, f"other: {pair_text or original}", obsolete_rule))
            continue
        if label is None or _label_matches(label, term) or _label_matches(label, live):
            rule = obsolete_rule or ("R19b_envo_id_only" if label is None else "R19a_envo_label_and_id")
            out.append((original, _faire_envo(live), rule))
            continue
        if _label_close(label, term) or _label_close(label, live):
            out.append((original, _faire_envo(live), obsolete_rule or "R19a2_envo_label_close_and_id"))
            continue
        # Label and ID disagree: keep both for manual review.
        out.append((original, f"other: {pair_text or original}", "R19i_envo_label_id_conflict"))
    return out


# ---------------------------------------------------------------------------
# R20: instrument -> "Name [OBI_xxxxxxx]" (FAIRe example:
# "Illumina HiSeq 1500 [OBI_0003386]"). FAIRe asks for OBI "where possible";
# instruments OBI does not list keep their name unchanged.

_OBI_MANUAL_FILE = Path(__file__).resolve().parent / "maps/obi_manual.csv"
_OBI_ITEM = re.compile(r"^(.+?)\s*\[\s*OBI[_:](\d{7})\s*\]$")


def _load_obi_manual() -> dict[str, str]:
    with _OBI_MANUAL_FILE.open(newline="") as handle:
        return {normalise_label(r["instrument"]): r["obi_id"] for r in csv.DictReader(handle)}


_OBI_MANUAL = _load_obi_manual()


def resolve_instrument(sequencers: dict[str, ObiSequencer], item: str) -> tuple[str, str] | None:
    """(new item, rule), or None when OBI has no matching sequencer."""
    by_label = {normalise_label(s.label): s for s in sequencers.values()}
    done = _OBI_ITEM.match(item)
    if done:
        seq = sequencers.get(f"OBI_{done.group(2)}")
        if seq:
            return f"{seq.label} [{seq.obi_id}]", "R20a_obi_label"
        return None
    norm = normalise_label(item)
    seq = by_label.get(norm) or by_label.get(f"illumina {norm}")
    if seq:
        return f"{seq.label} [{seq.obi_id}]", "R20a_obi_label"
    manual = _OBI_MANUAL.get(norm)
    if manual and manual in sequencers:
        seq = sequencers[manual]
        return f"{seq.label} [{seq.obi_id}]", "R20b_obi_manual_map"
    return None
