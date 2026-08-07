"""FAIRe checklist export: CSVs for the populated classes
(`projectMetadata`, `sampleMetadata`, `experimentRunMetadata`), matching
the sheet names and column order of the vendored
`FAIRe_checklist_v1.0.2_FULLtemplate.xlsx` where this pipeline has data.

`ampData`, `stdData`, `eLowQuantData`, `taxaRaw`, and `taxaFinal` are
omitted from exports for now: no source adapter or extraction step in this
pipeline currently produces amplification/standard-curve/taxonomic-
assignment records, so writing them would only create empty files.
`experimentRunMetadata` is populated from sample/assay-specific
experiment_run (library) entities. Sequencing runs remain separate linked
entities, so many library rows may correctly share one `seq_run_id`.

Alongside the per-class data files, `export_faire` also writes
`field_reference.csv` -- one row per FAIRe field, every column that
appears in any of the data files, with its requirement level and
`exact_mappings` (real cross-standard URIs, e.g. MIxS, that came for free
with the vendored FAIRe schema -- see standards/faire_registry.py, built
for Milestone 6b). This is schema-level reference data, not per-study
data, so it isn't squeezed into the data CSVs themselves (which must match
FULLtemplate.xlsx's exact column layout) -- it's a companion data
dictionary instead, built from the same `build_faire_registry()` the
standards registry uses rather than a second, possibly-drifting copy of
the same information.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.enums import EntityLevel, EntityRelationshipType, EntityRootStatus, SHAREABLE_ENTITY_LEVELS
from fair_ocean_agent.database.models import Entity, EntityRelationship, EntityStudy, StandardizedValue, Study
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA, resolve_project_id
from fair_ocean_agent.standards.faire_registry import build_faire_registry

FAIRE_SCHEMA_DIR = REPO_ROOT / "schemas" / "faire"

OMITTED_EMPTY_CLASSES = ("ampData", "stdData", "eLowQuantData", "taxaRaw", "taxaFinal")
# Backward-compatible alias for callers/tests that need to know which FAIRe
# classes are intentionally omitted until the pipeline has row-shaped data.
EMPTY_CLASSES = OMITTED_EMPTY_CLASSES

# Pipeline-internal traceability column, not a real FAIRe field -- deliberately
# NOT added to schemas/faire/classes.yaml (unlike expedition_id/
# ship_crs_expocode, a genuine NOAA/SEUS-MBON domain extension, this must
# never be mistaken for something submittable to NOAA/GBIF/OBIS as part of
# the real checklist) and deliberately excluded from field_reference.csv
# (which documents real FAIRe fields with real requirement levels/
# exact_mappings -- this column has neither). export_faire() merges every
# study in the database into one shared set of output CSVs with no
# per-study filter; this is the only thing that lets an outside observer
# trace which project/sample/experiment rows belong to the same study once
# more than one is being processed at once.
INTERNAL_STUDY_ID_FIELD = "internal_study_id"

# Same non-FAIRe, internal-only precedent as INTERNAL_STUDY_ID_FIELD above --
# not in schemas/faire/classes.yaml, excluded from field_reference.csv.
# Carries a canonical sample's alias entities' own native names (e.g. a
# supplement table's own "GC04_1") once identity/sample_alias_reconciliation.py
# has folded them into the canonical (accessioned) entity's row, so that
# native name isn't lost even though it no longer gets its own row.
INTERNAL_ALIAS_SAMPLE_IDS_FIELD = "internal_alias_sample_ids"

# samp_name/materialSampleID are the row's own identifying columns -- the
# real accession must win outright as the identifier, never get pipe-joined
# against an alias's native name (that native name is preserved separately,
# in INTERNAL_ALIAS_SAMPLE_IDS_FIELD).
_ALIAS_MERGE_EXCLUDED_FIELDS = frozenset({"samp_name", "materialSampleID"})


@lru_cache(maxsize=1)
def _load_classes() -> dict:
    with (FAIRE_SCHEMA_DIR / "classes.yaml").open() as f:
        data = yaml.safe_load(f)
    return data["classes"]


def class_columns(class_name: str) -> list[str]:
    return list(_load_classes()[class_name]["slots"])


def _study_wide_values(session: Session, study_id: str) -> dict[str, str]:
    """entity_id IS NULL standardized_values for a study: projectMetadata
    fields proper, plus sample-scoped LLM facts broadcast as a default
    (see mapping/faire.py's docstring)."""
    rows = session.execute(
        select(StandardizedValue.target_field, StandardizedValue.standardized_value).where(
            StandardizedValue.study_id == study_id,
            StandardizedValue.target_schema == TARGET_SCHEMA,
            StandardizedValue.entity_id.is_(None),
        )
    ).all()
    return {field: value for field, value in rows if value is not None}


def _linked_study_ids(session: Session, entity_id: str) -> list[str]:
    """Every Study this entity is linked to, home or shared (see EntityStudy's
    docstring in database/models.py) -- for SHAREABLE_ENTITY_LEVELS
    (sample/experiment_run/sequencing_run), this can be more than one when a
    second paper reuses the same real BioSample/run another study already
    created. Sorted for a deterministic pipe-joined internal_study_id
    column."""
    return sorted(session.scalars(select(EntityStudy.study_id).where(EntityStudy.entity_id == entity_id)))


def _entity_broadcast_is_authoritative(entity: Entity, study: Study) -> bool:
    """A shared entity's broadcast-style (study-wide LLM/text) facts only
    fill this entity's blanks from the study identity/root_determination.py
    determined to be its root -- never from a non-root linked study, and
    never while root determination is still pending/ambiguous ("unknown is
    preferable to guessing", same stance as identity/consistency.py).
    Subsumes the pre-root-determination "len(linked_study_ids) == 1" gate
    exactly for the non-shared case: root_status is eagerly DETERMINED/self
    at entity creation (identity/entity_linking.py::create_entity), so an
    entity that was never shared always satisfies this."""
    return entity.root_status == EntityRootStatus.DETERMINED.value and entity.root_study_id == study.study_id


def _entity_values(session: Session, entity_id: str) -> dict[str, str]:
    rows = session.execute(
        select(StandardizedValue.target_field, StandardizedValue.standardized_value).where(
            StandardizedValue.entity_id == entity_id,
            StandardizedValue.target_schema == TARGET_SCHEMA,
        )
    ).all()
    return {field: value for field, value in rows if value is not None}


def _linked_entity(
    session: Session,
    from_entity_id: str,
    relationship_type: EntityRelationshipType,
) -> Entity | None:
    entities = list(
        session.scalars(
            select(Entity)
            .join(EntityRelationship, EntityRelationship.to_entity_id == Entity.entity_id)
            .where(
                EntityRelationship.from_entity_id == from_entity_id,
                EntityRelationship.relationship_type == relationship_type.value,
            )
            .order_by(Entity.external_identifier, Entity.entity_id)
            .limit(2)
        )
    )
    if len(entities) > 1:
        raise ValueError(
            f"experiment entity {from_entity_id} has multiple {relationship_type.value} links; "
            "cannot emit an unambiguous FAIRe experimentRunMetadata row"
        )
    return entities[0] if entities else None


def _alias_entities(session: Session, canonical_entity_id: str) -> list[Entity]:
    """Reverse of _linked_entity's direction: every alias entity that
    identity/sample_alias_reconciliation.py has folded INTO this canonical
    entity. Deliberately doesn't reuse _linked_entity's "raise if >1"
    guard -- more than one supplement-derived alias legitimately folding
    into one real accession is the expected case here, unlike
    DERIVED_FROM_SAMPLE/USES_ASSAY/SEQUENCED_IN_RUN's 1:1 semantics."""
    return list(
        session.scalars(
            select(Entity)
            .join(EntityRelationship, EntityRelationship.from_entity_id == Entity.entity_id)
            .where(
                EntityRelationship.to_entity_id == canonical_entity_id,
                EntityRelationship.relationship_type == EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS.value,
            )
            .order_by(Entity.external_identifier, Entity.entity_id)
        )
    )


def _merge_field_values(
    primary: dict[str, str], secondary: dict[str, str], *, exclude: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Folds `secondary` into `primary`: a field present on only one side
    passes through unchanged; identical values on both sides collapse to
    one; genuinely differing non-null values get pipe-joined (primary's
    value first) per the user's own explicit request ("if there are
    conflicts we should list both with a pipe"). Fields in `exclude` are
    copied from `primary` only, never touched by `secondary`."""
    merged = dict(primary)
    for field, value in secondary.items():
        if field in exclude:
            continue
        if field not in merged or not merged[field]:
            merged[field] = value
        elif value and value != merged[field]:
            existing_values = merged[field].split("|")
            if value not in existing_values:
                merged[field] = "|".join([*existing_values, value])
    return merged


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return len(rows)


def _write_field_reference(output_dir: Path, columns_by_class: dict[str, list[str]]) -> int:
    field_to_classes: dict[str, list[str]] = {}
    for class_name, columns in columns_by_class.items():
        for field in columns:
            field_to_classes.setdefault(field, []).append(class_name)

    terms_by_field = {t["upstream_field_name"]: t for t in build_faire_registry()}
    rows = []
    for field, classes in sorted(field_to_classes.items()):
        term = terms_by_field.get(field, {})
        rows.append(
            {
                "faire_field": field,
                "faire_classes": "|".join(classes),
                "requirement_level_code": term.get("requirement_level_code", ""),
                "requirement_level_condition": term.get("requirement_level_condition") or "",
                "range": term.get("range", ""),
                "exact_mappings": "|".join(term.get("exact_mappings") or []),
            }
        )
    return _write_csv(
        output_dir / "field_reference.csv",
        ["faire_field", "faire_classes", "requirement_level_code", "requirement_level_condition", "range", "exact_mappings"],
        rows,
    )


def export_faire(session: Session, output_dir: str | Path) -> dict[str, int]:
    output_dir = Path(output_dir)
    counts: dict[str, int] = {}
    for class_name in OMITTED_EMPTY_CLASSES:
        stale_path = output_dir / f"{class_name}.csv"
        if stale_path.exists():
            stale_path.unlink()

    studies = list(session.scalars(select(Study)))

    project_columns = class_columns("projectMetadata")
    project_rows = []
    for study in studies:
        # Analysis-only paper: links to shared samples/runs (EntityStudy)
        # but is home to none of them -- did no original data collection of
        # its own, just reanalysis. Still exists in the network (Study row,
        # EntityStudy links, sampleMetadata/experimentRunMetadata rows for
        # anything it DOES home -- none, by this same condition), but never
        # gets a projectMetadata row. Evaluated unconditionally (not gated
        # on entity_component_status): this is a structural fact about
        # entity ownership, not a cross-study priority judgment like root
        # determination, so it's correct to check at any time.
        homed_entity_exists = (
            session.query(Entity.entity_id)
            .filter(
                Entity.study_id == study.study_id,
                Entity.entity_level.in_([level.value for level in SHAREABLE_ENTITY_LEVELS]),
            )
            .first()
            is not None
        )
        linked_entity_exists = (
            session.query(EntityStudy.entity_study_id).filter(EntityStudy.study_id == study.study_id).first()
            is not None
        )
        if linked_entity_exists and not homed_entity_exists:
            continue

        broadcast = _study_wide_values(session, study.study_id)
        project_id = resolve_project_id(session, study.study_id)
        # Real FAIRe's own projectMetadata export layout is one row per
        # assay_name, not one global row per study (see
        # schemas/faire/README.md) -- a paper describing more than one
        # assay tags each assay's facts with a real ASSAY Entity
        # (extraction/text.py's assay_tag), and mapping/faire.py gives each
        # its own StandardizedValue rows (entity_id set, not broadcast).
        #
        # Gated on assay entities that have at least one direct
        # StandardizedValue, not merely on the entity existing:
        # extraction/experiment_runs.py's materialize_legacy_experiment_runs
        # already creates ASSAY entities today purely as USES_ASSAY link
        # targets for structured ENA/BioProject assay_name facts -- those
        # facts land on the EXPERIMENT_RUN entity's row, never the assay
        # entity itself, so such an assay entity has zero StandardizedValue
        # rows of its own. Gating on mere existence would make every such
        # structured-only multi-assay study suddenly emit one
        # near-duplicate broadcast row per assay where it previously
        # emitted exactly one.
        assay_entities = list(
            session.scalars(
                select(Entity).where(
                    Entity.study_id == study.study_id, Entity.entity_level == EntityLevel.ASSAY.value
                )
            )
        )
        assays_with_values = [assay for assay in assay_entities if _entity_values(session, assay.entity_id)]
        if not assays_with_values:
            if project_id is None and not broadcast:
                continue  # nothing at all mapped for this study -- don't emit an all-blank row
            row = dict(broadcast)
            row["project_id"] = project_id or ""
            row[INTERNAL_STUDY_ID_FIELD] = study.study_id
            project_rows.append(row)
            continue
        for assay in assays_with_values:
            row = dict(broadcast)
            row.update(_entity_values(session, assay.entity_id))
            row.setdefault("assay_name", assay.external_identifier or assay.label or assay.entity_id)
            row["project_id"] = project_id or ""
            row[INTERNAL_STUDY_ID_FIELD] = study.study_id
            project_rows.append(row)
    counts["projectMetadata"] = _write_csv(
        output_dir / "projectMetadata.csv", [INTERNAL_STUDY_ID_FIELD, *project_columns], project_rows
    )

    sample_columns = class_columns("sampleMetadata")
    sample_rows = []
    for study in studies:
        # entities are keyed by home study_id (Entity.study_id, unchanged),
        # so a shared entity is still visited exactly once here, under its
        # home study -- no double-counting despite an entity now possibly
        # being linked to more than one Study.
        sample_entities = session.scalars(
            select(Entity).where(Entity.study_id == study.study_id, Entity.entity_level == EntityLevel.SAMPLE.value)
        )
        for entity in sample_entities:
            # A supplement-derived alias entity (identity/
            # sample_alias_reconciliation.py) never emits its own row --
            # its values are folded into its canonical (accessioned)
            # entity's row below instead.
            if _linked_entity(session, entity.entity_id, EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS) is not None:
                continue

            linked_study_ids = set(_linked_study_ids(session, entity.entity_id))
            # A shared sample's row must not carry any non-root study's
            # paper-specific interpretive broadcast defaults -- only the
            # root study's (identity/root_determination.py), or its own
            # entity-level facts, which are unconditionally safe regardless
            # of how many studies link to it.
            broadcast = _study_wide_values(session, study.study_id) if _entity_broadcast_is_authoritative(entity, study) else {}
            row = dict(broadcast)
            row.update(_entity_values(session, entity.entity_id))
            row["samp_name"] = entity.external_identifier or entity.entity_id

            alias_sample_ids = []
            for alias in _alias_entities(session, entity.entity_id):
                alias_broadcast = (
                    _study_wide_values(session, alias.study_id)
                    if _entity_broadcast_is_authoritative(alias, alias.study)
                    else {}
                )
                alias_row = dict(alias_broadcast)
                alias_row.update(_entity_values(session, alias.entity_id))
                row = _merge_field_values(row, alias_row, exclude=_ALIAS_MERGE_EXCLUDED_FIELDS)
                alias_sample_ids.append(alias.external_identifier or alias.entity_id)
                linked_study_ids.update(_linked_study_ids(session, alias.entity_id))
            if alias_sample_ids:
                row[INTERNAL_ALIAS_SAMPLE_IDS_FIELD] = "|".join(alias_sample_ids)

            row[INTERNAL_STUDY_ID_FIELD] = "|".join(sorted(linked_study_ids))
            sample_rows.append(row)
    counts["sampleMetadata"] = _write_csv(
        output_dir / "sampleMetadata.csv",
        [INTERNAL_STUDY_ID_FIELD, INTERNAL_ALIAS_SAMPLE_IDS_FIELD, *sample_columns],
        sample_rows,
    )

    experiment_columns = class_columns("experimentRunMetadata")
    experiment_rows = []
    for study in studies:
        experiment_entities = session.scalars(
            select(Entity).where(
                Entity.study_id == study.study_id,
                Entity.entity_level == EntityLevel.EXPERIMENT_RUN.value,
            )
        )
        for entity in experiment_entities:
            linked_study_ids = _linked_study_ids(session, entity.entity_id)
            broadcast = _study_wide_values(session, study.study_id) if _entity_broadcast_is_authoritative(entity, study) else {}
            row = dict(broadcast)
            row.update(_entity_values(session, entity.entity_id))
            sample = _linked_entity(
                session,
                entity.entity_id,
                EntityRelationshipType.DERIVED_FROM_SAMPLE,
            )
            assay = _linked_entity(
                session,
                entity.entity_id,
                EntityRelationshipType.USES_ASSAY,
            )
            sequencing_run = _linked_entity(
                session,
                entity.entity_id,
                EntityRelationshipType.SEQUENCED_IN_RUN,
            )
            if sample is not None:
                # Defends against a run's DERIVED_FROM_SAMPLE link pointing
                # at a supplement-derived alias entity (identity/
                # sample_alias_reconciliation.py) that no longer emits its
                # own sampleMetadata row -- prefer the real accessioned
                # entity so samp_name always resolves to a row that exists.
                canonical_sample = (
                    _linked_entity(session, sample.entity_id, EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS)
                    or sample
                )
                row.setdefault("samp_name", canonical_sample.external_identifier or canonical_sample.entity_id)
            if assay is not None:
                row.setdefault("assay_name", assay.external_identifier or assay.label or assay.entity_id)
            if sequencing_run is not None:
                row.setdefault("seq_run_id", sequencing_run.external_identifier or sequencing_run.entity_id)
            if entity.external_identifier and not entity.external_identifier.startswith("internal:"):
                row.setdefault("lib_id", entity.external_identifier)
            row[INTERNAL_STUDY_ID_FIELD] = "|".join(linked_study_ids)
            experiment_rows.append(row)
    counts["experimentRunMetadata"] = _write_csv(
        output_dir / "experimentRunMetadata.csv",
        [INTERNAL_STUDY_ID_FIELD, *experiment_columns],
        experiment_rows,
    )

    columns_by_class = {
        "projectMetadata": project_columns,
        "sampleMetadata": sample_columns,
        "experimentRunMetadata": experiment_columns,
    }
    counts["field_reference"] = _write_field_reference(output_dir, columns_by_class)

    return counts
