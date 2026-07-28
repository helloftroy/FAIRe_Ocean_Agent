# Vendored FAIRe checklist schema

Source: https://github.com/FAIR-eDNA/FAIRe_checklist
Commit: `042ced519c9a4e3808086e6078c12883cb884cd0`
Checklist version: **1.0.2** (per that repo's `latest_checklist/FAIRe_checklist_v1.0.2.xlsx`)

## What's vendored here and why

Only the three compiled, machine-readable artifacts are vendored:

- `schema.yaml` -- LinkML runtime schema (slot definitions: range, required,
  recommended, `requirement_level_code`, examples, `data_type` table
  membership).
- `classes.yaml` -- table groupings (`projectMetadata`, `sampleMetadata`,
  `ampData`, `stdData`, `experimentRunMetadata`, `eLowQuantData`, `taxaRaw`,
  `taxaFinal`) and the exact slot list + column order for each.
- `enums.yaml` -- controlled vocabularies (e.g. `platform_enum`,
  `target_gene_enum`). Some enums are genuinely closed (`platform_enum`,
  `assay_type_enum`); others (`env_broad_scale_enum`, `env_local_scale_enum`,
  `env_medium_enum`) list a single illustrative ENVO example rather than an
  exhaustive set and are treated as open text validated only by CURIE/URI
  shape, not membership.

The upstream repo's own comment in `schema.yaml` says these three files are
**generated from `slots/*.yaml`**, which is upstream's true field-level
primary source (one YAML file per term, ~300 files). That directory is not
vendored here -- it's provenance/debugging detail for upstream maintainers,
not something this pipeline's mapping layer needs at runtime. If a
discrepancy ever turns up between what's vendored here and upstream's
`slots/`, `slots/` wins; re-vendor by re-copying these three files from a
fresh clone.

`latest_checklist/FAIRe_checklist_v1.0.2*.xlsx` (the authoritative
human-readable reference and the `FULLtemplate` export-layout reference) are
**not vendored as binary files** in this repo; the export layout they define
(sheet names == class names above, column order == `classes.yaml` slot
order, one wide row per `samp_name`/`assay_name`/`seq_id` join key with
`requirement_level_code`/`section` header rows) is instead captured directly
in `exports/faire.py` and this README, having been inspected from the
user-provided clone during Milestone 6 development.

`working/FAIRe_checklist_v1.0_mapping.json` (cross-standard DwC/MIxS mapping
hints) was reviewed but is explicitly marked upstream as "work in progress...
not reviewed, finalized, or ready for public use" and is deliberately **not**
used as an authority anywhere in `mapping/faire.py` -- only as informal
background reading during rule-table design.

## Known coverage gap

This pipeline's raw_facts vocabulary (from source adapters in Milestones 2-3
and the LLM extraction prompt in Milestone 4) is coarser-grained than several
FAIRe fields expect. FAIRe splits PCR/extraction protocol detail into many
atomic fields (`pcr_primer_forward`, `pcr_cond`, `pcr_cycles`,
`annealingTemp`, `nucl_acid_ext`, `nucl_acid_ext_kit`, ...); the LLM
extraction prompt currently produces coarser blob-like facts (e.g. a single
`PCR_amplification_conditions` free-text string). `mapping/rules.py`
deliberately does not force these into the atomic fields they don't actually
match -- see that module's docstring and the "Milestone 6" section of
`README.md` for exactly which fields have full rule coverage today versus
which are out of scope pending a FAIRe-aware extraction prompt.
