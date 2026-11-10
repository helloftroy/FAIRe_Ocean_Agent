# Correcting the FAIRe export to strict FAIRe (NOAA fill-in)

Status: **in progress, session 1 of several (2026-11-10).**

Please check this document before the original CSVs are deleted. Each rule has an
id (R01, R02, …), which also appears in the `rule` column of the change logs, so
every changed value can be traced to the rule that changed it.

## What this produces

| | Path | Use |
|---|---|---|
| Original export (unchanged) | `data/snapshots/faire_2026-11/` | delete at the end, after review |
| **Strict FAIRe copy** | `data/snapshots/faire_2026-11_strict/` | the NOAA fill-in deliverable |
| Database | `data/snapshots/fair_ocean_2026-11.db` | our own analysis (unchanged) |

Files in the strict copy:

- `projectMetadata.csv`, `sampleMetadata.csv`, `experimentRunMetadata.csv`: the
  corrected tables.
- `extras_<table>.csv`: everything that was taken out of a FAIRe column, keyed by
  `internal_study_id` (and `samp_name` / `lib_id`). Nothing is dropped: the
  value is either in the FAIRe table, in an extras file, or both.
- `changes_<table>.csv`: every distinct change, as column, rule, original value,
  new value and the number of cells. The value-level rules are deterministic, so
  this list is complete and can be reviewed value by value.
- `review_other.csv`: every value written as `other: <original>` because it
  could not be mapped without guessing (see D0). These are for manual review.
- `review_flags.csv`: values left unchanged but flagged as implausible.

To rebuild (about 2 minutes; the original export is only read):

```
.venv/bin/python scripts/faire_strict/correct_to_faire.py \
    data/snapshots/faire_2026-11 data/snapshots/faire_2026-11_strict
```

Profiling script (fill rate and non-conforming values per column, read-only):
`scripts/faire_strict/profile_export.py`. Its last output is in
`data/snapshots/faire_strict_work/`.

## Reference (for the methods section)

All the reference material is saved in `reference/faire/`, with exact versions
in `VERSIONS.md` and checksums in `SHA256SUMS.txt`:

- **FAIRe checklist v1.0.2**, FAIR-eDNA/FAIRe_checklist, commit `061d8b6`
  (2026-10-06). `latest_checklist/FAIRe_checklist_v1.0.2.xlsx` is the
  authoritative checklist according to the repository README. v1.0.3 appears in
  the changelog but was withdrawn. Three per-term definition files (`slots/`)
  changed after the v1.0.2 spreadsheet. The only change that affects us:
  `pcr2_plate_id` moved to experimentRunMetadata.
- **FAIRe guidelines page** (fair-edna.github.io/guidelines.html), retrieved
  2026-11-10. It defines the missing-value terms, the ` | ` list separator,
  `other: FREE TEXT`, and ISO 8601 dates.
- **FAIRe-NOAA checklist** (NOAA AOML FAIReSheets, commit `47fddc3`). Saved
  for reference only. **Not the target** (decision 2026-11-10: plain FAIRe
  v1.0.2).
- Citation: Takahashi et al. (2025) Environmental DNA 7(3): e70100,
  https://doi.org/10.1002/edn3.70100

## Decisions already made (2026-11-10)

- **Target: plain FAIRe v1.0.2**, not FAIRe-NOAA.
- **Keep `internal_study_id`** as the first column of every table.
- **Keep all studies in one table**, not one submission per dataset. This is a
  fill-in, not a submission.
- **D0: values that cannot be mapped become `other: <original>`**, even in
  fields where FAIRe does not allow "other" (numeric fields, for example).
  You will review these by hand (`review_other.csv`).

## Deliberate deviations from the FAIRe template

1. **One row per study in projectMetadata.** The FAIRe template stores
   projectMetadata in long form: one row per term, and a value column per
   project or assay. Ours stays wide (one row per study, one column per term),
   to keep all studies in one table.
2. **`internal_study_id` column** in all three tables.
3. **No `# requirement_level_code` / `# section` header rows.** The template's
   sample and run sheets start with two comment rows. The strict CSVs have just
   the term header, so they stay plain CSVs. These rows could be added at the
   end if NOAA wants them.

## Rules applied so far

Counts are cells changed in the current build.

### R01: only FAIRe columns, in template order

The columns are the FAIRe v1.0.2 terms present in our export, in the
FULLtemplate's column order, with `internal_study_id` first. Non-FAIRe columns
move to `extras_<table>.csv`:

- **projectMetadata:** `sample_prep_0_1`, `primer_forward_source_unresolved`,
  `primer_reverse_source_unresolved`, `internal_downstream_analysis_techniques`,
  `funding_source`, `spreadsheet_headers` (empty).
- **sampleMetadata:** `internal_expedition_id`, `internal_alias_sample_ids`,
  `x_env_var_block`, `x_pulled_env_var`, `source_unmapped`,
  `eventDate_submitted`.
- **experimentRunMetadata:** `fastq_access_status`.

⚠ **Check:** FAIRe allows adding new terms when neither MIxS nor DwC has one
(guidelines, "Vocabulary basis"). If NOAA would value `funding_source` or the
environmental variables in `x_env_var_block` / `x_pulled_env_var`, they could
go back in as added terms. As it stands, strict means out.

### R02: list separator ` | `

FAIRe separates multiple values with space, vertical bar, space. Our export used
a bare `|`. All lists are rewritten with ` | `. This includes `seq_run_id` for
the 8,488 runs with several SRR runs.

After mapping, duplicate items in one cell are merged, for example
`Illumina MiSeq | Illumina HiSeq` → `ILLUMINA` (2,492 cells in project, 14,921
in sample). Every such merge is in the change log.

### R03: missing values use the exact INSDC wording

These are the FAIRe-allowed terms, listed in `scripts/faire_strict/checklist.py`.
- **Case fixes:** for example `Not applicable` → `not applicable`.
- **Bare category words become the matching term:**
  - `missing`, `not provided` → `missing: not provided`
  - `not collected` → `missing: not collected`
  - `restricted access` → `missing: restricted access`

Counts: 32,950 alias and 1,437 case cells.

⚠ **Check:** the bare word `missing` (most often in env_* fields from BioSample
records) is mapped to `missing: not provided`, which is the most neutral choice.

### R04: evidence text split off neg_cont_0_1 / pos_cont_0_1

Our project table wrote these fields as `flag | evidence sentence`. The 0/1
flag stays. The sentence moves to `extras_projectMetadata.csv`, in the columns
`neg_cont_0_1__evidence` and `pos_cont_0_1__evidence` (3,997 cells).

⚠ **D3, needs your decision:** `0 | no control/blank mention found` (6,031
cells) means the paper does not mention controls, not that it used none. The
current build writes `missing: not provided` for these. The alternative is to
keep `0`.

### R05: numbers, with units removed when the unit is FAIRe's standard unit

A value like `55 °C` becomes `55`, but only when the unit equals the term's
standard unit in the checklist. Only these units are recognised:
- °C (and its typographic variants)
- bp
- cycles
- m
- mm and mm²
- µm
- nM
- %
- reads

Other text handling:
- The lead-ins `stored at`, `kept at`, `frozen at`, `maintained at`,
  `store at` and `at` are removed.
- Unicode minus and dash characters become `-`, so `− 80 °C` becomes `-80`.
- Ranges become `low-high`, for example `0.22–1.6 µm` → `0.22-1.6`.

Unit matching is case-sensitive for metric units, so µM never matches µm.
Anything else, such as a different unit, a sentence, or several numbers, is left
to D0.

Counts: 1,838 project cells and 485,646 sample cells (mostly `samp_store_temp`
and `size_frac`).

**R05b:** `size_frac` given in µM (micromolar) is read as µm. A filter pore
size in µM is meaningless, so it must be a typo. This is a separate rule id so
it can be rejected (27 distinct values).

### R06: allowed-term case and "other" format

- **Case:** a value that differs from an allowed term only in letter case is
  corrected to that term. Examples: `Illumina` → `ILLUMINA` in `platform`, and
  `ml` → `mL` in `samp_vol_we_dna_ext_unit`. Counts: 11,702 sample cells, 137
  project cells.
- **Other format:** `other:metagenomics` → `other: metagenomics`, the format
  the guidelines require (1,010 cells).

### R07: unmapped values become `other: <original>` (decision D0)

Values are listed in `review_other.csv`. Current totals: **11,051 distinct
values**, mostly in columns the next stages will map (see "Work remaining").

### R08: platform from instrument keywords

Free-text platform names are mapped by keyword to FAIRe's platform list (the
INSDC platform names). Examples:
- Illumina, MiSeq, HiSeq, NovaSeq, NextSeq, iSeq → `ILLUMINA`
- 454, GS FLX → `LS454`
- Ion Torrent, PGM, S5 → `ION_TORRENT`
- MinION, GridION, PromethION → `OXFORD_NANOPORE`
- PacBio, Sequel → `PACBIO_SMRT`
- MGISEQ, DNBSEQ → `DNBSEQ`
- BGISEQ → `BGISEQ`
- Sanger, ABI 3730 → `CAPILLARY`

If a value names two different platforms, it is not mapped (D0). Count: 4,865
cells.

The model (`MiSeq`) is lost from `platform`, but `instrument` normally carries
it. Every original value is in the change log.

### R09: small, unambiguous vocabulary maps

- **`assay_type`:** `qPCR`, `quantitative PCR`, `ddPCR`, `digital PCR` →
  `targeted`.
- **`amp_vis_method`:**
  - agarose gel variants → `gel electrophoresis`
  - real-time / quantitative PCR → `qPCR`
  - droplet digital PCR → `dPCR`
- **`lib_layout`:** `paired`, `paired-end` → `paired end`; `single`,
  `single-end` → `single end`.

Bioanalyzer and TapeStation are deliberately **not** mapped. They are library
QC tools, not amplicon detection methods, so they go to D0.

### R10: Boolean words

`yes` / `true` → `1`, `no` / `false` → `0`.

### R11: sample labels out of samp_category

Our pipeline used `samp_category` to log each BioSample's display name: its
`source_material_id`, else `sample_name`, else title. This was your August
instruction, in `sources/ncbi.py::_sample_category_from_title_or_name`. These
labels (349,336 cells) are not FAIRe categories, so they move to
`extras_sampleMetadata.csv`, column `samp_category__sample_label`.

⚠ **D4, needs your decision:** what goes in `samp_category`, which is
mandatory? It can only be `sample`, `negative control`, `positive control` or
`PCR standard`. The field is blank for now. Options:
- (a) `sample` for every BioSample not identified as a control;
- (b) `missing: not provided`;
- (c) also copy the label into `materialSampleID` where that is empty.

### F01: plausibility flags (values not changed)

Storage temperatures of 50 °C or more are flagged in `review_flags.csv`: 9
distinct values, for example `220`, `280`, `2808`, `80`. These are probably
−20 / −80 °C with the minus sign, or a superscript, lost in PDF text.

## Decisions needed from you

| id | question | current build |
|---|---|---|
| D1 | `checkls_ver`: FAIRe types it numeric (example `1.2`), but the real version is `1.0.2`. Keep `1.0.2`? | `other: 1.0.2` (D0); proposed: keep `1.0.2` |
| D2 | `lib_layout` = `no files` (5,032 studies with no SRA files). Use `missing: not provided`? | `other: no files` |
| D3 | `0 \| no control/blank mention found` → `missing: not provided` or `0`? | `missing: not provided` |
| D4 | `samp_category` once the labels are removed (R11) | blank |
| D5 | Non-FAIRe columns (R01): drop as now, or add back any as new terms (`funding_source`, env variables)? | dropped to extras |

## Correction to something I said earlier

On 2026-11-10 I said `platform` and `instrument` were 0% filled in the run
table. They aren't in that table at all: FAIRe puts them in projectMetadata,
where they are 63% and 47% filled.

## Work remaining (next sessions)

Planned in this order. Counts are cells currently written as `other:` (or not
yet checked).

1. **Extraction method controlled vocabularies.**
   - `nucl_acid_ext_lysis` (614k cells, 1,930 distinct): for example
     bead-beating → `physical`, lysozyme / proteinase K → `enzymatic`,
     SDS / CTAB / Triton → `chemical`, freeze-thaw → `physical`.
   - `nucl_acid_ext_sep` (366k cells, 1,198 distinct): spin column →
     `column-based`, phenol-chloroform → `phenol chloroform`, …
   - `samp_store_sol` (220k), `filter_material` (36k).
   - Each needs a reviewed term-by-term map (a CSV in
     `scripts/faire_strict/maps/`). Sentences stay D0.
2. **Number + unit pairs:** `samp_vol_we_dna_ext` / `_unit`,
   `samp_size` / `samp_size_unit`, `pump_flow_rate` / `_unit`,
   `concentration` (+ `concentration_unit`, a FAIRe term we don't export
   yet). Rule: `15 ml` → value 15 and unit `mL`, only when the unit column is
   empty or already agrees.
3. **`target_gene`** (6,400 cells). Mostly safe: `12S` → `12S rRNA (SSU
   mitochondria)`, `18S` → `18S rRNA (SSU eukaryote)`, `COI`/`CO1` → `COI`.
   **16S is ambiguous**: FAIRe has both `16S rRNA (SSU prokaryote)` and
   `16S rRNA (LSU mitochondria)`. It needs a row-level rule using the
   target taxa (prokaryotes vs. vertebrates), else D0. Subregions (`ITS2`,
   `V4`) go to `target_subfragment` when it is empty.
4. **Ontology-coded fields** (needs ENVO and OBI downloads, saved with
   versions):
   - `env_broad_scale`, `env_local_scale`, `env_medium` →
     `label [ENVO:xxxxxxxx]`. They are 17–31% filled, and most are labels
     without IDs or IDs without labels.
   - `instrument` → `Name [OBI_xxxxxxx]`.
5. **`geo_loc_name`** against the INSDC country / sea list, in the format
   `Country: region, locality`. For example, `Chile Bay, Greenwich Island…`
   needs `Antarctica:` added. This can only be done where it's unambiguous.
6. **Dates and durations:**
   - `eventDate` as ISO 8601. Our values pass the format check, but this
     needs a closer look.
   - `samp_store_dur` as ISO 8601 durations: `24 h` → `PT24H`,
     `overnight` → D0.
7. **Mandatory and conditionally mandatory fields:**
   - Add the absent M terms as columns: `neg_cont_type`, `pos_cont_type`,
     `detected_notDetected`, and for targeted assays `pcr_assay_lod/loq`
     (+ units) and `thresholdQuantificationCycle`.
   - Fill empty M fields with the right missing-value term, respecting each
     term's condition (for example, `amp_vis_method` is mandatory only if
     `assay_type = targeted` and `pcr_0_1 = 1`; otherwise
     `not applicable`).
8. **Integer fields with sentences:** `pool_dna_num` (114k cells),
   `pcr_rep`, `biological_rep`. Sentences stay D0.
9. **Final validation:**
   - Re-run the profiler on the strict copy, aiming for zero
     non-conforming values outside `other:` and the missing-value terms.
   - Optionally run FAIRe-fier (the official validator) on a sample of
     studies.
