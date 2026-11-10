# Correcting the FAIRe export to strict FAIRe (NOAA fill-in)

Status: **in progress. Session 1 and session 2 (2026-11-10) done.**

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

Values are listed in `review_other.csv`. Current totals: **9,650 distinct
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

### R12: lib_layout `no files` → `none` (decision D2)

5,032 studies.

### R13: reviewed phrase maps for extraction method, storage solution and filter material

The rules are in `scripts/faire_strict/maps/controlled_terms.csv`: one line
per pattern, with the FAIRe term and the reason. Each pattern must match a
whole short item, at most 8 words, after normalising: lowercase, hyphens to
spaces. A sentence is never reduced to a term.

The categories follow FAIRe's own term definitions. For example, the
`nucl_acid_ext_lysis` definition says physical methods include "freeze-thaw
cycles or the uses of bead mills, homogenizers and sonicators", and thermal
methods "apply high temperatures".

| column | mapped items | still `other:` |
|---|---|---|
| nucl_acid_ext_lysis | 253,724 (bead-beating → physical, lysozyme → enzymatic, SDS / CTAB / Triton → chemical, heat → thermal, …) | 360,751 (1,885 distinct; mostly method sentences) |
| nucl_acid_ext_sep | 144,428 (spin column → column-based, phenol-chloroform → phenol chloroform, ethanol / isopropanol / PEG → precipitation, SPRI / AMPure → magnetic beads, …) | 221,459 |
| samp_store_sol | 11,749 (x% ethanol → ethanol, RA1 / lysis buffer → lysis buffer) | 207,836 (for example `sterilized SW`, RNAlater, PBS: genuinely not in FAIRe's list, so `other:` is correct) |
| filter_material | 12,255 (GF/F → glass fiber, PES → polyethersulfone, mixed cellulose ester → cellulose ester, nylon) | 23,985 |

Deliberately not mapped:
- `kit-proprietary lysis`.
- PVDF, PTFE, Durapore. FAIRe's `thermoplastic membrane` is too vague to put
  these under it.
- Sterivex. Sterivex-GP is PES, but Sterivex-GV is PVDF.
- `organic phase extraction`.
- `chloroform extraction` without phenol.

**R13b (inference, can be rejected):** polycarbonate membrane filters, and
the brands Isopore, Nuclepore and Cyclopore, → `track etched polycarbonate`.
FAIRe only lists track-etched polycarbonate, and membrane polycarbonate
filters are made by track etching.

Ethanol percentages (70%, 95%, …) are not part of the FAIRe term. The
original values are in the change log.

### R14: number and unit split into the value and `_unit` columns

Pairs:
- `samp_vol_we_dna_ext` / `_unit`
- `samp_size` / `_unit`
- `pump_flow_rate` / `_unit`
- `concentration` / `concentration_unit`, a FAIRe term **added as a column**
  because our export lacked it
- `host_tot_mass` / `_unit`
- `host_height` / `_unit`
- `host_length` / `_unit`

`15 ml` becomes `15` and unit `mL`, but only when:
- every number in the cell has the same unit;
- that unit is in the unit term's FAIRe list;
- the unit column is empty or already agrees.

If the unit column disagrees, both values are left for review. There is no
unit conversion (see D6). Unit spellings are matched case-sensitively wherever
case changes the meaning (m vs M, mL vs ML, g vs G).

Counts: `samp_vol_we_dna_ext` 58,169, `concentration` 22,754, `samp_size`
17,947, `pump_flow_rate` 1,271, host measurements 906.

**R14b:** unit-column spellings are normalised to FAIRe's (`liters` → `L`,
`ng/uL` → `ng/µl`).

### R15: target_gene

**R15:** unambiguous gene names are mapped to the FAIRe term. 2,356 cells.
Examples:
- `18S`, `18S rDNA` → `18S rRNA (SSU eukaryote)`
- `12S` → `12S rRNA (SSU mitochondria)`
- `28S` → `28S rRNA (LSU eukaryote)`
- `23S` → `23S rRNA (LSU prokaryote)`
- `CO1`, `cox1`, `cytochrome c oxidase I` → `COI`
- `cytochrome b` → `CytB`
- `ITS1`, `ITS2` → `ITS`
- `bacterial amoA` → `amoA`
- `mt16S` → `16S rRNA (LSU mitochondria)`
- `16S rRNA SSU` → `16S rRNA (SSU prokaryote)`

**Plain `16S` is ambiguous in FAIRe.** It can be `16S rRNA (SSU prokaryote)`
or `16S rRNA (LSU mitochondria)`, so it is decided from the same row's target
taxa:
- **R15b (inference, can be rejected):** `16S rRNA (SSU prokaryote)` when
  `targetTaxonomicAssay` or `targetTaxonomicScope` mention bacteria, archaea,
  prokaryotes or microbes, and no animal groups. 1,530 cells.
- **R15c (inference, can be rejected):** `16S rRNA (LSU mitochondria)` when
  `targetTaxonomicScope` itself names animal groups (fish, sharks, mammals,
  …) and no prokaryotes. 109 cells. `targetTaxonomicAssay` alone doesn't
  count here: it says "vertebrates" in many clearly bacterial studies,
  apparently as a default.
- Everything else stays `other:`. 1,298 cells, with no or mixed context.

**R15d:** `ITS1` / `ITS2` also go into `target_subfragment` (430 cells), so
the subregion isn't lost.

Whenever `target_gene` changes, the original cell is copied to
`extras_projectMetadata.csv` (`target_gene__original`).

Genes not in FAIRe's barcoding list (mcrA, nirS, nirK, pmoA, dsrB, …) are
correctly written as `other: <gene>`.

### R16: samp_category (decision D4)

Applied to every sample row whose `samp_category` is empty after R11. The
label is the BioSample label that R11 moved to extras.

- **R16a → `negative control`:** 1,451 rows. The label contains blank (but
  not "Blank Spring", a place name), NTC, no-template / non-template,
  negative control, neg ctrl, extraction / field / kit blank.
- **R16b → `positive control`:** 157 rows. The label contains mock
  community, mockmix, mock mixture, positive control, pos ctrl, Zymo mock.
  A bare "Mock" is **not** a positive control: in labels like "Mock - 3dpi"
  or "mock exp" it means a mock-inoculated treatment.
- **R16c → `other: Ctrl`:** 2,185 rows. Other labels with control / ctrl,
  such as `Apo_Ctrl_1_1` or `Control_SSW_1`, usually an experimental control
  group. The value is written in FAIRe's `other: …` format.
- **R16d → `sample`:** 1,005,529 rows. All the rest, including rows with no
  label.

### R17: mandatory fields

FAIRe requirement level M, applying each term's condition from the
checklist. Only empty cells are filled.

- **Condition applies, or there is no condition** → `missing: not provided`.
- **Control samples** (negative / positive control, PCR standard) →
  `not applicable: control sample` in `decimalLatitude`, `decimalLongitude`
  and the `env_*` fields. The checklist exempts controls from these terms;
  this is the exact INSDC term for that case.
- **Condition does not apply** → left empty, since the term isn't mandatory
  then. For example, `amp_vis_method` is mandatory only if `assay_type`
  includes `targeted`.
- **Condition unknown** (no `assay_type`) → targeted-only terms are left
  empty.
- **Sample and run rows** take the assay type from their study's
  projectMetadata row, via `internal_study_id`. A sample linked to several
  studies counts as targeted if any of them is.

Columns added, because these are FAIRe terms the original export lacked:
- **projectMetadata:** `pcr_assay_lod`, `pcr_assay_lod_unit`,
  `pcr_assay_loq`, `pcr_assay_loq_unit`, `thresholdQuantificationCycle`
  (targeted assays only).
- **sampleMetadata:** `neg_cont_type`, `pos_cont_type`,
  `detected_notDetected`.

Rules that infer a value rather than reading it:
- **R17b (inference):** `pcr_0_1` = `1` when `assay_type` is targeted or
  metabarcoding, which are PCR-based by definition. 157 studies.
- **R17c (inference from the label):**
  - `neg_cont_type`: field blank → `field negative` (15), extraction / kit
    blank → `extraction negative` (108), NTC / PCR blank / no-template →
    `PCR negative` (377). The other negative controls → `missing: not
    provided` (952).
  - `pos_cont_type` (free text): mock community → `mock community` (140).

Largest fills, which also show coverage (useful for the paper):
- **sampleMetadata, `missing: not provided`:**
  - `env_local_scale` 847k, `env_broad_scale` 838k, `env_medium` 699k
  - `decimalLatitude` / `decimalLongitude` 752k
  - `eventDate` 699k
  - `assay_name` 663k
  - `geo_loc_name` 601k
- **projectMetadata:** `project_contact` 7,524 studies, `assay_name` 5,392,
  `assay_type` 5,359, `recordedBy` 5,253, `pcr_0_1` 5,128.
- **experimentRunMetadata:** `assay_name` 359,971, `pcr_plate_id` 130,654
  (targeted studies).

### Our own artefact found in review

`(+1 similar variants omitted)` and similar values in `target_gene` (about 30
cells) are an elision marker written by our extraction, not data. They are in
`review_other.csv`.

### review_other.csv: `kind` column

Each entry is either:
- `phrase`: 8 words or fewer. Usually a value genuinely not in FAIRe's list,
  which is correct as `other:`.
- `sentence`: pasted method text, probably the wrong field.

The sentences are where review is most needed.

### F01: plausibility flags (values not changed)

Storage temperatures of 50 °C or more are flagged in `review_flags.csv`: 9
distinct values, for example `220`, `280`, `2808`, `80`. These are probably
−20 / −80 °C with the minus sign, or a superscript, lost in PDF text.

## Decisions

| id | question | decision |
|---|---|---|
| D1 | `checkls_ver`: FAIRe types it numeric (example `1.2`), but the real version is `1.0.2` | **Keep `1.0.2`** (2026-11-10) |
| D2 | `lib_layout` = `no files` (5,032 studies with no SRA files) | **Write `none`** (2026-11-10). Note: `none` is not in FAIRe's list (`paired end`, `single end`); it was chosen deliberately. Rule R12. |
| D3 | `0 \| no control/blank mention found` | **`missing: not provided`** (2026-11-10). Rule R04. |
| D4 | `samp_category` once the labels are removed (R11) | **Decided 2026-11-10**: negative / positive control from clear label keywords, `other: Ctrl` for other control labels, `sample` for all the rest. Rule R16. |
| D5 | Add non-FAIRe columns back as new terms? | **No** (2026-11-10). The NOAA copy is FAIRe terms only; the extras files keep the values. |
| D6 | Unit conversion: µL is not in FAIRe's volume list (`mL`, `L`, …) | **Decided 2026-11-10**: keep µL, no conversion, but still split the number and unit (`500 µL` → `500` + `µL`). About 10,800 cells. |
| D7 | `informationWithheld` is "Nothing indicated as withheld" in 9,298 studies. That is our pipeline's default text, not withheld information. Keep it, or blank it (moving the text to extras)? | **open**; kept for now |
| — | Inference rules R05b, R13b, R15b, R15c | **Accepted** 2026-11-10 |

### D4 background: what FAIRe says (decided above)

From the checklist:
- `samp_category` is **mandatory for every sample**, with no condition.
- Allowed values: `sample`, `negative control`, `positive control`,
  `PCR standard`, `other: …`.
- Description: "'Sample' includes biological and technical replicates. If
  negative or positive control was selected, provide further information
  under the terms neg_cont_type and pos_cont_type."

From the guidelines: a missing mandatory value takes an INSDC missing-value
term.

What our data has:
- **Before R11:** 349,336 sample rows had a label in `samp_category`. The
  other 660,053 sample rows had nothing. 67 rows had a real category.
- **Labels that look like controls**, by keyword:
  - 1,350 look like negative controls (`BLANK`, `NTC`, `Control_Blank`).
  - 229 look like positive controls (`Mock community`, `pos_control`).
  - 1,281 only say `Ctrl`. These are often an experimental control
    treatment (for example `Apo_Ctrl_1_1`), which FAIRe would call a
    `sample`, not a negative control.

Recommendation, which needs your OK:
- **`sample`** for every BioSample row, because public BioSamples are almost
  all real samples;
- **`negative control`** / **`positive control`** only where the label
  contains an unambiguous keyword (BLANK, NTC, no-template, negative control
  / mock community, positive control), under a separate rule id you can
  reject;
- `Ctrl` alone stays `sample`, and is listed for review.

The alternative is `missing: not provided` for every row without a known
category. That is FAIRe-correct, but it makes a mandatory field nearly empty
for 1M samples.

## Correction to something I said earlier

On 2026-11-10 I said `platform` and `instrument` were 0% filled in the run
table. They aren't in that table at all: FAIRe puts them in projectMetadata,
where they are 63% and 47% filled.

## Work remaining (next sessions)

Planned in this order. Counts are cells currently written as `other:` (or not
yet checked).

1. ~~Extraction method vocabularies~~: done (R13).
2. ~~Number and unit pairs~~: done (R14). D6 is still open.
3. ~~target_gene~~: done (R15).
4. ~~D4 samp_category~~: done (R16).
5. ~~Mandatory fields~~: done (R17).
6. **Ontology-coded fields** (needs ENVO and OBI downloads, saved with
   versions):
   - `env_broad_scale`, `env_local_scale`, `env_medium` →
     `label [ENVO:xxxxxxxx]`. They are 17–31% filled, and most are labels
     without IDs or IDs without labels.
   - `instrument` → `Name [OBI_xxxxxxx]`.
7. **`geo_loc_name`** against the INSDC country / sea list, in the format
   `Country: region, locality`. For example, `Chile Bay, Greenwich Island…`
   needs `Antarctica:` added. This can only be done where it's unambiguous.
8. **Dates and durations:**
   - `eventDate` as ISO 8601. Our values pass the format check, but this
     needs a closer look.
   - `samp_store_dur` as ISO 8601 durations: `24 h` → `PT24H`,
     `overnight` → D0.
9. **Integer fields with sentences:** `pool_dna_num` (114k cells),
   `pcr_rep`, `biological_rep`. Sentences stay D0.
10. **Final validation:**
   - Re-run the profiler on the strict copy, aiming for zero
     non-conforming values outside `other:` and the missing-value terms.
   - Optionally run FAIRe-fier (the official validator) on a sample of
     studies.
