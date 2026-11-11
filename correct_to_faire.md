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
- **ENVO** release 2026-10-23 and **OBI** release 2026-07-27, in
  `reference/ontologies/` (gzip copies, version IRIs, checksums). Used by
  R19 and R20.
- **INSDC `/geo_loc_name` vocabulary**, revised 2024-07-08, in
  `reference/insdc/` (the page as downloaded plus the extracted lists).
  Used by R21.

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

Values are listed in `review_other.csv`. Current totals: **9,813 distinct
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

### R18: storage durations → ISO 8601

Applies to `samp_store_dur` and `eventDurationValue`, which use the fixed
format `PnYnMnWnDTnHnMnS`.

- **Converted (35,201 cells, 34 distinct values):**
  - one number and one unit, optionally preceded by "for";
  - number words one to twelve;
  - units h / hour, min, s, day, week, month, year.
- **Examples:** `24 h` / `for 24 h` → `PT24H`, `30 min` → `PT30M`,
  `one hour` → `PT1H`, `4 weeks` → `P4W`.
- **Ranges:** `4–10 days` → `P4D/10D`, following the style of FAIRe's own
  example `P1W/4W`. This is done for date units only. Time-unit ranges such
  as `2-4 h` go to D0, because FAIRe gives no example for them.
- **Not converted, so `other:` (152,618 cells):**
  - vague durations (`overnight`, `briefly`);
  - open-ended durations (`until DNA extraction`, `until further
    processing`);
  - qualified durations (`up to 2 months`, `no more than 4 h`,
    `less than a week`);
  - sentences.

Before R18, these cells were kept unchanged even when they didn't match the
format.

### R19: env_broad_scale, env_local_scale, env_medium → `label [ENVO:id]`

FAIRe's format is `termLabel [termID]`, for example `estuarine biome
[ENVO:01000020]`. The reference is ENVO release 2026-10-23, saved in
`reference/ontologies/`. Code: `scripts/faire_strict/ontology.py`.

Our export wrote the label and the ID as two separate list items
(`marine biome | ENVO:00000447`). R19 therefore reads the raw cell, before
the R02 dedupe, and treats a label followed by an ID as one pair. Repeated
adjacent items are collapsed first: `marine biome | marine biome |
ENVO:00000447` is one pair. Missing-value items go through R03 as usual.
Whenever the cell changes, the full original cell is copied to
`extras_sampleMetadata.csv` as `<column>__original`.

The label written is always ENVO's current label for the ID. When an
unresolved label/ID pair becomes `other:`, it is written as one list item,
`other: <label> [<ID as written>]`. Writing it as `other: label | ENVO:id`
would split it into two list items.

| rule | what | cells (broad / local / medium) |
|---|---|---|
| R19a | label + ID, label is the ID's label or one of its synonyms (any type) | 25,975 / 23,498 / 21,762 |
| R19a2 | label + ID, label is a near typo (similarity ≥ 0.9) or one name's words contain the other's, e.g. `marine coral reeef biome`, `marine \| ENVO:00000447`. The ID decides. | 2,818 / 1,480 / 607 |
| R19b | ID only → ENVO label added | 6,783 / 9,056 / 7,372 |
| R19c | label only, exactly one ENVO label (case and spacing ignored) | 50,477 / 34,808 / 80,905 |
| R19d | label only, exactly one ENVO exact synonym, e.g. `freshwater` → `fresh water [ENVO:00002011]` | 1,513 / 2,007 / 2,337 |
| R19e | obsolete ID with an ENVO replacement → the replacement, e.g. `ENVO:00002151` → `ocean water [ENVO:00002149]` | 60 / 403 / 361 |
| R19m | reviewed manual map, `maps/envo_manual.csv` (19 rows, listed below) | 6,543 / – / 30,461 |
| R19f | obsolete ID, no ENVO replacement (e.g. `ENVO:00009003`, obsolete human-associated habitat) → `other:` | 2,185 / 692 / 104 |
| R19g | a non-ENVO ID (OHMI, UBERON, PO, PCO …) → `other:` | 889 / 2,314 / 1,530 |
| R19h | ENVO ID that does not exist (wrong digit count, or spreadsheet drag-fill such as `ENVO:06105017`, `…018`, …) → `other:` | 255 / 306 / 485 |
| R19i | label and ID disagree, e.g. `hydrothermal vent sediment \| ENVO:00005779` (= vegetable garden soil) → `other:` | 943 / 1,211 / 507 |
| R19j | spreadsheet formula text (`=IF(F62="Tissue",…`) → `other:` | 206 / 2 / 347 |
| R07 | no exact match → `other:` | rest |

Points to check:

- **ENVO renamed a term (see D8).** In this release ENVO:00002149 is
  labelled **"ocean water"**. FAIRe's v1.0.2 example (and many of our
  cells, about 5,400) still say `sea water`. The ID is kept and ENVO's
  current label is written: `ocean water [ENVO:00002149]`. The label-only
  word `seawater` matches the new class `seawater [ENVO:01001964]` exactly
  (R19c), and the manual map sends `sea water` / `marine water` there too.
- **Synonyms blocked.** Five exact synonyms are never used, because in our
  data they mean something else: tissue (= tissue paper), gas (gasoline),
  mountains (seamount), human biome (anthropogenic terrestrial biome),
  subway (subway train).
- **Matching is never fuzzy without an ID.** Near matches (R19a2) are used
  only when the cell also gives the ID.
- **Manual map (R19m), `scripts/faire_strict/maps/envo_manual.csv`:**
  - env_medium:
    - feces, faeces, stool, stool sample, fecal, fecal sample, human
      feces, human fecal sample → `fecal material [ENVO:00002003]`;
    - sea water, marine water, seawater sample, marine water sample →
      `seawater [ENVO:01001964]`;
    - coastal water → `coastal sea water [ENVO:00002150]`;
    - wastewater → `waste water [ENVO:00002001]`;
    - deep-sea sediment, deep sea sediment → `deep marine sediment
      [ENVO:00002113]`;
    - sediments → `sediment [ENVO:00002007]`.
  - env_broad_scale:
    - marine → `marine biome [ENVO:00000447]`;
    - coral reef biome → `marine coral reef biome [ENVO:01000049]`.
- **What stays `other:`.** Many of the remaining values are host-associated
  (gut, sputum, blood, vagina, human). ENVO has no class for these (they are
  UBERON anatomy terms), so they stay `other:` for review rather than being
  forced into a loose ENVO class.
- **F02 flag (value not changed).** FAIRe asks for the env_* terms to come
  from one ENVO branch each: biome for broad scale, layer for local scale,
  environmental material for medium. ENVO terms outside that branch are kept
  and listed in `review_flags.csv`. Cells flagged: broad 35,927, local
  65,058, medium 19,375. Most local-scale values are features rather than
  layers (`ocean`, `hospital`, `coral reef`), and FAIRe's own example
  `marine pelagic zone` is not a layer either. So F02 is information, not an
  error.

### R20: instrument → `Name [OBI_xxxxxxx]`

FAIRe example: `Illumina HiSeq 1500 [OBI_0003386]`. The reference is OBI
release 2026-07-27: the 39 named subclasses of DNA sequencer
(OBI_0400103). Code: `ontology.py`.

- **R20a (5,657 items):** exact OBI label (case ignored), also with
  `Illumina` added in front (`NextSeq 500` → `Illumina NextSeq 500
  [OBI_0002021]`).
- **R20b (699 items): reviewed map, `maps/obi_manual.csv`:**
  - MinION, PromethION → Oxford Nanopore …
  - Sequel, Sequel II, Sequel IIe, Revio → PacBio …
  - 454 GS FLX and 454 GS FLX Titanium → 454 Genome Sequencer FLX
    (Titanium is a reagent upgrade of the same instrument).
  - 454 GS 20 → 454 Genome Sequence 20 (OBI's spelling).
  - AB SOLiD System, AB SOLiD 4 System, AB SOLiD 3 Plus System,
    Helicos HeliScope.
- **Not mapped because ambiguous:**
  - GridION (OBI has only the Mk1 model);
  - Illumina HiSeq X (Five or Ten);
  - PacBio RS (OBI has only RS II);
  - 454 GS FLX+;
  - AB SOLiD System 3.0.
- **R20c (see D9):** instruments OBI does not list keep their name unchanged
  and are flagged F03 (516 items). These include:
  - Ion Torrent PGM, Proton and S5;
  - NextSeq 2000;
  - DNBSEQ-G400 and DNBSEQ-T7;
  - BGISEQ and MGISEQ;
  - iSeq 100;
  - Element AVITI;
  - AB capillary sequencers;
  - 454 GS Junior;
  - the unmapped names above.

  FAIRe says the term should come from OBI "where possible", so a plain name
  is valid when OBI has no term.
- **R20d (28 items):** `unspecified`, which is SRA's instrument_model
  placeholder, becomes `missing: not provided`.

### R21: geo_loc_name against the INSDC vocabulary

The FAIRe checklist links `insdc.org/country.html`, which now returns 404:
in 2024 INSDC renamed the `/country` qualifier to `/geo_loc_name`. The
reference is the INSDC `/geo_loc_name` vocabulary (revised 2024-07-08,
280 current names and 17 historical ones), saved in `reference/insdc/`.
Code: `scripts/faire_strict/geo.py`. Only the country or sea part is
checked or moved; the rest of the text is kept exactly as written.

- **keep (292,577 items):** the part before `:` (or the whole value) is
  already an INSDC name.
- **R21a (310):** case only (`NETHERLANDS` → `Netherlands`).
- **R21b (16):** unambiguous aliases, e.g. `United states of america` →
  `USA`, `Republic of Korea` → `South Korea`. The full list is `ALIASES`
  in `geo.py`.
- **R21d (52,543):** no `:`, and exactly one comma-separated part is an
  INSDC name, and it is the last part. That name is moved to the front:
  `Chile Bay, Greenwich Island, South Shetland Islands, Antarctica` →
  `Antarctica: Chile Bay, Greenwich Island, South Shetland Islands`
  (48,986 of these cells); `Halifax, Canada` → `Canada: Halifax`.
- **R21c:** the same when the name is the first part. No cells in this
  export.
- **R21f (2,896):** `GAZ:` prefix, from the GAZ gazetteer, dropped:
  `GAZ:United States of America` → `USA`.
- **R21g (1,750):** historical names with one current successor:
  - Czech Republic → Czechia (most cells);
  - Swaziland → Eswatini;
  - Burma → Myanmar;
  - also the Macedonia, East Timor, Zaire and Siam entries.
- **R21h / F04 (358):** historical names without a single successor
  (`Korea`, `Netherlands Antilles`) are kept and flagged.
- **Everything else → `other:` (54,523 cells, 487 distinct).** These are
  values with no country or sea name, or with several:
  - site names (`CAWS`, `Framlingham Forest reserve`, `Helgoland Roads`);
  - cruise descriptions;
  - US states without `USA`;
  - seas INSDC does not list (`Red Sea`, `Chesapeake Bay`).

  Adding the country would need per-site knowledge, so these are left for
  the manual review.

### R22: integer fields written as words

- **R22 (60 cells):** a whole cell that is one number word in an integer
  field: `three` → `3`.
- **R22b (126 cells):** in `*_rep` fields only, `duplicate(s)` → 2,
  `triplicate(s)` → 3, `quadruplicate(s)` → 4. In `pcr_cycles`,
  `triplicate` is a misplaced value, so there it stays `other:`.
- **Not converted, so `other:`:**
  - vague values (`multiple`, `several`, `two or three`, `5 to 10`);
  - pasted sentences in `pool_dna_num` (all of its filled cells; the field
    is the number of extracts pooled, and our extraction filled it with
    methods text).

### R23: bibliographicCitation → the DOI

FAIRe types `bibliographicCitation` as a DOI (example `DOI:
10.1111/mec.15382`). Our export wrote the full reference text.

- **R23 (3,844 studies):** the text contains exactly one DOI, which is
  written as `DOI: 10.xxxx/...`. The full reference text is kept in
  `extras_projectMetadata.csv` as `bibliographicCitation__full_text`.
- **`other:` (8 studies):** JSON-LD records with several DOIs or none (from
  PANGAEA). The full text is in the extras file as well.

### R24: host_life_stage → UBERON life cycle stage

FAIRe's format is `termLabel [termID]`, with the example `juvenile stage
[UBERON_0034919]`. The ID form is written with an underscore, as in FAIRe's
example. The IDs were checked against UBERON release 2026-11-05 (OLS
responses saved in `reference/ontologies/uberon_ols/`).

- **R24 (3,099 cells), any case:**
  - adult → `post-juvenile adult stage [UBERON_0000113]` (UBERON's label;
    "adult stage" is its synonym);
  - juvenile(s) → `juvenile stage [UBERON_0034919]`;
  - larva, larvae, larval → `larval stage [UBERON_0000069]`.
- **`other:` (1,427 cells):** everything else, for example `native polyp`,
  `Mature`, `grower pig`, `seedling`, `medusa`, `V8`. Before this change
  these cells were kept unchanged even though they don't match the format.

### R03 addition: INSDC spellings of the missing-value categories

INSDC writes, for example, `missing: third party data`, where FAIRe writes
`missing: not collected: third party data`. All eight INSDC forms are
mapped to the matching FAIRe term. They are the same categories, nested
differently.

### eventDate: checked, no change needed

All 310,654 filled values are `yyyy-mm-dd`, `yyyy-mm` or `yyyy`. The
checklist definition lists all three as valid ISO 8601. (The definition's
"Except: 2008-01; 2008" sentence is garbled in the upstream checklist; read
in full, it says these forms are compliant.) There are no times, so no time
zone is needed.

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
| D7 | `informationWithheld` is "Nothing indicated as withheld" in 9,298 studies. That is our pipeline's default text, not withheld information. Keep it, or blank it (moving the text to extras)? | **Keep** (2026-11-10) |
| D8 | ENVO renamed ENVO:00002149 from "sea water" to **"ocean water"** (release 2026-10-23); FAIRe's example still says `sea water [ENVO:00002149]` | **Keep ENVO's current label**, `ocean water [ENVO:00002149]` (2026-11-10) |
| D9 | Instruments OBI does not list (Ion Torrent, NextSeq 2000, DNBSEQ, AVITI …) | **Keep the plain name** (2026-11-10); flagged F03 for information only |
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

## Status (2026-11-10)

- All decisions D1–D9 are made.
- **`other:` values are left as they are.** The user decided not to review
  the 36,324 distinct values by hand. They stay in the NOAA copy as
  `other: <original>`, so nothing is lost.
- **Original export deleted** (`data/snapshots/faire_2026-11/`), at the user's
  request after validation. Before deleting:
  - its SHA-256 checksums were saved;
  - its two small side files (`api_paper_corrections.csv`,
    `field_reference.csv`) were copied.

  Both are in `data/snapshots/faire_strict_work/original_export_sidefiles/`.
  To re-run the correction, regenerate the export from
  `data/snapshots/fair_ocean_2026-11.db` with `export-faire` (about
  26 minutes) and run `correct_to_faire.py` again.
- **Fill summary:**
  - per-term counts for every FAIRe column:
    `data/snapshots/faire_strict_work/fill_summary/fill_summary.csv`, made
    by `scripts/faire_strict/fill_summary.py`;
  - manuscript figure: `fill_ecology.pdf/.svg/.png` in the same folder,
    made by `scripts/faire_strict/plot_fill.py`.

## Work remaining (next sessions)

Planned in this order. Counts are cells currently written as `other:` (or not
yet checked).

1. ~~Extraction method vocabularies~~: done (R13).
2. ~~Number and unit pairs~~: done (R14, with D6).
3. ~~target_gene~~: done (R15).
4. ~~D4 samp_category~~: done (R16).
5. ~~Mandatory fields~~: done (R17).
6. ~~Ontology-coded fields~~: done (R19 ENVO, R20 OBI; D8, D9 open).
7. ~~`geo_loc_name`~~: done (R21).
8. ~~Dates and durations~~: done (R18; eventDate checked).
9. ~~Integer fields with words~~: done (R22). Sentences stay `other:`
   (D0); `pool_dna_num` is entirely pasted methods text.
10. **Final validation:** profiler re-run on the strict copy (2026-11-10).
   Outside `other:` and the missing-value terms, the only non-conforming
   values left are the deliberate decisions:
   - `checkls_ver` = `1.0.2` (D1, 9,298 studies);
   - `lib_layout` = `none` (D2, 5,032 studies);
   - `µL` in three unit columns (D6, 21,944 cells).

   The env_* fields, geo_loc_name, the durations, integers, Booleans,
   controlled vocabularies and bibliographicCitation all conform. In total,
   1,769,877 cells (36,324 distinct values) are `other:` for the manual
   review in `review_other.csv`.

   Still to do:
   - Optionally run FAIRe-fier (the official validator) on a sample of
     studies.

## Coverage gaps found 2026-11-10 (not fixed yet; need the user's go-ahead)

Why lat/lon, date, depth and geo_loc_name look so low (25-40 % of samples):

1. **BioSample fetch cap.** `sources/ncbi.py` `MAX_SAMPLES_PER_PROJECT = 300`:
   in large BioProjects only the first 300 BioSamples get their attributes
   fetched. 1,009,474 sample entities, only 643,987 have any raw fact; the
   other ~365k (36 %) are accession-only rows with no metadata at all.
2. **Raw values never mapped.** Parseable values already in raw_facts but
   empty in the export (sample rows): lat/lon 92,177 (25.5 -> 34.6 %),
   eventDate 132,449 (30.8 -> 43.9 %), geo_loc_name 116,908 (40.1 -> 51.7 %),
   depth 21,127 (20.5 -> 22.6 %). Mostly samples with no
   standardized_values at all (MAP_FAIRE never ran for their facts; MAP_FAIRE
   completed for 5,686 of 10,100 studies), plus BioSample attribute
   spellings with no mapping rule ("collection date", "geographic location
   (latitude)", "geographic location (country and/or sea)", "country",
   "geo loc name", "geographic location (depth)", "Latitude", "lat lon", ...).
   Fix: add the rules, run enqueue-mapping-backfill + worker (deterministic,
   no LLM), re-export, re-run the strict correction.

Paper access (scripts/faire_strict/study_access.py, all 10,100 studies):
fulltext 5,166; paper linked but full text never retrieved (paywalled)
1,131; no linked paper 3,803. In the export: 1,121 / 9,503 studies and
66,412 / 1,009,389 samples are in paywalled studies. The fill figure now
shows "not reported, paper paywalled" as its own segment.
