# Data integrity audit (started 2026-11-10)

All numbers from the Nov 9 snapshot `data/snapshots/fair_ocean_2026-11.db`.
The live database is on the cluster; it keeps changing while discovery runs.

## User decisions (2026-11-10)

- Drop non-marine material **per sample and per BioProject**; then drop any
  study left with no marine samples and a non-marine title.
- **Freshwater stays.** Estuarine and brackish count as marine.
- Do not fetch the ~2.4M BioSamples past the 300 cap wholesale. After the
  non-marine removal, fill in only the samples still missing (expected small).
- An order of operations for the cluster is needed at the end.

## Finding 1: SAMEA/SAMD accessions fetched as NCBI UIDs (root cause of contamination)

`NcbiBioSampleAdapter.fetch_record_by_accessions` (sources/ncbi.py) passes
known accessions straight to efetch. efetch drops the prefix of SAMEA/SAMD
and reads the digits as a UID: SAMEA2620084 -> SAMN02620084 (unrelated).
Checked live 2026-11-10. Used by the empty-elink fallback and by
`_reconcile_missing_biosample_accessions` (workflow/handlers.py).

Chain: wrong SAMN sample -> its BioSample/SRA accessions become study
identifiers -> `_resolve_sequence_source_identifiers` resolves them to their
BioProjects (source `external_identifier_sequence_accession_resolution`) ->
those BioProjects are fetched -> hundreds to thousands of foreign samples.

Scope: 49,908 digit-collision SAMN samples in 499 studies; those studies
hold 539,431 of 1,009,474 sample entities. 292 of them also gained
BioProjects through sequence-accession resolution.
Example: STUDY-19dd2727d4f7 (Antarctic sponge, PRJEB18736): 215 collisions,
155 resolved BioProjects (SARS-CoV-2, malaria trial PRJNA983279, mouse,
soil, wastewater), 49,019 samples.

Fix (code): resolve every non-SAMN accession through esearch `[accn]` and
keep only returned records whose accession was requested. Done in
scripts/fetch_missing_biosamples.py; still needed in sources/ncbi.py.
Also restrict `_resolve_sequence_source_identifiers` to accessions found in
the paper itself, not accessions harvested from fetched samples.

## Finding 2: DataCite related DOIs merge studies

`DataCiteAdapter.find_related` emits every relatedIdentifier (References,
IsCitedBy, ...) as a DOI with STRUCTURED_SOURCE confidence, and
`resolve_or_create_study` merges on any DOI match without looking at the
relationship type. Result: STUDY-caa33f2ce2d5 holds 448 papers, 72,618
DataCite DOIs, 93,357 samples. Otherwise only 11 two-paper studies.
`merge_study_into` records no "absorbed into" link (572 studies merged).

## Finding 3: no marine screen; non-marine bulk

`marine_relevance_status` = unknown for all studies. scripts/marine_audit.py
(per sample: organism, else host, else isolation_source/env_broad_scale):
medical 257,649; marine 122,768; terrestrial 96,295; freshwater 26,466;
engineered 19,485; other host 16,150; no evidence 470,661. Much of the
medical/terrestrial bulk is Finding 1 contamination.

## Finding 4: mapping never ran for 4,029 studies with samples

MAP_FAIRE for a study with samples waits for CHECK_COMPONENT_SETTLED.
Settle tasks claimable since September were never claimed (worker limited
to other task types). Stalled components never get MAP_FAIRE at all.

## Finding 5: 300 BioSamples per BioProject cap

sources/ncbi.py MAX_SAMPLES_PER_PROJECT = 300. 1,188 capped projects,
~2.4M BioSamples never fetched; the largest are clinical projects pulled in
by Finding 1/2 (SARS-CoV-2, MalariaGEN, CDC HAI-Seq, PHE E. coli, HMP).
332k sample entities exist without attributes:
cluster/run_fetch_missing_biosamples.sbatch (running since 2026-11-10).

## Open

- Mapping-rule audit (scripts/faire_strict/mapping_inventory.py) running.
- Cleanup design for Finding 1/2 (needs user approval before touching data).

## Code fixes (2026-11-10, commit b6c8497)

- sources/ncbi.py `fetch_record_by_accessions`: esearch `[accn]` -> UID ->
  efetch, keep only requested accessions (regression test added).
- sources/datacite.py `find_related`: related DOIs only for same-work
  relations (IsSupplementTo, IsDocumentedBy, IsDescribedBy, IsPublishedIn,
  and inverses). Cited / citing / part DOIs no longer merge studies.
- Not done: `_resolve_sequence_source_identifiers` still resolves accessions
  harvested from fetched samples. User (2026-11-10): many studies were found
  sample-first, so legitimacy must NOT be judged from paper text.

## Finding 6: per-BioProject audit of the 989 studies with 2+ BioProjects

scripts/bioproject_audit.py -> marine_audit/bioproject_audit.csv.
User decision: target studies with >1 BioProject first, assess from there.

| verdict | BioProjects | samples |
|---|---|---|
| first_arrived | 989 | 87,389 |
| foreign_bug_chain | 2,214 | 267,212 |
| non_marine (>= 90 % classified non-marine) | 2,544 | 165,244 |
| jgi_gold (JGI: one BioProject per sequencing project) | 5,356 | 12,975 |
| paper_cited (accession in the paper text) | 1,591 | 60,552 |
| resolved_clean (resolution, study has no collisions) | 592 | 10,028 |
| review | 514 | 22,018 |

Spot check of foreign_bug_chain: cleft-lip GWAS, tick salivary glands, HMP
reference genome, grasshopper mouse, twins gut -- all foreign.
After dropping foreign_bug_chain + non_marine: 375 studies keep 1
BioProject, 331 keep 2, 283 keep 3+ (mostly JGI GOLD / reanalysis).

## Finding 7: mapping rules (gap 2)

scripts/faire_strict/mapping_inventory.py (10,412 raw name/level pairs) and
mapping_candidates.py (MIxS 7.0.1 pinned in reference/mixs/).
Sample level: 10,046 raw names, 64 mapped (38 % of sample facts).
134 candidate rules (excluding samp_name) add ~108k sample-term cells:
ENA spellings of lat/lon/date/depth/country, and FAIRe/MIxS-named
attributes with no sample-level rule at all (samp_vol_we_dna_ext,
samp_store_temp/dur/loc, nucl_acid_ext, size_frac_low, ph_meth, temp, alt,
local environmental context, ...). Other levels: almost all bibliographic
or bookkeeping; experiment_run library_layout (1,011) is unmapped.

Upper bound with all rules, raw value present (149,234 samples classified
marine/freshwater by organism/host/isolation source):
lat/lon 86 %, date 95 %, geo_loc_name 93 %, env_medium 93 %,
env_broad_scale 54 %, depth 48 %. Over all 1.01M samples: 33 / 44 / 45 /
40 / 21 / 12 %. The data is there; contamination and unmapped studies are
what pull the percentages down.
