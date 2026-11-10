# FAIRe reference versions used for the NOAA fill-in

Retrieved 2026-11-10 for the strict-FAIRe correction of the
fair_ocean_agent export (see correct_to_faire.md in the repo root).
Checksums of every file are in SHA256SUMS.txt.

## FAIRe checklist (FAIR eDNA collaboration)

- Repository: https://github.com/FAIR-eDNA/FAIRe_checklist
- Commit: 061d8b62d98b07adaef523acf44c1af289f43fa3 (2026-10-06, main)
- Checklist version: v1.0.2. The repository README names
  `latest_checklist/FAIRe_checklist_v1.0.2.xlsx` as the authoritative
  checklist. v1.0.3 is listed in CHANGELOG.md but was withdrawn
  (README: "v.1.02 has been restored as the correct excel files").
- Per-term definitions: `slots/` (338 files). Three slots changed after
  the v1.0.2 spreadsheet: assay_name (source "New"), eventDate (adds
  MIxS URI), pcr2_plate_id (moved to experimentRunMetadata).
- Citation: Takahashi M. et al. (2025) A metadata checklist and data
  formatting guidelines to make eDNA FAIR. Environmental DNA 7(3):
  e70100. https://doi.org/10.1002/edn3.70100
- Files here: FAIRe_checklist_v1.0.2.xlsx (authoritative),
  FAIRe_checklist_v1.0.2_FULLtemplate.xlsx, schema.yaml, classes.yaml,
  enums.yaml, slots/, README.md, CHANGELOG.md.

## FAIRe-NOAA checklist (NOAA AOML, FAIReSheets)

- Repository: https://github.com/aomlomics/FAIReSheets (MIT License)
- Commit: 47fddc3bcb49415d9ffcac3a6ea52e387901f300 (2026-11-09, main)
- Checklist: input/FAIRe_NOAA_checklist_v1.0.2.xlsx. Used for the Ocean
  DNA Explorer and edna2obis. Differs from FAIRe v1.0.2: adds 81 terms,
  drops 4, moves 38 terms between tables, changes the controlled
  vocabulary of 76 terms.
- Documentation: https://noaa-omics-dmg.readthedocs.io/en/latest/fairesheets.html
- Files here: the checklist, README.md, NOAA_config.yaml, and the
  FAIRe-to-NOAA conversion code (FAIRe2NOAA.py, FAIRe2NOAA_helpers.py)
  as a reference for how NOAA restructures FAIRe tables.

## FAIRe guidelines page

- URL: https://fair-edna.github.io/guidelines.html, retrieved 2026-11-10.
- Saved as FAIRe_guidelines/guidelines.html (original) and
  guidelines.txt (text only). Source of the missing-value terms (INSDC
  missing value vocabulary, exact wording), the " | " list separator,
  the "other: FREE TEXT" rule and ISO 8601 dates.
- The guidelines name FAIRe-fier (FAIR eDNA metadata verifier) as the
  official validator for projectMetadata and sampleMetadata.
