# MIxS (pinned 2026-11-10)

- `mixs.yaml`: GSC MIxS LinkML schema, version 7.0.1, from
  https://raw.githubusercontent.com/GenomicsStandardsConsortium/mixs/main/src/mixs/schema/mixs.yaml
  at commit 536af18e999c7296d8608f656a9d3cab388e7098 (2026-11-10).
- Used by scripts/faire_strict/mapping_candidates.py: MIxS slot names,
  titles and aliases, joined to FAIRe terms through the FAIRe registry's
  mixs exact_mappings, to find raw BioSample attribute spellings that have
  no mapping rule.
