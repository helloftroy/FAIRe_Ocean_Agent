# Vendored BeBOP protocol templates

Source: https://github.com/BeBOP-OBON/0_protocol_collection_template
Commit: `5a17dabb192e65d3d9ea39613492f8330c9e86fc`

The five `protocol_template_*.md` files (sampling, DNA extraction, PCR,
sequencing, bioinformatics) each carry a YAML frontmatter block with two
labeled sections: `# MIOP terms` (a fixed ~19-field protocol-document
description, matched against `schemas/miop/terms.yaml` by normalized
name -- see `standards/bebop_templates.py`) and `# FAIRe terms` (a
template-specific subset of FAIRe checklist fields relevant to that
protocol domain, matched by exact name against `schemas/faire/schema.yaml`).
This is exactly the "which FAIRe/MIOP field does each protocol section
use" information the compiled `standards/compiled/template_field_usage.csv`
captures.

`MIOP_definition.md` (a human-readable rendering of the MIOP fields, kept
for context) and `UPSTREAM_README.md` (that repo's own README) are also
vendored. **Not vendored**: `MIOP.md` (a blank per-protocol fill-in table
that just points at the real `miop` repo -- superseded here by
`schemas/miop/terms.yaml` itself) and the `pdf/` renderings (derived from
the markdown, not a separate source of information).

Every field across all 5 templates was checked against the vendored
FAIRe/MIOP schemas during Milestone 6 development and all resolved cleanly
(zero unmatched fields in either direction) -- so, as of this commit, there
are no `bebop:`-namespaced (template-specific, no upstream definition)
terms at all. See `standards/compiled/term_crosswalk.csv` for the
up-to-date, generated result of that check.
