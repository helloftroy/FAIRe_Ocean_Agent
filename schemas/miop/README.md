# Vendored MIOP schema

Source: https://github.com/BeBOP-OBON/miop
Commit: `be576bc7c9bce260905a1f6eec5c226104ad3ac4`

`terms.yaml` (canonical field definitions -- ~19 protocol-document-level
fields: methodology category, project, purpose, analyses, geographic
location, broad/local environmental context, environmental medium, target,
creator, materials/skills/time/personnel required, language, issued,
audience, publisher, hasVersion, license, maturity level) and `ranges.yaml`
(a generic "quantity value" pattern; not field-specific enumerations
despite the name) are vendored verbatim from `model/schema/`.
`UPSTREAM_README.md` is that repo's own README, kept for context.

MIOP describes metadata *about a protocol document itself* (who wrote it,
what it's for, what ontologies it references, license, version) -- this is
a fundamentally different scope from FAIRe (`schemas/faire/`), which
describes sample/sequencing data. See `standards/` (the compiled registry
built from this plus `schemas/bebop/`) for how the two relate.

Not physically merged with `schemas/bebop/` or `schemas/faire/` -- each
vendored directory stays a faithful, separately-attributed copy of its own
upstream repo. The only place MIOP, FAIRe, and BeBOP terms are combined is
the *compiled, derived* registry under `standards/compiled/`, produced by
`fair-ocean build-standards-registry` (see that command's help and
`src/fair_ocean_agent/standards/`).
