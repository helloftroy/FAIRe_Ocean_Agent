# Ontology versions used for the strict-FAIRe correction

Retrieved 2026-11-10. Used by scripts/faire_strict/ontology.py (rules R19
and R20 in correct_to_faire.md). Checksums are in SHA256SUMS.txt. The `.gz`
files are gzip copies of the downloads; the uncompressed SHA-256 is given below.

## ENVO (Environment Ontology)

- File: envo_2026-10-23.json.gz (obographs JSON)
- Downloaded from: http://purl.obolibrary.org/obo/envo.json
- Version IRI: http://purl.obolibrary.org/obo/envo/releases/2026-10-23/envo.json
- SHA-256 of the uncompressed download: 6e3c1f8352091d3b5575c97ba8e487220a8cf62456555a7628260bced00fda9d
- 4,454 classes, 447 of them obsolete.
- Note: in this release ENVO:00002149 is labelled "ocean water". FAIRe's
  v1.0.2 example still says "sea water [ENVO:00002149]". The new class
  "seawater" is ENVO:01001964.
- Citation: Buttigieg P.L. et al. (2016) The environment ontology in 2016:
  bridging domains with increased scope, semantic density, and
  interoperation. J Biomed Semantics 7:57. https://doi.org/10.1186/s13326-016-0097-6

## OBI (Ontology for Biomedical Investigations)

- File: obi_2026-07-27.owl.gz (OWL/XML)
- Downloaded from: http://purl.obolibrary.org/obo/obi.owl
- Version IRI: http://purl.obolibrary.org/obo/obi/2026-07-27/obi.owl
- SHA-256 of the uncompressed download: 8874abcca971ee26b4e4442198387916d788d14af3519e1040474908ed12b946
- Used for: the 39 named subclasses of "DNA sequencer" (OBI_0400103).
- Citation: Bandrowski A. et al. (2016) The Ontology for Biomedical
  Investigations. PLoS ONE 11(4): e0154556. https://doi.org/10.1371/journal.pone.0154556

## UBERON (three life cycle stage terms only)

- Folder: uberon_ols/. These are the OLS4 API responses for the UBERON
  ontology and for UBERON_0000113, UBERON_0034919, UBERON_0000069 and
  UBERON_0000104 (life cycle, FAIRe's branch for host_life_stage).
- Retrieved from https://www.ebi.ac.uk/ols4/api/ontologies/uberon on 2026-11-10.
- Version IRI (OLS): http://purl.obolibrary.org/obo/uberon/releases/2026-11-05/uberon.owl
- The full ontology was not downloaded: only these three terms are used
  (rule R24).
- Note: in this release UBERON_0000113 is labelled "post-juvenile adult
  stage"; "adult stage" is its exact synonym.
