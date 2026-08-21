from fair_ocean_agent.database.enums import EntityLevel, EntityRelationshipType, SupportType
from fair_ocean_agent.sources.sequence_file_heuristics import (
    ListedFile,
    SequenceDataStatus,
    classify_file_listing,
    synthesize_placeholder_sample_and_run_facts,
)


def test_confirmed_when_a_listed_file_is_directly_a_sequence_extension():
    files = [ListedFile("sample_R1.fastq.gz", 12_000_000), ListedFile("readme.txt", 100)]
    assert classify_file_listing(files) == SequenceDataStatus.CONFIRMED


def test_confirmed_when_archive_filename_contains_a_sequence_keyword():
    """Real case: 10.1002/edn3.184's Dryad dataset has
    Unfiltered_demultiplexed_fastq_files.zip alongside processed ASV/taxa
    CSV matrices -- the zip's own name should win over the CSVs."""
    files = [
        ListedFile("CoralITS2_acro_ASV_matrix.csv", 459_615),
        ListedFile("CoralITS2_acro_taxa_matrix.csv", 47_797),
        ListedFile("Unfiltered_demultiplexed_fastq_files.zip", 4_594_947_904),
    ]
    assert classify_file_listing(files) == SequenceDataStatus.CONFIRMED


def test_likely_when_large_archive_has_no_confirming_keyword():
    """Real case: 10.5281/zenodo.10381280's own files are Elas02.rar/
    MiFish.rar -- named after the primer/marker sets used, not "fastq" --
    but its own record metadata (checked separately) confirms these are the
    paper's real eDNA metabarcoding raw data. A keyword-only check would
    wrongly call this absent and risk dropping a paper with real data."""
    files = [ListedFile("Elas02.rar", 141_252_940), ListedFile("MiFish.rar", 259_064_594)]
    assert classify_file_listing(files) == SequenceDataStatus.LIKELY


def test_absent_when_only_small_or_non_archive_files_present():
    files = [ListedFile("poster.pdf", 500_000), ListedFile("README.md", 200)]
    assert classify_file_listing(files) == SequenceDataStatus.ABSENT


def test_absent_when_archive_present_but_too_small_to_trust():
    files = [ListedFile("supplementary_figures.zip", 40_000)]
    assert classify_file_listing(files) == SequenceDataStatus.ABSENT


def test_absent_for_empty_listing():
    assert classify_file_listing([]) == SequenceDataStatus.ABSENT


def test_synthesize_placeholder_facts_returns_empty_for_absent_status():
    assert synthesize_placeholder_sample_and_run_facts(repo="dryad", doi="10.5061/dryad.x", status=SequenceDataStatus.ABSENT) == []


def test_synthesize_placeholder_facts_produces_exactly_one_sample_and_one_run():
    """Per an explicit user request: "just one line in the sample and
    experiment metadata tables" -- not one row per file inside the
    dataset, no unzipping."""
    facts = synthesize_placeholder_sample_and_run_facts(
        repo="dryad", doi="10.5061/dryad.xksn02vdx", status=SequenceDataStatus.CONFIRMED
    )
    sample_ids = {f.entity_external_id for f in facts if f.entity_level == EntityLevel.SAMPLE}
    run_ids = {f.entity_external_id for f in facts if f.entity_level == EntityLevel.EXPERIMENT_RUN}
    assert sample_ids == {"internal:dryad:10.5061/dryad.xksn02vdx:sample"}
    assert run_ids == {"internal:dryad:10.5061/dryad.xksn02vdx:run"}


def test_synthesize_placeholder_facts_are_all_marked_inferred():
    facts = synthesize_placeholder_sample_and_run_facts(
        repo="zenodo", doi="10.5281/zenodo.10381280", status=SequenceDataStatus.LIKELY
    )
    assert all(f.support_type == SupportType.INFERRED for f in facts)


def test_synthesize_placeholder_run_links_back_to_the_sample_via_derived_from_sample():
    facts = synthesize_placeholder_sample_and_run_facts(
        repo="figshare", doi="10.6084/m9.figshare.1", status=SequenceDataStatus.CONFIRMED
    )
    run_fact = next(f for f in facts if f.entity_level == EntityLevel.EXPERIMENT_RUN)
    assert len(run_fact.entity_links) == 1
    link = run_fact.entity_links[0]
    assert link.entity_level == EntityLevel.SAMPLE
    assert link.external_identifier == "internal:figshare:10.6084/m9.figshare.1:sample"
    assert link.relationship_type == EntityRelationshipType.DERIVED_FROM_SAMPLE


def test_synthesize_placeholder_includes_materialsampleid_redirect_fact():
    """mapping/faire.py's own by-value redirect for materialSampleID needs
    a "sample_accession" fact at EntityLevel.SEQUENCING_RUN whose raw_value
    matches the SAMPLE entity's own external_identifier."""
    facts = synthesize_placeholder_sample_and_run_facts(
        repo="osf", doi="10.17605/OSF.IO/EZCUJ", status=SequenceDataStatus.CONFIRMED
    )
    redirect_fact = next(f for f in facts if f.fact_type_candidate == "sample_accession")
    assert redirect_fact.entity_level == EntityLevel.SEQUENCING_RUN
    assert redirect_fact.raw_value == "internal:osf:10.17605/OSF.IO/EZCUJ:sample"
