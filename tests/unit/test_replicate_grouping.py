"""Tests for sources/replicate_grouping.py: the shared, source-agnostic
detection of biological-replicate sample groups from sample-name suffix
patterns, reused by supplement_parsing.py and ncbi.py."""
from fair_ocean_agent.sources.replicate_grouping import ReplicateSignal, detect_replicate_groups


def test_detects_explicit_rep_marker_group_in_ascending_rep_order():
    names = {"S1": "Site_A_rep2", "S2": "Site_A_rep1"}
    groups = detect_replicate_groups(names)
    assert len(groups) == 1
    assert groups[0].signal is ReplicateSignal.EXPLICIT_REP_MARKER
    assert groups[0].members == ("S2", "S1")  # rep1 before rep2, regardless of input order


def test_explicit_marker_is_case_insensitive_and_accepts_replicate_spelling():
    names = {"S1": "S01-REP_1", "S2": "S01_replicate2"}
    groups = detect_replicate_groups(names)
    assert len(groups) == 1
    assert set(groups[0].members) == {"S1", "S2"}


def test_bare_trailing_digit_suffix_is_not_treated_as_replicate():
    names = {"S1": "Sample1", "S2": "Sample12"}
    assert detect_replicate_groups(names) == []


def test_short_prefix_signal_is_off_by_default():
    """The signal exists but is opt-in only -- confirmed live it isn't
    safe as a shared default (supplement_parsing.py's own table-row IDs
    are a much more collision-prone namespace)."""
    names = {"S1": "E2", "S2": "E3", "S3": "E4"}
    assert detect_replicate_groups(names) == []


def test_short_prefix_signal_groups_developmental_stage_replicates_when_enabled():
    """Real gap found live (PMC10988111): a developmental-stage time
    series named samples by a short stage-code prefix (P polyp, ES early
    strobila, AS advanced strobila, E ephyra) plus a bare replicate
    number, no separator at all -- e.g. "E2"/"E3". Only sources/ncbi.py
    opts into this, for real BioSample sample_name/title text."""
    names = {"S1": "E2", "S2": "E3", "S3": "E4", "S4": "AS1", "S5": "AS2"}
    groups = detect_replicate_groups(names, include_short_prefix_signal=True)
    by_members = {group.members: group.signal for group in groups}
    assert by_members == {
        ("S1", "S2", "S3"): ReplicateSignal.SHORT_PREFIX_NUMBER_SUFFIX,
        ("S4", "S5"): ReplicateSignal.SHORT_PREFIX_NUMBER_SUFFIX,
    }


def test_short_prefix_signal_groups_single_letter_sample_categories_when_enabled():
    names = {"S1": "P1", "S2": "P2", "S3": "P3"}
    groups = detect_replicate_groups(names, include_short_prefix_signal=True)
    assert len(groups) == 1
    assert groups[0].signal is ReplicateSignal.SHORT_PREFIX_NUMBER_SUFFIX
    assert groups[0].members == ("S1", "S2", "S3")


def test_short_prefix_signal_groups_descriptive_prefix_replicates_when_enabled():
    """Real gap found live (another gold paper): a biofilm sample series
    named by a descriptive, non-generic prefix with no separator at all
    before the bare replicate number -- "PB_Biofilm1"/"PB_Biofilm2"/
    "PB_Biofilm3" -- was missed entirely because the signal's base was
    originally capped at 1-2 letters. The base has no length cap now, only
    the "S"/generic-filler-word exclusions."""
    names = {"S1": "PB_Biofilm1", "S2": "PB_Biofilm2", "S3": "PB_Biofilm3"}
    groups = detect_replicate_groups(names, include_short_prefix_signal=True)
    assert len(groups) == 1
    assert groups[0].signal is ReplicateSignal.SHORT_PREFIX_NUMBER_SUFFIX
    assert groups[0].members == ("S1", "S2", "S3")


def test_short_prefix_signal_groups_embedded_number_with_constant_suffix_when_enabled():
    names = {"S1": "T_C1P", "S2": "T_C2P", "S3": "T_C3P"}
    groups = detect_replicate_groups(names, include_short_prefix_signal=True)
    assert len(groups) == 1
    assert groups[0].signal is ReplicateSignal.SHORT_PREFIX_NUMBER_SUFFIX
    assert groups[0].members == ("S1", "S2", "S3")


def test_short_prefix_signal_keeps_different_trailing_suffixes_separate():
    names = {"S1": "T_C1P", "S2": "T_C2Q", "S3": "T_C3P"}
    groups = detect_replicate_groups(names, include_short_prefix_signal=True)
    assert len(groups) == 1
    assert groups[0].members == ("S1", "S3")


def test_short_prefix_signal_still_excludes_s_prefix_even_when_enabled():
    """"S1"/"S2" is a real, common generic sequential sample-ID convention
    (already a tested exclusion for the default/table-parsing path) -- it
    must stay excluded even for a caller that opts into the short-prefix
    signal, not just when the signal is off."""
    names = {"S1": "S1", "S2": "S2"}
    assert detect_replicate_groups(names, include_short_prefix_signal=True) == []


def test_short_prefix_signal_still_excludes_full_word_bases_when_enabled():
    names = {"S1": "Sample1", "S2": "Sample12"}
    assert detect_replicate_groups(names, include_short_prefix_signal=True) == []


def test_detects_trailing_number_suffix_after_space_or_underscore_by_exact_prefix():
    names = {"S1": "LM_2", "S2": "LM 1", "S3": "LMM 2", "S4": "LMM 1"}
    groups = detect_replicate_groups(names)

    by_members = {group.members: group.signal for group in groups}
    assert by_members == {
        ("S2", "S1"): ReplicateSignal.TRAILING_NUMBER_SUFFIX,
        ("S4", "S3"): ReplicateSignal.TRAILING_NUMBER_SUFFIX,
    }


def test_trailing_number_suffix_ignores_generic_sample_titles():
    names = {"S1": "sample 1", "S2": "sample 2", "S3": "BioSample_3"}
    assert detect_replicate_groups(names) == []


def test_singleton_explicit_marker_produces_no_group():
    names = {"S1": "Site_A_rep1"}
    assert detect_replicate_groups(names) == []


def test_letter_suffix_group_detected_with_three_consecutive_members():
    names = {"S1": "Sample_A", "S2": "Sample_B", "S3": "Sample_C"}
    groups = detect_replicate_groups(names)
    assert len(groups) == 1
    assert groups[0].signal is ReplicateSignal.TRAILING_LETTER_SUFFIX
    assert groups[0].members == ("S1", "S2", "S3")


def test_letter_suffix_pair_below_minimum_group_size_is_not_grouped():
    """Guards the "Station_A"/"Station_B" false-positive case: two
    genuinely different sites must not be merged into one replicate
    group just because their names differ by a trailing letter."""
    names = {"S1": "Station_A", "S2": "Station_B"}
    assert detect_replicate_groups(names) == []


def test_letter_suffix_non_consecutive_letters_not_grouped():
    names = {"S1": "Site_A", "S2": "Site_C", "S3": "Site_F"}
    assert detect_replicate_groups(names) == []


def test_letter_suffix_signal_can_be_disabled():
    names = {"S1": "Sample_A", "S2": "Sample_B", "S3": "Sample_C"}
    assert detect_replicate_groups(names, include_letter_suffix_signal=False) == []


def test_group_members_are_the_callers_own_identifiers_not_the_name_strings():
    """Confirms the dict-key contract NCBI's adapter relies on: grouping
    runs against the *name* values, but the returned members are the
    *keys* (e.g. BioSample accessions), never the name strings."""
    names = {"SAMN1": "Site_A_rep1", "SAMN2": "Site_A_rep2"}
    groups = detect_replicate_groups(names)
    assert groups[0].members == ("SAMN1", "SAMN2")


def test_name_matched_by_explicit_marker_is_not_reconsidered_for_letter_suffix():
    names = {"S1": "Site_rep1", "S2": "Site_rep2", "S3": "Site_A", "S4": "Site_B", "S5": "Site_C"}
    groups = detect_replicate_groups(names)
    signals = {g.signal for g in groups}
    assert signals == {ReplicateSignal.EXPLICIT_REP_MARKER, ReplicateSignal.TRAILING_LETTER_SUFFIX}
