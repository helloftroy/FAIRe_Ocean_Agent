from __future__ import annotations

from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.models import MgnifyStudy


def is_marine_study(study: MgnifyStudy, config: SeedDiscoveryConfig) -> bool:
    text = " ".join(
        value
        for value in (
            study.biome,
            study.study_name,
            study.study_abstract,
        )
        if value
    ).casefold()
    if not text:
        return False
    if any(term.casefold() in text for term in config.rejected_biome_terms):
        return False
    if not config.accepted_biome_terms:
        return True
    return any(term.casefold() in text for term in config.accepted_biome_terms)


def experiment_type_status(study: MgnifyStudy, config: SeedDiscoveryConfig) -> str:
    if not study.experiment_types:
        return "experiment_type_unresolved"
    text = study.experiment_types.casefold()
    if any(term.casefold() in text for term in config.accepted_experiment_types):
        return "experiment_type_accepted"
    return "experiment_type_unresolved"
