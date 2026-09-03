"""Deterministic extraction of downstream analysis techniques.

This field is intentionally not LLM-generated. It scans methods-like sections
for a controlled list of downstream/statistical analysis terms, normalizes
case/separator variants back to canonical labels, and pipe-joins matches in
first-seen order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.sources.base import RawFactCandidate

FIELD_NAME = "internal_downstream_analysis_techniques"

_METHODS_TITLE_RE = re.compile(
    r"\b(?:methods?|materials?\s+and\s+methods?|methodology|experimental\s+(?:procedures?|design)|"
    r"sample\s+(?:collection|preparation)|sampling|dna\s+extraction|rna\s+extraction|"
    r"molecular\s+methods?|statistical\s+(?:analysis|analyses|methods?)|statistics|data\s+analys(?:is|es)|"
    r"bioinformatics?|community\s+analysis|multivariate\s+analysis|machine\s+learning|"
    r"supplementary\s+methods?)\b",
    re.IGNORECASE,
)
_NON_METHODS_TITLE_RE = re.compile(
    r"\b(?:abstract|introduction|background|results?|discussion|conclusions?|references?|bibliography)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

@dataclass(frozen=True)
class _TechniquePattern:
    canonical: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class _TechniqueMatch:
    canonical: str
    start: int
    end: int


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_DOWNSTREAM_TECHNIQUE_TERMS: tuple[str, ...] = (
    "relative abundance", "proportional abundance", "alpha diversity", "beta diversity", "Richness",
    "species richness", "observed richness", "observed species", "observed ASVs", "observed OTUs",
    "Shannon", "Simpson", "inverse Simpson", "Chao1", "ACE richness", "Pielou evenness",
    "Faith's phylogenetic diversity", "phylogenetic diversity", "Fisher's alpha",
    "Berger-Parker index", "Good's coverage", "rarefaction", "rarefaction curve", "extrapolation",
    "species accumulation curve", "sample coverage", "coverage-based rarefaction", "Hill numbers",
    "PCA", "PCoA", "NMDS", "multidimensional scaling", "correspondence analysis", "CCA", "RDA",
    "dbRDA", "detrended correspondence analysis", "canonical analysis of principal coordinates",
    "t-SNE", "UMAP", "Bray-Curtis", "Jaccard", "weighted UniFrac", "unweighted UniFrac",
    "generalized UniFrac", "Aitchison distance", "Euclidean distance", "Manhattan distance",
    "Canberra distance", "Kulczynski distance", "Morisita-Horn", "Sørensen-Dice",
    "Hellinger distance", "PERMANOVA", "nested PERMANOVA", "ANOSIM", "PERMDISP", "MRPP",
    "Mantel test", "partial Mantel test", "Procrustes analysis", "variation partitioning",
    "variance partitioning", "environmental vector fitting", "BIO-ENV", "differential abundance analysis",
    "ANCOM", "ANCOM-BC", "ALDEx2", "LEfSe", "indicator species analysis", "IndVal",
    "hierarchical clustering", "agglomerative clustering", "Ward clustering", "UPGMA", "k-means",
    "partitioning around medoids", "cluster analysis", "Dirichlet multinomial mixtures",
    "latent Dirichlet allocation", "co-occurrence network", "co-occurrence analysis",
    "association network", "correlation network", "ecological network", "network analysis",
    "network modularity", "centrality analysis", "degree centrality", "betweenness centrality",
    "Spearman correlation", "Pearson correlation", "Kendall correlation", "partial correlation",
    "correlation analysis", "linear regression", "logistic regression", "multinomial regression",
    "negative binomial regression", "beta regression", "generalized linear model",
    "generalized additive model", "mixed-effects model", "linear mixed model",
    "generalized linear mixed model", "zero-inflated model", "hurdle model",
    "multivariate generalized linear model", "distance-based linear model", "Random Forest",
    "support vector machine", "gradient boosting", "decision tree", "neural network",
    "deep learning", "k-nearest neighbors", "elastic net", "LASSO", "ridge regression",
    "partial least squares", "PLS-DA", "linear discriminant analysis",
    "quadratic discriminant analysis", "cross-validation", "k-fold cross-validation",
    "leave-one-out cross-validation", "ROC curve", "AUC", "accuracy", "precision", "recall",
    "F1 score", "confusion matrix", "feature importance", "permutation importance", "SHAP",
    "core microbiome", "core taxa", "core ASVs", "prevalence analysis", "occupancy analysis",
    "prevalence-abundance analysis", "detection frequency", "NRI", "NTI", "mean pairwise distance",
    "mean nearest taxon distance", "βMNTD", "βNTI", "null model analysis", "Raup-Crick",
    "normalized stochasticity ratio", "neutral community model", "niche breadth", "niche overlap",
    "source tracking", "source attribution", "microbial source tracking", "functional prediction",
    "functional inference", "metabolic inference", "habitat association", "environmental association",
    "distance-decay", "spatial autocorrelation", "dbMEM", "Moran's I", "Mantel correlogram",
    "temporal turnover", "community turnover", "beta diversity partitioning", "turnover component",
    "nestedness component", "additive diversity partitioning", "multiplicative diversity partitioning",
    "log transformation", "centered log-ratio transformation", "additive log-ratio transformation",
    "isometric log-ratio transformation", "Hellinger transformation", "arcsine square-root transformation",
    "presence-absence transformation", "total-sum scaling", "subsampling", "Shapiro-Wilk normality test",
    "Student's t-test", "ANOVA", "one-way ANOVA", "Tukey post hoc test", "Kruskal-Wallis test",
    "Dunn's test", "Conover-Iman test", "Fisher's LSD test", "Bonferroni correction",
    "Neighbor-joining", "phylogenetic tree", "bootstrap analysis", "forward selection", "backward selection",
)


_DOWNSTREAM_ALIASES: dict[str, tuple[str, ...]] = {
    "Bray-Curtis": ("Bray Curtis",),
    "Morisita-Horn": ("Morisita Horn",),
    "Sørensen-Dice": ("Sorensen-Dice", "Sørensen Dice", "Sorensen Dice"),
    "t-SNE": ("tSNE", "t SNE"),
    "βMNTD": ("beta MNTD",),
    "βNTI": ("beta NTI",),
    "Faith's phylogenetic diversity": ("Faiths phylogenetic diversity",),
    "Fisher's alpha": ("Fishers alpha",),
    "Good's coverage": ("Goods coverage",),
    "Moran's I": ("Morans I",),
    "Student's t-test": ("Students t-test", "Student t-test", "Student's t test", "Student t test"),
    "Fisher's LSD test": ("Fishers LSD test",),
    "Random Forest": ("random forests",),
    "Hellinger transformation": ("Hellinger transformed",),
    "presence-absence transformation": ("presence-absence", "presence absence"),
    "rarefaction": ("rarefied", "rarefying"),
    "rarefaction curve": ("rarefaction curves",),
    "PERMANOVA": ("permutation multivariate analysis of variance", "permutational multivariate analysis of variance"),
    "nested PERMANOVA": ("nested permutational multivariate analysis of variance",),
    "ANOSIM": ("analysis of similarity", "analysis of similarities"),
    "PERMDISP": ("multivariate homogeneity of group dispersion", "multivariate homogeneity of group dispersion/variance"),
    "NMDS": ("non-metric multidimensional scaling", "nonmetric multidimensional scaling"),
    "PCA": ("principal component analysis",),
    "PCoA": ("principal coordinate analysis", "principal coordinates analysis"),
    "RDA": ("redundancy analysis",),
    "dbRDA": ("distance-based redundancy analysis",),
    "CCA": ("canonical correspondence analysis",),
    "generalized linear model": ("generalised linear model",),
    "generalized additive model": ("generalised additive model",),
    "generalized linear mixed model": ("generalised linear mixed model",),
    "multivariate generalized linear model": ("multivariate generalised linear model",),
}


def _term_regex(term: str) -> re.Pattern[str]:
    pieces: list[str] = []
    for char in term:
        if char.isspace() or char in "-‐‑‒–—":
            pieces.append(r"[\s\-_‐‑‒–—]+")
        elif char in "'’":
            pieces.append(r"['’]?")
        else:
            pieces.append(re.escape(char))
    return _rx(rf"(?<![A-Za-z0-9]){''.join(pieces)}(?![A-Za-z0-9])")


def _pattern_for_term(term: str) -> re.Pattern[str]:
    variants = (term, *_DOWNSTREAM_ALIASES.get(term, ()))
    return _rx("|".join(_term_regex(variant).pattern for variant in variants))


_TECHNIQUE_PATTERNS: tuple[_TechniquePattern, ...] = tuple(
    _TechniquePattern(term, _pattern_for_term(term))
    for term in sorted(_DOWNSTREAM_TECHNIQUE_TERMS, key=len, reverse=True)
)


def _is_methods_section(title: str) -> bool:
    if not title or _NON_METHODS_TITLE_RE.search(title):
        return False
    return bool(_METHODS_TITLE_RE.search(title))


def _sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(normalized) if sentence.strip()]


def _matches_for_sentence(sentence: str) -> list[_TechniqueMatch]:
    matches: list[_TechniqueMatch] = []
    for spec in _TECHNIQUE_PATTERNS:
        for match in spec.pattern.finditer(sentence):
            matches.append(_TechniqueMatch(spec.canonical, match.start(), match.end()))
    matches.sort(key=lambda match: (match.start, -(match.end - match.start)))

    accepted: list[_TechniqueMatch] = []
    occupied: list[tuple[int, int]] = []
    for match in matches:
        if any(match.start < end and match.end > start for start, end in occupied):
            continue
        accepted.append(match)
        occupied.append((match.start, match.end))
    return accepted


def detect_downstream_analysis_techniques(
    texts: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    *,
    locator_prefix: str,
) -> list[RawFactCandidate]:
    techniques: list[str] = []
    evidence: list[str] = []
    seen: set[str] = set()

    for title, text in texts:
        if not _is_methods_section(title):
            continue
        sentences = _sentences(text)
        for sentence in sentences:
            for match in _matches_for_sentence(sentence):
                key = match.canonical.casefold()
                if key not in seen:
                    seen.add(key)
                    techniques.append(match.canonical)
                if sentence not in evidence:
                    evidence.append(sentence)

    if not techniques:
        return []
    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate=FIELD_NAME,
            raw_field_name=FIELD_NAME,
            raw_value=" | ".join(techniques),
            source_locator=f"{locator_prefix}:downstream_analysis_techniques",
            support_type=SupportType.EXPLICIT,
            evidence_quote=" | ".join(evidence),
            confidence_metadata={
                "detector": "deterministic_downstream_analysis_techniques",
                "source_quote_count": len(evidence),
            },
        )
    ]
