"""Deterministic extraction of downstream sequence-analysis techniques.

This field intentionally uses a conservative methods-only matcher. The goal is
to capture analyses applied to processed sequence-derived data (OTU/ASV tables,
taxonomic abundances, community matrices, etc.), while avoiding statistical
methods applied only to unrelated environmental variables.
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
_SEQUENCE_DATA_CONTEXT_RE = re.compile(
    r"\b(?:otu|otus|asv|asvs|amplicon\s+sequence\s+variants?|sequence\s+variants?|"
    r"taxonomic\s+(?:abundance|composition|profiles?|table|matrix)|taxa\s+abundance|"
    r"relative\s+abundance|abundance\s+(?:table|matrix|data)|feature\s+table|"
    r"biom\s+table|community\s+(?:matrix|composition|structure|profiles?|data)|"
    r"microbial\s+communities|microbiome|metabarcod|metagenom|gene\s+abundance|"
    r"functional\s+abundance|sequence-derived|sequencing\s+data|reads?|"
    r"taxonomic\s+groups?|phyloseq|bray[-\s]+curtis\s+dissimilarit(?:y|ies)|"
    r"multivariate\s+statistical\s+approaches?)\b",
    re.IGNORECASE,
)
_ENV_ONLY_CONTEXT_RE = re.compile(
    r"\b(?:environmental|physicochemical|physico-chemical|chemistry|nutrients?|temperature|"
    r"salinity|oxygen|ph|chlorophyll)\s+(?:variables?|parameters?|measurements?|data)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _TechniquePattern:
    canonical: str
    pattern: re.Pattern[str]
    context_required: re.Pattern[str] | None = None


@dataclass(frozen=True)
class _TechniqueMatch:
    canonical: str
    start: int
    end: int


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_TECHNIQUE_PATTERNS: tuple[_TechniquePattern, ...] = (
    _TechniquePattern("Hellinger", _rx(r"\bhellinger(?:[-\s]+transform(?:ed|ation)?)?\b")),
    _TechniquePattern("CLR", _rx(r"\b(?:cent(?:e|re)red\s+log[-\s]?ratio|clr[-\s]+transform(?:ed|ation)?|clr[-\s]+(?:abundance|data|counts?))\b")),
    _TechniquePattern("ALR", _rx(r"\b(?:additive\s+log[-\s]?ratio|alr[-\s]+transform(?:ed|ation)?)\b")),
    _TechniquePattern("ILR", _rx(r"\b(?:isometric\s+log[-\s]?ratio|ilr[-\s]+transform(?:ed|ation)?)\b")),
    _TechniquePattern("log1p transformation", _rx(r"\blog1p(?:[-\s]+transform(?:ed|ation)?)?\b")),
    _TechniquePattern("log transformation", _rx(r"\blog(?:arithmic)?[-\s]+transform(?:ed|ation)?\b")),
    _TechniquePattern("relative abundance", _rx(r"\b(?:relative\s+abundance|percentage\s+relative\s+abundance|proportions?)\b")),
    _TechniquePattern("total-sum scaling", _rx(r"\b(?:total[-\s]+sum\s+scal(?:ing|ed)|tss[-\s]+normalization)\b")),
    _TechniquePattern("CPM", _rx(r"\b(?:counts\s+per\s+million|cpm[-\s]+normaliz(?:ed|ation)|cpm[-\s]+counts?)\b")),
    _TechniquePattern("CSS", _rx(r"\b(?:cumulative\s+sum\s+scal(?:ing|ed)|css[-\s]+normaliz(?:ed|ation))\b")),
    _TechniquePattern("TMM", _rx(r"\b(?:trimmed\s+mean\s+of\s+m[-\s]+values|tmm[-\s]+normaliz(?:ed|ation))\b")),
    _TechniquePattern("RLE", _rx(r"\b(?:relative\s+log\s+expression|rle[-\s]+normaliz(?:ed|ation))\b")),
    _TechniquePattern("rarefaction", _rx(r"\braref(?:ied|action|ying)\b")),
    _TechniquePattern("presence-absence", _rx(r"\bpresence[-\s]+absence\b")),
    _TechniquePattern("z-score standardization", _rx(r"\b(?:z[-\s]?score(?:d)?|standardi[sz]ed)\b")),
    _TechniquePattern("Bray-Curtis", _rx(r"\bbray[-\s]+curtis\b")),
    _TechniquePattern("Jaccard", _rx(r"\bjaccard\b")),
    _TechniquePattern("Euclidean", _rx(r"\beuclidean\b")),
    _TechniquePattern("Aitchison", _rx(r"\baitchison\b")),
    _TechniquePattern("weighted UniFrac", _rx(r"\bweighted\s+unifrac\b")),
    _TechniquePattern("unweighted UniFrac", _rx(r"\bunweighted\s+unifrac\b")),
    _TechniquePattern("Canberra", _rx(r"\bcanberra\b")),
    _TechniquePattern("Morisita-Horn", _rx(r"\bmorisita[-\s]+horn\b")),
    _TechniquePattern("PCA", _rx(r"\b(?:principal\s+component\s+analysis|pca)\b")),
    _TechniquePattern("PCoA", _rx(r"\b(?:principal\s+coordinates?\s+analysis|principal\s+coordinate\s+analysis|pcoa)\b")),
    _TechniquePattern("NMDS", _rx(r"\b(?:non[-\s]+metric\s+multidimensional\s+scaling|nmds)\b")),
    _TechniquePattern("NMF", _rx(r"\b(?:non[-\s]+negative\s+matrix\s+factorization|nmf)\b")),
    _TechniquePattern("t-SNE", _rx(r"\b(?:t[-\s]?sne|t[-\s]+distributed\s+stochastic\s+neighbor\s+embedding)\b")),
    _TechniquePattern("UMAP", _rx(r"\bumap\b")),
    _TechniquePattern("MDS", _rx(r"\bmultidimensional\s+scaling\b")),
    _TechniquePattern("correspondence analysis", _rx(r"\bcorrespondence\s+analysis\b")),
    _TechniquePattern("DCA", _rx(r"\b(?:detrended\s+correspondence\s+analysis|dca)\b"), _rx(r"\b(?:ordination|correspondence|community|taxonomic|otu|asv)\b")),
    _TechniquePattern("MCA", _rx(r"\bmultiple\s+correspondence\s+analysis\b")),
    _TechniquePattern("PLS-DA", _rx(r"\b(?:partial\s+least\s+squares\s+discriminant\s+analysis|pls[-\s]?da)\b")),
    _TechniquePattern("PLS", _rx(r"\bpartial\s+least\s+squares\b")),
    _TechniquePattern("RDA", _rx(r"\b(?:redundancy\s+analysis|rda)\b"), _rx(r"\b(?:ordination|community|taxonomic|otu|asv|abundance|variation)\b")),
    _TechniquePattern("dbRDA", _rx(r"\b(?:dbrda|distance[-\s]+based\s+redundancy\s+analysis)\b")),
    _TechniquePattern("canonical correspondence analysis", _rx(r"\bcanonical\s+correspondence\s+analysis\b")),
    _TechniquePattern("CAP", _rx(r"\bcanonical\s+analysis\s+of\s+principal\s+coordinates\b")),
    _TechniquePattern("partial RDA", _rx(r"\bpartial\s+rda\b")),
    _TechniquePattern("partial CCA", _rx(r"\bpartial\s+canonical\s+correspondence\s+analysis\b")),
    _TechniquePattern("variation partitioning", _rx(r"\bvariation\s+partitioning\b")),
    _TechniquePattern("hierarchical clustering", _rx(r"\bhierarchical\s+cluster(?:ing|ed)?\b")),
    _TechniquePattern("k-means", _rx(r"\bk[-\s]+means\b")),
    _TechniquePattern("k-medoids", _rx(r"\bk[-\s]+medoids\b")),
    _TechniquePattern("PAM", _rx(r"\bpartitioning\s+around\s+medoids\b|\bpam\b"), _rx(r"\b(?:cluster|medoids|community|abundance)\b")),
    _TechniquePattern("DBSCAN", _rx(r"\bdbscan\b")),
    _TechniquePattern("Gaussian mixture model", _rx(r"\bgaussian\s+mixture\s+model\b")),
    _TechniquePattern("Dirichlet multinomial mixture", _rx(r"\bdirichlet\s+multinomial\s+mixture\b")),
    _TechniquePattern("spectral clustering", _rx(r"\bspectral\s+clustering\b")),
    _TechniquePattern("Ward clustering", _rx(r"\bward(?:'s)?\s+cluster(?:ing)?\b")),
    _TechniquePattern("community detection", _rx(r"\bcommunity\s+detection\b")),
    _TechniquePattern("observed richness", _rx(r"\bobserved\s+(?:richness|otus?|asvs?|features?)\b")),
    _TechniquePattern("Shannon", _rx(r"\bshannon(?:\s+(?:diversity|index))?\b")),
    _TechniquePattern("Simpson", _rx(r"\bsimpson(?:\s+(?:diversity|index))?\b")),
    _TechniquePattern("Pielou", _rx(r"\bpielou(?:'s)?(?:\s+evenness)?\b")),
    _TechniquePattern("Chao1", _rx(r"\bchao\s*1\b")),
    _TechniquePattern("ACE", _rx(r"\b(?:ace\s+(?:richness|estimator|index)|abundance[-\s]+based\s+coverage\s+estimator)\b")),
    _TechniquePattern("Faith phylogenetic diversity", _rx(r"\bfaith(?:'s)?\s+phylogenetic\s+diversity\b")),
    _TechniquePattern("Hill numbers", _rx(r"\bhill\s+numbers?\b")),
    _TechniquePattern("rarefaction curve", _rx(r"\brarefaction\s+curves?\b")),
    _TechniquePattern("PERMANOVA", _rx(r"\b(?:(?:permutation|permutational)\s+multivariate\s+analysis\s+of\s+variance|permanova)\b")),
    _TechniquePattern("ANOSIM", _rx(r"\b(?:analysis\s+of\s+similarit(?:y|ies)|anosim)\b")),
    _TechniquePattern("PERMDISP", _rx(r"\b(?:multivariate\s+homogeneity\s+of\s+group\s+dispersion(?:s)?(?:/variance)?|permdisp)\b")),
    _TechniquePattern("partial Mantel test", _rx(r"\bpartial\s+mantel\s+tests?\b")),
    _TechniquePattern("Mantel test", _rx(r"(?<!partial\s)\bmantel\s+tests?\b")),
    _TechniquePattern("Procrustes analysis", _rx(r"\bprocrustes\s+analysis\b")),
    _TechniquePattern("PROTEST", _rx(r"\bprotest\b")),
    _TechniquePattern("MRPP", _rx(r"\bmrpp\b")),
    _TechniquePattern("DESeq2", _rx(r"\bdeseq2\b")),
    _TechniquePattern("edgeR", _rx(r"\bedger\b")),
    _TechniquePattern("ANCOM-BC", _rx(r"\bancom[-\s]?bc\b")),
    _TechniquePattern("ANCOM", _rx(r"\bancom\b(?![-\s]?bc)")),
    _TechniquePattern("ALDEx2", _rx(r"\baldex2\b")),
    _TechniquePattern("LEfSe", _rx(r"\blefse\b")),
    _TechniquePattern("MaAsLin", _rx(r"\bmaaslin(?:2)?\b")),
    _TechniquePattern("indicator species analysis", _rx(r"\bindicator\s+species\s+analysis\b")),
    _TechniquePattern("SIMPER", _rx(r"\bsimper\b")),
    _TechniquePattern("Pearson correlation", _rx(r"\bpearson(?:'s)?\s+correlation\b")),
    _TechniquePattern("Spearman correlation", _rx(r"\bspearman(?:'s)?\s+(?:rank\s+)?correlation\b")),
    _TechniquePattern("SparCC", _rx(r"\bsparcc\b")),
    _TechniquePattern("SPIEC-EASI", _rx(r"\bspiec[-\s]?easi\b")),
    _TechniquePattern("FlashWeave", _rx(r"\bflashweave\b")),
    _TechniquePattern("CoNet", _rx(r"\bconet\b")),
    _TechniquePattern("CCLasso", _rx(r"\bcclasso\b")),
    _TechniquePattern("WGCNA", _rx(r"\bwgcna\b")),
    _TechniquePattern("proportionality analysis", _rx(r"\bproportionality\s+analysis\b")),
    _TechniquePattern("mutual information network", _rx(r"\bmutual\s+information\s+network\b")),
    _TechniquePattern("co-occurrence network", _rx(r"\bco[-\s]+occurrence\s+network\b")),
    _TechniquePattern("correlation network", _rx(r"\bcorrelation\s+network\b")),
    _TechniquePattern("graphical lasso", _rx(r"\bgraphical\s+lasso\b")),
    _TechniquePattern("probabilistic graphical model", _rx(r"\bprobabilistic\s+graphical\s+model\b")),
    _TechniquePattern("linear regression", _rx(r"\blinear\s+regression\b")),
    _TechniquePattern("multiple linear regression", _rx(r"\bmultiple\s+linear\s+regression\b")),
    _TechniquePattern("logistic regression", _rx(r"\blogistic\s+regression\b")),
    _TechniquePattern("generalized linear mixed model", _rx(r"\b(?:generalized|generalised)\s+linear\s+mixed\s+models?\b|\bglmm\b"), _rx(r"\b(?:model|regression|abundance|community|taxonomic)\b")),
    _TechniquePattern("generalized additive mixed model", _rx(r"\b(?:generalized|generalised)\s+additive\s+mixed\s+models?\b|\bgamm\b"), _rx(r"\b(?:model|spline|mgcv|regression|abundance|community)\b")),
    _TechniquePattern("generalized linear model", _rx(r"\b(?:generalized|generalised)\s+linear\s+models?\b|\bglm\b"), _rx(r"\b(?:model|regression|abundance|community|taxonomic)\b")),
    _TechniquePattern("generalized additive model", _rx(r"\b(?:generalized|generalised)\s+additive\s+models?\b|\bgam\b"), _rx(r"\b(?:model|spline|mgcv|predict|regression|abundance|community)\b")),
    _TechniquePattern("mixed-effects model", _rx(r"\bmixed[-\s]+effects?\s+models?\b")),
    _TechniquePattern("zero-inflated model", _rx(r"\bzero[-\s]+inflated\s+models?\b")),
    _TechniquePattern("hurdle model", _rx(r"\bhurdle\s+models?\b")),
    _TechniquePattern("negative binomial regression", _rx(r"\bnegative\s+binomial\s+regression\b")),
    _TechniquePattern("Poisson regression", _rx(r"\bpoisson\s+regression\b")),
    _TechniquePattern("random forest", _rx(r"\brandom\s+forests?\b")),
    _TechniquePattern("gradient boosting", _rx(r"\b(?:gradient\s+boosting|gradient\s+boosted\s+trees?|gradient\s+boosting\s+machine)\b")),
    _TechniquePattern("XGBoost", _rx(r"\bxgboost\b")),
    _TechniquePattern("LightGBM", _rx(r"\blightgbm\b")),
    _TechniquePattern("CatBoost", _rx(r"\bcatboost\b")),
    _TechniquePattern("support vector machine", _rx(r"\bsupport\s+vector\s+machines?\b")),
    _TechniquePattern("k-nearest neighbors", _rx(r"\bk[-\s]+nearest\s+neighbou?rs?\b")),
    _TechniquePattern("decision tree", _rx(r"\bdecision\s+trees?\b")),
    _TechniquePattern("extra trees", _rx(r"\bextra\s+trees?\b")),
    _TechniquePattern("AdaBoost", _rx(r"\badaboost\b")),
    _TechniquePattern("elastic net", _rx(r"\belastic\s+net\b")),
    _TechniquePattern("LASSO", _rx(r"\blasso\b")),
    _TechniquePattern("ridge regression", _rx(r"\bridge\s+regression\b")),
    _TechniquePattern("partial least squares regression", _rx(r"\bpartial\s+least\s+squares\s+regression\b")),
    _TechniquePattern("Gaussian process", _rx(r"\bgaussian\s+process(?:es)?\b")),
    _TechniquePattern("naive Bayes", _rx(r"\bna[iï]ve\s+bayes\b")),
    _TechniquePattern("multilayer perceptron", _rx(r"\bmultilayer\s+perceptron\b")),
    _TechniquePattern("artificial neural network", _rx(r"\bartificial\s+neural\s+network\b|\bann\b"), _rx(r"\b(?:neural|network|model|classifier)\b")),
    _TechniquePattern("CNN", _rx(r"\b(?:convolutional\s+neural\s+network|cnn)\b"), _rx(r"\b(?:neural|network|model|deep|classifier)\b")),
    _TechniquePattern("RNN", _rx(r"\b(?:recurrent\s+neural\s+network|rnn)\b"), _rx(r"\b(?:neural|network|model|deep|classifier)\b")),
    _TechniquePattern("LSTM", _rx(r"\blstm\b")),
    _TechniquePattern("transformer", _rx(r"\btransformer(?:\s+model)?\b")),
    _TechniquePattern("autoencoder", _rx(r"\bautoencoder\b")),
    _TechniquePattern("variational autoencoder", _rx(r"\bvariational\s+autoencoder\b")),
    _TechniquePattern("graph neural network", _rx(r"\bgraph\s+neural\s+network\b|\bgnn\b"), _rx(r"\b(?:neural|network|model|deep|classifier)\b")),
    _TechniquePattern("sequence embedding", _rx(r"\bsequence\s+embeddings?\b")),
    _TechniquePattern("protein language model", _rx(r"\bprotein\s+language\s+models?\b")),
    _TechniquePattern("DNA language model", _rx(r"\bdna\s+language\s+models?\b")),
    _TechniquePattern("Moran's I", _rx(r"\bmoran(?:'s)?\s+i\b")),
    _TechniquePattern("spatial autocorrelation", _rx(r"\bspatial\s+autocorrelation\b")),
    _TechniquePattern("distance-decay analysis", _rx(r"\bdistance[-\s]+decay\s+analysis\b")),
    _TechniquePattern("spatial interpolation", _rx(r"\bspatial\s+interpolation\b")),
    _TechniquePattern("kriging", _rx(r"\bkriging\b")),
    _TechniquePattern("species distribution model", _rx(r"\bspecies\s+distribution\s+models?\b")),
    _TechniquePattern("geographically weighted regression", _rx(r"\bgeographically\s+weighted\s+regression\b")),
    _TechniquePattern("spatial eigenvector mapping", _rx(r"\bspatial\s+eigenvector\s+mapping\b")),
    _TechniquePattern("MEM", _rx(r"\b(?:moran(?:'s)?\s+eigenvector\s+maps?|mem\s+spatial\s+variables?)\b")),
    _TechniquePattern("dbMEM", _rx(r"\bdbmem\b")),
    _TechniquePattern("time-series analysis", _rx(r"\btime[-\s]+series\s+analysis\b")),
    _TechniquePattern("cross-correlation", _rx(r"\bcross[-\s]+correlation\b")),
    _TechniquePattern("autocorrelation analysis", _rx(r"\bautocorrelation\s+analysis\b")),
    _TechniquePattern("seasonal decomposition", _rx(r"\bseasonal\s+decomposition\b")),
    _TechniquePattern("change-point analysis", _rx(r"\bchange[-\s]+point\s+analysis\b")),
    _TechniquePattern("dynamic time warping", _rx(r"\bdynamic\s+time\s+warping\b")),
    _TechniquePattern("autoregressive model", _rx(r"\bautoregressive\s+models?\b")),
)


def _is_methods_section(title: str) -> bool:
    if not title or _NON_METHODS_TITLE_RE.search(title):
        return False
    return bool(_METHODS_TITLE_RE.search(title))


def _sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(normalized) if sentence.strip()]


def _has_sequence_context(window: str) -> bool:
    if not _SEQUENCE_DATA_CONTEXT_RE.search(window):
        return False
    # A chemistry-only sentence can appear inside the same statistical methods
    # paragraph as sequence analyses. Keep it out unless molecular data are
    # explicitly present in that local window too.
    if _ENV_ONLY_CONTEXT_RE.search(window) and not re.search(r"\b(?:otu|asv|taxonomic|community|microbi|sequence|gene\s+abundance|abundance)\b", window, re.IGNORECASE):
        return False
    return True


def _matches_for_sentence(sentence: str, window: str) -> list[_TechniqueMatch]:
    matches: list[_TechniqueMatch] = []
    for spec in _TECHNIQUE_PATTERNS:
        if spec.context_required is not None and not spec.context_required.search(window):
            continue
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

    for section_index, (title, text) in enumerate(texts, start=1):
        if not _is_methods_section(title):
            continue
        sentences = _sentences(text)
        for index, sentence in enumerate(sentences):
            window = " ".join(sentences[max(0, index - 1) : min(len(sentences), index + 2)])
            if not _has_sequence_context(window):
                continue
            for match in _matches_for_sentence(sentence, window):
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
