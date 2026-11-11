#!/usr/bin/env python3
"""Manuscript figure: how completely ecologically important FAIRe terms
are filled in the strict-FAIRe copy.

Reads fill_summary.csv (scripts/faire_strict/fill_summary.py) and writes
fill_ecology.{pdf,svg,png} next to it. Two panels: sample-level terms
(% of samples) and study / library-level terms (% of studies or libraries).
Each bar: FAIRe-conforming value | present but non-standard ("other:") |
not reported, paper paywalled (study's article full text never retrieved;
needs the ACCESS_CSV columns of fill_summary.py) | not reported. No title
baked in (caption goes in the manuscript).

Usage: python scripts/faire_strict/plot_fill.py FILL_SUMMARY_CSV
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
SURFACE = "#ffffff"
C_VALUE = "#2a78d6"   # categorical slot 1
C_OTHER = "#eb6834"   # categorical slot 2 (validated pair)
C_PAYWALL = "#a9a8a1"  # neutral, darker: not reported, paper paywalled
C_NONE = "#e7e6e1"    # neutral track: not reported

# (group, [(table, term, label)]). Order = reading order.
SAMPLE_GROUPS = [
    ("Where and when", [
        ("sampleMetadata", "decimalLatitude", "Latitude / longitude"),
        ("sampleMetadata", "geo_loc_name", "Geographic location"),
        ("sampleMetadata", "eventDate", "Collection date"),
        ("sampleMetadata", "minimumDepthInMeters", "Depth"),
    ]),
    ("Environment", [
        ("sampleMetadata", "env_broad_scale", "Broad-scale environment"),
        ("sampleMetadata", "env_local_scale", "Local environment"),
        ("sampleMetadata", "env_medium", "Environmental medium"),
        ("sampleMetadata", "host_species", "Host species"),
    ]),
    ("Collection and storage", [
        ("sampleMetadata", "samp_collect_device", "Collection device"),
        ("sampleMetadata", "size_frac", "Filter pore size"),
        ("sampleMetadata", "filter_material", "Filter material"),
        ("sampleMetadata", "samp_store_temp", "Storage temperature"),
        ("sampleMetadata", "samp_store_sol", "Storage solution"),
        ("sampleMetadata", "samp_store_dur", "Storage duration"),
    ]),
    ("Nucleic acid extraction", [
        ("sampleMetadata", "samp_vol_we_dna_ext", "Volume / mass extracted"),
        ("sampleMetadata", "nucl_acid_ext_kit", "Extraction kit"),
        ("sampleMetadata", "nucl_acid_ext_lysis", "Lysis method"),
    ]),
]
STUDY_GROUPS = [
    ("Amplification", [
        ("projectMetadata", "target_gene", "Target gene"),
        ("projectMetadata", "target_subfragment", "Target region"),
        ("projectMetadata", "pcr_primer_forward", "Forward primer sequence"),
        ("projectMetadata", "pcr_primer_reverse", "Reverse primer sequence"),
        ("projectMetadata", "ampliconSize", "Amplicon size"),
        ("projectMetadata", "annealingTemp", "Annealing temperature"),
        ("projectMetadata", "pcr_cycles", "PCR cycles"),
        ("projectMetadata", "pcr_rep", "PCR replicates"),
    ]),
    ("Controls", [
        ("projectMetadata", "neg_cont_0_1", "Negative controls reported"),
        ("projectMetadata", "pos_cont_0_1", "Positive controls reported"),
    ]),
    ("Sequencing", [
        ("projectMetadata", "platform", "Platform"),
        ("projectMetadata", "instrument", "Instrument"),
        ("projectMetadata", "seq_kit", "Sequencing kit"),
    ]),
    ("Bioinformatics", [
        ("projectMetadata", "otu_seq_comp_appr", "Clustering / ASV approach"),
        ("projectMetadata", "otu_db", "Reference database"),
    ]),
    ("Raw data (per library)", [
        ("experimentRunMetadata", "seq_run_id", "Sequence run accession"),
        ("experimentRunMetadata", "filename", "Read file name"),
    ]),
]


def load(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with path.open(newline="") as handle:
        out = {}
        for r in csv.DictReader(handle):
            n = int(r["rows"])
            out[(r["table"], r["term"])] = {
                "value": int(r["value"]) / n * 100,
                "other": int(r["other"]) / n * 100,
                "paywall": int(r.get("missing_no_fulltext") or 0) / n * 100,
                "rows": n,
            }
        return out


def draw_panel(ax, groups, data, unit_note: str, n_libs_note: str = "") -> None:
    y = 0.0
    ticks, labels, terms = [], [], []
    group_rows = []
    bar_h = 0.62
    for g_index, (group, items) in enumerate(groups):
        if g_index:
            y += 0.55  # gap between groups
        group_rows.append((y - 0.05, group))
        y += 0.75
        for table, term, label in items:
            d = data[(table, term)]
            v, o, pw = d["value"], d["other"], d["paywall"]
            ax.barh(y, 100, height=bar_h, color=C_NONE, linewidth=0, zorder=1)
            if v > 0:
                ax.barh(y, v, height=bar_h, color=C_VALUE, linewidth=0, zorder=2)
            if o > 0:
                ax.barh(y, o, left=v, height=bar_h, color=C_OTHER, linewidth=0, zorder=2)
                # 2px-style surface gap between the two fills
                if v > 0:
                    ax.plot([v, v], [y - bar_h / 2, y + bar_h / 2], color=SURFACE, lw=0.9, zorder=3,
                            solid_capstyle="butt")
            if pw > 0:
                ax.barh(y, pw, left=v + o, height=bar_h, color=C_PAYWALL, linewidth=0, zorder=2)
                if v + o > 0:
                    ax.plot([v + o, v + o], [y - bar_h / 2, y + bar_h / 2], color=SURFACE, lw=0.9, zorder=3,
                            solid_capstyle="butt")
            present = v + o
            text = f"{present:.0f}%" if present >= 1 else ("<1%" if present > 0 else "0%")
            ax.text(101.5, y, text, va="center", ha="left", fontsize=6.5, color=INK, zorder=4)
            ticks.append(y)
            labels.append(label)
            terms.append(term)
            y += 1.0
    ax.set_ylim(y - 0.3, -0.6)
    ax.set_xlim(0, 100)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7, color=INK)
    ax.tick_params(axis="y", length=0, pad=4)
    for tick_y, term in zip(ticks, terms):
        ax.annotate(term, xy=(0, tick_y), xycoords=("axes fraction", "data"), xytext=(-4, -6.2),
                    textcoords="offset points", ha="right", va="center", fontsize=5,
                    color=INK_3, family="Menlo", annotation_clip=False)
    for gy, group in group_rows:
        group = group.format(n_libs=n_libs_note)
        ax.annotate(group, xy=(0, gy), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7,
                    fontweight="bold", color=INK, annotation_clip=False)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=6.5, color=INK_2)
    ax.tick_params(axis="x", length=2, width=0.5, color=INK_3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_3)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_xlabel(unit_note, fontsize=7, color=INK_2, labelpad=4)


def main() -> None:
    path = Path(sys.argv[1])
    data = load(path)
    n_samples = data[("sampleMetadata", "samp_name")]["rows"]
    n_studies = data[("projectMetadata", "project_id")]["rows"]
    n_libs = data[("experimentRunMetadata", "lib_id")]["rows"]
    fig = plt.figure(figsize=(7.2, 5.2), facecolor=SURFACE)
    left = fig.add_axes([0.205, 0.115, 0.25, 0.77])
    right = fig.add_axes([0.695, 0.115, 0.25, 0.77])
    draw_panel(left, SAMPLE_GROUPS, data, f"% of samples (n = {n_samples:,})")
    draw_panel(right, STUDY_GROUPS, data,
               f"% of studies (n = {n_studies:,})\nraw data: % of libraries (n = {n_libs:,})")
    for ax, letter in ((left, "a"), (right, "b")):
        ax.text(-0.80, 1.035, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
                color=INK, va="bottom")
    handles = [
        Patch(color=C_VALUE, label="FAIRe-conforming value"),
        Patch(color=C_OTHER, label="Reported, non-standard (other:)"),
        Patch(color=C_PAYWALL, label="Not reported, paper paywalled"),
        Patch(color=C_NONE, label="Not reported"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.985), ncol=4, frameon=False,
               fontsize=7, handlelength=1.2, handleheight=0.9, columnspacing=1.4, labelcolor=INK)
    out = path.parent / "fill_ecology"
    for ext in ("pdf", "svg"):
        fig.savefig(f"{out}.{ext}", facecolor=SURFACE)
    fig.savefig(f"{out}.png", dpi=300, facecolor=SURFACE)
    print(f"wrote {out}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
