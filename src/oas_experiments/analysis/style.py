"""Shared figure style: thesis font sizes, a fixed colour per method family and
a fixed marker per protocol type.

Call :func:`apply` once before plotting (the figures CLI does).
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib as mpl

WIDTH = 6.3  # inches, the text width of the thesis

# Okabe-Ito without yellow and black (too light / no hue), plus wine and olive.
# Checked for lightness, chroma and colour-blind separation with a palette
# validator; the lighter colours always come with a legend.
PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
    "#882255",
    "#999933",
)
LINESTYLES = ("-", "--", ":", "-.")
PROTOCOL_MARKERS = {"warm": "o", "cold": "s"}
FALLBACK_MARKER = "D"

# Sizes for an 11pt thesis: figures are included at their natural width.
RC = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,  # embed TrueType, so text stays text in the PDF
}


def apply() -> None:
    """Set the thesis style for all following figures."""
    mpl.rcParams.update(RC)


def family_styles(families: Iterable[str]) -> dict[str, dict]:
    """Colour, first line style and marker fill per family, by sorted name.

    Families after the eighth reuse the colours with a different first line
    style and hollow markers, so no two families look the same.
    """
    styles = {}
    for i, family in enumerate(sorted(set(families))):
        styles[family] = {
            "color": PALETTE[i % len(PALETTE)],
            "dash": i // len(PALETTE),
            "fill": "full" if i < len(PALETTE) else "none",
        }
    return styles


def protocol_marker(protocol_type: str) -> str:
    """Marker of a protocol type (warm, cold)."""
    return PROTOCOL_MARKERS.get(protocol_type, FALLBACK_MARKER)
