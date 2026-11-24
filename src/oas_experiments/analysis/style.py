"""Shared figure style: thesis font sizes, a fixed colour per method family, a
fixed marker per protocol type and the order in which methods are listed.

Call :func:`apply` once before plotting (the figures CLI does).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import matplotlib as mpl
import pandas as pd

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
    # Computer Modern, like the LaTeX text (cmr10 ships with matplotlib)
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no minus sign of its own
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
    # cmr10 has a zero timestamp, which fontTools warns about on every save
    logging.getLogger("fontTools").setLevel(logging.ERROR)


_pinned: frozenset[str] = frozenset()


def pin_families(families: Iterable[str]) -> None:
    """Always assign colours over these families too, so figures drawn from a
    subset (e.g. one protocol) keep each family's colour of the whole
    experiment."""
    global _pinned
    _pinned = frozenset(families)


_order: tuple[str, ...] = ()


def pin_order(families: Iterable[str]) -> None:
    """List families in this order (e.g. the order of the config) in tables
    and legends; other families follow by name."""
    global _order
    _order = tuple(dict.fromkeys(families))


def family_rank(family: str) -> tuple[int, str]:
    """Sort key of a family: its place in :func:`pin_order`, then its name."""
    return (_order.index(family) if family in _order else len(_order), family)


def method_order(frame: pd.DataFrame) -> list[str]:
    """The methods of ``frame`` (with ``method`` and ``method_family``) in
    family order (:func:`family_rank`), then by name within a family."""
    pairs = frame[["method_family", "method"]].drop_duplicates()
    keys = sorted(
        (family_rank(f), m) for f, m in pairs.itertuples(index=False, name=None)
    )
    return list(dict.fromkeys(m for _, m in keys))


def family_styles(families: Iterable[str]) -> dict[str, dict]:
    """Colour, first line style and marker fill per family, by sorted name
    (over ``families`` and the families of :func:`pin_families`).

    Families after the eighth reuse the colours with a different first line
    style and hollow markers, so no two families look the same.
    """
    styles = {}
    for i, family in enumerate(sorted(set(families) | _pinned)):
        styles[family] = {
            "color": PALETTE[i % len(PALETTE)],
            "dash": i // len(PALETTE),
            "fill": "full" if i < len(PALETTE) else "none",
        }
    return styles


def protocol_marker(protocol_type: str) -> str:
    """Marker of a protocol type (warm, cold)."""
    return PROTOCOL_MARKERS.get(protocol_type, FALLBACK_MARKER)
