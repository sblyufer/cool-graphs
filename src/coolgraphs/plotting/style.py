"""Shared matplotlib style for the paper figures.

Colour-vision-deficiency-safe categorical palette (blue, orange, violet,
green), a single-hue sequential map for continuous parameters, and a
blue-grey-red diverging map for signed heat flux.

    from coolgraphs.plotting import style
    style.use()
    fig, ax = style.figure("single")
    ax.plot(x, y, **style.series(0, "GVAE"))
    style.save(fig, "figures/roc")   # writes .pdf and .png
"""
from __future__ import annotations

import os

# --- the validated palette -------------------------------------------------
# Hex values are from the reference design system; the ORDER below is the one
# that passed validation on a white surface. Changing the order changes the
# result -- re-run the validator if you do.
CATEGORICAL = ["#2a78d6", "#eb6834", "#6250d6", "#008300",
               # Slots 5-8 are the remaining reference hues, kept so a fifth
               # series has somewhere defined to go. They were NOT validated
               # as part of the leading four; if you need five or more, either
               # re-validate or (better) facet.
               "#e34948", "#1baf7a", "#eda100", "#e87ba4"]

SCATTER_2 = ["#2a78d6", "#eb6834"]            # all-pairs safe
SCATTER_3 = ["#2a78d6", "#eb6834", "#1baf7a"]  # all-pairs safe

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#2a78d6", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#3987e5", "#9ec5f4",
             "#e8e8e6",                        # neutral midpoint = "no flux"
             "#f3a3a2", "#e34948", "#c22f2e", "#8f1f1e"]

# SCATTER VARIANTS -- why these exist
# The ramps above are the design system's, and they are correct for a heatmap
# or a choropleth, where the lightest step fills a cell and "near zero" should
# recede. Rendered as 3-pixel dots on white paper they fail: the bottom two
# steps of SEQUENTIAL and the neutral midpoint of DIVERGING are so close to
# the page that those points simply vanish. For the flux figure that is the
# worst possible place to lose data -- near-zero flux is the physically
# ambiguous band the threshold argument is about, and a reader must be able to
# see how populated it is.
#
# These variants raise the floor so every point stays visible. The ordering,
# hues and monotonic lightness are unchanged.
SEQUENTIAL_SCATTER = SEQUENTIAL[1:]            # drop the near-white step
DIVERGING_SCATTER = ["#0d366b", "#256abf", "#3987e5", "#86b6ef",
                     "#a8a7a2",                # a grey you can actually see
                     "#ef8584", "#e34948", "#c22f2e", "#8f1f1e"]

# Ink. Text never wears a series colour; a coloured mark beside it carries
# identity instead.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d4"

# Greyscale-safe secondary encoding, paired 1:1 with CATEGORICAL.
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)),
              (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# NeurIPS text width is 5.5 in; a single column is 2.65 in.
WIDTHS = {"single": 2.65, "wide": 5.5, "half": 3.3, "full": 5.5}


def colormap(kind: str = "sequential", scatter: bool = True):
    """A matplotlib colormap from the validated ramps.

    ``scatter=True`` (the default) uses the raised-floor variants, because the
    overwhelming majority of colour-mapped figures here are scatter plots of
    the latent space. Pass ``scatter=False`` for a heatmap or an image, where
    the full ramp down to near-white is correct.
    """
    from matplotlib.colors import LinearSegmentedColormap
    stops = {
        ("sequential", True): SEQUENTIAL_SCATTER,
        ("sequential", False): SEQUENTIAL,
        ("diverging", True): DIVERGING_SCATTER,
        ("diverging", False): DIVERGING,
    }[(kind, bool(scatter))]
    return LinearSegmentedColormap.from_list(f"coolgraphs_{kind}", stops)


def use(font_scale: float = 1.0) -> None:
    """Apply the house style to matplotlib's global settings.

    Call once per script, before creating figures.
    """
    import matplotlib as mpl

    base = 8 * font_scale
    mpl.rcParams.update({
        # Match the paper's body text. mathtext in 'stix' sits beside Times
        # without the jarring default sans-serif math.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,

        # Recessive chrome: the data should be the darkest thing on the page.
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,

        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),

        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,

        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Type 42 keeps text as text in the PDF: selectable, searchable, and
        # accepted by publishers that reject Type 3 fonts.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figure(width: str = "single", height: float | None = None, **kw):
    """A correctly sized figure. ``width`` is 'single', 'half' or 'wide'."""
    import matplotlib.pyplot as plt
    w = WIDTHS.get(width, WIDTHS["single"])
    return plt.subplots(figsize=(w, height or w * 0.72), **kw)


def series(i: int, label: str | None = None, marker: bool = False) -> dict:
    """Keyword arguments for series ``i``: colour, linestyle, optional marker.

    Colour and dash pattern move together, so the figure stays readable in
    greyscale and under colour-vision deficiency without a second thought.
    """
    kw = {"color": CATEGORICAL[i % len(CATEGORICAL)],
          "linestyle": LINESTYLES[i % len(LINESTYLES)]}
    if label:
        kw["label"] = label
    if marker:
        kw["marker"] = MARKERS[i % len(MARKERS)]
        kw["markevery"] = max(1, 0)
    return kw


def finish(ax, xlabel=None, ylabel=None, title=None, legend=None, **kw):
    """Labels and a legend, applied consistently.

    ``legend`` defaults to showing one whenever two or more labelled artists
    exist -- a single series needs no legend box, because the axis label or
    caption already names it.
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", color=INK)
    handles, labels = ax.get_legend_handles_labels()
    if legend is None:
        legend = len(labels) >= 2
    if legend and labels:
        ax.legend(**{"loc": "best", **kw})
    return ax


def save(fig, stem: str, formats=("pdf", "png")) -> list:
    """Write the figure once per format. PDF for LaTeX, PNG for quick looks."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    out = []
    for f in formats:
        p = f"{stem}.{f}"
        fig.savefig(p, format=f)
        out.append(p)
    return out


def facet_topologies(topologies, width: str = "wide", ncols: int = 4,
                     sharex: bool = True, sharey: bool = True):
    """A grid of panels, one per topology.

    Eight topologies in one scatter cannot be made colour-safe -- no ordering
    of any eight hues clears the all-pairs separation floors. Faceting removes
    the problem instead of papering over it, and it reads better: per-topology
    structure is the comparison a reader actually wants to make.
    """
    import matplotlib.pyplot as plt
    topos = sorted(set(topologies))
    nrows = (len(topos) + ncols - 1) // ncols
    w = WIDTHS.get(width, WIDTHS["wide"])
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, w * 0.30 * nrows),
                             sharex=sharex, sharey=sharey, squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(topos):]:
        ax.set_visible(False)
    return fig, dict(zip(topos, flat))
