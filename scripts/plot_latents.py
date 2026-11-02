#!/usr/bin/env python
"""3-D PCA view of the graph-level latent space (Fig. 3 of the paper).

    python scripts/plot_latents.py --run results/final/gvae_s0 --paper
    python scripts/plot_latents.py --run results/final/gvae_s0 \
        --colour label T beta0 alpha_range alpha_mean etaR

Reads the <split>.npz arrays written by scripts/train.py, so no model or GPU
is needed. One panel per --colour entry: 'label', 'topology', or any column
of the saved meta matrix (T, beta0, etaR, etaL, lamE, lamM, alpha_mean,
alpha_range, n_sites, heat_flux, ...). Continuous colours are percentile
clipped, and wide-range quantities such as heat flux are signed-log
transformed first (named on the colour bar). The explained variance of the
three components is printed. --method tsne / umap are available for
exploration; PCA is used in the paper.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# label and topology are categorical; the rest are recorded meta columns.
# The default is the four panels of the paper figure.
DEFAULT_COLOUR = ["label", "T", "beta0", "etaR"]

# Columns put through a signed log before colouring. These span orders of
# magnitude, so percentile clipping alone leaves every point but a handful at
# one end of the colour map.
#
# NO AUTOMATIC DETECTION. An earlier version inferred this from the ratio of
# the largest to smallest non-zero magnitude, which misfires on any quantity
# that CROSSES ZERO: alpha_mean ranges over about [-0.1, 1.2] eV, a span of
# 1.3, but passes near zero, so the ratio test saw a huge dynamic range that
# does not exist and log-scaled a linear quantity. A colour bar that says it is
# logarithmic when the quantity is not is worse than an ugly colour bar, so the
# rule is now explicit. Add more with --log-cols.
DEFAULT_SIGNED_LOG = {"heat_flux"}

# `meta` stores code identifiers; a figure should carry the paper's notation.
# TITLES are descriptive, for the panel heading; CBARS are the symbol and unit,
# for the colour bar, matching the published Fig. 3. Anything not listed falls
# back to the column name with underscores turned into spaces.
PRETTY_TITLE = {
    "label": "Energy flux label",
    "topology": "Topology",
    "T": "Temperature $T$",
    "beta0": "Inter-site coupling $\\beta_0$",
    "etaR": "Electrode coupling $\\eta^{0,R}$",
    "etaL": "Electrode coupling $\\eta^{0,L}$",
    "lamM": "Material decay $\\lambda_M$",
    "lamE": "Electrode decay $\\lambda_E$",
    "alpha_mean": "Mean on-site energy $\\bar{\\alpha}$",
    "alpha_std": "On-site energy spread $\\sigma_\\alpha$",
    "alpha_range": "On-site energy range $\\Delta\\alpha$",
    "alpha_min": "Minimum on-site energy",
    "alpha_max": "Maximum on-site energy",
    "n_sites": "System size $n$",
    "heat_flux": "Energy flux $F_E$",
}
PRETTY_CBAR = {
    "T": "$T$ (K)",
    "beta0": "$\\beta_0$ (eV)",
    "etaR": "$\\eta^{0,R}$ (eV)",
    "etaL": "$\\eta^{0,L}$ (eV)",
    "lamM": "$\\lambda_M$",
    "lamE": "$\\lambda_E$",
    "alpha_mean": "$\\bar{\\alpha}$ (eV)",
    "alpha_std": "$\\sigma_\\alpha$ (eV)",
    "alpha_range": "$\\Delta\\alpha$ (eV)",
    "alpha_min": "$\\alpha_{\\min}$ (eV)",
    "alpha_max": "$\\alpha_{\\max}$ (eV)",
    "n_sites": "$n$",
    "heat_flux": "$F_E$ (eV/s)",
}


def signed_log10(v):
    """sign(v) * log10(1 + |v|); handles the signed flux."""
    v = np.asarray(v, float)
    return np.sign(v) * np.log10(1.0 + np.abs(v))


def zoom_3d(ax, factor):
    """Shrink the empty margin a 3-D axes reserves around its projection box.

    matplotlib sizes a 3-D axes' bounding box far larger than the drawn cube,
    so bbox_inches='tight' still leaves a wide border and the plot ends up
    about a third of a subfigure. set_box_aspect(..., zoom=) fixes this on
    matplotlib >= 3.6; older versions are left alone rather than poked through
    the deprecated ax.dist.
    """
    if factor == 1.0:
        return
    try:
        ax.set_box_aspect(None, zoom=factor)
    except TypeError:
        pass
    except Exception:
        pass


def reduce_3d(z, method, seed=0, perplexity=30.0, n_neighbors=15):
    """[N, D] -> [N, 3], plus a caption saying how. D == 3 is used as-is."""
    if z.shape[1] == 3:
        return z, "latent dimensions (no projection: latent_dim is 3)"
    if method == "pca":
        from sklearn.decomposition import PCA
        p = PCA(n_components=3, random_state=seed)
        out = p.fit_transform(z)
        ev = p.explained_variance_ratio_
        cap = ("PCA (%.1f%% + %.1f%% + %.1f%% = %.1f%% variance)"
               % (100 * ev[0], 100 * ev[1], 100 * ev[2], 100 * ev[:3].sum()))
        return out, cap
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(perplexity, max(5.0, (len(z) - 1) / 3))
        out = TSNE(n_components=3, init="pca", random_state=seed,
                   perplexity=perp).fit_transform(z)
        return out, "t-SNE 3-D (perplexity %g, seed %d)" % (perp, seed)
    if method == "umap":
        try:
            import umap
        except ImportError:
            raise SystemExit(
                "umap-learn is not installed.\n"
                "Use --method pca or --method tsne, or install it with\n"
                "  pip install --break-system-packages umap-learn")
        out = umap.UMAP(n_components=3, random_state=seed,
                        n_neighbors=n_neighbors).fit_transform(z)
        return out, "UMAP 3-D (n_neighbors %d, seed %d)" % (n_neighbors, seed)
    raise SystemExit("unknown --method %r" % method)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="a run directory, e.g. results/final/gvae_s0")
    ap.add_argument("--split", default="test",
                    help="train/val/test/ext")
    ap.add_argument("--colour", "--color", nargs="+", default=DEFAULT_COLOUR,
                    dest="colour")
    ap.add_argument("--method", default="pca",
                    choices=["pca", "tsne", "umap"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--max-points", type=int, default=20000)
    ap.add_argument("--clip", type=float, nargs=2, default=(2.0, 98.0),
                    metavar=("LO", "HI"))
    ap.add_argument("--no-signed-log", action="store_true",
                    help="colour heat_flux linearly; a handful of extreme "
                         "values will then flatten every other point")
    ap.add_argument("--log-cols", nargs="+", default=None,
                    help="extra columns to signed-log, beyond heat_flux")
    ap.add_argument("--panel-cols", type=int, default=3,
                    help="columns in the combined panel figure; it wraps onto "
                         "further rows rather than growing one row wider")
    ap.add_argument("--paper", action="store_true",
                    help="figures sized for inclusion in LaTeX: no burnt-in "
                         "title, no projection footer (both belong in the "
                         "caption), tightened 3-D margins, smaller fonts")
    ap.add_argument("--zoom", type=float, default=None,
                    help="3-D box zoom; >1 shrinks the empty border. Default "
                         "1.0, or 1.12 under --paper. Much above 1.15 the box "
                         "overflows its cell in a panel and the axis labels "
                         "are clipped off the canvas")
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="multiply every font size; 1.3-1.5 for a 2x2 panel "
                         "scaled down to one column of a printed page")
    ap.add_argument("--font-family", default=None,
                    choices=["serif", "sans"],
                    help="serif is STIX, which is Times-metric and has math "
                         "glyphs that match its text, so it sits correctly in "
                         "a NeurIPS page; sans is DejaVu. Default: serif under "
                         "--paper, otherwise whatever the repo style sets")
    ap.add_argument("--bottom-margin", type=float, default=0.18,
                    help="figure fraction reserved below the bottom row for "
                         "its x label. Raise it if 'PC 1' is still clipped; "
                         "surplus is white space LaTeX can trim")
    ap.add_argument("--legend-y", type=float, default=0.44,
                    help="top of the legend box in axes coordinates; lower it "
                         "to clear the panel title and the point cloud")
    ap.add_argument("--legend-ncol", type=int, default=1,
                    help="2 turns the key into a thin horizontal strip rather "
                         "than a block")
    ap.add_argument("--legend-counts", action="store_true",
                    help="put the per-class n in the key. Off by default: it "
                         "roughly doubles the width of the box, and the counts "
                         "belong in the caption")
    ap.add_argument("--no-zlabel", action="store_true",
                    help="drop the 'PC 3' label. It shares the right-hand "
                         "side of a 3-D axes with the colour bar, so on a "
                         "tight panel one of the two has to go")
    ap.add_argument("--axis-labels", default=None,
                    choices=["long", "short"],
                    help="'Latent Dimension 1' or 'PC 1'. Default long, or "
                         "short under --paper, where the long form does not "
                         "fit a quarter-width panel")
    ap.add_argument("--elev", type=float, default=22.0,
                    help="3-D view elevation, degrees")
    ap.add_argument("--azim", type=float, default=-60.0,
                    help="3-D view azimuth, degrees")
    ap.add_argument("--point-size", type=float, default=6.0)
    ap.add_argument("--interactive", action="store_true",
                    help="also write a rotatable plotly HTML")
    ap.add_argument("--no-panel", action="store_true",
                    help="skip the combined multi-panel figure")
    ap.add_argument("--out-dir", default=None,
                    help="default: <run>/figures3d")
    args = ap.parse_args()

    # --paper only changes defaults, so an explicit flag still wins
    if args.zoom is None:
        args.zoom = 1.12 if args.paper else 1.0
    if args.axis_labels is None:
        args.axis_labels = "short" if args.paper else "long"
    log_cols = set(DEFAULT_SIGNED_LOG) | set(args.log_cols or ())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 registers '3d'

    from coolgraphs.training import artifacts
    from coolgraphs.utils.meta import META_COLS
    try:
        from coolgraphs.plotting import style
        style.use()
    except Exception:
        style = None

    # After style.use(), so it overrides the repo defaults rather than being
    # overridden by them. Matplotlib's out-of-the-box pairing draws body text
    # in one family and $math$ in another, which is the main reason a default
    # figure reads as unprofessional next to typeset text.
    fam = args.font_family or ("serif" if args.paper else None)
    if fam == "serif":
        plt.rcParams.update({
            "font.family": "serif",
            # STIX first: Times-metric, ships with matplotlib, so this needs
            # no font installed on the cluster.
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
        })
    elif fam == "sans":
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "mathtext.fontset": "dejavusans",
        })

    path = os.path.join(args.run, "%s.npz" % args.split)
    if not os.path.exists(path):
        have = sorted(f[:-4] for f in os.listdir(args.run)
                      if f.endswith(".npz")) if os.path.isdir(args.run) else []
        raise SystemExit(
            "no %s.npz in %s\n%s" % (args.split, args.run,
                                     "splits present: " + ", ".join(have)
                                     if have else
                                     "no saved splits; has the run finished?"))

    d = artifacts.load_split(path)
    z, y = d["z"], d["y"]
    if z is None or not len(z):
        raise SystemExit("%s carries no latents to plot" % path)
    print("%s: %d graphs, %d positives, z%s"
          % (path, len(z), int(y.sum()), z.shape))

    # Subsample BEFORE projecting, stratified on the label so the positives --
    # the point of the figure -- are not thinned away.
    sel = np.arange(len(z))
    if len(z) > args.max_points:
        rng = np.random.default_rng(args.seed)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        n_neg = max(0, args.max_points - len(pos))
        sel = np.sort(np.concatenate(
            [pos, rng.choice(neg, size=min(n_neg, len(neg)), replace=False)]))
        print("  subsampled to %d (%d positives kept in full)"
              % (len(sel), len(pos)))

    z3, caption = reduce_3d(z[sel], args.method, args.seed, args.perplexity,
                            args.n_neighbors)
    print("  projection: %s" % caption)

    out_dir = args.out_dir or os.path.join(args.run, "figures3d")
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def pretty(what):
        plain = what.replace("_", " ")
        return (PRETTY_CBAR.get(what, plain), PRETTY_TITLE.get(what, plain))

    def values_for(what):
        """-> (values, cmap, is_categorical, cbar_label, title) or None."""
        if what == "label":
            # categorical: no colour bar, so no bar label
            return y[sel].astype(float), "coolwarm", True, "", \
                PRETTY_TITLE["label"]
        if what == "topology":
            topo = d.get("topo")
            if topo is None:
                return None
            topo = np.asarray(topo)[sel]
            cats = sorted(set(topo.tolist()))
            codes = np.array([cats.index(t) for t in topo], dtype=float)
            return codes, "tab10", True, "", PRETTY_TITLE["topology"]
        if what not in META_COLS:
            print("  [skip] %s: not a recorded column. Available: %s"
                  % (what, ", ".join(META_COLS)))
            return None
        vals = artifacts.meta_column(d, what)[sel].astype(float)
        if not np.isfinite(vals).any():
            print("  [skip] %s: no finite values" % what)
            return None
        cbar, title = pretty(what)
        if what in log_cols and not args.no_signed_log:
            vals = signed_log10(vals)
            cbar = "signed $\\log_{10}$ " + cbar
            print("  [log] %s: signed log10 applied" % what)
        return vals, "viridis", False, cbar, title

    def draw(ax, what, vals, cmap, categorical):
        if what == "label":
            # Paper Fig. 3(a): positive flux blue, negative red.
            handles = []
            for v, lab, col, size, alpha, zo in (
                    (0, "Negative flux", "#C44E52",
                     args.point_size, 0.30, 1),
                    (1, "Positive flux", "#4C72B0",
                     args.point_size * 1.8, 0.90, 3)):
                m = vals == v
                ax.scatter(z3[m, 0], z3[m, 1], z3[m, 2], s=size, alpha=alpha,
                           c=col, linewidths=0, depthshade=True, zorder=zo)
                # Proxy handles: a legend built from the scatters themselves
                # inherits their 6-point markers and the negatives' alpha of
                # 0.30, which is a faint speck at print size. The key is
                # meant to be read, so its markers are drawn large and opaque
                # whatever the plotted points look like.
                txt = ("%s ($n{=}%d$)" % (lab, int(m.sum()))
                       if args.legend_counts else lab)
                handles.append(Line2D([], [], marker="o", linestyle="none",
                                      markersize=7 * args.font_scale,
                                      markerfacecolor=col,
                                      markeredgecolor="none", alpha=1.0,
                                      label=txt))
            # frameon and facecolor are set explicitly: the repository's
            # plotting style turns legend frames off, so framealpha alone
            # leaves the key floating over the points with nothing behind it.
            ax.legend(handles=handles, loc="upper left",
                      bbox_to_anchor=(0.0, args.legend_y),
                      fontsize=8.5 * args.font_scale, ncol=args.legend_ncol,
                      frameon=True, facecolor="white", edgecolor="0.75",
                      framealpha=1.0,
                      borderpad=0.4, handletextpad=0.4, labelspacing=0.35,
                      columnspacing=1.0)
            return None
        if categorical:
            topo = np.asarray(d.get("topo"))[sel]
            handles = []
            for i, t in enumerate(sorted(set(topo.tolist()))):
                m = topo == t
                h = ax.scatter(z3[m, 0], z3[m, 1], z3[m, 2], s=args.point_size,
                               alpha=0.7, linewidths=0, depthshade=True)
                handles.append(Line2D([], [], marker="o", linestyle="none",
                                      markersize=8 * args.font_scale,
                                      markerfacecolor=h.get_facecolor()[0],
                                      markeredgecolor="none", alpha=1.0,
                                      label=str(t)))
            ax.legend(handles=handles, loc="upper left",
                      bbox_to_anchor=(0.0, args.legend_y),
                      fontsize=8.5 * args.font_scale, ncol=2,
                      frameon=True, facecolor="white", edgecolor="0.75",
                      framealpha=1.0,
                      handletextpad=0.6, labelspacing=0.5)
            return None
        finite = np.isfinite(vals)
        lo, hi = np.percentile(vals[finite], args.clip)
        if hi <= lo:
            lo, hi = float(vals[finite].min()), float(vals[finite].max())
        sc = ax.scatter(z3[finite, 0], z3[finite, 1], z3[finite, 2],
                        c=np.clip(vals[finite], lo, hi), cmap=cmap,
                        s=args.point_size, alpha=0.75, linewidths=0,
                        depthshade=True)
        return sc

    S = args.font_scale

    def dress(ax, small=False):
        # Axis labels as the paper's Fig. 3 has them; ticks kept, since the
        # published figure shows the latent coordinate values. The long form
        # overruns a quarter-width panel, hence --axis-labels short.
        fmt = ("PC %d" if args.axis_labels == "short"
               else "Latent Dimension %d")
        fs = 11 * S
        # labelpad keeps the axis label clear of its own tick labels; without
        # it the 3-D tick text runs straight into the label.
        # Small labelpad on purpose. A 3-D axes draws its x/y labels well
        # outside its own rectangle, and the further out they sit the more
        # likely they fall off the canvas on the bottom row. Keeping them
        # close to the cube is the reliable half of the fix; the generous
        # bottom margin below is the other half.
        ax.set_xlabel(fmt % 1, fontsize=fs, labelpad=2 * S)
        ax.set_ylabel(fmt % 2, fontsize=fs, labelpad=2 * S)
        # The z label is the one that collides with a colour bar: at this
        # azimuth matplotlib draws the z axis on the right, which is where the
        # bar goes. Suppressing it entirely is cleaner than fighting for the
        # space -- all three axes are principal components and the caption
        # says so. --axis-labels long restores it.
        if args.axis_labels == "short" and not args.no_zlabel:
            ax.set_zlabel(fmt % 3, fontsize=fs, labelpad=10 * S)
        elif args.axis_labels != "short":
            ax.set_zlabel(fmt % 3, fontsize=fs, labelpad=10 * S)
        ax.tick_params(labelsize=10 * S, pad=2)
        ax.view_init(elev=args.elev, azim=args.azim)
        zoom_3d(ax, args.zoom)

    def add_cbar(fig, sc, ax, label):
        # fraction/pad rather than shrink/pad: fraction makes the colour bar
        # steal a fixed share of the axes, which keeps it outside the 3-D box
        # instead of on top of it. pad is generous because the z axis and its
        # label occupy the same side of the axes as the bar.
        cb = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.17, aspect=22)
        # Label horizontally ABOVE the bar, as the published Fig. 3 has it,
        # rather than rotated alongside: a rotated label competes with the z
        # axis for the same strip of space, and a symbol like $\eta^{0,R}$ is
        # much easier to read upright.
        cb.ax.set_title(label, fontsize=11 * S, pad=6 * S)
        cb.ax.tick_params(labelsize=9.5 * S)
        return cb

    panels = []
    run_name = os.path.basename(os.path.normpath(args.run))

    # --- one figure per colour, as the paper uses them -------------------
    for what in args.colour:
        got = values_for(what)
        if got is None:
            continue
        vals, cmap, categorical, label, title = got
        panels.append((what, vals, cmap, categorical, label, title))

        fig = plt.figure(figsize=(4.8, 4.2) if args.paper else (5.4, 4.8))
        ax = fig.add_subplot(111, projection="3d")
        sc = draw(ax, what, vals, cmap, categorical)
        if sc is not None:
            add_cbar(fig, sc, ax, label)
        dress(ax)
        # Under --paper the run id, split and projection go in the LaTeX
        # caption instead: burnt into the image they cannot be edited, they
        # duplicate the caption, and they are the first thing a reader crops.
        if not args.paper:
            ax.set_title("%s - %s - %s" % (run_name, args.split, what),
                         loc="left", fontsize=9)
            fig.text(0.99, 0.01, caption, ha="right", va="bottom",
                     fontsize=6.5, color="#666666")
        stem = os.path.join(out_dir, "latent3d_%s_%s_%s"
                            % (args.split, args.method, what))
        if style is not None and hasattr(style, "save"):
            try:
                written += style.save(fig, stem)
            except Exception:
                fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
                written.append(stem + ".png")
        else:
            fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
            fig.savefig(stem + ".pdf", bbox_inches="tight")
            written += [stem + ".png", stem + ".pdf"]
        plt.close(fig)
        print("  wrote %s" % stem)

    if not panels:
        raise SystemExit("nothing to plot: none of %s was usable"
                         % args.colour)

    # --- the combined panel, for a slide ---------------------------------
    # Wraps onto rows rather than growing one row wider: six panels in a single
    # row is a 29-inch figure that nothing can display or print usefully.
    if not args.no_panel:
        n = len(panels)
        ncols = max(1, min(args.panel_cols, n))
        nrows = int(math.ceil(n / ncols))
        cell = (4.6, 4.2) if args.paper else (4.8, 4.6)
        fig = plt.figure(figsize=(cell[0] * ncols, cell[1] * nrows))
        for i, (what, vals, cmap, categorical, label, title) in enumerate(panels):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            sc = draw(ax, what, vals, cmap, categorical)
            if sc is not None:
                add_cbar(fig, sc, ax, label)
            dress(ax, small=True)
            # The panel letter goes INSIDE the axes. set_title places text
            # above the axes bounding box, which for a 3-D axes sits well
            # clear of the drawn cube and lands on the row above.
            ax.text2D(0.5, 1.0, "(%s) %s" % ("abcdefgh"[i], title),
                      transform=ax.transAxes, fontsize=13 * S,
                      ha="center", va="top")
        if not args.paper:
            fig.suptitle("%s - %s - %s" % (run_name, args.split, caption),
                         fontsize=10 * S)
        # NOT tight_layout: it cannot measure a 3-D axes, so it under-allocates
        # and the next row's titles land on this row's tick labels. Explicit
        # spacing plus bbox_inches='tight' at save time is reliable here.
        # wspace must leave room for each panel's colour bar AND its tick
        # labels: at 0.02 the next column's axes background painted over the
        # left column's colour-bar numbers.
        #
        # bottom has to be generous. matplotlib's tight bounding box does not
        # reliably account for a 3-D axes' labels, so bbox_inches='tight' at
        # save time will NOT rescue an x label that falls below the canvas --
        # the room has to be reserved here. 0.06 was still clipping "PC 1" on
        # the bottom row once the fonts grew.
        # bottom and hspace are deliberately large. The bottom row's x label
        # kept falling off the canvas at 0.06, then 0.11, then 0.12; rather
        # than keep guessing where matplotlib places a 3-D axis label, this
        # reserves clearly more room than the label can need. Any surplus is
        # white space, which LaTeX can trim; a clipped label cannot be
        # recovered. hspace matches, so the top row's label has the same room.
        fig.subplots_adjust(left=0.01, right=0.99,
                            top=0.97 if args.paper else 0.93,
                            bottom=args.bottom_margin,
                            wspace=0.14, hspace=0.22)
        stem = os.path.join(out_dir, "latent3d_%s_%s_panel"
                            % (args.split, args.method))
        # NO bbox_inches='tight' here. Its crop box is computed from the
        # artists matplotlib's tight-bbox machinery knows about, and that
        # machinery does not account for a 3-D axes' axis labels. The margin
        # reserved by subplots_adjust above is therefore CROPPED AWAY again and
        # "PC 1" is cut off the bottom row -- which is why enlarging `bottom`
        # never fixed it. Saving the full canvas keeps the reserved margin.
        fig.savefig(stem + ".png", dpi=200)
        fig.savefig(stem + ".pdf")
        written += [stem + ".png", stem + ".pdf"]
        plt.close(fig)
        print("  wrote %s (%d x %d)" % (stem, nrows, ncols))

    # --- rotatable HTML ---------------------------------------------------
    if args.interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("  [skip] --interactive: plotly is not installed "
                  "(pip install --break-system-packages plotly)")
        else:
            fig = make_subplots(
                rows=1, cols=len(panels),
                specs=[[{"type": "scene"} for _ in panels]],
                subplot_titles=[p[5] for p in panels])
            for i, (what, vals, cmap, categorical, label, title) in enumerate(panels):
                scale = "RdBu" if what == "label" else "Viridis"
                fig.add_trace(
                    go.Scatter3d(
                        x=z3[:, 0], y=z3[:, 1], z=z3[:, 2], mode="markers",
                        marker=dict(size=max(1.5, args.point_size / 3.0),
                                    color=vals, colorscale=scale, opacity=0.6,
                                    showscale=True,
                                    colorbar=dict(title=label,
                                                  x=(i + 1) / len(panels) - 0.02)),
                        name=label, showlegend=False),
                    row=1, col=i + 1)
            fig.update_layout(title="%s - %s - %s"
                                    % (run_name, args.split, caption),
                              height=620, width=620 * len(panels))
            html = os.path.join(out_dir, "latent3d_%s_%s.html"
                                % (args.split, args.method))
            fig.write_html(html)
            written.append(html)
            print("  wrote %s" % html)

    with open(os.path.join(out_dir, "latent3d_figures.json"), "w") as f:
        json.dump({"run": args.run, "split": args.split, "method": args.method,
                   "caption": caption, "n_plotted": int(len(sel)),
                   "n_total": int(len(z)), "coloured_by": args.colour,
                   "signed_log": not args.no_signed_log,
                   "view": {"elev": args.elev, "azim": args.azim},
                   "files": written}, f, indent=2)
    print("\n%d files in %s" % (len(written), out_dir))
    print("Replots of saved arrays. Recolour, reproject or re-angle freely; "
          "nothing here needs the GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
