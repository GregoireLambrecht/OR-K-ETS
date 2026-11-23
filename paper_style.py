"""House style of the OR-K-ETS-2026 paper panels (figure regeneration spec).

    import paper_style as ps
    ps.apply()                      # once, at the top of the plotting script
    fig, ax = ps.line_panel()       # fixed size + fixed margins
    ps.save(fig, base)              # base.pdf + base.png, white, no title

One axes per file, drawn at final size (no LaTeX rescaling), no titles.
Panels of one row share figsize AND margins, so the saved files have exactly
the same size: savefig.bbox is left at None on purpose (a tight bbox would
crop every file differently). save() warns if something sticks out.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.edgecolor": "white",
    # no Times on the cluster: STIX (bundled with matplotlib) is Times-like
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "Nimbus Roman",
                   "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 2.5,
    "axes.grid": True, "grid.linewidth": 0.3, "grid.color": "0.85",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    # fixed sizes: see the module docstring
    "savefig.bbox": None, "savefig.pad_inches": 0.02,
}

DPI = 300

# sizes (inches), text width 6.5 in
LINE = (2.1, 1.6)          # exploitability / convergence, 3 per row
SURF_MARKET = (3.1, 2.3)   # price, pollution, 2 per row
SURF_REVENUE = (2.1, 1.8)  # revenue per type, 3 per row
RUNTIME = (3.1, 2.1)

# fixed margins, in inches, so the axes boxes of a row coincide
LINE_MARGINS = dict(left=0.40, right=0.06, bottom=0.30, top=0.05)

# type identity: colour (unchanged) + marker + line style for B/W print
TYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TYPE_NAMES = ("Private producer", "Large public producer", "Financial institution")
TYPE_TAGS = ("priv", "pub", "fi")
TYPE_MARKERS = ("o", "s", "^")
TYPE_LINESTYLES = ("-", (0, (4, 1.5)), (0, (1, 1)))

# axis wording, identical in every grid
AXIS_LABEL = {'init_alloc': 'Initial allocation (%)',
              'kappa': r'Tax slope $\kappa^2$',
              'Afloor': r'Price floor (USD/tCO$_2$)',
              'financial_inst': 'Financial institutions (%)'}
AXIS_SCALE = {'init_alloc': 100.0, 'financial_inst': 100.0}
PRICE_LABEL = 'Average allowance\nprice (USD/tCO$_2$)'
POLLUTION_LABEL = 'Pollution per\nproducer (ktCO$_2$)'
REVENUE_LABEL = 'Revenue\n(millions, USD)'


def apply():
    mpl.rcParams.update(STYLE)


def line_panel(figsize=LINE, margins=LINE_MARGINS):
    fig, ax = plt.subplots(figsize=figsize)
    w, h = figsize
    fig.subplots_adjust(left=margins['left'] / w, right=1 - margins['right'] / w,
                        bottom=margins['bottom'] / h, top=1 - margins['top'] / h)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    return fig, ax


def _check_inside(fig, base):
    r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r)
    W, H = fig.get_size_inches()
    tol = 0.01
    if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > W + tol or bb.y1 > H + tol:
        print(f"  WARNING {os.path.basename(base)}: content outside the figure "
              f"({bb.x0:.2f},{bb.y0:.2f})-({bb.x1:.2f},{bb.y1:.2f}) vs {W}x{H} in")


def save(fig, base, check=True):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    if check:
        fig.canvas.draw()
        _check_inside(fig, base)
    fig.savefig(base + '.pdf', transparent=False, dpi=DPI)
    fig.savefig(base + '.png', transparent=False, dpi=DPI)
    plt.close(fig)
    print(f"-> {base}.pdf (+ .png)")
