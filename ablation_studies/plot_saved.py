"""Standalone plotting: ONLY reads outputs/data/<quick|formal>/summary_A/B/C.csv.
Does not import core.py, numba, or any Monte-Carlo algorithm.
Original function interface maintained: read_saved_summary(), draw().
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, NullFormatter
import plot_settings as S

ROOT = Path(__file__).resolve().parent


def read_saved_summary(experiment: str, mode: str | None = None) -> pd.DataFrame:
    experiment = experiment.upper()
    if experiment not in ("A", "B", "C"):
        raise ValueError("experiment should be A, B or C")
    mode = mode or S.PLOT_DATA_MODE
    if mode not in ("quick", "formal"):
        raise ValueError("PLOT_DATA_MODE must be 'quick' or 'formal'")
    path = ROOT / "outputs" / "data" / mode / f"summary_{experiment}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到汇总数据：{path}\n"
            "请确认正式数据已保存；只绘图不需要重新运行 generate_*.py。"
        )
    df = pd.read_csv(path)
    needed = {"mu", "cv", "n", "t"} | set(S.METHODS_TO_PLOT)
    absent = needed - set(df.columns)
    if absent:
        raise ValueError(f"{path.name} 缺少字段：{sorted(absent)}")
    if df.empty:
        raise ValueError(f"{path} 为空")
    return df


def _panels(experiment: str, df: pd.DataFrame):
    if experiment == "A":
        vals = sorted(df["mu"].unique())
        return (
            [(f"({chr(97+i)})  $\\bar{{\\theta}}={v:.2f}$",
              df[np.isclose(df["mu"], v)].sort_values("cv")) for i, v in enumerate(vals)],
            "cv", "Relative heterogeneity CV",
        )
    vals = sorted(df["cv"].unique())
    xvar = "n" if experiment == "B" else "t"
    xlabel = r"Buyers per period $n$" if experiment == "B" else "Target period $t$"
    return (
        [(f"({chr(97+i)})  CV={v:.1f}",
          df[np.isclose(df["cv"], v)].sort_values(xvar)) for i, v in enumerate(vals)],
        xvar, xlabel,
    )


def _y_axis_limits(groups):
    vals = np.concatenate([g[S.METHODS_TO_PLOT].to_numpy(float).ravel()
                           for _, g in groups])
    if not np.isfinite(vals).all() or np.any(vals <= 0):
        raise ValueError("绘图数据存在非正值、NaN 或 Inf，不能使用对数纵轴")
    low, high = float(S.Y_MIN), float(S.Y_MAX)
    if S.Y_AXIS_MODE == "auto":
        if vals.min() < low:
            low = 10.0 ** np.floor(np.log10(vals.min() * 0.9))
        if vals.max() > high:
            high = 10.0 ** np.ceil(np.log10(vals.max() * 1.1))
    elif S.Y_AXIS_MODE == "fixed":
        if vals.min() < low or vals.max() > high:
            raise ValueError(f"绘图数据超出纵轴范围 [{low}, {high}]，请使用 auto")
    else:
        raise ValueError("Y_AXIS_MODE must be 'auto' or 'fixed'")
    ticks = [x for x in S.Y_TICKS if low <= x <= high]
    return low, high, ticks


def draw(experiment: str, mode: str | None = None) -> Path:
    experiment = experiment.upper()
    mode = mode or S.PLOT_DATA_MODE
    summary = read_saved_summary(experiment, mode)
    groups, xvar, xlabel = _panels(experiment, summary)
    if len(groups) != 3:
        raise ValueError(f"实验{experiment}应包含3个子图，实际为{len(groups)}个")

    # B: display n = 40..120, leaving CSV and simulations untouched.
    if experiment == "B":
        groups = [(title, g[g["n"].between(S.B_X_MIN, S.B_X_MAX)])
                  for title, g in groups]
    for title, g in groups:
        if g.empty:
            raise ValueError(f"实验{experiment}子图 {title} 无数据，请检查CSV或B轴范围")
    low, high, yticks = _y_axis_limits(groups)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": S.FONT_FAMILY,
        "font.size": S.AXIS_FONT_PT, "axes.titlesize": S.PANEL_FONT_PT,
        "axes.labelsize": S.AXIS_FONT_PT,
        "xtick.labelsize": S.TICK_FONT_PT,
        "ytick.labelsize": S.TICK_FONT_PT,
        "axes.linewidth": .55, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "stix", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3,
        figsize=(S.FIGURE_WIDTH_IN, S.FIGURE_HEIGHT_IN), sharey=True)
    handles = []
    for k, (ax, (subtitle, g)) in enumerate(zip(axes, groups)):
        for method in S.METHODS_TO_PLOT:
            line, = ax.plot(g[xvar].to_numpy(float), g[method].to_numpy(float),
                color=S.COLORS[method], linestyle=S.LINE_STYLES[method],
                marker=S.MARKERS[method], markerfacecolor=S.COLORS[method],
                markeredgecolor=S.COLORS[method],
                markeredgewidth=S.MARKER_EDGE_WIDTH,
                linewidth=S.LINE_WIDTH, markersize=S.MARKER_SIZE,
                label=S.METHOD_LABELS.get(method, method), zorder=3)
            if k == 0:
                handles.append(line)
        ax.set_title(subtitle, pad=S.PANEL_TITLE_PAD_PT, fontweight="normal")
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in yticks]))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        xticks = S.X_TICKS[experiment]
        xlabels = set(S.X_LABELED_TICKS[experiment])
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FixedFormatter(
            [f"{v:g}" if v in xlabels else "" for v in xticks]))
        if experiment == "B":
            ax.set_xlim(S.B_X_MIN - 3, S.B_X_MAX + 3)
        else:
            xmin, xmax = float(g[xvar].min()), float(g[xvar].max())
            margin = (xmax - xmin) * .045
            ax.set_xlim(xmin - margin, xmax + margin)
        ax.tick_params(axis="both", which="major", length=2.0,
                       width=.55, direction="out", pad=1.0)
        ax.tick_params(axis="y", which="minor", length=.95,
                       width=.32, direction="out")
        ax.spines[["top", "right"]].set_visible(False)
        if S.DRAW_MAJOR_GRID:
            ax.grid(axis="y", which="major", color="#B7C0C9",
                    alpha=.40, lw=.40, zorder=0)
            if S.DRAW_MINOR_GRID:
                ax.grid(axis="y", which="minor", color="#D7DEE5",
                        alpha=.30, lw=.26, zorder=0)
        if k > 0 and not S.SHOW_Y_TICK_LABELS_ON_EACH_PANEL:
            ax.tick_params(axis="y", labelleft=False)
        # Prevent endpoint tick labels on the rightmost panel from being cut off.
        fig.canvas.draw()
        labels = [text for text in ax.get_xticklabels() if text.get_text()]
        if len(labels) > 1:
            labels[0].set_ha("left")
            labels[-1].set_ha("right")

    axes[0].set_ylabel("Relative revenue loss (%)",
                       fontsize=S.AXIS_FONT_PT, labelpad=1.0)
    fig.subplots_adjust(left=S.LEFT, right=S.RIGHT, top=S.TOP,
                        bottom=S.BOTTOM, wspace=S.SUBPLOT_GAP)
    fig.text(.55, S.SHARED_XLABEL_Y, xlabel,
             ha="center", va="center", fontsize=S.AXIS_FONT_PT)
    fig.legend(handles,
        [S.METHOD_LABELS.get(m, m) for m in S.METHODS_TO_PLOT],
        loc="lower center", ncol=S.LEGEND_COLUMNS,
        bbox_to_anchor=(.50, S.LEGEND_Y), fontsize=S.LEGEND_FONT_PT,
        frameon=False, handlelength=S.LEGEND_HANDLE_LENGTH,
        columnspacing=S.LEGEND_COLUMN_SPACING,
        labelspacing=S.LEGEND_LABEL_SPACING,
        handletextpad=.24, borderaxespad=.0)

    if mode == "quick" and S.SHOW_QUICK_WATERMARK:
        fig.text(.5, .986, "QUICK TEST", fontsize=8,
                 ha="center", va="top", color="#9B3D3D")
    out = ROOT / "outputs" / "figures"
    out.mkdir(exist_ok=True, parents=True)
    name = f"Figure_7_2_{experiment}" + ("_QUICK_ONLY" if mode == "quick" else "")
    # No bbox_inches='tight': retains the exact 3.48-inch width and 8-pt font size.
    for ext in S.SAVE_FORMATS:
        fig.savefig(out / f"{name}.{ext}", dpi=S.DPI)
    plt.close(fig)
    dest = out / f"{name}.png"
    print(f"[{experiment}] 从保存的CSV重绘完成（未重新模拟）：{dest}")
    return dest
