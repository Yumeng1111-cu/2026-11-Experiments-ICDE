"""Standalone 7.1 plotting: reads previously generated CSV ONLY.

No core.py, numba, scipy, simulation or Monte Carlo imports.
No data CSV is written by this module.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plot_settings as S

ROOT = Path(__file__).resolve().parent
DATA_REQUIRED = {
    "1": ('experiment1_mismatch.csv',),
    "2": ("experiment2_tradeoff.csv",),
    "3": ("experiment3_heterogeneity.csv", "experiment3_sample_size.csv"),
}


def source_path(filename: str, mode: str) -> Path:
    if mode not in ("formal", "quick"):
        raise ValueError("mode must be 'formal' or 'quick'")
    direct = ROOT / "outputs" / "data" / mode / filename
    if direct.is_file():
        return direct
    legacy = ROOT / "outputs" / "data" / filename
    if S.ALLOW_LEGACY_DATA_PATH and legacy.is_file():
        print(f"[plot] 使用旧项目汇总数据: {legacy}", flush=True)
        return legacy
    raise FileNotFoundError(
        f"缺少 {direct}\n"
        f"请先运行 python reproduce.py --quick（测试）或 python reproduce.py（正式），"
        f"或运行 python generate_all.py 生成 CSV 后再绘图。"
    )


def load(filename: str, columns: set[str], mode: str) -> pd.DataFrame:
    path = source_path(filename, mode)
    d = pd.read_csv(path)
    missing = columns - set(d.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    if d.empty:
        raise ValueError(f"{path} 为空")
    return d


def _font_setup():
    available = {f.name for f in fm.fontManager.ttflist}
    font = S.FONT_FAMILY if S.FONT_FAMILY in available else S.FALLBACK_SERIF
    if font not in available:
        font = "DejaVu Serif"
    if font != S.FONT_FAMILY:
        print(f"[plot] 当前系统未安装 {S.FONT_FAMILY}，使用 {font}；Windows 上安装 TNR 可自动恢复。", flush=True)
    plt.rcParams.update({
        "font.family": font,
        "mathtext.fontset": "stix",
        "font.size": S.FONT_PT,
        "axes.labelsize": S.LABEL_FONT_PT,
        "axes.titlesize": S.PANEL_TITLE_FONT_PT,
        "axes.titleweight": S.PANEL_TITLE_WEIGHT,
        "xtick.labelsize": S.TICK_FONT_PT,
        "ytick.labelsize": S.TICK_FONT_PT,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def _axis(ax):
    # 只显示原有左侧和底部坐标轴
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # 加粗原有横纵坐标轴，不新增线条
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    # 刻度线
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=0.8,
        pad=2
    )

    # 网格
    if S.SHOW_GRID:
        ax.grid(
            axis="y",
            which="major",
            color="#A4ADB8",
            alpha=S.GRID_ALPHA,
            linewidth=S.GRID_LINEWIDTH
        )

    if S.SHOW_MINOR_GRID:
        ax.grid(
            axis="y",
            which="minor",
            color="#A4ADB8",
            alpha=S.GRID_ALPHA * 0.45,
            linewidth=S.GRID_LINEWIDTH * 0.65
        )

    ax.set_axisbelow(True)


def _sigma_name(sigma: float) -> str:
    return rf"$\sigma_\theta={sigma:.{S.SIGMA_LEGEND_DIGITS}f}$"


def _save(fig, experiment: str) -> Path:
    dest = ROOT / "outputs" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    stem = S.OUTPUT_BASENAMES[experiment]
    for fmt in S.SAVE_FORMATS:
        path = dest / f"{stem}.{fmt}"
        try:
            # 不用 bbox_inches='tight'：避免单栏宽度被自动扩展或裁切变得不可控
            fig.savefig(path, dpi=S.DPI, facecolor="white")
        except PermissionError:
            # Windows 正在打开 PDF 时可能锁定文件；输出新版而不停止整个绘图流程
            backup = dest / f"{stem}_new.{fmt}"
            fig.savefig(backup, dpi=S.DPI, facecolor="white")
            print(f"[plot] 原文件被占用，已保存为 {backup.name}", flush=True)
    plt.close(fig)
    print(f"[plot] 实验{experiment}: {dest / (stem + '.pdf')} (数据未改动)", flush=True)
    return dest / f"{stem}.png"


def plot_1(mode: str):
    df = load("experiment1_mismatch.csv", {
        "sigma_theta", "historical_distance_d", "population_mismatch"}, mode)
    fig, ax = plt.subplots(figsize=S.FIGURE_SIZE["1"])
    sigmas = sorted(df.sigma_theta.unique())
    for i, sigma in enumerate(sigmas):
        d = df[np.isclose(df.sigma_theta, sigma)].sort_values("historical_distance_d")
        ax.plot(d.historical_distance_d, d.population_mismatch,
                color=S.SIGMA_COLORS[i % len(S.SIGMA_COLORS)],
                linestyle=S.SIGMA_LINE_STYLES[i % len(S.SIGMA_LINE_STYLES)],
                linewidth=S.EXP1_LINE_WIDTH,
                marker=S.SIGMA_MARKERS[i % len(S.SIGMA_MARKERS)],
                markersize=S.EXP1_MARKER_SIZE, label=_sigma_name(sigma))
    ax.set_xlabel(S.AXIS_LABELS["1"][0]); ax.set_ylabel(S.AXIS_LABELS["1"][1])
    ax.set_xlim(float(df.historical_distance_d.min()), float(df.historical_distance_d.max()))
    if S.EXP1_X_TICKS is not None: ax.set_xticks(S.EXP1_X_TICKS)
    if S.EXP1_Y_TICKS is not None: ax.set_yticks(S.EXP1_Y_TICKS)
    _axis(ax)
    # 横纵轴名称字体
    ax.xaxis.label.set_fontsize(S.EXP1_X_LABEL_FONT_PT)
    ax.yaxis.label.set_fontsize(S.EXP1_Y_LABEL_FONT_PT)

    # 横纵轴名称与刻度的间距
    ax.xaxis.labelpad = S.EXP1_X_LABEL_PAD
    ax.yaxis.labelpad = S.EXP1_Y_LABEL_PAD

    # 刻度数字字体、刻度线粗细
    ax.tick_params(
        axis="x",
        which="major",
        labelsize=S.EXP1_X_TICK_FONT_PT,
        width=S.EXP1_TICK_WIDTH,
        length=S.EXP1_TICK_LENGTH,
        pad=S.EXP1_TICK_PAD
    )

    ax.tick_params(
        axis="y",
        which="major",
        labelsize=S.EXP1_Y_TICK_FONT_PT,
        width=S.EXP1_TICK_WIDTH,
        length=S.EXP1_TICK_LENGTH,
        pad=S.EXP1_TICK_PAD
    )

    # 只调整已有坐标轴，不增加线条
    ax.spines["bottom"].set_linewidth(S.EXP1_X_AXIS_WIDTH)
    ax.spines["left"].set_linewidth(S.EXP1_Y_AXIS_WIDTH)

    ax.legend(loc=S.EXP1_LEGEND_LOC, ncol=S.SIGMA_LEGEND_COLUMNS["1"],
              fontsize=S.EXP1_LEGEND_FONT_PT, frameon=False, handlelength=1.7,
              columnspacing=0.7, labelspacing=0.25, borderaxespad=0.15)
    fig.subplots_adjust(**S.LAYOUT["1"])
    return _save(fig, "1")


def plot_2(mode: str):
    df = load("experiment2_tradeoff.csv", {
        "sigma_theta", "method", "memory_parameter", "relative_revenue_loss_percent"}, mode)
    sigmas = sorted(df.sigma_theta.unique())
    fig, axs = plt.subplots(1, 2, figsize=S.FIGURE_SIZE["2"], sharey=True)
    for j, (ax, method, lab) in enumerate(zip(axs, ("SW", "EW"), ("2_sw", "2_ew"))):
        for i, sigma in enumerate(sigmas):
            sub = df[(df.method == method) & np.isclose(df.sigma_theta, sigma)].sort_values("memory_parameter")
            if sub.empty: raise ValueError(f"实验2缺少 {method} / sigma={sigma:g} 曲线")
            ax.plot(sub.memory_parameter, sub.relative_revenue_loss_percent,
                    color=S.SIGMA_COLORS[i % len(S.SIGMA_COLORS)],
                    linestyle=S.SIGMA_LINE_STYLES[i % len(S.SIGMA_LINE_STYLES)],
                    linewidth=S.LINE_WIDTH,
                    marker=S.SIGMA_MARKERS[i % len(S.SIGMA_MARKERS)],
                    markersize=S.MARKER_SIZE, label=_sigma_name(sigma))
        ax.set_xlabel(S.AXIS_LABELS[lab][0])
        if j == 0: ax.set_ylabel(S.AXIS_LABELS[lab][1])
        ax.set_yscale(S.EXP2_Y_SCALE)
        if S.EXP2_Y_LIMITS is not None: ax.set_ylim(*S.EXP2_Y_LIMITS)
        ticks = S.EXP2_SW_X_TICKS if method == "SW" else S.EXP2_EW_X_TICKS
        if ticks is not None: ax.set_xticks(ticks)
        _axis(ax)
        if S.SHOW_PANEL_TITLES: ax.set_title(S.PANEL_TITLES["2"][j], pad=5)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=S.SIGMA_LEGEND_COLUMNS["2"], frameon=False,
               fontsize=S.LEGEND_FONT_PT, loc="lower center", bbox_to_anchor=(.5,S.EXP2_LEGEND_Y),
               handlelength=1.75, columnspacing=1.45)
    fig.subplots_adjust(**S.LAYOUT["2"])
    return _save(fig, "2")


def plot_3(mode: str):
    cols = {"sigma_theta", "n", "h_revenue", "h_bound", "lambda_revenue", "lambda_bound"}
    a = load("experiment3_heterogeneity.csv", cols, mode).sort_values("sigma_theta")
    b = load("experiment3_sample_size.csv", cols, mode).sort_values("n")
    fig, axs = plt.subplots(2, 2, figsize=S.FIGURE_SIZE["3"])
    specs = [
        (axs[0,0], a.sigma_theta, a.h_revenue, a.h_bound, "3_sigma_h", S.EXP3_SIGMA_X_TICKS),
        (axs[0,1], a.sigma_theta, a.lambda_revenue, a.lambda_bound, "3_sigma_lambda", S.EXP3_SIGMA_X_TICKS),
        (axs[1,0], b.n, b.h_revenue, b.h_bound, "3_n_h", S.EXP3_N_X_TICKS),
        (axs[1,1], b.n, b.lambda_revenue, b.lambda_bound, "3_n_lambda", S.EXP3_N_X_TICKS),
    ]
    for i, (ax, x, rev, bound, lab, xticks) in enumerate(specs):
        for kind, series in (("revenue", rev), ("bound", bound)):
            if kind == "bound" and not S.SHOW_BOUND_OPTIMAL:
                continue

            st = S.OPTIMUM_STYLES[kind]
            ax.plot(
                x, series,
                label=st["label"],
                color=st["color"],
                linewidth=S.LINE_WIDTH,
                linestyle=st["linestyle"],
                marker=st["marker"],
                markersize=S.MARKER_SIZE
            )
        # 设置横纵坐标名称
        ax.set_xlabel(
            S.AXIS_LABELS[lab][0],
            fontsize=S.LABEL_FONT_PT,
            labelpad=4
        )

        ax.set_ylabel(
            S.AXIS_LABELS[lab][1],
            fontsize=S.LABEL_FONT_PT,
            labelpad=4
        )

        # 横轴刻度
        if xticks is not None:
            ax.set_xticks(xticks)

        ax.tick_params(
            axis="both",
            labelsize=S.TICK_FONT_PT
        )
        _axis(ax)
        if S.SHOW_PANEL_TITLES: ax.set_title(S.PANEL_TITLES["3"][i], pad=6)
    fig.subplots_adjust(**S.LAYOUT["3"])
    return _save(fig, "3")


def draw(experiment: str | int, mode: str | None = None) -> Path:
    _font_setup()
    e = str(experiment)
    if e not in ("1", "2", "3"):
        raise ValueError("experiment must be 1, 2, or 3")
    m = mode or S.PLOT_DATA_MODE
    return {"1":plot_1, "2":plot_2, "3":plot_3}[e](m)
