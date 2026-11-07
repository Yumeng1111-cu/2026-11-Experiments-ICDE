"""7.1 理论验证图：所有可调绘图参数都在这里。

绘图模块不会导入 core.py、numba 或任何 Monte Carlo 代码。
请先运行 generate_all.py 生成 CSV，再运行 plot_all.py。

本项目以保存的 CSV 为唯一数据来源。独立生成脚本使用 settings.py 的
QUICK_MODE / TEST_REPS / PILOT_REPS，且不会由绘图入口调用。
"""
from __future__ import annotations

# 对应 outputs/data/formal 或 outputs/data/quick。
PLOT_DATA_MODE = "formal"
# 本仓库不包含旧数据路径，防止意外读取过期结果。
ALLOW_LEGACY_DATA_PATH = False
SAVE_FORMATS = ("png", "pdf", "svg")
DPI = 600

# IEEE 图片文字建议 8 pt；在 Windows 上安装 Times New Roman 时将优先使用。
FONT_FAMILY = "Times New Roman"
FALLBACK_SERIF = "Liberation Serif"
FONT_PT = 13.0
TICK_FONT_PT = 12
LABEL_FONT_PT = 13
PANEL_TITLE_FONT_PT = 13
LEGEND_FONT_PT = 13
PANEL_TITLE_WEIGHT = "normal"  # "normal" 或 "bold"

# 图的最终物理宽高（英寸）。修改不会改变字体 pt 大小。
# 实验1：单栏；实验2：双栏两子图；实验3：双栏2×2四子图。
# 需要统一单栏时可改成 (3.48, ...)；但多子图建议使用双栏。
FIGURE_SIZE = {
    "1": (3.48, 2.50),
    "2": (7.2, 3),
    "3": (7.12, 6),
}

LINE_WIDTH = 1.5
MARKER_SIZE = 3.8
GRID_ALPHA = 0.23
GRID_LINEWIDTH = 0.55
SHOW_GRID = True
SHOW_MINOR_GRID = False
SHOW_PANEL_TITLES = True
AXIS_LINE_WIDTH = 1.0
TICK_LINE_WIDTH = 0.8
TICK_LENGTH = 3.0
# 布局留白调节：控制每张图的上下左右、子图间距；坐标均为整张图占比。
LAYOUT = {
    "1": dict(left=0.15, right=0.95, top=0.94, bottom=0.19),
    "2": dict(left=0.07, right=0.99, top=0.88, bottom=0.3, wspace=0.10),
    "3": dict(left=0.07, right=0.99, top=0.95, bottom=0.08, wspace=0.30, hspace=0.35),
}

# 颜色与方法/参数标记。实验1、2按 sigma 顺序使用这五种颜色。
SIGMA_COLORS = ["#315F97", "#D38B44", "#328C80", "#8462A9", "#C64B5A"]
SIGMA_LINE_STYLES = ["-", "-", "-", "-", "-"]
SIGMA_MARKERS = [None, None, None, None, None]
SIGMA_LEGEND_DIGITS = 3
SIGMA_LEGEND_COLUMNS = {"1": 2, "2": 5}

# 实验3：收益最优与理论界最优。也可以单独调整线型/标记。
SHOW_BOUND_OPTIMAL = False  # False：只画收益最优；True：两种都画
OPTIMUM_STYLES = {
    "revenue": dict(label="Revenue-optimal", color="#D38B44", linestyle="-", marker="o"),
}
COMPARATIVE_LEGEND_COLUMNS = 2

# 图内标注完整文字，避免只写一个符号；三张图的横纵轴均可直接修改。
AXIS_LABELS = {
    "1": (r"Age gap $d_{s,t}$", "Demand mismatch"),
    "2_sw": (r"Window length, $h$", "Relative revenue loss (%)"),
    "2_ew": (r"Forgetting parameter, $\lambda$", "Relative revenue loss (%)"),
    "3_sigma_h": (r"Time-sensitivity heterogeneity, $\sigma_\theta$", r"Optimal window length, $h_t^{\mathrm{rev}}$"),
    "3_sigma_lambda": (r"Time-sensitivity heterogeneity, $\sigma_\theta$", r"Optimal forgetting rate, $\lambda_t^{\mathrm{rev}}$"),
    "3_n_h": (r"Buyers per period, $n$", r"Optimal window length, $h_t^{\mathrm{rev}}$"),
    "3_n_lambda": (r"Buyers per period, $n$", r"Optimal forgetting rate, $\lambda_t^{\mathrm{rev}}$"),

}
PANEL_TITLES = {
    "2": ["(a) Sliding window", "(b) Exponential weighting"],
    "3": ["(a) SW vs. heterogeneity", "(b) EW vs. heterogeneity",
          "(c) SW vs. sample size", "(d) EW vs. sample size"],
}

# 轴刻度可独立设置。None 为自动。这里不改变数据或拟合。
EXP1_X_TICKS = (0, 20, 40, 60, 79)
EXP1_Y_TICKS = None
EXP2_SW_X_TICKS = (1, 10, 20, 40, 60, 79)
EXP2_EW_X_TICKS = (0, 0.1, 0.2, 0.3, 0.4)
EXP2_Y_SCALE = "linear"  # "linear" 或 "log"
EXP2_Y_LIMITS = None       # 如 (0.1, 25)，不建议强制截断数据
EXP3_SIGMA_X_TICKS = (0.0, 0.005, 0.01, 0.015, 0.02)
EXP3_N_X_TICKS = (10, 20, 30, 50, 100)

# 图例的位置：实验1在图内，实验2/3在图下方。
EXP1_LEGEND_LOC = "upper left"
EXP2_LEGEND_Y = 0.00
EXP3_LEGEND_Y = 0.025

# 若需要可替换图名；用于与原来的论文 LaTeX 图片引用保持兼容。
OUTPUT_BASENAMES = {
    "1": "Experiment1_Mismatch",
    "2": "Experiment2_MemoryTradeoff",
    "3": "Experiment3_ComparativeStatics_4Panels",
}
# ========== 实验1：独立绘图设置 ==========

# 字体大小（pt）
EXP1_X_LABEL_FONT_PT = 10.0      # 横轴名称
EXP1_Y_LABEL_FONT_PT = 10.0      # 纵轴名称
EXP1_X_TICK_FONT_PT = 10.0       # 横轴刻度数字
EXP1_Y_TICK_FONT_PT = 10.0       # 纵轴刻度数字
EXP1_LEGEND_FONT_PT = 10.0       # 图例文字

# 坐标轴
EXP1_X_AXIS_WIDTH = 1.0        # 横轴粗细
EXP1_Y_AXIS_WIDTH = 1.0        # 纵轴粗细
EXP1_TICK_WIDTH = 0.8          # 刻度线粗细
EXP1_TICK_LENGTH = 3.0         # 刻度线长度

# 曲线
EXP1_LINE_WIDTH = 1.0          # 曲线粗细
EXP1_MARKER_SIZE = 3.5         # 数据点大小

# 坐标轴名称与刻度之间的距离
EXP1_X_LABEL_PAD = 4
EXP1_Y_LABEL_PAD = 4
EXP1_TICK_PAD = 2
