"""ICDE 7.2 ablation figures; change appearance here, NOT simulation data.

This file remains compatible with the original standalone plot_saved.py API.
All figures are 3-across panels at IEEE single-column width. CSV path:
    outputs/data/<PLOT_DATA_MODE>/summary_A/B/C.csv

Install Times New Roman on Windows to obtain this exact font. At native 3.48-inch
figure width and LaTeX width=\\columnwidth, all chart text is set to >= 8 pt.
"""

# IMPORTANT: use the 5000-repetition results already saved in your project.
PLOT_DATA_MODE = "formal"    # "quick" for preview only
FIGURE_WIDTH_IN = 3.48
FIGURE_HEIGHT_IN = 2.15

DPI = 600
SAVE_FORMATS = ("png", "pdf", "svg")
FONT_FAMILY = ["Times New Roman", "Liberation Serif", "Nimbus Roman"]
TICK_FONT_PT = 6.0
AXIS_FONT_PT = 6.5
PANEL_FONT_PT = 6.5
LEGEND_FONT_PT = 6.5

# Fixed layout in figure-relative coordinates (0..1). Every subplot shares
# one ylabel, one xlabel, and one bottom legend.
LEFT = 0.1
RIGHT = 0.963
TOP = 0.93
BOTTOM = 0.28

SUBPLOT_GAP = 0.16

PANEL_TITLE_PAD_PT = 3.0
SHARED_XLABEL_Y = 0.185
LEGEND_Y = 0.01
LEGEND_COLUMNS = 3
LEGEND_HANDLE_LENGTH = 2
LEGEND_COLUMN_SPACING = 2
LEGEND_LABEL_SPACING = 0.5

Y_AXIS_MODE = "auto"  # "fixed" raises an error instead of clipping data
Y_MIN = 0.073
Y_MAX = 145.0
Y_TICKS = (0.1, 0.2, 0.5, 1, 2,  10, 100)
DRAW_MINOR_GRID = False
DRAW_MAJOR_GRID = True
SHOW_Y_TICK_LABELS_ON_EACH_PANEL = False

# Keep all tick marks, show only selected tick numbers.
X_TICKS = {
    "A": (0.2, 0.3, 0.4, 0.5),
    "B": (40, 60, 80, 100, 120),
    "C": (40, 50, 60, 70, 80, 90, 100),
}
X_LABELED_TICKS = {
    "A": (0.2, 0.3, 0.4, 0.5),
    "B": (40, 80, 120),
    "C": (40, 60, 80, 100),
}
B_X_MIN = 40
B_X_MAX = 120

LINE_WIDTH = 1.15
MARKER_SIZE = 2.55
MARKER_EDGE_WIDTH = 0.25
SHOW_QUICK_WATERMARK = True

# Keys MUST remain identical to existing CSV column names.
METHODS_TO_PLOT = [
    "Full-history ERM",
    "Calibrated Full-history",
    "Uncalibrated SW-FV",
    "ACRP-FV",
    "Oracle-SW",
    "Oracle-EW",
]
# Display labels can be changed freely without changing CSV data.
METHOD_LABELS = {
        "Full-history ERM": "U-FH",
    "Calibrated Full-history": "C-FH",
    "Uncalibrated SW-FV": "U-SW-FV",
    "ACRP-FV": "ACRP-FV",
    "Oracle-SW": "Oracle-SW",
    "Oracle-EW": "Oracle-EW",
}
COLORS = {
    "Full-history ERM": "#5F7289",
    "Calibrated Full-history": "#43AA8B",
    "Uncalibrated SW-FV": "#C89B1F",
    "ACRP-FV": "#C84D62",
    "Oracle-SW": "#8535B8",
    "Oracle-EW": "#222222",
}
MARKERS = {
    "Full-history ERM": "o",
    "Calibrated Full-history": "s",
    "Uncalibrated SW-FV": "^",
    "ACRP-FV": "D",
    "Oracle-SW": "v",
    "Oracle-EW": "P",
}
LINE_STYLES = {
    "Full-history ERM": "-",
    "Calibrated Full-history": "-",
    "Uncalibrated SW-FV": "-",
    "ACRP-FV": "-",
    "Oracle-SW": "--",
    "Oracle-EW": "--",
}
