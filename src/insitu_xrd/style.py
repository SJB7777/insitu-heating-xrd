"""그림 공통 스타일 (matplotlib · Plotly)."""

# Origin 기본 그래프 느낌: 사방 테두리, 안쪽 눈금(주/보조), 굵은 축, 격자 없음, Arial
ORIGIN_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "mathtext.default": "regular",
    "axes.linewidth": 1.5,
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "axes.grid": False,
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# 온도 구간 색 (matplotlib · Plotly 공용)
SEG_COLOR = {"heat": "#d62728", "hold": "#8c8c8c", "cool": "#1f77b4", "gap": "#d9d9d9"}

# 실험 / 피크 트랙 선 색
EXP_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#8c564b"]
TRACK_COLORS = ["#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#8c564b", "#bcbd22"]

# 면적 vs 온도: 구간 종류별 선 모양
KIND_LS = {"heat": "-", "cool": "--", "hold": ":"}
KIND_DASH = {"heat": "solid", "cool": "dash", "hold": "dot"}
