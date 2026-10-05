"""여러 실험의 피크별 결정화 분율 X (정규화 면적) vs 온도 비교 + T50.

설정 파일(TOML, 기본 experiments.toml)에 실험별 이미지 폴더 · 온도 로그(recipe) 를 적으면
    uv run python main.py compare [experiments.toml]
→ 피크마다 패널 하나: 각 실험의 X vs PV 온도, 범례에 'x1  T50 401 °C'

비정질 기준 나누기: R = I / I_amorphous − 1  (I_amorphous = 사용 구간 처음 n_norm 프레임 평균)
     디텍터 가장자리 세기 감소(곱셈)와 비정질 배경이 상쇄되어 결정 피크만 봉우리로 남음

적분 창 (실험·피크마다 자동):
     설정의 2θ 는 '이 근처를 봐라' 는 참고값. 결정화가 끝난 마지막 n_norm 프레임의 평균 R 에서
     참고값 ± search 안의 피크를 가우시안(+직선 배경)으로 피팅 → 중심 c, 반치폭 FWHM
     → 창 = c ± fwhm_k × FWHM  (기본 1.5 배: 가우시안 면적의 ~99.9 %)

X  = (A − A₀) / (A₁ − A₀)
     A  = 위 창에서 R 의 면적 (창 양끝을 잇는 직선 배경 제거)
     A₀ = 사용 구간 처음 n_norm 프레임의 중앙값 (전이 전), A₁ = 마지막 n_norm 프레임의 중앙값 (전이 후)
T50 = X 가 처음으로 level(기본 0.5) 을 위로 넘는 온도 (앞뒤 프레임 사이 선형보간)

이미지 폴더 · 온도 로그는 설정 파일에 적힌 경로 그대로 사용 (자동으로 찾지 않음).
적분 결과는 out/cache/ 에 저장해 두고 이미지 폴더가 그대로면 다시 계산하지 않음.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import segments as sg
from .config import OUT_DIR, SETTINGS_FILE, load_toml, project_path
from .geometry import Geometry
from .heatmap import Config, beam_ok, integrate_folder, regrid, time_mean
from .integrate import ProfileOptions
from .peaks import fit_peak
from .style import EXP_COLORS, ORIGIN_RC
from .temperature import PV, TempLog, fmt_kst, parse_time
from .tracks import level_temperature, window_area

DEFAULT_CONFIG = SETTINGS_FILE
BASELINE_TOL = 30.0     # 첫 사용 프레임이 승온 시작보다 이만큼 [°C] 넘게 늦으면 '전이 전' 기준이 없다고 봄


# ═════════════════════════════ 설정 ═════════════════════════════
@dataclass(slots=True)
class Experiment:
    name: str
    images: list[Path]
    recipe: list[Path]
    start: str | None = None
    end: str | None = None
    color: str | None = None


@dataclass(slots=True)
class Settings:
    peaks: dict[str, float]           # 피크 이름 → 참고 2θ (이 근처에서 실제 피크를 찾음)
    exps: list[Experiment]
    segment: str = "heat"             # "heat" = 승온 구간만, "all" = 전체
    level: float = 0.5
    n_norm: int = 10
    t_range: tuple[float, float] | None = None
    tth_bin: float = 0.02
    search: float = 0.5               # 참고 2θ ± search 안에서 피크를 찾음 [deg]
    fwhm_k: float = 1.5               # 적분 창 = 중심 ± fwhm_k × FWHM
    path: Path = DEFAULT_CONFIG


def _as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def load_settings(path: Path = DEFAULT_CONFIG, need_exps: bool = True) -> Settings:
    """설정 파일 읽기. 경로는 적힌 그대로 (상대경로면 프로젝트 폴더 기준)."""
    path = Path(path)
    raw = load_toml(path)
    o = raw.get("options", {})
    exps = [Experiment(name=str(e["name"]),
                       images=[project_path(s) for s in _as_list(e["images"])],
                       recipe=[project_path(s) for s in _as_list(e.get("recipe", []))],
                       start=e.get("start"), end=e.get("end"), color=e.get("color"))
            for e in raw.get("exp", [])]
    if not exps and need_exps:
        raise SystemExit(f"❌ {path}: [[exp]] 항목이 없음")
    # 숫자 하나 = 참고 2θ. [시작, 끝] 으로 적은 예전 형식은 가운데를 참고값으로
    peaks = {k: float(v) if isinstance(v, (int, float)) else (float(v[0]) + float(v[1])) / 2
             for k, v in raw.get("peaks", {}).items()}
    if not peaks:
        raise SystemExit(f"❌ {path}: [peaks] 항목이 없음")
    tr = o.get("t_range")
    return Settings(peaks=peaks, exps=exps, segment=o.get("segment", "heat"),
                    level=float(o.get("level", 0.5)), n_norm=int(o.get("n_norm", 10)),
                    t_range=(float(tr[0]), float(tr[1])) if tr else None,
                    tth_bin=float(o.get("tth_bin", 0.02)), search=float(o.get("search", 0.5)),
                    fwhm_k=float(o.get("fwhm_k", 1.5)), path=path)


def find_exp(name: str, path: Path = DEFAULT_CONFIG) -> Experiment | None:
    """설정 파일의 [[exp]] 중 이름이 name 인 것 (overview x1 처럼 쓸 때)."""
    if not Path(path).exists():
        return None
    return next((e for e in load_settings(path, need_exps=False).exps if e.name == name), None)


def check_paths(exps: list[Experiment]) -> None:
    """계산 시작 전에 없는 경로를 한꺼번에 알려 줌."""
    missing = [f"   {e.name}: {p}" for e in exps for p in (*e.images, *e.recipe) if not p.exists()]
    if missing:
        raise SystemExit("❌ 경로 없음 (experiments.toml 의 images / recipe 확인):\n" + "\n".join(missing))


# ═════════════════════════════ 실험 하나 ═════════════════════════════
@dataclass(slots=True)
class ExpResult:
    name: str
    table: pd.DataFrame             # 프레임별 file, time_kst, T_pv, kind, A_<peak>, X_<peak>
    t50: dict[str, float]
    note: str = ""
    color: str = "#000000"
    ok: bool = True                 # False = 전이 전 데이터가 없어 X 를 정규화할 수 없음
    windows: dict[str, tuple[float, float, float, float]] | None = None  # 피크 → (중심, FWHM, 창 시작, 끝)


def _clip_window(x: np.ndarray, ok: np.ndarray, c: float, lo: float, hi: float) -> tuple[float, float]:
    """[lo, hi] 를 중심 c 에서 이어지는 ok 구간 안으로 줄임 (갭(NaN) 은 통과, 가장자리 낙하는 막음)."""
    i = int(np.clip(np.searchsorted(x, c), 0, len(x) - 1))
    a = i
    while a > 0 and ok[a - 1] and x[a - 1] >= lo:
        a -= 1
    b = i
    while b < len(x) - 1 and ok[b + 1] and x[b + 1] <= hi:
        b += 1
    return max(lo, float(x[a])), min(hi, float(x[b]))


def peak_window(x: np.ndarray, prof: np.ndarray, guess: float, search: float,
                k: float) -> tuple[float, float, float]:
    """참고 2θ 근처 피크의 (중심, FWHM, 적분 창 반폭 = k×FWHM).
    가우시안+직선 배경 피팅, 실패하면 반값 지점 직접 찾기."""
    fit = fit_peak(x, prof, guess, half_window=search)
    dx = x[1] - x[0]
    if np.isfinite(fit.fwhm) and abs(fit.center - guess) <= search and 2 * dx < fit.fwhm < 2 * search:
        return fit.center, fit.fwhm, k * fit.fwhm
    # 대안: 범위 양끝을 잇는 직선 배경을 빼고 최댓값의 절반이 되는 좌우 지점
    m = (np.abs(x - guess) <= search) & np.isfinite(prof)
    xs, ys = x[m], prof[m]
    if len(xs) < 6:
        return guess, np.nan, search
    ys = ys - np.interp(xs, [xs[0], xs[-1]], [ys[:3].mean(), ys[-3:].mean()])
    i = int(np.argmax(ys))
    half = ys[i] / 2
    left = np.flatnonzero(ys[:i] < half)
    right = np.flatnonzero(ys[i:] < half)
    if not len(left) or not len(right) or ys[i] <= 0:
        return guess, np.nan, search
    l0, r0 = left[-1], i + right[0]
    xl = np.interp(half, [ys[l0], ys[l0 + 1]], [xs[l0], xs[l0 + 1]])
    xr = np.interp(half, [ys[r0], ys[r0 - 1]], [xs[r0], xs[r0 - 1]])
    fwhm = xr - xl
    return float(xs[i]), float(fwhm), float(k * fwhm)


def peak_areas(x: np.ndarray, Z: np.ndarray, base: np.ndarray, final: np.ndarray, st: Settings,
               say=print) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, np.ndarray]]:
    """피크별 (중심, FWHM, 창 시작, 끝) 과 프레임별 면적.
    base 프레임(전이 전, 비정질) 평균으로 나눔: R = I / I_amorphous − 1
      → 디텍터 가장자리 세기 감소(곱셈 효과)와 비정질 배경이 상쇄되어 결정 피크만 깔끔한 봉우리로 남음.
    창은 final 프레임(결정화가 끝난 상태) 평균 R 에서 [peaks] 참고값 근처 피크를 찾아 정함."""
    pre = time_mean(Z[base])
    cover = np.isfinite(pre) & (pre > 0.3 * np.nanmedian(pre))     # 디텍터가 거의 안 닿는 2θ 제외
    with np.errstate(all="ignore"):
        R = np.where(cover, Z / pre - 1.0, np.nan)
    ref = time_mean(R[final])
    dx = x[1] - x[0]
    windows, areas = {}, {}
    for pk, guess in st.peaks.items():
        c, fwhm, hw = peak_window(x, ref, guess, st.search, st.fwhm_k)
        lo, hi = _clip_window(x, cover | np.isnan(pre), c, c - hw, c + hw)
        windows[pk] = (c, fwhm, lo, hi)
        cut = " ⚠️ 디텍터 범위 끝에서 잘림" if (lo > c - hw + dx or hi < c + hw - dx) else ""
        say(f"   {pk:>6}: 중심 {c:.3f}°  FWHM {fwhm:.3f}°  →  적분 {lo:.2f}–{hi:.2f}° "
            f"(±{st.fwhm_k:g}·FWHM){cut}" if np.isfinite(fwhm) else
            f"   {pk:>6}: ⚠️ {guess}° ±{st.search}° 에서 피크를 못 찾음 → {lo:.2f}–{hi:.2f}° 적분")
        areas[pk] = np.array([window_area(x, r, (lo + hi) / 2, (hi - lo) / 2) for r in R])
    return windows, areas


@dataclass(slots=True)
class Analysis:
    use: np.ndarray                 # 프레임별: 계산에 쓴 프레임 (승온 · 온도 범위 · 빔 정상)
    kind: np.ndarray                # 프레임별 구간 종류 heat/hold/cool
    ok: bool                        # False = 전이 전 데이터가 없어 정규화 불가
    msg: str
    t50: dict[str, float]
    windows: dict[str, tuple[float, float, float, float]]   # 피크 → (중심, FWHM, 창 시작, 끝)
    A: dict[str, np.ndarray]
    X: dict[str, np.ndarray]


def analyze(x: np.ndarray, Z: np.ndarray, t: np.ndarray, T: np.ndarray, log: TempLog,
            st: Settings, verbose: bool = True) -> Analysis:
    """프레임 프로파일 Z (시간순) + 프레임 온도 T(PV) → 피크별 X, T50. compare · overview 공용."""
    say = print if verbose else (lambda *a, **k: None)
    has_T = np.isfinite(T)
    # 구간 종류 (승온/유지/하온)
    kind = np.full(len(t), "", dtype=object)
    heat_T0 = np.nan
    if has_T.any():
        tt = t[has_T]
        # 로그 전체로 감지 (프레임 범위로 자르면 승온 시작 온도를 잘못 앎)
        det = sg.detect(log, min(tt[0], log.t[0]), max(tt[-1], log.t[-1]))
        kind = np.where(has_T, sg.kinds_at(t, det.segments), "")
        heats = [s for s in det.segments if s.kind == "heat"]
        if heats:
            heat_T0 = max(heats, key=lambda s: s.duration).T0

    use = has_T.copy()
    if st.segment == "heat":
        use &= kind == "heat"
    if st.t_range:
        use &= (T >= st.t_range[0]) & (T <= st.t_range[1])
    if use.sum() < 2 * st.n_norm:
        say(f"   ⚠️  사용할 프레임이 {use.sum()} 개뿐 (segment={st.segment}, t_range={st.t_range})")
    use &= beam_ok(Z, use)          # 빔 꺼진 프레임 제외 (사용 프레임 세기 중앙값 기준)

    ok, msg = True, ""
    if use.any():
        T_first = float(np.nanmin(T[use]))
        T_need = max(st.t_range[0] if st.t_range else -np.inf,
                     heat_T0 if np.isfinite(heat_T0) else -np.inf)
        if np.isfinite(T_need) and T_first - T_need > BASELINE_TOL:
            ok = False
            msg = (f"이미지가 {T_first:.0f} °C 부터만 있음 (승온은 {T_need:.0f} °C 부터) → "
                   "전이 전 기준이 없어 X·T50 계산 안 함")
            say(f"   ❌ {msg}")
    else:
        ok, msg = False, "사용할 프레임 없음"

    t50, Xs = {}, {}
    iu_all = np.flatnonzero(use)
    if not len(iu_all):
        iu_all = np.flatnonzero(has_T)
    windows, As = peak_areas(x, Z, iu_all[:st.n_norm], iu_all[-st.n_norm:], st, say)
    for pk, A in As.items():
        X = np.full(len(A), np.nan)
        iu = np.flatnonzero(use & np.isfinite(A))
        if len(iu) >= 2:
            n = min(st.n_norm, max(1, len(iu) // 4))
            a0, a1 = np.median(A[iu[:n]]), np.median(A[iu[-n:]])
            X[iu] = (A[iu] - a0) / ((a1 - a0) or np.nan)
        As[pk], Xs[pk] = A, X
        t50[pk] = level_temperature(T[use], X[use], st.level) if ok else np.nan
        if ok:
            say(f"   {pk:>6}: T50 = {t50[pk]:.1f} °C" if np.isfinite(t50[pk]) else f"   {pk:>6}: T50 없음")
    return Analysis(use, kind, ok, msg, t50, windows, As, Xs)


@dataclass(slots=True)
class ExpData:
    x: np.ndarray                   # 2θ
    Z: np.ndarray                   # 세기 [frame, 2θ], 시간순
    t: np.ndarray                   # 촬영시각 Unix [frame]
    files: np.ndarray
    T: np.ndarray                   # 프레임 PV 온도 (로그 밖 NaN)
    log: TempLog | None             # 온도 로그가 없으면 None
    note: str = ""


def load_exp(exp: Experiment, st: Settings, geo: Geometry, workers: int | None) -> ExpData:
    """실험의 이미지 폴더(들) 적분(캐시) → 시간순으로 합치고 start/end 로 자르고 온도 매칭."""
    print(f"\n🧪 {exp.name}: {', '.join(str(p) for p in exp.images)}  |  "
          f"{', '.join(p.name for p in exp.recipe) or '온도 로그 없음'}")
    parts = [integrate_folder(Config(folder=f, geometry=geo, workers=workers,
                                     profile=ProfileOptions(tth_bin=st.tth_bin)))
             for f in exp.images]
    x = parts[0][0]
    Z = np.vstack([regrid(x, p[0], p[1]) for p in parts])
    t = np.concatenate([p[2] for p in parts])
    files = np.concatenate([p[3] for p in parts])
    order = np.argsort(t, kind="stable")
    Z, t, files = Z[order], t[order], files[order]

    # 시간 범위 제한 (선택)
    keep = np.ones(len(t), bool)
    day = fmt_kst(t[0])[:10]
    if exp.start:
        keep &= t >= parse_time(exp.start, day)
    if exp.end:
        keep &= t <= parse_time(exp.end, day)
    Z, t, files = Z[keep], t[keep], files[keep]

    # 온도: 로그가 실제로 있는 시각만 (바깥은 NaN)
    if not exp.recipe:          # 온도 로그 없음 → 온도 NaN (avrami 는 그래도 n·k 계산)
        return ExpData(x, Z, t, files, np.full(len(t), np.nan), None, "온도 로그 없음")
    log = TempLog.load(exp.recipe)
    T = log.at(t, PV)
    n_T = int(np.isfinite(T).sum())
    note = ""
    if n_T < len(t):
        note = f"온도 있는 프레임 {n_T}/{len(t)} (로그 {log.span[0][11:]}~{log.span[1][11:]} 밖은 제외)"
        print(f"   ⚠️  {note}")
    return ExpData(x, Z, t, files, T, log, note)


def run_exp(exp: Experiment, st: Settings, geo: Geometry, workers: int | None,
            color: str = "#000000") -> ExpResult:
    if not exp.recipe:
        raise SystemExit(f"❌ {exp.name}: recipe(온도 로그) 가 필요함")
    d = load_exp(exp, st, geo, workers)
    x, Z, t, files, T, log, note = d.x, d.Z, d.t, d.files, d.T, d.log, d.note
    an = analyze(x, Z, t, T, log, st)
    if an.msg:
        note = (note + " / " if note else "") + an.msg
    tab = pd.DataFrame({"file": files, "time_kst": [fmt_kst(v) for v in t], "T_pv": T,
                        "kind": an.kind, "used": an.use})
    for pk in st.peaks:
        tab[f"A_{pk}"], tab[f"X_{pk}"] = an.A[pk], an.X[pk]
    return ExpResult(exp.name, tab, an.t50, note, color=color, ok=an.ok, windows=an.windows)


# ═════════════════════════════ 그림 ═════════════════════════════
def _lab(r: ExpResult, pk: str) -> str:
    t50 = r.t50[pk]
    return f"{r.name}  T50 {t50:.0f} °C" if np.isfinite(t50) else f"{r.name}  T50 –"


NO_BASELINE = "no pre-transition images"


def plot(results: list[ExpResult], st: Settings) -> plt.Figure:
    peaks = list(st.peaks)
    with plt.rc_context(ORIGIN_RC):
        fig, axes = plt.subplots(1, len(peaks), figsize=(5.6 * len(peaks), 4.6), sharey=True,
                                 squeeze=False)
        for ax, pk in zip(axes[0], peaks):
            for r in results:
                if not r.ok:
                    ax.plot([], [], "o", color="0.65", ms=4, label=f"{r.name}  {NO_BASELINE}")
                    continue
                d = r.table[r.table["used"]]
                ax.plot(d["T_pv"], d[f"X_{pk}"], "-o", color=r.color, ms=3, lw=1.3, label=_lab(r, pk))
            ax.axhline(st.level, color="0.6", lw=0.9, ls=(0, (5, 3)), zorder=0)
            ax.set_ylim(-0.1, 1.2)
            ax.set_title(pk, fontsize=14)
            ax.set_xlabel("PV (°C)")
            ax.legend(loc="lower right", fontsize=10)
            if st.t_range:
                ax.set_xlim(*st.t_range)
        axes[0][0].set_ylabel("crystallised fraction X\n(normalised area)")
        fig.tight_layout()
    return fig


def write_html(results: list[ExpResult], st: Settings, path: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    peaks = list(st.peaks)
    fig = make_subplots(rows=1, cols=len(peaks), subplot_titles=peaks, shared_yaxes=True,
                        horizontal_spacing=0.06)
    for j, pk in enumerate(peaks, start=1):
        lg = "legend" if j == 1 else f"legend{j}"
        for r in results:
            if not r.ok:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", legend=lg,
                                         name=f"{r.name}  {NO_BASELINE}",
                                         marker=dict(color="#aaa", size=6)), 1, j)
                continue
            d = r.table[r.table["used"]]
            fig.add_trace(go.Scatter(
                x=d["T_pv"], y=d[f"X_{pk}"], mode="lines+markers", name=_lab(r, pk), legendgroup=r.name,
                legend=lg, marker=dict(size=4, color=r.color), line=dict(color=r.color, width=1.5),
                customdata=np.stack([d["time_kst"].str[11:19], d["file"]], axis=1),
                hovertemplate=(f"<b>{r.name} {pk}</b><br>T = %{{x:.1f}} °C<br>X = %{{y:.3f}}"
                               "<br>%{customdata[0]} · %{customdata[1]}<extra></extra>")), 1, j)
        fig.add_hline(y=st.level, line=dict(color="#999", dash="dash", width=1), row=1, col=j)
    leg = {("legend" if j == 1 else f"legend{j}"):
           dict(x=fig.layout[f"xaxis{'' if j == 1 else j}"].domain[1] - 0.005, xanchor="right",
                y=0.03, yanchor="bottom", bgcolor="rgba(255,255,255,0.8)", font=dict(size=12))
           for j in range(1, len(peaks) + 1)}
    fig.update_layout(template="simple_white", height=560, font=dict(family="Arial", size=14),
                      margin=dict(t=60, l=90, r=30, b=60), **leg)
    fig.update_xaxes(title="PV (°C)", showline=True, mirror=True, ticks="inside", linewidth=1.5,
                     **({"range": list(st.t_range)} if st.t_range else {}))
    fig.update_yaxes(showline=True, mirror=True, ticks="inside", linewidth=1.5, range=[-0.1, 1.2])
    fig.update_yaxes(title="crystallised fraction X (normalised area)", col=1)
    fig.write_html(path, include_plotlyjs=True, config={"displaylogo": False, "scrollZoom": True,
                                                         "toImageButtonOptions": {"scale": 3}})
    return path


# ═════════════════════════════ 실행 ═════════════════════════════
def run(config: Path = DEFAULT_CONFIG, geo: Geometry = Geometry(), workers: int | None = None,
        show: bool = True) -> list[ExpResult]:
    st = load_settings(config)
    print(f"📋 {st.path}  —  실험 {len(st.exps)}개, 피크 {', '.join(f'{k} ~{v}°' for k, v in st.peaks.items())}"
          f"  (± {st.search}° 에서 찾아 ±{st.fwhm_k:g}·FWHM 적분)")
    check_paths(st.exps)
    results = [run_exp(e, st, geo, workers, e.color or EXP_COLORS[i % len(EXP_COLORS)])
               for i, e in enumerate(st.exps)]
    d = OUT_DIR / "compare"
    d.mkdir(parents=True, exist_ok=True)
    stem = st.path.stem
    summary = pd.DataFrame([{"exp": r.name, **{f"T50 {k}": round(v, 1) for k, v in r.t50.items()},
                             **{f"{k} {lab}": round(w[i], 3) for k, w in (r.windows or {}).items()
                                for i, lab in ((0, "center"), (1, "FWHM"))},
                             "note": r.note} for r in results])
    print("\n📊 T50 (X = " + f"{st.level:g} 을 넘는 온도)\n" + summary.to_string(index=False))
    summary.to_csv(d / f"{stem}_T50.csv", index=False, encoding="utf-8-sig")
    pd.concat([r.table.assign(exp=r.name) for r in results]).to_csv(
        d / f"{stem}_frames.csv", index=False, encoding="utf-8-sig")
    fig = plot(results, st)
    png = d / f"{stem}.png"
    with plt.rc_context(ORIGIN_RC):
        fig.savefig(png)
    plt.close(fig)
    page = write_html(results, st, d / f"{stem}.html")
    print(f"\n💾 {png}\n💾 {page}\n💾 {d / f'{stem}_T50.csv'}")
    if show:
        import webbrowser
        webbrowser.open(page.resolve().as_uri())
    return results
