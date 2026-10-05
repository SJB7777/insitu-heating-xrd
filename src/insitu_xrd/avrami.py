"""결정화 변화 X(t) → Avrami(JMAK) 지수 n, 속도상수 k, (같은 시료의 등온 온도가 2개 이상이면) 활성화 에너지 Ea.

온도 구간을 판별하지 않고 데이터의 변화만 봄 → 등온·승온 어떤 측정에도 그대로 돌아감.
(승온 측정이면 n, k 는 '겉보기' 값이라 비등온 경고가 뜨고 Arrhenius 에서 빠짐)

설정: experiments.toml 의 [[iso]]  (측정 하나 = 비정질 시료 한 조각)
    [[iso]]
    sample = "x1"                 # 같은 sample 끼리 묶어 Arrhenius
    images = ['D:\\...\\x1_iso370']
    recipe = ['D:\\...\\recipe_....csv']   # (선택) 없으면 온도 없이 n, k 만
    T_iso  = 370                  # (선택) 이름 붙이기용 설정 온도. 실제 온도는 로그에서 계산
    start  = "10:05"              # (선택) 이 시각 사이 프레임만 (X 의 0 / 1 기준이 되는 처음·끝을 정함)
    end    = "11:30"
    t0     = "10:15:02"           # (선택) 변화 시작 시각을 직접 지정. 생략하면 JMAK 피팅으로 구함

    uv run python main.py avrami                  ← [[iso]] 전부
    uv run python main.py avrami --only x1        ← 이름 또는 sample 로 일부만
    uv run python main.py avrami --exp            ← [[iso]] 대신 [[exp]] (기존 승온 데이터로 시험)

분석
 1) X(t): compare 와 같은 방식 — 비정질로 나눈 R = I/I₀ − 1 의 피크 창(중심 ± fwhm_k·FWHM) 면적.
    X = 0 은 처음 n_norm 프레임, X = 1 은 마지막 n_norm 프레임
 2) t0: JMAK  X = 1 − exp(−(k·(t − t0))ⁿ)  를 n, k, t0 모두 자유롭게 피팅 (t0 = 변화가 실제로 시작된 시각).
    설정에 t0 를 적으면 그 시각을 쓰고, JMAK 의 τ 는 그 뒤 잠복시간으로 따로 맞춤
 3) Avrami 플롯: ln[−ln(1−X)] = n·ln(t − t0) + n·ln k   (fit_range 안의 X) → n, k
 4) 온도: X 가 10–90 % 인 동안의 PV 평균. 그동안 iso_span 넘게 변하면 비등온 → Arrhenius 제외
 5) 같은 sample 의 (등온) 온도가 2개 이상: ln k vs 1/T → Ea

결과: out/avrami/<설정이름>_avrami.png / .csv / _arrhenius.csv / _frames.csv
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.stats import linregress

from .compare import (DEFAULT_CONFIG, Experiment, Settings, _as_list, check_paths, load_exp,
                      load_settings, peak_areas)
from .config import OUT_DIR, load_toml, project_path
from .geometry import Geometry
from .heatmap import beam_ok
from .style import EXP_COLORS, ORIGIN_RC
from .temperature import fmt_kst, parse_time
from .tracks import level_temperature

KB_EV = 8.617333e-5               # 볼츠만 상수 [eV/K]
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v"]


# ═════════════════════════════ 설정 ═════════════════════════════
@dataclass(slots=True)
class IsoRun:
    exp: Experiment               # name · images · recipe · start/end
    sample: str
    t0: str | None = None         # 변화 시작 시각 직접 지정 (KST)


@dataclass(slots=True)
class IsoOptions:
    fit_range: tuple[float, float] = (0.15, 0.85)
    iso_span: float = 5.0                     # X 10–90 % 동안 온도 변화가 이보다 크면 비등온 [°C]
    edge_min: float = 5.0                     # 처음/마지막 이 시간 [min] 동안 X 변화로 기준 상태 확인
    edge_tol: float = 0.05                    # 그 동안 X 가 이보다 더 변하면 경고


def load_iso(path: Path = DEFAULT_CONFIG, section: str = "iso") -> tuple[Settings, list[IsoRun], IsoOptions]:
    """설정 파일의 [peaks]·[options] + [[iso]] (section="exp" 면 [[exp]]) 목록."""
    st = load_settings(path, need_exps=False)
    raw = load_toml(path)
    o = raw.get("options", {})
    opt = IsoOptions(fit_range=tuple(float(v) for v in o.get("fit_range", (0.15, 0.85))),
                     iso_span=float(o.get("iso_span", 5.0)))
    runs = []
    for i, e in enumerate(raw.get(section, []), start=1):
        sample = str(e.get("sample") or e.get("name") or f"run{i}")
        name = e.get("name") or (f"{sample}_{e['T_iso']:g}" if "T_iso" in e else sample)
        exp = Experiment(name=str(name), images=[project_path(s) for s in _as_list(e["images"])],
                         recipe=[project_path(s) for s in _as_list(e.get("recipe", []))],
                         start=e.get("start"), end=e.get("end"), color=e.get("color"))
        runs.append(IsoRun(exp, sample, e.get("t0")))
    if not runs:
        raise SystemExit(f"❌ {path}: [[{section}]] 항목이 없음 (README 의 '등온 실험' 참고)")
    return st, runs, opt


# ═════════════════════════════ 피팅 ═════════════════════════════
def _jmak(t, n, k, t0):
    return 1 - np.exp(-np.clip(k * (t - t0), 0, None) ** n)


def jmak_fit(t: np.ndarray, X: np.ndarray, t0_fixed: float | None = None) -> dict:
    """X = 1 − exp(−(k·(t − t0))ⁿ). t0_fixed 가 없으면 t0 도 피팅 (여러 초기값 중 잔차 최소).
    t [min]. 반환 t0 는 같은 시간축의 값."""
    out = dict(n_jmak=np.nan, k_jmak=np.nan, t0_fit=np.nan)
    m = np.isfinite(t) & np.isfinite(X)
    t, X = t[m], X[m]
    t05, t50 = (level_temperature(t, X, v, m=3) for v in (0.05, 0.5))
    if len(t) < 8 or not np.isfinite(t50):
        return out
    t05 = t05 if np.isfinite(t05) else t[0]
    lo_t0 = t[0] - 2 * (t50 - t[0])                  # t0 는 첫 프레임보다 조금 앞일 수도 있음
    best = None
    for n0 in (1.0, 2.0, 3.0, 4.0):
        for f in (0.0, 0.5, 1.0):
            t00 = t0_fixed if t0_fixed is not None else max(lo_t0 + 1e-6, t05 - f * (t50 - t05))
            if t50 - t00 <= 0:
                continue
            k0 = np.log(2) ** (1 / n0) / (t50 - t00)
            try:
                if t0_fixed is None:
                    p, _ = curve_fit(_jmak, t, X, p0=[n0, k0, t00], maxfev=20000,
                                     bounds=([0.3, k0 / 1e3, lo_t0], [10.0, k0 * 1e3, t50 - 1e-6]))
                else:
                    (pn, pk), _ = curve_fit(lambda tt, n, k: _jmak(tt, n, k, t0_fixed), t, X,
                                            p0=[n0, k0], bounds=([0.3, k0 / 1e3], [10.0, k0 * 1e3]),
                                            maxfev=20000)
                    p = (pn, pk, t0_fixed)
            except (RuntimeError, ValueError):
                continue
            sse = float(np.sum((_jmak(t, *p) - X) ** 2))
            if best is None or sse < best[0]:
                best = (sse, p)
    if best is not None:
        out.update(n_jmak=float(best[1][0]), k_jmak=float(best[1][1]), t0_fit=float(best[1][2]))
    return out


def avrami_fit(tm: np.ndarray, X: np.ndarray, rng: tuple[float, float]) -> dict:
    """Avrami 플롯 직선 피팅: ln(−ln(1−X)) = n·ln t + n·ln k. tm = t − t0 [min].
    X 가 처음 rng[1] 을 넘기 전(주된 변화 구간)의 점만 — 변화 뒤 잡음으로 범위에 들어온 점은 뺌."""
    m = (tm > 0) & np.isfinite(X) & (X > rng[0]) & (X < rng[1])
    good = np.flatnonzero(np.isfinite(X))
    if len(good):
        above = np.flatnonzero(median_filter(X[good], size=3, mode="nearest") >= rng[1])
        if len(above):
            m &= tm <= tm[good[above[0]]]
    out = dict(n=np.nan, n_err=np.nan, k=np.nan, r2=np.nan, npts=int(m.sum()))
    if m.sum() < 5:
        return out
    lr = linregress(np.log(tm[m]), np.log(-np.log(1 - X[m])))
    n = float(lr.slope)
    out.update(n=n, n_err=float(lr.stderr), k=float(np.exp(lr.intercept / n)) if n > 0 else np.nan,
               r2=float(lr.rvalue ** 2), t_fit_end=float(tm[m].max()))
    return out


# ═════════════════════════════ 측정 하나 ═════════════════════════════
@dataclass(slots=True)
class IsoResult:
    run: IsoRun
    T_iso: float = np.nan             # X 10–90 % 동안 평균 PV [°C]
    T_span: float = np.nan            # 그동안 PV 변화 폭 [°C]
    isothermal: bool = False
    t0: float = np.nan                # Unix (변화 시작)
    tm: np.ndarray = field(default_factory=lambda: np.empty(0))   # 프레임별 t − t0 [min]
    T: np.ndarray = field(default_factory=lambda: np.empty(0))
    files: np.ndarray = field(default_factory=lambda: np.empty(0))
    X: dict[str, np.ndarray] = field(default_factory=dict)
    fits: dict[str, dict] = field(default_factory=dict)          # 피크 → n, k, t50, ...
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.fits)


def _edge_change(tm: np.ndarray, X: np.ndarray, first: bool, minutes: float) -> float:
    """처음(또는 마지막) minutes 분 동안 X 의 선형 변화량."""
    m = np.isfinite(X) & ((tm <= tm[np.isfinite(X)][0] + minutes) if first
                          else (tm >= tm[np.isfinite(X)][-1] - minutes))
    return float(np.polyfit(tm[m], X[m], 1)[0] * minutes) if m.sum() >= 3 else np.nan


def _normalized(x, Z, ok, st: Settings, say=print) -> dict[str, np.ndarray]:
    """피크별 X: 처음 n_norm 정상 프레임 = 0, 마지막 n_norm 정상 프레임 = 1."""
    idx = np.flatnonzero(ok)
    base, final = idx[:st.n_norm], idx[-st.n_norm:]
    _, areas = peak_areas(x, Z, base, final, st, say)
    out = {}
    for pk, A in areas.items():
        a0, a1 = np.nanmedian(A[base]), np.nanmedian(A[final])
        X = (A - a0) / ((a1 - a0) or np.nan)
        X[~ok] = np.nan
        out[pk] = X
    return out


def _change_end(tt: np.ndarray, X: np.ndarray, opt: IsoOptions, drift: float = 0.1,
                flat: float = 0.1) -> int | None:
    """주된 변화(X 가 0.5 를 넘는 상승)가 끝나 평평해진 뒤, 이후 데이터가 그 수준에서 drift 넘게 벗어나면
    (승온·냉각으로 세기·위치가 변하는 등) 평평해진 구간 끝 인덱스. 계속 평평하면(등온) None.
    평평함 = edge_min 분 창의 변화 속도가 최대 속도의 flat 배 미만."""
    good = np.flatnonzero(np.isfinite(X))
    if len(good) < 10:
        return None
    xs = median_filter(X[good], size=5, mode="nearest")
    ts = tt[good]
    up = np.flatnonzero(xs >= 0.5)
    w = max(3, int(round(opt.edge_min / np.median(np.diff(ts)))))     # edge_min 분의 프레임 수
    if not len(up) or len(xs) <= w + 1:
        return None
    rate = np.abs(xs[w:] - xs[:-w])                                     # 창마다 변화량
    for j in range(up[0], len(xs) - w):
        if rate[j] < flat * rate.max():                                 # 평평해짐
            level = np.median(xs[j:j + w + 1])
            if np.any(np.abs(xs[j + w + 1:] - level) > drift):
                return int(good[j + w])
            return None
    return None


def analyze_iso(run: IsoRun, x: np.ndarray, Z: np.ndarray, t: np.ndarray, T: np.ndarray,
                files: np.ndarray, st: Settings, opt: IsoOptions = IsoOptions()) -> IsoResult:
    """프레임 프로파일 Z (시간순, start/end 로 이미 자른 것) → 피크별 X(t), n, k."""
    # 온도 로그가 있으면 로그 시간대 밖 프레임(덮어쓰기 후 남은 예전 측정 등)은 이 측정이 아님 → 제외
    if np.isfinite(T).any() and not np.isfinite(T).all():
        keep = np.isfinite(T)
        print(f"   온도 로그 밖 프레임 {int((~keep).sum())} 개 제외")
        Z, t, T, files = Z[keep], t[keep], T[keep], files[keep]
    ok = beam_ok(Z)
    if ok.sum() < 4 * st.n_norm:
        res = IsoResult(run, T=T, files=files)
        res.notes.append(f"프레임이 {ok.sum()} 개뿐")
        return res

    # 1) X(t): 처음 n_norm 프레임 = 0, 마지막 n_norm 프레임 = 1
    tt = (t - t[0]) / 60.0                           # 첫 프레임 기준 [min]
    Xs = _normalized(x, Z, ok, st, say=lambda *a: None)
    # 변화가 끝난 뒤 데이터가 다시 크게 변하면(승온 계속·냉각 등) 거기서 자름 → X=1 = 변화 직후 상태
    cut = None if run.exp.end else _change_end(tt, Xs[next(iter(Xs))], opt)
    if cut is not None and cut < len(t) - st.n_norm:
        k = cut + 1
        x_note = (f"분석 구간: 변화가 끝나 평평해진 {fmt_kst(t[cut])[11:]} 까지 "
                  f"(이후 {len(t) - k} 프레임은 X 가 다시 변해서 제외; end 로 직접 지정 가능)")
        Z, t, T, files, ok, tt = Z[:k], t[:k], T[:k], files[:k], ok[:k], tt[:k]
    else:
        x_note = None
    res = IsoResult(run, T=T, files=files)
    note = res.notes.append
    if x_note:
        note(x_note)
    res.X = _normalized(x, Z, ok, st)

    # 2) t0: 지정값 또는 첫 피크의 JMAK 피팅 (모든 피크가 같은 t0 를 씀)
    pk0 = next(iter(res.X))
    if run.t0:
        t0 = parse_time(run.t0, fmt_kst(t[0])[:10])
        src = "지정"
    else:
        j = jmak_fit(tt, res.X[pk0])
        if not np.isfinite(j["t0_fit"]):
            note(f"{pk0}: X 가 0.5 를 넘지 않음 (변화 없음) 또는 JMAK 피팅 실패")
            return res
        t0 = t[0] + j["t0_fit"] * 60
        src = f"{pk0} JMAK 피팅"
    res.t0 = t0
    res.tm = tm = (t - t0) / 60.0
    if t0 < t[0] - 30:
        note(f"t0 ({src}) 가 첫 이미지보다 {(t[0] - t0) / 60:.1f} 분 앞 — 변화가 기록 전에 시작됨")

    # 3) 피크별 Avrami 플롯 + JMAK (t0 고정 → 지정 t0 일 때는 τ 를 따로)
    for pk, X in res.X.items():
        f = avrami_fit(tm, X, opt.fit_range)
        j = jmak_fit(tm, X, None if run.t0 else 0.0)
        f.update(n_jmak=j["n_jmak"], k_jmak=j["k_jmak"], tau=j["t0_fit"])
        f["t50"] = level_temperature(tm, X, 0.5, m=3)
        res.fits[pk] = f
        if run.t0 and np.isfinite(f["tau"]) and abs(f["n"] - f["n_jmak"]) > 0.15 and abs(f["tau"]) > 0.5:
            note(f"{pk}: 지정 t0 뒤 잠복시간 τ = {f['tau']:.1f} min → Avrami 플롯 n {f['n']:.2f} 보다 "
                 f"JMAK n {f['n_jmak']:.2f} 이 더 믿을 만함")
        rise0 = _edge_change(tm, X, True, opt.edge_min)
        rise1 = _edge_change(tm, X, False, opt.edge_min)
        if rise0 > opt.edge_tol:
            note(f"{pk}: 처음부터 이미 변화 중 (처음 {opt.edge_min:g}분 X +{rise0:.2f}) → X=0 기준 왜곡")
        if rise1 > opt.edge_tol:
            note(f"{pk}: 아직 포화 안 됨 (마지막 {opt.edge_min:g}분 X +{rise1:.2f}) → X=1 기준 과소, n 왜곡 가능")

    # 4) 온도: 변화(X 10–90 %) 동안의 PV
    X0 = res.X[pk0]
    act = (X0 > 0.1) & (X0 < 0.9) & np.isfinite(T)
    if act.sum() >= 2:
        res.T_iso = float(np.mean(T[act]))
        res.T_span = float(np.ptp(T[act]))
        res.isothermal = res.T_span <= opt.iso_span
        if not res.isothermal:
            note(f"비등온: X 10–90 % 동안 T {T[act].min():.0f} → {T[act].max():.0f} °C — n, k 는 겉보기 값 "
                 "(Arrhenius 제외)")
    elif not np.isfinite(T).any():
        note("온도 없음 (recipe 없음 또는 로그 시간대 밖) — n, k 만")
    return res


def _arr_fit(T: pd.Series, k: pd.Series) -> tuple[float, float, float, float]:
    """ln k = ln k0 − Ea/(kB·T) → (Ea [eV], 오차, ln k0, R²). 점 2개면 오차·R² 는 NaN."""
    ok = np.isfinite(k) & np.isfinite(T) & (k > 0)
    if T[ok].round().nunique() < 2:
        return (np.nan,) * 4
    lr = linregress(1 / (T[ok] + 273.15), np.log(k[ok]))
    many = ok.sum() > 2
    return (-lr.slope * KB_EV, lr.stderr * KB_EV if many else np.nan, lr.intercept,
            lr.rvalue ** 2 if many else np.nan)


def arrhenius(table: pd.DataFrame) -> pd.DataFrame:
    """같은 sample · 피크의 등온 측정들: ln k vs 1/T → Ea [eV] (온도 2개 이상일 때만).
    Ea_eV = Avrami 플롯 k 로, Ea_jmak_eV = JMAK 피팅 k 로."""
    rows = []
    for (sample, pk), d in table[table["isothermal"]].groupby(["sample", "peak"], sort=False):
        Ea, err, lnk0, r2 = _arr_fit(d["T_iso"], d["k"])
        if not np.isfinite(Ea):
            continue
        Ej, ej, _, _ = _arr_fit(d["T_iso"], d["k_jmak"])
        rows.append({"sample": sample, "peak": pk, "n_temps": int(d["T_iso"].round().nunique()),
                     "Ea_eV": Ea, "Ea_err_eV": err, "lnk0": lnk0, "r2": r2,
                     "Ea_jmak_eV": Ej, "Ea_jmak_err_eV": ej})
    return pd.DataFrame(rows, columns=["sample", "peak", "n_temps", "Ea_eV", "Ea_err_eV", "lnk0", "r2",
                                       "Ea_jmak_eV", "Ea_jmak_err_eV"])


# ═════════════════════════════ 그림 ═════════════════════════════
def _style(results: list[IsoResult]) -> dict[str, tuple[str, str, str]]:
    """측정별 (색, 선 모양, 마커): 색 = sample, 선·마커 = 그 sample 안의 온도 순서."""
    samples = list(dict.fromkeys(r.run.sample for r in results))
    out = {}
    for si, s in enumerate(samples):
        rs = sorted((r for r in results if r.run.sample == s), key=lambda r: np.nan_to_num(r.T_iso, nan=1e9))
        for j, r in enumerate(rs):
            c = r.run.exp.color or EXP_COLORS[si % len(EXP_COLORS)]
            out[r.run.exp.name] = (c, LINESTYLES[j % len(LINESTYLES)], MARKERS[j % len(MARKERS)])
    return out


def plot(results: list[IsoResult], arr: pd.DataFrame, st: Settings, opt: IsoOptions) -> plt.Figure:
    peaks = list(st.peaks)
    sty = _style(results)
    good = [r for r in results if r.ok]
    with plt.rc_context(ORIGIN_RC):
        fig, axes = plt.subplots(len(peaks), 3, figsize=(17, 4.8 * len(peaks)), squeeze=False)
        for row, pk in zip(axes, peaks):
            ax_x, ax_a, ax_r = row
            for r in good:
                c, ls, mk = sty[r.run.exp.name]
                f = r.fits[pk]
                X, tm = r.X[pk], r.tm
                T = f"{r.T_iso:.0f} °C" + ("" if r.isothermal else " (ramp)") if np.isfinite(r.T_iso) else "T –"
                lab = f"{r.run.exp.name}  {T}  n={f['n']:.2f}"
                ax_x.plot(tm, X, mk, color=c, ms=3, alpha=0.6, mfc="none")
                if np.isfinite(f["n_jmak"]):
                    ts = np.linspace(np.nanmin(tm), np.nanmax(tm), 400)
                    ax_x.plot(ts, _jmak(ts, f["n_jmak"], f["k_jmak"], f["tau"]), ls=ls, color=c, lw=1.8,
                              label=lab)
                else:
                    ax_x.plot([], [], ls=ls, color=c, label=lab)
                # Avrami 플롯: 피팅 범위 점은 채우고 나머지는 흐리게
                with np.errstate(all="ignore"):
                    lx, ly = np.log(tm), np.log(-np.log(1 - X))
                inr = (tm > 0) & (X > opt.fit_range[0]) & (X < opt.fit_range[1])
                if np.isfinite(f["n"]):                    # 피팅에 쓴 점만 (주된 변화 구간)
                    inr &= tm <= f["t_fit_end"]
                ax_a.plot(lx[(tm > 0) & ~inr], ly[(tm > 0) & ~inr], mk, color=c, ms=3, alpha=0.25, mfc="none")
                if inr.any():
                    ax_a.plot(lx[inr], ly[inr], mk, color=c, ms=4)
                if np.isfinite(f["n"]):
                    xs = np.array([lx[inr].min(), lx[inr].max()])
                    ax_a.plot(xs, f["n"] * (xs + np.log(f["k"])), ls=ls, color=c, lw=1.6,
                              label=f"{r.run.exp.name}  n = {f['n']:.2f} ± {f['n_err']:.2f}")
            ax_x.axhline(0.5, color="0.6", lw=0.9, ls=(0, (5, 3)), zorder=0)
            ax_x.axvline(0, color="0.6", lw=0.9, ls=":", zorder=0)
            ax_x.set(xlabel="t − t$_0$ (min)", ylabel="X", ylim=(-0.1, 1.15))
            ax_x.set_title(f"{pk}  X(t)  (lines: JMAK fit)", fontsize=12, loc="left")
            ax_x.legend(fontsize=9, loc="lower right")
            for v in opt.fit_range:
                ax_a.axhline(np.log(-np.log(1 - v)), color="0.7", lw=0.8, ls=":")
            ax_a.set(xlabel="ln[(t − t$_0$) / min]", ylabel="ln[−ln(1 − X)]")
            ax_a.set_title(f"{pk}  Avrami plot  (filled: {opt.fit_range[0]:g} < X < {opt.fit_range[1]:g})",
                           fontsize=12, loc="left")
            ax_a.legend(fontsize=9, loc="upper left")
            # Arrhenius (등온 측정만)
            for r in good:
                c, _, mk = sty[r.run.exp.name]
                k = r.fits[pk]["k"]
                if r.isothermal and np.isfinite(k):
                    ax_r.plot(1000 / (r.T_iso + 273.15), np.log(k), mk, color=c, ms=8, mec="black")
            for _, a in arr[arr["peak"] == pk].iterrows():
                rs = [r for r in good if r.run.sample == a["sample"] and r.isothermal]
                c = sty[rs[0].run.exp.name][0]
                inv = np.array([1 / (r.T_iso + 273.15) for r in rs])
                xs = np.array([inv.min(), inv.max()])
                err = f" ± {a['Ea_err_eV']:.2g}" if np.isfinite(a["Ea_err_eV"]) else ""
                jm = f"  (JMAK {a['Ea_jmak_eV']:.2f})" if np.isfinite(a["Ea_jmak_eV"]) else ""
                ax_r.plot(1000 * xs, a["lnk0"] - a["Ea_eV"] / KB_EV * xs, color=c, lw=1.5,
                          label=f"{a['sample']}  E$_a$ = {a['Ea_eV']:.2f}{err} eV{jm}")
            ax_r.set(xlabel="1000 / T (K$^{-1}$)", ylabel="ln(k / min$^{-1}$)")
            ax_r.set_title(f"{pk}  Arrhenius (k from Avrami plot)", fontsize=12, loc="left")
            if len(arr[arr["peak"] == pk]):
                ax_r.legend(fontsize=9)
            else:
                ax_r.text(0.5, 0.5, "Ea: needs ≥ 2 isothermal\ntemperatures per sample", transform=ax_r.transAxes,
                          ha="center", va="center", fontsize=11, color="0.4")
        fig.tight_layout()
    return fig


# ═════════════════════════════ 실행 ═════════════════════════════
def run(config: Path = DEFAULT_CONFIG, geo: Geometry = Geometry(), workers: int | None = None,
        only: list[str] | None = None, show: bool = True, section: str = "iso") -> list[IsoResult]:
    st, runs, opt = load_iso(config, section)
    if only:
        runs = [r for r in runs if r.exp.name in only or r.sample in only]
        if not runs:
            raise SystemExit(f"❌ --only {' '.join(only)} 에 맞는 [[{section}]] 없음")
    print(f"📋 {Path(config)} [[{section}]]  —  측정 {len(runs)}개, 피크 {', '.join(st.peaks)}, "
          f"Avrami 피팅 {opt.fit_range[0]:g} < X < {opt.fit_range[1]:g}")
    check_paths([r.exp for r in runs])
    results = []
    for r in runs:
        d = load_exp(r.exp, st, geo, workers)
        res = analyze_iso(r, d.x, d.Z, d.t, d.T, d.files, st, opt)
        if np.isfinite(res.t0):
            T = (f"T = {res.T_iso:.1f} °C (변화 폭 {res.T_span:.1f} °C)" if np.isfinite(res.T_iso) else "T 없음")
            print(f"   t0 = {fmt_kst(res.t0)} ({'지정' if r.t0 else 'JMAK 피팅'}),  {T}")
        for pk, f in res.fits.items():
            print(f"   {pk:>6}: n = {f['n']:.2f} ± {f['n_err']:.2f}  k = {f['k']:.3g} /min  "
                  f"t50 = {f['t50']:.1f} min  R² = {f['r2']:.3f}  ({f['npts']} pts)  |  "
                  f"JMAK n = {f['n_jmak']:.2f}")
        for msg in res.notes:
            print(f"   ⚠️  {msg}")
        results.append(res)

    table = pd.DataFrame([{"run": r.run.exp.name, "sample": r.run.sample, "peak": pk,
                           "T_iso": r.T_iso, "T_span": r.T_span, "isothermal": r.isothermal,
                           "t0_kst": fmt_kst(r.t0), **f, "note": " / ".join(r.notes)}
                          for r in results for pk, f in r.fits.items()])
    for r in results:
        if not r.ok:
            print(f"\n❌ {r.run.exp.name}: 분석 못 함 — {' / '.join(r.notes)}")
    if table.empty:
        raise SystemExit("❌ 분석된 측정이 없음")
    arr = arrhenius(table)

    d = OUT_DIR / "avrami"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(config).stem}_{section}_avrami"
    cols = ["run", "peak", "T_iso", "isothermal", "n", "n_err", "k", "t50", "r2", "n_jmak"]
    print("\n📊 Avrami (k [1/min], t50 = t0 부터 X=0.5 까지 [min])\n"
          + table[cols].round(3).to_string(index=False))
    if len(arr):
        print("\n📈 Arrhenius\n" + arr.round(3).to_string(index=False))
    table.to_csv(d / f"{stem}.csv", index=False, encoding="utf-8-sig")
    arr.to_csv(d / f"{stem}_arrhenius.csv", index=False, encoding="utf-8-sig")
    pd.concat([pd.DataFrame({"run": r.run.exp.name, "file": r.files, "t_min": r.tm, "T_pv": r.T,
                             **{f"X_{pk}": X for pk, X in r.X.items()}}) for r in results if r.ok]
              ).to_csv(d / f"{stem}_frames.csv", index=False, encoding="utf-8-sig")
    fig = plot(results, arr, st, opt)
    png = d / f"{stem}.png"
    with plt.rc_context(ORIGIN_RC):
        fig.savefig(png)
    print(f"\n💾 {png}\n💾 {d / f'{stem}.csv'}\n💾 {d / f'{stem}_arrhenius.csv'}")
    if show:
        plt.show()
    plt.close(fig)
    return results
