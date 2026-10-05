"""통합 CLI.

사용:
    python main.py timing   [FOLDER]                 # 파일 생성/EPICS 시각 비교 → CSV
    python main.py center   [FILE] --width 100       # 중앙 스트립 픽셀 프로파일
    python main.py profile  [FILE] --width 10        # 단일 이미지 2θ 프로파일 + 피크 피팅
    python main.py heatmap  [FOLDER] --temp --at 5 15:55:00   # 시간 히트맵 + 온도
    python main.py overview FOLDER [FOLDER ...] [CSV] # 온도 + 히트맵 + 온도 구간 한 장
    python main.py overview x1                       # experiments.toml 의 [[exp]] 이름 → 그 images · recipe
    python main.py compare  [experiments.toml]       # 실험별 피크 X vs 온도, T50 비교
    python main.py avrami   [experiments.toml]       # 등온 [[iso]] → Avrami n, k, (Ea)
    python main.py merge-temp [CSV ...]              # 온도 로그 병합
    python main.py temp     15:55:00 FILE.h5 FOLDER  # 특정 시각/프레임의 온도

이미지 폴더 · 파일은 실제 경로를 그대로 적음 (자동으로 찾지 않음).
결과물은 모두 out/<샘플 폴더명>/ 에 저장됨. 각 명령의 옵션은 `python main.py <명령> -h`.
"""
import argparse
from pathlib import Path

from .config import DEFAULT_FILE, DEFAULT_FOLDER, SETTINGS_FILE
from .geometry import Geometry
from .temperature import DEFAULT_LOG, TEMP_DIR


def _add_geometry(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("geometry (기본값: experiments.toml [geometry])")
    d = Geometry()
    g.add_argument("--alpha", type=float, default=d.alpha_deg, help="디텍터 암 2θ [deg]")
    g.add_argument("--sdd", type=float, default=d.sdd * 1e3, help="시료→PONI 거리 [mm]")
    g.add_argument("--poni", type=float, nargs=2, metavar=("X", "Y"), default=(d.xc, d.yc),
                   help="PONI (x, y) [px]")
    g.add_argument("--energy", type=float, default=d.energy_kev, help="X-선 에너지 [keV]")


def _geometry(a: argparse.Namespace) -> Geometry:
    return Geometry(sdd=a.sdd * 1e-3, xc=a.poni[0], yc=a.poni[1], alpha_deg=a.alpha,
                    energy_kev=a.energy)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="insitu-xrd", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<명령>")

    p = sub.add_parser("timing", help="파일 생성시각 vs EPICS 촬영시각 → CSV")
    p.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER)
    p.add_argument("--pattern", default="*.h5")

    p = sub.add_parser("center", help="중앙 스트립 세로 픽셀 프로파일 + 피크")
    p.add_argument("file", nargs="?", type=Path, default=DEFAULT_FILE)
    p.add_argument("--width", type=int, default=100, help="스트립 폭 [px]")
    p.add_argument("--reduce", choices=["mean", "sum"], default="mean")
    p.add_argument("--prominence", type=float, default=1, help="음수면 자동 (3σ)")
    p.add_argument("--min-distance", type=int, default=10, help="피크 간 최소 간격 [px]")
    p.add_argument("--no-show", action="store_true")

    p = sub.add_parser("profile", help="단일 이미지 2θ 프로파일 + 피크 피팅")
    p.add_argument("file", nargs="?", type=Path, default=DEFAULT_FILE)
    p.add_argument("--width", type=int, default=10, help="스트립 폭 [px]")
    p.add_argument("--bin", type=float, default=0.02, help="2θ 빈 [deg]")
    p.add_argument("--smooth", type=int, default=7, help="Savitzky–Golay 창 (0=끔)")
    p.add_argument("--prominence", type=float, default=None, help="미지정 시 자동")
    p.add_argument("--min-distance", type=float, default=0.15, help="피크 간 최소 간격 [deg]")
    p.add_argument("--no-show", action="store_true")
    _add_geometry(p)

    p = sub.add_parser("heatmap", help="폴더 전체 → 2θ 시간 히트맵 (+ 온도)")
    p.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER)
    p.add_argument("--pattern", default="*.h5")
    p.add_argument("-j", "--workers", type=int, default=None)
    p.add_argument("--region", choices=["full", "strip"], default="full")
    p.add_argument("--width", type=int, default=10, help="region=strip 일 때 폭 [px]")
    p.add_argument("--bin", type=float, default=0.02, help="2θ 빈 [deg]")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--log", action="store_true", help="log color scale")
    p.add_argument("--seconds", action="store_true", help="시간축 단위를 초로")
    p.add_argument("--temp", nargs="?", type=Path, const=DEFAULT_LOG, default=None, metavar="CSV",
                   help=f"온도 로그 CSV (경로 생략 시 {DEFAULT_LOG.name})")
    p.add_argument("--at", nargs="+", default=[], metavar="T",
                   help="표시할 시각: 숫자=상대시간(분/초), 'HH:MM:SS'=KST 시각")
    p.add_argument("--no-show", action="store_true")
    _add_geometry(p)

    p = sub.add_parser("overview", help="폴더(들) + 온도 CSV → 온도·히트맵·구간 한 장 (Origin 스타일)")
    p.add_argument("paths", nargs="+", type=Path,
                   help="h5 이미지 폴더 경로 (여러 개면 시간순으로 이어 붙임) [+ 온도 로그 .csv, "
                        "생략 시 시간이 맞는 로그 자동 선택]. experiments.toml 의 실험 이름(x1 등)도 가능")
    p.add_argument("--pattern", default="*.h5")
    p.add_argument("-j", "--workers", type=int, default=None)
    p.add_argument("--bin", type=float, default=0.02, help="2θ 빈 [deg]")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--log", action="store_true", help="log color scale")
    p.add_argument("--seconds", action="store_true", help="시간축 단위를 초로")
    p.add_argument("--cmap", default="inferno", help="컬러맵 (예: jet, viridis)")
    p.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"), default=(25.0, 99.7),
                   help="히트맵 색 범위 = 세기 백분위 LO~HI (기본 25 99.7, 대비 약하게: 1 99.5)")
    p.add_argument("--no-segments", action="store_true", help="승온/유지/하온 구간 표시 끄기")
    p.add_argument("--hold-rate", type=float, default=1.5,
                   help="|dT/dt| 가 이보다 작으면 유지 구간 [°C/min]")
    p.add_argument("--min-seg", type=float, default=90.0, help="이보다 짧은 구간은 무시 [s]")
    p.add_argument("--no-match", action="store_true",
                   help="폴더 여러 개일 때 폴더별 세기 수준 맞추지 않음")
    p.add_argument("--peaks", type=float, nargs="+", default=(), metavar="2θ",
                   help="추적할 2θ 위치를 직접 추가 (자동 검출에 더해서)")
    p.add_argument("--no-track", action="store_true", help="피크 추적(나타남/사라짐) 끄기")
    p.add_argument("--peaks-config", type=Path, default=None, metavar="TOML",
                   help="전이 온도 상자에 쓸 피크 설정 (생략 시 experiments.toml 의 [peaks]·[options])")
    p.add_argument("--no-transition", action="store_true", help="전이 온도 상자 끄기")
    p.add_argument("--keep-orphans", action="store_true",
                   help="온도 로그와 시간이 안 겹치는 이미지 블록(덮어쓰기 잔여 등)도 그대로 표시")
    p.add_argument("--level", type=float, default=0.5,
                   help="전이 기준: 정규화 면적(최댓값=1)이 이 값을 넘는 시점 (기본 0.5)")
    p.add_argument("--browser", action="store_true",
                   help="--no-show 여도 인터랙티브 HTML 을 브라우저로 열기")
    p.add_argument("--mpl", action="store_true",
                   help="브라우저 대신 matplotlib 창 두 개로 보기 (시간축 그림 / 2θ·온도축 그림)")
    p.add_argument("--no-html", action="store_true", help="인터랙티브 HTML 만들지 않음")
    p.add_argument("--no-fill", action="store_true",
                   help="로그 끊긴 구간을 이웃 램프 외삽으로 추정하지 않음")
    p.add_argument("--no-show", action="store_true")
    _add_geometry(p)

    p = sub.add_parser("compare", help="설정 파일의 실험들: 피크별 결정화 분율 X vs 온도 + T50")
    p.add_argument("config", nargs="?", type=Path, default=None,
                   help="실험 목록 TOML (생략 시 프로젝트 폴더의 experiments.toml)")
    p.add_argument("-j", "--workers", type=int, default=None)
    p.add_argument("--no-show", action="store_true", help="브라우저로 열지 않음")
    _add_geometry(p)

    p = sub.add_parser("avrami", help="등온 측정([[iso]]) → Avrami 지수 n · 속도상수 k · 활성화 에너지 Ea")
    p.add_argument("config", nargs="?", type=Path, default=None,
                   help="설정 TOML (생략 시 프로젝트 폴더의 experiments.toml)")
    p.add_argument("--only", nargs="+", default=None, metavar="NAME",
                   help="이 이름(또는 sample)의 [[iso]] 만 분석")
    p.add_argument("--exp", action="store_true",
                   help="[[iso]] 대신 [[exp]] 목록 분석 (기존 승온 데이터로 시험; 비등온이라 n·k 는 겉보기 값)")
    p.add_argument("-j", "--workers", type=int, default=None)
    p.add_argument("--no-show", action="store_true", help="그림 창 띄우지 않음")
    _add_geometry(p)

    p = sub.add_parser("merge-temp", help="온도 로그 CSV 여러 개 → 하나로 병합")
    p.add_argument("csvs", nargs="*", type=Path, help=f"생략 시 {TEMP_DIR / 'raw'}/*.csv 전부")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_LOG)

    p = sub.add_parser("temp", help="특정 시각 / h5 프레임 / 폴더의 온도 조회")
    p.add_argument("targets", nargs="+",
                   help="'HH:MM:SS' 또는 'YYYY-MM-DD HH:MM:SS' (KST), .h5 파일, h5 폴더")
    p.add_argument("--csv", type=Path, default=DEFAULT_LOG, help="온도 로그 CSV")
    p.add_argument("--max-gap", type=float, default=5.0, help="로그 샘플과 이보다 멀면 NaN [s]")
    return ap


def _overview_inputs(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """overview 인자 → (이미지 폴더들, 온도 CSV 들). 없는 경로가 experiments.toml 의 실험 이름이면
    그 실험의 images · recipe 를 그대로 씀."""
    from .compare import find_exp
    folders, csvs = [], []
    for p in paths:
        if p.suffix.lower() == ".csv":
            csvs.append(p)
        elif p.exists():
            folders.append(p)
        elif (e := find_exp(str(p))) is not None:
            print(f"📋 {SETTINGS_FILE.name} [[exp]] {e.name}: {', '.join(map(str, e.images))}")
            folders += e.images
            csvs += e.recipe
        else:
            raise SystemExit(f"❌ 폴더 없음: {p}  (전체 경로 또는 {SETTINGS_FILE.name} 의 실험 이름)")
    if not folders:
        raise SystemExit("이미지 폴더를 하나 이상 지정하세요")
    return folders, csvs


def _num_or_str(v: str) -> float | str:
    try:
        return float(v)
    except ValueError:
        return v


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)

    # 무거운 import (matplotlib, scipy 등) 는 실제로 쓰는 명령에서만
    match a.cmd:
        case "timing":
            from . import timing
            timing.run(a.folder, a.pattern)

        case "center":
            from . import center_profile
            center_profile.run(a.file, a.width, a.reduce,
                               prominence=None if a.prominence < 0 else a.prominence,
                               min_distance=a.min_distance, show=not a.no_show)

        case "profile":
            from . import tth_profile
            tth_profile.run(a.file, _geometry(a), strip_width=a.width, tth_bin=a.bin,
                            smooth=a.smooth, prominence=a.prominence,
                            min_distance=a.min_distance, show=not a.no_show)

        case "heatmap":
            from . import heatmap as hm
            from .integrate import ProfileOptions
            cfg = hm.Config(
                folder=a.folder, pattern=a.pattern, geometry=_geometry(a),
                profile=ProfileOptions(tth_bin=a.bin, region=a.region, strip_width=a.width,
                                       normalize=a.normalize),
                plot=hm.PlotOptions(log_color=a.log, time_unit="s" if a.seconds else "min",
                                    show=not a.no_show),
                temp_log=a.temp, at=tuple(_num_or_str(v) for v in a.at), workers=a.workers,
            )
            hm.run(cfg)

        case "overview":
            from . import heatmap as hm
            from . import overview
            from .integrate import ProfileOptions
            from .segments import SegmentOptions
            from .tracks import TrackOptions
            folders, csvs = _overview_inputs(a.paths)
            cfg = hm.Config(
                folder=folders[0], pattern=a.pattern, geometry=_geometry(a),
                profile=ProfileOptions(tth_bin=a.bin, normalize=a.normalize),
                plot=hm.PlotOptions(log_color=a.log, time_unit="s" if a.seconds else "min",
                                    cmap=a.cmap, show=not a.no_show, clip=tuple(a.clip)),
                temp_log=csvs or None, workers=a.workers,      # 로그 없으면 overview 가 자동 선택
            )
            overview.run(cfg, None if a.no_segments else
                         SegmentOptions(hold_rate=a.hold_rate, min_duration=a.min_seg,
                                        fill_gaps=not a.no_fill),
                         extra=folders[1:], match=not a.no_match, html=not a.no_html,
                         browser=a.browser, mpl=a.mpl, keep_orphans=a.keep_orphans,
                         peaks_cfg=None if a.no_transition else (a.peaks_config or SETTINGS_FILE),
                         track_opt=None if a.no_track else TrackOptions(manual=tuple(a.peaks), level=a.level))

        case "compare":
            from . import compare
            compare.run(a.config or SETTINGS_FILE, _geometry(a), a.workers, show=not a.no_show)

        case "avrami":
            from . import avrami
            avrami.run(a.config or SETTINGS_FILE, _geometry(a), a.workers, only=a.only,
                       show=not a.no_show, section="exp" if a.exp else "iso")

        case "merge-temp":
            from . import temperature
            csvs = a.csvs or sorted((TEMP_DIR / "raw").glob("*.csv"))
            df = temperature.merge(csvs, a.out)
            print(f"{len(csvs)} files → {a.out}  ({len(df)} rows, "
                  f"{df['Timestamp'].iloc[0][:19]} ~ {df['Timestamp'].iloc[-1][:19]})")

        case "temp":
            from . import temperature
            temperature.lookup(a.targets, a.csv, a.max_gap)


if __name__ == "__main__":
    main()
