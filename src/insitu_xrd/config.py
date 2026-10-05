"""프로젝트 공통 설정.

실험마다 바뀌는 값(디텍터 기하 · 빔 에너지 · 기본 이미지 경로)은 프로젝트 폴더의 experiments.toml
[geometry] · [paths] 에서 읽음. 파일이나 항목이 없으면 아래 기본값 사용.
여기에는 코드 쪽 상수(HDF5 키, EPICS 기준시각, 출력 폴더 구조)만 둠.

이미지 경로는 자동으로 찾지 않음: experiments.toml 과 명령줄에 실제 경로를 그대로 적음.
"""
import tomllib
from pathlib import Path

# ===== 프로젝트 폴더 (코드 기준, 고정) =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"          # 로컬 보조 데이터 (온도 로그 등)
OUT_DIR = PROJECT_ROOT / "out"            # 모든 결과물 (out/<샘플>/...)
CACHE_DIR = OUT_DIR / "cache"             # 적분 결과 캐시 (지워도 다시 계산됨)
SETTINGS_FILE = PROJECT_ROOT / "experiments.toml"

# ===== HDF5 (areaDetector NeXus) =====
DSET = "entry/data/data"
ATTR = "entry/instrument/NDAttributes"
EPICS_EPOCH_OFFSET = 631_152_000          # 1990-01-01 UTC - 1970-01-01 UTC [s]


def load_toml(path: Path) -> dict:
    """TOML 읽기. 형식 오류면 어느 파일인지 알려 주고 종료."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"❌ {Path(path).name} 형식 오류: {e}") from e


def project_path(s: str | Path) -> Path:
    """설정 파일 속 경로: 전체 경로는 그대로, 상대경로는 프로젝트 폴더 기준."""
    p = Path(s)
    return p if p.is_absolute() else PROJECT_ROOT / p


_S = load_toml(SETTINGS_FILE) if SETTINGS_FILE.exists() else {}
_geo = _S.get("geometry", {})
_paths = _S.get("paths", {})

# ===== 디텍터 기하 / 빔 — experiments.toml [geometry] =====
PONI_PX = tuple(float(v) for v in _geo.get("poni_px", (514.0, 850.0)))   # PONI (x, y) [px]
SDD_MM = float(_geo.get("sdd_mm", 861.14))                                # 시료 → PONI 거리 [mm]
ENERGY_KEV = float(_geo.get("energy_kev", 13.0))                          # X-선 에너지 [keV]
ALPHA_DEG = float(_geo.get("alpha_deg", 18.9))                            # 디텍터 암 2θ [deg]
PIXEL_UM = float(_geo.get("pixel_um", 75.0))                              # 픽셀 크기 [µm]

# ===== 경로 생략한 명령(timing, center, profile, heatmap)의 기본값 — experiments.toml [paths] =====
DEFAULT_FOLDER = project_path(_paths.get("default_folder", r"Z:\exp\hkim\261004\images\testx1"))
DEFAULT_FILE = project_path(_paths.get("default_file", DEFAULT_FOLDER / "testx1_00089.h5"))


def out_dir(sample: str) -> Path:
    """out/<sample>/ 을 만들고 반환."""
    d = OUT_DIR / sample
    d.mkdir(parents=True, exist_ok=True)
    return d
