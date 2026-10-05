# insitu-heating-xrd — 승온 in-situ 2D XRD 분석 (261004 빔타임)

Eiger HDF5 디텍터 이미지 → 2θ 프로파일 / 시간 히트맵 / 피크 추적.

## 구조

```
main.py                  진입점 (서브커맨드 CLI)
src/insitu_xrd/
  config.py              HDF5 키 등 코드 상수 + experiments.toml [geometry]·[paths] 읽기
  style.py               그림 공통 스타일 (Origin rc, 구간·피크 색)
  geometry.py            Geometry (캘리브레이션 값, 2θ/q/d 변환)
  io.py                  이미지 / EPICS 타임스탬프 읽기
  integrate.py           RadialIntegrator (이미지 → 2θ 프로파일)
  peaks.py               피크 검출 · 가우시안 피팅
  timing.py              [timing]  파일 생성시각 vs EPICS 촬영시각
  center_profile.py      [center]  중앙 스트립 픽셀 프로파일
  tth_profile.py         [profile] 단일 이미지 2θ 프로파일
  heatmap.py             [heatmap] 폴더 전체 시간 히트맵 (+ 온도)
  temperature.py         [merge-temp, temp] 온도 로그 병합 / 시각별 온도 조회
  segments.py            온도 프로파일 → 승온/유지/하온 구간 자동 분할
  tracks.py              피크 추적 · 전이(기준선 넘나듦) 판단
  overview.py            [overview] 온도 + 히트맵 + 구간별 프로파일 한 장
  interactive.py         overview 의 인터랙티브 HTML (Plotly)
  compare.py             [compare] 실험별 X vs 온도 · T50
data/temperature/
  raw/                   온도 컨트롤러 원본 로그
  manual_data_20261004_x1.csv   병합본 (온도 조회 기본값)
out/<샘플>/              모든 결과물 (git 제외)
out/cache/               적분 결과 캐시 (지워도 다시 계산됨)
```

## 사용

```powershell
uv sync
uv run python main.py                        # 인자 없이 (VS Code ▶ 포함) = compare (experiments.toml)
uv run python main.py --no-show              # 위와 같되 브라우저 안 염 (명령 없이 옵션만 → 기본 명령에)
uv run python main.py -h                     # 명령 목록
uv run python main.py timing  Z:\exp\hkim\261004\images\testx1
uv run python main.py center  Z:\...\testx1_00089.h5 --width 100
uv run python main.py profile Z:\...\testx1_00089.h5 --alpha 18.9
uv run python main.py heatmap Z:\...\testx1 --log -j 8 --no-show
```

### 온도 + 히트맵 + 온도 구간 한 장 (Origin 스타일)

```powershell
uv run python main.py overview x1                                    # experiments.toml 의 [[exp]] x1 (images · recipe 그대로)
uv run python main.py overview Z:\exp\hkim\261004\images\test2x1
uv run python main.py overview Z:\...\testx1 Z:\...\test2x1          # 폴더 여러 개 → 시간순으로 이어 붙임
uv run python main.py overview Z:\...\test2x1 data\temperature\other.csv   # 온도 로그 지정 (생략 시 자동 선택)
```

가로축 = 시간: 위에 구간 라벨 → 온도 그래프 → 2θ 히트맵(세로=2θ) 이 시간축 공유, 오른쪽에 구간 경계 프로파일 → `out/<샘플>/<샘플>_overview.png` (300 dpi).
폴더 여러 개면 결과 이름은 `testx1+test2x1`, 이미지 없는 시간은 회색 "no images".

승온 / 유지 / 하온 구간 자동 감지 (`segments.py`, 온도 로그 SV 기준, 없으면 PV):
- 구간 경계를 온도·히트맵에 세로 점선으로, 구간 라벨(`Heat 10 °C/min · 400 → 650 °C · 25 min · 16:11–16:36`)은 그래프 위 바깥에
- 오른쪽: 각 경계 시점의 2θ 프로파일을 시간순으로 쌓고 시간·온도 표시 (빔 꺼진 어두운 프레임, 이미지 없는 경계는 건너뜀)
- 온도 로그가 끊긴 시간은 이웃 램프를 연장해 추정 → 회색 점선 `est.`, 온도 앞에 `~`, CSV `estimated=True`
  (예: 15:23–15:51 은 로그 없음 → 284 °C 부터 10 °C/min 역외삽으로 ~15:25 에 승온 시작 추정)
- 폴더 여러 개면 폴더별 세기 수준을 마지막 폴더에 맞춤 (배율 출력, testx1 ≈ ×8.4)
- 출력: `<샘플>_segments.csv`, `_frames.csv` 에 `segment`, `temp_profile`, `temp_estimated` 열

**피크 추적 (상변이 추정)** (`tracks.py`):
- 프레임마다 피크를 찾고 이웃 프레임끼리 이어 붙여 추적 (열팽창으로 위치가 조금씩 움직여도 따라감)
- 히트맵 위에 피크 위치(P1, P2 …) 점선, 아래 패널에 **피크 면적 vs 시간**, 오른쪽 아래 **피크 면적 vs 온도**
  (승온 실선 / 하온 점선 / 유지 점선 → 승온·하온 차이(히스테리시스) 확인)
- 전이 시점 = 정규화 면적(최댓값=1)이 0.5 를 위로/아래로 넘는 프레임 (▲/▼ + 온도, `--level` 로 변경). 잠깐 튀는 건 무시, 이미지 공백 너머는 판단 안 함
  (예: testx1 18.40° · 21.38° 가 374–376 °C 에서 나타남, IGOx1 19.0° · 21.86° 가 395–405 °C)
- `--peaks 18.4 21.4` 로 위치 직접 추가, `--no-track` 으로 끄기
- 출력: `<샘플>_peaks.csv` (프레임별 위치·면적·검출 여부), `<샘플>_peak_events.csv`

**전이 온도 상자** (왼쪽 맨 아래 피크 범례 옆, 시료·시작 시각 포함 — 왼쪽 아래만 캡처해도 됨): `experiments.toml` 의 `[peaks]`·`[options]` 로 compare 와 같은 방식의 T50
(승온 중 X = 0.5) 을 피크(hkl)별로 크게 표시. 추적된 피크도 같은 hkl 이름 · T50 으로 표시됨. 결과 `<샘플>_transition.csv`.
다른 설정: `--peaks-config 파일.toml`, 끄기: `--no-transition`.

**온도 로그 자동 선택**: overview 에 CSV 를 안 주면 `data/temperature` 아래(하위 폴더 포함) 로그 중 프레임 절대시각과 가장 많이 겹치는 것을 고름.

**덮어쓰기 잔여 블록 자동 제외**: 온도 매칭은 절대시각(EPICS) 기준. 이미지 공백으로 나뉜 블록 중 온도 로그와 한 프레임도 안 겹치는 블록
(예: IGOx4 의 03:12–03:18 이전 측정 35 프레임)은 다른 측정으로 보고 그림·분석에서 뺌 (콘솔에 표시, `--keep-orphans` 로 유지).

**보기**: `--no-show` 없이 실행하면 인터랙티브 HTML 이 브라우저로 열림 (화면 폭에 맞춰져 잘리지 않음).
`--mpl` 이면 대신 matplotlib 창 두 개 — ① 시간축 그림(온도·히트맵·피크 면적 vs 시간) ② 2θ·온도축 그림(경계 프로파일·피크 면적 vs 온도) — 를 화면 크기에 맞춰 나란히 띄움.
PNG 는 항상 한 장짜리로 저장.

**인터랙티브 HTML** (`<샘플>_overview.html`, 자동 생성, 인터넷 없이 열림; `--browser` 로 `--no-show` 여도 열기, `--no-html` 로 끄기):
- 마우스 휠/드래그로 확대, 더블클릭 원래대로. 온도·히트맵 시간축 연동
- 확대하면 온도 그래프 아래 온도 눈금이 촘촘해지고(최소 3 s 간격까지), 온도 세로축도 보이는 범위에 맞춰짐
- 히트맵 위 마우스: 시간 / KST / 온도 / 2θ / 세기. 클릭하면 그 시점 프로파일을 오른쪽에 (Shift+클릭 누적)
- 히트맵 **Alt+클릭**: 그 2θ 의 세기 vs 시간을 피크 패널에 추가
- 오른쪽 범례 클릭으로 프로파일 켜고 끄기, 카메라 아이콘으로 PNG 저장
- matplotlib 창(`--mpl`)에서도 확대하면 온도 눈금이 촘촘해짐

옵션: `--clip 25 99.7` (히트맵 색 범위 = 세기 백분위, 기본값이 고대비; 예전처럼 `--clip 1 99.5`), `--cmap jet`, `--log`, `--normalize`, `--seconds`, `--bin`, `--no-show`,
`--hold-rate 1.5` (이보다 느리면 유지 [°C/min]), `--min-seg 90` (더 짧은 구간 무시 [s]),
`--no-segments`, `--no-fill` (로그 공백 추정 끄기), `--no-match` (폴더별 세기 맞춤 끄기)

### 디텍터 기하 / 빔

`experiments.toml` 의 `[geometry]` 만 고치면 모든 명령 기본값이 바뀜 (`config.py` 는 이 값을 읽기만 함, 형식 오류면 모든 명령이 멈추고 알려 줌):

```toml
[geometry]
poni_px    = [514.0, 850.0]   # PONI (x, y) [px]
sdd_mm     = 861.14           # 시료 → PONI 거리 [mm]
energy_kev = 13.0             # X-선 에너지 [keV]
alpha_deg  = 18.9             # 디텍터 암 2θ [deg]
pixel_um   = 75.0

[paths]                       # 경로 생략한 명령(timing, center, profile, heatmap)의 기본값
default_folder = 'Z:\exp\hkim\261004\images\testx1'
default_file   = 'Z:\exp\hkim\261004\images\testx1\testx1_00089.h5'
```

항목이나 파일이 없으면 기존 기본값 사용. HDF5 키·EPICS 기준시각 같은 코드 상수는 `config.py` 에 그대로.

일회성으로는 `--poni 514 850 --sdd 861.14 --energy 13 --alpha 18.9`.

### 실험 비교: 피크별 결정화 분율 X vs 온도 + T50

`experiments.toml` 에 실험마다 이미지 폴더 · 온도 로그(recipe) 의 **전체 경로**를 적고
(자동으로 찾지 않음 — 2nd 측정 · 로컬 백업 · Z: 원본이 섞여 있어서 적힌 경로만 씀):

```toml
[[exp]]
name   = "x1"
images = ['D:\USER_DATA\hkim\261004\data\IGOx1']                 # 여러 개면 시간순으로 합침
recipe = ['D:\USER_DATA\hkim\261004\data\temperature\recipe_data_20261004_x1.csv']
```


```powershell
uv run python main.py compare                  # experiments.toml 사용
uv run python main.py compare 다른설정.toml
```

- 피크마다 패널 하나, 가로축 = PV 온도, 범례 `x2  T50 397 °C`
- `[peaks]` 의 2θ 는 참고값: 실험마다 결정화 끝난 상태에서 그 근처(±`search`) 피크를 가우시안 피팅 →
  중심·반치폭(FWHM) → **중심 ± `fwhm_k`(1.5) × FWHM** 을 적분 (중심·FWHM 은 `_T50.csv` 에 기록)
- 각 프레임을 비정질(전이 전) 프로파일로 나눈 R = I/I₀ − 1 에서 적분 → 디텍터 가장자리 세기 감소·비정질 배경 상쇄
- X = (A − A₀)/(A₁ − A₀): A₀ = 처음 n_norm 프레임, A₁ = 마지막 n_norm 프레임 (기본 승온 구간 · 250–520 °C)
- T50 = X 가 0.5 를 처음 넘는 온도 (프레임 사이 선형보간)
- 이미지가 전이 이후부터만 있으면(전이 전 기준 없음) 그 실험은 T50 계산 안 하고 범례에 표시
- 없는 경로가 있으면 계산 전에 한꺼번에 알려 주고 멈춤
- 결과: `out/compare/<설정이름>.png / .html / _T50.csv / _frames.csv`
- 적분 결과는 `out/cache/` 에 캐시 (overview · heatmap 과 공용; 폴더 파일·기하·적분 옵션이 그대로면 재사용)
- 13 keV 에서 (222) ≈ 18.8°, (400) ≈ 21.7°

### 온도

```powershell
uv run python main.py merge-temp                        # raw/*.csv → 병합본 (새 로그 추가 시 재실행)
uv run python main.py temp 16:55:00                     # 특정 시각(KST)의 온도
uv run python main.py temp Z:\...\test2x1_00086.h5      # 특정 프레임의 온도
uv run python main.py temp Z:\...\test2x1               # 폴더 전체 프레임별 온도
uv run python main.py heatmap Z:\...\test2x1 --temp --at 0 10 16:55:00
```

- `--temp [CSV]` : 히트맵 옆에 온도(PV/SV) 패널. 경로 생략 시 병합본 사용
- `--at T ...`   : 숫자 = 첫 프레임 기준 상대시간(분, `--seconds` 면 초), `HH:MM:SS` = KST 시각.
  해당 프레임을 가로선으로 표시하고, 그 시점 프로파일을 온도와 함께 오른쪽 패널에 겹쳐 그림
- 결과: `out/<샘플>/<샘플>_frames.csv` (프레임별 시각·온도), `<샘플>_tth_heatmap.png/.npz`
- 로그가 끊긴 구간(가장 가까운 기록과 5초 넘게 차이)은 온도 NaN

이미지 폴더/파일은 **실제 경로**를 그대로 적음 (자동으로 찾지 않음). overview 는 `experiments.toml` 의 실험 이름(`x1`)도 받음.
경로를 생략하면 `experiments.toml` `[paths]` 의 `default_folder` / `default_file` 사용.
`uv run insitu-xrd <명령>` 또는 `python -m insitu_xrd <명령>` 도 동일.

파이썬에서 직접:

```python
from pathlib import Path
from insitu_xrd import heatmap, tth_profile, Geometry

geo = Geometry(alpha_deg=20.0)
tth_profile.run(Path(r"Z:\...\testx1_00089.h5"), geo, show=False)
from insitu_xrd.temperature import DEFAULT_LOG, TempLog
res = heatmap.run(heatmap.Config(folder=Path(r"Z:\...\test2x1"), geometry=geo,
                                 temp_log=DEFAULT_LOG, at=(10, "16:55:00")))
res.frames                                  # 프레임별 file / time_kst / time_rel / temp_pv / temp_sv
TempLog().at([1791100537.0])                # Unix 시각 → 온도
```
