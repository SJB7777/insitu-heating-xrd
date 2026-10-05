"""디텍터 기하 (캘리브레이션 값) 와 2θ / q / d 변환."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import config as C

FArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Geometry:
    """기본값은 config.py 의 캘리브레이션 블록 (내부 단위: 거리 m)."""
    pixel: float = C.PIXEL_UM * 1e-6     # 픽셀 크기 [m]
    sdd: float = C.SDD_MM * 1e-3         # 시료 → PONI 거리 [m]
    xc: float = C.PONI_PX[0]             # PONI x [px]
    yc: float = C.PONI_PX[1]             # PONI y [px]
    alpha_deg: float = C.ALPHA_DEG       # 디텍터 암의 2θ 모터 값 [deg]
    energy_kev: float = C.ENERGY_KEV

    @property
    def wavelength(self) -> float:  # [Å]
        return 12.398420 / self.energy_kev

    def tth_map(self, shape: tuple[int, int]) -> FArray:
        """각 픽셀의 true 2θ [deg].
        디텍터 좌표 u(가로), v(위쪽) 를 실험실 좌표로 옮긴 뒤 직접빔(z)과의 각도를 계산.
          P = SDD·n + u·ex + v·ev,  n=(0,sinα,cosα), ev=(0,cosα,-sinα)
          2θ = arccos( Pz / |P| ) = arccos( (SDD·cosα − v·sinα) / sqrt(SDD²+u²+v²) )
        α=0 이면 2θ = atan( sqrt(u²+v²) / SDD ) 와 같음."""
        y, x = np.mgrid[: shape[0], : shape[1]]
        u = (x - self.xc) * self.pixel
        v = (self.yc - y) * self.pixel          # 이미지 위쪽이 +v (2θ 증가 방향)
        a = np.deg2rad(self.alpha_deg)
        cos_tth = (self.sdd * np.cos(a) - v * np.sin(a)) / np.sqrt(self.sdd**2 + u**2 + v**2)
        return np.rad2deg(np.arccos(np.clip(cos_tth, -1, 1)))

    def tth_to_q(self, tth: FArray | float) -> FArray:
        return 4 * np.pi * np.sin(np.deg2rad(np.asarray(tth) / 2)) / self.wavelength

    def q_to_tth(self, q: FArray | float) -> FArray:
        s = np.clip(np.asarray(q) * self.wavelength / (4 * np.pi), -1, 1)
        return 2 * np.rad2deg(np.arcsin(s))

    def d_spacing(self, tth: float) -> float:
        return self.wavelength / (2 * np.sin(np.deg2rad(tth / 2)))
