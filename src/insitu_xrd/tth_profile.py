"""단일 이미지 → 2θ 프로파일 (중앙 스트립 / 전체 링 평균) + 피크 피팅."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import out_dir
from .geometry import Geometry
from .integrate import ProfileOptions, RadialIntegrator, strip_bounds
from .io import load_image
from .peaks import detect_peaks, fit_peak


def run(file: Path, geo: Geometry = Geometry(), strip_width: int = 10, tth_bin: float = 0.02,
        smooth: int = 7, prominence: float | None = None, min_distance: float = 0.15,
        save: bool = True, show: bool = True) -> plt.Figure:
    img = load_image(file)
    tth = geo.tth_map(img.shape)
    x0, x1 = strip_bounds(img.shape[1], strip_width)

    def profile(region):
        opt = ProfileOptions(tth_bin=tth_bin, region=region, strip_width=strip_width, min_fill=0)
        integ = RadialIntegrator.build(geo, opt, img.shape, tth)
        return integ.centers, integ(img)

    results = {
        f"strip x={x0}-{x1 - 1}": profile("strip"),
        "full image (ring avg)": profile("full"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    ax = axes[0]
    ax.imshow(img, vmin=0, vmax=np.nanpercentile(img, 99.5), aspect="auto")
    cs = ax.contour(tth, levels=10, colors="w", linewidths=0.6)
    ax.clabel(cs, fmt="%.1f°", fontsize=7)
    ax.axvspan(x0, x1, color="r", alpha=0.3)
    ax.set_title(f"{file.name}  (2θ contours)")

    for ax, (name, (x, y)) in zip(axes[1:], results.items()):
        ys, pk = detect_peaks(x, y, prominence, min_distance, smooth)
        ax.plot(x, y, lw=0.6, alpha=0.5, label="raw")
        ax.plot(x, ys, lw=1.2, label="smoothed")
        ax.plot(x[pk], ys[pk], "rx", ms=8)
        for p in pk:
            ax.annotate(f"{x[p]:.2f}", (x[p], ys[p]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8, color="r")
        ax.set_xlabel("2θ (deg)")
        ax.set_ylabel("mean intensity")
        ax.set_title(f"{name}: {len(pk)} peaks")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        print(f"\n[{name}]")
        for p in pk:
            fit = fit_peak(x, y, x[p])
            c = fit.center if np.isfinite(fit.center) else x[p]
            print(f"  2θ(max) = {x[p]:8.3f}°  2θ(fit) = {fit.center:8.4f} ± {fit.center_err:.4f}°"
                  f"  FWHM = {fit.fwhm:.3f}°  I = {ys[p]:.2f}"
                  f"  q = {geo.tth_to_q(c):.4f} Å⁻¹  d = {geo.d_spacing(c):.4f} Å")

    fig.tight_layout()
    if save:
        save_file = out_dir(file.parent.name) / f"{file.stem}_tth_profile.png"
        fig.savefig(save_file, dpi=150)
        print(f"\n그림 저장: {save_file}")
    if show:
        plt.show()
    return fig
