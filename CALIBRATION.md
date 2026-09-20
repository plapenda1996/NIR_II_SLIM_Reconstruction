# Calibration: values in the paper vs. values in the scripts

Lateral scale of the reconstruction grid (pinhole-array pitch calibration, Supplementary Fig. 5; the
high-magnification value was checked against USAF group-4 elements, Fig. 1e–g):

| objective | 512-px grid | 256-px grid (2× binned; videos, segmentation and ear stacks) | 1024-px grid (2× upsampled; lymphatic analysis) |
|---|---|---|---|
| high magnification (5.2×) | 2.808 µm/px (frame 1.44 mm) | 5.616 µm/px | – |
| low magnification (1.9×)  | 8.5 µm/px (frame 4.35 mm)  | 17.0 µm/px  | 4.25 µm/px |

For any other grid: µm/px = frame width / number of pixels across the full frame (cropping does not change it).

| analysis | value in the script / at run time | value used in the paper | how the reported numbers were obtained |
|---|---|---|---|
| zebrafish chambers (Fig. 2c–f) | 10 µm/px, 600 volumes s⁻¹, 20-µm planes | 5.616 µm/px | areas and dV/dt × (5.616/10)² = 0.3154 (`python/source_data/rescale_source_data.py`) |
| zebrafish strain rate (Fig. 2g,h) | 600 volumes s⁻¹ | same | independent of the pixel size |
| zebrafish valve (`leaflet_unet.py`: `UM_PER_PIXEL = 10`) | 10 µm/px, 600 volumes s⁻¹ | 5.616 µm/px | only normalised signals and times are reported; lengths from this script must be × 0.5616 |
| mouse-ear bolus speed (Fig. 3g) | 40 µm per analysis-canvas unit, 20 volumes s⁻¹ | 17.0 µm/px on the 256-px grid | canvas units → grid px fitted from the drawn centerlines (ROI1 0.672, ROI2 0.754 px/unit); speeds × 0.2856 / 0.3204 |
| lymphatic bolus speed (Fig. 3l,m) | 4.0 µm/px; 100 (Fig. 3l) and 30 (Fig. 3m) volumes s⁻¹ | 4.25 µm/px | lengths and speeds × 1.0625 (885 ± 149 → 941 ± 158 µm s⁻¹) |
| Supplementary Videos | bars drawn for 128-px frames (11 / 30.3 µm/px) | 5.616 / 17.0 µm/px (256-px frames) | bars redrawn with `python/figures_videos/fix_video_overlays.py`; the presets in `Depth_cycle_video_gui.py` are now the 256-px values |
| mouse paw (Supplementary Fig. 7) | 2.93 µm/px passed with `--px-um` (script fallback 4.0) | 2.808 µm/px | scale bars differ by 4 % |

The GUI defaults of `python/mouse_ear/ROI_analysis_*.py` (17.0 µm/px, 20 volumes s⁻¹) and the CONFIG of
`python/zebrafish_heart/analyze_heart.py` (600 volumes s⁻¹, 5.616 µm/px, 20-µm planes) were set to the values
above on 2026-09-20; the earlier values are recorded in `MANIFEST.md`.
