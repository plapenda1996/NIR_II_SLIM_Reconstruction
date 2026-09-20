# zebrafish_heart

- `analyze_heart.py` – SimpleElastix-based analysis: per-plane chamber areas, dV/dt and perimeter strain rate.
- `valve_analysis.py` – atrioventricular-valve analysis (kymograph, leaflet annotation, opening events). It imports shared
  constants and helpers from `leaflet_unet.py`; the U-Net tracking branch of that module was not used for the paper.
- The chamber U-Net (MATLAB) is in `../../matlab/zebrafish_unet/`.

**Published Fig. 2c–f.** The plotted "area" is the lumen-area output of the earlier script `compare_heart_analysis.py`
(ventricular minus atrial cross-sectional area per plane), run at 600 volumes s⁻¹, 20-µm plane spacing and 10 µm/px, and rescaled
to 5.616 µm/px (× 0.3154; see `../../CALIBRATION.md`). `analyze_heart.py` computes the areas from the ventricular mask instead and
therefore does not reproduce those curves exactly. *Add `compare_heart_analysis.py` to this folder before release.*
