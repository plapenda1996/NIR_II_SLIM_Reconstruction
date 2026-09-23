# zebrafish_heart

- `analyze_heart.py` – SimpleElastix-based analysis: per-plane chamber areas, dV/dt and perimeter strain rate
  (Fig. 2c–h). CONFIG holds the values reported in the paper (600 volumes s⁻¹, 5.616 µm/px, 20-µm planes;
  see `../../CALIBRATION.md`).
- `valve_analysis.py` – atrioventricular-valve analysis (kymograph, leaflet annotation, opening events). It imports shared
  constants and helpers from `leaflet_unet.py`; the U-Net tracking branch of that module was not used for the paper.
- `elastix_bspline_params_resolved.txt` – the resolved B-spline parameter map used for the registration.
- The chamber U-Net (MATLAB) is in `../../matlab/zebrafish_unet/`.
