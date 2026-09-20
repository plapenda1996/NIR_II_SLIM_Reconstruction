# zebrafish_heart

- `matlab_unet/` – chamber U-Net training/inference (MATLAB). Both scripts are kept because the run that produced the
  deposited masks could not be identified from the files alone (see `MANIFEST.md`).
- `analyze_heart_ver6.py` – SimpleElastix-based analysis: per-plane chamber areas, dV/dt and perimeter strain rate.
- `leaflet_unet.py`, `valve_analysis.py` – atrioventricular-valve analysis. The U-Net tracking branch in
  `leaflet_unet.py` was not used for the paper; `valve_analysis.py` imports shared constants and helpers from it.

**Published Fig. 2c–f.** The plotted "area" is the lumen-area output of the earlier script
`compare_heart_analysis_ver5.py` (ventricular minus atrial cross-sectional area per plane), run at 600 volumes s⁻¹,
20-µm plane spacing and 10 µm/px, and rescaled to 5.616 µm/px (× 0.3154; see `../CALIBRATION.md`).
`analyze_heart_ver6.py` computes the areas from the ventricular mask instead and therefore does not reproduce those
curves exactly. *Add `compare_heart_analysis_ver5.py` to this folder before release.*
