# characterization — not in this repository

The Methods describe bead / phantom characterisation that this repository cannot
currently support:

| Claimed | Status |
|---|---|
| Bead PSF FWHM vs depth and vs field position (Supp. Figs. 3-5) | **script not found** |
| USAF / phantom line profiles, lateral resolution (Fig. 1g) | **script not found** |
| SBR / CNR vs depth (Supp. Fig. 6) | produced by `../figures_videos/paw_suppfig.py` (`sbr_cnr_vs_depth`, `measure_fwhm_vs_depth`) |

The bead project folder on the analysis machine
(`E:\260220_bead_imaging`) contains the acquired data and calibration files
(`dense_bead*.tif`, `*_geo_*.mat`, `*_psf_*.mat`, `cal*.mat`) plus a six-line
placeholder `main.py` and a napari viewer, but no measurement code. Either the
analysis was done interactively (Fiji / napari / MATLAB command line) or the script
lives on another machine.

Before release, either add the script here or amend the Code Availability statement.
