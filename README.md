# NIR-II SLIM — reconstruction and analysis code

Squeezed light-field imaging microscopy (SLIM) for volumetric imaging in the second
near-infrared window (1,000–1,700 nm). A single 640 × 512 InGaAs sensor records a light
field that has been rotated by a Dove prism and anamorphically compressed, so that many
angular views share the detector without tiling it. One camera frame is deconvolved into
a volume, giving continuous volumetric imaging at up to 600 volumes s⁻¹. This repository
holds the MATLAB reconstruction code and the Python and MATLAB analysis code that
produced the figures and the numbers reported in the paper.

## Repository map

```
NIR_II_SLIM_Reconstruction/
├── main.mlapp                  reconstruction GUI (App Designer) — entry point
├── matlab/slim_app/            reconstruction implementation and utilities
├── matlab/third_party/MIMT     third-party MATLAB image-manipulation toolbox
├── environments/               dependency lists, MATLAB toolboxes, DeepCAD-RT provenance
├── zebrafish_heart/            chamber registration/strain, valve segmentation and kinetics,
│                               MATLAB U-Net chamber segmentation
├── mouse_ear/                  bolus kymographs and wavefront velocity
├── stitching/                  extended-field tile registration, blending, depth colour-coding
├── lymphatics/                 pulse detection and statistics, denoising batch scripts
├── characterization/           (see note — bead/USAF scripts are not in this repository)
├── simulations/                (see note — Supplementary Note scripts are not in this repository)
├── figures_videos/             figure panels and Supplementary Videos
├── MANIFEST.md                 every copied file: origin, hash, figure panels, environment
└── PATHS.md                    every hard-coded data path and the dataset it refers to
```

The MATLAB reconstruction code was already in the repository and its layout is unchanged;
the analysis folders were added alongside it.

## Pipeline → figure table

| Script | Produces | Input data | Environment |
|---|---|---|---|
| `main.mlapp`, `matlab/slim_app/` | all reconstructed volumes | raw `.raw` + `*_geo_*.mat` + `*_psf_*.mat` | MATLAB R2024a |
| `zebrafish_heart/matlab_unet/test_segmentation_XYZT*.m` | chamber masks, Fig. 2b–h | reconstructed zebrafish stacks | MATLAB R2024a |
| `zebrafish_heart/analyze_heart_ver6.py` | chamber areas per plane, dV/dt, strain rate — Fig. 2c–h | `segmentation_4D_RGB_*.tif` | `requirements_heart_valve_py314.txt` |
| `zebrafish_heart/elastix_bspline_params_resolved.txt` | the resolved B-spline parameter map actually used | — | SimpleElastix 3.0.0a1.post183 |
| `zebrafish_heart/leaflet_unet.py` | AV valve leaflet masks — Fig. 2i–m | reconstructed zebrafish stacks | `requirements_heart_valve_py314.txt` |
| `zebrafish_heart/valve_analysis.py` | valve opening/closing kinetics — Fig. 2i–m | leaflet masks | `requirements_heart_valve_py314.txt` |
| `mouse_ear/ROI_analysis_mouse_ear.py` | position–time kymographs, wavefront slope — Fig. 3e–g | 20-vps ear recordings | `requirements_heart_valve_py314.txt` |
| `mouse_ear/ROI_analysis_2.py` | arrival-time cross-check — Fig. 3e–g | same | `requirements_heart_valve_py314.txt` |
| `stitching/stitch_tool.py` + `stitching/slim_recon/` | tile registration, blending, depth colour-coded maps — Fig. 3d, 3j | multi-tile recordings | `requirements_stitch_py311.txt` |
| `lymphatics/pulse_gui_main.py` | reconstruction + pulse-analysis GUI; writes `*_pulseresults.json` | raw lymphatic recordings | `requirements_stitch_py311.txt` |
| `lymphatics/pulse_pipeline_standalone.py` | non-interactive reproduction of the pulse pipeline — Fig. 3k–m | denoised lymphatic stacks | `requirements_pulse_py314.txt` |
| `lymphatics/analyze_pulseresults.py` | pulse velocity (885 ± 149 µm s⁻¹), transport distance, FWHM, bolus end positions — Fig. 3k–m | `*_pulseresults.json` | `requirements_pulse_py314.txt` |
| `lymphatics/deepcad_batch/deepcad_train_pick.py` | per-recording DeepCAD models | raw ≥ 100-vps recordings | `requirements_heart_valve_py314.txt` |
| `lymphatics/deepcad_batch/deepcad_denoise_batch.py` | denoised stacks — Fig. 3j–m, Supp. Fig. 6d | raw + trained `.pth` | `requirements_heart_valve_py314.txt` |
| `figures_videos/paw_suppfig.py` | SBR/CNR and FWHM vs depth (Supp. Fig. 6), paw panels (Supp. Fig. 7) | reconstructed `record_*_rl.mat` | `requirements_heart_valve_py314.txt` |
| `figures_videos/Depth_cycle_video_gui.py` | Supplementary Videos 1–6 | per-depth TIFF stacks | `requirements_heart_valve_py314.txt` |
| `figures_videos/fill_to_dashed_outline.py` | dashed chamber outlines used in Fig. 2 panels | `segmentation_4D_RGB.tif` | `requirements_heart_valve_py314.txt` |

## Data

All datasets, trained DeepCAD-RT checkpoints and the SimpleElastix wheel are deposited on
Zenodo: **10.5281/zenodo.22184824**.

No data files are stored in this repository. The scripts were copied unmodified from the
analysis machine, so they still contain the absolute paths they were run with
(`E:\Selected data\...`, `D:\NIR_SLIM\...`). Every one of those literals is listed in
[`PATHS.md`](PATHS.md) together with the Zenodo dataset it refers to. To reproduce a
result, download the dataset and either edit the literal in the script or place the data
at the same path. `stitch_tool.py` and `pulse_gui_main.py` also accept paths through their
file dialogs, so no edit is needed when they are used interactively.

## Environments

**MATLAB** — R2024a Update 3 with the Image Processing, Deep Learning, Computer Vision
and Parallel Computing toolboxes; GPU acceleration with CUDA 12.2. See
[`environments/MATLAB_toolboxes.txt`](environments/MATLAB_toolboxes.txt).

**Python** — four environments, listed in `environments/requirements_*.txt`. They were
captured with `uv pip freeze` from the machine that produced the results.

| File | Used by |
|---|---|
| `requirements_heart_valve_py314.txt` | zebrafish heart and valve, mouse ear, denoising, figures/videos |
| `requirements_pulse_py314.txt` | lymphatic pulse analysis (same environment as above) |
| `requirements_stitch_py311.txt` | stitching, `pulse_gui_main.py` |
| `requirements_bead_py311.txt` | bead/phantom acquisition project (kept for completeness) |

**Windows interpreter note.** On the analysis machine the plain `python` command resolves
to Python 3.13, which does not have SimpleITK or the PyTorch build used here. All Python
analyses were run with the 3.14 interpreter explicitly:

```bat
py -3.14 <script.py>
:: = C:\Users\qicui\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

**SimpleElastix.** `simpleitk-simpleelastix 3.0.0a1.post183-g61ffa` (ITK 6.0). This is a
pre-release snapshot build and may not remain installable from PyPI; the wheel and the
resolved elastix parameter file are deposited on Zenodo. The resolved parameter map is
also in this repository at `zebrafish_heart/elastix_bspline_params_resolved.txt`.

**DeepCAD-RT.** Not redistributed here. The upstream commit, the two local modifications,
verification hashes and the training parameters are in
[`environments/DEEPCAD_VERSION.md`](environments/DEEPCAD_VERSION.md).

**PYTHONPATH.** `stitching/slim_recon/` is imported by both `stitching/stitch_tool.py` and
`lymphatics/pulse_gui_main.py`. Run those two from the `stitching/` folder, or add it to
`PYTHONPATH`.

## How to reproduce each pipeline

Run times were not recorded except where stated.

**Reconstruction (all figures)**

```matlab
>> cd matlab/slim_app
>> main                 % or open main.mlapp
% load raw / geo / psf, set depth range and nDepth, Prepare Reconstruction, Recon.
% output: record_*_rl.mat (or per-depth TIFF stacks)
```

**Zebrafish heart — chambers (Fig. 2b–h)**

```matlab
>> cd zebrafish_heart/matlab_unet
>> test_segmentation_XYZT_fish_251110    % trains/applies the chamber U-Net
% output: segmentation_4D_RGB_*.tif
```

```bat
py -3.14 zebrafish_heart\analyze_heart_ver6.py
:: output: analysis_results_elastix_WT\, analysis_results_elastix_WEA\,
::         analysis_results_elastix_comparison\ (strain heat maps, 3D surfaces,
::         per-plane areas, dV/dt)
```

**Zebrafish heart — valve (Fig. 2i–m)**

```bat
py -3.14 zebrafish_heart\leaflet_unet.py
py -3.14 zebrafish_heart\valve_analysis.py
```

**Mouse ear bolus (Fig. 3e–g)**

```bat
py -3.14 mouse_ear\ROI_analysis_mouse_ear.py
:: interactive: draw the ROI along the vessel, export the kymograph and slope fit
py -3.14 mouse_ear\ROI_analysis_2.py          :: arrival-time cross-check
```

**Extended-field stitching (Fig. 3d, 3j)**

```bat
cd stitching
py -3.11 stitch_tool.py
:: interactive: preview tiles, register, blend, export the depth colour-coded mosaic
```

**Lymphatics (Fig. 3k–m)**

```bat
:: 1. denoise (>= 100 vps recordings only)
py -3.14 lymphatics\deepcad_batch\deepcad_train_pick.py        :: one model per recording
py -3.14 lymphatics\deepcad_batch\deepcad_denoise_batch.py --model <name> <files>
:: 2. pulse detection -> *_pulseresults.json
cd stitching && py -3.11 ..\lymphatics\pulse_gui_main.py
::    or non-interactively:
py -3.14 lymphatics\pulse_pipeline_standalone.py
:: 3. statistics
py -3.14 lymphatics\analyze_pulseresults.py <folder>\*_pulseresults.json
:: output: results_final\ (aggregate.json, per_pulse.csv, bolus_positions.csv,
::         pulse_fig1..3 .png/.svg, summary.txt)
```

**Figures and videos**

```bat
py -3.14 figures_videos\paw_suppfig.py --recon record_..._rl.mat --raw record_....raw --fps 10
:: output: <stem>_suppfig\ (panels SVG/PDF/PNG, FWHM and SBR/CNR source-data CSVs, pptx)
py -3.14 figures_videos\Depth_cycle_video_gui.py
:: interactive: pick folder, range, contrast, overlays, export MP4
py -3.14 figures_videos\fill_to_dashed_outline.py
```

## Acquisition-rate policy

Cardiac recordings were acquired at 600 volumes s⁻¹; lymphatic dynamics at
100 volumes s⁻¹ and denoised with DeepCAD-RT; ear bolus-transit recordings at
20 volumes s⁻¹ and **not** denoised; extended-field maps as stated in the Methods.
Calibration PSF stacks were acquired with 30 µm and 150 µm z-steps, while volumes were
reconstructed onto 20 µm, 60 µm and 12 µm axial grids depending on the experiment; the
reconstruction interpolates the measured PSFs onto the reconstruction grid.

## License

Code in this repository: MIT (see [`LICENSE`](LICENSE)), except
`matlab/third_party/MIMT`, which carries its own licence. Data on Zenodo: CC BY 4.0.

## Citation

See [`CITATION.cff`](CITATION.cff). Preprint DOI 10.64898/2026.08.13.744709; dataset DOI
10.5281/zenodo.22184824.

## Contact

Corresponding author: gaol@ucla.edu
