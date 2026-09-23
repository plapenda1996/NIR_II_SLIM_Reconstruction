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
├── matlab/
│   ├── slim_app/                reconstruction GUI — main.mlapp (entry point), main_exported.m, utils/
│   ├── zebrafish_unet/          chamber U-Net training / inference (masks for Fig. 2b–h)
│   ├── third_party/             how to obtain the external MIMT toolbox (required, not redistributed)
│   └── MATLAB_toolboxes.txt     MATLAB release and toolboxes
├── python/
│   ├── zebrafish_heart/         chamber areas, dV/dt, strain rate; valve kinetics
│   ├── mouse_ear/               bolus kymographs, wavefront velocity, Fig. 3f traces
│   ├── lymphatics_stitching/    tile stitching (Fig. 3d, 3j), pulse GUI and statistics (Fig. 3k–m), DeepCAD-RT batch scripts
│   ├── figures_videos/          figure panels, Supplementary Videos, overlay correction
│   ├── source_data/             rescaling of the analysis exports to the calibrated pixel sizes
│   └── environments/            Python dependency lists, DeepCAD-RT provenance
├── CALIBRATION.md               pixel sizes and acquisition rates: values in the paper vs. values in the scripts
├── MANIFEST.md                  every copied file: origin, hash, figure panels, environment
└── PATHS.md                     every hard-coded data path and the dataset it refers to
```

MATLAB code lives under `matlab/`, Python code under `python/`. The stitching and lymphatic scripts share one
folder because `pulse_gui_main.py` imports `slim_recon` and `analyze_pulseresults`, and launches `stitch_tool.py`,
from its own directory (this is how they were laid out on the analysis machine).

## Pipeline → figure table

| Script | Produces | Input data | Environment |
|---|---|---|---|
| `matlab/slim_app/main.mlapp`, `matlab/slim_app/` | all reconstructed volumes | raw `.raw` + `*_geo_*.mat` + `*_psf_*.mat` | MATLAB R2024a |
| `matlab/zebrafish_unet/test_segmentation_XYZT*.m` | chamber masks, Fig. 2b–h | reconstructed zebrafish stacks | MATLAB R2024a |
| `python/zebrafish_heart/analyze_heart.py` | per-plane areas, dV/dt and perimeter strain rate (Fig. 2g,h); for the published Fig. 2c–f values see [`python/zebrafish_heart/README.md`](python/zebrafish_heart/README.md) | `segmentation_4D_RGB_*.tif` | `python/environments/requirements_analysis.txt` |
| `python/zebrafish_heart/elastix_bspline_params_resolved.txt` | the resolved B-spline parameter map actually used | — | SimpleElastix 3.0.0a1.post183 |
| `python/zebrafish_heart/valve_analysis.py` | valve opening/closing kinetics — Fig. 2i–m | reconstructed zebrafish stacks, manual leaflet annotations | `python/environments/requirements_analysis.txt` |
| `python/zebrafish_heart/leaflet_unet.py` | constants and helper functions imported by `valve_analysis.py`; its U-Net tracking branch was **not** used for the paper (`USE_UNET_VALVE_TRACKING = False`) | — | `python/environments/requirements_analysis.txt` |
| `python/mouse_ear/ROI_analysis_mouse_ear.py` | position–time kymographs, wavefront slope — Fig. 3e–g; Fig. 3f trace export together with `fig3f_roi_trace.py` | 20-volumes-s⁻¹ ear recordings | `python/environments/requirements_analysis.txt` |
| `python/mouse_ear/ROI_analysis_arrival_time.py` | arrival-time cross-check — Fig. 3e–g | same | `python/environments/requirements_analysis.txt` |
| `python/lymphatics_stitching/stitch_tool.py` + `slim_recon/` | tile registration, blending, depth colour-coded maps — Fig. 3d, 3j | multi-tile recordings | `python/environments/requirements_stitching.txt` |
| `python/lymphatics_stitching/pulse_gui_main.py` | reconstruction + pulse-analysis GUI; writes `*_pulseresults.json` | raw lymphatic recordings | `python/environments/requirements_stitching.txt` |
| `python/lymphatics_stitching/pulse_pipeline_standalone.py` | non-interactive reproduction of the pulse pipeline — Fig. 3k–m | denoised lymphatic stacks | `python/environments/requirements_analysis.txt` |
| `python/lymphatics_stitching/analyze_pulseresults.py` | pulse velocity (885 ± 149 µm s⁻¹ with the 4.0 µm/px used at run time = 941 ± 158 µm s⁻¹ after calibration to 4.25 µm/px, see `CALIBRATION.md`), transport distance, FWHM, bolus end positions — Fig. 3k–m | `*_pulseresults.json` | `python/environments/requirements_analysis.txt` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_train_pick.py` | per-recording DeepCAD-RT models | raw lymphatic recordings | `python/environments/requirements_analysis.txt` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_denoise_batch.py` | denoised stacks — Fig. 3j–m, Supp. Fig. 6d | raw + trained `.pth` | `python/environments/requirements_analysis.txt` |
| `python/figures_videos/paw_suppfig.py` | SBR/CNR and FWHM vs depth (Supp. Fig. 6), paw panels (Supp. Fig. 7) | reconstructed `record_*_rl.mat` | `python/environments/requirements_analysis.txt` |
| `python/figures_videos/Depth_cycle_video_gui.py` | Supplementary Videos 1–7 (overlays corrected afterwards with `fix_video_overlays.py`) | per-depth TIFF stacks | `python/environments/requirements_analysis.txt` |
| `python/figures_videos/fix_video_overlays.py` | scale bars / time stamps of the released Supplementary Videos (post-export correction) | exported MP4 | `python/environments/requirements_analysis.txt` + ffmpeg |
| `python/figures_videos/fill_to_dashed_outline.py` | dashed chamber outlines used in Fig. 2 panels | `segmentation_4D_RGB.tif` | `python/environments/requirements_analysis.txt` |
| `python/figures_videos/plot_fig3m.py` | Fig. 3m plot body | Fig. 3m source-data CSV | `python/environments/requirements_analysis.txt` |
| `python/source_data/rescale_source_data.py` | Source Data values of Fig. 2c–f, 3g, 3l, 3m from the analysis exports (pixel-size calibration only) | analysis CSV exports | `python/environments/requirements_analysis.txt` |

## Data

All datasets are deposited on Zenodo: **10.5281/zenodo.22184824** (CC BY 4.0, 11.5 GB).
No data files are stored in this repository.

| Zenodo file | contents | used for |
|---|---|---|
| `fig2_zebrafish_WT.raw`, `fig2_zebrafish_WEA.raw` | raw camera frames of the wild-type and *wea* zebrafish heart recordings, 600 volumes s⁻¹ | Fig. 2, Supplementary Videos 1–2 |
| `fig3d-g_mouse_ear.raw` | raw camera frames of the mouse-ear recordings (extended-field map and bolus transit, 20 volumes s⁻¹) | Fig. 3d–g, Supplementary Videos 3–4 |
| `fig3j_lymphatics_stitch_tiles.raw` | raw tiles of the lymphatic extended-field map | Fig. 3j |
| `fig3k-m_lymphatics_roi1.raw`, `fig3k-m_lymphatics_roi2.raw` | raw camera frames of the lymphatic dynamics recordings (100 and 30 volumes s⁻¹) | Fig. 3k–m, Supplementary Videos 6–7 |
| `Supple_fig7_mouse_paw.raw` | raw camera frames of the hind-paw recording | Supplementary Fig. 7, Supplementary Video 5 |
| `Source_data.zip` | source-data CSV files of the main figures, the Fig. 1 USAF/phantom data, representative reconstructed volumes, and the `*_geo_*.mat` / `*_psf_*.mat` calibration files of every dataset | all figures |

Each `.raw` file is a sequence of 640 × 512 uint16 camera frames, little-endian, with no
header; reconstruction needs the matching `*_geo_*.mat` and `*_psf_*.mat` files from
`Source_data.zip`. The trained DeepCAD-RT checkpoints and the SimpleElastix wheel are not
deposited; retrain with `python/lymphatics_stitching/deepcad_batch/deepcad_train_pick.py`
(parameters in `para.yaml`) and see the SimpleElastix note below.

The scripts were copied unmodified from the analysis machine, so they still contain the
absolute paths they were run with (`E:\Selected data\...`, `D:\NIR_SLIM\...`). Every one of
those literals is listed in [`PATHS.md`](PATHS.md) together with the Zenodo file it refers
to. To reproduce a result, download the file and either edit the literal in the script or
place the data at the same path. `stitch_tool.py` and `pulse_gui_main.py` also accept paths
through their file dialogs, so no edit is needed when they are used interactively.

## Environments

**MATLAB** — R2024a Update 3 with the Image Processing, Deep Learning, Computer Vision
and Parallel Computing toolboxes; GPU acceleration with CUDA 12.2. See
[`matlab/MATLAB_toolboxes.txt`](matlab/MATLAB_toolboxes.txt).

**Python** — dependency lists in `python/environments/`, captured with `uv pip freeze` on the machine that produced the results.

| File | Python | Used by | label in script headers / MANIFEST |
|---|---|---|---|
| `requirements_analysis.txt` | 3.14 | zebrafish heart and valve, mouse ear, lymphatic pulse statistics, denoising, figures/videos, source data | `heart_valve_py314`, `pulse_py314` |
| `requirements_stitching.txt` | 3.11 | `stitch_tool.py`, `pulse_gui_main.py` | `stitch_py311` |
| `requirements_bead.txt` | 3.11 | bead/phantom acquisition project (kept for completeness) | — |

**Windows interpreter note.** On the analysis machine the plain `python` command resolves
to Python 3.13, which does not have SimpleITK or the PyTorch build used here. All Python
analyses were run with the 3.14 interpreter explicitly:

```bat
py -3.14 <script.py>
```

**SimpleElastix.** `simpleitk-simpleelastix 3.0.0a1.post183-g61ffa` (ITK 6.0). This is a
pre-release snapshot build and may not remain installable from PyPI; the wheel is not
redistributed. The resolved elastix parameter map actually used is in this repository at
`python/zebrafish_heart/elastix_bspline_params_resolved.txt`.

**DeepCAD-RT.** Not redistributed here. The upstream commit, the two local modifications,
verification hashes and the training parameters are in
[`python/environments/DEEPCAD_VERSION.md`](python/environments/DEEPCAD_VERSION.md).

**Imports.** `slim_recon/`, `stitch_tool.py`, `pulse_gui_main.py` and `analyze_pulseresults.py` are in the same folder, and so are
`valve_analysis.py` / `leaflet_unet.py` and `ROI_analysis_mouse_ear.py` / `fig3f_roi_trace.py`; no `PYTHONPATH` setting is needed.

## How to reproduce each pipeline

Run times were not recorded except where stated.

**Reconstruction (all figures)** — MIMT must be on the MATLAB path (see `matlab/third_party/README.md`).

```matlab
>> cd matlab/slim_app
>> main                 % or open main.mlapp (the app adds utils/ relative to this folder)
% load raw / geo / psf, set depth range and nDepth, Prepare Reconstruction, Recon.
% output: record_*_rl.mat (or per-depth TIFF stacks)
```

**Zebrafish heart — chambers (Fig. 2b–h)**

```matlab
>> cd matlab/zebrafish_unet
>> test_segmentation_XYZT_fish          % trains/applies the chamber U-Net
% output: segmentation_4D_RGB_*.tif
```

```bat
py -3.14 python\zebrafish_heart\analyze_heart.py
:: output: analysis_results_elastix_WT\, analysis_results_elastix_WEA\, analysis_results_elastix_comparison\
```

**Zebrafish heart — valve (Fig. 2i–m)**

```bat
py -3.14 python\zebrafish_heart\valve_analysis.py
```

**Mouse ear bolus (Fig. 3e–g)**

```bat
py -3.14 python\mouse_ear\ROI_analysis_mouse_ear.py
:: interactive: draw the ROI along the vessel, export the kymograph, slope fit and Fig. 3f traces
py -3.14 python\mouse_ear\ROI_analysis_arrival_time.py      :: arrival-time cross-check
```

**Extended-field stitching (Fig. 3d, 3j)**

```bat
py -3.11 python\lymphatics_stitching\stitch_tool.py
:: interactive: preview tiles, register, blend, export the depth colour-coded mosaic
```

**Lymphatics (Fig. 3k–m)**

```bat
:: 1. denoise (lymphatic recordings)
py -3.14 python\lymphatics_stitching\deepcad_batch\deepcad_train_pick.py        :: one model per recording
py -3.14 python\lymphatics_stitching\deepcad_batch\deepcad_denoise_batch.py --model <n> <files>
:: 2. pulse detection -> *_pulseresults.json
py -3.11 python\lymphatics_stitching\pulse_gui_main.py
::    or non-interactively:
py -3.14 python\lymphatics_stitching\pulse_pipeline_standalone.py
:: 3. statistics
py -3.14 python\lymphatics_stitching\analyze_pulseresults.py <folder>\*_pulseresults.json
```

**Figures, videos and Source Data**

```bat
py -3.14 python\figures_videos\paw_suppfig.py --recon record_..._rl.mat --raw record_....raw --fps 10
py -3.14 python\figures_videos\Depth_cycle_video_gui.py
py -3.14 python\figures_videos\fix_video_overlays.py in.mp4 out.mp4 --label "200 µm" --um-per-px 5.616
py -3.14 python\figures_videos\plot_fig3m.py SourceData_Fig3m.csv
py -3.14 python\source_data\rescale_source_data.py --in-dir analysis_exports --out-dir source_data_final
```

## Not included in this release

The numerical simulations of Supplementary Notes 1 and 2 (Supplementary Figs. 8–10, Supplementary Table 1) and the scripts for the
calibration-pinhole, USAF-target and bead measurements (Fig. 1g, Supplementary Figs. 3–5) are available from the corresponding author on request.

## Calibration

Pixel sizes and acquisition rates used for every reported number, the values that were set in the
scripts at run time, and the conversion between the two are listed in [`CALIBRATION.md`](CALIBRATION.md).

## Acquisition-rate policy

Cardiac recordings were acquired at 600 volumes s⁻¹; lymphatic dynamics at
100 volumes s⁻¹ (Fig. 3j–l) or 30 volumes s⁻¹ (pulse-to-pulse dataset, Fig. 3m), denoised with DeepCAD-RT; ear bolus-transit recordings at
20 volumes s⁻¹ and **not** denoised; extended-field maps as stated in the Methods.
Calibration PSF stacks were acquired with 30 µm and 150 µm z-steps, while volumes were
reconstructed onto 20 µm, 60 µm and 12 µm axial grids depending on the experiment; the
reconstruction interpolates the measured PSFs onto the reconstruction grid.

## License

Code in this repository: MIT (see [`LICENSE`](LICENSE)). The MIMT toolbox used by the reconstruction
GUI is third-party software and is not redistributed here (see `matlab/third_party/README.md`). Data on Zenodo: CC BY 4.0.

## Citation

Preprint DOI 10.64898/2026.08.13.744709; dataset DOI 10.5281/zenodo.22184824
(https://doi.org/10.5281/zenodo.22184824).

## Contact

Author: <plapenda@ucla.edu>  
Corresponding author: <gaol@ucla.edu>
