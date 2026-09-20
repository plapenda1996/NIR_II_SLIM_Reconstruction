# MANIFEST

Every file copied into this repository from the analysis machine.
Files are byte-identical to the versions that produced the results, except for a
three-line provenance header added at the top of each (after any shebang /
coding line). `sha256 (repo)` is the file as committed here; `sha256 (source)`
is the original before the header was added.

| repo path | original path | size (B) | figure panels / numbers | environment |
|---|---|---|---|---|
| `python/zebrafish_heart/analyze_heart.py` | `D:\NIR2SLIM\NIR-II-SLIM-code\SimpleElastix\Analyze_heart_ver6.py` | 47,003 | Fig. 2c-h | `heart_valve_py314` |
| `python/zebrafish_heart/elastix_bspline_params_resolved.txt` | `D:\NIR2SLIM\NIR-II-SLIM-code\SimpleElastix\TransformParameters.0.txt` | 4,081 | Fig. 2c-h (resolved elastix parameter map) | `SimpleElastix 3.0.0a1.post183` |
| `python/zebrafish_heart/leaflet_unet.py` | `E:\Selected data\250902_fish\leaflet_unet.py` | 193,887 | Fig. 2i-m | `heart_valve_py314` |
| `matlab/zebrafish_unet/test_segmentation_XYZT.m` | `D:\NIR_SLIM2\test_segmentation_XYZT.m` | 19,713 | Fig. 2b-h masks | `MATLAB R2024a` |
| `matlab/zebrafish_unet/test_segmentation_XYZT_fish.m` | `D:\NIR_SLIM2\test_segmentation_XYZT_fish_251110.m` | 40,096 | Fig. 2b-h masks | `MATLAB R2024a` |
| `python/zebrafish_heart/valve_analysis.py` | `E:\Selected data\250902_fish\valve_analysis.py` | 355,465 | Fig. 2i-m | `heart_valve_py314` |
| `python/mouse_ear/ROI_analysis_arrival_time.py` | `D:\NIR2SLIM\demotion\ROI_analysis_2.py` | 124,029 | Fig. 3e-g (arrival-time cross-check) | `heart_valve_py314` |
| `python/mouse_ear/ROI_analysis_mouse_ear.py` | `D:\NIR2SLIM\demotion\ROI_analysis_mouse_ear.py` | 413,210 | Fig. 3e-g | `heart_valve_py314` |
| `python/lymphatics_stitching/slim_recon/__init__.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\__init__.py` | 0 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/blend_export.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\blend_export.py` | 39,453 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/data_io.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\data_io.py` | 12,729 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/denoise.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\denoise.py` | 10,418 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/enhance.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\enhance.py` | 10,496 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/layered_export.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\layered_export.py` | 18,943 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/pipeline.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\pipeline.py` | 16,284 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/preprocessing.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\preprocessing.py` | 14,835 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/reconstruction.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\reconstruction.py` | 9,220 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/viewer.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\viewer.py` | 20,228 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/visualization.py` | `E:\260325_stitch_process_dy_ver2\slim_recon\visualization.py` | 7,891 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/stitch_tool.py` | `E:\260325_stitch_process_dy_ver2\stitch_tool.py` | 71,849 | Fig. 3d, 3j | `stitch_py311` |
| `python/lymphatics_stitching/analyze_pulseresults.py` | `E:\260325_stitch_process_dy_ver2\analyze_pulseresults_revised.py` | 55,602 | Fig. 3k-m; 885 +/- 149 um/s as run at 4.0 um/px (941 +/- 158 after calibration, see CALIBRATION.md), transport distance, FWHM | `pulse_py314` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_denoise_batch.py` | `D:\NIR2SLIM\DeepCAD_RT_pytorch_ver2\Deepcad denoise batch.py` | 22,064 | Fig. 3j-m, Supp. Fig. 6d | `heart_valve_py314` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_train_pick.py` | `D:\NIR2SLIM\DeepCAD_RT_pytorch_ver2\deepcad_train_pick.py` | 6,091 | Fig. 3j-m, Supp. Fig. 6d | `heart_valve_py314` |
| `python/lymphatics_stitching/deepcad_batch/para.yaml` | `D:\NIR2SLIM\DeepCAD_RT_pytorch\pth\record_03102025_170648_40hz\para.yaml` | 405 | representative training/inference parameter file | `heart_valve_py314` |
| `python/lymphatics_stitching/pulse_gui_main.py` | `E:\260325_stitch_process_dy_ver2\main.py` | 1,645,990 | produces *_pulseresults.json consumed by analyze_pulseresults.py | `stitch_py311` |
| `python/lymphatics_stitching/pulse_pipeline_standalone.py` | `E:\260325_stitch_process_dy_ver2\pulse_pipeline_standalone.py` | 75,729 | Fig. 3k-m | `pulse_py314` |
| `python/figures_videos/Depth_cycle_video_gui.py` | `E:\Selected data\250902_fish\Depth cycle video gui.py.bak` | 101,401 | Supp. Videos 1-6 | `heart_valve_py314` |
| `python/figures_videos/fill_to_dashed_outline.py` | `E:\Selected data\250902_fish\fill_to_dashed_outline.py` | 8,105 | Fig. 2 panels (dashed chamber outlines) | `heart_valve_py314` |
| `python/figures_videos/paw_suppfig.py` | `E:\Selected data\260310_mouse_paw\paw_suppfig_revised.py` | 176,335 | Supp. Fig. 6 (SBR/CNR, FWHM vs z), Supp. Fig. 7 | `heart_valve_py314` |

## Hashes

| repo path | sha256 (repo) | sha256 (source) |
|---|---|---|
| `python/zebrafish_heart/analyze_heart.py` | `d28d2dacee151308a513c8e9279b4d53…` | `4373359cfad6f524e76e0cecf0d3de5e…` |
| `python/zebrafish_heart/elastix_bspline_params_resolved.txt` | `a149530e276b9c21bf2f5b60feddc673…` | `f8baea0846e3bbc2161c7ef0d4a73513…` |
| `python/zebrafish_heart/leaflet_unet.py` | `970f7b0b6f9160b9dccbb200a9eeae97…` | `65b61d773720102493dffc7bb9a2a665…` |
| `matlab/zebrafish_unet/test_segmentation_XYZT.m` | `14b85b5b27d5b2fb083f6c1f6599193c…` | `e5b28023ebfb14cf630968b462533887…` |
| `matlab/zebrafish_unet/test_segmentation_XYZT_fish.m` | `9f6533c2a037d33b5062473fc8ffb11b…` | `fd6a60dcf61814c3de429c7173a8b6bf…` |
| `python/zebrafish_heart/valve_analysis.py` | `36d6e91902f131e6630bfd1c1c49b178…` | `2550cabf7ccfd33720005d1bed6f0667…` |
| `python/mouse_ear/ROI_analysis_arrival_time.py` | `b0cd0e1580fb3f78c14fd63418eaf8ab…` | `63adfc3d4bd96690f7a806e83348ef07…` |
| `python/mouse_ear/ROI_analysis_mouse_ear.py` | `ea8869257dab17c86ba43935d7dbdb00…` | `997ad3f7b6bd5c217b2e5903cc668271…` |
| `python/lymphatics_stitching/slim_recon/__init__.py` | `7e122f673a2f94e5dfb16de3d9ff8f7c…` | `e3b0c44298fc1c149afbf4c8996fb924…` |
| `python/lymphatics_stitching/slim_recon/blend_export.py` | `ce15d789ed0ff23da480937a3d134200…` | `34a993a2ac8eaf8e7690ccf1e50ec968…` |
| `python/lymphatics_stitching/slim_recon/data_io.py` | `8101e424859e11086946a34dfa4df30a…` | `06eb97944d5c99a79ef09be100e412f0…` |
| `python/lymphatics_stitching/slim_recon/denoise.py` | `ef24d5e9f7e71f3ff2f07b5729121bfe…` | `01ef3c937aca447a0b9c3560184c50a3…` |
| `python/lymphatics_stitching/slim_recon/enhance.py` | `8b1a2ab816970699b7933f327cee5fad…` | `7debf1873d1c93eaa07aabeb5c0e9771…` |
| `python/lymphatics_stitching/slim_recon/layered_export.py` | `29bde035883875871927b70011502f5b…` | `a75b6ee61169d87fa24d9449f0f6902d…` |
| `python/lymphatics_stitching/slim_recon/pipeline.py` | `b701ee547954524d71b1b2da9518e109…` | `93063aeddebe7d52ac8b561a62ae67e5…` |
| `python/lymphatics_stitching/slim_recon/preprocessing.py` | `dde903537fb01fe496347ab8ec02235d…` | `0cef97fc8737ded91a949f2532a98831…` |
| `python/lymphatics_stitching/slim_recon/reconstruction.py` | `a924b6e4aa720d920cda020ed50bc8f3…` | `3644c841cc262110505befcb4ac3cf74…` |
| `python/lymphatics_stitching/slim_recon/viewer.py` | `bc7cc18f2fd25a59e12a551f05069ea2…` | `d2e18b9bcd2cd567ae8dca13f517ee07…` |
| `python/lymphatics_stitching/slim_recon/visualization.py` | `aca1aa92f462bda28f5d4b9478c7d57f…` | `9cf423dafdbe902512ad5c0b1906b804…` |
| `python/lymphatics_stitching/stitch_tool.py` | `59ef248996ee2352997902a0e9c4eb05…` | `0e71fbde8c6dd279ef5bb9957031a37b…` |
| `python/lymphatics_stitching/analyze_pulseresults.py` | `a9ce7123fa704659bf8e23029ed4520d…` | `9b74725cec43cd47f93aa8d3e8d2618c…` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_denoise_batch.py` | `8331e2ed06c659a46c9093199962f9a3…` | `b0d1ff1590fc04af3e700f3a3a57ad7c…` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_train_pick.py` | `db16e05ae6a36603542c17e68aa57c45…` | `9700b81e1d81e18e9b8bfe66113d1530…` |
| `python/lymphatics_stitching/deepcad_batch/para.yaml` | `a709a3245d0f4173282c44b9b4ae09aa…` | `a69e322c08d05863af01f6349a77cd08…` |
| `python/lymphatics_stitching/pulse_gui_main.py` | `60e9b94f5820c504a1528dca5abefc11…` | `92efc64428687ee8b58b07a4438fa993…` |
| `python/lymphatics_stitching/pulse_pipeline_standalone.py` | `a564881f0bd583f18d601bb2f263c5c7…` | `80b7080f54f7a269c47058f998d9c8ed…` |
| `python/figures_videos/Depth_cycle_video_gui.py` | `69e09b06b923439849d8c3643f3ddec7…` | `4dfde8742bbe954e7bc01488d5b48ced…` |
| `python/figures_videos/fill_to_dashed_outline.py` | `dab0004411abb0510b869d722d790d24…` | `c11ed4c72614a39f4a97fedf7b584381…` |
| `python/figures_videos/paw_suppfig.py` | `2c8bd9dbb9975431baa0328e0e0f9862…` | `1bdf6f7b769ef89865808359a180bec3…` |

## Version choices where more than one candidate existed

These were ambiguous on disk. The choice made is recorded here so it can be
reversed without re-running the search.

| repo file | chosen source | rejected candidates | reason |
|---|---|---|---|
| `python/figures_videos/Depth_cycle_video_gui.py` | `Depth cycle video gui.py.bak` (2,380 lines, md5 `2d6aedb1…`) | the live `Depth cycle video gui.py` (2,605 lines, md5 `0ef5960c…`) | the live file was extended on 2026-09-17 with label-editing / upscale-export features **after** the Supplementary Videos were exported; the `.bak` is the version that produced them |
| `python/zebrafish_heart/valve_analysis.py` | `valve_analysis.py` (6,997 lines, 2026-06-24 16:36) | `valve_analysis_0624.py`, `- Copy`, `- Copy (2)`, `- Copy (3)`, `(1)` | newest and largest of six variants with identical docstring |
| `matlab/zebrafish_unet/` | **both** `test_segmentation_XYZT_fish_251110.m` (1,027 lines, 2025-11-10) and `test_segmentation_XYZT.m` (504 lines, 2026-08-20) | `test_segmentation.m`, `_curve.m`, `_dual.m`, `Test_segmentation_XYZT_v2.m` | could not be resolved from the files alone: the `_fish_251110` variant is fish-specific and largest, the `_XYZT` variant is newest. Both are included; **the corresponding author should delete whichever was not used** |
| `python/mouse_ear/ROI_analysis_mouse_ear.py` | `D:\NIR2SLIM\demotion\` copy | `D:\NIR_SLIM2\demotion\` copy | the two are sha256-identical; the NIR2SLIM copy was taken |
| `python/lymphatics_stitching/analyze_pulseresults.py` | `analyze_pulseresults_revised.py` (939 lines) | `analyze_pulseresults.py` (721 lines) | the revised version is the one that emits bolus end positions |
| `python/lymphatics_stitching/pulse_gui_main.py` | `main.py` (28,490 lines, 2026-09-01) | `main _0901.py` (sha-identical duplicate), `main _0720.py`, `main_0624/0705/0710.py`, `main - Copy*.py` | newest; matches the ~28k-line GUI named in the Methods |
| `python/figures_videos/paw_suppfig.py` | `E:\Selected data\260310_mouse_paw\paw_suppfig_revised.py` | `paw_suppfig.py` (older) and older downloaded copies | newest in the project folder rather than a browser-download copy |

## Deliberately not copied

- DeepCAD-RT source — third-party; see `python/environments/DEEPCAD_VERSION.md`.
- All data files (`.tif`, `.mat`, `.raw`, `.mp4`, `.pth`, `.npz`) — on Zenodo.
- Superseded script versions listed in the table above.
- `PulseFlowAnalyzer.py` / `PulseFlowAnalyzer_v2.py` (`D:\NIR2SLIM\demotion\`, 2026-04) — an earlier concatenated-stack pulse/flow tool superseded by `pulse_pipeline_standalone.py`; **confirm before release** if any reported number came from it.
- `heart3d.py` (parametric cartoon heart model), `compare_coverage.py`, `Nature style plots.py`, `depth scalebar export.py` (code fragments, not runnable scripts).

> The MIMT toolbox that was previously vendored under `matlab/third_party/MIMT/` has been removed; see `matlab/third_party/README.md`.



## Changes made for the public release (2026-09-20)

**Layout.** MATLAB code is under `matlab/`, Python code under `python/`. Version and date suffixes were removed from the file names
(the original names are in the "original path" column above):

| file in this repository | name when copied |
|---|---|
| `python/zebrafish_heart/analyze_heart.py` | `analyze_heart_ver6.py` |
| `python/mouse_ear/ROI_analysis_arrival_time.py` | `ROI_analysis_2.py` |
| `python/mouse_ear/ROI_analysis_mouse_ear.py` | `ROI_analysis_mouse_ear_ver2.py` (successor of `ROI_analysis_mouse_ear.py`: same code plus the Fig. 3f export and line-ROI save/load functions; no existing line changed) |
| `matlab/zebrafish_unet/test_segmentation_XYZT_fish.m` | `test_segmentation_XYZT_fish_251110.m` |
| `python/environments/requirements_analysis.txt` | `requirements_heart_valve_py314.txt` (identical `requirements_pulse_py314.txt` dropped) |
| `python/environments/requirements_stitching.txt`, `requirements_bead.txt` | `requirements_stitch_py311.txt`, `requirements_bead_py311.txt` |

`stitch_tool.py`, `slim_recon/`, `pulse_gui_main.py`, `pulse_pipeline_standalone.py` and `analyze_pulseresults.py` are again in one
folder (`python/lymphatics_stitching/`), as on the analysis machine: `pulse_gui_main.py` imports `slim_recon` and
`analyze_pulseresults` and launches `stitch_tool.py` from its own directory.

**Content.** Files are byte-identical to the versions that were run (apart from the provenance header) except:

| file | change |
|---|---|
| `python/zebrafish_heart/analyze_heart.py` | CONFIG `fps` 400 → 600, `pixel_size` 15.0 → 5.616, `z_spacing` 30.0 → 20.0 (values reported in the paper); comments refer to `compare_heart_analysis` without its version suffix |
| `python/figures_videos/Depth_cycle_video_gui.py` | `MAG_PRESETS`: high 11.0 → 5.616 µm/px; low 30.3 → 17.0 µm/px, dz 100 → 60 µm, bar 1000 → 600 µm (256-px frames) |
| `python/mouse_ear/ROI_analysis_mouse_ear.py`, `ROI_analysis_arrival_time.py` | GUI defaults 11.0 µm/px, 30 fps → 17.0 µm/px, 20 volumes s⁻¹ |
| `python/environments/requirements_*.txt` | one comment line with the Python version added at the top |

Removed: `*.bak`, `__pycache__/`, the duplicate root `main.mlapp` (the GUI must be started from `matlab/slim_app/`, where `utils/` is),
the vendored MIMT toolbox (see `matlab/third_party/README.md`) and the empty `characterization/` and `simulations/` placeholders.

Added:

| file | purpose | sha256 (repo) |
|---|---|---|
| `python/mouse_ear/fig3f_roi_trace.py` | Fig. 3f ROI traces and Source Data export | `7b332c905112d7b41303d012950975b2…` |
| `python/figures_videos/fix_video_overlays.py` | post-export correction of the Supplementary Video overlays | `ec969c2e65a9dbf2a14255dc7b0a4f6b…` |
| `python/figures_videos/plot_fig3m.py` | Fig. 3m plot body | `559d0b5a2d59d7a8206bbae9899d048c…` |
| `python/source_data/rescale_source_data.py` | pixel-size rescaling that yields the Source Data of Fig. 2c–f, 3g, 3l, 3m | `c164523ebb65b66d53fd757b09870c09…` |
