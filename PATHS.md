# Hard-coded data paths

The scripts were copied unmodified, so the absolute paths they were run with are
still in the source. Nothing is read from these paths automatically — every one
must be edited (or the data placed at the same location) before a script will run.

Dataset files themselves are on Zenodo: **10.5281/zenodo.22184824**. Calibration files
(`*_geo_*.mat`, `*_psf_*.mat`) are inside `Source_data.zip`.

| script | literal path in the source | what it is | Zenodo file |
|---|---|---|---|
| `python/figures_videos/Depth_cycle_video_gui.py` | `E:\Selected data\250902_fish\6` | reconstructed zebrafish heart stacks (wild type) | reconstruct `fig2_zebrafish_WT.raw` with `matlab/slim_app` |
| `python/figures_videos/paw_suppfig.py` | `C:/Windows/Fonts/arial.ttf` | figure font (Windows system font) | - |
| `python/lymphatics_stitching/deepcad_batch/deepcad_denoise_batch.py` | `C:\data\fov1.tif` | placeholder in the usage example of the docstring | - |
| `python/lymphatics_stitching/deepcad_batch/deepcad_denoise_batch.py` | `C:\data\fov2.tif` | placeholder in the usage example of the docstring | - |
| `python/lymphatics_stitching/deepcad_batch/deepcad_train_pick.py` | `E:\260424_mouse_paw\deepcad_input` | DeepCAD training input stacks | `fig3k-m_lymphatics_roi1.raw`, `fig3k-m_lymphatics_roi2.raw` |
| `python/lymphatics_stitching/pulse_gui_main.py` | `C:/Windows/Fonts/arial.ttf` | figure font (Windows system font) | - |
| `python/lymphatics_stitching/pulse_gui_main.py` | `D:\NIR_SLIM2\demotion` | earlier analysis working folder | - |
| `python/lymphatics_stitching/pulse_gui_main.py` | `D:\NIR_SLIM\DeepCAD_RT_pytorch` | local DeepCAD-RT checkout (see DEEPCAD_VERSION.md) | - |
| `python/lymphatics_stitching/pulse_gui_main.py` | `D:\NIR_SLIM\demotion` | earlier analysis working folder | - |
| `python/lymphatics_stitching/pulse_gui_main.py` | `E:\Selected data\260130_mouse_lym\Dynamic\denoised\denoised_truncated` | DeepCAD-denoised lymphatic dynamics | not deposited — regenerate with `deepcad_batch/` from `fig3k-m_lymphatics_roi*.raw` |
| `python/lymphatics_stitching/pulse_gui_main.py` | `E:\Selected data\260209_mouse_lym\260209_geo_1.mat` | geometric calibration | `Source_data.zip` (`*_geo_*.mat`, low-magnification objective) |
| `python/lymphatics_stitching/pulse_gui_main.py` | `E:\Selected data\260209_mouse_lym\260209_psf_1.mat` | measured PSF stack | `Source_data.zip` (`*_psf_*.mat`, low-magnification objective) |
| `python/lymphatics_stitching/pulse_gui_main.py` | `E:\Selected data\260209_mouse_lym\Stitch\record_09022026_142008_20hz.raw` | raw lymphatic recording, 20 vps | `fig3j_lymphatics_stitch_tiles.raw` |
| `python/lymphatics_stitching/pulse_gui_main.py` | `E:\\260325_stitch_process_dy_ver2\\analyze_pulseresults.py\n` | printed hint in a help string, not a read path | - |
| `python/lymphatics_stitching/slim_recon/denoise.py` | `D:\NIR_SLIM\DeepCAD_RT_pytorch` | local DeepCAD-RT checkout (see DEEPCAD_VERSION.md) | - |
| `python/lymphatics_stitching/stitch_tool.py` | `C:/Windows/Fonts/arial.ttf` | figure font (Windows system font) | - |
| `python/lymphatics_stitching/stitch_tool.py` | `E:\Selected data\260209_mouse_lym\260209_geo_1.mat` | geometric calibration | `Source_data.zip` (`*_geo_*.mat`, low-magnification objective) |
| `python/lymphatics_stitching/stitch_tool.py` | `E:\Selected data\260209_mouse_lym\260209_psf_1.mat` | measured PSF stack | `Source_data.zip` (`*_psf_*.mat`, low-magnification objective) |
| `python/lymphatics_stitching/stitch_tool.py` | `E:\Selected data\260209_mouse_lym\Stitch\record_09022026_142008_20hz.raw` | raw lymphatic recording, 20 vps | `fig3j_lymphatics_stitch_tiles.raw` |
| `matlab/zebrafish_unet/test_segmentation_XYZT.m` | `E:\Selected data\250922_mutant_fish\28` | reconstructed zebrafish heart stacks (mutant) | reconstruct `fig2_zebrafish_WEA.raw` with `matlab/slim_app` |
| `matlab/zebrafish_unet/test_segmentation_XYZT_fish.m` | `E:\250902_fish\6` | reconstructed zebrafish heart stacks (wild type) | reconstruct `fig2_zebrafish_WT.raw` with `matlab/slim_app` |

## How to repoint a script

1. Download the file from Zenodo (and unpack `Source_data.zip`) anywhere.
2. Open the script and replace the literal above with your own path. The paths
   are plain string literals at module level or in a `CONFIG` dict near the top;
   none is constructed at run time.
3. For `stitch_tool.py` and `pulse_gui_main.py` the paths are also settable in the
   GUI file dialogs, so no edit is required if you open the files interactively.
4. `C:/Windows/Fonts/arial.ttf` is only used for figure text; on a non-Windows
   system point it at any TrueType font.
