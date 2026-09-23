# MANIFEST

Every code file in this repository. The scripts are byte-identical to the versions that
produced the results, except for a two-line header (title and environment) at the top of
each file, after any shebang / coding line, and the content changes listed at the end of
this file. `sha256 (repo)` is the file as committed here; `sha256 (source)` is the first
32 hex digits of the original file before the header was added.

## Files

| repo path | size (B) | figure panels / numbers | environment |
|---|---|---|---|
| `python/zebrafish_heart/analyze_heart.py` | 47,358 | Fig. 2c-h | `heart_valve_py314` |
| `python/zebrafish_heart/elastix_bspline_params_resolved.txt` | 4,229 | Fig. 2c-h (resolved elastix parameter map) | `SimpleElastix 3.0.0a1.post183` |
| `python/zebrafish_heart/leaflet_unet.py` | 193,994 | Fig. 2i-m | `heart_valve_py314` |
| `python/zebrafish_heart/valve_analysis.py` | 355,562 | Fig. 2i-m | `heart_valve_py314` |
| `matlab/zebrafish_unet/test_segmentation_XYZT.m` | 19,833 | Fig. 2b-h masks | `MATLAB R2024a` |
| `matlab/zebrafish_unet/test_segmentation_XYZT_fish.m` | 40,216 | Fig. 2b-h masks | `MATLAB R2024a` |
| `python/mouse_ear/ROI_analysis_arrival_time.py` | 124,236 | Fig. 3e-g (arrival-time cross-check) | `heart_valve_py314` |
| `python/mouse_ear/ROI_analysis_mouse_ear.py` | 432,691 | Fig. 3e-g | `heart_valve_py314` |
| `python/mouse_ear/fig3f_roi_trace.py` | 40,105 | Fig. 3f ROI traces and Source Data export | `heart_valve_py314` |
| `python/lymphatics_stitching/analyze_pulseresults.py` | 55,737 | Fig. 3k-m; 885 +/- 149 um/s as run at 4.0 um/px (941 +/- 158 after calibration, see CALIBRATION.md), transport distance, FWHM | `pulse_py314` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_denoise_batch.py` | 22,189 | Fig. 3j-m, Supp. Fig. 6d | `heart_valve_py314` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_train_pick.py` | 6,213 | Fig. 3j-m, Supp. Fig. 6d | `heart_valve_py314` |
| `python/lymphatics_stitching/deepcad_batch/para.yaml` | 555 | representative training/inference parameter file | `heart_valve_py314` |
| `python/lymphatics_stitching/pulse_gui_main.py` | 1,646,146 | produces *_pulseresults.json consumed by analyze_pulseresults.py | `stitch_py311` |
| `python/lymphatics_stitching/pulse_pipeline_standalone.py` | 75,819 | Fig. 3k-m | `pulse_py314` |
| `python/lymphatics_stitching/slim_recon/__init__.py` | 158 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/blend_export.py` | 39,611 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/data_io.py` | 12,887 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/denoise.py` | 10,576 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/enhance.py` | 10,654 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/layered_export.py` | 19,101 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/pipeline.py` | 16,442 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/preprocessing.py` | 14,993 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/reconstruction.py` | 9,378 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/viewer.py` | 20,386 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/slim_recon/visualization.py` | 8,049 | Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py) | `stitch_py311` |
| `python/lymphatics_stitching/stitch_tool.py` | 71,943 | Fig. 3d, 3j | `stitch_py311` |
| `python/figures_videos/Depth_cycle_video_gui.py` | 101,707 | Supp. Videos 1-6 | `heart_valve_py314` |
| `python/figures_videos/fill_to_dashed_outline.py` | 8,234 | Fig. 2 panels (dashed chamber outlines) | `heart_valve_py314` |
| `python/figures_videos/fix_video_overlays.py` | 5,586 | post-export correction of the Supplementary Video overlays | `heart_valve_py314` |
| `python/figures_videos/paw_suppfig.py` | 176,490 | Supp. Fig. 6 (SBR/CNR, FWHM vs z), Supp. Fig. 7 | `heart_valve_py314` |
| `python/figures_videos/plot_fig3m.py` | 2,331 | Fig. 3m plot body | `heart_valve_py314` |
| `python/source_data/rescale_source_data.py` | 4,638 | pixel-size rescaling that yields the Source Data of Fig. 2c–f, 3g, 3l, 3m | `heart_valve_py314` |
| `python/environments/requirements_analysis.txt` | 1,368 | Python dependency list | — |
| `python/environments/requirements_bead.txt` | 2,592 | Python dependency list | — |
| `python/environments/requirements_stitching.txt` | 773 | Python dependency list | — |
| `matlab/slim_app/main.mlapp` | 368,429 | reconstruction GUI — entry point (all reconstructed volumes) | MATLAB R2024a |
| `matlab/slim_app/main_exported.m` | 234,751 | exported source of `main.mlapp` | MATLAB R2024a |
| `matlab/slim_app/utils/Image4DViewer.m` | 40,339 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/ImageStackViewer.m` | 16,992 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/addScaleBar.m` | 2,369 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/backward_model.m` | 1,424 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/blend_images.m` | 341 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/cal_capture.mlapp` | 46,392 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/cal_dark.mlapp` | 42,720 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/cal_dark_3d.mlapp` | 23,499 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/cal_psf_estimation.mlapp` | 151,473 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/cal_registration.mlapp` | 74,175 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/coordinate_transform.m` | 12,859 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/correctHotPixels.m` | 5,262 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/createCircularMask.m` | 1,736 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/cropPSF.m` | 1,221 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/crop_image.m` | 1,279 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/darkChannelRemoveBackground.m` | 38,819 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/deconvPSF.m` | 794 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/deconvlucy_gpu.m` | 7,388 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/depthIntensityMap.m` | 1,446 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/fittingPSF.m` | 3,855 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/formatTime.m` | 528 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/forward_model.m` | 1,083 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/generate_roi_masks.m` | 1,650 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/getRawPSF.m` | 2,246 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/get_dark_channel_gpu.m` | 870 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/init_camera.m` | 2,819 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/init_stage.m` | 2,702 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/interactive_stitch_viewer.m` | 9,626 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/interp_transformations.m` | 1,192 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/interpolate_PSFs.m` | 1,134 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/load_hdf5_cam.m` | 949 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/load_raw_block.m` | 2,021 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/load_raw_cam.m` | 1,272 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/optimizeCentroidTransform.m` | 1,461 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/psf2otf_gpu.m` | 4,925 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/psf_lowrank_fit.m` | 23,982 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/rectCentroid.m` | 460 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/restoreCroppedImage.m` | 1,277 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/save_animation.mlapp` | 41,681 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/shift_and_sum.m` | 5,180 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/slim_app/utils/warpPSFs.m` | 1,448 | reconstruction GUI helper (`utils/`) | MATLAB R2024a |
| `matlab/MATLAB_toolboxes.txt` | 437 | MATLAB release and toolboxes | MATLAB R2024a |

## Hashes

| repo path | sha256 (repo) | sha256 (source) |
|---|---|---|
| `python/zebrafish_heart/analyze_heart.py` | `ac56ea08ce7a9204d8398a2aae942e297931749765344bb2742c6d467ffbe55c` | `4373359cfad6f524e76e0cecf0d3de5e…` |
| `python/zebrafish_heart/elastix_bspline_params_resolved.txt` | `ae1f09c381bf03b154faf743574130a7a812ee996a0692e8b2a0cde68eeecb20` | `f8baea0846e3bbc2161c7ef0d4a73513…` |
| `python/zebrafish_heart/leaflet_unet.py` | `2b89b21f4a3bf8368510e2b53e00b27d97b2747574b58c7dc652da0ed16a3294` | `65b61d773720102493dffc7bb9a2a665…` |
| `python/zebrafish_heart/valve_analysis.py` | `3124d906f236960892c8e4844d4b962da21e76af6a1d0de31b82372b9f443b26` | `2550cabf7ccfd33720005d1bed6f0667…` |
| `matlab/zebrafish_unet/test_segmentation_XYZT.m` | `0e9b2543cabc4b91f892a68a27baef39902eac2f388bff42fb137bbac80b381b` | `e5b28023ebfb14cf630968b462533887…` |
| `matlab/zebrafish_unet/test_segmentation_XYZT_fish.m` | `fc767036b988eb98059ef22df711b878b06ce09de72309cc0414f1a49e817354` | `fd6a60dcf61814c3de429c7173a8b6bf…` |
| `python/mouse_ear/ROI_analysis_arrival_time.py` | `4c177726a949d53e7af3c1aea28ab03145d1dbde9a13a4fe3e7b5adb40e1ab9b` | `63adfc3d4bd96690f7a806e83348ef07…` |
| `python/mouse_ear/ROI_analysis_mouse_ear.py` | `6aaf7972124fa5d3e9debebd969198196f3ab4e8f2adf97bb67dfb8887d8ef45` | `997ad3f7b6bd5c217b2e5903cc668271…` |
| `python/mouse_ear/fig3f_roi_trace.py` | `7b332c905112d7b41303d012950975b251ae149e7b8fa8180092158f139fd352` | — |
| `python/lymphatics_stitching/analyze_pulseresults.py` | `b1c8929dcc7b6c3263038671007b579131169bd6fd8583f08eb3f9196d6ea442` | `9b74725cec43cd47f93aa8d3e8d2618c…` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_denoise_batch.py` | `cb25da499ff5d536a66ebd996a0ed9cbb851bef7433126d050af149693e01407` | `b0d1ff1590fc04af3e700f3a3a57ad7c…` |
| `python/lymphatics_stitching/deepcad_batch/deepcad_train_pick.py` | `1404b4b8ac1814c23b469428a6b4a4d1d9a45d723d0dceb838eebaaa35fac109` | `9700b81e1d81e18e9b8bfe66113d1530…` |
| `python/lymphatics_stitching/deepcad_batch/para.yaml` | `54e9fa522a6927173c77dd7d9e9efa6bb1fe82b1cd0bfc0c244ca8f28706fca7` | `a69e322c08d05863af01f6349a77cd08…` |
| `python/lymphatics_stitching/pulse_gui_main.py` | `73a6fa85f4e734c8f6f581e0ab56008a9370b5695a25404df8a9dc3f3190326b` | `92efc64428687ee8b58b07a4438fa993…` |
| `python/lymphatics_stitching/pulse_pipeline_standalone.py` | `a71700e0c29024c5984894a5c8af30b1f839baecb5c7d4d36fbca614863a7af0` | `80b7080f54f7a269c47058f998d9c8ed…` |
| `python/lymphatics_stitching/slim_recon/__init__.py` | `382133c610f78e84214cc2c4ee877bcd1c0852e7662097de3946aa48be7c1e2c` | `e3b0c44298fc1c149afbf4c8996fb924…` |
| `python/lymphatics_stitching/slim_recon/blend_export.py` | `fdb8c5fd0341b123a0b978dc06b15ce21b672c23f619757659b786762500f39d` | `34a993a2ac8eaf8e7690ccf1e50ec968…` |
| `python/lymphatics_stitching/slim_recon/data_io.py` | `ecf32102066da19a79aa9167229a5c5fb9b2e6a612f6c12ea61e2add40b1bcd1` | `06eb97944d5c99a79ef09be100e412f0…` |
| `python/lymphatics_stitching/slim_recon/denoise.py` | `fb6f1a53f6ca42fd494eefe36c78e7b2b6a1d49bd05a0d731b38d4d4b1eafe10` | `01ef3c937aca447a0b9c3560184c50a3…` |
| `python/lymphatics_stitching/slim_recon/enhance.py` | `851b6387fad31c7a83626ad90f886d10e0c0e1af93a377b038a9eaaf4e3b981c` | `7debf1873d1c93eaa07aabeb5c0e9771…` |
| `python/lymphatics_stitching/slim_recon/layered_export.py` | `5b0f4e6c3715efc1b01b115dbf7d228147b60feec1348de1d3c62d0951d2a00e` | `a75b6ee61169d87fa24d9449f0f6902d…` |
| `python/lymphatics_stitching/slim_recon/pipeline.py` | `4944547528900cb3884764313a806b291d8422c466a5cb683dcd3abc73a0e294` | `93063aeddebe7d52ac8b561a62ae67e5…` |
| `python/lymphatics_stitching/slim_recon/preprocessing.py` | `a49283337a1349b0ccd30154da5761170c1337ba0aab5d99cabdeb7ee3cf08a2` | `0cef97fc8737ded91a949f2532a98831…` |
| `python/lymphatics_stitching/slim_recon/reconstruction.py` | `0d23a23e04704c54da11f40eb756eadeee285567b22a0454884ca7bbfbfa1153` | `3644c841cc262110505befcb4ac3cf74…` |
| `python/lymphatics_stitching/slim_recon/viewer.py` | `8491847d2df43a1f1d670562bc357a6270e92d929b8b108745789fe69352be43` | `d2e18b9bcd2cd567ae8dca13f517ee07…` |
| `python/lymphatics_stitching/slim_recon/visualization.py` | `b0b1f94126e6b9af660df6d11778e9baaea10fc27eee8abf1548b18a98fc984f` | `9cf423dafdbe902512ad5c0b1906b804…` |
| `python/lymphatics_stitching/stitch_tool.py` | `e1004745dd920e06bd2d2e695f559ffc8645280ff7a6ba0c711f336ae7489675` | `0e71fbde8c6dd279ef5bb9957031a37b…` |
| `python/figures_videos/Depth_cycle_video_gui.py` | `83f7f8889fb1728ce32faf816d14d122add1d4d00094aee5c8f7ab578526bfde` | `4dfde8742bbe954e7bc01488d5b48ced…` |
| `python/figures_videos/fill_to_dashed_outline.py` | `c2f2c6a466693c95bce904cae868042018e3189b576d32c23629fa6e7f7b5b69` | `c11ed4c72614a39f4a97fedf7b584381…` |
| `python/figures_videos/fix_video_overlays.py` | `ec969c2e65a9dbf2a14255dc7b0a4f6b83361f5cc0e5cf82ba045f9f8ed25e74` | — |
| `python/figures_videos/paw_suppfig.py` | `e14a29423c5b4e67ffbb395ad6282556eabf8c81fab2727aebdef6b263ed1dad` | `1bdf6f7b769ef89865808359a180bec3…` |
| `python/figures_videos/plot_fig3m.py` | `559d0b5a2d59d7a8206bbae9899d048c7505b273d362ee55504d12f6f5b0f282` | — |
| `python/source_data/rescale_source_data.py` | `c164523ebb65b66d53fd757b09870c0930d81c406dec3de74bb35da187ea38d5` | — |
| `python/environments/requirements_analysis.txt` | `855f5384967245ba26a5f7d771f7d6648143bff9c13180881c4375a014542c7a` | — |
| `python/environments/requirements_bead.txt` | `7f25f3d18ff074bffeafe35fbe93fd968f53b56c4b82b9ca2a7bce2cbb1980b3` | — |
| `python/environments/requirements_stitching.txt` | `36f313fce40928c3aceaa62f9f458eaef1790646343fe4f0d6b95c9b893ac938` | — |
| `matlab/slim_app/main.mlapp` | `32dd67a0cea51b9bf0aceb59512c4c70951a09a0590457923ad8080c2ee50683` | — |
| `matlab/slim_app/main_exported.m` | `61b395583cb000b3c597fb34e55422284948884d102b6d0eed4d51594499c873` | — |
| `matlab/slim_app/utils/Image4DViewer.m` | `d2486abb646a07404eaeb946ee79fea4bcb2577d23721324de35688c5e4c2c19` | — |
| `matlab/slim_app/utils/ImageStackViewer.m` | `4d9ff07c7b30ed59e77d4673756d1e173529e327817f004f00417ccde4dc4ba2` | — |
| `matlab/slim_app/utils/addScaleBar.m` | `229372970db8b1d032f372316fc1d056a71de6ed6bacb13e5f073f4e17c0b0cc` | — |
| `matlab/slim_app/utils/backward_model.m` | `e39b08aae8df2617e1f044719898a90c402f7db4c3a87caf2ea96fad29efc09f` | — |
| `matlab/slim_app/utils/blend_images.m` | `0ed7bb531a641c6d22e68d6b6f393b65f24f81d4130b0a0740722ed57db84f7a` | — |
| `matlab/slim_app/utils/cal_capture.mlapp` | `350896c4fb6f5d35edaa31c7c8466faf0833208c4c7898bae47c7d405155690c` | — |
| `matlab/slim_app/utils/cal_dark.mlapp` | `0c7b8f2bc75470a272e4937a0848ba7e2c32dd8cc54816aa4d85d2e579a5b276` | — |
| `matlab/slim_app/utils/cal_dark_3d.mlapp` | `07e6e5bc9f98c4a9170e7656b800886c164284a3fc3610608a7b52edff2cc313` | — |
| `matlab/slim_app/utils/cal_psf_estimation.mlapp` | `5aee18ad91b6b6299be8a3052c460f6fc4ba318a6412c8c8af6bc103bf069b70` | — |
| `matlab/slim_app/utils/cal_registration.mlapp` | `3c8b2d718e96f994c76f4e9f0d74edab35009a3d612b67ff8f87dcc9c758b229` | — |
| `matlab/slim_app/utils/coordinate_transform.m` | `879c9f0dd78dd7839c2a74bfb250095588fa4a1bca1f1badaaec12a96113aec7` | — |
| `matlab/slim_app/utils/correctHotPixels.m` | `63e8dcc694f1de943ebc22fa448589bc294d921c9c951dd25b62235d60515ca2` | — |
| `matlab/slim_app/utils/createCircularMask.m` | `7027604c3575dac515a9a20ec7a914fc20589a91400060054be3d222216e05d4` | — |
| `matlab/slim_app/utils/cropPSF.m` | `3286a60d0233093813289e9b2125ea557ddb8ac16011c298df3123db913fd801` | — |
| `matlab/slim_app/utils/crop_image.m` | `277f7c1da905a971f827cf0f26aee59caa6089114e3e7be02ee8093cf040ef73` | — |
| `matlab/slim_app/utils/darkChannelRemoveBackground.m` | `7518f39b90cbdb53d99983dc8cbf95d77372e1c0ccd7d896e102fcf1b6c9ecf5` | — |
| `matlab/slim_app/utils/deconvPSF.m` | `94fad98a58a7f53dcb0dd2b598802d90505e9a98d7536d4fa82f50912b2b7f00` | — |
| `matlab/slim_app/utils/deconvlucy_gpu.m` | `a92586397825da70cf2f43b7a3827a4044a9afeea3ab6268a1baccc81676f951` | — |
| `matlab/slim_app/utils/depthIntensityMap.m` | `6c83875aee09e36c8d6868610f6362ec88e2212c9e5c1f94c197d6ed262d344f` | — |
| `matlab/slim_app/utils/fittingPSF.m` | `484cf24f592a5779f28012c1eb6e46b42859106facac507d592391a4b7fd1d71` | — |
| `matlab/slim_app/utils/formatTime.m` | `a6fdd294a64d8c3e812f650789bb848a92c38539a7bc6b1c997f4459ab685b5d` | — |
| `matlab/slim_app/utils/forward_model.m` | `0eb041d474ab190f7400da9dd220df99379bab113dc9bab2b10cda83931e5831` | — |
| `matlab/slim_app/utils/generate_roi_masks.m` | `0ea5c9ff325565662333639f7b70937a96e7fb1e32aa7ecd8382cbccfdf85258` | — |
| `matlab/slim_app/utils/getRawPSF.m` | `a06263d49f9f6ad9eb8e681d9e29e9a6f57f94b76ef961976caa9122f414f29d` | — |
| `matlab/slim_app/utils/get_dark_channel_gpu.m` | `3d9d00b2653414a6b6e361ed7f5daba69ffef7adeb0a5bca8b91a44609337853` | — |
| `matlab/slim_app/utils/init_camera.m` | `04effa60af9401db118242e807865d0b71aa0ac0a944fbcd35b8755a791ee35f` | — |
| `matlab/slim_app/utils/init_stage.m` | `596c16cd839a9e51eb3d2306c8312d7df5e3b6537e2526befacbe7fa0a6d6dcd` | — |
| `matlab/slim_app/utils/interactive_stitch_viewer.m` | `5194137ceeaed38eb67f9025db2509e42cb65f78c79355e0f682fa95e708d8b4` | — |
| `matlab/slim_app/utils/interp_transformations.m` | `f8b21e0634b4de08e3b683489348f5b2b4c25d1ba44d5044536752d637708e7d` | — |
| `matlab/slim_app/utils/interpolate_PSFs.m` | `eeca3190bb517f29bc8685cc5d0763a59ccfd2a22d3723957505183962f732f0` | — |
| `matlab/slim_app/utils/load_hdf5_cam.m` | `936378e0451068e971858d41840e74f6a4054ea5a8ee38a0bc0497ee320e490d` | — |
| `matlab/slim_app/utils/load_raw_block.m` | `48570c4a075652b88ebc84a954a60c0b533855b64ae68c2af540db5fe8c698dc` | — |
| `matlab/slim_app/utils/load_raw_cam.m` | `a1eabb26f9bbe451150c797a30259573944fe34166e1b3b9d96658e4cd025936` | — |
| `matlab/slim_app/utils/optimizeCentroidTransform.m` | `23002c9cf68c8fa83fd891256933050d1a17a6a9177f0aa19e7f3382cbe51772` | — |
| `matlab/slim_app/utils/psf2otf_gpu.m` | `98830ab0fc0f29116015a7a6700ca2dc3133ece687b59987b95234115d1d851e` | — |
| `matlab/slim_app/utils/psf_lowrank_fit.m` | `7b2797d80ebcabf6f6e1d82fb717ce497079136fdb4786170b3582337ebc4ba8` | — |
| `matlab/slim_app/utils/rectCentroid.m` | `3c54e628bf8ae786d6ee99856c2e63a80e81b5cb848ddc7605f47ac7adf333e1` | — |
| `matlab/slim_app/utils/restoreCroppedImage.m` | `a9272ef57daedc245d3241592d9f64c06a9d77e9efdeba3e81f98ed82fe5833d` | — |
| `matlab/slim_app/utils/save_animation.mlapp` | `89ebf5363a43bff71412b928bdc36a604b0988272aa94d80630246955c118f54` | — |
| `matlab/slim_app/utils/shift_and_sum.m` | `54a041666a7116afdc16367eb1eae71f6900d0dea1639640f038ed74f4be3e1a` | — |
| `matlab/slim_app/utils/warpPSFs.m` | `c1813a3a234d20396f8ad5cd6d021d8f1da2b26af9ab4ec88767a822704522dc` | — |
| `matlab/MATLAB_toolboxes.txt` | `4d07d4b82b1ddf2ef856b23378c87501bf29da8afe6240ac8208f0e50165a42c` | — |

## Deliberately not copied

- DeepCAD-RT source — third-party; see `python/environments/DEEPCAD_VERSION.md`.
- All data files (`.tif`, `.mat`, `.raw`, `.mp4`) — on Zenodo (10.5281/zenodo.22184824); `.pth` checkpoints and `.npz` intermediates — not deposited.
- Code fragments that are not runnable scripts (a parametric cartoon heart model, plotting-style snippets, a depth scale-bar export snippet).

> The MIMT toolbox is not vendored; see `matlab/third_party/README.md`.

## Changes relative to the files as run

**Layout.** MATLAB code is under `matlab/`, Python code under `python/`. `stitch_tool.py`, `slim_recon/`, `pulse_gui_main.py`,
`pulse_pipeline_standalone.py` and `analyze_pulseresults.py` are in one folder (`python/lymphatics_stitching/`), as on the
analysis machine: `pulse_gui_main.py` imports `slim_recon` and `analyze_pulseresults` and launches `stitch_tool.py` from its own
directory.

**Content.** Files are byte-identical to the versions that were run (apart from the header) except:

| file | change |
|---|---|
| `python/zebrafish_heart/analyze_heart.py` | CONFIG `fps` 400 → 600, `pixel_size` 15.0 → 5.616, `z_spacing` 30.0 → 20.0 (values reported in the paper) |
| `python/figures_videos/Depth_cycle_video_gui.py` | `MAG_PRESETS`: high 11.0 → 5.616 µm/px; low 30.3 → 17.0 µm/px, dz 100 → 60 µm, bar 1000 → 600 µm (256-px frames) |
| `python/mouse_ear/ROI_analysis_mouse_ear.py`, `ROI_analysis_arrival_time.py` | GUI defaults 11.0 µm/px, 30 fps → 17.0 µm/px, 20 volumes s⁻¹ |
| `python/environments/requirements_*.txt` | one comment line with the Python version added at the top |

Removed: `*.bak`, `__pycache__/`, a duplicate root copy of `main.mlapp` (the GUI must be started from `matlab/slim_app/`, where `utils/` is)
and the vendored MIMT toolbox (see `matlab/third_party/README.md`).

Added for the release (not part of the original analysis folders):

| file | purpose |
|---|---|
| `python/mouse_ear/fig3f_roi_trace.py` | Fig. 3f ROI traces and Source Data export |
| `python/figures_videos/fix_video_overlays.py` | post-export correction of the Supplementary Video overlays |
| `python/figures_videos/plot_fig3m.py` | Fig. 3m plot body |
| `python/source_data/rescale_source_data.py` | pixel-size rescaling that yields the Source Data of Fig. 2c–f, 3g, 3l, 3m |
