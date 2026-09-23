# DeepCAD-RT provenance

Upstream: https://github.com/cabooster/DeepCAD-RT, commit
`0f53a5e09b4355d03b5ab2d9923b6caa96ddd643` (28 May 2025)

Local copy: `D:\NIR2SLIM\DeepCAD_RT_pytorch` (not redistributed here; obtain from
upstream at the commit above)

Local modifications: output-path handling only (timestamp removed from output folder /
checkpoint names in `test_collection.py` and `train_collection.py`) — network, loss and
training loop unchanged.

## Verification hashes

git blob SHA-1 of `deepcad/*.py` after LF normalisation:

```
c7f16599ff4c9f7567e3d8bea48cec651dd3d6ec  buildingblocks.py
5fc1472cd09c347a93b6ea08ffae9ce25754a707  data_process.py
5b4d47ed1c671fea2ee265387121892b04bde3bc  model_3DUnet.py
8c2f80c1c34f86f1107c38e6ffc8dad2820ae2d3  movie_display.py
3724914749ad78a6cf0e69ac9ce7935b1394260a  network.py
15a0b129a99d05cbc73be2bee6e1a40a33f1c40d  test_collection.py   (local)
fd258bc2806e99712b74176fe870c6488f90c941  train_collection.py  (local)
3c2efda4c81c7e52da72efa5b03d11d6fe09bf5c  utils.py
```

Six of the eight files match the upstream commit exactly. `test_collection.py` and
`train_collection.py` are the two locally modified files; their hashes match no upstream
commit between March 2022 and May 2025, confirming the changes are local rather than a
different upstream version.

## The local modification

```diff
--- test_collection.py    (upstream 0f53a5e)
+++ test_collection.py    (local)
@@
-        self.output_path = self.output_dir + '//' + 'DataFolderIs_' + self.datasets_name \
-                           + '_' + current_time + '_ModelFolderIs_' + self.denoise_model
+        self.output_path = self.output_dir
@@
-            output_path_name = self.output_path + '//' + pth_name.replace('.pth', '')
+            output_path_name = self.output_path
```

```diff
--- train_collection.py   (upstream 0f53a5e)
+++ train_collection.py   (local)
@@
-        pth_name = self.datasets_name + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
+        pth_name = self.datasets_name
```

Two `print()` progress lines were also added to `test_collection.py`. No other change.

Consequence to be aware of: with the timestamp removed, re-running a dataset overwrites
the previous output folder instead of creating a new one.

## Training / inference parameters

5 epochs, lr 2e-5, batch 1, Adam (0.5, 0.999), fmap 16, patch 150x150x150 (x,y,t),
gaps 90/90/22-40, overlap 0.4, scale 1.0; epoch-5 checkpoint used for inference.
One model per recording.

## Applied to

Fluorescence recordings at >= 100 volumes/s (lymphatic dynamics, Fig. 3j-m;
Supp. Fig. 6d). **Not** applied to dark-field zebrafish data or to 20-vps ear
recordings.

Environment: Python 3.14.3, PyTorch 2.11.0+cu128, RTX 4090.

Trained checkpoints (`.pth`) are not deposited. The training parameters
(`para.yaml`) are included here at `../lymphatics_stitching/deepcad_batch/para.yaml`,
and `deepcad_train_pick.py` retrains one model per recording from the raw
lymphatic recordings on Zenodo (10.5281/zenodo.22184824).
