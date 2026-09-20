# Third-party MATLAB dependency

The reconstruction GUI calls three functions of the **MATLAB Image Manipulation Toolbox (MIMT)** by DGM
(MATLAB File Exchange): `imblend`, `mergedown` and `colorpict`. MIMT is not redistributed in this repository.
Install it from the File Exchange / Add-On Explorer, or place a copy in `matlab/third_party/MIMT/` and add it to
the MATLAB path before starting `matlab/slim_app/main.mlapp`.
