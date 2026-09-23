# Third-party MATLAB dependency: MIMT

The reconstruction GUI calls three functions of the **MATLAB Image Manipulation Toolbox (MIMT)** by DGM:
`imblend`, `mergedown` and `colorpict` (these in turn need MIMT's support functions and colour-space lookup tables).

MIMT is third-party software and is **not redistributed in this repository**. Get it from
- MATLAB File Exchange 53786 "Image Manipulation Toolbox" – https://www.mathworks.com/matlabcentral/fileexchange/53786 (source: https://github.com/291ce4321ac/MIMT), or
- the smaller File Exchange 52513 "Image blending functions" (MIMT blend tools) – https://www.mathworks.com/matlabcentral/fileexchange/52513

and add it, with sub-folders, to the MATLAB path before starting `matlab/slim_app/main.mlapp`:

```matlab
addpath(genpath('matlab/third_party/MIMT'))   % if unpacked here; any other location works as well
savepath                                       % optional
```

   If a copy is committed to this repository, keep the toolbox's own licence file next to it. `.gitignore` excludes `*.mat`
   everywhere except `matlab/third_party/MIMT/`, so the toolbox's lookup tables would be tracked.
