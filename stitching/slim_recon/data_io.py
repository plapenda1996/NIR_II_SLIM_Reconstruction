# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
# copied from E:\260325_stitch_process_dy_ver2\slim_recon\data_io.py on 2026-09-17
"""Data loading for NIR SLIM light-field microscopy."""

import numpy as np
import scipy.io as sio
from pathlib import Path
import subprocess
import shutil
import warnings
import textwrap


# Raw camera parameters
RAW_WIDTH = 640
RAW_HEIGHT = 512
BYTES_PER_PIXEL = 2  # int16
RAW_OFFSET = 400


def load_raw(filepath: str, start_frame: int = 0, num_frames: int | None = None) -> np.ndarray:
    """Load raw int16 video file.

    Args:
        filepath: Path to .raw file.
        start_frame: First frame index (0-based).
        num_frames: Number of frames to load. None = all remaining.

    Returns:
        Array of shape (H, W, N) as float64, with offset correction.
    """
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    pixels_per_frame = RAW_WIDTH * RAW_HEIGHT
    bytes_per_frame = pixels_per_frame * BYTES_PER_PIXEL
    total_frames = file_size // bytes_per_frame

    if num_frames is None:
        num_frames = total_frames - start_frame
    num_frames = min(num_frames, total_frames - start_frame)

    offset_bytes = start_frame * bytes_per_frame
    total_pixels = pixels_per_frame * num_frames

    with open(filepath, 'rb') as f:
        f.seek(offset_bytes)
        data = np.fromfile(f, dtype=np.int16, count=total_pixels)

    # MATLAB: reshape(im, [width, height, N]) in column-major (Fortran) order,
    # then permute([2,1,3]) to get (height, width, N)
    data = data.reshape((RAW_WIDTH, RAW_HEIGHT, num_frames), order='F')
    data = data.transpose(1, 0, 2)  # permute([2,1,3]) -> (H, W, N)

    # Offset correction and clamp
    im = data.astype(np.float64) + RAW_OFFSET
    im[im < 0] = 0

    # Edge pixel correction
    im[0, :, :] = im[1, :, :]
    im[-1, :, :] = im[-2, :, :]
    im[:, 0, :] = im[:, 1, :]
    im[:, -1, :] = im[:, -2, :]

    return im


def load_tiff(filepath: str, start_frame: int = 0, num_frames: int | None = None) -> np.ndarray:
    """Load a TIFF stack as (H, W, N) float64.

    Handles both (T, H, W) and (H, W, T) layouts automatically.

    Args:
        filepath: Path to .tif/.tiff file.
        start_frame: First frame index (0-based).
        num_frames: Number of frames to load. None = all.

    Returns:
        Array of shape (H, W, N) as float64.
    """
    import tifffile
    data = tifffile.imread(filepath)
    if data.ndim == 2:
        # Single frame
        return data.astype(np.float64)[:, :, np.newaxis]
    elif data.ndim == 3:
        # Detect layout: if first dim is large and last two are similar → (T, H, W)
        if data.shape[0] > data.shape[1] or data.shape[0] > data.shape[2]:
            # (T, H, W) — most common TIFF stack layout
            im = data.transpose(1, 2, 0).astype(np.float64)
        else:
            # (H, W, T) already
            im = data.astype(np.float64)
    else:
        raise ValueError(f"Unexpected TIFF dimensions: {data.shape}")

    T = im.shape[2]
    if num_frames is None:
        num_frames = T - start_frame
    end = min(start_frame + num_frames, T)
    return im[:, :, start_frame:end]


def load_data_auto(filepath: str, start_frame: int = 0, num_frames: int | None = None) -> np.ndarray:
    """Load data from raw or TIFF file automatically based on extension.

    Args:
        filepath: Path to .raw or .tif/.tiff file.
        start_frame: First frame index.
        num_frames: Number of frames.

    Returns:
        Array of shape (H, W, N) as float64.
    """
    filepath = str(filepath)
    if filepath.lower().endswith(('.tif', '.tiff')):
        return load_tiff(filepath, start_frame, num_frames)
    else:
        return load_raw(filepath, start_frame, num_frames)


def get_raw_frame_count(filepath: str) -> int:
    """Get total number of frames in a raw file."""
    file_size = Path(filepath).stat().st_size
    return file_size // (RAW_WIDTH * RAW_HEIGHT * BYTES_PER_PIXEL)


def load_geo_calibration(filepath: str) -> dict:
    """Load geometric calibration .mat file.

    Handles both:
    - Original MATLAB files with affinetform2d objects (tforms unavailable)
    - Exported files with plain numeric arrays (tform_list_A)

    Returns dict with keys:
        ellipse_mask: list of 2D boolean arrays (one per view)
        resample_size: (H, W) tuple
        stretched_size: (H, W) tuple
        tform_list: ndarray (nView, nFrame, 3, 3) or None if unavailable
        tform_init: ndarray or None
        center_frame_idx: int
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = sio.loadmat(filepath, squeeze_me=True)

    # ellipse_mask
    ellipse_mask_raw = mat['ellipse_mask']
    if isinstance(ellipse_mask_raw, np.ndarray) and ellipse_mask_raw.dtype == object:
        ellipse_mask = [m.astype(bool) for m in ellipse_mask_raw]
    else:
        ellipse_mask = [ellipse_mask_raw.astype(bool)]

    resample_size = tuple(int(x) for x in np.atleast_1d(mat['resample_size']))
    stretched_size = tuple(int(x) for x in np.atleast_1d(mat['stretched_size']))
    center_frame_idx = int(mat.get('center_frame_idx', 0))

    # Try to load transforms (may be plain arrays or MATLAB objects)
    tform_list = _try_load_tforms(mat, 'tform_list', 'tform_list_A')
    tform_init = _try_load_tforms(mat, 'tform_init', 'tform_init_A')

    if tform_list is None and _needs_tform_export(filepath):
        # Check if exported version already exists alongside
        exported = Path(filepath).parent / 'geo_exported.mat'
        if exported.exists():
            print(f"  Auto-loading exported file: {exported}")
            return load_geo_calibration(str(exported))
        else:
            print("  WARNING: tform_list contains MATLAB objects that scipy cannot read.")

    return {
        'ellipse_mask': ellipse_mask,
        'resample_size': resample_size,
        'stretched_size': stretched_size,
        'tform_list': tform_list,
        'tform_init': tform_init,
        'center_frame_idx': center_frame_idx,
    }


def load_psf_calibration(filepath: str) -> dict:
    """Load PSF calibration .mat file.

    Returns dict with keys:
        patch_size: (H, W) tuple
        PSFs: ndarray (H, W, num_frame, num_view)
        optimized_tform_list: ndarray (nView, nFrame, 3, 3) or None
        pixel_size_x, pixel_size_y: float (um)
        PSFs_warped: ndarray (H, W, nDepth) or None (pre-warped synthetic PSF)
        PSFs_warped_all_view: ndarray (H, W, nDepth, nView) or None
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = sio.loadmat(filepath, squeeze_me=True)

    patch_size = tuple(int(x) for x in np.atleast_1d(mat['patch_size']))
    PSFs = mat['PSFs'].astype(np.float64)
    pixel_size_x = float(mat['pixel_size_x'])
    pixel_size_y = float(mat['pixel_size_y'])

    # Try transforms
    opt_tforms = _try_load_tforms(mat, 'optimized_tform_list', 'optimized_tform_list_A')

    if opt_tforms is None and _needs_tform_export(filepath):
        exported = Path(filepath).parent / 'psf_exported.mat'
        if exported.exists():
            print(f"  Auto-loading exported file: {exported}")
            return load_psf_calibration(str(exported))

    # Pre-warped PSFs (if available)
    PSFs_warped = mat.get('PSFs_warped', None)
    if PSFs_warped is not None:
        PSFs_warped = PSFs_warped.astype(np.float64)

    PSFs_warped_all_view = mat.get('PSFs_warped_all_view', None)
    if PSFs_warped_all_view is not None:
        PSFs_warped_all_view = PSFs_warped_all_view.astype(np.float64)

    return {
        'patch_size': patch_size,
        'PSFs': PSFs,
        'optimized_tform_list': opt_tforms,
        'pixel_size_x': pixel_size_x,
        'pixel_size_y': pixel_size_y,
        'PSFs_warped': PSFs_warped,
        'PSFs_warped_all_view': PSFs_warped_all_view,
    }


def export_mat_tforms(geo_path: str, psf_path: str, output_dir: str | None = None) -> tuple[str, str]:
    """Call MATLAB to export affinetform2d objects as plain numeric arrays.

    Args:
        geo_path: Path to original geo calibration .mat.
        psf_path: Path to original PSF calibration .mat.
        output_dir: Output directory. Defaults to same directory as inputs.

    Returns:
        (geo_exported_path, psf_exported_path)

    Raises:
        RuntimeError: If MATLAB is not found or export fails.
    """
    matlab_bin = shutil.which('matlab')
    if matlab_bin is None:
        raise RuntimeError(
            "MATLAB not found in PATH. Cannot auto-export affinetform2d objects.\n"
            "Either add MATLAB to PATH, or manually run export_tforms.m in MATLAB."
        )

    if output_dir is None:
        output_dir = str(Path(geo_path).parent)

    geo_out = str(Path(output_dir) / 'geo_exported.mat')
    psf_out = str(Path(output_dir) / 'psf_exported.mat')

    # Build inline MATLAB script
    # Use forward slashes for MATLAB path compatibility
    g = geo_path.replace('\\', '/')
    p = psf_path.replace('\\', '/')
    od = output_dir.replace('\\', '/')

    matlab_code = textwrap.dedent(f"""\
        S = load('{g}');
        tform_list_A = [];
        tform_init_A = [];
        if isfield(S,'tform_list')
            t = S.tform_list; [n1,n2] = size(t);
            tform_list_A = zeros(n1,n2,3,3);
            for i=1:n1, for j=1:n2, tform_list_A(i,j,:,:)=t(i,j).A; end, end
        end
        if isfield(S,'tform_init')
            ti = S.tform_init;
            if numel(ti)==1
                tform_init_A = reshape(ti.A,[1,1,3,3]);
            else
                [n1,n2]=size(ti);
                tform_init_A=zeros(n1,n2,3,3);
                for i=1:n1, for j=1:n2, tform_init_A(i,j,:,:)=ti(i,j).A; end, end
            end
        end
        ellipse_mask=S.ellipse_mask; resample_size=S.resample_size;
        stretched_size=S.stretched_size; center_frame_idx=S.center_frame_idx;
        save('{od}/geo_exported.mat','ellipse_mask','resample_size','stretched_size',...
             'center_frame_idx','tform_list_A','tform_init_A','-v6');
        S2 = load('{p}');
        optimized_tform_list_A = [];
        if isfield(S2,'optimized_tform_list')
            t2=S2.optimized_tform_list; [n1,n2]=size(t2);
            optimized_tform_list_A=zeros(n1,n2,3,3);
            for i=1:n1, for j=1:n2, optimized_tform_list_A(i,j,:,:)=t2(i,j).A; end, end
        end
        PSFs=S2.PSFs; patch_size=S2.patch_size;
        pixel_size_x=S2.pixel_size_x; pixel_size_y=S2.pixel_size_y;
        PSFs_warped=[]; PSFs_warped_all_view=[];
        if isfield(S2,'PSFs_warped'), PSFs_warped=S2.PSFs_warped; end
        if isfield(S2,'PSFs_warped_all_view'), PSFs_warped_all_view=S2.PSFs_warped_all_view; end
        save('{od}/psf_exported.mat','PSFs','patch_size','pixel_size_x','pixel_size_y',...
             'optimized_tform_list_A','PSFs_warped','PSFs_warped_all_view','-v6');
    """)

    print(f"  Calling MATLAB to export tform objects...")
    result = subprocess.run(
        [matlab_bin, '-batch', matlab_code],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"MATLAB export failed:\n{result.stderr}")

    if not Path(geo_out).exists() or not Path(psf_out).exists():
        raise RuntimeError(f"MATLAB ran but output files not created.\nstdout: {result.stdout}")

    print(f"  Exported: {geo_out}")
    print(f"  Exported: {psf_out}")
    return geo_out, psf_out


def _needs_tform_export(filepath: str) -> bool:
    """Check if a .mat file contains affinetform2d objects that need exporting."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = sio.loadmat(filepath, squeeze_me=True)
    # If there's a 'None' key with s0/s1/s2/arr struct, it's an unreadable MATLAB object
    if 'None' in mat:
        val = mat['None']
        if hasattr(val, 'dtype') and val.dtype.names and 's2' in val.dtype.names:
            return True
    return False


def _try_load_tforms(mat: dict, key_obj: str, key_numeric: str) -> np.ndarray | None:
    """Try to load transform matrices from a .mat dict.

    First tries the numeric exported key (e.g., 'tform_list_A'),
    then the original key, checking if it's actually numeric.
    Returns None if the data is a MATLAB class object.
    """
    # Try exported numeric array first
    if key_numeric in mat:
        val = mat[key_numeric]
        if isinstance(val, np.ndarray) and val.dtype != object:
            return val.astype(np.float64)

    # Try original key
    if key_obj in mat:
        val = mat[key_obj]
        if isinstance(val, np.ndarray) and val.dtype != object:
            # Already numeric
            if val.ndim >= 2 and val.shape[-1] == 3 and val.shape[-2] == 3:
                return val.astype(np.float64)

    return None
