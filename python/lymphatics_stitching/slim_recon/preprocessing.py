# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
# copied from E:\260325_stitch_process_dy_ver2\slim_recon\preprocessing.py on 2026-09-17
"""Preprocessing pipeline for SLIM light-field reconstruction."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from skimage.transform import resize, AffineTransform, warp


def select_frame(im_data: np.ndarray, frame_idx: int, num_avg: int = 1,
                 clip_range: tuple = (0.0, 1.0), gamma: float = 1.0) -> tuple[np.ndarray, float]:
    """Select and preprocess a frame from raw video data.

    Args:
        im_data: (H, W, N) raw video data.
        frame_idx: Frame index (0-based, after averaging).
        num_avg: Number of raw frames to average per displayed frame.
        clip_range: (low, high) percentile clip in [0, 1].
        gamma: Gamma correction value.

    Returns:
        (im_adj, im_amp): Preprocessed 2D frame and its max amplitude.
    """
    start = frame_idx * num_avg
    end = min(start + num_avg, im_data.shape[2])
    im_adj = np.mean(im_data[:, :, start:end], axis=2)

    im_amp = float(np.max(im_adj))

    # Rescale to [0, 1] then apply clip + gamma (like MATLAB imadjust)
    im_min, im_max = np.min(im_adj), np.max(im_adj)
    if im_max > im_min:
        im_adj = (im_adj - im_min) / (im_max - im_min)
    else:
        im_adj = np.zeros_like(im_adj)

    low, high = clip_range
    im_adj = np.clip((im_adj - low) / (high - low + 1e-10), 0, 1)
    if gamma != 1.0:
        im_adj = im_adj ** gamma

    return im_adj, im_amp


def extract_subapertures(im_adj: np.ndarray, ellipse_mask: list,
                         resample_size: tuple, stretched_size: tuple) -> np.ndarray:
    """Extract and stretch sub-aperture images from a preprocessed frame.

    Corresponds to MATLAB init_data():
    1. Resize transposed image to resample_size
    2. For each view, crop elliptical region and stretch to circle

    Args:
        im_adj: (H, W) preprocessed 2D frame.
        ellipse_mask: List of boolean masks, one per view.
        resample_size: Target size for initial resampling (H, W).
        stretched_size: Target size after stretching ellipses to circles (H, W).

    Returns:
        measurements: (H_s, W_s, num_views) array of sub-aperture images.
    """
    # Transpose and resize (MATLAB does imresize(im_adj', resample_size))
    # MATLAB imresize defaults to bicubic with antialiasing
    im_resized = resize(im_adj.T, resample_size, order=3, preserve_range=True, anti_aliasing=True)

    num_views = len(ellipse_mask)
    H_s, W_s = stretched_size
    measurements = np.zeros((H_s, W_s, num_views), dtype=np.float64)

    for i in range(num_views):
        mask = ellipse_mask[i]
        # Crop to bounding box of the mask
        rows, cols = np.where(mask)
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()

        im_sub = im_resized.copy()
        im_sub[~mask] = 0
        im_sub = im_sub[r_min:r_max + 1, c_min:c_max + 1]

        # Stretch to target size (making ellipse circular)
        im_stretched = resize(im_sub, stretched_size, order=3, preserve_range=True, anti_aliasing=True)

        # Rescale to [0, 1]
        s_min, s_max = im_stretched.min(), im_stretched.max()
        if s_max > s_min:
            im_stretched = (im_stretched - s_min) / (s_max - s_min)
        measurements[:, :, i] = im_stretched

    return measurements


def interp_transformations(tform_array: np.ndarray, lambda_vals: np.ndarray,
                           method: str = 'cubic') -> np.ndarray:
    """Interpolate affine transformation matrices across depth.

    Args:
        tform_array: (num_view, num_frame, 3, 3) or (num_frame, 3, 3) array.
        lambda_vals: Normalized depth values, shape (num_depth,).
        method: Interpolation method ('cubic', 'linear', etc.).

    Returns:
        Interpolated transforms: (num_view, num_depth, 3, 3) or (num_depth, 3, 3).
    """
    single_view = (tform_array.ndim == 3)
    if single_view:
        tform_array = tform_array[np.newaxis, ...]

    B, C, _, _ = tform_array.shape
    num_depth = len(lambda_vals)
    xi = np.linspace(0, 1, C)
    tform_interp = np.zeros((B, num_depth, 3, 3), dtype=np.float64)

    for i in range(B):
        # Flatten each 3x3 matrix in Fortran order to match MATLAB
        elements = tform_array[i].reshape(C, 9, order='F')
        elements_interp = np.zeros((num_depth, 9), dtype=np.float64)
        for k in range(9):
            fn = interp1d(xi, elements[:, k], kind=method, fill_value='extrapolate')
            elements_interp[:, k] = fn(lambda_vals)
        tform_interp[i] = elements_interp.reshape(num_depth, 3, 3, order='F')

    if single_view:
        return tform_interp[0]
    return tform_interp


def interpolate_psfs(PSFs: np.ndarray, lambda_vals: np.ndarray,
                     pca_k: int = 5) -> np.ndarray:
    """PCA-based PSF interpolation across depth.

    Args:
        PSFs: (H, W, num_frame, num_view) PSF data.
        lambda_vals: Normalized depth values, shape (num_depth,).
        pca_k: Number of PCA components to keep.

    Returns:
        PSF_interp: (H, W, num_depth, num_view) interpolated PSFs.
    """
    pH, pW, num_frame, num_view = PSFs.shape
    num_depth = len(lambda_vals)
    PSF_interp = np.zeros((pH, pW, num_depth, num_view), dtype=np.float64)
    x_orig = np.linspace(0, 1, num_frame)

    for i in range(num_view):
        # SVD decomposition (Fortran-order reshape to match MATLAB)
        A = PSFs[:, :, :, i].reshape((-1, num_frame), order='F')
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T

        U_k = U[:, :pca_k]
        S_k = np.diag(s[:pca_k])
        V_k = V[:, :pca_k]

        # Interpolate PCA coefficients
        V_interp = np.zeros((num_depth, pca_k))
        for j in range(pca_k):
            fn = interp1d(x_orig, V_k[:, j], kind='cubic', fill_value='extrapolate')
            V_interp[:, j] = fn(lambda_vals)

        # Reconstruct
        X_recon = U_k @ S_k @ V_interp.T  # (M, num_depth)
        X_recon = np.maximum(X_recon, 0)
        X_recon = X_recon.reshape((pH, pW, num_depth), order='F')

        PSF_interp[:, :, :, i] = X_recon

        # Normalize per view
        total = np.sum(PSF_interp[:, :, :, i])
        if total > 0:
            PSF_interp[:, :, :, i] /= total

    return PSF_interp


def compute_relative_transforms(backward_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor backward transforms into reference (depth-invariant) and relative (depth-dependent) parts.

    Args:
        backward_list: (num_view, num_depth, 3, 3) backward transforms.

    Returns:
        tform_bkwd_ref: (num_view, 3, 3) reference transforms (at mid-depth).
        tform_bkwd_rel: (num_view, num_depth, 3, 3) relative transforms.
    """
    num_view, num_depth = backward_list.shape[:2]
    mid_z = num_depth // 2

    tform_bkwd_ref = np.zeros((num_view, 3, 3), dtype=np.float64)
    tform_bkwd_rel = np.zeros((num_view, num_depth, 3, 3), dtype=np.float64)

    for i in range(num_view):
        tform_bkwd_ref[i] = backward_list[i, mid_z]
        A_ref_fwd = np.linalg.inv(backward_list[i, mid_z])
        for j in range(num_depth):
            tform_bkwd_rel[i, j] = backward_list[i, j] @ A_ref_fwd

    return tform_bkwd_ref, tform_bkwd_rel


def warp_psfs(PSFs_interp: np.ndarray, output_size: tuple,
              backward_rel_list: np.ndarray) -> np.ndarray:
    """Warp interpolated PSFs using relative transforms and average across views.

    Args:
        PSFs_interp: (pH, pW, num_depth, num_view) interpolated PSFs.
        output_size: (H, W) output image size.
        backward_rel_list: (num_view, num_depth, 3, 3) relative transforms.

    Returns:
        psf_syn: (H, W, num_depth) synthetic PSF averaged over views.
    """
    pH, pW, nz, nViews = PSFs_interp.shape
    H, W = output_size

    # Pad PSFs to output size
    pad_h = H - pH
    pad_w = W - pW
    PSFs_padded = np.pad(PSFs_interp, ((0, pad_h), (0, pad_w), (0, 0), (0, 0)), mode='constant')
    # Center the PSF
    shift_h = H // 2 - pH // 2
    shift_w = W // 2 - pW // 2
    PSFs_padded = np.roll(PSFs_padded, (shift_h, shift_w), axis=(0, 1))

    psf_syn = np.zeros((H, W, nz), dtype=np.float64)

    for z in range(nz):
        view_sum = np.zeros((H, W), dtype=np.float64)
        for v in range(nViews):
            A = backward_rel_list[v, z]
            tform = AffineTransform(matrix=A)
            warped = warp(PSFs_padded[:, :, z, v], tform.inverse,
                         output_shape=(H, W), mode='constant', cval=0, preserve_range=True)
            s = np.sum(warped)
            if s > 0:
                warped /= s
            view_sum += warped
        psf_syn[:, :, z] = view_sum / nViews

    return psf_syn


def apply_ref_transforms(measurements: np.ndarray,
                         tform_bkwd_ref: np.ndarray) -> np.ndarray:
    """Apply depth-invariant reference backward transforms to each view.

    Args:
        measurements: (H, W, num_views) sub-aperture images.
        tform_bkwd_ref: (num_views, 3, 3) reference transforms.

    Returns:
        measurements_trans: (H, W, num_views) transformed images.
    """
    H, W, nViews = measurements.shape
    measurements_trans = np.zeros_like(measurements)

    for i in range(nViews):
        tform = AffineTransform(matrix=tform_bkwd_ref[i])
        measurements_trans[:, :, i] = warp(
            measurements[:, :, i], tform.inverse,
            output_shape=(H, W), mode='constant', cval=0, preserve_range=True
        )

    return measurements_trans


def generate_roi_masks(forward_list: np.ndarray, image_size: tuple,
                       roi_ratio: float = 0.9, blur_range: float = 7.0) -> np.ndarray:
    """Generate soft ROI masks per view.

    Args:
        forward_list: (num_view, num_depth, 3, 3) forward transforms.
        image_size: (H, W).
        roi_ratio: Ellipse size ratio.
        blur_range: Gaussian blur sigma for soft edges.

    Returns:
        roi_mask: (H, W, num_views) soft masks.
    """
    H, W = image_size
    # Create centered elliptical mask
    cy, cx = H / 2, W / 2
    ry, rx = (H * roi_ratio) / 2, (W * roi_ratio) / 2
    y, x = np.ogrid[:H, :W]
    roi_center = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1
    roi_center = gaussian_filter(roi_center.astype(np.float64), blur_range)

    num_view, num_depth = forward_list.shape[:2]
    roi_mask = np.zeros((H, W, num_view), dtype=np.float64)

    for i in range(num_view):
        accumulated = np.zeros((H, W), dtype=np.float64)
        for j in range(num_depth):
            tform = AffineTransform(matrix=forward_list[i, j])
            warped = warp(roi_center, tform.inverse,
                         output_shape=(H, W), mode='constant', cval=0, preserve_range=True)
            accumulated += warped
        accumulated = np.minimum(accumulated, 1.0)
        roi_mask[:, :, i] = accumulated

    return roi_mask


def create_circular_mask(center_x: float, center_y: float, roi_ratio: float,
                         img_size: tuple, blur_radius: float = 5.0) -> np.ndarray:
    """Create a feathered circular mask using sigmoid function.

    Args:
        center_x, center_y: Mask center in pixels.
        roi_ratio: Ratio of mask diameter to smaller image dimension.
        img_size: (H, W).
        blur_radius: Edge feathering radius in pixels.

    Returns:
        mask: (H, W) array with values in [0, 1].
    """
    H, W = img_size
    radius = min(H, W) * roi_ratio / 2
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    if blur_radius <= 0:
        mask = (dist <= radius).astype(np.float64)
    else:
        mask = 1.0 / (1.0 + np.exp((dist - radius) / blur_radius))

    return np.clip(mask, 0, 1)


def apodize_measurements(measurements: np.ndarray, roi_ratio: float = 0.85,
                          blur_radius: float = 8.0) -> np.ndarray:
    """Apply soft circular window to each view's measurements to suppress edge artifacts.

    Args:
        measurements: (H, W, nViews) sub-aperture images.
        roi_ratio: Window diameter as fraction of image size.
        blur_radius: Sigmoid edge feathering radius in pixels.

    Returns:
        Apodized measurements (same shape).
    """
    H, W, nViews = measurements.shape
    window = create_circular_mask(W / 2, H / 2, roi_ratio, (H, W), blur_radius)
    return measurements * window[:, :, np.newaxis]


def compute_depth_coverage_mask(backward_rel_list: np.ndarray,
                                image_size: tuple,
                                roi_ratio: float = 0.9,
                                blur_radius: float = 8.0) -> np.ndarray:
    """Compute per-depth coverage mask showing how many views overlap at each pixel.

    For each depth, warps a centered circular mask through each view's relative
    transform and averages. Regions where fewer views contribute get lower values.

    Args:
        backward_rel_list: (nViews, nDepths, 3, 3) relative backward transforms.
        image_size: (H, W).
        roi_ratio: Circular mask diameter ratio.
        blur_radius: Edge feathering.

    Returns:
        coverage: (H, W, nDepths) array in [0, 1], where 1 = full view overlap.
    """
    H, W = image_size
    nViews, nDepths = backward_rel_list.shape[:2]

    # Base mask: centered circle
    base_mask = create_circular_mask(W / 2, H / 2, roi_ratio, (H, W), blur_radius)

    coverage = np.zeros((H, W, nDepths), dtype=np.float64)

    for z in range(nDepths):
        depth_sum = np.zeros((H, W), dtype=np.float64)
        for v in range(nViews):
            A = backward_rel_list[v, z]
            tform = AffineTransform(matrix=A)
            warped = warp(base_mask, tform.inverse,
                         output_shape=(H, W), mode='constant', cval=0,
                         preserve_range=True)
            depth_sum += warped
        # Normalize by number of views -> 1.0 where all views overlap
        coverage[:, :, z] = depth_sum / nViews

    return coverage


def psf2otf(psf: np.ndarray, output_size: tuple) -> np.ndarray:
    """Convert PSF to OTF (Optical Transfer Function).

    Args:
        psf: PSF array.
        output_size: Desired output shape.

    Returns:
        OTF as complex array.
    """
    # Pad PSF to output size
    pad_sizes = [(0, s - p) for p, s in zip(psf.shape, output_size)]
    psf_padded = np.pad(psf, pad_sizes, mode='constant')

    # Circularly shift so center of PSF is at (0,0,...,0)
    for ax in range(psf.ndim):
        psf_padded = np.roll(psf_padded, -(psf.shape[ax] // 2), axis=ax)

    otf = np.fft.fftn(psf_padded)

    # Discard imaginary part if within roundoff
    max_abs = np.max(np.abs(otf))
    max_imag = np.max(np.abs(np.imag(otf)))
    if max_abs > 0 and max_imag / max_abs < psf.size * np.finfo(float).eps:
        otf = np.real(otf)

    return otf
