# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
"""Visualization utilities for 3D reconstruction results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def show_depth_slices(volume: np.ndarray, num_slices: int = 8,
                      depth_range: tuple = (0, 1), cmap: str = 'gray',
                      gamma: float = 0.5,
                      title: str = 'Depth Slices', save_path: str | None = None):
    """Display selected depth slices from a 3D volume.

    Args:
        volume: (H, W, D) 3D reconstructed volume.
        num_slices: Number of slices to display.
        depth_range: (min, max) depth range labels.
        cmap: Colormap.
        gamma: Display gamma (<1 compresses highlights, shows more detail).
        title: Figure title.
        save_path: If provided, save figure to this path.
    """
    D = volume.shape[2]
    indices = np.linspace(0, D - 1, num_slices, dtype=int)
    depths = np.linspace(depth_range[0], depth_range[1], D)

    cols = min(num_slices, 4)
    rows = (num_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_slices == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Normalize volume to [0, 1] then apply gamma for display
    vmin = np.percentile(volume, 0.5)
    vmax = np.percentile(volume, 99.9)

    for i, idx in enumerate(indices):
        slc = np.clip((volume[:, :, idx] - vmin) / (vmax - vmin + 1e-10), 0, 1)
        slc = slc ** gamma
        axes[i].imshow(slc, cmap=cmap, vmin=0, vmax=1)
        axes[i].set_title(f'd={depths[idx]:.3f}')
        axes[i].axis('off')

    for i in range(len(indices), len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def show_mip(volume: np.ndarray, depth_range: tuple = (0, 1),
             gamma: float = 0.5,
             title: str = 'MIP', save_path: str | None = None,
             plow: float = 0.5, phigh: float = 99.9,
             denoise_sigma: float = 0.0,
             clahe: bool = True, clahe_clip: float = 2.0,
             depth_cmap: str = 'turbo'):
    """Display Maximum Intensity Projection with depth color coding.

    Args:
        volume: (H, W, D) 3D volume.
        depth_range: (shallow, deep) depth labels.
        gamma: Display gamma (<1 compresses highlights, shows more detail).
        title: Figure title.
        save_path: If provided, save to this path.
        plow: Lower percentile for intensity clipping (default 0.5).
        phigh: Upper percentile for intensity clipping (default 99.9).
        denoise_sigma: If > 0, apply Gaussian smoothing per slice before MIP.
        clahe: If True, apply CLAHE to enhance local contrast on grayscale MIP.
        clahe_clip: Clip limit for CLAHE (higher = more contrast).
        depth_cmap: Colormap for depth-coded MIP (default 'turbo').
    """
    from scipy.ndimage import gaussian_filter

    H, W, D = volume.shape

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    # ── Pre-MIP per-slice processing ──
    vol_proc = volume.copy()

    # Per-slice Gaussian denoising
    if denoise_sigma > 0:
        for d in range(D):
            vol_proc[:, :, d] = gaussian_filter(vol_proc[:, :, d], sigma=denoise_sigma)

    # Per-slice bilateral denoising (edge-preserving, better than Gaussian)
    if has_cv2 and denoise_sigma > 0:
        for d in range(D):
            u8 = (np.clip(vol_proc[:, :, d], vol_proc.min(), vol_proc.max()) * 255 /
                  (vol_proc.max() + 1e-10)).astype(np.uint8)
            u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=40, sigmaSpace=40)
            vol_proc[:, :, d] = u8.astype(np.float64) * vol_proc.max() / 255.0

    # Robust percentile normalization
    vmin = np.percentile(vol_proc, plow)
    vmax = np.percentile(vol_proc, phigh)
    vol_norm = np.clip((vol_proc - vmin) / (vmax - vmin + 1e-10), 0, 1)

    # MIP: find max depth index and max value
    max_idx = np.argmax(vol_norm, axis=2)
    max_val = np.max(vol_norm, axis=2)

    # Apply gamma
    max_val_disp = max_val ** gamma

    # CLAHE for local contrast enhancement
    if clahe and has_cv2:
        mip_u8 = (max_val_disp * 255).astype(np.uint8)
        cl = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        mip_clahe = cl.apply(mip_u8)
        max_val_enhanced = mip_clahe.astype(np.float64) / 255.0
    else:
        max_val_enhanced = max_val_disp

    # Background subtraction on MIP (percentile floor removal)
    bg_floor = np.percentile(max_val_enhanced[max_val_enhanced > 0], 5) if np.any(max_val_enhanced > 0) else 0
    max_val_bg_sub = np.clip(max_val_enhanced - bg_floor, 0, None)
    mx = max_val_bg_sub.max()
    if mx > 0:
        max_val_bg_sub /= mx

    # Create depth-coded color image
    cmap_obj = plt.colormaps.get_cmap(depth_cmap)
    depth_norm = max_idx.astype(float) / max(D - 1, 1)
    depth_colors = cmap_obj(depth_norm)[:, :, :3]
    intensity = max_val_bg_sub[:, :, np.newaxis]
    rgb = depth_colors * intensity

    # ── Figure (4 panels) ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                              gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    fig.patch.set_facecolor('black')

    # Panel 1: Raw MIP
    axes[0].imshow(max_val_disp, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('MIP (raw)', color='white', fontsize=10,
                       fontfamily='Arial', fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Enhanced MIP (CLAHE + BG removal)
    axes[1].imshow(max_val_bg_sub, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('MIP (enhanced)', color='white', fontsize=10,
                       fontfamily='Arial', fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Depth-coded MIP
    axes[2].imshow(rgb)
    axes[2].set_title('MIP (depth-coded)', color='white', fontsize=10,
                       fontfamily='Arial', fontweight='bold')
    axes[2].axis('off')

    # Panel 4: Depth map only (for structure visualization)
    # Mask out low-signal regions
    sig_mask = max_val_bg_sub > 0.05
    depth_map = np.where(sig_mask, depth_norm, np.nan)
    axes[3].imshow(depth_map, cmap=depth_cmap, vmin=0, vmax=1)
    axes[3].set_title('Depth map', color='white', fontsize=10,
                       fontfamily='Arial', fontweight='bold')
    axes[3].axis('off')

    # Depth colorbar
    sm = plt.cm.ScalarMappable(cmap=depth_cmap,
                                norm=plt.Normalize(depth_range[0], depth_range[1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[2:], fraction=0.03, pad=0.04, shrink=0.8)
    cbar.set_label('Depth', fontsize=9, color='white')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=8)

    fig.suptitle(title, fontsize=12, fontfamily='Arial',
                  fontweight='bold', color='white')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                     facecolor='black')
        print(f"Saved: {save_path}")
    plt.close(fig)


def show_subapertures(measurements: np.ndarray, title: str = 'Sub-aperture Views',
                      save_path: str | None = None):
    """Display extracted sub-aperture views."""
    nViews = measurements.shape[2]
    fig, axes = plt.subplots(1, nViews, figsize=(4 * nViews, 4))
    if nViews == 1:
        axes = [axes]
    for i in range(nViews):
        axes[i].imshow(measurements[:, :, i], cmap='gray')
        axes[i].set_title(f'View {i + 1}')
        axes[i].axis('off')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
