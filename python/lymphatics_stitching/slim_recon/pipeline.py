# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
# copied from E:\260325_stitch_process_dy_ver2\slim_recon\pipeline.py on 2026-09-17
"""End-to-end reconstruction pipeline for SLIM light-field microscopy.

This module ties together all stages:
1. Load raw data
2. Load geometric & PSF calibration
3. Select & preprocess a frame
4. Extract sub-apertures
5. Interpolate transforms & PSFs (or use pre-warped PSFs)
6. Prepare reconstruction operators (OTFs, masks)
7. Run GPU-accelerated RL deconvolution
8. Visualize 3D reconstruction
"""

import numpy as np
from skimage.transform import AffineTransform, warp

from . import data_io
from . import preprocessing as pp
from . import reconstruction as recon
from . import visualization as vis
from .viewer import ReconViewer


class SLIMPipeline:
    """Full reconstruction pipeline for a single frame."""

    def __init__(self):
        # Raw data
        self.im_data = None

        # Calibration
        self.geo_cal = None
        self.psf_cal = None

        # Preprocessing results
        self.im_adj = None
        self.im_amp = None
        self.measurements = None
        self.measurements_trans = None

        # Transform interpolation
        self.forward_list = None
        self.backward_list = None
        self.tform_bkwd_ref = None
        self.tform_bkwd_rel = None

        # PSF / OTF
        self.psf_all_view = None
        self.OTFs = None

        # Masks
        self.roi_mask = None
        self.obj_mask = None

        # Coverage mask for edge suppression
        self.depth_coverage = None

        # Reconstruction
        self.recon_result = None

        # Parameters
        self.depth_range = (0.0, 1.0)
        self.n_depth = 32
        self.n_iters = 10
        self.dampar = 0.0
        self.readout = 0.0
        self.psf_n = 1
        self.pca_k = 5

    def load_data(self, raw_path: str, num_frames: int | None = None):
        """Step 1: Load raw or TIFF video data."""
        print(f"Loading data: {raw_path}")
        self.im_data = data_io.load_data_auto(raw_path, num_frames=num_frames)
        print(f"  Shape: {self.im_data.shape} (H, W, N)")

    def load_calibration(self, geo_path: str, psf_path: str):
        """Step 2: Load geometric and PSF calibration.

        If the .mat files contain MATLAB affinetform2d objects that scipy
        cannot read, automatically calls MATLAB to export them as plain
        numeric arrays (requires MATLAB in PATH).
        """
        print(f"Loading geometric calibration: {geo_path}")
        self.geo_cal = data_io.load_geo_calibration(geo_path)

        # Auto-export if tforms are missing and MATLAB objects detected
        if (self.geo_cal['tform_list'] is None
                and data_io._needs_tform_export(geo_path)):
            print("  Transforms unavailable — attempting MATLAB auto-export...")
            try:
                geo_exp, psf_exp = data_io.export_mat_tforms(geo_path, psf_path)
                self.geo_cal = data_io.load_geo_calibration(geo_exp)
                psf_path = psf_exp  # use exported PSF file below
            except RuntimeError as e:
                print(f"  Auto-export failed: {e}")

        print(f"  Views: {len(self.geo_cal['ellipse_mask'])}, "
              f"resample: {self.geo_cal['resample_size']}, "
              f"stretched: {self.geo_cal['stretched_size']}")

        print(f"Loading PSF calibration: {psf_path}")
        self.psf_cal = data_io.load_psf_calibration(psf_path)
        print(f"  PSF shape: {self.psf_cal['PSFs'].shape}, "
              f"pixel_size: ({self.psf_cal['pixel_size_x']:.2f}, {self.psf_cal['pixel_size_y']:.2f}) um")

        if self.psf_cal['PSFs_warped_all_view'] is not None:
            print(f"  Pre-warped PSFs available: {self.psf_cal['PSFs_warped_all_view'].shape}")

    def preprocess_frame(self, frame_idx: int = 0, num_avg: int = 1,
                         clip_range: tuple = (0.0, 1.0), gamma: float = 1.0):
        """Step 3: Select and preprocess a frame."""
        print(f"Preprocessing frame {frame_idx} (avg={num_avg})")
        self.im_adj, self.im_amp = pp.select_frame(
            self.im_data, frame_idx, num_avg, clip_range, gamma
        )
        print(f"  Frame shape: {self.im_adj.shape}, amplitude: {self.im_amp:.2f}")

    def extract_views(self):
        """Step 3b: Extract sub-aperture views from the preprocessed frame."""
        print("Extracting sub-aperture views...")
        self.measurements = pp.extract_subapertures(
            self.im_adj,
            self.geo_cal['ellipse_mask'],
            self.geo_cal['resample_size'],
            self.geo_cal['stretched_size'],
        )
        print(f"  Measurements shape: {self.measurements.shape}")

    def prepare_reconstruction(self, depth_range: tuple | None = None,
                               n_depth: int | None = None,
                               psf_n: int = 1):
        """Step 4: Prepare all reconstruction operators.

        Args:
            depth_range: (min, max) normalized depth range.
            n_depth: Number of depth slices.
            psf_n: PSF sharpening exponent. PSF^n is applied after warping
                   and re-normalized per (depth, view) slice. Higher values
                   sharpen the PSF (narrower main lobe, reduced sidelobes).
        """
        if depth_range is not None:
            self.depth_range = depth_range
        if n_depth is not None:
            self.n_depth = n_depth
        self.psf_n = psf_n

        stretched = self.geo_cal['stretched_size']
        H, W = stretched
        has_tforms = (self.psf_cal['optimized_tform_list'] is not None)

        if has_tforms:
            self._prepare_with_transforms()
        else:
            self._prepare_with_prewarped()

        # Object mask
        print("Creating object mask...")
        self.obj_mask = pp.create_circular_mask(W / 2, H / 2, 0.9, (H, W), blur_radius=5)

        print("Preparation complete.")

    def _prepare_with_transforms(self):
        """Full path: interpolate transforms, PSFs, warp, compute OTFs."""
        lambda_vals = np.linspace(self.depth_range[0], self.depth_range[1], self.n_depth)
        stretched = self.geo_cal['stretched_size']
        H, W = stretched

        opt_tforms = self.psf_cal['optimized_tform_list']
        nView, nFrame = opt_tforms.shape[:2]

        # Forward transforms = inverse of backward
        print("Interpolating transforms...")
        fwd_tforms = np.zeros_like(opt_tforms)
        for i in range(nView):
            for j in range(nFrame):
                fwd_tforms[i, j] = np.linalg.inv(opt_tforms[i, j])

        self.forward_list = pp.interp_transformations(fwd_tforms, lambda_vals)
        self.backward_list = pp.interp_transformations(opt_tforms, lambda_vals)

        # Relative transforms
        print("Computing relative transforms...")
        self.tform_bkwd_ref, self.tform_bkwd_rel = pp.compute_relative_transforms(self.backward_list)

        # ROI masks
        print("Generating ROI masks...")
        self.roi_mask = pp.generate_roi_masks(self.forward_list, stretched)

        # Apply reference transforms
        print("Applying reference transforms...")
        self.measurements_trans = pp.apply_ref_transforms(self.measurements, self.tform_bkwd_ref)

        # Apodize measurements to suppress edge artifacts
        print("Apodizing measurements...")
        self.measurements_trans = pp.apodize_measurements(
            self.measurements_trans, roi_ratio=0.85, blur_radius=8.0
        )

        # Compute depth-dependent coverage mask for post-processing
        print("Computing depth coverage mask...")
        self.depth_coverage = pp.compute_depth_coverage_mask(
            self.tform_bkwd_rel, stretched, roi_ratio=0.90, blur_radius=8.0
        )

        # Interpolate PSFs
        print("Interpolating PSFs (PCA)...")
        PSF_interp = pp.interpolate_psfs(self.psf_cal['PSFs'], lambda_vals, pca_k=self.pca_k)

        # Warp PSFs (BEFORE exponentiation, matching MATLAB init_psf order)
        print("Warping PSFs...")
        self.psf_all_view = self._warp_psfs_multiview(
            PSF_interp, stretched, self.tform_bkwd_rel
        )

        # PSF sharpening: exponentiation AFTER warping (MATLAB: psf_all_view.^n)
        if self.psf_n > 1:
            self.psf_all_view = self.psf_all_view ** self.psf_n

        # Normalize per (depth, view) spatial slice (MATLAB: sum over dims [1,2])
        for v in range(self.psf_all_view.shape[3]):
            for z in range(self.psf_all_view.shape[2]):
                s = np.sum(self.psf_all_view[:, :, z, v])
                if s > 0:
                    self.psf_all_view[:, :, z, v] /= s

        # OTFs (includes DC normalization)
        print("Computing OTFs...")
        self.OTFs = recon.prepare_multiview_otf(self.psf_all_view)
        print(f"  OTFs shape: {self.OTFs.shape}")

    def _prepare_with_prewarped(self):
        """Fast path: use pre-warped PSFs from calibration file."""
        print("Using pre-warped PSFs (transforms unavailable)...")
        stretched = self.geo_cal['stretched_size']
        H, W = stretched

        psf_all_view = self.psf_cal['PSFs_warped_all_view']  # (256, 256, 21, 5)
        if psf_all_view is None:
            raise RuntimeError(
                "Neither transforms nor pre-warped PSFs are available. "
                "Run export_tforms.m in MATLAB to export transforms."
            )

        nPsfH, nPsfW, nPsfDepth, nViews = psf_all_view.shape
        print(f"  Pre-warped PSF shape: {psf_all_view.shape}")

        # Check size match with stretched_size
        if nPsfH != H or nPsfW != W:
            print(f"  WARNING: PSF size ({nPsfH},{nPsfW}) != stretched_size ({H},{W}).")
            print(f"  Using PSF size as reconstruction grid.")

        # The pre-warped PSFs have nPsfDepth depth slices; use as-is or resample
        self.n_depth = nPsfDepth
        self.psf_all_view = psf_all_view.copy()

        # PSF sharpening and per-slice normalization (matching MATLAB init_psf)
        if self.psf_n > 1:
            self.psf_all_view = self.psf_all_view ** self.psf_n
        for v in range(nViews):
            for z in range(nPsfDepth):
                s = np.sum(self.psf_all_view[:, :, z, v])
                if s > 0:
                    self.psf_all_view[:, :, z, v] /= s

        # For measurements_trans: without reference transforms, use raw measurements
        # The pre-warped PSFs already account for the geometric transforms
        self.measurements_trans = pp.apodize_measurements(
            self.measurements, roi_ratio=0.85, blur_radius=8.0
        )
        print(f"  Using {nViews} views, {self.n_depth} depth slices")

        # ROI mask: uniform since we can't compute view-specific masks without transforms
        print("  Using uniform ROI masks (no transforms available)")
        self.roi_mask = np.ones((nPsfH, nPsfW, nViews), dtype=np.float64)

        # OTFs (computed from processed psf_all_view, includes DC normalization)
        print("Computing OTFs from pre-warped PSFs...")
        self.OTFs = recon.prepare_multiview_otf(self.psf_all_view)
        print(f"  OTFs shape: {self.OTFs.shape}")

    def _warp_psfs_multiview(self, PSFs_interp, output_size, backward_rel_list):
        """Warp PSFs per view."""
        pH, pW, nz, nViews = PSFs_interp.shape
        H, W = output_size

        pad_h = H - pH
        pad_w = W - pW
        PSFs_padded = np.pad(PSFs_interp, ((0, pad_h), (0, pad_w), (0, 0), (0, 0)), mode='constant')
        shift_h = H // 2 - pH // 2
        shift_w = W // 2 - pW // 2
        PSFs_padded = np.roll(PSFs_padded, (shift_h, shift_w), axis=(0, 1))

        psf_all_view = np.zeros((H, W, nz, nViews), dtype=np.float64)

        for z in range(nz):
            for v in range(nViews):
                A = backward_rel_list[v, z]
                tform = AffineTransform(matrix=A)
                warped = warp(PSFs_padded[:, :, z, v], tform.inverse,
                             output_shape=(H, W), mode='constant', cval=0, preserve_range=True)
                s = np.sum(warped)
                if s > 0:
                    warped /= s
                psf_all_view[:, :, z, v] = warped

        return psf_all_view

    def reconstruct(self, n_iters: int | None = None, dampar: float | None = None,
                    readout: float | None = None, view_indices: list | None = None,
                    coverage_mode: str = 'post_hoc', coverage_power: float = 1.0):
        """Step 5: Run RL reconstruction.

        Args:
            n_iters: Number of RL iterations.
            dampar: Damping parameter.
            readout: Readout noise level.
            view_indices: View subset indices.
            coverage_mode: How to apply depth coverage mask:
                'per_iter' - apply inside RL loop (prevents artifact accumulation)
                'post_hoc' - apply after RL (original behavior)
                'none' - don't apply coverage mask
            coverage_power: Exponent applied to coverage mask. Higher values
                give sharper edge cutoff (1.0 = linear, 3.0 = steep, 5.0 = near step).
        """
        if n_iters is not None:
            self.n_iters = n_iters
        if dampar is not None:
            self.dampar = dampar
        if readout is not None:
            self.readout = readout

        # Prepare coverage mask with power
        cov_powered = None
        if self.depth_coverage is not None and coverage_mode != 'none':
            cov_powered = self.depth_coverage ** coverage_power

        use_per_iter = (coverage_mode == 'per_iter' and cov_powered is not None)

        print(f"Starting RL reconstruction ({self.n_iters} iterations, "
              f"coverage={coverage_mode}, power={coverage_power:.1f})...")
        self.recon_result = recon.deconvlucy_gpu(
            self.measurements_trans,
            self.OTFs,
            num_iter=self.n_iters,
            dampar=self.dampar,
            roi_mask=self.roi_mask,
            readout=self.readout,
            view_idx_list=view_indices,
            depth_coverage=cov_powered if use_per_iter else None,
        )

        # Post-hoc coverage mask (only if not already applied per-iteration)
        if coverage_mode == 'post_hoc' and cov_powered is not None:
            n_recon_depth = self.recon_result.shape[2]
            n_cov_depth = cov_powered.shape[2]
            if n_recon_depth == n_cov_depth:
                self.recon_result = self.recon_result * cov_powered
            else:
                from scipy.ndimage import zoom
                scale = [1, 1, n_recon_depth / n_cov_depth]
                cov_resized = zoom(cov_powered, scale, order=1)
                self.recon_result = self.recon_result * cov_resized

        # Normalize to [0, 1]
        rmin, rmax = self.recon_result.min(), self.recon_result.max()
        if rmax > rmin:
            self.recon_result = (self.recon_result - rmin) / (rmax - rmin)

        print(f"  Reconstruction shape: {self.recon_result.shape}")

    def visualize(self, num_slices: int = 8, gamma: float = 0.5,
                  save_prefix: str | None = None):
        """Step 6: Save static visualization.

        Args:
            num_slices: Number of depth slices to show.
            gamma: Display gamma. <1 compresses highlights.
            save_prefix: If provided, save images as {prefix}_slices.png etc.
        """
        if self.recon_result is None:
            print("No reconstruction result to visualize.")
            return

        vis.show_depth_slices(
            self.recon_result,
            num_slices=num_slices,
            depth_range=self.depth_range,
            gamma=gamma,
            title='Reconstruction Depth Slices',
            save_path=f"{save_prefix}_slices.png" if save_prefix else None,
        )

        vis.show_mip(
            self.recon_result,
            depth_range=self.depth_range,
            gamma=gamma,
            title='Maximum Intensity Projection',
            save_path=f"{save_prefix}_mip.png" if save_prefix else None,
        )

    def view(self):
        """Step 6 (interactive): Open interactive viewer with sliders."""
        if self.recon_result is None:
            print("No reconstruction result to visualize.")
            return
        viewer = ReconViewer(self.recon_result, self.depth_range)
        viewer.show()
