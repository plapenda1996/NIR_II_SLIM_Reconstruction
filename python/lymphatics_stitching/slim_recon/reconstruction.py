# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
"""GPU-accelerated Richardson-Lucy reconstruction for SLIM light-field microscopy.

Uses PyTorch for GPU operations. The forward/backward models implement
3D convolution via FFT with per-view OTFs.
"""

import torch
import numpy as np
import time


def forward_model(reconstruction: torch.Tensor, OTFs: torch.Tensor) -> torch.Tensor:
    """Forward projection: 3D volume -> 2D multi-view projections.

    Implements circular convolution with OTFs then extracts z=0 slice.

    Args:
        reconstruction: (H, W, D) 3D volume.
        OTFs: (H, W, D, nViews) OTF array.

    Returns:
        projections: (H, W, nViews) 2D projections.
    """
    H, W, D = reconstruction.shape
    nViews = OTFs.shape[3]

    # Center and FFT the reconstruction
    shift = torch.tensor([-(H // 2), -(W // 2), -(D // 2)], device=reconstruction.device)
    recon_shifted = torch.roll(reconstruction, shifts=tuple(shift.tolist()), dims=(0, 1, 2))
    recon_fft = torch.fft.fftn(recon_shifted)

    # Multiply by OTFs (broadcast over views)
    # recon_fft: (H, W, D) -> (H, W, D, 1) for broadcasting
    conv_fft = recon_fft.unsqueeze(-1) * OTFs  # (H, W, D, nViews)

    # Inverse FFT along xy dimensions, then along z
    conv_fft = torch.fft.ifft(conv_fft, dim=2)
    conv = torch.fft.ifft2(conv_fft, dim=(0, 1))

    # Shift back (only xy, not z or views)
    projections = torch.roll(conv.real, shifts=(H // 2, W // 2), dims=(0, 1))

    # Extract z=0 slice (index 0)
    projections = projections[:, :, 0, :]  # (H, W, nViews)

    return projections


def backward_model(projections: torch.Tensor, OTFs: torch.Tensor) -> torch.Tensor:
    """Backward projection (adjoint): 2D multi-view projections -> 3D volume.

    Embeds projections at mid-z, applies conjugate OTF convolution, averages views.

    Args:
        projections: (H, W, nViews) 2D projections.
        OTFs: (H, W, D, nViews) OTF array.

    Returns:
        reconstruction: (H, W, D) 3D volume.
    """
    H, W, nViews = projections.shape
    D = OTFs.shape[2]
    # MATLAB: mid_z = floor(nz/2) in 1-indexed = (D // 2 - 1) in 0-indexed
    mid_z = D // 2 - 1

    # Embed projections at mid-z
    proj_emb = torch.zeros(H, W, D, nViews, dtype=projections.dtype, device=projections.device)
    proj_emb[:, :, mid_z, :] = projections

    # ifftshift along spatial and depth dims
    proj_emb = torch.fft.ifftshift(proj_emb, dim=0)
    proj_emb = torch.fft.ifftshift(proj_emb, dim=1)
    proj_emb = torch.fft.ifftshift(proj_emb, dim=2)

    # FFT: 2D spatial + 1D depth
    proj_fft = torch.fft.fft2(proj_emb, dim=(0, 1))
    proj_fft = torch.fft.fft(proj_fft, dim=2)

    # Conjugate OTF convolution
    recon_fft = proj_fft * torch.conj(OTFs)

    # Inverse FFT
    recon = torch.fft.ifft(recon_fft, dim=2)
    recon = torch.fft.ifft2(recon, dim=(0, 1))

    # fftshift back
    recon = torch.fft.fftshift(recon.real, dim=0)
    recon = torch.fft.fftshift(recon, dim=1)
    recon = torch.fft.fftshift(recon, dim=2)

    # Average over views
    reconstruction = torch.mean(recon, dim=3)

    return reconstruction


def deconvlucy_gpu(measurements: np.ndarray,
                   otf_np: np.ndarray,
                   num_iter: int = 10,
                   dampar: float = 0.0,
                   roi_mask: np.ndarray | None = None,
                   readout: float = 0.0,
                   view_idx_list: list | None = None,
                   depth_coverage: np.ndarray | None = None,
                   device: str = 'cuda') -> np.ndarray:
    """GPU-accelerated Richardson-Lucy deconvolution with Biggs-Andrews acceleration.

    Args:
        measurements: (H, W, nViews) transformed measurements.
        otf_np: (H, W, D, nViews) OTF array (numpy, complex).
        num_iter: Number of RL iterations.
        dampar: Damping parameter for noise suppression.
        roi_mask: (H, W, nViews) weight mask. None = uniform weights.
        readout: Readout noise level.
        view_idx_list: List of view indices to use. None = all.
        depth_coverage: (H, W, D) per-voxel coverage mask in [0, 1].
            If provided, applied as a multiplicative constraint at each
            RL iteration to prevent edge artifacts from accumulating.
            This is more effective than post-hoc masking because it
            stops the RL from amplifying partially-covered edge voxels.
        device: PyTorch device.

    Returns:
        result: (H, W, D) reconstructed 3D volume (numpy).
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Select views
    if view_idx_list is not None:
        measurements = measurements[:, :, view_idx_list]
        otf_np = otf_np[:, :, :, view_idx_list]
        if roi_mask is not None:
            roi_mask = roi_mask[:, :, view_idx_list]

    H, W, nViews = measurements.shape
    D = otf_np.shape[2]

    # Move to GPU
    meas = torch.tensor(measurements, dtype=torch.float32, device=dev)
    OTFs = torch.tensor(otf_np, dtype=torch.complex64, device=dev)

    if roi_mask is not None:
        weight = torch.tensor(roi_mask, dtype=torch.float32, device=dev)
    else:
        weight = torch.ones(H, W, nViews, dtype=torch.float32, device=dev)

    # Depth coverage constraint (applied per-iteration)
    if depth_coverage is not None:
        cov = torch.tensor(depth_coverage, dtype=torch.float32, device=dev)
    else:
        cov = None

    READOUT = readout

    # Weighted input
    wI = torch.clamp(weight * (READOUT + meas), min=0)

    # Scale factor: backward_model(weight)
    scale = backward_model(weight, OTFs) + torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps))

    DAMPAR22 = (dampar ** 2) / 2

    # Initialize estimate via back-projection
    J_cur = torch.clamp(backward_model(meas, OTFs), min=0)
    if cov is not None:
        J_cur = J_cur * cov
    J_prev = torch.zeros_like(J_cur)
    # Acceleration vectors
    accel_0 = torch.zeros(J_cur.numel(), dtype=torch.float32, device=dev)
    accel_1 = torch.zeros_like(accel_0)

    lam = 0.0

    start_time = time.time()
    for k in range(num_iter):
        # Biggs-Andrews acceleration
        if k > 1:
            num = torch.dot(accel_0, accel_1)
            denom = torch.dot(accel_1, accel_1) + torch.finfo(torch.float32).eps
            lam = float(torch.clamp(num / denom, 0, 1))

        # Accelerated prediction
        Y = torch.clamp(J_cur + lam * (J_cur - J_prev), min=0)

        # Forward projection
        reblurred = forward_model(Y, OTFs) + READOUT
        reblurred = torch.clamp(reblurred, min=torch.finfo(torch.float32).eps)

        # Ratio
        ratio = wI / reblurred

        # Damping
        if DAMPAR22 > 0:
            gm = 10
            g = (wI * torch.log(ratio + torch.finfo(torch.float32).eps) + reblurred - wI) / DAMPAR22
            g = torch.clamp(g, max=1.0)
            G = (g ** (gm - 1)) * (gm - (gm - 1) * g)
            im_ratio = 1 + G * (ratio - 1)
        else:
            im_ratio = ratio

        # Update
        J_prev = J_cur.clone()
        J_cur = torch.clamp(Y * backward_model(im_ratio, OTFs) / scale, min=0)

        # Per-iteration depth coverage constraint:
        # Suppress edge voxels where fewer views overlap, preventing
        # artifacts from accumulating through RL iterations.
        if cov is not None:
            J_cur = J_cur * cov

        # Acceleration tracking
        accel_1 = accel_0.clone()
        accel_0 = (J_cur - Y).reshape(-1)

        elapsed = time.time() - start_time
        eta = (elapsed / (k + 1)) * (num_iter - k - 1)
        print(f"  RL iter {k + 1}/{num_iter}, elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s", end='\r')

    total = time.time() - start_time
    print(f"\n  RL done in {total:.2f}s")

    return J_cur.cpu().numpy()


def prepare_otf(psf_syn: np.ndarray, output_size: tuple) -> np.ndarray:
    """Compute per-view 3D OTFs from synthetic PSF.

    For the single-PSF case (averaged across views), the same OTF is
    broadcast to all views.

    Args:
        psf_syn: (H, W, D) synthetic PSF (already at output_size in H,W).
        output_size: (H, W, D) — should match psf_syn shape.

    Returns:
        OTF: (H, W, D) complex OTF array.
    """
    H, W, D = psf_syn.shape

    # Circularly shift PSF center to (0,0,0)
    psf_shifted = psf_syn.copy()
    for ax in range(3):
        psf_shifted = np.roll(psf_shifted, -(psf_syn.shape[ax] // 2), axis=ax)

    otf = np.fft.fftn(psf_shifted)
    return otf


def prepare_multiview_otf(psf_all_view: np.ndarray) -> np.ndarray:
    """Compute per-view 3D OTFs.

    Args:
        psf_all_view: (H, W, D, nViews) warped PSFs per view.

    Returns:
        OTFs: (H, W, D, nViews) complex OTF array.
    """
    H, W, D, nViews = psf_all_view.shape
    OTFs = np.zeros((H, W, D, nViews), dtype=np.complex128)

    for v in range(nViews):
        psf_v = psf_all_view[:, :, :, v].copy()
        for ax in range(3):
            psf_v = np.roll(psf_v, -(psf_all_view.shape[ax] // 2), axis=ax)
        otf_v = np.fft.fftn(psf_v)
        # Normalize by DC component (MATLAB: otf / abs(otf(1,1,1)))
        dc = np.abs(otf_v[0, 0, 0])
        if dc > 0:
            otf_v = otf_v / dc
        OTFs[:, :, :, v] = otf_v

    return OTFs
