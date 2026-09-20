# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
# copied from E:\260325_stitch_process_dy_ver2\slim_recon\denoise.py on 2026-09-17
"""Shot noise suppression for raw SLIM data.

Provides multiple denoising methods:
  1. DeepCAD (3D U-Net, deep learning) — best for temporal stacks
  2. Non-local Means (NLM) — good quality, no training needed
  3. BM3D-style bilateral temporal filtering — fast, edge-preserving
  4. Temporal averaging — simplest baseline

All functions accept (H, W, T) float arrays and return denoised arrays.
"""

import numpy as np
import sys
import os


# ── DeepCAD integration ───────────────────────────────────────
DEEPCAD_PATH = r"D:\NIR_SLIM\DeepCAD_RT_pytorch"

# Default pretrained models
DEEPCAD_MODELS = {
    'default': ('record_24092025_170203', 'E_05_Iter_3120.pth'),
    '30hz': ('record_24092025_152254_30hz', 'E_05_Iter_1290.pth'),
    'fast': ('record_24092025_152254', 'E_05_Iter_1290.pth'),
}


def load_deepcad_model(model_name='default', fmap=16, gpu='0', deepcad_path=None):
    """Load a DeepCAD 3D U-Net ONCE and return (model, device, pth_path).

    Pass the returned (model, device) to denoise_deepcad() to reuse it across many
    files (BATCH denoising) without reloading the weights for every file."""
    import torch

    dc_path = deepcad_path or DEEPCAD_PATH
    if dc_path not in sys.path:
        sys.path.insert(0, dc_path)

    from deepcad.network import Network_3D_Unet

    # Resolve model path
    if model_name in DEEPCAD_MODELS:
        model_dir, model_file = DEEPCAD_MODELS[model_name]
        pth_path = os.path.join(dc_path, 'pth', model_dir, model_file)
    elif os.path.isfile(model_name):
        pth_path = model_name
    else:
        raise FileNotFoundError(f"Model not found: {model_name}")

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model weights not found: {pth_path}")

    print(f"DeepCAD: loading model from {pth_path}")

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = Network_3D_Unet(in_channels=1, out_channels=1,
                             f_maps=fmap, final_sigmoid=True)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device, pth_path


def denoise_deepcad(stack, model_name='default', patch_xy=150, patch_t=150,
                     overlap=0.4, fmap=16, gpu='0', deepcad_path=None,
                     progress_callback=None, model=None, device=None):
    """Denoise a (H, W, T) stack using DeepCAD 3D U-Net.

    Args:
        stack: (H, W, T) float array, raw data.
        model_name: Key in DEEPCAD_MODELS or path to .pth file.
        patch_xy: Spatial patch size.
        patch_t: Temporal patch size.
        overlap: Overlap factor between patches.
        fmap: Feature map count (must match trained model).
        gpu: GPU index string.
        deepcad_path: Path to DeepCAD repo (default: DEEPCAD_PATH).

    Returns:
        Denoised (H, W, T) float array.
    """
    import torch

    # Load the model only if one wasn't supplied. Batch denoising calls
    # load_deepcad_model() ONCE and passes (model, device) here, so the weights
    # are NOT reloaded for every file (single-file callers pass nothing -> same
    # behavior as before).
    if model is None:
        model, device, _ = load_deepcad_model(
            model_name, fmap=fmap, gpu=gpu, deepcad_path=deepcad_path)

    H, W, T = stack.shape
    # Normalize to [0, 1]
    smin, smax = stack.min(), stack.max()
    if smax > smin:
        norm_stack = (stack - smin) / (smax - smin)
    else:
        norm_stack = stack.copy()

    # Process in temporal chunks with overlap
    pt = min(patch_t, T)
    stride_t = max(1, int(pt * (1 - overlap)))
    output = np.zeros_like(norm_stack)
    count = np.zeros(T, dtype=np.float32)

    # Spatial: process full frame (pad if needed)
    pad_h = (patch_xy - H % patch_xy) % patch_xy if H > patch_xy else patch_xy - H
    pad_w = (patch_xy - W % patch_xy) % patch_xy if W > patch_xy else patch_xy - W

    # Calculate total chunks for progress
    total_chunks = len(range(0, T, stride_t))
    chunk_idx = 0

    for t_start in range(0, T, stride_t):
        t_end = min(t_start + pt, T)
        if t_end - t_start < 2:
            continue

        chunk = norm_stack[:, :, t_start:t_end]  # (H, W, chunk_t)
        ct = chunk.shape[2]

        # Pad temporally if needed
        if ct < pt:
            pad_t = pt - ct
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, pad_t)), mode='reflect')

        # Convert to torch: (1, 1, T, H, W)
        tensor = torch.from_numpy(chunk.transpose(2, 0, 1)).float()  # (T, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T, H, W)

        with torch.no_grad():
            try:
                denoised = model(tensor)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"  GPU OOM at chunk t={t_start}, falling back to CPU")
                    torch.cuda.empty_cache()
                    model_cpu = model.cpu()
                    denoised = model_cpu(tensor.cpu())
                    model.to(device)
                else:
                    raise

        # Extract result
        result = denoised.squeeze().cpu().numpy()  # (T, H, W) or (H, W) if T=1
        if result.ndim == 2:
            result = result[np.newaxis, :, :]
        result = result[:ct].transpose(1, 2, 0)  # (H, W, ct)

        output[:, :, t_start:t_end] += result
        count[t_start:t_end] += 1.0
        chunk_idx += 1

        pct = int(chunk_idx / total_chunks * 100)
        print(f"  DeepCAD: processed frames {t_start}-{t_end-1}/{T} ({pct}%)")
        if progress_callback:
            progress_callback(pct, chunk_idx, total_chunks,
                              f"Frames {t_start}-{t_end-1}/{T}")

    # Average overlapping regions
    for t in range(T):
        if count[t] > 0:
            output[:, :, t] /= count[t]

    # Rescale back
    if smax > smin:
        output = output * (smax - smin) + smin

    print(f"DeepCAD: done ({T} frames)")
    return output


# ── Non-local Means (OpenCV) ──────────────────────────────────
def denoise_nlm(stack, h=10, template_size=7, search_size=21,
                progress_callback=None):
    """Denoise using Non-Local Means (per-frame, OpenCV)."""
    import cv2
    H, W, T = stack.shape
    output = np.empty_like(stack)

    smin, smax = stack.min(), stack.max()
    for t in range(T):
        frame = stack[:, :, t]
        if smax > smin:
            u8 = ((frame - smin) / (smax - smin) * 255).astype(np.uint8)
        else:
            u8 = np.zeros((H, W), dtype=np.uint8)

        denoised = cv2.fastNlMeansDenoising(u8, None, h=h,
                                              templateWindowSize=template_size,
                                              searchWindowSize=search_size)
        output[:, :, t] = denoised.astype(np.float64) / 255.0 * (smax - smin) + smin

        if t % 10 == 0:
            pct = int((t + 1) / T * 100)
            print(f"  NLM: {t+1}/{T} frames ({pct}%)")
            if progress_callback:
                progress_callback(pct, t + 1, T, f"NLM frame {t+1}/{T}")

    print(f"NLM: done ({T} frames)")
    return output


# ── Bilateral temporal filter ─────────────────────────────────
def denoise_bilateral_temporal(stack, d=5, sigma_color=50, sigma_space=50,
                                temporal_window=3, progress_callback=None):
    """Denoise using bilateral filter + temporal averaging."""
    import cv2
    H, W, T = stack.shape

    smin, smax = stack.min(), stack.max()
    if smax <= smin:
        return stack.copy()

    norm = ((stack - smin) / (smax - smin) * 255).astype(np.uint8)

    filtered = np.empty_like(norm)
    for t in range(T):
        filtered[:, :, t] = cv2.bilateralFilter(norm[:, :, t], d,
                                                  sigma_color, sigma_space)
        if t % 10 == 0:
            pct = int((t + 1) / T * 50)  # first half = bilateral
            print(f"  Bilateral: {t+1}/{T} ({pct}%)")
            if progress_callback:
                progress_callback(pct, t + 1, T, f"Bilateral frame {t+1}/{T}")

    if temporal_window > 1:
        hw = temporal_window // 2
        output = np.empty_like(filtered, dtype=np.float64)
        for t in range(T):
            t0 = max(0, t - hw)
            t1 = min(T, t + hw + 1)
            output[:, :, t] = np.mean(filtered[:, :, t0:t1].astype(np.float64), axis=2)
            if t % 10 == 0:
                pct = 50 + int((t + 1) / T * 50)  # second half = temporal
                if progress_callback:
                    progress_callback(pct, t + 1, T, f"Temporal avg {t+1}/{T}")
        filtered = output.astype(np.uint8)

    result = filtered.astype(np.float64) / 255.0 * (smax - smin) + smin
    if progress_callback:
        progress_callback(100, T, T, "Done")
    print(f"Bilateral temporal: done ({T} frames)")
    return result


# ── Simple temporal averaging ─────────────────────────────────
def denoise_temporal_avg(stack, window=5):
    """Denoise by averaging neighboring frames (sliding window).

    Args:
        stack: (H, W, T) float array.
        window: Number of frames to average.

    Returns:
        Denoised (H, W, T) float array.
    """
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(stack, size=window, axis=2)


# ── Convenience: denoise with method selection ────────────────
def denoise(stack, method='bilateral', **kwargs):
    """Denoise a (H, W, T) stack with the selected method.

    Args:
        stack: (H, W, T) float array.
        method: 'deepcad', 'nlm', 'bilateral', 'temporal'.
        **kwargs: Passed to the selected method.

    Returns:
        Denoised (H, W, T) float array.
    """
    methods = {
        'deepcad': denoise_deepcad,
        'nlm': denoise_nlm,
        'bilateral': denoise_bilateral_temporal,
        'temporal': denoise_temporal_avg,
    }
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
    return methods[method](stack, **kwargs)
