# NIR-II SLIM - zebrafish heart chamber volumes & strain - produces Fig. 2c-h
# environment: heart_valve_py314
# CONFIG fps / pixel_size / z_spacing set to the values reported in the paper on 2026-09-20 (see MANIFEST.md)
# copied from D:\NIR2SLIM\NIR-II-SLIM-code\SimpleElastix\Analyze_heart_ver6.py on 2026-09-17
"""
Zebrafish Heart Full Analysis - SimpleElastix version

기존 analyze_heart.py와의 차이:
  - 프레임 간 wall 변형을 SimpleElastix B-spline 비강체 등록으로 추정
  - 등록에서 얻은 displacement field로 contour 점을 warp하여
    실제 point correspondence 기반 strain rate / WSS 계산
    (기존 arc-length 재샘플링 방식은 점 대응이 보장되지 않았음)
  - 등록은 (t-1, t) 쌍당 1회만 수행하고 결과를 overlay 생성과
    대표 contour 추출에 공유 (중복 등록 제거)

ver5 추가 (compare_heart_analysis 스타일 strain rate 추출):
  - strain rate 부호 유지: 국소 신장(+) / 수축(-), 단위 s^-1
    (WSS는 |SR| 기반이므로 ver4와 수치 동일 — 기존 출력 불변)
  - compute_wss_timeseries 결과 dict에 'red_sr'/'blue_sr' 저장
  - Z별 평균 strain rate heatmap (compare_heart와 동일 스타일/컬러맵)
  - 점별 원본 CSV + heatmap 원본(점 평균) CSV 저장

ver6 추가 (두 샘플 자동 순차 실행):
  - SAMPLES 목록의 WT/WEA 두 스택을 순서대로 자동 분석
    (compare_heart_analysis_ver5에서 쓰던 두 TIF 스택)
  - 촬영 조건을 compare_heart_analysis_ver5와 동일하게 변경:
    z_depths 5, fps 400, pixel_size 15.0, z_spacing 30.0
  - 기존 __main__ 본문을 run_sample(cfg)로 함수화, 샘플별 출력 폴더 분리
  - 두 샘플 공통 대칭 스케일의 비교 heatmap 추가 생성

설치: pip install SimpleITK-SimpleElastix
  (일반 SimpleITK에는 ElastixImageFilter가 없음.
   일반 SimpleITK가 이미 있으면 먼저 pip uninstall SimpleITK)
"""

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')  # GUI 없이 렌더링
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure, morphology
from scipy import ndimage, interpolate
from io import BytesIO
from PIL import Image
import os
import time
import warnings
warnings.filterwarnings('ignore')  # Warning 무시

# --- SimpleElastix ---
try:
    import SimpleITK as sitk
except ImportError:
    raise ImportError(
        "SimpleITK를 찾을 수 없습니다. 설치: pip install SimpleITK-SimpleElastix")

if not hasattr(sitk, 'ElastixImageFilter'):
    raise ImportError(
        "일반 SimpleITK가 설치되어 있습니다 (ElastixImageFilter 없음).\n"
        "  pip uninstall SimpleITK\n"
        "  pip install SimpleITK-SimpleElastix")

# ============================================
# 설정값
# ============================================
CONFIG = {
    # filepath / output_folder는 아래 SAMPLES에서 샘플별로 지정됩니다
    'z_depths': 5,                # 촬영 조건: compare_heart_analysis_ver5와 동일
    'fps': 600,                   # acquisition rate, volumes/s (was 400 in the copied file)
    'pixel_size': 5.616,          # µm/pixel on the 256-px segmentation grid (was 15.0; see CALIBRATION.md)
    'z_spacing': 20.0,            # µm between reconstructed planes (was 30.0)
    'viscosity': 0.003,

    # 테스트용: 각 Z에서 앞 N프레임만 사용 (None이면 전체)
    # 처음 돌릴 때 30~50으로 먼저 확인 권장 (등록이 오래 걸림)
    'test_frames': None,

    # --- SimpleElastix B-spline 등록 설정 ---
    'elx_grid_spacing': 24.0,     # B-spline 제어점 최종 간격 (px)
    'elx_resolutions': 3,         # multi-resolution 단계 수
    'elx_iterations': 150,        # 해상도당 최대 반복
    'elx_samples': 2048,          # metric 샘플 수
    'label_smooth_sigma': 2.0,    # 등록용 label 이미지 Gaussian smoothing (px)

    # --- Strain rate 추출 표시 설정 (compare_heart 스타일) ---
    'strain_unit': 'per_s',       # 'per_s' | 'percent_per_s' | 'percent'
    'red_label': 'Atrium',        # red wall 표시 이름 (compare_heart: red=atrium)
    'blue_label': 'Ventricle',    # blue wall 표시 이름 (blue=ventricle)
}

# ============================================
# 자동 순차 실행 샘플 목록 (compare_heart_analysis_ver5와 동일한 두 스택)
# ============================================
SAMPLES = [
    {'label': 'WT',
     'filepath': 'segmentation_4D_RGB_0902_5stack.tif',
     'output_folder': 'analysis_results_elastix_WT'},
    {'label': 'WEA',
     'filepath': 'segmentation_4D_RGB_0923_swapped.tif',
     'output_folder': 'analysis_results_elastix_WEA'},
]

# 두 샘플 공통 스케일 비교 heatmap 저장 폴더
COMPARISON_FOLDER = 'analysis_results_elastix_comparison'

# ============================================
# Strain 단위 정의 및 컬러맵 (compare_heart_analysis와 동일)
# ============================================
# calculate_point_strain_rate가 계산하는 값은 strain RATE 입니다:
#     strain_rate = (L_warp - L_prev) / (L_prev * dt)   [부호 있음, s^-1]
# 픽셀 크기는 분자/분모에서 상쇄되므로 단위에 영향을 주지 않습니다.
#   'per_s'         : s^-1   — strain rate imaging 표준 표기 (기본값)
#   'percent_per_s' : %/s    — 위 값 x100
#   'percent'       : %      — strain_rate x dt = 프레임 간 증분 strain
#                              (누적 strain 아님에 주의)
STRAIN_UNIT_DEFS = {
    'per_s': {
        'scale': lambda fps: 1.0,
        'unit': r's$^{-1}$',
        'name': 'Strain rate',
    },
    'percent_per_s': {
        'scale': lambda fps: 100.0,
        'unit': r'%$\cdot$s$^{-1}$',
        'name': 'Strain rate',
    },
    'percent': {
        'scale': lambda fps: 100.0 / fps,
        'unit': r'%',
        'name': 'Incremental strain',
    },
}

if CONFIG['strain_unit'] not in STRAIN_UNIT_DEFS:
    raise ValueError(
        f"CONFIG['strain_unit']는 {list(STRAIN_UNIT_DEFS)} 중 하나여야 합니다 "
        f"(입력값: {CONFIG['strain_unit']!r})")

_STRAIN_DEF = STRAIN_UNIT_DEFS[CONFIG['strain_unit']]
STRAIN_SCALE = _STRAIN_DEF['scale'](CONFIG['fps'])   # raw(s^-1) -> 표시 단위
STRAIN_UNIT = _STRAIN_DEF['unit']
STRAIN_NAME = _STRAIN_DEF['name']
STRAIN_LABEL = f'{STRAIN_NAME} ({STRAIN_UNIT})'

def create_custom_colormaps():
    """compare_heart_analysis의 고대비 strain 컬러맵과 동일 (균등 배치)"""
    atrium_colors = ['#006400', '#228B22', '#32CD32', '#ADFF2F', '#FFFFE0',
                     '#FFD700', '#FF6347', '#DC143C', '#8B0000']
    ventricle_colors = ['#00008B', '#0000CD', '#4169E1', '#87CEEB', '#F5F5F5',
                        '#FFA07A', '#FF6347', '#DC143C', '#8B0000']
    atrium_cmap = LinearSegmentedColormap.from_list('atrium_strain',
                                                    atrium_colors)
    ventricle_cmap = LinearSegmentedColormap.from_list('ventricle_strain',
                                                       ventricle_colors)
    return atrium_cmap, ventricle_cmap

ATRIUM_CMAP, VENTRICLE_CMAP = create_custom_colormaps()

# ============================================
# 1. TIF 로드 및 분할
# ============================================
def load_and_split_by_z(filepath, n_z_depths):
    stack = tifffile.imread(filepath)
    total_frames = stack.shape[0]
    frames_per_z = total_frames // n_z_depths

    print(f"TIF 로드: {stack.shape}")
    print(f"Z-depth당 프레임: {frames_per_z}")

    z_stacks = {}
    for z in range(n_z_depths):
        start = z * frames_per_z
        end = start + frames_per_z
        z_stacks[z] = stack[start:end]
        print(f"  Z{z}: frame {start+1} ~ {end}")

    return z_stacks, frames_per_z

# ============================================
# 2. Red/Blue Mask, Contour 및 Area 추출
# ============================================
def extract_contours_and_area(frame):
    """RGB 이미지에서 Red/Blue mask, contour, 면적 추출"""

    if len(frame.shape) < 3:
        return None, None, 0, 0, None, None

    red = frame[:, :, 0].astype(float)
    green = frame[:, :, 1].astype(float)
    blue = frame[:, :, 2].astype(float)

    red_mask = (red > 100) & (red > green + 30) & (red > blue + 30)
    blue_mask = (blue > 100) & (blue > red + 30) & (blue > green + 30)

    # 작은 객체 제거
    red_mask = morphology.remove_small_objects(red_mask.astype(bool), min_size=51)
    blue_mask = morphology.remove_small_objects(blue_mask.astype(bool), min_size=51)
    red_mask = ndimage.binary_fill_holes(red_mask)
    blue_mask = ndimage.binary_fill_holes(blue_mask)

    # 면적 계산
    red_area = np.sum(red_mask)
    blue_area = np.sum(blue_mask)

    # Contour 추출
    red_contours = measure.find_contours(red_mask.astype(float), 0.5)
    blue_contours = measure.find_contours(blue_mask.astype(float), 0.5)

    red_contour = max(red_contours, key=len) if red_contours else None
    blue_contour = max(blue_contours, key=len) if blue_contours else None

    return red_contour, blue_contour, red_area, blue_area, red_mask, blue_mask

# ============================================
# 3. Contour 정규화
# ============================================
def normalize_contour(contour, n_points=100):
    if contour is None or len(contour) < 3:
        return None

    diff = np.diff(contour, axis=0)
    dist = np.sqrt((diff**2).sum(axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(dist)])

    total_length = cum_dist[-1]
    if total_length == 0:
        return None

    uniform_dist = np.linspace(0, total_length, n_points)

    interp_y = interpolate.interp1d(cum_dist, contour[:, 0], kind='linear')
    interp_x = interpolate.interp1d(cum_dist, contour[:, 1], kind='linear')

    return np.column_stack([interp_y(uniform_dist), interp_x(uniform_dist)])

# ============================================
# 4. SimpleElastix 등록 및 displacement field
# ============================================
def masks_to_label_image(red_mask, blue_mask, sigma):
    """등록용 label 이미지: red=1, blue=2 를 Gaussian smoothing 후 sitk 이미지로"""
    label = np.zeros(red_mask.shape, dtype=np.float32) if red_mask is not None \
        else None
    if label is None:
        return None
    label += red_mask.astype(np.float32) * 1.0
    label += blue_mask.astype(np.float32) * 2.0
    label = ndimage.gaussian_filter(label, sigma=sigma)
    return sitk.GetImageFromArray(label)

def register_bspline(fixed_img, moving_img, cfg):
    """B-spline 비강체 등록. fixed=frame(t-1), moving=frame(t).
    반환: transform parameter map (elastix)"""
    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(fixed_img)
    elx.SetMovingImage(moving_img)

    pmap = sitk.GetDefaultParameterMap(
        'bspline',
        cfg['elx_resolutions'],
        cfg['elx_grid_spacing'])
    pmap['MaximumNumberOfIterations'] = [str(cfg['elx_iterations'])]
    pmap['NumberOfSpatialSamples'] = [str(cfg['elx_samples'])]
    pmap['WriteResultImage'] = ['false']

    elx.SetParameterMap(pmap)
    elx.LogToConsoleOff()
    elx.LogToFileOff()
    elx.Execute()
    return elx.GetTransformParameterMap()

def get_displacement_field(transform_pmap, moving_img):
    """transformix로 dense displacement field 계산.
    반환: (H, W, 2) numpy array. [..., 0]=dx(col 방향), [..., 1]=dy(row 방향).
    field는 fixed 좌표 x에서 T(x) - x, 즉 fixed(t-1) → moving(t) 이동량."""
    tfx = sitk.TransformixImageFilter()
    tfx.SetTransformParameterMap(transform_pmap)
    tfx.SetMovingImage(moving_img)
    tfx.ComputeDeformationFieldOn()
    tfx.LogToConsoleOff()
    tfx.LogToFileOff()
    tfx.Execute()
    field = tfx.GetDeformationField()
    return sitk.GetArrayFromImage(field).astype(np.float64)

def warp_contour(contour_rc, disp):
    """(row, col) contour 점들을 displacement field로 warp.
    frame(t-1) 좌표 → frame(t) 좌표"""
    if contour_rc is None:
        return None
    rows = contour_rc[:, 0]
    cols = contour_rc[:, 1]
    dx = ndimage.map_coordinates(disp[:, :, 0], [rows, cols],
                                 order=1, mode='nearest')
    dy = ndimage.map_coordinates(disp[:, :, 1], [rows, cols],
                                 order=1, mode='nearest')
    return np.column_stack([rows + dy, cols + dx])

# ============================================
# 5. Strain Rate 및 WSS 계산 (point correspondence 기반)
# ============================================
def calculate_point_strain_rate(contour_t0, contour_t1, dt):
    """contour_t1은 contour_t0을 warp한 것이므로 점 i끼리 실제 대응됨.
    반환: 부호 있는 strain rate (s^-1). +는 국소 신장, -는 국소 수축."""
    if contour_t0 is None or contour_t1 is None:
        return None

    def local_lengths(contour):
        diff = np.diff(contour, axis=0)
        lengths = np.sqrt((diff**2).sum(axis=1))
        return np.concatenate([lengths, [lengths[0]]])

    len_t0 = local_lengths(contour_t0)
    len_t1 = local_lengths(contour_t1)

    strain_rate = (len_t1 - len_t0) / (len_t0 * dt + 1e-10)
    return strain_rate  # 부호 유지 (ver5)

def calculate_point_wss(strain_rate, viscosity):
    if strain_rate is None:
        return None
    return viscosity * np.abs(strain_rate) * 1000  # mPa (크기 — ver4와 동일 값)

# ============================================
# 6. Z-stack 전체에 대한 등록 기반 WSS 시계열 계산
# ============================================
def compute_wss_timeseries(z_stack, z_index, cfg, n_points=100):
    """각 (t-1, t) 쌍에 대해:
      1) red+blue label 이미지 B-spline 등록 (쌍당 1회)
      2) displacement field로 frame(t-1) contour를 frame(t)로 warp
      3) 대응점 기반 strain rate → WSS
    반환: 프레임별 dict 리스트
      {'red': contour, 'blue': contour, 'red_wss': arr, 'blue_wss': arr,
       'red_sr': arr, 'blue_sr': arr}  # sr: 부호 있는 strain rate (s^-1)
      frame 0은 WSS 없음. contour는 warp된 좌표 (frame t 기준)."""

    n_frames = z_stack.shape[0]
    dt = 1.0 / cfg['fps']

    print(f"\n  Z{z_index}: B-spline 등록 + WSS 계산... "
          f"({n_frames - 1} pairs)")

    results = []

    rc, bc, _, _, rm, bm = extract_contours_and_area(z_stack[0])
    prev_red = normalize_contour(rc, n_points)
    prev_blue = normalize_contour(bc, n_points)
    prev_label = masks_to_label_image(rm, bm, cfg['label_smooth_sigma'])

    results.append({'red': prev_red, 'blue': prev_blue,
                    'red_wss': None, 'blue_wss': None,
                    'red_sr': None, 'blue_sr': None})

    t0 = time.time()

    for i in range(1, n_frames):
        rc, bc, _, _, rm, bm = extract_contours_and_area(z_stack[i])
        curr_red = normalize_contour(rc, n_points)
        curr_blue = normalize_contour(bc, n_points)
        curr_label = masks_to_label_image(rm, bm, cfg['label_smooth_sigma'])

        red_warp = blue_warp = None
        red_wss = blue_wss = None
        red_sr = blue_sr = None

        try:
            tp = register_bspline(prev_label, curr_label, cfg)
            disp = get_displacement_field(tp, curr_label)

            red_warp = warp_contour(prev_red, disp)
            blue_warp = warp_contour(prev_blue, disp)

            red_sr = calculate_point_strain_rate(prev_red, red_warp, dt)
            blue_sr = calculate_point_strain_rate(prev_blue, blue_warp, dt)
            red_wss = calculate_point_wss(red_sr, cfg['viscosity'])
            blue_wss = calculate_point_wss(blue_sr, cfg['viscosity'])
        except Exception as e:
            print(f"    [경고] Frame {i} 등록 실패: {e}")

        # 표시할 contour: warp 성공 시 warp된 것, 실패 시 해당 프레임 자체 contour
        results.append({
            'red': red_warp if red_warp is not None else curr_red,
            'blue': blue_warp if blue_warp is not None else curr_blue,
            'red_wss': red_wss,
            'blue_wss': blue_wss,
            'red_sr': red_sr,
            'blue_sr': blue_sr,
        })

        prev_red = curr_red
        prev_blue = curr_blue
        prev_label = curr_label

        if i % 25 == 0 or i == n_frames - 1:
            elapsed = time.time() - t0
            eta = elapsed / i * (n_frames - 1 - i)
            print(f"    {i}/{n_frames-1} pairs "
                  f"({100*i/(n_frames-1):.0f}%) | "
                  f"경과 {elapsed:.0f}s | 남은 예상 {eta:.0f}s")

    return results

# ============================================
# 6b. Strain Rate 추출/저장/플롯 (compare_heart 스타일)
# ============================================
def to_display_units(strain_array):
    """Raw strain rate (s^-1) -> CONFIG['strain_unit'] 표시 단위 변환"""
    return np.asarray(strain_array, dtype=float) * STRAIN_SCALE

def build_strain_matrices(all_wss_timeseries, cfg, n_points=100):
    """compute_wss_timeseries 결과에서 compare_heart의 z_data와 동일한 형태로
    strain rate 행렬 구성.

    반환: {z: {'times': (n_frames-1,),
               'red_strain':  (n_frames-1, n_points),
               'blue_strain': (n_frames-1, n_points)}}   # 부호 있음, s^-1
    frame 0은 이전 프레임이 없어 제외 (compare_heart와 동일 규약).
    등록 실패 프레임은 NaN 행. red=atrium, blue=ventricle 대응:
    compare_heart의 'atrium_strain'/'ventricle_strain'과 같은 구조."""
    fps = cfg['fps']
    strain_data = {}
    for z, frames in all_wss_timeseries.items():
        red_rows, blue_rows, times = [], [], []
        for i, f in enumerate(frames[1:], start=1):
            red_rows.append(f['red_sr'] if f['red_sr'] is not None
                            else np.full(n_points, np.nan))
            blue_rows.append(f['blue_sr'] if f['blue_sr'] is not None
                             else np.full(n_points, np.nan))
            times.append(i / fps)
        strain_data[z] = {
            'times': np.array(times),
            'red_strain': np.array(red_rows),
            'blue_strain': np.array(blue_rows),
        }
    return strain_data

def save_strain_csvs(strain_data, output_folder):
    """원본 strain rate (항상 s^-1, 부호 있음) CSV 저장.
      - z{z}_red_strain_rate.csv / z{z}_blue_strain_rate.csv:
          time_s + contour 점별 열 (p000..p099)
      - strain_rate_mean_by_z.csv:
          프레임별 점 평균 (heatmap의 원본 데이터, 열: red_z0.., blue_z0..)"""
    n_z = len(strain_data)
    for z, d in strain_data.items():
        n_points = d['red_strain'].shape[1]
        header = 'time_s,' + ','.join(f'p{i:03d}' for i in range(n_points))
        for wall in ('red', 'blue'):
            arr = np.column_stack([d['times'], d[f'{wall}_strain']])
            path = os.path.join(output_folder, f'z{z}_{wall}_strain_rate.csv')
            np.savetxt(path, arr, delimiter=',', header=header, comments='',
                       fmt='%.6g')

    times = strain_data[0]['times']
    cols, names = [times], ['time_s']
    for wall in ('red', 'blue'):
        for z in range(n_z):
            cols.append(np.nanmean(strain_data[z][f'{wall}_strain'], axis=1))
            names.append(f'{wall}_z{z}')
    path = os.path.join(output_folder, 'strain_rate_mean_by_z.csv')
    np.savetxt(path, np.column_stack(cols), delimiter=',',
               header=','.join(names), comments='', fmt='%.6g')
    print(f"  Strain CSV 저장: z별 점별 원본 {n_z * 2}개 "
          f"+ strain_rate_mean_by_z.csv (단위 s^-1, 부호 있음)")

def compute_strain_limits(strain_data, plo=5, phi=95):
    """0 중심 대칭 컬러 스케일 (표시 단위 기준) — compare_heart와 동일.
    strain_data: 단일 dict 또는 dict 리스트 (여러 샘플 공통 스케일 계산용)"""
    if isinstance(strain_data, dict):
        strain_data = [strain_data]
    chunks = []
    for sd in strain_data:
        for d in sd.values():
            chunks.append(np.asarray(d['red_strain']).ravel())
            chunks.append(np.asarray(d['blue_strain']).ravel())
    values = to_display_units(np.concatenate(chunks))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    vlo = np.percentile(values, plo)
    vhi = np.percentile(values, phi)
    vabs = max(abs(vlo), abs(vhi))
    if vabs == 0:
        vabs = 1.0
    return -vabs, vabs

def enhance_strain_data(strain_matrix):
    """행 내 NaN 보간 + 표시 단위 변환 (compare_heart와 동일.
    저장된 원본은 항상 s^-1로 유지 — 표시용 변환)"""
    data = np.asarray(strain_matrix, dtype=float).copy()
    for i in range(data.shape[0]):
        row = data[i]
        nan_mask = np.isnan(row)
        if nan_mask.any() and not nan_mask.all():
            valid_idx = np.where(~nan_mask)[0]
            nan_idx = np.where(nan_mask)[0]
            if len(valid_idx) > 1:
                data[i, nan_idx] = np.interp(nan_idx, valid_idx, row[valid_idx])
    return to_display_units(data)

def style_heatmap_axis(ax):
    """Heatmap용 축 설정 (compare_heart와 동일)"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3.0)
        spine.set_color('#333333')
    ax.tick_params(axis='both', which='major', labelsize=22, width=2.0, length=8)

def add_strain_colorbar(im, ax, shrink=0.8, pad=0.05, aspect=12,
                        label_fontsize=21):
    """단위가 표기된 strain 컬러바 (compare_heart와 동일)"""
    cbar = plt.colorbar(im, ax=ax, shrink=shrink, pad=pad, aspect=aspect)
    cbar.set_label(STRAIN_LABEL, fontsize=label_fontsize,
                   fontweight='medium', labelpad=8)
    cbar.ax.tick_params(labelsize=20, width=1.5, length=6)
    cbar.outline.set_linewidth(2.0)
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    return cbar

def plot_strain_heatmaps(strain_data, output_folder, n_z_depths,
                         vlim=None, title_prefix=''):
    """compare_heart의 strain heatmap과 동일 스타일 (단일 샘플: red/blue 2행).
    y = Z-depth, x = 시간, 값 = contour 점 평균 strain rate (부호 있음).
    0 중심 대칭 5-95 percentile 스케일, gaussian interpolation.
    vlim=(vmin, vmax) 지정 시 그 스케일 사용, title_prefix는 제목 앞에 붙음"""
    if vlim is not None:
        vmin, vmax = vlim
    else:
        vmin, vmax = compute_strain_limits(strain_data, plo=5, phi=95)
    times = strain_data[0]['times']

    wall_configs = [
        ('red', CONFIG['red_label'], ATRIUM_CMAP),
        ('blue', CONFIG['blue_label'], VENTRICLE_CMAP),
    ]

    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica',
                                             'DejaVu Sans'],
                         'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(2, 1, figsize=(16, 7), facecolor='white')

        for ax, (wall, wall_name, cmap) in zip(axes, wall_configs):
            strain_by_z = []
            for z in range(n_z_depths):
                strain_by_z.append(
                    np.nanmean(strain_data[z][f'{wall}_strain'], axis=1))
            strain_matrix = enhance_strain_data(np.array(strain_by_z))

            im = ax.imshow(strain_matrix, aspect='auto', cmap=cmap,
                           extent=[times[0], times[-1],
                                   n_z_depths - 0.5, -0.5],
                           vmin=vmin, vmax=vmax,
                           interpolation='gaussian')
            style_heatmap_axis(ax)
            ax.set_ylabel('Z-depth', fontsize=24, fontweight='medium')
            ax.set_xlabel('Time (s)', fontsize=24, fontweight='medium')
            ax.set_title(f'{title_prefix}{wall_name} {STRAIN_NAME}',
                         fontsize=26,
                         fontweight='bold', pad=10, color='#222222')
            ax.set_yticks(range(n_z_depths))
            ax.set_yticklabels([f'Z{i}' for i in range(n_z_depths)],
                               fontsize=22)
            add_strain_colorbar(im, ax, shrink=0.85, pad=0.02, aspect=12)

        plt.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(output_folder, f'strain_heatmaps.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
    print("  Strain heatmap 저장: strain_heatmaps.png / .svg")

def plot_comparison_strain_heatmaps(all_strain, output_folder, n_z_depths):
    """두 샘플 공통 대칭 스케일 비교 heatmap (compare_heart 왼쪽 열과 동일 구성).
    행: 샘플1-Atrium / 샘플1-Ventricle / 샘플2-Atrium / 샘플2-Ventricle
    all_strain: {label: strain_data}  (build_strain_matrices 결과)
    스케일은 모든 샘플/Z를 아우르는 5-95 percentile 0 중심 대칭."""
    labels = list(all_strain.keys())
    vmin, vmax = compute_strain_limits(list(all_strain.values()),
                                       plo=5, phi=95)

    row_configs = []
    for label in labels:
        row_configs.append((label, 'red', CONFIG['red_label'], ATRIUM_CMAP))
        row_configs.append((label, 'blue', CONFIG['blue_label'],
                            VENTRICLE_CMAP))

    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica',
                                             'DejaVu Sans'],
                         'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(len(row_configs), 1,
                                 figsize=(16, 3.5 * len(row_configs)),
                                 facecolor='white')

        for ax, (label, wall, wall_name, cmap) in zip(axes, row_configs):
            sd = all_strain[label]
            times = sd[0]['times']
            strain_by_z = []
            for z in range(n_z_depths):
                strain_by_z.append(
                    np.nanmean(sd[z][f'{wall}_strain'], axis=1))
            strain_matrix = enhance_strain_data(np.array(strain_by_z))

            im = ax.imshow(strain_matrix, aspect='auto', cmap=cmap,
                           extent=[times[0], times[-1],
                                   n_z_depths - 0.5, -0.5],
                           vmin=vmin, vmax=vmax,
                           interpolation='gaussian')
            style_heatmap_axis(ax)
            ax.set_ylabel('Z-depth', fontsize=24, fontweight='medium')
            ax.set_xlabel('Time (s)', fontsize=24, fontweight='medium')
            ax.set_title(f'{label} - {wall_name} {STRAIN_NAME}',
                         fontsize=26, fontweight='bold', pad=10,
                         color='#222222')
            ax.set_yticks(range(n_z_depths))
            ax.set_yticklabels([f'Z{i}' for i in range(n_z_depths)],
                               fontsize=22)
            add_strain_colorbar(im, ax, shrink=0.85, pad=0.02, aspect=12)

        plt.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(output_folder,
                                     f'strain_heatmaps_comparison.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"  비교 heatmap 저장: {output_folder}/"
          f"strain_heatmaps_comparison.png / .svg")

# ============================================
# 7. WSS Overlay 프레임 생성
# ============================================
def create_wss_overlay_frame(frame, red_contour, blue_contour,
                              red_wss, blue_wss, line_width=4):
    """단일 프레임의 WSS overlay 이미지 생성"""

    h, w = frame.shape[:2]
    dpi = 100
    fig_w = w / dpi
    fig_h = h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(frame)

    # Red contour
    if red_contour is not None and red_wss is not None:
        points = red_contour[:, ::-1].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        wss_seg = (red_wss[:-1] + red_wss[1:]) / 2

        vmin, vmax = np.nanmin(wss_seg), np.nanmax(wss_seg)
        if vmax > vmin:
            norm = plt.Normalize(vmin, vmax)
            lc = LineCollection(segments, cmap='hot', norm=norm, linewidth=line_width)
            lc.set_array(wss_seg)
            ax.add_collection(lc)

    # Blue contour
    if blue_contour is not None and blue_wss is not None:
        points = blue_contour[:, ::-1].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        wss_seg = (blue_wss[:-1] + blue_wss[1:]) / 2

        vmin, vmax = np.nanmin(wss_seg), np.nanmax(wss_seg)
        if vmax > vmin:
            norm = plt.Normalize(vmin, vmax)
            lc = LineCollection(segments, cmap='cool', norm=norm, linewidth=line_width)
            lc.set_array(wss_seg)
            ax.add_collection(lc)

    ax.axis('off')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    fig.tight_layout(pad=0)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    plt.close(fig)

    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    img_pil = Image.fromarray(img_array)
    img_resized = img_pil.resize((w, h), Image.LANCZOS)

    return np.array(img_resized)

# ============================================
# 8. 전체 시간 WSS Overlay TIF 스택 생성
# ============================================
def create_wss_overlay_stack(z_stack, z_index, wss_frames, output_folder):
    """미리 계산된 등록 기반 WSS 시계열로 overlay TIF 스택 생성"""

    n_frames = z_stack.shape[0]
    print(f"\n  Z{z_index}: WSS overlay 스택 생성 중... ({n_frames} frames)")

    overlay_frames = []
    for i in range(n_frames):
        if i % 100 == 0 and i > 0:
            print(f"    Frame {i}/{n_frames} ({100*i/n_frames:.0f}%)")

        d = wss_frames[i]
        overlay = create_wss_overlay_frame(
            z_stack[i], d['red'], d['blue'], d['red_wss'], d['blue_wss'])
        overlay_frames.append(overlay)

    overlay_stack = np.array(overlay_frames)
    output_path = os.path.join(output_folder, f'z{z_index}_wss_overlay.tif')
    tifffile.imwrite(output_path, overlay_stack)

    print(f"    저장: {output_path}")

    return overlay_stack

# ============================================
# 9. 3D Volume 계산 (Stroke Volume)
# ============================================
def calculate_3d_volume(z_stacks, n_z_depths, frames_per_z, pixel_size, z_spacing):
    """각 시점에서 3D Volume 계산 (등록 불필요, 기존과 동일)"""

    print("\n3D Volume 계산 중...")

    red_areas = np.zeros((frames_per_z, n_z_depths))
    blue_areas = np.zeros((frames_per_z, n_z_depths))

    for z in range(n_z_depths):
        print(f"  Z{z} 면적 추출...")
        for t in range(frames_per_z):
            frame = z_stacks[z][t]
            _, _, r_area, b_area, _, _ = extract_contours_and_area(frame)
            red_areas[t, z] = r_area * (pixel_size ** 2)
            blue_areas[t, z] = b_area * (pixel_size ** 2)

    red_volumes = np.sum(red_areas, axis=1) * z_spacing
    blue_volumes = np.sum(blue_areas, axis=1) * z_spacing
    total_volumes = red_volumes + blue_volumes

    times = np.arange(frames_per_z) / CONFIG['fps']

    return times, red_volumes, blue_volumes, total_volumes

def calculate_stroke_volume(volumes):
    """Stroke Volume 계산"""

    edv = np.max(volumes)
    esv = np.min(volumes)
    sv = edv - esv
    ef = (sv / edv) * 100 if edv > 0 else 0

    return {
        'EDV': edv,
        'ESV': esv,
        'SV': sv,
        'EF': ef,
        'EDV_frame': np.argmax(volumes),
        'ESV_frame': np.argmin(volumes),
    }

def plot_volume_analysis(times, red_vol, blue_vol, total_vol, output_folder):
    """Volume 분석 결과 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Volume over time
    axes[0, 0].plot(times, red_vol/1e9, 'r-', label='Red Wall', alpha=0.7)
    axes[0, 0].plot(times, blue_vol/1e9, 'b-', label='Blue Wall', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Volume (×10⁹ µm³)')
    axes[0, 0].set_title('Wall Volume over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Total volume
    axes[0, 1].plot(times, total_vol/1e9, 'purple', alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Volume (×10⁹ µm³)')
    axes[0, 1].set_title('Total Volume over Time')
    axes[0, 1].grid(True, alpha=0.3)

    sv_data = calculate_stroke_volume(total_vol)
    axes[0, 1].axhline(sv_data['EDV']/1e9, color='green', linestyle='--', label=f"EDV: {sv_data['EDV']/1e9:.2f}")
    axes[0, 1].axhline(sv_data['ESV']/1e9, color='orange', linestyle='--', label=f"ESV: {sv_data['ESV']/1e9:.2f}")
    axes[0, 1].legend()

    # Volume change
    dt = times[1] - times[0] if len(times) > 1 else 1/400
    vol_rate = np.gradient(total_vol, dt) / 1e9
    axes[1, 0].plot(times, vol_rate, 'purple', alpha=0.7)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('dV/dt (×10⁹ µm³/s)')
    axes[1, 0].set_title('Volume Rate of Change')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)

    # Stroke Volume 정보
    axes[1, 1].axis('off')
    sv_text = f"""
    ═══════════════════════════════
    STROKE VOLUME ANALYSIS
    ═══════════════════════════════

    End Diastolic Volume (EDV):
        {sv_data['EDV']/1e9:.4f} ×10⁹ µm³
        (Frame {sv_data['EDV_frame']})

    End Systolic Volume (ESV):
        {sv_data['ESV']/1e9:.4f} ×10⁹ µm³
        (Frame {sv_data['ESV_frame']})

    Stroke Volume (SV):
        {sv_data['SV']/1e9:.4f} ×10⁹ µm³

    Ejection Fraction (EF):
        {sv_data['EF']:.1f} %

    ═══════════════════════════════
    """
    axes[1, 1].text(0.1, 0.5, sv_text, transform=axes[1, 1].transAxes,
                    fontsize=12, fontfamily='monospace', verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'volume_analysis.png'), dpi=200)
    plt.close()

    return sv_data

# ============================================
# 10. 3D Surface Visualization
# ============================================
def visualize_contours_3d_surface(all_z_contours, output_folder, n_points=100):
    """Z-depth별 contour를 3D surface로 시각화"""

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    n_z = len(all_z_contours)

    red_x, red_y, red_z = [], [], []
    blue_x, blue_y, blue_z = [], [], []

    for z in range(n_z):
        data = all_z_contours[z]

        if data['red'] is not None:
            contour = normalize_contour(data['red'], n_points)
            if contour is not None:
                red_x.append(contour[:, 1])
                red_y.append(contour[:, 0])
                red_z.append(np.full(n_points, z))

        if data['blue'] is not None:
            contour = normalize_contour(data['blue'], n_points)
            if contour is not None:
                blue_x.append(contour[:, 1])
                blue_y.append(contour[:, 0])
                blue_z.append(np.full(n_points, z))

    if len(red_x) >= 2:
        red_x = np.array(red_x)
        red_y = np.array(red_y)
        red_z = np.array(red_z)
        ax.plot_surface(red_x, red_y, red_z, color='red', alpha=0.6)

    if len(blue_x) >= 2:
        blue_x = np.array(blue_x)
        blue_y = np.array(blue_y)
        blue_z = np.array(blue_z)
        ax.plot_surface(blue_x, blue_y, blue_z, color='blue', alpha=0.6)

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z-depth')
    ax.set_title('3D Heart Wall Surface')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.6, label='Red Wall'),
                       Patch(facecolor='blue', alpha=0.6, label='Blue Wall')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '3d_surface.png'), dpi=200)
    plt.close()

    print(f"3D Surface 저장: 3d_surface.png")

def visualize_3d_surface_with_wss(all_z_contours, output_folder, n_points=100):
    """WSS 값으로 색상 입힌 3D surface"""

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    n_z = len(all_z_contours)

    red_x, red_y, red_z, red_wss = [], [], [], []
    blue_x, blue_y, blue_z, blue_wss = [], [], [], []

    for z in range(n_z):
        data = all_z_contours[z]

        if data['red'] is not None and data['red_wss'] is not None:
            contour = normalize_contour(data['red'], n_points)
            wss = data['red_wss']
            if contour is not None and len(wss) == n_points:
                red_x.append(contour[:, 1])
                red_y.append(contour[:, 0])
                red_z.append(np.full(n_points, z))
                red_wss.append(wss)

        if data['blue'] is not None and data['blue_wss'] is not None:
            contour = normalize_contour(data['blue'], n_points)
            wss = data['blue_wss']
            if contour is not None and len(wss) == n_points:
                blue_x.append(contour[:, 1])
                blue_y.append(contour[:, 0])
                blue_z.append(np.full(n_points, z))
                blue_wss.append(wss)

    if len(red_x) >= 2:
        red_x = np.array(red_x)
        red_y = np.array(red_y)
        red_z = np.array(red_z)
        red_wss = np.array(red_wss)
        red_wss_norm = red_wss / (np.nanmax(red_wss) + 1e-10)
        ax.plot_surface(red_x, red_y, red_z, facecolors=plt.cm.hot(red_wss_norm),
                        alpha=0.8, shade=False)

    if len(blue_x) >= 2:
        blue_x = np.array(blue_x)
        blue_y = np.array(blue_y)
        blue_z = np.array(blue_z)
        blue_wss = np.array(blue_wss)
        blue_wss_norm = blue_wss / (np.nanmax(blue_wss) + 1e-10)
        ax.plot_surface(blue_x, blue_y, blue_z, facecolors=plt.cm.cool(blue_wss_norm),
                        alpha=0.8, shade=False)

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z-depth')
    ax.set_title('3D Heart Wall with WSS')

    sm_red = plt.cm.ScalarMappable(cmap='hot')
    sm_red.set_array([])
    plt.colorbar(sm_red, ax=ax, shrink=0.3, location='left', label='Red WSS (mPa)')

    sm_blue = plt.cm.ScalarMappable(cmap='cool')
    sm_blue.set_array([])
    plt.colorbar(sm_blue, ax=ax, shrink=0.3, location='right', label='Blue WSS (mPa)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '3d_surface_wss.png'), dpi=200)
    plt.close()

    print(f"3D Surface WSS 저장: 3d_surface_wss.png")

# ============================================
# 11. 대표 Contour 추출 (등록 결과 재사용)
# ============================================
def extract_representative_contours(all_wss_timeseries, n_use=200):
    """미리 계산된 등록 기반 WSS 시계열에서 각 Z의 대표 contour와
    평균 WSS 추출 (등록 재수행 없음)"""

    all_z_data = {}

    for z, frames in all_wss_timeseries.items():
        print(f"  Z{z}: 대표 contour 추출...")

        use = frames[1:n_use]

        red_pairs = [(f['red'], f['red_wss']) for f in use
                     if f['red'] is not None and f['red_wss'] is not None]
        blue_pairs = [(f['blue'], f['blue_wss']) for f in use
                      if f['blue'] is not None and f['blue_wss'] is not None]

        mean_red_wss = (np.nanmean([w for _, w in red_pairs], axis=0)
                        if red_pairs else None)
        mean_blue_wss = (np.nanmean([w for _, w in blue_pairs], axis=0)
                         if blue_pairs else None)

        rep_red = red_pairs[len(red_pairs) // 2][0] if red_pairs else None
        rep_blue = blue_pairs[len(blue_pairs) // 2][0] if blue_pairs else None

        all_z_data[z] = {
            'red': rep_red,
            'blue': rep_blue,
            'red_wss': mean_red_wss,
            'blue_wss': mean_blue_wss
        }

    return all_z_data

# ============================================
# 단일 샘플 파이프라인 (기존 __main__ 본문을 함수화)
# ============================================
def run_sample(cfg):

    print("="*60)
    print(f"Sample: {cfg['label']}  ({cfg['filepath']})")
    print("="*60)

    os.makedirs(cfg['output_folder'], exist_ok=True)

    # 1. 로드 & 분할
    print("\n[1/6] TIF 로드 및 Z-depth 분할...")
    z_stacks, frames_per_z = load_and_split_by_z(cfg['filepath'], cfg['z_depths'])

    if cfg['test_frames'] is not None:
        n_test = int(cfg['test_frames'])
        z_stacks = {z: s[:n_test] for z, s in z_stacks.items()}
        frames_per_z = min(frames_per_z, n_test)
        print(f"\n  [테스트 모드] 각 Z 앞 {frames_per_z}프레임만 사용")

    # 2. 등록 기반 WSS 시계열 계산 (Z별, 쌍당 등록 1회)
    print("\n[2/6] SimpleElastix B-spline 등록 + WSS 계산...")
    print(f"  (총 {cfg['z_depths']} × {frames_per_z - 1} = "
          f"{cfg['z_depths'] * (frames_per_z - 1)} 회 등록 — 시간이 오래 걸립니다)")

    all_wss_timeseries = {}
    for z in range(cfg['z_depths']):
        all_wss_timeseries[z] = compute_wss_timeseries(
            z_stacks[z], z, cfg)

    # 3. Strain rate 추출 (부호 있음) — heatmap + 원본 CSV
    print("\n[3/6] Strain rate 추출 (heatmap + CSV)...")
    strain_data = build_strain_matrices(all_wss_timeseries, cfg)
    save_strain_csvs(strain_data, cfg['output_folder'])
    plot_strain_heatmaps(strain_data, cfg['output_folder'],
                         cfg['z_depths'],
                         title_prefix=f"{cfg['label']} - ")

    # 4. WSS Overlay TIF 스택 생성 (각 Z별)
    print("\n[4/6] WSS Overlay TIF 스택 생성...")
    for z in range(cfg['z_depths']):
        create_wss_overlay_stack(
            z_stacks[z], z,
            all_wss_timeseries[z],
            cfg['output_folder']
        )

    # 5. 3D Volume & Stroke Volume 계산
    print("\n[5/6] 3D Volume 및 Stroke Volume 계산...")
    times, red_vol, blue_vol, total_vol = calculate_3d_volume(
        z_stacks,
        cfg['z_depths'],
        frames_per_z,
        cfg['pixel_size'],
        cfg['z_spacing']
    )

    sv_data = plot_volume_analysis(times, red_vol, blue_vol, total_vol, cfg['output_folder'])

    # Stroke Volume 결과 저장
    with open(os.path.join(cfg['output_folder'], 'stroke_volume.txt'), 'w') as f:
        f.write("STROKE VOLUME ANALYSIS\n")
        f.write("="*40 + "\n\n")
        f.write(f"End Diastolic Volume (EDV): {sv_data['EDV']/1e9:.6f} ×10^9 um^3\n")
        f.write(f"End Systolic Volume (ESV):  {sv_data['ESV']/1e9:.6f} ×10^9 um^3\n")
        f.write(f"Stroke Volume (SV):         {sv_data['SV']/1e9:.6f} ×10^9 um^3\n")
        f.write(f"Ejection Fraction (EF):     {sv_data['EF']:.1f} %\n")
        f.write(f"\nEDV at frame: {sv_data['EDV_frame']}\n")
        f.write(f"ESV at frame: {sv_data['ESV_frame']}\n")

    print(f"\n  Stroke Volume: {sv_data['SV']/1e9:.4f} ×10⁹ µm³")
    print(f"  Ejection Fraction: {sv_data['EF']:.1f}%")

    # 6. 대표 Contour 추출 및 3D 시각화 (등록 결과 재사용)
    print("\n[6/6] 3D Surface 시각화...")
    all_z_contours = extract_representative_contours(all_wss_timeseries)

    visualize_contours_3d_surface(all_z_contours, cfg['output_folder'])
    visualize_3d_surface_with_wss(all_z_contours, cfg['output_folder'])

    print(f"\n{cfg['label']} 완료!")
    print("="*60)
    print(f"결과: {os.path.abspath(cfg['output_folder'])}")
    print("="*60)
    print("\n생성된 파일:")
    print("  - z{N}_wss_overlay.tif    : 각 Z의 전체시간 WSS overlay 스택 (등록 기반)")
    print("  - volume_analysis.png     : Volume 분석 그래프")
    print("  - stroke_volume.txt       : Stroke Volume 결과")
    print("  - 3d_surface.png          : 3D Surface")
    print("  - strain_heatmaps.png/.svg: Z별 평균 strain rate heatmap (부호 있음)")
    print("  - z{N}_red/blue_strain_rate.csv : 점별 원본 strain rate (s^-1)")
    print("  - strain_rate_mean_by_z.csv     : 프레임별 점 평균 (heatmap 원본)")
    print("  - 3d_surface_wss.png      : 3D Surface (WSS 색상)")

    return strain_data

# ============================================
# 메인: 두 샘플 자동 순차 실행
# ============================================
if __name__ == "__main__":

    print("="*60)
    print("Zebrafish Heart - Full Analysis (SimpleElastix)")
    print(f"자동 순차 실행: {', '.join(s['label'] for s in SAMPLES)}")
    print("="*60)

    # 긴 등록 도중의 실패를 막기 위해 시작 전에 입력 파일 존재 확인
    missing = [s['filepath'] for s in SAMPLES
               if not os.path.isfile(s['filepath'])]
    if missing:
        raise FileNotFoundError(
            "다음 입력 TIF를 찾을 수 없습니다: " + ", ".join(missing))

    all_strain = {}
    for idx, sample in enumerate(SAMPLES, start=1):
        print(f"\n{'#'*60}")
        print(f"# [{idx}/{len(SAMPLES)}] {sample['label']} 분석")
        print(f"{'#'*60}")
        cfg = {**CONFIG, **sample}
        all_strain[sample['label']] = run_sample(cfg)

    # 두 샘플 공통 대칭 스케일 비교 heatmap (compare_heart 왼쪽 열 구성)
    if len(all_strain) >= 2:
        print("\n두 샘플 공통 스케일 비교 heatmap 생성...")
        os.makedirs(COMPARISON_FOLDER, exist_ok=True)
        plot_comparison_strain_heatmaps(all_strain, COMPARISON_FOLDER,
                                        CONFIG['z_depths'])

    print("\n전체 완료!")
    print("="*60)
    for s in SAMPLES:
        print(f"  {s['label']:>4}: {os.path.abspath(s['output_folder'])}")
    if len(all_strain) >= 2:
        print(f"  비교: {os.path.abspath(COMPARISON_FOLDER)}")
    print("="*60)