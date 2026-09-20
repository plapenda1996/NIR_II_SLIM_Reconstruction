#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NIR-II SLIM - segmentation overlay styling - produces Fig. 2 panels (dashed chamber outlines)
# environment: heart_valve_py314
# copied from E:\Selected data\250902_fish\fill_to_dashed_outline.py on 2026-09-17
"""
segmentation_4D_RGB.tif 의 라벨 표시 방식을 바꾸는 스크립트.

현재  : 선택 영역이 통째로 빨강/파랑으로 칠해져 있음 (solid fill)
변경후: 영역 내부의 채색을 걷어내고(원래 영상 복원),
        영역의 "가장자리만" 빨강/파랑 점선으로 표시.

처리 단위: 8 depth x 901 frame = 7208 프레임을 한 장씩 스트리밍으로
          읽고/쓰기 때문에 스택이 커도 메모리에 한 번에 올리지 않음.

필요 패키지: tifffile, numpy, scipy, scikit-image  (모두 표준 pip 설치 가능)
    pip install tifffile numpy scipy scikit-image
"""

import numpy as np
import tifffile
from skimage import measure, draw
from scipy import ndimage

# ============================================================================
# 설정 (필요에 맞게 수정)
# ============================================================================
IN_PATH   = "segmentation_4D_RGB.tif"          # 입력 파일
OUT_PATH  = "segmentation_4D_RGB_dashed.tif"   # 출력 파일

N_DEPTHS  = 8
N_FRAMES  = 901          # depth 당 프레임 수 (검증용, 실제 페이지 수와 비교만 함)

# --- 색 검출 ---
COLOR_THRESH = 40        # 주(主)채널이 나머지보다 이만큼 크면 채색으로 판정.
                         # 점선이 너무 안 잡히면 낮추고(예 25), 노이즈가 잡히면 올리세요(예 60).

# --- 점선 모양 ---
DASH = 3                 # 점선에서 "켜진" 구간 길이 (px)
GAP  = 2                 # 점선에서 "꺼진" 구간 길이 (px)
THICKNESS = 1            # 선 두께 (px). 굵게 하려면 2~3.

# --- 점선 색 (R,G,B) ---
RED_COLOR  = (255, 0, 0)
BLUE_COLOR = (0, 0, 255)

# --- 내부 채색 제거 방식 ---
STRIP_FILL       = True   # True면 내부 채색을 걷어냄. False면 채색은 그대로 두고 점선만 덧그림.
RESTORE_BRIGHTNESS = True # True면 반투명(alpha) 오버레이의 원래 밝기까지 복원.
                          # (불투명 오버레이는 원본이 소실되어 복원 불가 → 아래 경고 참고)
MAX_ALPHA = 0.92          # 밝기 복원 시 alpha 상한 (불투명 영역에서 0 나눗셈 방지)

# 작은 잡티(노이즈) 영역은 무시: 이 픽셀 수보다 작은 덩어리는 윤곽선을 안 그림
MIN_REGION_AREA = 8
# ============================================================================


def to_hwc(frame):
    """프레임을 (H, W, 3) 형태로 정규화. planar(3,H,W)나 그레이도 처리."""
    a = np.asarray(frame)
    if a.ndim == 2:                      # 그레이 → 3채널 복제
        a = np.repeat(a[..., None], 3, axis=-1)
    elif a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))   # (3,H,W) → (H,W,3)
    if a.shape[-1] > 3:                  # RGBA 등 → 앞 3채널만
        a = a[..., :3]
    if a.dtype != np.uint8:              # 16bit 등 → 8bit로 스케일
        a = (a.astype(np.float32) / a.max() * 255).astype(np.uint8) if a.max() > 0 else a.astype(np.uint8)
    return np.ascontiguousarray(a)


def detect_overlay_masks(img):
    """빨강/파랑 채색 영역 마스크와 '어두워진 원본 그레이' 추정치 반환."""
    R = img[..., 0].astype(np.int16)
    G = img[..., 1].astype(np.int16)
    B = img[..., 2].astype(np.int16)
    red_mask  = (R - np.maximum(G, B)) > COLOR_THRESH
    blue_mask = (B - np.maximum(R, G)) > COLOR_THRESH
    base_min  = np.min(img, axis=-1)     # 반투명 오버레이에서 비주채널 = (1-a)*원본
    return red_mask, blue_mask, base_min


def remove_small(mask):
    if MIN_REGION_AREA <= 1:
        return mask
    lbl, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    keep = np.isin(lbl, np.nonzero(sizes >= MIN_REGION_AREA)[0] + 1)
    return keep


def restore_gray(img, red_mask, blue_mask):
    """순색 alpha 오버레이 아래의 원래 그레이 복원.
       빨강: R=(1-a)g+255a, G=B=(1-a)g  →  a=(R-G)/255, g=G/(1-a)."""
    out = np.min(img, axis=-1).astype(np.float32)   # 기본값: 어두워진 그레이
    Rf, Gf, Bf = (img[..., 0].astype(np.float32),
                  img[..., 1].astype(np.float32),
                  img[..., 2].astype(np.float32))
    if RESTORE_BRIGHTNESS:
        a = np.clip((Rf - Gf) / 255.0, 0, MAX_ALPHA)
        out[red_mask] = (Gf / (1.0 - a))[red_mask]
        a = np.clip((Bf - Gf) / 255.0, 0, MAX_ALPHA)
        out[blue_mask] = (Gf / (1.0 - a))[blue_mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def dashed_contour(binary_mask):
    """마스크 경계를 따라 등간격 점선 픽셀 마스크 생성."""
    out = np.zeros(binary_mask.shape, bool)
    period = DASH + GAP
    for c in measure.find_contours(binary_mask.astype(float), 0.5):
        if len(c) < 2:
            continue
        seg = np.sqrt((np.diff(c, axis=0) ** 2).sum(1))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        on = (s % period) < DASH       # 호(arc) 길이 기준으로 켜짐/꺼짐
        for i in range(len(c) - 1):
            if on[i] and on[i + 1]:
                rr, cc = draw.line(int(round(c[i, 0])),   int(round(c[i, 1])),
                                   int(round(c[i+1, 0])), int(round(c[i+1, 1])))
                out[rr, cc] = True
    return out


def convert_frame(frame):
    img = to_hwc(frame)
    red_mask, blue_mask, base_min = detect_overlay_masks(img)
    red_mask  = remove_small(red_mask)
    blue_mask = remove_small(blue_mask)

    out = img.copy()
    if STRIP_FILL:
        gray = (restore_gray(img, red_mask, blue_mask) if RESTORE_BRIGHTNESS
                else base_min)
        colored = red_mask | blue_mask
        out[colored] = np.stack([gray]*3, axis=-1)[colored]

    rd = dashed_contour(red_mask)
    bd = dashed_contour(blue_mask)
    if THICKNESS > 1:
        k = np.ones((THICKNESS, THICKNESS), bool)
        rd = ndimage.binary_dilation(rd, k)
        bd = ndimage.binary_dilation(bd, k)
    out[rd] = RED_COLOR
    out[bd] = BLUE_COLOR

    # 불투명 fill 자동 점검용 통계 반환
    nondom_in_color = base_min[red_mask | blue_mask]
    return out, (red_mask.sum(), blue_mask.sum()),  \
        (float(np.median(nondom_in_color)) if nondom_in_color.size else None)


def main():
    opaque_warned = False
    with tifffile.TiffFile(IN_PATH) as tif:
        pages = tif.pages
        total = len(pages)
        print(f"입력 페이지 수: {total}  (기대값 {N_DEPTHS}x{N_FRAMES}={N_DEPTHS*N_FRAMES})")
        if total != N_DEPTHS * N_FRAMES:
            print("  ⚠ 페이지 수가 기대값과 다릅니다. 그래도 전체를 그대로 처리합니다.")

        with tifffile.TiffWriter(OUT_PATH, bigtiff=False) as writer:
            for i in range(total):
                frame = pages[i].asarray()
                out, (nr, nb), nondom = convert_frame(frame)

                # 첫 채색 프레임에서 불투명 여부 점검
                if (not opaque_warned) and nondom is not None and (nr + nb) > 0:
                    if nondom < 10:
                        print("\n  ⚠ 채색이 '불투명(opaque)'으로 보입니다. "
                              "이 경우 RGB tif만으로는 영역 내부의 원본 영상을 복원할 수 없습니다.\n"
                              "    내부는 어둡게 표시되며, 깨끗한 결과를 원하면 "
                              "segmentation_data_XYZT_dual.mat 마스크 + 원본 데이터에서 "
                              "다시 렌더링하는 방식이 필요합니다 (요청 주세요).\n")
                    opaque_warned = True

                writer.write(out, photometric="rgb", contiguous=True)

                if i % 200 == 0 or i == total - 1:
                    depth = i // N_FRAMES
                    fr = i % N_FRAMES
                    print(f"  {i+1:>5}/{total}  (depth {depth}, frame {fr})")

    print(f"\n완료 → {OUT_PATH}")


if __name__ == "__main__":
    main()
