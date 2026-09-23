#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NIR-II SLIM - self-supervised denoising (inference) - produces Fig. 3j-m, Supp. Fig. 6d
# environment: heart_valve_py314
"""
DeepCAD 배치 디노이징 스크립트 (순수 Python 버전)
====================================================
MATLAB의 ``deepcad_denoise.m`` 가 하던 일을 Python으로 옮긴 것입니다.
선택한 여러 개의 .tif 스택을 **하나씩 순서대로** DeepCAD 모델로 디노이징합니다.
(DeepCAD는 self-supervised 방식으로 형광 영상의 shot noise를 제거합니다.)

핵심 특징
---------
1. 학습된 모델(.pth)을 **딱 한 번만** 메모리에 올립니다.
   → 파일마다 모델을 다시 로딩하지 않으므로, MATLAB 래퍼처럼 매번
     testing_class 를 새로 만드는 것보다 여러 파일 처리 시 훨씬 빠릅니다.
2. 선택한 파일을 **한 개씩, 정해진 순서대로** 처리하고
   파일별 진행상황 / 소요시간을 출력합니다.
3. DeepCAD 원본의 전처리·추론·스티칭 함수
   (``test_preprocess_chooseOne``, ``testset``,
    ``singlebatch_test_save``, ``multibatch_test_save``)를
   **그대로 재사용**하므로 결과가 원본 파이프라인과 동일합니다.

DeepCAD 프로젝트 구조 (deepcad_path = 프로젝트 루트)
----------------------------------------------------
    DeepCAD_root/
      ├─ deepcad/              <- 파이썬 패키지 (network.py, data_process.py ...)
      ├─ pth/
      │    └─ <denoise_model>/
      │         └─ best_model.pth   (또는 ...pth 파일들)
      └─ deepcad_denoise_batch.py   <- (권장) 이 스크립트를 여기에 두면
                                        --deepcad-path 를 생략할 수 있습니다.

사용법
------
(1) GUI 창으로 파일 선택 — 가장 간단함:
        python deepcad_denoise_batch.py --model mouse_202506142239

(2) 처리할 파일을 인자로 직접 지정 — 적은 순서 그대로 처리됩니다:
        python deepcad_denoise_batch.py --model mouse_202506142239 \
               C:/data/fov1.tif C:/data/fov2.tif C:/data/fov3.tif

(3) 아래 FILES 리스트에 경로를 적어두고 그냥 실행:
        python deepcad_denoise_batch.py --model mouse_202506142239

주요 옵션 (전부 denoise_interface.py 의 기본값과 동일)
    --model            (필수) pth/ 아래 모델 폴더 이름
    --deepcad-path     DeepCAD 프로젝트 루트 (기본: 이 스크립트가 있는 폴더)
    --output-dir       결과 저장 폴더 (기본: ./denoise_results)
    --pth-file         특정 .pth 파일을 직접 지정 (모델 폴더 자동탐색 대신)
    --patch-xy 150  --patch-t 150  --overlap 0.4
    --fmap 16  --scale-factor 1  --test-datasize 100000
    --gpu 0  --num-workers 0
"""

import os
import sys
import time
import shutil
import tempfile
import argparse
from types import SimpleNamespace

import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage import io
from tqdm import tqdm


# ---------------------------------------------------------------------------
# .raw frame spec (NIR-II C-RED 2): headerless (frames, H, W) uint16.
# one frame = 512 * 640 * 2 bytes. Overridable via --raw-height/--raw-width.
# ---------------------------------------------------------------------------
RAW_HEIGHT = 512
RAW_WIDTH = 640
RAW_DTYPE = np.uint16


# ---------------------------------------------------------------------------
# (선택) 여기에 경로를 적어두면 인자 없이 실행해도 이 파일들이 처리됩니다.
# 적은 순서대로 처리됩니다. 비워두면 GUI 창이 뜹니다.
# 예: FILES = [r"C:\data\fov1.tif", r"C:\data\fov2.tif"]
# ---------------------------------------------------------------------------
FILES: list[str] = []


def _raw_to_temp_tif(raw_path, height=None, width=None, dtype=None):
    """Read a headerless .raw stack (frames, H, W) READ-ONLY and stage it as a
    temporary .tif inside its OWN temporary folder (placed next to the .raw so it
    stays on the same volume and never bloats C:). The temp .tif keeps the .raw
    stem so the output name is derived from the original filename.

    Returns (temp_dir, temp_tif_path). The original .raw is never modified.
    Raises a clear ValueError if the file size is not a whole number of frames."""
    height = RAW_HEIGHT if height is None else int(height)
    width = RAW_WIDTH if width is None else int(width)
    dtype = RAW_DTYPE if dtype is None else dtype
    bytes_per_frame = height * width * np.dtype(dtype).itemsize
    file_size = os.path.getsize(raw_path)
    if file_size % bytes_per_frame != 0:
        raise ValueError(
            f"[{os.path.basename(raw_path)}] file size ({file_size} bytes) is not an "
            f"integer multiple of the frame size ({bytes_per_frame} bytes = "
            f"{height}x{width} {np.dtype(dtype).name}). Check height/width/dtype or a "
            f"possible header.")
    total_frames = file_size // bytes_per_frame

    # READ-ONLY memmap of the original (never loads the whole file eagerly)
    mm = np.memmap(raw_path, dtype=dtype, mode="r",
                   shape=(total_frames, height, width))
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    temp_dir = tempfile.mkdtemp(prefix=f"_deepcad_tmp_{stem}_",
                                dir=os.path.dirname(os.path.abspath(raw_path)))
    temp_tif = os.path.join(temp_dir, stem + ".tif")
    # write the (T,H,W) uint16 stack as a plain multi-page TIFF for DeepCAD
    tifffile.imwrite(temp_tif, np.asarray(mm))
    del mm
    print(f"   [.raw] {os.path.basename(raw_path)} -> temp tif "
          f"({total_frames} frames, {height}x{width}, {np.dtype(dtype).name})")
    return temp_dir, temp_tif


def _import_deepcad(deepcad_path):
    """DeepCAD 프로젝트 루트를 sys.path 에 넣고 필요한 함수들을 import 한다."""
    deepcad_path = os.path.abspath(deepcad_path)
    if deepcad_path not in sys.path:
        sys.path.insert(0, deepcad_path)
    try:
        from deepcad.network import Network_3D_Unet
        from deepcad.data_process import (
            test_preprocess_chooseOne,
            testset,
            multibatch_test_save,
            singlebatch_test_save,
        )
    except ImportError as e:
        raise ImportError(
            "DeepCAD 패키지를 import 할 수 없습니다.\n"
            f"  --deepcad-path 가 'deepcad' 폴더를 포함한 프로젝트 루트를 "
            f"가리키는지 확인하세요.\n"
            f"  현재 경로: {deepcad_path}\n"
            f"  원본 오류: {e}"
        )
    return (
        Network_3D_Unet,
        test_preprocess_chooseOne,
        testset,
        multibatch_test_save,
        singlebatch_test_save,
    )


class DeepCADDenoiser:
    """학습된 DeepCAD 모델을 한 번만 로딩해두고, 파일별로 디노이징을 수행한다."""

    def __init__(
        self,
        deepcad_path,
        denoise_model,
        pth_dir="pth",
        pth_file=None,
        fmap=16,
        gpu="0",
        patch_x=150,
        patch_y=150,
        patch_t=150,
        overlap_factor=0.4,
        scale_factor=1,
        test_datasize=100000,
        num_workers=0,
    ):
        # --- DeepCAD 함수 import (프로젝트 경로 등록) ---
        (
            self._Net,
            self._preprocess,
            self._Testset,
            self._multibatch_save,
            self._singlebatch_save,
        ) = _import_deepcad(deepcad_path)

        self.deepcad_path = os.path.abspath(deepcad_path)

        # --- patch / 전처리 파라미터 저장 ---
        self.patch_x = int(patch_x)
        self.patch_y = int(patch_y)
        self.patch_t = int(patch_t)
        self.overlap_factor = float(overlap_factor)
        self.scale_factor = scale_factor
        self.test_datasize = int(test_datasize)
        self.num_workers = int(num_workers)

        # --- GPU 환경설정 (torch CUDA 초기화 전에 해야 함) ---
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ngpu = str(gpu).count(",") + 1
        # 원본 testing_class 와 동일: batch_size = 사용 GPU 개수 (CPU면 1)
        self.batch_size = ngpu if self.device.type == "cuda" else 1

        # --- 사용할 .pth 파일 결정 ---
        if pth_file is None:
            model_dir = os.path.join(self.deepcad_path, pth_dir, denoise_model)
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"모델 폴더가 없습니다: {model_dir}")
            pths = sorted(f for f in os.listdir(model_dir) if f.lower().endswith(".pth"))
            if not pths:
                raise FileNotFoundError(f"'{model_dir}' 안에 .pth 파일이 없습니다.")
            pth_file = os.path.join(model_dir, pths[-1])  # 이름순 마지막(최신) 모델
        if not os.path.isfile(pth_file):
            raise FileNotFoundError(f".pth 파일을 찾을 수 없습니다: {pth_file}")
        self.pth_file = pth_file
        self.model_stem = os.path.splitext(os.path.basename(pth_file))[0]

        # --- 네트워크 생성 + 가중치 로딩 (한 번만) ---
        net = self._Net(
            in_channels=1, out_channels=1, f_maps=int(fmap), final_sigmoid=True
        )
        state = torch.load(pth_file, map_location="cpu")
        # DataParallel 로 저장된 경우 'module.' 접두사가 있을 수 있어 안전하게 제거
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        net.load_state_dict(state)
        net.eval()
        net.to(self.device)
        # 다중 GPU면 DataParallel 로 감싼다
        if self.device.type == "cuda" and ngpu > 1:
            net = nn.DataParallel(net, device_ids=list(range(ngpu)))
        self.model = net

        print("=" * 64)
        print("DeepCAD Denoiser 준비 완료")
        print(f"  device        : {self.device} (batch_size={self.batch_size})")
        print(f"  model (.pth)  : {self.pth_file}")
        print(f"  patch         : {self.patch_x} x {self.patch_y} x {self.patch_t}"
              f"  (overlap={self.overlap_factor})")
        print("=" * 64)

    # -- test_preprocess_chooseOne 이 기대하는 args 네임스페이스를 만든다 --
    def _make_args(self, folder, stem):
        a = SimpleNamespace()
        a.patch_x = self.patch_x
        a.patch_y = self.patch_y
        a.patch_t = self.patch_t
        a.overlap_factor = self.overlap_factor
        # set_params 와 동일한 방식으로 gap 계산
        a.gap_x = int(self.patch_x * (1 - self.overlap_factor))
        a.gap_y = int(self.patch_y * (1 - self.overlap_factor))
        a.gap_t = int(self.patch_t * (1 - self.overlap_factor))
        a.scale_factor = self.scale_factor
        a.test_datasize = self.test_datasize
        a.print_img_name = False
        a.datasets_path = folder        # 전처리 함수가 폴더를 walk 한다
        a.datasets_name = stem          # patch 이름용 (파일 내부에서만 쓰임)
        return a

    def denoise_file(self, input_path, output_dir):
        """tif 또는 raw 한 개를 디노이징해서 결과 tif 를 저장한다.

        .raw 입력은 읽기전용(memmap)으로 열어 임시 .tif 로 변환한 뒤 (원본은 절대
        수정/삭제하지 않음) 기존 tif 파이프라인을 그대로 태우고, 끝나면 임시 폴더를
        지운다. 출력 이름은 원본 .raw stem 기준으로 유지된다.

        Returns:
            (output_path, denoised_array)
        """
        input_path = os.path.abspath(input_path)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(input_path)

        # ---- .raw 면 임시 .tif 로 스테이징 (자기만의 임시 폴더에) ----
        ext = os.path.splitext(input_path)[1].lower()
        temp_dir = None
        try:
            if ext == ".raw":
                temp_dir, work_path = _raw_to_temp_tif(input_path)
            else:
                work_path = input_path

            folder = os.path.dirname(work_path)
            fname = os.path.basename(work_path)
            stem, _ = os.path.splitext(fname)       # raw 입력이면 원본 .raw stem 과 동일
            opt = self._make_args(folder, stem)

            # test_preprocess_chooseOne 은 폴더 안 파일을 정렬해 index 로 고르므로,
            # 이 파일의 index 를 계산해 넘긴다. (raw 는 단독 임시폴더라 index=0)
            files_in_folder = sorted(list(os.walk(folder, topdown=False))[-1][-1])
            if fname not in files_in_folder:
                raise FileNotFoundError(f"'{fname}' 을(를) 폴더에서 찾을 수 없습니다: {folder}")
            img_id = files_in_folder.index(fname)

            # ---- 전처리: 스택을 겹치는 3D patch 들로 분할 ----
            (name_list, noise_img, coordinate_list,
             im_name, img_mean, input_data_type) = self._preprocess(opt, img_id)
            print(f"   입력 형태(T,H,W): {tuple(noise_img.shape)}  dtype={input_data_type}"
                  f"  patch 수={len(name_list)}")

            denoise_img = np.zeros(noise_img.shape)

            test_data = self._Testset(name_list, coordinate_list, noise_img)
            testloader = DataLoader(
                test_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            n_batches = len(testloader)

            # ---- 추론 + 스티칭 (원본 test_collection.test() 와 동일한 로직) ----
            #      tqdm 진행 막대로 patch 진행상황을 보여준다.
            with torch.no_grad():
                pbar = tqdm(testloader, total=n_batches, desc="   denoise",
                            unit="patch", dynamic_ncols=True, leave=False)
                for it, (noise_patch, single_coordinate) in enumerate(pbar):
                    inp = noise_patch.to(self.device).float()
                    out = self.model(inp)

                    output_image = np.squeeze(out.cpu().detach().numpy())
                    raw_image = np.squeeze(inp.cpu().detach().numpy())

                    # batch_size == 1 이면 (T,H,W) 3차원, 다중 GPU면 (B,T,H,W) 4차원
                    turns = 1 if output_image.ndim == 3 else output_image.shape[0]

                    if turns > 1:
                        for idx in range(turns):
                            (op, rp, sw, ew, sh, eh, ss, es) = self._multibatch_save(
                                single_coordinate, idx, output_image, raw_image
                            )
                            op = op + img_mean
                            rp = rp + img_mean
                            denoise_img[ss:es, sh:eh, sw:ew] = (
                                op * (np.sum(rp) / np.sum(op)) ** 0.5
                            )
                    else:
                        (op, rp, sw, ew, sh, eh, ss, es) = self._singlebatch_save(
                            single_coordinate, output_image, raw_image
                        )
                        op = op + img_mean
                        rp = rp + img_mean
                        denoise_img[ss:es, sh:eh, sw:ew] = (
                            op * (np.sum(rp) / np.sum(op)) ** 0.5
                        )

                pbar.close()

            # ---- 마무리: intensity scale 복원 + 원본 dtype 복원 ----
            output_img = denoise_img.squeeze().astype(np.float32) * self.scale_factor
            output_img = output_img.astype(input_data_type)

            os.makedirs(output_dir, exist_ok=True)
            out_name = f"{stem}_{self.model_stem}_output.tif"
            out_path = os.path.join(output_dir, out_name)
            io.imsave(out_path, output_img, check_contrast=False)
            return out_path, output_img
        finally:
            # raw 변환용 임시 폴더 정리 (원본 .raw 는 건드리지 않음)
            if temp_dir is not None and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


def select_files(cli_files):
    """처리할 파일 목록을 결정한다.
    우선순위: (1) 커맨드라인 인자 → (2) FILES 리스트 → (3) GUI 선택창
    """
    if cli_files:
        return [os.path.abspath(f) for f in cli_files]
    if FILES:
        return [os.path.abspath(f) for f in FILES]
    # GUI 파일 선택창
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        paths = filedialog.askopenfilenames(
            title="디노이징할 .tif / .raw 파일들을 선택하세요 (여러 개 선택 가능)",
            filetypes=[("Image stacks", "*.tif *.tiff *.raw"),
                       ("TIFF files", "*.tif *.tiff"),
                       ("RAW files", "*.raw"),
                       ("All files", "*.*")],
        )
        root.destroy()
        return list(paths)
    except Exception as e:
        print(f"[!] 파일 선택창을 열 수 없습니다 ({e}).")
        print("    파일 경로를 인자로 직접 주거나, 스크립트 상단 FILES 리스트를 채우세요.")
        return []


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="DeepCAD 로 .tif / .raw 스택들을 하나씩 순서대로 디노이징합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("files", nargs="*",
                   help="처리할 .tif/.tiff/.raw 파일들 (없으면 GUI/FILES 사용)")
    p.add_argument("--raw-height", type=int, default=RAW_HEIGHT,
                   help=".raw 프레임 높이 (기본 512)")
    p.add_argument("--raw-width", type=int, default=RAW_WIDTH,
                   help=".raw 프레임 너비 (기본 640)")
    p.add_argument("--model", required=True,
                   help="pth/ 아래 모델 폴더 이름 (예: mouse_202506142239)")
    p.add_argument("--deepcad-path", default=os.path.dirname(os.path.abspath(__file__)),
                   help="DeepCAD 프로젝트 루트 경로")
    p.add_argument("--output-dir", default="denoise_results", help="결과 저장 폴더")
    p.add_argument("--pth-dir", default="pth", help="deepcad-path 기준 pth 폴더 이름")
    p.add_argument("--pth-file", default=None,
                   help="특정 .pth 파일 직접 지정 (모델 폴더 자동탐색 대신)")
    p.add_argument("--patch-xy", type=int, default=150, help="patch 의 가로/세로 크기")
    p.add_argument("--patch-t", type=int, default=150, help="patch 의 시간(프레임) 크기")
    p.add_argument("--overlap", type=float, default=0.4, help="patch 간 overlap factor")
    p.add_argument("--fmap", type=int, default=16, help="feature map 개수 (모델 복잡도)")
    p.add_argument("--scale-factor", type=float, default=1.0, help="intensity scaling")
    p.add_argument("--test-datasize", type=int, default=100000,
                   help="스택당 처리할 최대 프레임 수")
    p.add_argument("--gpu", default="0", help="GPU 인덱스 (예: '0' 또는 '0,1')")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker 수 (Windows 는 0 권장)")
    return p


def main():
    args = build_arg_parser().parse_args()

    # .raw frame geometry overrides (default 512 x 640)
    global RAW_HEIGHT, RAW_WIDTH
    RAW_HEIGHT = int(args.raw_height)
    RAW_WIDTH = int(args.raw_width)

    files = select_files(args.files)
    if not files:
        sys.exit("처리할 파일이 없습니다. 종료합니다.")

    # 처리 순서 출력
    print("\n[처리할 파일 순서]")
    for i, f in enumerate(files, 1):
        print(f"  {i:>2}. {f}")
    print(f"총 {len(files)}개 파일\n")

    # 모델은 딱 한 번만 로딩
    denoiser = DeepCADDenoiser(
        deepcad_path=args.deepcad_path,
        denoise_model=args.model,
        pth_dir=args.pth_dir,
        pth_file=args.pth_file,
        fmap=args.fmap,
        gpu=args.gpu,
        patch_x=args.patch_xy,
        patch_y=args.patch_xy,
        patch_t=args.patch_t,
        overlap_factor=args.overlap,
        scale_factor=args.scale_factor,
        test_datasize=args.test_datasize,
        num_workers=args.num_workers,
    )

    output_dir = os.path.abspath(args.output_dir)
    results, failures = [], []
    total_t0 = time.time()

    # 파일을 하나씩 순서대로 처리
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] ▶ {os.path.basename(f)}")
        try:
            t0 = time.time()
            out_path, _ = denoiser.denoise_file(f, output_dir)
            dt = time.time() - t0
            results.append((f, out_path, dt))
            print(f"   ✓ 완료 ({dt:.1f}s) → {out_path}")
        except Exception as e:
            failures.append((f, str(e)))
            print(f"   ✗ 실패: {e}")
            # 한 파일이 실패해도 나머지는 계속 처리

    # 최종 요약
    total_dt = time.time() - total_t0
    print("\n" + "=" * 64)
    print(f"전체 완료: 성공 {len(results)} / 실패 {len(failures)}"
          f"  (총 {total_dt:.1f}s)")
    if results:
        print(f"결과 폴더: {output_dir}")
    if failures:
        print("\n[실패한 파일]")
        for f, err in failures:
            print(f"  - {os.path.basename(f)}: {err}")
    print("=" * 64)


if __name__ == "__main__":
    main()