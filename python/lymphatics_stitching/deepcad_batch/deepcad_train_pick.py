#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# NIR-II SLIM - self-supervised denoising (training) - produces Fig. 3j-m, Supp. Fig. 6d
# environment: heart_valve_py314
"""
deepcad_train_pick.py  -  train a DeepCAD model on FILES YOU SELECT
===================================================================
DeepCAD is self-supervised: it trains directly on your (noisy) .tif stacks.
This launcher lets you PICK the training .tif files in a dialog, gathers them
into a clean training folder (HARD-LINKED, so no gigabytes are copied), and runs
DeepCAD's own training_class on exactly those files. The trained model lands in
  <project>/pth/<training_folder_name>_<...>/   and can then be used to denoise.

Why a separate training folder?  DeepCAD trains on EVERY .tif in datasets_path,
so we isolate your selection in its own folder (originals are never modified).

Run (from the DeepCAD_RT_pytorch_ver2 project folder):
    python deepcad_train_pick.py
    python deepcad_train_pick.py --epochs 10 --name paw_240424 --gpu 0
"""
import os
import sys
import time
import shutil
import argparse

PROJECT = os.path.dirname(os.path.abspath(__file__))


def pick_tif_files(initial_dir):
    """GUI multi-select of .tif training files (console fallback if no GUI)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        files = list(filedialog.askopenfilenames(
            title="Select the .tif stack(s) to TRAIN DeepCAD on (multi-select OK)",
            initialdir=initial_dir if os.path.isdir(initial_dir) else None,
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]))
        root.destroy()
        return files
    except Exception as e:
        print(f"[picker] GUI unavailable ({e}); console fallback.")
        raw = input("Paths to .tif training file(s), ';'-separated: ").strip().strip('"')
        return [p.strip().strip('"') for p in raw.split(";") if p.strip()]


def gather_training_folder(files, train_root, link=True):
    """Put the selected files into a fresh training folder. Hard-link by default
    (instant, no extra disk); fall back to copy if linking fails (cross-volume)."""
    os.makedirs(train_root, exist_ok=True)
    used = []
    for src in files:
        src = os.path.abspath(src)
        if not os.path.isfile(src):
            print(f"  [skip] not a file: {src}"); continue
        dst = os.path.join(train_root, os.path.basename(src))
        if os.path.exists(dst):
            os.remove(dst)
        try:
            if link:
                os.link(src, dst)          # hard link (same NTFS volume) - instant
            else:
                raise OSError("copy requested")
        except OSError:
            print(f"  [copy] {os.path.basename(src)} (hard-link unavailable; copying)")
            shutil.copy2(src, dst)
        used.append(dst)
    return used


def main():
    ap = argparse.ArgumentParser(description="Train DeepCAD on files you pick.")
    ap.add_argument("--name", default=None,
                    help="training set / model name (default: train_pick_<timestamp>)")
    ap.add_argument("--epochs", type=int, default=20, help="training epochs")
    ap.add_argument("--gpu", default="0", help="GPU index, e.g. '0' or '0,1'")
    ap.add_argument("--patch-xy", type=int, default=150)
    ap.add_argument("--patch-t", type=int, default=150)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--fmap", type=int, default=16)
    ap.add_argument("--train-size", type=int, default=2000, help="patches per epoch")
    ap.add_argument("--select-img-num", type=int, default=1000,
                    help="frames used from the START of each stack")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--initial-dir", default=r"E:\260424_mouse_paw\deepcad_input",
                    help="folder the file picker opens in")
    ap.add_argument("--copy", action="store_true",
                    help="copy files into the training folder instead of hard-linking")
    args = ap.parse_args()

    # 1) pick the training files
    files = pick_tif_files(args.initial_dir)
    if not files:
        sys.exit("No file selected. Cancelled.")
    name = args.name or ("train_pick_" + time.strftime("%Y%m%d_%H%M%S"))
    # put the training folder on the SAME volume as the inputs so hard-links are
    # instant (no gigabytes copied); the model still saves to <project>/pth.
    train_root = os.path.join(os.path.dirname(os.path.abspath(files[0])),
                              "_deepcad_train", name)

    print("\n[selected training files]")
    for i, f in enumerate(files, 1):
        print(f"  {i:>2}. {f}")
    used = gather_training_folder(files, train_root, link=not args.copy)
    if not used:
        sys.exit("No usable training files. Cancelled.")
    print(f"\nTraining folder: {train_root}  ({len(used)} stack(s))")

    # 2) build DeepCAD training params (matches train_interface.py / train_collection)
    if PROJECT not in sys.path:
        sys.path.insert(0, PROJECT)
    os.chdir(PROJECT)
    from deepcad.train_collection import training_class

    train_dict = {
        "patch_x": args.patch_xy, "patch_y": args.patch_xy, "patch_t": args.patch_t,
        "overlap_factor": args.overlap, "scale_factor": 1,
        "select_img_num": args.select_img_num, "train_datasets_size": args.train_size,
        "datasets_path": train_root.replace("\\", "/"),
        "pth_dir": "./pth",
        "n_epochs": args.epochs, "lr": 0.00005, "b1": 0.5, "b2": 0.999,
        "fmap": args.fmap, "GPU": args.gpu, "num_workers": args.num_workers,
        "visualize_images_per_epoch": False, "save_test_images_per_epoch": True,
    }
    print("\n[training params]")
    for k, v in train_dict.items():
        print(f"  {k}: {v}")
    print(f"\nModel will be saved under: {os.path.join(PROJECT, 'pth')}\\<{name}_...>\n")

    # 3) train
    t0 = time.time()
    training_class(train_dict).run()
    print(f"\nDone. Training took {(time.time()-t0)/60:.1f} min. "
          f"Use the model in pth/ with 'Deepcad denoise batch.py' to denoise.")


if __name__ == "__main__":
    main()
