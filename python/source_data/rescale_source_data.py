#!/usr/bin/env python3
"""Rescale the numerical analysis exports to the calibrated pixel sizes used in the paper.

The analysis scripts in this repository were run with nominal pixel sizes (see ../CALIBRATION.md).
The values reported in the figures and in the Source Data files were obtained by rescaling those
exports with the calibrated pixel sizes; nothing else is changed (times, normalised signals and
strain rates do not depend on the pixel size).

    heart  (Fig. 2c-f): area and dV/dt      x (5.616 / 10.0)**2           = 0.31540
    ear    (Fig. 3g)  : path length, speed  x 17.0 * k / 40.0   (k = 256-grid px per analysis-canvas unit;
                                                                  ROI1 k = 0.672 -> 0.2856, ROI2 k = 0.754 -> 0.3204)
    lymph  (Fig. 3l,m): length, speed       x 4.25 / 4.0                   = 1.0625

usage:  python rescale_source_data.py --in-dir analysis_exports --out-dir source_data_final
"""
import argparse, os
import numpy as np
import pandas as pd

P_HEART_RUN, P_HEART_CAL = 10.0, 5.616        # um/px on the 256-px segmentation grid
P_EAR_RUN, P_EAR_GRID = 40.0, 17.0            # um per canvas unit as run; um/px on the 256-px low-magnification grid
K_EAR = {"ROI1": 0.672, "ROI2": 0.754}        # 256-grid px per canvas unit (fit of the drawn centerlines, rms < 0.15 px)
P_LYMPH_RUN, P_LYMPH_CAL = 4.0, 4.25          # um/px on the 1024-px (2x upsampled) low-magnification grid


def heart(in_dir, out_dir):
    f = (P_HEART_CAL / P_HEART_RUN) ** 2
    for name in ("SourceData_Fig2c_WT_area_by_depth.csv", "SourceData_Fig2d_WEA_area_by_depth.csv"):
        d = pd.read_csv(os.path.join(in_dir, name)); out = pd.DataFrame({"time_ms": d.time_ms})
        for c in d.columns[1:]:
            out[c.replace("_1e6_um2", "_1e4_um2")] = (d[c] * f * 100).round(5)      # x1e6 um^2 -> x1e4 um^2
        out.to_csv(os.path.join(out_dir, name), index=False)
    for name in ("SourceData_Fig2e_WT_volume_rate.csv", "SourceData_Fig2f_WEA_volume_rate.csv"):
        d = pd.read_csv(os.path.join(in_dir, name))
        pd.DataFrame({"time_ms": d.time_ms,
                      "dVdt_raw_nL_per_s": (d.dVdt_raw_1e9_um3_per_s * f * 1000).round(4),          # 1 nL = 1e6 um^3
                      "dVdt_plotted_nL_per_s": (d.dVdt_plotted_1e9_um3_per_s * f * 1000).round(4)}
                     ).to_csv(os.path.join(out_dir, name), index=False)


def ear(in_dir, out_dir):
    for roi, name in (("ROI1", "SourceData_Fig3g_ROI1_bolus_speed.csv"), ("ROI2", "SourceData_Fig3g_ROI2_bolus_speed.csv")):
        d = pd.read_csv(os.path.join(in_dir, name)); um = P_EAR_GRID * K_EAR[roi]; f = um / P_EAR_RUN
        keep = d.local_speed_true_mm_s.notna()                                   # drop centerline segments without data
        pd.DataFrame({"segment": d.segment, "s_frac_start": d.s_frac_start, "s_um_start": (d.s_px_start * um).round(2),
                      "x_canvas": d.x_px, "y_canvas": d.y_px,
                      "x_px_256grid_rel": ((d.x_px - d.x_px.iloc[0]) * K_EAR[roi]).round(3),
                      "y_px_256grid_rel": ((d.y_px - d.y_px.iloc[0]) * K_EAR[roi]).round(3),
                      "local_speed_mm_s": (d.local_speed_true_mm_s * f).round(5),
                      "local_speed_smoothed_mm_s": (d.local_speed_true_smoothed_mm_s * f).round(5)})[keep
                     ].to_csv(os.path.join(out_dir, name), index=False)


def lymph(in_dir, out_dir):
    f = P_LYMPH_CAL / P_LYMPH_RUN
    name = "SourceData_Fig3l_source_data_bolus_speed.csv"; d = pd.read_csv(os.path.join(in_dir, name))
    d["s_um"] = (d.s_um * f).round(3); d["v_raw_um_s"] = (d.v_raw_um_s * f).round(4); d["v_display_um_s"] = (d.v_display_um_s * f).round(4)
    d.to_csv(os.path.join(out_dir, name), index=False)
    name = "SourceData_Fig3m_source_data_multiple_lymph_pulses.csv"; d = pd.read_csv(os.path.join(in_dir, name))
    for c in ("speed_um_s", "se_um_s", "ci95_lower_um_s", "ci95_upper_um_s", "transport_distance_um", "bolus_fwhm_um"):
        d[c] = (d[c] * f).round(1)
    d["px_um"] = P_LYMPH_CAL; d.to_csv(os.path.join(out_dir, name), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", choices=["heart", "ear", "lymph"], nargs="*", default=["heart", "ear", "lymph"])
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    for part in a.only:
        {"heart": heart, "ear": ear, "lymph": lymph}[part](a.in_dir, a.out_dir); print("rescaled:", part)
