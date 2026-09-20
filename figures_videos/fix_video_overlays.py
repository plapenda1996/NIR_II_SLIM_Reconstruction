#!/usr/bin/env python3
"""Correct the burnt-in overlays of the exported Supplementary Videos (256 x 256 px MP4).

Why: Depth_cycle_video_gui.py drew the scale bars with pixel sizes defined for a 128-px frame, so the bars in the
256-px exports were ~2x too short. The released videos were corrected with this script instead of being re-exported:
the old bar and label are blacked out (the corner lies outside the circular field of view) and redrawn at the
calibrated scale (high magnification 5.616 um/px, low magnification 17.0 um/px on the 256-px grid).
Optional: rewrite the time stamp for a different acquisition rate, replace or remove the playback-speed line.

examples
  python fix_video_overlays.py in.mp4 out.mp4 --label "200 µm" --um-per-px 5.616                      # Videos 1, 2
  python fix_video_overlays.py in.mp4 out.mp4 --label "600 µm" --um-per-px 17.0                       # Videos 3-5
  python fix_video_overlays.py in.mp4 out.mp4 --label "600 µm" --um-per-px 17.0 --retime-fps 30 --speed-text "real time"   # Video 6
  python fix_video_overlays.py in.mp4 out.mp4 --label "200 µm" --um-per-px 5.616 --remove-line2 5 26 58 41           # Video 7 (z-sweep)
Requires: opencv-python, numpy, pillow, ffmpeg on PATH.
"""
import argparse, subprocess
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--label", required=True); ap.add_argument("--um-per-px", type=float, required=True)
ap.add_argument("--font", default="DejaVuSans.ttf"); ap.add_argument("--crf", type=int, default=12)
ap.add_argument("--retime-fps", type=float, help="rewrite the 't = ... s' stamp as frame / this acquisition rate")
ap.add_argument("--speed-text", help="replacement for the playback-speed line (2nd overlay line)")
ap.add_argument("--line1-box", type=int, nargs=4, default=(4, 7, 84, 25), metavar=("X0", "Y0", "X1", "Y1"))
ap.add_argument("--line2-box", type=int, nargs=4, default=(4, 25, 160, 40), metavar=("X0", "Y0", "X1", "Y1"))
ap.add_argument("--remove-line2", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="black out this box (e.g. 'real time' on a z-sweep)")
a = ap.parse_args()

cap = cv2.VideoCapture(a.src); fps = cap.get(cv2.CAP_PROP_FPS); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release(); frames = np.array(frames); n, H, W, _ = frames.shape
g = frames[::max(1, n // 80)].max(axis=3).astype(np.float32); tmed, tmax = np.median(g, axis=0), g.max(axis=0)

# old bar + label = static bright pixels in the bottom-right corner
y0r, x0r = int(H * 0.84), int(W * 0.74); ys, xs = np.where(tmed[y0r:, x0r:] > 35)
bx0, bx1, by0, by1 = xs.min() + x0r, xs.max() + x0r, ys.min() + y0r, ys.max() + y0r
bar_rows = [y for y in range(by0, by1 + 1) if (tmed[y, bx0:bx1 + 1] > 120).sum() >= 12 and y > by1 - 5]
lab_rows = np.where((tmed[by0:min(bar_rows) - 1, bx0:bx1 + 1] > 60).any(axis=1))[0] + by0; lab_h = lab_rows.max() - lab_rows.min() + 1
barx = np.where(tmed[bar_rows[0], :] > 120)[0]; barx = barx[barx >= x0r]; x_end = int(barx.max())
L = int(round(float(a.label.split()[0]) / a.um_per_px)); x_start = x_end - L + 1

def glyph(text, size):
    font = ImageFont.truetype(a.font, size); im = Image.new("L", (400, 60), 0); ImageDraw.Draw(im).text((4, 8), text, font=font, fill=255)
    arr = np.array(im); yy, xx = np.where(arr > 40); return arr[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
def fit(text, h):
    s = 24
    while s > 6 and glyph(text, s).shape[0] > h: s -= 1
    return s
def paste(f, gl, x, y):
    h, w = gl.shape; al = (gl.astype(np.float32) / 255.0)[:, :, None]; f[y:y + h, x:x + w] = (f[y:y + h, x:x + w] * (1 - al) + 255 * al).astype(np.uint8)

lab = glyph(a.label, fit(a.label, lab_h)); gx = min(int(round((x_start + x_end) / 2 - lab.shape[1] / 2)), W - 2 - lab.shape[1])
cover = (min(bx0 - 4, x_start - 3, gx - 3), max(0, by0 - 4), min(W - 1, bx1 + 4), min(H - 1, by1 + 4))
box = np.zeros((H, W), bool); box[cover[1]:cover[3] + 1, cover[0]:cover[2] + 1] = True
overlay = cv2.dilate((tmed > 25).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
print(f"old bar {len(barx)} px -> new bar {L} px ({L * a.um_per_px:.0f} um); background under the cover box <= {tmax[box & ~overlay].max():.0f}/255")
s1 = fit("t = 0.00 s", a.line1_box[3] - a.line1_box[1] - 7) if a.retime_fps else None
g2 = glyph(a.speed_text, fit(a.speed_text, 9)) if a.speed_text else None

enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
                        "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.dst], stdin=subprocess.PIPE)
for i, f in enumerate(frames):
    f = f.copy(); f[cover[1]:cover[3] + 1, cover[0]:cover[2] + 1] = 0
    f[min(bar_rows):max(bar_rows) + 1, x_start:x_end + 1] = 255; paste(f, lab, gx, lab_rows.max() - lab.shape[0] + 1)
    if a.retime_fps:
        x0, y0, x1, y1 = a.line1_box; f[y0:y1, x0:x1] = 0; g1 = glyph(f"t = {i / a.retime_fps:.2f} s", s1); paste(f, g1, x0 + 4, y1 - 4 - g1.shape[0] + 1)
    if g2 is not None:
        x0, y0, x1, y1 = a.line2_box; f[y0:y1, x0:x1] = 0; paste(f, g2, x0 + 5, y1 - 4 - g2.shape[0] + 2)
    if a.remove_line2:
        x0, y0, x1, y1 = a.remove_line2; f[y0:y1 + 1, x0:x1 + 1] = 0
    enc.stdin.write(f.tobytes())
enc.stdin.close(); enc.wait(); print("wrote", a.dst, f"({n} frames, {fps:g} fps)")
