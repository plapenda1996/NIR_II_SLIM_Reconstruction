# NIR-II SLIM - extended-field stitching (support module) - produces Fig. 3d, 3j (imported by stitch_tool.py / pulse_gui_main.py)
# environment: stitch_py311
# copied from E:\260325_stitch_process_dy_ver2\slim_recon\viewer.py on 2026-09-17
"""Interactive viewer for 3D reconstruction results using matplotlib widgets."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import hsv_to_rgb


class ReconViewer:
    """Interactive viewer for a reconstructed 3D volume.

    Provides sliders for:
    - Depth index (z-slice)
    - Gamma correction
    - Percentile low/high (display range)
    - Background removal (percentile floor, median subtraction)
    - CLAHE enhancement
    - Bilateral denoising
    - Circular mask
    - MIP toggle
    """

    def __init__(self, volume: np.ndarray, depth_range: tuple = (0, 1),
                  pixel_size_um: float = 11.0, scalebar_um: float = 100.0,
                  scalebar_color: str = 'white', cmap_name: str = 'turbo',
                  gui=None):
        self.volume = volume
        self.H, self.W, self.D = volume.shape
        self.depth_range = depth_range
        self.depths = np.linspace(depth_range[0], depth_range[1], self.D)
        # Optional back-reference to the SLIMGui so every enhancement change
        # publishes the current settings into ``gui.recon_enhance`` (single
        # source-of-truth consumed by the Lymphatic Pulse Analyzer).
        self.gui = gui

        # Display annotation defaults. ``pixel_size_um`` is required to draw
        # a physically-meaningful scale bar; ``scalebar_um`` is the bar's
        # physical length, ``scalebar_color`` is its colour, and ``cmap_name``
        # is the depth colormap used for the depth-coded MIP + colorbar.
        self.pixel_size_um = float(pixel_size_um)
        self.scalebar_um = float(scalebar_um)
        self.scalebar_color = str(scalebar_color)
        self.cmap_name = str(cmap_name)

        # Precompute MIP
        self.mip_val = np.max(volume, axis=2)
        self.mip_idx = np.argmax(volume, axis=2)

        self._build_ui()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.subplots_adjust(bottom=0.32)
        self.fig.canvas.manager.set_window_title('SLIM Recon Viewer')
        self.fig.patch.set_facecolor('#1a1a1a')

        ax_slice = self.fig.add_axes([0.02, 0.38, 0.31, 0.58])
        ax_mip_g = self.fig.add_axes([0.35, 0.38, 0.31, 0.58])
        ax_mip_c = self.fig.add_axes([0.68, 0.38, 0.31, 0.58])

        for ax in [ax_slice, ax_mip_g, ax_mip_c]:
            ax.set_facecolor('black')
            ax.axis('off')

        self.axes = [ax_slice, ax_mip_g, ax_mip_c]

        # State
        self.gamma = 0.5
        self.pct_lo = 0.5
        self.pct_hi = 99.9
        self.z_idx = self.D // 2
        self.bg_mode = 'none'
        self.bg_pct = 5.0
        self.bg_medsz = 15
        self.clahe_on = False
        self.clahe_clip = 2.0
        self.denoise_on = False
        self.denoise_str = 5
        self.mask_on = False
        self.mask_cx = self.W // 2
        self.mask_cy = self.H // 2
        self.mask_r = min(self.H, self.W) // 2 - 5

        # Initial images
        slc, mip_g, mip_c = self._get_displays()
        self.im_slice = ax_slice.imshow(slc, cmap='gray', vmin=0, vmax=1)
        self.title_slice = ax_slice.set_title(self._slice_title(), color='white', fontsize=9)

        self.im_mip_g = ax_mip_g.imshow(mip_g, cmap='gray', vmin=0, vmax=1)
        ax_mip_g.set_title('MIP (grayscale)', color='white', fontsize=9)

        self.im_mip_c = ax_mip_c.imshow(mip_c)
        ax_mip_c.set_title('MIP (depth-coded)', color='white', fontsize=9)

        # Scale bar (one per image axis) + depth colorbar on the depth-coded
        # MIP axis. These are matplotlib primitives, so they render correctly
        # in both the interactive view and any saved figure (PNG/SVG/PDF):
        # SVG export keeps them as true vector elements.
        self._add_scalebar_to_axes([ax_slice, ax_mip_g, ax_mip_c])
        self._add_depth_colorbar(ax_mip_c)

        # Sliders
        sb = '#f0f0f0'
        lc = 'black'
        sc1, sc2, sc3 = '#4a90d9', '#e07040', '#40a040'

        ax_z = self.fig.add_axes([0.08, 0.26, 0.55, 0.022], facecolor=sb)
        ax_g = self.fig.add_axes([0.08, 0.225, 0.28, 0.022], facecolor=sb)
        ax_lo = self.fig.add_axes([0.42, 0.225, 0.21, 0.022], facecolor=sb)
        ax_hi = self.fig.add_axes([0.08, 0.19, 0.28, 0.022], facecolor=sb)

        self.s_z = Slider(ax_z, 'Depth', 0, self.D - 1, valinit=self.z_idx, valstep=1, color=sc1)
        self.s_gamma = Slider(ax_g, 'Gamma', 0.1, 2.0, valinit=self.gamma, color=sc1)
        self.s_lo = Slider(ax_lo, 'Pct Lo', 0, 50, valinit=self.pct_lo, valstep=0.5, color=sc1)
        self.s_hi = Slider(ax_hi, 'Pct Hi', 50, 100, valinit=self.pct_hi, valstep=0.5, color=sc1)

        # BG removal
        ax_bgpct = self.fig.add_axes([0.08, 0.155, 0.28, 0.022], facecolor=sb)
        ax_bgmed = self.fig.add_axes([0.42, 0.155, 0.21, 0.022], facecolor=sb)
        self.s_bgpct = Slider(ax_bgpct, 'BG Pctl', 0, 30, valinit=self.bg_pct, color=sc3)
        self.s_bgmed = Slider(ax_bgmed, 'BG Med', 3, 51, valinit=self.bg_medsz, valstep=2, color=sc3)

        # Enhancement
        ax_cl = self.fig.add_axes([0.08, 0.12, 0.28, 0.022], facecolor=sb)
        ax_dn = self.fig.add_axes([0.42, 0.12, 0.21, 0.022], facecolor=sb)
        self.s_clahe = Slider(ax_cl, 'CLAHE', 0.5, 10.0, valinit=self.clahe_clip, color=sc3)
        self.s_denoise = Slider(ax_dn, 'Denoise', 3, 15, valinit=self.denoise_str, valstep=1, color=sc3)

        # Mask
        ax_mcx = self.fig.add_axes([0.08, 0.085, 0.18, 0.022], facecolor=sb)
        ax_mcy = self.fig.add_axes([0.30, 0.085, 0.18, 0.022], facecolor=sb)
        ax_mr = self.fig.add_axes([0.52, 0.085, 0.18, 0.022], facecolor=sb)
        self.s_mcx = Slider(ax_mcx, 'Mask X', 0, self.W - 1, valinit=self.mask_cx, valstep=1, color=sc2)
        self.s_mcy = Slider(ax_mcy, 'Mask Y', 0, self.H - 1, valinit=self.mask_cy, valstep=1, color=sc2)
        self.s_mr = Slider(ax_mr, 'Mask R', 5, max(self.H, self.W), valinit=self.mask_r, valstep=1, color=sc2)

        for s in [self.s_z, self.s_gamma, self.s_lo, self.s_hi,
                  self.s_bgpct, self.s_bgmed, self.s_clahe, self.s_denoise,
                  self.s_mcx, self.s_mcy, self.s_mr]:
            s.label.set_color(lc)
            s.valtext.set_color(lc)
            s.on_changed(self._on_change)

        # Buttons
        bw, bh = 0.065, 0.026
        bx = 0.74

        ax_b1 = self.fig.add_axes([bx, 0.26, bw, bh])
        self.btn_mask = Button(ax_b1, 'Mask: OFF')
        self.btn_mask.on_clicked(self._toggle_mask)

        ax_b2 = self.fig.add_axes([bx, 0.225, bw, bh])
        self.btn_bg = Button(ax_b2, 'BG: OFF')
        self.btn_bg.on_clicked(self._toggle_bg)

        ax_b3 = self.fig.add_axes([bx + 0.075, 0.225, bw, bh])
        self.btn_clahe = Button(ax_b3, 'CLAHE: OFF')
        self.btn_clahe.on_clicked(self._toggle_clahe)

        ax_b4 = self.fig.add_axes([bx, 0.19, bw, bh])
        self.btn_denoise = Button(ax_b4, 'Denoise: OFF')
        self.btn_denoise.on_clicked(self._toggle_denoise)

        # Save button — opens a Tk filedialog and writes the current figure
        # at high-resolution (300 dpi raster, vector SVG/PDF).
        ax_b5 = self.fig.add_axes([bx + 0.075, 0.19, bw, bh])
        self.btn_save = Button(ax_b5, 'Save Fig…')
        self.btn_save.on_clicked(self._save_figure)

        # Dedicated export for the depth-coded MIP only (single panel + scale
        # bar + depth colorbar). Defaults to SVG so the colorbar / labels
        # land as editable vector primitives.
        ax_b6 = self.fig.add_axes([bx, 0.155, bw, bh])
        self.btn_save_color = Button(ax_b6, 'Save Color MIP…')
        self.btn_save_color.on_clicked(self._save_color_mip)

        # Publish initial defaults so the Lymphatic Pulse Analyzer can read them
        # without waiting for the first slider interaction.
        self._publish_settings()

    # ── Scale bar + depth colorbar overlays ───────────────────────────────
    def _add_scalebar_to_axes(self, axes_list):
        """Draw a matplotlib scale bar (rectangle + label) on each axis.

        Uses ``self.pixel_size_um`` to convert ``self.scalebar_um`` to pixels.
        Drawn in axis data coordinates so the bar moves with the image when
        the user pans/zooms (and survives savefig as a vector primitive).
        """
        from matplotlib.patches import Rectangle
        if self.pixel_size_um <= 0:
            return
        bar_px = self.scalebar_um / self.pixel_size_um
        label_text = self._format_scalebar_length(self.scalebar_um)
        for ax in axes_list:
            margin_x = self.W * 0.04
            margin_y = self.H * 0.04
            thick = max(2.0, self.H * 0.01)
            x1 = self.W - margin_x - bar_px
            x2 = self.W - margin_x
            y_bot = self.H - margin_y
            ax.add_patch(Rectangle(
                (x1, y_bot - thick), bar_px, thick,
                facecolor=self.scalebar_color, edgecolor='none', zorder=10))
            ax.text((x1 + x2) / 2, y_bot - thick - 4,
                    label_text,
                    color=self.scalebar_color, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='black', edgecolor='none',
                              alpha=0.35))

    @staticmethod
    def _format_scalebar_length(um):
        """Match the stitch tool: μm below 1000, mm at or above."""
        if um >= 1000.0:
            mm = um / 1000.0
            if abs(mm - round(mm)) < 1e-6:
                return f"{int(round(mm))} mm"
            return f"{mm:g} mm"
        return f"{um:.0f} μm"

    def _add_depth_colorbar(self, ax):
        """Inset a vertical depth colorbar on the right side of ``ax``.

        Maps the cmap to the configured ``depth_range``. Ticks are shown at
        the lower and upper bounds (with a midpoint), labelled in the same
        units as ``depth_range``; if the units are normalised (0–1) the
        labels still convey colour↔depth correspondence.
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib as _mpl
        cmap = plt.colormaps.get_cmap(self.cmap_name)
        # Inset on the right edge of the image axis.
        cax = inset_axes(ax, width="3.5%", height="55%",
                         loc='center right', borderpad=-2.0)
        d_lo, d_hi = float(self.depth_range[0]), float(self.depth_range[1])
        norm = _mpl.colors.Normalize(vmin=d_lo, vmax=d_hi)
        cb = _mpl.colorbar.ColorbarBase(
            cax, cmap=cmap, norm=norm, orientation='vertical')
        # Three numeric ticks (top / middle / bottom) with auto μm→mm
        # switching at the high end so deep volumes don't crowd the labels.
        ticks = [d_lo, (d_lo + d_hi) / 2.0, d_hi]
        tick_labels = [self._format_scalebar_length(t) for t in ticks]
        cb.set_ticks(ticks)
        cb.set_ticklabels(tick_labels)
        # Pick an axis label unit based on the larger endpoint so the bar
        # label reads "Depth (mm)" for deep volumes, "Depth (μm)" otherwise.
        cb_unit = 'mm' if max(abs(d_lo), abs(d_hi)) >= 1000.0 else 'μm'
        cb.set_label(f'Depth ({cb_unit})', color='white',
                      fontsize=8, labelpad=4)
        cb.outline.set_edgecolor('white')
        cax.tick_params(colors='white', labelsize=7, length=2, width=0.5)

    # ── Save (high-res raster or vector SVG/PDF) ──────────────────────────
    def _save_figure(self, _event=None):
        """Save a clean copy of the 3 image panels + overlays to disk.

        Pops a Tk filedialog. Output format chosen by the file extension:
        ``.svg`` / ``.pdf`` → vector (scale bar, colorbar, labels stay
        editable vector primitives; the image stays raster). ``.png`` /
        ``.tif`` → 300 dpi raster.
        """
        try:
            import tkinter as _tk
            from tkinter import filedialog as _fd
            # FigureCanvasTkAgg already created a root; reuse rather than
            # spawning a new one (a second Tk root would deadlock).
            root = _tk._default_root or _tk.Tk()
            try:
                root.withdraw()
            except Exception:
                pass
            path = _fd.asksaveasfilename(
                title="Save Recon Figure",
                defaultextension=".png",
                filetypes=[("PNG (300 dpi)", "*.png"),
                            ("SVG (vector)", "*.svg"),
                            ("PDF (vector)", "*.pdf"),
                            ("TIFF (300 dpi)", "*.tif"),
                            ("All", "*.*")])
        except Exception:
            path = None
        if not path:
            return
        self._save_to_path(path)

    def _save_color_mip(self, _event=None):
        """Export ONLY the depth-coded MIP panel (image + scale bar + depth
        colorbar) as a high-resolution file, defaulting to vector SVG.

        Pops a Tk filedialog. ``.svg`` / ``.pdf`` → vector; ``.png`` / ``.tif``
        → 300 dpi raster.
        """
        try:
            import tkinter as _tk
            from tkinter import filedialog as _fd
            root = _tk._default_root or _tk.Tk()
            try:
                root.withdraw()
            except Exception:
                pass
            path = _fd.asksaveasfilename(
                title="Save Depth-Coded MIP",
                defaultextension=".svg",
                filetypes=[("SVG (vector)", "*.svg"),
                            ("PDF (vector)", "*.pdf"),
                            ("PNG (300 dpi)", "*.png"),
                            ("TIFF (300 dpi)", "*.tif"),
                            ("All", "*.*")])
        except Exception:
            path = None
        if not path:
            return
        self._save_color_mip_to_path(path)

    def _save_color_mip_to_path(self, path):
        """Single-panel render of just the depth-coded MIP."""
        _, _, mip_c = self._get_displays()
        fig2 = plt.figure(figsize=(6, 6), facecolor='white')
        ax = fig2.add_subplot(1, 1, 1)
        ax.imshow(mip_c)
        ax.set_title('MIP (depth-coded)', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        self._add_scalebar_to_axes([ax])
        self._add_depth_colorbar(ax)
        ext = path.lower().rsplit('.', 1)[-1]
        kwargs = {'bbox_inches': 'tight', 'facecolor': fig2.get_facecolor()}
        if ext in ('svg', 'pdf'):
            kwargs['transparent'] = False
        else:
            kwargs['dpi'] = 300
        fig2.savefig(path, **kwargs)
        plt.close(fig2)
        print(f"[ReconViewer] saved depth-coded MIP {path}")

    def _save_to_path(self, path):
        """Render a clean 3-panel figure (no sliders/buttons) + overlays
        and write it to ``path``. Uses the current state of the volume."""
        slc, mip_g, mip_c = self._get_displays()
        # Build a fresh figure so saved output omits the slider widgets.
        fig2 = plt.figure(figsize=(15, 5), facecolor='white')
        gs = fig2.add_gridspec(1, 3, wspace=0.04)
        ax1 = fig2.add_subplot(gs[0, 0]); ax1.imshow(slc, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(self._slice_title(), fontsize=9)
        ax2 = fig2.add_subplot(gs[0, 1]); ax2.imshow(mip_g, cmap='gray', vmin=0, vmax=1)
        ax2.set_title('MIP (grayscale)', fontsize=9)
        ax3 = fig2.add_subplot(gs[0, 2]); ax3.imshow(mip_c)
        ax3.set_title('MIP (depth-coded)', fontsize=9)
        for ax in (ax1, ax2, ax3):
            ax.set_xticks([]); ax.set_yticks([])
        # Add overlays
        saved_color = self.scalebar_color
        self.scalebar_color = 'white' if saved_color == 'white' else saved_color
        self._add_scalebar_to_axes([ax1, ax2, ax3])
        self._add_depth_colorbar(ax3)
        self.scalebar_color = saved_color
        # DPI is irrelevant for vector formats but matplotlib accepts it.
        ext = path.lower().rsplit('.', 1)[-1]
        kwargs = {'bbox_inches': 'tight', 'facecolor': fig2.get_facecolor()}
        if ext in ('svg', 'pdf'):
            kwargs['transparent'] = False
        else:
            kwargs['dpi'] = 300
        fig2.savefig(path, **kwargs)
        plt.close(fig2)
        print(f"[ReconViewer] saved {path}")

    # ── Canonical enhancement settings dict (single source of truth) ──────
    # Read by :func:`slim_recon.enhance.enhance_frame`. Whenever any UI
    # widget mutates a setting, we also publish into ``self.gui.recon_enhance``
    # so the Lymphatic Pulse Analyzer can reproduce the displayed pixels.
    def _settings_dict(self):
        return dict(
            gamma=float(self.gamma),
            pct_lo=float(self.pct_lo), pct_hi=float(self.pct_hi),
            bg_mode=str(self.bg_mode), bg_pct=float(self.bg_pct),
            bg_medsz=int(self.bg_medsz),
            denoise_on=bool(self.denoise_on), denoise_str=int(self.denoise_str),
            clahe_on=bool(self.clahe_on), clahe_clip=float(self.clahe_clip),
            mask_on=bool(self.mask_on),
            mask_cx=int(self.mask_cx), mask_cy=int(self.mask_cy),
            mask_r=int(self.mask_r),
            mask_softness=2.0,                      # ReconViewer convention
        )

    def _publish_settings(self):
        if self.gui is not None:
            try:
                self.gui.recon_enhance = dict(self._settings_dict())
                self.gui.recon_enhance['__source__'] = 'ReconViewer'
            except Exception:
                pass

    def _get_displays(self):
        from .enhance import enhance_frame
        cmap_obj = plt.colormaps.get_cmap('turbo')
        s = self._settings_dict()

        # Depth slice — gamma uses volume-percentile clip
        slc = enhance_frame(self.volume[:, :, self.z_idx], s, src_volume=self.volume)

        # Grayscale MIP — same pipeline
        mip_g = enhance_frame(self.mip_val, s, src_volume=self.volume)

        # Depth-coded MIP
        depth_norm = self.mip_idx.astype(np.float32) / max(self.D - 1, 1)
        depth_rgb = cmap_obj(depth_norm)[:, :, :3]
        mip_c = (depth_rgb * mip_g[:, :, np.newaxis]).astype(np.float32)

        return slc, mip_g, mip_c

    def _slice_title(self):
        d = self.depths[self.z_idx]
        return f'Depth z={self.z_idx}  (d={d:.3f})'

    def _on_change(self, _val):
        self.z_idx = int(self.s_z.val)
        self.gamma = self.s_gamma.val
        self.pct_lo = self.s_lo.val
        self.pct_hi = self.s_hi.val
        self.bg_pct = self.s_bgpct.val
        self.bg_medsz = int(self.s_bgmed.val)
        self.clahe_clip = self.s_clahe.val
        self.denoise_str = int(self.s_denoise.val)
        self.mask_cx = int(self.s_mcx.val)
        self.mask_cy = int(self.s_mcy.val)
        self.mask_r = int(self.s_mr.val)

        slc, mip_g, mip_c = self._get_displays()
        self.im_slice.set_data(slc)
        self.im_mip_g.set_data(mip_g)
        self.im_mip_c.set_data(mip_c)
        self.title_slice.set_text(self._slice_title())
        self.fig.canvas.draw_idle()
        self._publish_settings()

    def _toggle_mask(self, _):
        self.mask_on = not self.mask_on
        self.btn_mask.label.set_text(f'Mask: {"ON" if self.mask_on else "OFF"}')
        self._on_change(None)

    def _toggle_bg(self, _):
        modes = ['none', 'percentile', 'median']
        idx = (modes.index(self.bg_mode) + 1) % len(modes)
        self.bg_mode = modes[idx]
        labels = {'none': 'BG: OFF', 'percentile': 'BG: Pctl', 'median': 'BG: Med'}
        self.btn_bg.label.set_text(labels[self.bg_mode])
        self._on_change(None)

    def _toggle_clahe(self, _):
        self.clahe_on = not self.clahe_on
        self.btn_clahe.label.set_text(f'CLAHE: {"ON" if self.clahe_on else "OFF"}')
        self._on_change(None)

    def _toggle_denoise(self, _):
        self.denoise_on = not self.denoise_on
        self.btn_denoise.label.set_text(f'Denoise: {"ON" if self.denoise_on else "OFF"}')
        self._on_change(None)

    def show(self):
        plt.show()
