# NIR-II SLIM - mouse ear bolus velocity - produces Fig. 3e-g
# environment: heart_valve_py314
# copied from D:\NIR2SLIM\demotion\ROI_analysis_mouse_ear.py on 2026-09-17
# %%
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
from PIL import Image, ImageTk
from scipy import ndimage
from scipy.signal import medfilt, find_peaks
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import os
import io

# NumPy 2.0 removed np.trapz (renamed to np.trapezoid). Support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

class TIFAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("3D TIF Stack ROI Analyzer with Blood Flow Analysis")
        # Fit the window to the screen so nothing (controls / image / status bar)
        # falls off the bottom on small laptop displays.
        _sw = self.root.winfo_screenwidth()
        _sh = self.root.winfo_screenheight()
        _w = min(1400, _sw - 40)
        _h = min(900, _sh - 80)          # leave room for the taskbar
        self.root.geometry(f"{_w}x{_h}+20+20")
        self.root.minsize(900, 500)
        
        # Data variables
        self.image_stack = None
        self.current_frame = 0
        self.background_roi_1 = None  # First background line
        self.background_roi_2 = None  # Second background line
        self.foreground_roi = None
        self.waypoints = []          # first-class float waypoints for the 'waypoint' method
        self._wp_next_id = 0
        self.foreground_line_width = 5
        self.background_line_width = 5
        self.median_filter_window = 5
        self.apply_filter = False
        self.background_segments = 10

        # Outlier removal settings
        self.remove_outliers = False
        self.outlier_threshold = 3.0
        self.outlier_method = 'zscore'  # 'zscore', 'iqr', or 'modified_z'
        self.remove_outliers = False
        
        # 添加存储TIF文件路径的变量
        self.tif_file_path = None
        
        # Scale settings
        self.pixel_size = 11.0  # μm per pixel (default for typical mouse imaging)
        self.frame_rate = 30.0  # fps
        
        # Time markers for analysis range
        self.start_marker = None  # Frame index
        self.end_marker = None    # Frame index
        
        # Image display settings
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.show_scalebar = True
        
        # Analysis results
        self.foreground_signal = None
        self.background_signal = None
        self.difference_signal = None
        
        # GUI state
        self.drawing_mode = None  # 'background1', 'background2', 'foreground', None
        self.temp_points = []
        self.roi_plots_open = False
        
        # Drawing visualization
        self.drawing_lines = []
        
        # 播放控制状态
        self.is_playing = False
        self.play_speed = 50  # ms between frames
        self.play_job = None
        
        # 存储打开的分析窗口和进度线
        self.analysis_windows = []  # 存储 {'window': window, 'axes': [ax1, ax2, ...], 'lines': [line1, line2, ...]}

        self.min_value = 0.0
        self.max_value = 1.0
        self.subtract_background = True
        self.normalization_mode = "min_subtract"

        # ==================== MOUSE EAR VESSEL ANALYSIS SETTINGS ====================
        # Physical / fluid properties (blood at body temperature, adjustable in GUI)
        self.blood_viscosity = 3.5e-3      # Pa*s (dynamic viscosity of whole blood, ~3-4 cP)
        self.blood_density = 1060.0        # kg/m^3
        # Number of perpendicular cross-section profiles sampled along the vessel
        self.n_cross_sections = 20
        # Half-length (in pixels) of each perpendicular profile line on each side of the vessel
        self.profile_half_len = 20
        # How the vessel "edge" is defined for diameter: 'fwhm' or 'threshold'
        self.diameter_method = 'fwhm'
        # Kymograph streak-angle velocity settings
        self.velocity_min_speed = 0.0      # mm/s lower bound for reporting
        # Cached mouse-ear results so multiple panels can share one extraction
        self.vessel_results = None
        # ---- Vessel mask / perfusion visualisation subsystem (Parts 0-3) ----
        # A binary vessel segmentation underpins the isochrone arrival map, the
        # structural-vs-functional distance maps and the multi-segment scaling
        # analysis. Built explicitly (analysis), inspected/edited, and cached.
        self.vessel_mask = None            # bool HxW  : anatomical vessel pixels
        self.fov_mask = None               # bool HxW  : illuminated aperture (FOV)
        self.vesselness = None             # float HxW : cached Frangi response
        self._vessel_meta = None           # dict      : params + source + stats
        self.segment_table = []            # list of per-segment records (Part 3)
        self._arrival_pixmap = None        # dict      : cached per-pixel arrival map
        # ---- Breathing / motion frame exclusion ----
        # Frames corrupted by respiratory motion are excluded from the temporal
        # mean image (morphometry) and the kymograph (velocity). Detection uses a
        # per-frame phase-correlation displacement vs the temporal-mean reference.
        self.excluded_frames = set()       # set of excluded frame indices (global)
        self.breathing_scores = None       # cached (idx, scores, threshold) for plotting
        self.breathing_sigma = 3.0         # robust-MAD sigma for auto-detection
        # ==========================================================================

        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel — wrapped in a height-capped scrollable canvas so every
        # function stays reachable on small laptop screens. `control_frame` below
        # is the INNER frame, so all downstream controls are added to it unchanged.
        control_outer = ttk.LabelFrame(main_frame, text="Controls (scroll for more ▾)")
        control_outer.pack(fill=tk.X, pady=(0, 10))
        ctrl_canvas = tk.Canvas(control_outer, highlightthickness=0, borderwidth=0)
        ctrl_scroll = ttk.Scrollbar(control_outer, orient='vertical',
                                    command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        control_frame = ttk.Frame(ctrl_canvas)
        _ctrl_win = ctrl_canvas.create_window((0, 0), window=control_frame, anchor='nw')
        # Cap the visible height to ~45% of the screen; scroll if content exceeds it.
        _ctrl_cap = int(self.root.winfo_screenheight() * 0.45)

        def _ctrl_on_inner(_evt=None):
            ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all'))
            need = control_frame.winfo_reqheight()
            ctrl_canvas.configure(height=min(need, _ctrl_cap))
        control_frame.bind('<Configure>', _ctrl_on_inner)
        # keep the inner frame as wide as the canvas
        ctrl_canvas.bind('<Configure>',
                         lambda e: ctrl_canvas.itemconfigure(_ctrl_win, width=e.width))

        # Mouse wheel scrolls the controls ONLY while the pointer is over them, so
        # the image canvas's own wheel-zoom binding is left untouched.
        def _ctrl_wheel(evt):
            d = getattr(evt, 'delta', 0)
            if d:
                ctrl_canvas.yview_scroll(int(-d / 120) or (-1 if d > 0 else 1), 'units')
            elif getattr(evt, 'num', None) == 4:
                ctrl_canvas.yview_scroll(-1, 'units')
            elif getattr(evt, 'num', None) == 5:
                ctrl_canvas.yview_scroll(1, 'units')

        def _ctrl_bind_wheel(_e):
            ctrl_canvas.bind_all('<MouseWheel>', _ctrl_wheel)
            ctrl_canvas.bind_all('<Button-4>', _ctrl_wheel)
            ctrl_canvas.bind_all('<Button-5>', _ctrl_wheel)

        def _ctrl_unbind_wheel(_e):
            ctrl_canvas.unbind_all('<MouseWheel>')
            ctrl_canvas.unbind_all('<Button-4>')
            ctrl_canvas.unbind_all('<Button-5>')
        ctrl_canvas.bind('<Enter>', _ctrl_bind_wheel)
        ctrl_canvas.bind('<Leave>', _ctrl_unbind_wheel)

        # File operations
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load TIF Stack", 
                  command=self.load_tif_stack).pack(side=tk.LEFT, padx=(0, 10))
        
        # Scale settings
        scale_frame = ttk.LabelFrame(control_frame, text="Scale Settings")
        scale_frame.pack(fill=tk.X, padx=5, pady=5)
        
        scale_inputs_frame = ttk.Frame(scale_frame)
        scale_inputs_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scale_inputs_frame, text="Pixel Size (μm/pixel):").pack(side=tk.LEFT)
        self.pixel_size_var = tk.DoubleVar(value=11.0)
        pixel_entry = ttk.Entry(scale_inputs_frame, textvariable=self.pixel_size_var, width=8)
        pixel_entry.pack(side=tk.LEFT, padx=(5, 15))
        pixel_entry.bind('<KeyRelease>', self.update_scale_settings)
        
        ttk.Label(scale_inputs_frame, text="Frame Rate (fps):").pack(side=tk.LEFT)
        self.frame_rate_var = tk.DoubleVar(value=30.0)
        fps_entry = ttk.Entry(scale_inputs_frame, textvariable=self.frame_rate_var, width=8)
        fps_entry.pack(side=tk.LEFT, padx=(5, 0))
        fps_entry.bind('<KeyRelease>', self.update_scale_settings)

        ttk.Label(scale_inputs_frame, text="Min:").pack(side=tk.LEFT)
        self.min_var = tk.DoubleVar(value=0.0)
        fps_entry = ttk.Entry(scale_inputs_frame, textvariable=self.min_var, width=8)
        fps_entry.pack(side=tk.LEFT, padx=(5, 0))
        fps_entry.bind('<KeyRelease>', self.update_frame)

        ttk.Label(scale_inputs_frame, text="Max:").pack(side=tk.LEFT)
        self.max_var = tk.DoubleVar(value=1.0)
        fps_entry = ttk.Entry(scale_inputs_frame, textvariable=self.max_var, width=8)
        fps_entry.pack(side=tk.LEFT, padx=(5, 0))
        fps_entry.bind('<KeyRelease>', self.update_frame)
        
        # Frame selection
        frame_frame = ttk.Frame(control_frame)
        frame_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(frame_frame, text="Current Frame:").pack(side=tk.LEFT)
        self.frame_var = tk.IntVar()
        self.frame_scale = ttk.Scale(frame_frame, from_=0, to=0, 
                                   variable=self.frame_var,
                                   command=self.update_frame)
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        self.frame_label = ttk.Label(frame_frame, text="0/0")
        self.frame_label.pack(side=tk.LEFT)
        
        # 播放控制
        play_frame = ttk.Frame(control_frame)
        play_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_button = ttk.Button(play_frame, text="▶ Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(play_frame, text="Speed (ms):").pack(side=tk.LEFT)
        self.play_speed_var = tk.IntVar(value=50)
        ttk.Spinbox(play_frame, from_=10, to=500, width=5,
                   textvariable=self.play_speed_var,
                   command=self.update_play_speed).pack(side=tk.LEFT, padx=(5, 15))
        
        # 添加播放范围提示
        self.play_range_var = tk.StringVar(value="Will play full range")
        ttk.Label(play_frame, textvariable=self.play_range_var, 
                 font=('TkDefaultFont', 8), foreground='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # Time markers
        marker_frame = ttk.LabelFrame(control_frame, text="Analysis Time Range")
        marker_frame.pack(fill=tk.X, padx=5, pady=5)
        
        marker_buttons_frame = ttk.Frame(marker_frame)
        marker_buttons_frame.pack(fill=tk.X, pady=5)
        
        start_btn = ttk.Button(marker_buttons_frame, text="Set Start Marker", 
                  command=self.set_start_marker)
        start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        end_btn = ttk.Button(marker_buttons_frame, text="Set End Marker", 
                  command=self.set_end_marker)
        end_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(marker_buttons_frame, text="Clear Markers", 
                  command=self.clear_markers).pack(side=tk.LEFT, padx=(0, 5))
        
        # Add helpful instruction
        ttk.Label(marker_buttons_frame, text="(Sets current frame as marker)", 
                 font=('TkDefaultFont', 8), foreground='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        marker_info_frame = ttk.Frame(marker_frame)
        marker_info_frame.pack(fill=tk.X, pady=5)
        
        self.marker_info_var = tk.StringVar(value="No markers set - using full time range")
        ttk.Label(marker_info_frame, textvariable=self.marker_info_var, 
                 font=('TkDefaultFont', 8)).pack(side=tk.LEFT)
        
        # ROI controls
        roi_frame = ttk.LabelFrame(control_frame, text="ROI Drawing")
        roi_frame.pack(fill=tk.X, padx=5, pady=5)
        
        roi_buttons_frame = ttk.Frame(roi_frame)
        roi_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(roi_buttons_frame, text="Draw Background ROI 1 (Line)", 
                  command=self.start_background_roi_1).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(roi_buttons_frame, text="Draw Background ROI 2 (Line)", 
                  command=self.start_background_roi_2).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(roi_buttons_frame, text="Draw Foreground ROI (Line)", 
                  command=self.start_foreground_roi).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(roi_buttons_frame, text="Clear All ROIs", 
                  command=self.clear_rois).pack(side=tk.LEFT, padx=(0, 5))
        # v2: line-ROI persistence (fig3f_roi_trace.ROISet json) — 재현/기탁용
        ttk.Button(roi_buttons_frame, text="Save ROIs…",
                  command=self.save_line_rois).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Button(roi_buttons_frame, text="Load ROIs…",
                  command=self.load_line_rois).pack(side=tk.LEFT, padx=(0, 5))
        
        # Line width settings
        line_width_frame = ttk.Frame(roi_frame)
        line_width_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(line_width_frame, text="Foreground Width:").pack(side=tk.LEFT)
        self.foreground_line_width_var = tk.IntVar(value=5)
        ttk.Spinbox(line_width_frame, from_=1, to=50, width=5,
                   textvariable=self.foreground_line_width_var,
                   command=self.update_foreground_line_width).pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(line_width_frame, text="Background Width:").pack(side=tk.LEFT)
        self.background_line_width_var = tk.IntVar(value=5)
        ttk.Spinbox(line_width_frame, from_=1, to=50, width=5,
                   textvariable=self.background_line_width_var,
                   command=self.update_background_line_width).pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(line_width_frame, text="Background Segments:").pack(side=tk.LEFT)
        self.background_segments_var = tk.IntVar(value=10)
        ttk.Spinbox(line_width_frame, from_=5, to=50, width=5,
                   textvariable=self.background_segments_var,
                   command=self.update_background_segments).pack(side=tk.LEFT, padx=(5, 0))
        
        # Filter controls
        filter_frame = ttk.LabelFrame(control_frame, text="Noise Reduction")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        filter_controls_frame = ttk.Frame(filter_frame)
        filter_controls_frame.pack(fill=tk.X, pady=5)

        self.filter_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_controls_frame, text="Apply Median Filter", 
                    variable=self.filter_enabled_var,
                    command=self.toggle_filter).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(filter_controls_frame, text="Window Size:").pack(side=tk.LEFT)
        self.filter_window_var = tk.IntVar(value=5)
        filter_spinbox = ttk.Spinbox(filter_controls_frame, from_=3, to=51, width=5,
                                textvariable=self.filter_window_var,
                                command=self.update_filter_window,
                                increment=2)
        filter_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        filter_spinbox.bind('<KeyRelease>', lambda e: self.update_filter_window())
        filter_spinbox.bind('<Button-1>', lambda e: self.update_filter_window())
        filter_spinbox.bind('<ButtonRelease-1>', lambda e: self.update_filter_window())

        # Outlier removal controls
        outlier_controls_frame = ttk.Frame(filter_frame)
        outlier_controls_frame.pack(fill=tk.X, pady=5)

        self.outlier_removal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(outlier_controls_frame, text="Remove Outliers", 
                    variable=self.outlier_removal_var,
                    command=self.toggle_outlier_removal).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(outlier_controls_frame, text="Threshold (σ):").pack(side=tk.LEFT)
        self.outlier_threshold_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(outlier_controls_frame, from_=1.0, to=5.0, width=5,
                textvariable=self.outlier_threshold_var,
                command=self.update_outlier_threshold,
                increment=0.5).pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(outlier_controls_frame, text="Method:").pack(side=tk.LEFT)
        self.outlier_method_var = tk.StringVar(value="zscore")
        outlier_method_combo = ttk.Combobox(outlier_controls_frame, 
                                        textvariable=self.outlier_method_var,
                                        width=12, state='readonly')
        outlier_method_combo['values'] = ('zscore', 'iqr', 'modified_z', 'moving_window', 'morphological')
        outlier_method_combo.pack(side=tk.LEFT, padx=(5, 15))
        outlier_method_combo.bind('<<ComboboxSelected>>', self.update_outlier_method)

        ttk.Label(outlier_controls_frame, text="Morph Size:").pack(side=tk.LEFT)
        self.morph_size_var = tk.IntVar(value=5)
        ttk.Spinbox(outlier_controls_frame, from_=3, to=15, width=5,
                textvariable=self.morph_size_var,
                command=self.update_morph_size,
                increment=2).pack(side=tk.LEFT, padx=(5, 0))
        
        # Normalization controls (放在 Filter controls 之后)
        norm_frame = ttk.LabelFrame(control_frame, text="Signal Processing")
        norm_frame.pack(fill=tk.X, padx=5, pady=5)

        norm_controls_frame = ttk.Frame(norm_frame)
        norm_controls_frame.pack(fill=tk.X, pady=5)

        # 背景减除开关
        self.bg_subtract_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(norm_controls_frame, text="Subtract Background", 
                    variable=self.bg_subtract_var,
                    command=self.update_bg_subtract).pack(side=tk.LEFT, padx=(0, 10))

        # 归一化模式选择
        ttk.Label(norm_controls_frame, text="Normalization:").pack(side=tk.LEFT)
        self.norm_mode_var = tk.StringVar(value="min_subtract")
        norm_combo = ttk.Combobox(norm_controls_frame, textvariable=self.norm_mode_var,
                                width=15, state='readonly')
        norm_combo['values'] = ('min_subtract', 'percentile_10', 'percentile_20', 
                                'percentile_50', 'normalize_0_1')
        norm_combo.pack(side=tk.LEFT, padx=(5, 0))
        norm_combo.bind('<<ComboboxSelected>>', self.update_norm_mode)
        


        # Analysis controls
        analysis_frame = ttk.LabelFrame(control_frame, text="Analysis")
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        analysis_buttons_frame = ttk.Frame(analysis_frame)
        analysis_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_buttons_frame, text="Show Combined Signals", 
                  command=self.show_combined_signals).pack(side=tk.LEFT, padx=(0, 10))
        # v2: Fig. 3f Source Data (CSV + Nature-style PDF/SVG/PNG) — 숫자는 Show Combined Signals 와 동일
        ttk.Button(analysis_buttons_frame, text="Export Fig 3f (PDF+CSV)",
                  command=self.export_fig3f_source_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(analysis_buttons_frame, text="Videokymography", 
                  command=self.show_videokymography).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(analysis_buttons_frame, text="Heart Rate Analysis", 
                  command=self.show_heart_rate_analysis).pack(side=tk.LEFT, padx=(0, 10))
        
        # Video export controls - separate frame for better organization
        video_frame = ttk.Frame(analysis_frame)
        video_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(video_frame, text="Save MP4 (Image Only)", 
                  command=self.save_mp4).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(video_frame, text="Save MP4 with Signal", 
                  command=self.save_mp4_with_signal).pack(side=tk.LEFT, padx=(0, 10))

        # ==================== MOUSE EAR VESSEL ANALYSIS PANEL ====================
        ear_frame = ttk.LabelFrame(control_frame, text="Mouse Ear Vessel Analysis")
        ear_frame.pack(fill=tk.X, padx=5, pady=5)

        # Row 1: fluid / geometry parameters
        ear_params_frame = ttk.Frame(ear_frame)
        ear_params_frame.pack(fill=tk.X, pady=3)

        ttk.Label(ear_params_frame, text="Cross-sections:").pack(side=tk.LEFT)
        self.n_cross_sections_var = tk.IntVar(value=20)
        ttk.Spinbox(ear_params_frame, from_=3, to=200, width=5,
                    textvariable=self.n_cross_sections_var,
                    command=self.update_ear_params).pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(ear_params_frame, text="Profile half-len (px):").pack(side=tk.LEFT)
        self.profile_half_len_var = tk.IntVar(value=20)
        ttk.Spinbox(ear_params_frame, from_=5, to=200, width=5,
                    textvariable=self.profile_half_len_var,
                    command=self.update_ear_params).pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(ear_params_frame, text="Diameter def:").pack(side=tk.LEFT)
        self.diameter_method_var = tk.StringVar(value="fwhm")
        diam_combo = ttk.Combobox(ear_params_frame, textvariable=self.diameter_method_var,
                                  width=10, state='readonly')
        diam_combo['values'] = ('fwhm', 'threshold')
        diam_combo.pack(side=tk.LEFT, padx=(5, 0))
        diam_combo.bind('<<ComboboxSelected>>', self.update_ear_params)

        # Row 2: blood properties
        ear_fluid_frame = ttk.Frame(ear_frame)
        ear_fluid_frame.pack(fill=tk.X, pady=3)

        ttk.Label(ear_fluid_frame, text="Blood viscosity (cP):").pack(side=tk.LEFT)
        self.blood_viscosity_var = tk.DoubleVar(value=3.5)
        ttk.Entry(ear_fluid_frame, textvariable=self.blood_viscosity_var,
                  width=6).pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(ear_fluid_frame, text="Blood density (kg/m³):").pack(side=tk.LEFT)
        self.blood_density_var = tk.DoubleVar(value=1060.0)
        ttk.Entry(ear_fluid_frame, textvariable=self.blood_density_var,
                  width=7).pack(side=tk.LEFT, padx=(5, 0))

        # Row 3: analysis buttons
        ear_buttons_frame = ttk.Frame(ear_frame)
        ear_buttons_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(ear_buttons_frame, text="1. Vessel Morphometry (FWHM)",
                   command=self.analyze_vessel_morphometry).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ear_buttons_frame, text="2. Flow Velocity (Kymo-slope)",
                   command=self.analyze_flow_velocity).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ear_buttons_frame, text="3. Flow Rate + Regime",
                   command=self.analyze_flow_rate_and_regime).pack(side=tk.LEFT, padx=(0, 6))

        ear_buttons_frame2 = ttk.Frame(ear_frame)
        ear_buttons_frame2.pack(fill=tk.X, pady=(3, 0))

        ttk.Button(ear_buttons_frame2, text="4. Vasomotion (Low-freq)",
                   command=self.analyze_vasomotion).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ear_buttons_frame2, text="5. Summary Figure Panel",
                   command=self.generate_summary_figure).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ear_buttons_frame2, text="6. Flow Speed Map (Δframe peak)",
                   command=self.analyze_flow_speed_map).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ear_buttons_frame2, text="Export Results (CSV)",
                   command=self.export_vessel_results).pack(side=tk.LEFT, padx=(0, 6))

        # Row 3b: vessel-mask / perfusion visualisations (Parts 0-3)
        ear_perf_frame = ttk.Frame(ear_frame)
        ear_perf_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(ear_perf_frame, text="Vessel Mask…",
                   command=self.open_vessel_mask_editor).pack(side=tk.LEFT, padx=(0, 6))

        # Row 4: breathing / motion frame exclusion
        ear_breath_frame = ttk.Frame(ear_frame)
        ear_breath_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(ear_breath_frame, text="Breathing exclusion  σ:").pack(side=tk.LEFT)
        self.breathing_sigma_var = tk.DoubleVar(value=3.0)
        ttk.Entry(ear_breath_frame, textvariable=self.breathing_sigma_var,
                  width=5).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Button(ear_breath_frame, text="Detect Breathing Frames",
                   command=self.detect_breathing_frames).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ear_breath_frame, text="Exclude Current",
                   command=self.exclude_current_frame).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ear_breath_frame, text="Re-include Current",
                   command=self.include_current_frame).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ear_breath_frame, text="Clear Excluded",
                   command=self.clear_excluded_frames).pack(side=tk.LEFT, padx=(0, 6))
        self.breathing_status_var = tk.StringVar(value="no frames excluded")
        ttk.Label(ear_breath_frame, textvariable=self.breathing_status_var,
                  foreground="#b00").pack(side=tk.LEFT, padx=(8, 0))

        # Row 5: Δframe flow-speed seeded peak-tracking controls (Panel 6)
        ear_flow_frame = ttk.Frame(ear_frame)
        ear_flow_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(ear_flow_frame, text="Seed end:").pack(side=tk.LEFT)
        self.flow_seed_end_var = tk.StringVar(value="auto")
        ttk.Combobox(ear_flow_frame, textvariable=self.flow_seed_end_var, width=10,
                     state="readonly",
                     values=("auto", "proximal", "distal")).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow_frame, text="method:").pack(side=tk.LEFT)
        self.flow_method_var = tk.StringVar(value="dp")
        ttk.Combobox(ear_flow_frame, textvariable=self.flow_method_var, width=7,
                     state="readonly", values=("dp", "greedy")).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow_frame, text="v_max(mm/s):").pack(side=tk.LEFT)
        self.flow_vmax_var = tk.StringVar(value="auto")
        ttk.Entry(ear_flow_frame, textvariable=self.flow_vmax_var, width=6).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow_frame, text="v_min:").pack(side=tk.LEFT)
        self.flow_vmin_var = tk.DoubleVar(value=0.0)
        ttk.Entry(ear_flow_frame, textvariable=self.flow_vmin_var, width=5).pack(side=tk.LEFT, padx=(4, 0))

        # Row 6: seeded-tracking tuning parameters (Panel 6)
        ear_flow2_frame = ttk.Frame(ear_frame)
        ear_flow2_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(ear_flow2_frame, text="prominence(σ):").pack(side=tk.LEFT)
        self.flow_prom_var = tk.DoubleVar(value=3.0)
        ttk.Entry(ear_flow2_frame, textvariable=self.flow_prom_var, width=5).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow2_frame, text="max miss(frames):").pack(side=tk.LEFT)
        self.flow_maxmiss_var = tk.IntVar(value=5)
        ttk.Entry(ear_flow2_frame, textvariable=self.flow_maxmiss_var, width=5).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow2_frame, text="Δmap smooth:").pack(side=tk.LEFT)
        self.flow_smooth_var = tk.DoubleVar(value=1.0)
        ttk.Entry(ear_flow2_frame, textvariable=self.flow_smooth_var, width=5).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow2_frame, text="seed window(µm):").pack(side=tk.LEFT)
        self.flow_seedwin_var = tk.StringVar(value="auto")
        ttk.Entry(ear_flow2_frame, textvariable=self.flow_seedwin_var, width=6).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Button(ear_flow2_frame, text="Δframe Inspector",
                   command=self.open_dframe_inspector).pack(side=tk.LEFT, padx=(4, 0))

        # Row 7: flow-speed METHOD selection + arrival-time (transit) controls
        ear_flow3_frame = ttk.Frame(ear_frame)
        ear_flow3_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(ear_flow3_frame, text="Flow method:").pack(side=tk.LEFT)
        self.flow_method_kind_var = tk.StringVar(value="dframe_peak")
        ttk.Combobox(ear_flow3_frame, textvariable=self.flow_method_kind_var, width=12,
                     state="readonly",
                     values=("dframe_peak", "arrival_time", "waypoint", "both")).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow3_frame, text="timing:").pack(side=tk.LEFT)
        self.flow_timing_feature_var = tk.StringVar(value="t50")
        ttk.Combobox(ear_flow3_frame, textvariable=self.flow_timing_feature_var, width=9,
                     state="readonly",
                     values=("t50", "max_slope", "xcorr", "ttp")).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow3_frame, text="arrival dir:").pack(side=tk.LEFT)
        self.flow_arrival_dir_var = tk.StringVar(value="auto")
        ttk.Combobox(ear_flow3_frame, textvariable=self.flow_arrival_dir_var, width=6,
                     state="readonly", values=("auto", "+", "-")).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow3_frame, text="baseline pct:").pack(side=tk.LEFT)
        self.flow_arrival_bpct_var = tk.DoubleVar(value=10.0)
        ttk.Entry(ear_flow3_frame, textvariable=self.flow_arrival_bpct_var, width=5).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(ear_flow3_frame, text="t-smooth:").pack(side=tk.LEFT)
        self.flow_arrival_smooth_var = tk.IntVar(value=5)
        ttk.Entry(ear_flow3_frame, textvariable=self.flow_arrival_smooth_var, width=4).pack(side=tk.LEFT, padx=(4, 0))

        # Row 8: arrival-time trend-outlier rejection + local-speed window (analysis)
        ear_flow4_frame = ttk.Frame(ear_frame)
        ear_flow4_frame.pack(fill=tk.X, pady=(3, 0))
        self.flow_trend_reject_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ear_flow4_frame, text="trend reject",
                        variable=self.flow_trend_reject_var).pack(side=tk.LEFT)
        ttk.Label(ear_flow4_frame, text="k:").pack(side=tk.LEFT, padx=(6, 0))
        self.flow_trend_k_var = tk.DoubleVar(value=2.5)
        ttk.Entry(ear_flow4_frame, textvariable=self.flow_trend_k_var, width=4).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(ear_flow4_frame, text="max resid(s):").pack(side=tk.LEFT)
        self.flow_max_resid_var = tk.StringVar(value="")
        ttk.Entry(ear_flow4_frame, textvariable=self.flow_max_resid_var, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(ear_flow4_frame, text="local win(µm):").pack(side=tk.LEFT)
        self.flow_local_win_var = tk.StringVar(value="auto")
        ttk.Entry(ear_flow4_frame, textvariable=self.flow_local_win_var, width=7).pack(side=tk.LEFT, padx=(2, 0))

        # Row 9: adaptive local-speed estimator params (analysis)
        ear_flow5_frame = ttk.Frame(ear_frame)
        ear_flow5_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(ear_flow5_frame, text="min span(µm):").pack(side=tk.LEFT)
        self.flow_min_span_var = tk.StringVar(value="auto")
        ttk.Entry(ear_flow5_frame, textvariable=self.flow_min_span_var, width=7).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(ear_flow5_frame, text="w_max(µm):").pack(side=tk.LEFT)
        self.flow_w_max_var = tk.StringVar(value="auto")
        ttk.Entry(ear_flow5_frame, textvariable=self.flow_w_max_var, width=7).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(ear_flow5_frame, text="min pts:").pack(side=tk.LEFT)
        self.flow_local_min_pts_var = tk.IntVar(value=4)
        ttk.Entry(ear_flow5_frame, textvariable=self.flow_local_min_pts_var, width=4).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(ear_flow5_frame, text="edge frames:").pack(side=tk.LEFT)
        self.flow_edge_frames_var = tk.IntVar(value=2)
        ttk.Entry(ear_flow5_frame, textvariable=self.flow_edge_frames_var, width=4).pack(side=tk.LEFT, padx=(2, 10))
        self.flow_shrink_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ear_flow5_frame, text="shrink→global",
                        variable=self.flow_shrink_var).pack(side=tk.LEFT)
        ttk.Label(ear_flow5_frame, text="  k_profile(v_mean/v_front):").pack(side=tk.LEFT)
        self.flow_k_profile_var = tk.DoubleVar(value=0.6)
        ttk.Combobox(ear_flow5_frame, textvariable=self.flow_k_profile_var, width=6,
                     values=("0.6", "0.5", "0.7", "1.0")).pack(side=tk.LEFT, padx=(2, 0))

        # Row 10: waypoint method controls + persistence (analysis)
        ear_flow6_frame = ttk.Frame(ear_frame)
        ear_flow6_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(ear_flow6_frame, text="WP radius:").pack(side=tk.LEFT)
        self.flow_wp_radius_var = tk.StringVar(value="auto")
        ttk.Entry(ear_flow6_frame, textvariable=self.flow_wp_radius_var, width=5).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(ear_flow6_frame, text="stat:").pack(side=tk.LEFT)
        self.flow_wp_stat_var = tk.StringVar(value="median")
        ttk.Combobox(ear_flow6_frame, textvariable=self.flow_wp_stat_var, width=7, state="readonly",
                     values=("median", "mean")).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(ear_flow6_frame, text="n_cand:").pack(side=tk.LEFT)
        self.flow_wp_ncand_var = tk.IntVar(value=5)
        ttk.Entry(ear_flow6_frame, textvariable=self.flow_wp_ncand_var, width=4).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(ear_flow6_frame, text="assign:").pack(side=tk.LEFT)
        self.flow_wp_assign_var = tk.StringVar(value="dp")
        ttk.Combobox(ear_flow6_frame, textvariable=self.flow_wp_assign_var, width=7, state="readonly",
                     values=("dp", "greedy")).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(ear_flow6_frame, text="Choose WPs…", command=self.choose_waypoints).pack(side=tk.LEFT, padx=2)
        ttk.Button(ear_flow6_frame, text="Save WP…", command=self.save_waypoints).pack(side=tk.LEFT, padx=2)
        ttk.Button(ear_flow6_frame, text="Load WP…", command=self.load_waypoints).pack(side=tk.LEFT, padx=2)
        ttk.Button(ear_flow6_frame, text="Snap", command=self.snap_waypoints_to_vessel).pack(side=tk.LEFT, padx=2)
        ttk.Button(ear_flow6_frame, text="Undo snap", command=self.undo_snap_waypoints).pack(side=tk.LEFT, padx=2)
        # ========================================================================
        
        # Image display
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Display")
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for image display
        self.canvas = tk.Canvas(self.image_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for ROI drawing
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self.finish_roi_drawing)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        
        # Bind mouse wheel for zoom
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux
        
        # Bind middle mouse button for panning
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)
        
        # Bind canvas resize
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Make canvas focusable for mouse wheel events
        self.canvas.focus_set()
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-r>', self.reset_view)
        self.root.bind('<Control-0>', self.reset_view)
        self.root.bind('<space>', lambda e: self.toggle_play())  # 空格键播放/暂停
        self.root.bind('<Control-d>', lambda e: self.debug_progress_lines())  # Ctrl+D调试进度线
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready | Mouse wheel: zoom | Middle click: pan | Ctrl+R: reset view | Space: play/pause | Ctrl+D: debug")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def toggle_play(self):
        """切换播放状态"""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load a TIF stack first")
            return
            
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """开始播放"""
        self.is_playing = True
        self.play_button.config(text="⏸ Pause")
        self.update_play_range_info()
        self.play_next_frame()
        
        # 更新状态显示
        start_frame, end_frame = self.get_play_range()
        range_text = f"Playing frames {start_frame}-{end_frame}"
        self.status_var.set(range_text + " | Space: pause")
    
    def stop_playback(self):
        """停止播放"""
        self.is_playing = False
        self.play_button.config(text="▶ Play")
        
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None
            
        self.status_var.set("Playback stopped | Space: play")
    
    def play_next_frame(self):
        """播放下一帧"""
        if not self.is_playing:
            return
            
        start_frame, end_frame = self.get_play_range()
        
        # 如果当前帧超出播放范围，回到开始
        if self.current_frame < start_frame or self.current_frame >= end_frame:
            self.current_frame = start_frame
        else:
            self.current_frame += 1
            if self.current_frame >= end_frame:
                self.current_frame = start_frame  # 循环播放
        
        # 更新界面
        self.frame_var.set(self.current_frame)
        self.frame_label.config(text=f"{self.current_frame}/{len(self.image_stack)-1}")
        self.update_marker_info()
        self.display_current_frame()
        
        # 更新所有分析窗口的进度线
        if self.analysis_windows:  # 只有当有窗口时才更新
            self.update_all_progress_lines()
        
        # 调度下一次更新
        self.play_job = self.root.after(self.play_speed_var.get(), self.play_next_frame)
    
    def get_play_range(self):
        """获取播放范围"""
        if self.image_stack is None:
            return 0, 0
            
        total_frames = len(self.image_stack)
        
        # 如果设置了marker，使用marker范围，否则使用全部范围
        if self.start_marker is not None or self.end_marker is not None:
            start_frame = self.start_marker if self.start_marker is not None else 0
            end_frame = self.end_marker if self.end_marker is not None else total_frames - 1
        else:
            start_frame = 0
            end_frame = total_frames - 1
            
        # 确保范围有效
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        
        return start_frame, end_frame + 1  # +1 for range()
    
    def update_play_range_info(self):
        """更新播放范围信息"""
        start_frame, end_frame = self.get_play_range()
        end_frame -= 1  # 调整显示
        
        if self.start_marker is not None or self.end_marker is not None:
            self.play_range_var.set(f"Playing frames {start_frame}-{end_frame}")
        else:
            self.play_range_var.set(f"Playing full range (0-{len(self.image_stack)-1})")
    
    def update_play_speed(self):
        """更新播放速度"""
        # 播放速度会在下次调度时生效
        pass
    
    def debug_progress_lines(self):
        """调试进度线状态"""
        print(f"=== PROGRESS LINES DEBUG ===")
        print(f"Total analysis windows: {len(self.analysis_windows)}")
        print(f"Current frame: {self.current_frame}")
        print(f"Current time: {self.current_frame / self.frame_rate:.3f}s")
        
        for i, window_info in enumerate(self.analysis_windows):
            try:
                window_info['window'].winfo_exists()
                print(f"Window {i}: EXISTS")
                print(f"  - Axes count: {len(window_info['axes'])}")
                print(f"  - Lines count: {len(window_info['lines'])}")
                print(f"  - Canvas: {type(window_info['canvas'])}")
            except tk.TclError:
                print(f"Window {i}: DESTROYED")
            except Exception as e:
                print(f"Window {i}: ERROR - {e}")
        print("==========================")
    
    def update_all_progress_lines(self):
        """更新所有分析窗口的进度线"""
        current_time = self.current_frame / self.frame_rate
        
        # 清理已关闭的窗口 - 使用更安全的检查方法
        valid_windows = []
        for w in self.analysis_windows:
            try:
                # 检查窗口是否仍然存在
                w['window'].winfo_exists()
                valid_windows.append(w)
            except tk.TclError:
                # 窗口已被销毁
                continue
        
        self.analysis_windows = valid_windows
        
        for window_info in self.analysis_windows:
            try:
                for i, ax in enumerate(window_info['axes']):
                    if i < len(window_info['lines']):
                        # 更新进度线位置
                        line = window_info['lines'][i]
                        line.set_xdata([current_time, current_time])
                        
                        # 获取y轴范围并更新
                        ylim = ax.get_ylim()
                        line.set_ydata(ylim)
                
                # 强制刷新画布
                try:
                    window_info['canvas'].draw()
                    window_info['canvas'].flush_events()
                except:
                    # 如果canvas已经被销毁，忽略错误
                    pass
                
            except Exception as e:
                print(f"Error updating progress line: {e}")
                continue
    
    def add_progress_lines_to_window(self, window, fig, axes_with_time, canvas=None):
        """为窗口添加进度线"""
        try:
            current_time = self.current_frame / self.frame_rate
            progress_lines = []
            
            for ax in axes_with_time:
                # 添加红色虚线作为进度线
                ylim = ax.get_ylim()
                line = ax.axvline(x=current_time, color='red', linestyle='--', 
                                linewidth=2, alpha=0.8, zorder=10)
                progress_lines.append(line)
            
            # 如果没有传入canvas，尝试搜索
            if canvas is None:
                def find_canvas_recursive(widget):
                    try:
                        # 检查widget本身是否是FigureCanvasTkAgg
                        if isinstance(widget, FigureCanvasTkAgg):
                            return widget
                        
                        # 检查children
                        for child in widget.winfo_children():
                            result = find_canvas_recursive(child)
                            if result:
                                return result
                                
                    except Exception as e:
                        print(f"Error checking widget {widget}: {e}")
                        
                    return None
                
                canvas = find_canvas_recursive(window)
            
            if canvas:
                self._complete_progress_line_setup(window, axes_with_time, progress_lines, canvas)
            else:
                print("Warning: Could not find canvas in window")
                
        except Exception as e:
            print(f"Error adding progress lines: {e}")
            import traceback
            traceback.print_exc()
    
    def _complete_progress_line_setup(self, window, axes_with_time, progress_lines, canvas):
        """完成进度线设置"""
        try:
            # 存储窗口信息
            window_info = {
                'window': window,
                'axes': axes_with_time,
                'lines': progress_lines,
                'canvas': canvas
            }
            self.analysis_windows.append(window_info)
            
            # 绑定窗口关闭事件
            def on_window_close():
                try:
                    # 从列表中移除
                    if window_info in self.analysis_windows:
                        self.analysis_windows.remove(window_info)
                except:
                    pass
                try:
                    window.destroy()
                except:
                    pass
            
            window.protocol("WM_DELETE_WINDOW", on_window_close)
            
            # 初始绘制
            canvas.draw()
            
            print(f"Successfully added progress lines to {len(axes_with_time)} axes")
            
        except Exception as e:
            print(f"Error completing progress line setup: {e}")
    
    def update_scale_settings(self, event=None):
        """Update scale settings"""
        try:
            self.pixel_size = self.pixel_size_var.get()
            self.frame_rate = self.frame_rate_var.get()
            
            # 更新播放范围信息
            self.update_play_range_info()
            # 更新所有进度线
            if self.analysis_windows:  # 只有当有窗口时才更新
                self.update_all_progress_lines()
        except:
            pass  # Ignore invalid input during typing
    
    def load_tif_stack(self):
        """Load TIF stack file"""
        file_path = filedialog.askopenfilename(
            title="Select TIF Stack",
            filetypes=[("TIF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # 保存TIF文件路径
            self.tif_file_path = file_path
            
            # Load TIF stack
            img = Image.open(file_path)
            frames = []
            
            try:
                while True:
                    frames.append(np.array(img))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            
            self.image_stack = np.array(frames)
            
            # Check if we have existing ROIs to preserve
            has_existing_setup = (self.background_roi_1 is not None or 
                                self.background_roi_2 is not None or 
                                self.foreground_roi is not None or
                                self.start_marker is not None or 
                                self.end_marker is not None)
            
            if has_existing_setup:
                self.status_var.set(f"Loaded {len(frames)} frames, shape: {self.image_stack.shape} | ROIs and markers preserved")
            else:
                self.status_var.set(f"Loaded {len(frames)} frames, shape: {self.image_stack.shape}")
            
            # Update frame controls
            self.frame_scale.configure(to=len(frames)-1)
            self.frame_var.set(0)
            self.current_frame = 0
            self.frame_label.config(text=f"0/{len(frames)-1}")
            
            # Keep existing ROIs and markers - but validate markers against new image size
            # This allows users to reload images while preserving their analysis setup
            if self.start_marker is not None and self.start_marker >= len(frames):
                self.start_marker = len(frames) - 1
                print(f"Adjusted start_marker to {self.start_marker} (image size: {len(frames)})")
            
            if self.end_marker is not None and self.end_marker >= len(frames):
                self.end_marker = len(frames) - 1
                print(f"Adjusted end_marker to {self.end_marker} (image size: {len(frames)})")
            
            # Clear cached signals to force regeneration
            self.foreground_signal = None
            self.background_signal = None
            self.difference_signal = None
            
            # Update marker info display
            self.update_marker_info()
            self.update_play_range_info()
            
            self.display_current_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load TIF stack: {str(e)}")
    
    def update_frame(self, value=None):
        """Update current frame display"""
        if self.image_stack is None:
            return
        self.min_value = self.min_var.get()
        self.max_value = self.max_var.get()
        self.current_frame = int(self.frame_var.get())
        self.frame_label.config(text=f"{self.current_frame}/{len(self.image_stack)-1}")
        self.update_marker_info()  # Update marker info when frame changes
        self.display_current_frame()
        
        # 更新所有分析窗口的进度线
        if self.analysis_windows:  # 只有当有窗口时才更新
            self.update_all_progress_lines()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if self.image_stack is None:
            return
            
        # Get mouse position
        mouse_x = self.canvas.canvasx(event.x)
        mouse_y = self.canvas.canvasy(event.y)
        
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:  # Zoom in
            zoom_change = 1.2
        else:  # Zoom out
            zoom_change = 1.0 / 1.2
        
        # Update zoom factor
        old_zoom = self.zoom_factor
        self.zoom_factor *= zoom_change
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))  # Limit zoom range
        
        # Adjust pan to zoom around mouse cursor
        zoom_ratio = self.zoom_factor / old_zoom
        canvas_center_x = self.canvas.winfo_width() / 2
        canvas_center_y = self.canvas.winfo_height() / 2
        
        self.pan_x = canvas_center_x + (self.pan_x - canvas_center_x) * zoom_ratio + (mouse_x - canvas_center_x) * (1 - zoom_ratio)
        self.pan_y = canvas_center_y + (self.pan_y - canvas_center_y) * zoom_ratio + (mouse_y - canvas_center_y) * (1 - zoom_ratio)
        
        self.display_current_frame()
    
    def start_pan(self, event):
        """Start panning with middle mouse button"""
        self.last_pan_x = event.x
        self.last_pan_y = event.y
    
    def do_pan(self, event):
        """Pan the image"""
        if hasattr(self, 'last_pan_x'):
            dx = event.x - self.last_pan_x
            dy = event.y - self.last_pan_y
            self.pan_x += dx
            self.pan_y += dy
            self.last_pan_x = event.x
            self.last_pan_y = event.y
            self.display_current_frame()
    
    def end_pan(self, event):
        """End panning"""
        if hasattr(self, 'last_pan_x'):
            delattr(self, 'last_pan_x')
        if hasattr(self, 'last_pan_y'):
            delattr(self, 'last_pan_y')
    
    def set_start_marker(self):
        """Set start marker at current frame"""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load a TIF stack first")
            return
        
        # Ensure current_frame is within bounds
        max_frame = len(self.image_stack) - 1
        if self.current_frame > max_frame:
            self.current_frame = max_frame
            self.frame_var.set(self.current_frame)
            
        self.start_marker = self.current_frame
        self.update_marker_info()
        self.update_play_range_info()
        self.status_var.set(f"Start marker set at frame {self.current_frame} ({self.current_frame * 1000 / self.frame_rate:.1f} ms)")
        
        # Clear cached signals to force regeneration with new range
        self.foreground_signal = None
        self.background_signal = None
        self.difference_signal = None
    
    def set_end_marker(self):
        """Set end marker at current frame"""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load a TIF stack first")
            return
        
        # Ensure current_frame is within bounds
        max_frame = len(self.image_stack) - 1
        if self.current_frame > max_frame:
            self.current_frame = max_frame
            self.frame_var.set(self.current_frame)
            
        self.end_marker = self.current_frame
        self.update_marker_info()
        self.update_play_range_info()
        self.status_var.set(f"End marker set at frame {self.current_frame} ({self.current_frame * 1000 / self.frame_rate:.1f} ms)")
        
        # Clear cached signals to force regeneration with new range
        self.foreground_signal = None
        self.background_signal = None
        self.difference_signal = None
    
    def clear_markers(self):
        """Clear all time markers"""
        self.start_marker = None
        self.end_marker = None
        self.update_marker_info()
        self.update_play_range_info()
        self.status_var.set("Time markers cleared - using full range")
        
        # Clear cached signals
        self.foreground_signal = None
        self.background_signal = None
        self.difference_signal = None
        
        # Debug output
        print("Markers cleared, full range will be used")
    
    def update_marker_info(self):
        """Update marker information display"""
        if self.start_marker is None and self.end_marker is None:
            self.marker_info_var.set("No markers set - using full time range")
        elif self.start_marker is not None and self.end_marker is not None:
            start_time = self.start_marker * 1000 / self.frame_rate
            end_time = self.end_marker * 1000 / self.frame_rate
            duration = end_time - start_time
            self.marker_info_var.set(f"Range: {start_time:.1f}-{end_time:.1f} ms (duration: {duration:.1f} ms)")
        elif self.start_marker is not None:
            start_time = self.start_marker * 1000 / self.frame_rate
            self.marker_info_var.set(f"Start: {start_time:.1f} ms - End marker not set")
        else:
            end_time = self.end_marker * 1000 / self.frame_rate
            self.marker_info_var.set(f"Start marker not set - End: {end_time:.1f} ms")
    
    def get_analysis_frame_range(self):
        """Get the frame range for analysis based on markers"""
        if self.image_stack is None:
            return 0, 0
        
        total_frames = len(self.image_stack)
        
        if self.start_marker is None and self.end_marker is None:
            return 0, total_frames - 1
        
        start_frame = self.start_marker if self.start_marker is not None else 0
        end_frame = self.end_marker if self.end_marker is not None else total_frames - 1
        
        # Ensure valid range - critical fix for index bounds
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        
        # Additional safety check - ensure end_frame is actually valid
        if end_frame >= total_frames:
            end_frame = total_frames - 1
        if start_frame >= total_frames:
            start_frame = total_frames - 1
        
        # Ensure start <= end
        if start_frame > end_frame:
            start_frame = end_frame
        
        return start_frame, end_frame
    
    def debug_frame_info(self):
        """Debug function to print frame information"""
        if self.image_stack is not None:
            total_frames = len(self.image_stack)
            start_frame, end_frame = self.get_analysis_frame_range()
            print("=== FRAME DEBUG INFO ===")
            print(f"Total frames in stack: {total_frames}")
            print(f"Valid frame indices: 0 to {total_frames - 1}")
            print(f"Current frame: {self.current_frame}")
            print(f"Start marker: {self.start_marker}")
            print(f"End marker: {self.end_marker}")
            print(f"Analysis range: {start_frame} to {end_frame}")
            print(f"Range length: {end_frame - start_frame + 1}")
            print("======================")
        else:
            print("No image stack loaded")
    
    def reset_view(self, event=None):
        """Reset zoom and pan to default (Ctrl+R or Ctrl+0)"""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.display_current_frame()
        self.status_var.set("View reset to default | Zoom: 1.0x")
    
    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        if self.image_stack is not None:
            self.display_current_frame()
    
    def display_current_frame(self):
        """Display current frame with ROIs, zoom, pan, and scalebar"""
        if self.image_stack is None:
            return
            
        # Get current frame
        frame = self.image_stack[self.current_frame].copy()
        
        # Normalize for display
        # if frame.dtype != np.uint8:
        #     frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)

        max_value = np.iinfo(frame.dtype).max
        frame = np.double(frame) / np.double(max_value)
        frame = np.clip(frame, self.min_value, self.max_value)
        frame = (frame - self.min_value) / (self.max_value - self.min_value)* 255
        frame = frame.astype(np.uint8)
        
        # Convert to RGB for ROI overlay
        if len(frame.shape) == 2:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            display_frame = frame.copy()
        
        # Create separate overlays for background and foreground
        background_overlay = display_frame.copy()
        foreground_overlay = display_frame.copy()

        # Draw background ROIs (will be more transparent)
        if self.background_roi_1 is not None:
            # Draw line
            for i in range(len(self.background_roi_1) - 1):
                cv2.line(background_overlay, tuple(self.background_roi_1[i]), 
                        tuple(self.background_roi_1[i+1]), (0, 255, 0), 1)
            
            # Draw expanded area
            expanded_roi = self.create_expanded_line_roi(self.background_roi_1, self.background_line_width)
            if expanded_roi is not None:
                cv2.fillPoly(background_overlay, [expanded_roi], (0, 255, 0))

        if self.background_roi_2 is not None:
            # Draw line
            for i in range(len(self.background_roi_2) - 1):
                cv2.line(background_overlay, tuple(self.background_roi_2[i]), 
                        tuple(self.background_roi_2[i+1]), (0, 200, 100), 1)
            
            # Draw expanded area
            expanded_roi = self.create_expanded_line_roi(self.background_roi_2, self.background_line_width)
            if expanded_roi is not None:
                cv2.fillPoly(background_overlay, [expanded_roi], (0, 200, 100))

        # Draw foreground ROI (keep current transparency)
        if self.foreground_roi is not None:
            # Draw line
            for i in range(len(self.foreground_roi) - 1):
                cv2.line(foreground_overlay, tuple(self.foreground_roi[i]), 
                        tuple(self.foreground_roi[i+1]), (255, 0, 0), 1)
            
            # Draw expanded area
            expanded_roi = self.create_expanded_line_roi(self.foreground_roi, self.foreground_line_width)
            if expanded_roi is not None:
                cv2.fillPoly(foreground_overlay, [expanded_roi], (255, 0, 0))

        # Apply different transparency levels
        # Background: more transparent (lower alpha, e.g., 0.05 = 5% overlay)
        display_frame = cv2.addWeighted(display_frame, 0.95, background_overlay, 0.05, 0)
        # Foreground: keep current transparency (0.15 = 15% overlay)
        display_frame = cv2.addWeighted(display_frame, 0.85, foreground_overlay, 0.15, 0)

        # Waypoint markers: green = included in flow-speed analysis, grey = excluded.
        # Numbered so the user can match them to the 'Choose WPs…' list and the
        # 'show WP' selector in the flow-speed figure.
        if getattr(self, 'waypoints', None):
            for i, w in enumerate(self.waypoints):
                px, py = int(round(w['x'])), int(round(w['y']))
                on = bool(w.get('enabled', True))
                col = (0, 220, 0) if on else (150, 150, 150)
                if on:
                    cv2.circle(display_frame, (px, py), 3, col, -1)
                else:
                    cv2.circle(display_frame, (px, py), 3, col, 1)
                cv2.putText(display_frame, str(i), (px + 4, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)

        # Draw temporary drawing lines (more visible during drawing)
        for line in self.drawing_lines:
            for i in range(len(line) - 1):
                cv2.line(display_frame, tuple(line[i]), tuple(line[i+1]), (255, 255, 0), 1)
        
        # Add frame number and time info in top-left corner
        margin = 5
        text_y_start = margin + 20
        text_line_height = 15
        
        # Calculate current time
        current_time = self.current_frame / self.frame_rate
        total_frames = len(self.image_stack)
        
        # Add frame number text
        frame_text = f"Frame: {self.current_frame}/{total_frames-1}"
        cv2.putText(display_frame, frame_text, 
                   (margin, text_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add time text
        time_text = f"Time: {current_time:.3f}s"
        cv2.putText(display_frame, time_text, 
                   (margin, text_y_start + text_line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add scalebar (200 μm)
        if self.show_scalebar:
            scalebar_length_um = 200.0
            scalebar_length_pixels = int(scalebar_length_um / self.pixel_size)
            
            # Position scalebar in top-right corner
            margin_sb = 20
            scalebar_y = margin_sb + 10
            scalebar_x_end = display_frame.shape[1] - margin_sb
            scalebar_x_start = scalebar_x_end - scalebar_length_pixels
            
            # Draw scalebar
            cv2.line(display_frame, (scalebar_x_start, scalebar_y), 
                    (scalebar_x_end, scalebar_y), (255, 255, 255), 3)
            
            # Add text
            cv2.putText(display_frame, f"{scalebar_length_um:.0f} μm", 
                       (scalebar_x_start, scalebar_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(display_frame)
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Apply zoom
            img_width = int(pil_image.width * self.zoom_factor)
            img_height = int(pil_image.height * self.zoom_factor)
            
            if img_width > 0 and img_height > 0:
                pil_image = pil_image.resize((img_width, img_height), Image.LANCZOS)
            
            # Calculate display position with pan
            display_x = int(canvas_width/2 + self.pan_x - img_width/2)
            display_y = int(canvas_height/2 + self.pan_y - img_height/2)
        else:
            display_x = 0
            display_y = 0
        
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(display_x, display_y, image=self.photo, anchor=tk.NW)
        
        # Update status with zoom info
        zoom_info = f" | Zoom: {self.zoom_factor:.1f}x"
        current_status = self.status_var.get()
        if " | Zoom:" in current_status:
            base_status = current_status.split(" | Zoom:")[0]
        else:
            base_status = current_status
        self.status_var.set(base_status + zoom_info)
    
    def start_background_roi_1(self):
        """Start drawing first background ROI"""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load a TIF stack first")
            return
            
        self.drawing_mode = 'background1'
        self.temp_points = []
        self.drawing_lines = []
        self.status_var.set("Click to draw background ROI 1 line. Double-click to finish.")
    
    def start_background_roi_2(self):
        """Start drawing second background ROI"""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load a TIF stack first")
            return
            
        self.drawing_mode = 'background2'
        self.temp_points = []
        self.drawing_lines = []
        self.status_var.set("Click to draw background ROI 2 line. Double-click to finish.")
    
    def start_foreground_roi(self):
        """Start drawing foreground ROI"""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load a TIF stack first")
            return
            
        self.drawing_mode = 'foreground'
        self.temp_points = []
        self.drawing_lines = []
        self.status_var.set("Click to draw foreground ROI line. Double-click to finish.")
    
    def clear_rois(self):
        """Clear all ROIs"""
        self.background_roi_1 = None
        self.background_roi_2 = None
        self.foreground_roi = None
        self.temp_points = []
        self.drawing_lines = []
        self.drawing_mode = None
        self.status_var.set("All ROIs cleared")
        self.display_current_frame()
    
    def update_foreground_line_width(self):
        """Update foreground line width"""
        self.foreground_line_width = self.foreground_line_width_var.get()
        if self.foreground_roi is not None:
            self.display_current_frame()
    
    def update_background_line_width(self):
        """Update background line width"""
        self.background_line_width = self.background_line_width_var.get()
        if self.background_roi_1 is not None or self.background_roi_2 is not None:
            self.display_current_frame()
    
    def update_background_segments(self):
        """Update background segments"""
        self.background_segments = self.background_segments_var.get()
        # Clear cached signals to force regeneration
        self.foreground_signal = None
        self.background_signal = None
        self.difference_signal = None
    
    def toggle_filter(self):
        """Toggle filter application"""
        self.apply_filter = self.filter_enabled_var.get()
        self.update_filter_window()

    def toggle_outlier_removal(self):
        """Toggle outlier removal"""
        self.remove_outliers = self.outlier_removal_var.get()
        # Clear cached signals to force regeneration
        self.foreground_signal = None
        self.background_signal = None
        self.difference_signal = None
        
        if self.remove_outliers:
            self.status_var.set(f"Outlier removal enabled ({self.outlier_method}, threshold={self.outlier_threshold}σ)")
        else:
            self.status_var.set("Outlier removal disabled")

    def update_outlier_threshold(self):
        """Update outlier threshold"""
        self.outlier_threshold = self.outlier_threshold_var.get()
        if self.remove_outliers:
            # Clear cached signals to force regeneration
            self.foreground_signal = None
            self.background_signal = None
            self.difference_signal = None
            self.status_var.set(f"Outlier threshold updated to {self.outlier_threshold}σ")

    def update_outlier_method(self, event=None):
        """Update outlier detection method"""
        self.outlier_method = self.outlier_method_var.get()
        if self.remove_outliers:
            # Clear cached signals to force regeneration
            self.foreground_signal = None
            self.background_signal = None
            self.difference_signal = None
            self.status_var.set(f"Outlier method updated to {self.outlier_method}")

    def update_morph_size(self):
        """Update morphological filter size"""
        self.morph_size = self.morph_size_var.get()
        if self.remove_outliers and self.outlier_method == 'morphological':
            # Clear cached signals to force regeneration
            self.foreground_signal = None
            self.background_signal = None
            self.difference_signal = None
            self.status_var.set(f"Morphological filter size updated to {self.morph_size}")
    def apply_morphological_filter(self, signal, structure_size=5):
        """
        Apply morphological opening to remove pulse-like noise
        Excellent for breathing artifacts with known duration
        
        Parameters:
        -----------
        signal : numpy array
            Input signal
        structure_size : int
            Size of structuring element (should be larger than pulse width)
        
        Returns:
        --------
        cleaned_signal : numpy array
            Signal with pulse noise removed
        """
        from scipy.ndimage import grey_opening, grey_closing
        
        # Apply morphological opening (removes positive pulses/spikes)
        opened = grey_opening(signal, size=structure_size)
        
        # Apply morphological closing (removes negative pulses/dips)  
        cleaned = grey_closing(opened, size=structure_size)
        
        return cleaned
    def remove_outliers_from_signal(self, signal):
        """
        Remove outliers from signal and interpolate gaps
        Now supports moving window detection for consecutive outlier bursts
        
        Parameters:
        -----------
        signal : numpy array
            Input signal
            
        Returns:
        --------
        cleaned_signal : numpy array
            Signal with outliers removed and interpolated
        outlier_mask : numpy array
            Boolean mask indicating outlier positions
        """
        if len(signal) < 3:
            return signal, np.zeros(len(signal), dtype=bool)
        
        outlier_mask = np.zeros(len(signal), dtype=bool)
        
        if self.outlier_method == 'zscore':
            # Z-score method: outliers are points beyond threshold standard deviations
            mean = np.mean(signal)
            std = np.std(signal)
            if std > 0:
                z_scores = np.abs((signal - mean) / std)
                outlier_mask = z_scores > self.outlier_threshold
        
        elif self.outlier_method == 'iqr':
            # IQR method: outliers are beyond Q1 - threshold*IQR or Q3 + threshold*IQR
            q1 = np.percentile(signal, 25)
            q3 = np.percentile(signal, 75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - self.outlier_threshold * iqr
                upper_bound = q3 + self.outlier_threshold * iqr
                outlier_mask = (signal < lower_bound) | (signal > upper_bound)
        
        elif self.outlier_method == 'modified_z':
            # Modified Z-score using median absolute deviation (more robust)
            median = np.median(signal)
            mad = np.median(np.abs(signal - median))
            if mad > 0:
                modified_z_scores = 0.6745 * (signal - median) / mad
                outlier_mask = np.abs(modified_z_scores) > self.outlier_threshold
        
        elif self.outlier_method == 'moving_window':
            # Moving window method - better for consecutive outlier bursts (breathing artifacts)
            window_size = 7  # Analyze 7-frame windows
            for i in range(len(signal)):
                # Get window around point
                start = max(0, i - window_size // 2)
                end = min(len(signal), i + window_size // 2 + 1)
                window = signal[start:end]
                
                # Calculate local statistics (more robust to bursts)
                local_median = np.median(window)
                local_mad = np.median(np.abs(window - local_median))
                
                if local_mad > 0:
                    # Use MAD-based threshold
                    deviation = np.abs(signal[i] - local_median) / local_mad
                    # More sensitive threshold for burst detection
                    if deviation > self.outlier_threshold * 1.5:
                        outlier_mask[i] = True
        
        # If no outliers detected, return original signal
        if not np.any(outlier_mask):
            return signal.copy(), outlier_mask
        
        # Create cleaned signal
        cleaned_signal = signal.copy()
        
        # Get indices of valid (non-outlier) points
        valid_indices = np.where(~outlier_mask)[0]
        outlier_indices = np.where(outlier_mask)[0]
        
        # Count consecutive outliers for diagnostics
        if len(outlier_indices) > 0:
            consecutive_counts = []
            count = 1
            for i in range(1, len(outlier_indices)):
                if outlier_indices[i] == outlier_indices[i-1] + 1:
                    count += 1
                else:
                    consecutive_counts.append(count)
                    count = 1
            consecutive_counts.append(count)
            max_consecutive = max(consecutive_counts)
            if max_consecutive > 1:
                print(f"  Detected outlier bursts: max {max_consecutive} consecutive frames")
        
        # If too many outliers, return original signal
        if len(valid_indices) < 4:
            print(f"Warning: Too few valid points ({len(valid_indices)}) for interpolation")
            return signal.copy(), outlier_mask
        
        # Interpolate outlier values using valid points
        try:
            # Use cubic spline for smoother interpolation of burst gaps
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(valid_indices, signal[valid_indices], bc_type='natural')
            cleaned_signal[outlier_mask] = cs(outlier_indices)
            
        except Exception as e:
            # Fallback to Akima interpolation (better for bursts than linear)
            try:
                from scipy.interpolate import Akima1DInterpolator
                akima = Akima1DInterpolator(valid_indices, signal[valid_indices])
                cleaned_signal[outlier_mask] = akima(outlier_indices)
            except:
                # Last resort: linear interpolation
                try:
                    interp_func = interp1d(valid_indices, signal[valid_indices], 
                                        kind='linear', bounds_error=False, 
                                        fill_value='extrapolate')
                    cleaned_signal[outlier_mask] = interp_func(outlier_indices)
                except:
                    print(f"Warning: All interpolation failed: {e}")
                    return signal.copy(), outlier_mask
        
        return cleaned_signal, outlier_mask

    def remove_outliers_from_signal(self, signal):
        """
        Remove outliers from signal and interpolate gaps
        
        Parameters:
        -----------
        signal : numpy array
            Input signal
            
        Returns:
        --------
        cleaned_signal : numpy array
            Signal with outliers removed and interpolated
        outlier_mask : numpy array
            Boolean mask indicating outlier positions
        """
        if len(signal) < 3:
            return signal, np.zeros(len(signal), dtype=bool)
        
        outlier_mask = np.zeros(len(signal), dtype=bool)
        
        if self.outlier_method == 'zscore':
            # Z-score method: outliers are points beyond threshold standard deviations
            mean = np.mean(signal)
            std = np.std(signal)
            if std > 0:
                z_scores = np.abs((signal - mean) / std)
                outlier_mask = z_scores > self.outlier_threshold
        
        elif self.outlier_method == 'iqr':
            # IQR method: outliers are beyond Q1 - threshold*IQR or Q3 + threshold*IQR
            q1 = np.percentile(signal, 25)
            q3 = np.percentile(signal, 75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - self.outlier_threshold * iqr
                upper_bound = q3 + self.outlier_threshold * iqr
                outlier_mask = (signal < lower_bound) | (signal > upper_bound)
        
        elif self.outlier_method == 'modified_z':
            # Modified Z-score using median absolute deviation (more robust)
            median = np.median(signal)
            mad = np.median(np.abs(signal - median))
            if mad > 0:
                modified_z_scores = 0.6745 * (signal - median) / mad
                outlier_mask = np.abs(modified_z_scores) > self.outlier_threshold
        
        # If no outliers detected, return original signal
        if not np.any(outlier_mask):
            return signal.copy(), outlier_mask
        
        # Create cleaned signal
        cleaned_signal = signal.copy()
        
        # Get indices of valid (non-outlier) points
        valid_indices = np.where(~outlier_mask)[0]
        outlier_indices = np.where(outlier_mask)[0]
        
        # If too many outliers, return original signal
        if len(valid_indices) < 3:
            print(f"Warning: Too few valid points ({len(valid_indices)}) for interpolation")
            return signal.copy(), outlier_mask
        
        # Interpolate outlier values using valid points
        try:
            # Use linear interpolation for points within range
            interp_func = interp1d(valid_indices, signal[valid_indices], 
                                kind='linear', bounds_error=False, 
                                fill_value='extrapolate')
            
            # Replace outlier values with interpolated values
            cleaned_signal[outlier_mask] = interp_func(outlier_indices)
            
        except Exception as e:
            print(f"Warning: Interpolation failed: {e}")
            return signal.copy(), outlier_mask
        
        return cleaned_signal, outlier_mask
    
    def update_bg_subtract(self):
        """更新背景减除设置"""
        self.subtract_background = self.bg_subtract_var.get()
        print(f"Background subtraction: {self.subtract_background}")

    def update_norm_mode(self, event=None):
        """更新归一化模式"""
        self.normalization_mode = self.norm_mode_var.get()
        print(f"Normalization mode: {self.normalization_mode}")
        
    def update_filter_window(self):
        """Update median filter window size and apply if enabled"""
        try:
            new_window = self.filter_window_var.get()
            if new_window % 2 == 0:
                new_window += 1
                self.filter_window_var.set(new_window)
            
            if new_window != self.median_filter_window:
                self.median_filter_window = new_window
                self.apply_filter = self.filter_enabled_var.get()
                
                # Clear cached signals to force regeneration with new filter
                self.foreground_signal = None
                self.background_signal = None
                self.difference_signal = None
                
                # Update status to show filter change
                if self.apply_filter:
                    self.status_var.set(f"Filter window updated to {self.median_filter_window}")
        except tk.TclError:
            # Handle invalid input during typing
            pass
    
    def on_canvas_click(self, event):
        """Handle canvas click for ROI drawing"""
        if self.drawing_mode is None:
            return
            
        # Convert canvas coordinates to image coordinates
        x, y = self.canvas_to_image_coords(event.x, event.y)
        if x is not None and y is not None:
            self.temp_points.append([int(x), int(y)])
            
            # Update drawing visualization
            if len(self.temp_points) > 1:
                self.drawing_lines = [np.array(self.temp_points, dtype=np.int32)]
                self.display_current_frame()
    
    def on_canvas_drag(self, event):
        """Handle canvas drag"""
        pass
    
    def on_canvas_release(self, event):
        """Handle canvas release"""
        pass
    
    def on_canvas_motion(self, event):
        """Handle mouse motion during drawing"""
        if self.drawing_mode is not None and len(self.temp_points) > 0:
            x, y = self.canvas_to_image_coords(event.x, event.y)
            if x is not None and y is not None:
                # Show preview line
                preview_points = self.temp_points + [[int(x), int(y)]]
                self.drawing_lines = [np.array(preview_points, dtype=np.int32)]
                self.display_current_frame()
    
    def finish_roi_drawing(self, event):
        """Finish ROI drawing on double-click"""
        if self.drawing_mode is None or len(self.temp_points) < 2:
            return
            
        if self.drawing_mode == 'background1':
            self.background_roi_1 = np.array(self.temp_points, dtype=np.int32)
            self.status_var.set("Background ROI 1 created")
        elif self.drawing_mode == 'background2':
            self.background_roi_2 = np.array(self.temp_points, dtype=np.int32)
            self.status_var.set("Background ROI 2 created")
        elif self.drawing_mode == 'foreground':
            self.foreground_roi = np.array(self.temp_points, dtype=np.int32)
            # Also keep the clicks as first-class FLOAT waypoints (sub-pixel), an
            # additional view of the same points used by the 'waypoint' method.
            self.set_waypoints_from_points(self.temp_points)
            self.status_var.set("Foreground ROI created")
        
        self.drawing_mode = None
        self.temp_points = []
        self.drawing_lines = []
        self.display_current_frame()
    
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates with zoom and pan"""
        if self.image_stack is None:
            return None, None
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return None, None
            
        img_height, img_width = self.image_stack[0].shape[:2]
        
        # Account for zoom and pan
        zoomed_width = img_width * self.zoom_factor
        zoomed_height = img_height * self.zoom_factor
        
        # Calculate image position on canvas
        display_x = int(canvas_width/2 + self.pan_x - zoomed_width/2)
        display_y = int(canvas_height/2 + self.pan_y - zoomed_height/2)
        
        # Convert canvas coordinates to zoomed image coordinates
        img_x_zoomed = canvas_x - display_x
        img_y_zoomed = canvas_y - display_y
        
        # Convert to original image coordinates
        img_x = img_x_zoomed / self.zoom_factor
        img_y = img_y_zoomed / self.zoom_factor
        
        if 0 <= img_x < img_width and 0 <= img_y < img_height:
            return img_x, img_y
        
        return None, None
    
    def create_expanded_line_roi(self, line_points, width):
        """Create expanded ROI around line with given width"""
        if len(line_points) < 2:
            return None
            
        try:
            # Create a mask for the line with width
            img_shape = self.image_stack[0].shape[:2]
            mask = np.zeros(img_shape, dtype=np.uint8)
            
            for i in range(len(line_points) - 1):
                cv2.line(mask, tuple(line_points[i]), tuple(line_points[i+1]), 255, width)
            
            # Find contours of the expanded area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                return largest_contour.reshape(-1, 2)
            
        except Exception as e:
            print(f"Error creating expanded ROI: {e}")
        
        return None
    
    def extract_segmented_signals(self, exclude_breathing=False):
        """Extract segmented foreground and background signals within marker range.

        If `exclude_breathing` is True, breathing-flagged frames are dropped from
        the time axis (used by the videokymograph). Off by default so uniformly
        sampled analyses (FFT/heartbeat) keep their constant frame interval."""
        if (self.image_stack is None or self.foreground_roi is None or 
            self.background_roi_1 is None or self.background_roi_2 is None):
            return None, None
        
        # Get analysis frame range
        start_frame, end_frame = self.get_analysis_frame_range()
        total_frames = len(self.image_stack)
        
        # Critical fix: Ensure end_frame is within bounds for range() function
        if end_frame >= total_frames:
            end_frame = total_frames - 1
        if start_frame >= total_frames:
            start_frame = total_frames - 1
        if start_frame > end_frame:
            start_frame = end_frame
            
        # Debug output
        print(f"extract_segmented_signals: start={start_frame}, end={end_frame}, total={total_frames}")
        
        # Calculate line lengths
        def line_length(points):
            total = 0
            for i in range(len(points) - 1):
                dx = points[i+1][0] - points[i][0]
                dy = points[i+1][1] - points[i][1]
                total += np.sqrt(dx*dx + dy*dy)
            return total
        
        fg_length = line_length(self.foreground_roi)
        bg1_length = line_length(self.background_roi_1)
        bg2_length = line_length(self.background_roi_2)
        
        # Create segments
        foreground_signals = []
        background_signals = []
        
        # Use safe range - ensure we don't go beyond array bounds
        if exclude_breathing:
            frame_range = self._included_frames(start_frame, min(end_frame, total_frames - 1))
            if len(frame_range) < 1:
                frame_range = list(range(start_frame, min(end_frame + 1, total_frames)))
            print(f"Breathing exclusion ON: {len(frame_range)} frames kept "
                  f"({len(self.excluded_frames)} excluded globally)")
        else:
            frame_range = range(start_frame, min(end_frame + 1, total_frames))
        print(f"Actual frame range: {list(frame_range)[:5]}...{list(frame_range)[-5:] if len(list(frame_range)) > 5 else list(frame_range)}")
        
        for frame_idx in frame_range:
            # Double-check bounds
            if frame_idx >= total_frames:
                print(f"ERROR: frame_idx {frame_idx} >= total_frames {total_frames}")
                break
                
            frame = self.image_stack[frame_idx]
            
            # Extract foreground segments
            fg_segment_signals = []
            for seg_idx in range(self.background_segments):
                # Calculate position along foreground line
                target_distance = (seg_idx + 0.5) * fg_length / self.background_segments
                
                # Find corresponding point
                current_distance = 0
                seg_center = None
                
                for i in range(len(self.foreground_roi) - 1):
                    dx = self.foreground_roi[i+1][0] - self.foreground_roi[i][0]
                    dy = self.foreground_roi[i+1][1] - self.foreground_roi[i][1]
                    segment_length = np.sqrt(dx*dx + dy*dy)
                    
                    if current_distance + segment_length >= target_distance:
                        t = (target_distance - current_distance) / segment_length if segment_length > 0 else 0
                        seg_center = [
                            int(self.foreground_roi[i][0] + t * dx),
                            int(self.foreground_roi[i][1] + t * dy)
                        ]
                        break
                    
                    current_distance += segment_length
                
                if seg_center is not None:
                    # Create circular ROI
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, tuple(seg_center), self.foreground_line_width//2, 255, -1)
                    
                    fg_pixels = frame[mask > 0]
                    fg_segment_signals.append(np.mean(fg_pixels) if len(fg_pixels) > 0 else 0)
                else:
                    fg_segment_signals.append(0)
            
            # Extract background segments (average of both background lines)
            bg_segment_signals = []
            for seg_idx in range(self.background_segments):
                bg_values = []
                
                # Background 1
                target_distance = (seg_idx + 0.5) * bg1_length / self.background_segments
                current_distance = 0
                seg_center = None
                
                for i in range(len(self.background_roi_1) - 1):
                    dx = self.background_roi_1[i+1][0] - self.background_roi_1[i][0]
                    dy = self.background_roi_1[i+1][1] - self.background_roi_1[i][1]
                    segment_length = np.sqrt(dx*dx + dy*dy)
                    
                    if current_distance + segment_length >= target_distance:
                        t = (target_distance - current_distance) / segment_length if segment_length > 0 else 0
                        seg_center = [
                            int(self.background_roi_1[i][0] + t * dx),
                            int(self.background_roi_1[i][1] + t * dy)
                        ]
                        break
                    
                    current_distance += segment_length
                
                if seg_center is not None:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, tuple(seg_center), self.background_line_width//2, 255, -1)
                    bg_pixels = frame[mask > 0]
                    bg_values.append(np.mean(bg_pixels) if len(bg_pixels) > 0 else 0)
                
                # Background 2
                target_distance = (seg_idx + 0.5) * bg2_length / self.background_segments
                current_distance = 0
                seg_center = None
                
                for i in range(len(self.background_roi_2) - 1):
                    dx = self.background_roi_2[i+1][0] - self.background_roi_2[i][0]
                    dy = self.background_roi_2[i+1][1] - self.background_roi_2[i][1]
                    segment_length = np.sqrt(dx*dx + dy*dy)
                    
                    if current_distance + segment_length >= target_distance:
                        t = (target_distance - current_distance) / segment_length if segment_length > 0 else 0
                        seg_center = [
                            int(self.background_roi_2[i][0] + t * dx),
                            int(self.background_roi_2[i][1] + t * dy)
                        ]
                        break
                    
                    current_distance += segment_length
                
                if seg_center is not None:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, tuple(seg_center), self.background_line_width//2, 255, -1)
                    bg_pixels = frame[mask > 0]
                    bg_values.append(np.mean(bg_pixels) if len(bg_pixels) > 0 else 0)
                
                # Average background values and smooth
                bg_segment_signals.append(np.mean(bg_values) if bg_values else 0)
            
            # Smooth background signals to prevent jumps
            if len(bg_segment_signals) > 2:
                bg_segment_signals = ndimage.gaussian_filter1d(bg_segment_signals, sigma=1.0)
            
            foreground_signals.append(fg_segment_signals)
            background_signals.append(bg_segment_signals)
        
        # Convert to arrays and apply filtering if enabled
        fg_array = np.array(foreground_signals).T  # Shape: (segments, frames)
        bg_array = np.array(background_signals).T
        
        print(f"Final arrays shape: fg={fg_array.shape}, bg={bg_array.shape}")
        

        # Apply outlier removal first if enabled
        if self.remove_outliers:
            if self.outlier_method == 'morphological':
                # Use morphological filter for periodic pulse noise
                print(f"Applying morphological filter (size={self.morph_size})...")
                for i in range(fg_array.shape[0]):
                    fg_array[i] = self.apply_morphological_filter(fg_array[i], self.morph_size)
                    bg_array[i] = self.apply_morphological_filter(bg_array[i], self.morph_size)
                print("Morphological filtering complete")
            else:
                # Use statistical outlier detection + interpolation
                print(f"Applying {self.outlier_method} outlier detection (threshold={self.outlier_threshold}σ)...")
                total_outliers = 0
                for i in range(fg_array.shape[0]):
                    fg_array[i], fg_outliers = self.remove_outliers_from_signal(fg_array[i])
                    bg_array[i], bg_outliers = self.remove_outliers_from_signal(bg_array[i])
                    total_outliers += np.sum(fg_outliers) + np.sum(bg_outliers)
                
                if total_outliers > 0:
                    print(f"Removed and interpolated {total_outliers} outlier points")

        # Then apply median filter if enabled
        if self.apply_filter and self.median_filter_window > 1:
            for i in range(fg_array.shape[0]):
                fg_array[i] = medfilt(fg_array[i], kernel_size=self.median_filter_window)
                bg_array[i] = medfilt(bg_array[i], kernel_size=self.median_filter_window)
        return fg_array, bg_array
    
    def show_combined_signals(self):
        """Show combined foreground and background signals using segmented approach"""
        if (self.foreground_roi is None or self.background_roi_1 is None or self.background_roi_2 is None):
            messagebox.showwarning("Warning", "Please draw foreground and both background ROIs first")
            return
        
        # Extract segmented signals
        fg_signals, bg_signals = self.extract_segmented_signals()
        if fg_signals is None:
            messagebox.showerror("Error", "Failed to extract signals")
            return
        
        # Calculate average signals
        self.foreground_signal = np.mean(fg_signals, axis=0)
        self.background_signal = np.mean(bg_signals, axis=0)
        self.difference_signal = self.foreground_signal - self.background_signal
        
        # Create plot window
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Combined Signal Analysis")
        plot_window.geometry("1000x600")
        
        fig = Figure(figsize=(12, 8))
        
        # Convert time axis to ms (starting from analysis start)
        start_frame, end_frame = self.get_analysis_frame_range()
        time_axis = (np.arange(len(self.foreground_signal)) + start_frame) / self.frame_rate
        
        # Foreground signal
        ax1 = fig.add_subplot(221)
        ax1.plot(time_axis, self.foreground_signal, 'b-', linewidth=2, label='Foreground')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Mean Intensity')
        ax1.set_title('Foreground ROI Signal')
        ax1.grid(True, alpha=0.3)
        
        # Background signal
        ax2 = fig.add_subplot(222)
        ax2.plot(time_axis, self.background_signal, 'g-', linewidth=2, label='Background')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Mean Intensity')
        ax2.set_title('Background ROI Signal (Segmented Average)')
        ax2.grid(True, alpha=0.3)
        
        # Difference signal
        ax3 = fig.add_subplot(223)
        ax3.plot(time_axis, self.difference_signal, 'r-', linewidth=2, label='Foreground - Background')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Difference Signal (Foreground - Background)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Combined view
        ax4 = fig.add_subplot(224)
        ax4.plot(time_axis, self.foreground_signal, 'b-', linewidth=1, label='Foreground', alpha=0.7)
        ax4.plot(time_axis, self.background_signal, 'g-', linewidth=1, label='Background', alpha=0.7)
        ax4.plot(time_axis, self.difference_signal, 'r-', linewidth=2, label='Difference')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Intensity')
        ax4.set_title('Combined View')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add filter status with prominent display
        filter_status = f"Segmented background subtraction ({self.background_segments} segments). "

        if self.remove_outliers:
            if self.outlier_method == 'morphological':
                filter_status += f"MORPHOLOGICAL FILTER (size={self.morph_size}). "
            else:
                filter_status += f"OUTLIER REMOVAL ({self.outlier_method}, threshold={self.outlier_threshold}σ). "
        else:
            filter_status += "NO OUTLIER REMOVAL. "
            
        if self.apply_filter:
            filter_status += f"MEDIAN FILTER (window={self.median_filter_window})"
        else:
            filter_status += "NO MEDIAN FILTER"
        
        status_label = ttk.Label(plot_window, text=filter_status, font=('TkDefaultFont', 9, 'bold'))
        status_label.pack(pady=5)
        
        # 添加进度线到所有时间序列子图 - 在canvas创建后调用
        time_axes = [ax1, ax2, ax3, ax4]
        self.add_progress_lines_to_window(plot_window, fig, time_axes, canvas)
        
        self.roi_plots_open = True
    
    # ------------------------------------------------------------------
    # v2 additions: line-ROI persistence + Fig. 3f export (see fig3f_roi_trace.py)
    # ------------------------------------------------------------------
    def _fig3f_module(self):
        """fig3f_roi_trace.py 를 이 스크립트와 같은 폴더(또는 sys.path)에서 import."""
        import importlib, sys as _sys
        here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        if here not in _sys.path:
            _sys.path.insert(0, here)
        try:
            return importlib.import_module('fig3f_roi_trace')
        except ImportError:
            messagebox.showerror("Missing module",
                                 "fig3f_roi_trace.py 를 이 스크립트와 같은 폴더에 두세요.")
            return None

    def save_line_rois(self):
        """foreground / background 1,2 polyline + width/segments/marker/필터 설정을 JSON 으로 저장."""
        f3 = self._fig3f_module()
        if f3 is None:
            return
        try:
            self.update_scale_settings()
        except Exception:
            pass
        try:
            roi = f3.ROISet.from_gui(self, label='ROI')
        except ValueError as e:
            messagebox.showwarning("Warning", str(e))
            return
        p = filedialog.asksaveasfilename(title="Save line ROIs", defaultextension=".json",
                                         filetypes=[("ROI json", "*.json")])
        if not p:
            return
        roi.label = os.path.splitext(os.path.basename(p))[0].replace('_roi', '')
        roi.note = f"tif={os.path.basename(getattr(self, 'tif_file_path', '') or '')}"
        roi.to_json(p)
        self.status_var.set(f"Saved line ROIs → {os.path.basename(p)}")

    def load_line_rois(self):
        f3 = self._fig3f_module()
        if f3 is None:
            return
        p = filedialog.askopenfilename(title="Load line ROIs", filetypes=[("ROI json", "*.json")])
        if not p:
            return
        roi = f3.ROISet.from_json(p)
        self.foreground_roi = np.array(roi.foreground, dtype=np.int32)
        self.background_roi_1 = np.array(roi.background_1, dtype=np.int32)
        self.background_roi_2 = np.array(roi.background_2, dtype=np.int32)
        self.foreground_line_width = int(roi.fg_width)
        self.background_line_width = int(roi.bg_width)
        self.background_segments = int(roi.n_segments)
        for var, val in ((getattr(self, 'foreground_line_width_var', None), roi.fg_width),
                         (getattr(self, 'background_line_width_var', None), roi.bg_width),
                         (getattr(self, 'background_segments_var', None), roi.n_segments)):
            try:
                var.set(val)
            except Exception:
                pass
        self.start_marker, self.end_marker = roi.start_frame, roi.end_frame
        self.remove_outliers = bool(roi.remove_outliers)
        self.outlier_method = roi.outlier_method
        self.outlier_threshold = float(roi.outlier_threshold)
        self.morph_size = int(roi.morph_size)
        self.apply_filter = bool(roi.apply_median)
        self.median_filter_window = int(roi.median_window)
        self.foreground_signal = self.background_signal = self.difference_signal = None
        try:
            self.set_waypoints_from_points([list(pt) for pt in roi.foreground])
        except Exception:
            pass
        self.update_marker_info()
        self.display_current_frame()
        self.status_var.set(f"Loaded line ROIs ← {os.path.basename(p)} (label={roi.label})")

    # Okabe-Ito colour-blind safe palette (the Fig. 3f submitted panels used blue / vermilion)
    FIG3F_PALETTE = [
        ('blue (Fig 3f left)', '#0072B2'), ('red (Fig 3f right)', '#D62728'),
        ('crimson', '#C0392B'), ('dark red', '#A50F15'),
        ('vermilion (orange-red)', '#D55E00'), ('orange', '#E69F00'),
        ('green', '#009E73'), ('sky', '#56B4E9'), ('purple', '#CC79A7'),
        ('yellow', '#F0E442'), ('black', '#000000'),
    ]

    FIG3F_SMOOTH = [
        ('none (raw trace)', 'none'),
        ('LOWESS local linear fit', 'lowess'),
        ('moving average (time window)', 'movavg'),
        ('Savitzky-Golay', 'savgol'),
        ('smoothing spline', 'spline'),
    ]

    def _fig3f_options_dialog(self):
        """Fig 3f export 옵션. dict 를 돌려주고, 취소하면 None."""
        from tkinter import colorchooser
        n_ex_total = len(getattr(self, 'excluded_frames', set()))
        try:
            s_f, e_f = self.get_analysis_frame_range()
            n_ex_range = sum(1 for i in self.excluded_frames if s_f <= i <= e_f)
            n_range = e_f - s_f + 1
        except Exception:
            n_ex_range, n_range = n_ex_total, 0

        dlg = tk.Toplevel(self.root)
        dlg.title("Fig 3f export options")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)
        pad = dict(padx=8, pady=3)
        result = {}

        body = ttk.Frame(dlg)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=6)
        r = 0

        ttk.Label(body, text="Panel label:").grid(row=r, column=0, sticky='w', **pad)
        label_var = tk.StringVar(value="ROI_blue_-300um")
        ttk.Entry(body, textvariable=label_var, width=28).grid(row=r, column=1, columnspan=2, sticky='w', **pad)
        r += 1

        ttk.Separator(body, orient='horizontal').grid(row=r, column=0, columnspan=3, sticky='ew', pady=6)
        r += 1

        # ---- breathing exclusion ----
        excl_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Exclude breathing / motion frames before exporting",
                        variable=excl_var).grid(row=r, column=0, columnspan=3, sticky='w', **pad)
        r += 1
        if n_ex_total == 0:
            msg = ("no frames are flagged yet — run 'Detect Breathing Frames' first, "
                   "otherwise this option changes nothing")
            fg = '#b00'
        else:
            msg = ("%d frame(s) flagged in total, %d inside the analysis range%s"
                   % (n_ex_total, n_ex_range,
                      (" of %d" % n_range) if n_range else ""))
            fg = '#060'
        ttk.Label(body, text=msg, foreground=fg, wraplength=330).grid(
            row=r, column=0, columnspan=3, sticky='w', padx=26)
        r += 1

        gap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Break the line at rejected spans (do not interpolate across)",
                        variable=gap_var).grid(row=r, column=0, columnspan=3, sticky='w', padx=26)
        r += 1
        shade_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Shade the rejected time spans in grey",
                        variable=shade_var).grid(row=r, column=0, columnspan=3, sticky='w', padx=26)
        r += 1

        ttk.Separator(body, orient='horizontal').grid(row=r, column=0, columnspan=3, sticky='ew', pady=6)
        r += 1

        # ---- appearance ----
        ttk.Label(body, text="Trace colour:").grid(row=r, column=0, sticky='w', **pad)
        names = [n for n, _ in self.FIG3F_PALETTE]
        colour_var = tk.StringVar(value=names[0])
        hex_var = tk.StringVar(value=self.FIG3F_PALETTE[0][1])
        combo = ttk.Combobox(body, textvariable=colour_var, values=names,
                             state='readonly', width=22)
        combo.grid(row=r, column=1, sticky='w', **pad)
        swatch = tk.Label(body, text="       ", background=hex_var.get(), relief='solid', borderwidth=1)
        swatch.grid(row=r, column=2, sticky='w', **pad)

        def on_combo(_=None):
            for n, h in self.FIG3F_PALETTE:
                if n == colour_var.get():
                    hex_var.set(h)
                    swatch.config(background=h)
                    break
        combo.bind('<<ComboboxSelected>>', on_combo)
        r += 1

        def pick_custom():
            rgb, hx = colorchooser.askcolor(color=hex_var.get(), parent=dlg,
                                            title="Fig 3f trace colour")
            if hx:
                hex_var.set(hx)
                swatch.config(background=hx)
                colour_var.set("custom %s" % hx)
        ttk.Button(body, text="Custom colour...", command=pick_custom).grid(
            row=r, column=1, sticky='w', **pad)
        r += 1

        ttk.Label(body, text="Line width (pt):").grid(row=r, column=0, sticky='w', **pad)
        lw_var = tk.DoubleVar(value=0.9)
        ttk.Entry(body, textvariable=lw_var, width=8).grid(row=r, column=1, sticky='w', **pad)
        r += 1

        ttk.Label(body, text="Plotted quantity:").grid(row=r, column=0, sticky='w', **pad)
        sig_var = tk.StringVar(value='fg')
        ttk.Combobox(body, textvariable=sig_var, state='readonly', width=22,
                     values=['fg', 'diff']).grid(row=r, column=1, sticky='w', **pad)
        ttk.Label(body, text="fg = raw ROI mean (Fig 3f);  diff = fg - bg",
                  foreground='#555').grid(row=r + 1, column=0, columnspan=3, sticky='w', padx=26)
        r += 2

        ttk.Label(body, text="Panel size (mm):").grid(row=r, column=0, sticky='w', **pad)
        szf = ttk.Frame(body)
        szf.grid(row=r, column=1, columnspan=2, sticky='w', **pad)
        w_var = tk.DoubleVar(value=42.0)
        h_var = tk.DoubleVar(value=32.0)
        ttk.Entry(szf, textvariable=w_var, width=7).pack(side=tk.LEFT)
        ttk.Label(szf, text=" x ").pack(side=tk.LEFT)
        ttk.Entry(szf, textvariable=h_var, width=7).pack(side=tk.LEFT)
        r += 1

        grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Background grid", variable=grid_var).grid(
            row=r, column=0, columnspan=3, sticky='w', **pad)
        r += 1
        zero_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Force y axis to start at 0 (submitted figure did this)",
                        variable=zero_var).grid(row=r, column=0, columnspan=3, sticky='w', **pad)
        r += 1

        ttk.Separator(body, orient='horizontal').grid(row=r, column=0, columnspan=3, sticky='ew', pady=6)
        r += 1

        # ---- time axis ----
        t0_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Time axis: t = 0 at the analysis start marker",
                        variable=t0_var).grid(row=r, column=0, columnspan=3, sticky='w', **pad)
        r += 1
        try:
            _s_f, _e_f = self.get_analysis_frame_range()
            _fr = float(self.frame_rate_var.get())
            _msg = ("start marker = frame %d -> would otherwise start at %.2f s; "
                    "unchecking keeps absolute time" % (_s_f, _s_f / _fr if _fr else float('nan')))
        except Exception:
            _msg = "unchecking keeps absolute time from the first recorded frame"
        ttk.Label(body, text=_msg, foreground='#555', wraplength=330).grid(
            row=r, column=0, columnspan=3, sticky='w', padx=26)
        r += 1

        # ---- smoothing ----
        ttk.Label(body, text="Smoothing / fit:").grid(row=r, column=0, sticky='w', **pad)
        sm_names = [n for n, _ in self.FIG3F_SMOOTH]
        sm_var = tk.StringVar(value=sm_names[1])          # LOWESS by default
        ttk.Combobox(body, textvariable=sm_var, values=sm_names, state='readonly',
                     width=22).grid(row=r, column=1, columnspan=2, sticky='w', **pad)
        r += 1
        ttk.Label(body, text="Window / bandwidth (s):").grid(row=r, column=0, sticky='w', **pad)
        win_var = tk.DoubleVar(value=1.0)
        ttk.Entry(body, textvariable=win_var, width=8).grid(row=r, column=1, sticky='w', **pad)
        ord_var = tk.IntVar(value=2)
        ttk.Label(body, text="savgol order:").grid(row=r, column=1, sticky='e', **pad)
        ttk.Entry(body, textvariable=ord_var, width=4).grid(row=r, column=2, sticky='w', **pad)
        r += 1
        ttk.Label(body, text="the window is in SECONDS, so it stays correct where breathing frames "
                             "were removed and the sampling is uneven",
                  foreground='#555', wraplength=330).grid(row=r, column=0, columnspan=3,
                                                          sticky='w', padx=26)
        r += 1
        raw_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Show the raw trace faintly behind the smoothed curve",
                        variable=raw_var).grid(row=r, column=0, columnspan=3, sticky='w', padx=26)
        r += 1
        rawstyle_var = tk.StringVar(value='line')
        rsf = ttk.Frame(body)
        rsf.grid(row=r, column=0, columnspan=3, sticky='w', padx=44)
        ttk.Radiobutton(rsf, text="as a thin line", variable=rawstyle_var,
                        value='line').pack(side=tk.LEFT)
        ttk.Radiobutton(rsf, text="as dots", variable=rawstyle_var,
                        value='points').pack(side=tk.LEFT, padx=(10, 0))
        r += 1
        ttk.Label(body, text="the CSV always keeps the RAW values; the smoothed curve is added as an "
                             "extra column and is a display aid only",
                  foreground='#555', wraplength=330).grid(row=r, column=0, columnspan=3,
                                                          sticky='w', padx=26)
        r += 1

        btns = ttk.Frame(dlg)
        btns.pack(fill=tk.X, padx=10, pady=(4, 10))

        def ok():
            try:
                meth = dict(self.FIG3F_SMOOTH)[sm_var.get()]
                result.update(
                    label=label_var.get().strip(),
                    exclude_breathing=bool(excl_var.get()),
                    signal=sig_var.get(),
                    t0_at_start_marker=bool(t0_var.get()),
                    smooth=(None if meth == 'none' else
                            dict(method=meth, window_s=float(win_var.get()),
                                 order=int(ord_var.get()))),
                    style=dict(color=hex_var.get(), lw=float(lw_var.get()),
                               break_gaps=bool(gap_var.get()),
                               mark_excluded=bool(shade_var.get()),
                               grid=bool(grid_var.get()),
                               y_from_zero=bool(zero_var.get()),
                               show_raw=bool(raw_var.get()),
                               raw_style=rawstyle_var.get(),
                               panel_w_mm=float(w_var.get()), panel_h_mm=float(h_var.get())))
            except (ValueError, tk.TclError) as e:
                messagebox.showerror("Invalid value", str(e), parent=dlg)
                return
            if not result['label']:
                messagebox.showwarning("Label", "Panel label 을 입력하세요", parent=dlg)
                result.clear()
                return
            dlg.destroy()

        ttk.Button(btns, text="Export", command=ok).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=(0, 6))
        dlg.bind('<Return>', lambda e: ok())
        dlg.bind('<Escape>', lambda e: dlg.destroy())
        dlg.update_idletasks()
        self.root.wait_window(dlg)
        return result or None

    def export_fig3f_source_data(self):
        """Fig. 3f: 현재 ROI/marker/필터 설정으로 Source Data CSV + Nature 스타일 그림 저장.

        - 숫자: self.extract_segmented_signals() → Show Combined Signals 와 동일
        - 시간축: frame / fps.  fps 는 GUI 의 Frame Rate 입력값을 쓰되 기본값(30) 그대로면 거부
          (마우스 귀 dynamic 은 20 volumes/s — fps 가 틀리면 Fig 3g 속도까지 통째로 스케일됨)
        - ROI 는 <label>_roi.json 으로 함께 저장되어 fig3f_roi_trace.py 로 헤드리스 재현 가능
        """
        f3 = self._fig3f_module()
        if f3 is None:
            return
        if self.image_stack is None:
            messagebox.showwarning("Warning", "TIF 스택을 먼저 로드하세요")
            return
        try:
            self.update_scale_settings()
        except Exception:
            pass
        opt = self._fig3f_options_dialog()
        if not opt:
            return
        label = opt['label']
        try:
            fps = f3.resolve_fps(self, None)
        except ValueError as e:
            messagebox.showerror("Frame rate", str(e))
            return
        out_dir = filedialog.askdirectory(title="Output folder for Fig 3f source data")
        if not out_dir:
            return
        try:
            out = f3.export_from_gui(self, out_dir=out_dir, fps=fps, label=label,
                                     fps_source='GUI Frame Rate field',
                                     signal=opt['signal'],
                                     exclude_breathing=opt['exclude_breathing'],
                                     t0_at_start_marker=opt['t0_at_start_marker'],
                                     smooth=opt['smooth'],
                                     style=opt['style'])
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            return
        tr = out['traces']
        i = int(np.argmax(tr[opt['signal']]))
        excl_txt = (f"{out['n_excluded']} breathing frames removed"
                    if out['exclude_breathing'] else "breathing exclusion OFF")
        self.status_var.set(f"Fig 3f exported: {os.path.basename(out['csv'])}  "
                            f"(n={len(tr['frame'])} frames @ {fps:g} fps, {excl_txt}, "
                            f"peak {tr[opt['signal']][i]:.0f} at {tr['time_s'][i]:.2f} s)")
        messagebox.showinfo("Fig 3f export",
                            f"CSV : {out['csv']}\nROI : {out['roi_json']}\nFig : {out['figure'][0]}\n\n"
                            f"fps = {fps:g} volumes/s, {len(tr['frame'])} frames "
                            f"({tr['time_s'][0]:.2f}–{tr['time_s'][-1]:.2f} s"
                            + (f", t=0 at frame {out['t0_frame']}" if out['t0_frame'] is not None
                               else ", absolute time") + ")\n"
                            f"{excl_txt}; signal={opt['signal']}, colour={opt['style']['color']}\n"
                            f"smoothing: {out['smooth'] or 'none'}")

    def show_videokymography(self):
        """Show videokymography analysis"""
        if (self.foreground_roi is None or self.background_roi_1 is None or self.background_roi_2 is None):
            messagebox.showwarning("Warning", "Please draw foreground and both background ROIs first")
            return
        
        # Create dialog for sub-ROI length
        dialog = tk.Toplevel(self.root)
        dialog.title("Videokymography Settings")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Sub-ROI Length (μm):").pack(pady=10)
        
        length_var = tk.DoubleVar(value=10.0 * self.pixel_size)
        ttk.Entry(dialog, textvariable=length_var, width=10).pack(pady=5)
        
        def generate_kymography():
            sub_roi_length_um = length_var.get()
            sub_roi_length_pixels = sub_roi_length_um / self.pixel_size
            dialog.destroy()
            self._generate_kymography(sub_roi_length_pixels)
        
        ttk.Button(dialog, text="Generate", command=generate_kymography).pack(pady=10)
    
    def _generate_kymography(self, sub_roi_length):
        """Generate videokymography plot - ENHANCED VERSION with flexible normalization"""
        try:
            # ========== 配置选项 ==========
            # 使用实例变量替代硬编码
            SUBTRACT_BACKGROUND = self.subtract_background
            NORMALIZATION_MODE = self.normalization_mode
            # ==============================
            
            # Get the analysis frame range first
            start_frame, end_frame = self.get_analysis_frame_range()
            total_frames = len(self.image_stack)
            
            print(f"Kymography: start={start_frame}, end={end_frame}, total={total_frames}")
            
            # Extract segmented background signals for the analysis range.
            # exclude_breathing=True drops motion frames so the background columns
            # stay aligned with the breathing-excluded sub-ROI columns below.
            fg_signals, bg_signals = self.extract_segmented_signals(exclude_breathing=True)
            if fg_signals is None:
                messagebox.showerror("Error", "Failed to extract background signals")
                return
            
            # Calculate sub-ROIs along the foreground line
            line_points = self.foreground_roi
            
            # Calculate total line length
            total_length = 0
            for i in range(len(line_points) - 1):
                dx = line_points[i+1][0] - line_points[i][0]
                dy = line_points[i+1][1] - line_points[i][1]
                total_length += np.sqrt(dx*dx + dy*dy)
            
            num_sub_rois = max(1, int(total_length // sub_roi_length))
            
            # Generate sub-ROI signals
            sub_roi_signals = []
            positions_um = []
            
            # Safe frame range that matches the extracted signals — must use the
            # SAME breathing-excluded frames so columns align with bg_signals.
            frame_range = self._included_frames(start_frame, min(end_frame, total_frames - 1))
            if len(frame_range) < 1:
                frame_range = list(range(start_frame, min(end_frame + 1, total_frames)))
            num_analysis_frames = len(frame_range)
            n_excluded_kymo = (min(end_frame, total_frames - 1) - start_frame + 1) - num_analysis_frames
            
            print(f"Analysis frames: {num_analysis_frames}, bg_signals shape: {bg_signals.shape}")
            
            for sub_idx in range(num_sub_rois):
                # Calculate position along line
                target_distance = sub_idx * sub_roi_length
                positions_um.append(target_distance * self.pixel_size)
                
                # Find corresponding point on line
                current_distance = 0
                sub_roi_center = None
                
                for i in range(len(line_points) - 1):
                    dx = line_points[i+1][0] - line_points[i][0]
                    dy = line_points[i+1][1] - line_points[i][1]
                    segment_length = np.sqrt(dx*dx + dy*dy)
                    
                    if current_distance + segment_length >= target_distance:
                        t = (target_distance - current_distance) / segment_length if segment_length > 0 else 0
                        sub_roi_center = [
                            int(line_points[i][0] + t * dx),
                            int(line_points[i][1] + t * dy)
                        ]
                        break
                    
                    current_distance += segment_length
                
                if sub_roi_center is None:
                    sub_roi_signals.append(np.zeros(num_analysis_frames))
                    continue
                
                # Extract signal for this sub-ROI
                sub_roi_signal = []
                
                for relative_frame_idx, absolute_frame_idx in enumerate(frame_range):
                    if absolute_frame_idx >= total_frames:
                        break
                        
                    frame = self.image_stack[absolute_frame_idx]
                    
                    # Create circular mask
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, tuple(sub_roi_center), int(sub_roi_length//2), 255, -1)
                    
                    # Extract foreground signal
                    fg_pixels = frame[mask > 0]
                    fg_mean = np.mean(fg_pixels) if len(fg_pixels) > 0 else 0
                    
                    # Find corresponding background segment
                    bg_seg_idx = min(sub_idx * self.background_segments // num_sub_rois, 
                                self.background_segments - 1)
                    
                    # Get background signal
                    if (bg_seg_idx < bg_signals.shape[0] and 
                        relative_frame_idx < bg_signals.shape[1]):
                        bg_mean = bg_signals[bg_seg_idx, relative_frame_idx]
                    else:
                        if relative_frame_idx < bg_signals.shape[1]:
                            bg_mean = np.mean(bg_signals[:, relative_frame_idx])
                        else:
                            bg_mean = 0
                    
                    # 根据配置决定是否减背景
                    if SUBTRACT_BACKGROUND:
                        sub_roi_signal.append(fg_mean - bg_mean)
                    else:
                        sub_roi_signal.append(fg_mean)
                
                signal = np.array(sub_roi_signal)
                
                # Apply filter if enabled
                if self.apply_filter and self.median_filter_window > 1:
                    signal = medfilt(signal, kernel_size=self.median_filter_window)
                    signal = ndimage.gaussian_filter1d(signal, sigma=0.5)
                
                # ========== 新的归一化处理 ==========
                signal_min = np.min(signal)
                signal_max = np.max(signal)
                signal_range = signal_max - signal_min
                
                if NORMALIZATION_MODE == "min_subtract":
                    # 原版：只减最小值
                    signal = signal - signal_min
                    norm_label = "Min Subtracted"
                    
                elif NORMALIZATION_MODE == "percentile_10":
                    # 减掉范围的10%
                    threshold = signal_min + 0.10 * signal_range
                    signal = signal - threshold
                    signal = np.maximum(signal, 0)  # 确保非负
                    norm_label = "10% Threshold"
                    
                elif NORMALIZATION_MODE == "percentile_20":
                    # 减掉范围的20%
                    threshold = signal_min + 0.20 * signal_range
                    signal = signal - threshold
                    signal = np.maximum(signal, 0)
                    norm_label = "20% Threshold"
                    
                elif NORMALIZATION_MODE == "percentile_50":
                    # 减掉范围的50%
                    threshold = signal_min + 0.50 * signal_range
                    signal = signal - threshold
                    signal = np.maximum(signal, 0)
                    norm_label = "50% Threshold"
                    
                elif NORMALIZATION_MODE == "normalize_0_1":
                    # 归一化到0-1
                    if signal_range > 0:
                        signal = (signal - signal_min) / signal_range
                    else:
                        signal = signal - signal_min
                    norm_label = "Normalized (0-1)"
                
                else:
                    norm_label = "Raw"
                # ====================================
                
                sub_roi_signals.append(signal)
            
            if not sub_roi_signals:
                messagebox.showerror("Error", "Could not generate sub-ROI signals")
                return
            
            # Ensure all signals have the same length
            min_length = min(len(signal) for signal in sub_roi_signals)
            sub_roi_signals = [signal[:min_length] for signal in sub_roi_signals]
            
            # Create kymography matrix
            kymo_matrix = np.array(sub_roi_signals)
            
            # Create plot window
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Videokymography Analysis")
            plot_window.geometry("1200x800")
            
            fig = Figure(figsize=(14, 10))
            
            # Time axis from the REAL kept-frame indices (breathing-excluded frames
            # are dropped, so a gap shows as a gap rather than faking continuity).
            _kept = np.asarray(frame_range[:kymo_matrix.shape[1]], dtype=float)
            time_axis = _kept / self.frame_rate
            
            # Main kymography plot
            ax1 = fig.add_subplot(131)
            im = ax1.imshow(kymo_matrix, aspect='auto', cmap='viridis', origin='lower', 
                        extent=[time_axis[0], time_axis[-1], positions_um[0], positions_um[-1]])
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position (μm)')
            
            # 动态标题根据设置
            bg_text = "FG-BG" if SUBTRACT_BACKGROUND else "FG only"
            _excl_text = (f", {n_excluded_kymo} breathing frames removed"
                          if n_excluded_kymo > 0 else "")
            ax1.set_title(f'Videokymography\n({bg_text}, {norm_label}{_excl_text})')
            fig.colorbar(im, ax=ax1, label='Intensity')
            
            # Show ALL sample time traces
            ax2 = fig.add_subplot(132)
            num_traces = len(sub_roi_signals)
            colors = plt.cm.viridis(np.linspace(0, 1, num_traces))
            
            for i in range(num_traces):
                signal = sub_roi_signals[i]
                ax2.plot(time_axis[:len(signal)], signal, color=colors[i], 
                        label=f'{positions_um[i]:.1f} μm' if i < 10 else None,
                        linewidth=1.0, alpha=0.8)
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel(f'Intensity ({norm_label})')
            ax2.set_title(f'All Sample Time Traces (n={num_traces})')
            if num_traces <= 10:
                ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Average signal across vessel
            ax3 = fig.add_subplot(133)
            avg_signal = np.mean(kymo_matrix, axis=0)
            ax3.plot(time_axis[:len(avg_signal)], avg_signal, 'purple', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel(f'Average Intensity ({norm_label})')
            ax3.set_title(f'Average Signal Across Vessel\n({bg_text}, {norm_label})')
            ax3.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add settings info
            settings_text = f"Settings: {self.pixel_size:.2f} μm/pixel, {self.frame_rate:.1f} fps"

            if self.remove_outliers:
                if self.outlier_method == 'morphological':
                    settings_text += f" | MORPHOLOGICAL FILTER (size={self.morph_size})"
                else:
                    settings_text += f" | OUTLIER REMOVAL ({self.outlier_method}, σ={self.outlier_threshold})"
            else:
                settings_text += " | NO OUTLIER REMOVAL"
                
            if self.apply_filter:
                settings_text += f" | MEDIAN FILTER (window={self.median_filter_window})"
            else:
                settings_text += " | NO MEDIAN FILTER"
                
            # 添加处理方式说明
            settings_text += f" | Background: {'SUBTRACTED' if SUBTRACT_BACKGROUND else 'NOT SUBTRACTED'}"
            settings_text += f" | Normalization: {norm_label}"
            settings_text += f" | Analysis: frames {start_frame}-{end_frame}"
            
            status_label = ttk.Label(plot_window, text=settings_text, font=('TkDefaultFont', 9, 'bold'))
            status_label.pack(pady=5)
            
            # Add progress lines
            time_axes = [ax2, ax3]
            self.add_progress_lines_to_window(plot_window, fig, time_axes, canvas)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate videokymography: {str(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
    
    def show_heart_rate_analysis(self):
        """Simplified heart rate analysis focusing on peak detection"""
        if (self.foreground_roi is None or self.background_roi_1 is None or self.background_roi_2 is None):
            messagebox.showwarning("Warning", "Please draw foreground and both background ROIs first")
            return
        
        try:
            # Extract signals with proper bounds checking
            fg_signals, bg_signals = self.extract_segmented_signals()
            if fg_signals is None:
                messagebox.showerror("Error", "Failed to extract signals")
                return
            
            # Calculate difference signal
            avg_fg = np.mean(fg_signals, axis=0)
            avg_bg = np.mean(bg_signals, axis=0)
            diff_signal = avg_fg - avg_bg
            
            # Safety check for empty signals
            if len(diff_signal) == 0:
                messagebox.showerror("Error", "No signal data in selected range")
                return
            
            # FFT analysis
            fft_signal = fft(diff_signal - np.mean(diff_signal))
            freqs = fftfreq(len(diff_signal), 1/self.frame_rate)
            
            # Focus on physiological range (for mice: 5-15 Hz, i.e., 300-900 bpm)
            valid_idx = (freqs > 5) & (freqs < 15)
            power_spectrum = np.abs(fft_signal[valid_idx])
            valid_freqs = freqs[valid_idx]
            
            # Create analysis window
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Heart Rate Analysis")
            plot_window.geometry("1000x600")
            
            fig = Figure(figsize=(12, 8))
            
            # Time axis in ms (adjusted for analysis range)
            start_frame, end_frame = self.get_analysis_frame_range()
            time_axis= (np.arange(len(diff_signal)) + start_frame) / self.frame_rate
            
            # Signal with peaks
            ax1 = fig.add_subplot(221)
            ax1.plot(time_axis, diff_signal, 'b-', linewidth=1)
            
            # Find and mark peaks
            peaks, properties = find_peaks(diff_signal, height=np.std(diff_signal)*0.5, 
                                         distance=int(self.frame_rate*0.05))  # Min 50ms between peaks
            if len(peaks) > 0:
                ax1.plot(time_axis[peaks], diff_signal[peaks], 'ro', markersize=4)
                
                # Calculate heart rate from peaks
                if len(peaks) > 1:
                    peak_intervals = np.diff(peaks) / self.frame_rate  # seconds
                    mean_interval = np.mean(peak_intervals)
                    heart_rate_from_peaks = 60 / mean_interval if mean_interval > 0 else 0
                    ax1.set_title(f'Signal with Peaks\nHR from peaks: {heart_rate_from_peaks:.1f} bpm')
                else:
                    ax1.set_title('Signal with Peaks\nInsufficient peaks detected')
            else:
                ax1.set_title('Signal with Peaks\nNo peaks detected')
            
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Intensity Difference')
            ax1.grid(True, alpha=0.3)
            
            # Power spectrum with peak detection
            ax2 = fig.add_subplot(222)
            if len(power_spectrum) > 0:
                freq_bpm = valid_freqs * 60
                ax2.plot(freq_bpm, power_spectrum, 'g-', linewidth=1)
                
                # Find peaks in frequency domain
                freq_peaks, _ = find_peaks(power_spectrum, height=np.max(power_spectrum)*0.1)
                
                if len(freq_peaks) > 0:
                    # Mark significant frequency peaks
                    for i, peak_idx in enumerate(freq_peaks[:5]):  # Show top 5 peaks
                        peak_freq = valid_freqs[peak_idx]
                        peak_bpm = peak_freq * 60
                        peak_power = power_spectrum[peak_idx]
                        
                        ax2.plot(peak_bpm, peak_power, 'ro', markersize=6)
                        ax2.annotate(f'{peak_bpm:.0f} bpm', 
                                   xy=(peak_bpm, peak_power),
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=9, ha='left')
                
                ax2.set_xlabel('Heart Rate (bpm)')
                ax2.set_ylabel('Power')
                ax2.set_title('Frequency Analysis - Detected Peaks')
                ax2.set_xlim(300, 900)  # Mouse heart rate range
            else:
                ax2.text(0.5, 0.5, 'No valid frequency data', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('Frequency Analysis')
            
            ax2.grid(True, alpha=0.3)
            
            # Heart rate over time (sliding window)
            ax3 = fig.add_subplot(223)
            if len(peaks) > 3:
                window_size = min(10, len(peaks)//2)
                sliding_hr = []
                sliding_times = []
                
                for i in range(window_size, len(peaks)):
                    window_peaks = peaks[i-window_size:i]
                    if len(window_peaks) > 1:
                        intervals = np.diff(window_peaks) / self.frame_rate
                        mean_interval = np.mean(intervals)
                        hr = 60 / mean_interval if mean_interval > 0 else 0
                        sliding_hr.append(hr)
                        sliding_times.append(time_axis[peaks[i]])
                
                if sliding_hr:
                    ax3.plot(sliding_times, sliding_hr, 'r-', linewidth=2)
                    ax3.set_ylabel('Heart Rate (bpm)')
                    ax3.set_title('Heart Rate Over Time (Sliding Window)')
                else:
                    ax3.text(0.5, 0.5, 'Insufficient data for sliding window', 
                            transform=ax3.transAxes, ha='center', va='center')
                    ax3.set_title('Heart Rate Over Time')
            else:
                ax3.text(0.5, 0.5, 'Insufficient peaks for time analysis', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Heart Rate Over Time')
            
            ax3.set_xlabel('Time (s)')
            ax3.grid(True, alpha=0.3)
            
            # Summary statistics
            ax4 = fig.add_subplot(224)
            ax4.axis('off')
            
            summary_text = "Heart Rate Analysis Summary:\n\n"
            
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / self.frame_rate
                mean_hr = 60 / np.mean(peak_intervals)
                std_hr = 60 * np.std(peak_intervals) / (np.mean(peak_intervals)**2)
                summary_text += f"Peak-based HR: {mean_hr:.1f} ± {std_hr:.1f} bpm\n"
                summary_text += f"Number of peaks detected: {len(peaks)}\n"
            
            if len(freq_peaks) > 0 and len(power_spectrum) > 0:
                dominant_freq_idx = freq_peaks[np.argmax(power_spectrum[freq_peaks])]
                dominant_freq = valid_freqs[dominant_freq_idx]
                dominant_hr = dominant_freq * 60
                summary_text += f"Dominant frequency HR: {dominant_hr:.1f} bpm\n"
            
            summary_text += f"\nSettings:\n"
            summary_text += f"Frame rate: {self.frame_rate:.1f} fps\n"
            summary_text += f"Filter: {'Applied' if self.apply_filter else 'Not applied'}\n"
            summary_text += f"Segments: {self.background_segments}"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 添加进度线到时间序列子图 - 在canvas创建后调用
            time_axes = [ax1, ax3]  # ax1有时间轴，ax3有心率随时间变化
            self.add_progress_lines_to_window(plot_window, fig, time_axes, canvas)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze heart rate: {str(e)}")
    
    def generate_default_mp4_filename(self):
        """生成默认的MP4文件名，基于TIF文件名"""
        if self.tif_file_path is None:
            return "output_video.mp4"
        
        # 获取文件名（不含路径）
        tif_filename = os.path.basename(self.tif_file_path)
        
        # 移除扩展名
        name_without_ext = os.path.splitext(tif_filename)[0]
        
        # 添加分析信息作为后缀
        suffix = "_analysis"
        
        # 如果设置了时间范围，添加范围信息
        if self.start_marker is not None or self.end_marker is not None:
            start_frame, end_frame = self.get_analysis_frame_range()
            suffix += f"_frames{start_frame}-{end_frame}"
        
        # 生成完整文件名
        default_filename = f"{name_without_ext}{suffix}.mp4"
        
        return default_filename
    
    def generate_default_mp4_with_signal_filename(self):
        """生成带信号图的MP4文件名"""
        if self.tif_file_path is None:
            return "output_video_with_signal.mp4"
        
        # 获取文件名（不含路径）
        tif_filename = os.path.basename(self.tif_file_path)
        
        # 移除扩展名
        name_without_ext = os.path.splitext(tif_filename)[0]
        
        # 添加分析信息作为后缀
        suffix = "_analysis_with_signal"
        
        # 如果设置了时间范围，添加范围信息
        if self.start_marker is not None or self.end_marker is not None:
            start_frame, end_frame = self.get_analysis_frame_range()
            suffix += f"_frames{start_frame}-{end_frame}"
        
        # 生成完整文件名
        default_filename = f"{name_without_ext}{suffix}.mp4"
        
        return default_filename
    
    def save_mp4(self):
        """Save frames between markers as MP4 video with default filename"""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load a TIF stack first")
            return
        
        # Get frame range
        start_frame, end_frame = self.get_analysis_frame_range()
        total_frames = end_frame - start_frame + 1
        
        if total_frames < 1:
            messagebox.showwarning("Warning", "Invalid frame range")
            return
        
        # 生成默认文件名
        default_filename = self.generate_default_mp4_filename()
        
        # Get save path with default filename
        save_path = filedialog.asksaveasfilename(
            title="Save MP4 Video",
            initialfile=default_filename,  # 设置默认文件名
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # Create progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Saving MP4...")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Center the progress window
            progress_window.update_idletasks()
            x = (progress_window.winfo_screenwidth() // 2) - (progress_window.winfo_width() // 2)
            y = (progress_window.winfo_screenheight() // 2) - (progress_window.winfo_height() // 2)
            progress_window.geometry(f"+{x}+{y}")
            
            ttk.Label(progress_window, text="Generating MP4 video...").pack(pady=10)
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            progress_label = ttk.Label(progress_window, text="Processing frame 0/0")
            progress_label.pack(pady=5)
            
            # Force window to display
            progress_window.update()
            
            # Get frame dimensions from first frame
            first_frame = self._generate_display_frame(start_frame)
            if first_frame is None:
                progress_window.destroy()
                messagebox.showerror("Error", "Failed to generate first frame")
                return
                
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.frame_rate  # Use the actual frame rate
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                progress_window.destroy()
                messagebox.showerror("Error", "Failed to create video writer")
                return
            
            # Generate frames
            for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                # Update progress
                progress = (i / total_frames) * 100
                progress_var.set(progress)
                progress_label.config(text=f"Processing frame {i+1}/{total_frames}")
                progress_window.update()
                
                # Generate frame with all overlays
                display_frame = self._generate_display_frame(frame_idx)
                if display_frame is not None:
                    # Convert RGB to BGR for OpenCV
                    bgr_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr_frame)
                else:
                    print(f"Warning: Failed to generate frame {frame_idx}")
            
            # Cleanup
            video_writer.release()
            progress_window.destroy()
            
            # Calculate video info
            duration = total_frames / fps
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            
            messagebox.showinfo("Success", 
                            f"MP4 saved successfully!\n\n"
                            f"File: {os.path.basename(save_path)}\n"
                            f"Frames: {total_frames}\n"
                            f"Duration: {duration:.2f} seconds\n"
                            f"FPS: {fps:.1f}\n"
                            f"Size: {file_size:.1f} MB")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Failed to save MP4: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_mp4_with_signal(self):
        """Save MP4 video with image on left and signal plot on right"""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load a TIF stack first")
            return
        
        # Check if ROIs are drawn and signals can be extracted
        if (self.foreground_roi is None or self.background_roi_1 is None or self.background_roi_2 is None):
            messagebox.showwarning("Warning", "Please draw foreground and both background ROIs first")
            return
        
        # Extract signals first
        fg_signals, bg_signals = self.extract_segmented_signals()
        if fg_signals is None:
            messagebox.showerror("Error", "Failed to extract signals")
            return
        
        # Calculate difference signal
        avg_fg = np.mean(fg_signals, axis=0)
        avg_bg = np.mean(bg_signals, axis=0)
        difference_signal = avg_fg - avg_bg
        
        # Get frame range
        start_frame, end_frame = self.get_analysis_frame_range()
        total_frames = end_frame - start_frame + 1
        
        if total_frames < 1:
            messagebox.showwarning("Warning", "Invalid frame range")
            return
        
        # Generate default filename
        default_filename = self.generate_default_mp4_with_signal_filename()
        
        # Get save path
        save_path = filedialog.asksaveasfilename(
            title="Save MP4 Video with Signal",
            initialfile=default_filename,
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # Create progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Saving MP4 with Signal...")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Center the progress window
            progress_window.update_idletasks()
            x = (progress_window.winfo_screenwidth() // 2) - (progress_window.winfo_width() // 2)
            y = (progress_window.winfo_screenheight() // 2) - (progress_window.winfo_height() // 2)
            progress_window.geometry(f"+{x}+{y}")
            
            ttk.Label(progress_window, text="Generating MP4 video with signal...").pack(pady=10)
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            progress_label = ttk.Label(progress_window, text="Processing frame 0/0")
            progress_label.pack(pady=5)
            
            # Force window to display
            progress_window.update()
            
            # Get first frame to determine dimensions
            first_image_frame = self._generate_display_frame(start_frame)
            if first_image_frame is None:
                progress_window.destroy()
                messagebox.showerror("Error", "Failed to generate first frame")
                return
            
            # Calculate combined frame dimensions
            img_height, img_width = first_image_frame.shape[:2]
            plot_width = img_width  # Make plot same width as image
            combined_width = img_width + plot_width
            combined_height = img_height
            
            # Create time axis for signal
            time_axis = (np.arange(len(difference_signal)) + start_frame) / self.frame_rate
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.frame_rate
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (combined_width, combined_height))
            
            if not video_writer.isOpened():
                progress_window.destroy()
                messagebox.showerror("Error", "Failed to create video writer")
                return
            
            # Generate frames
            for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                # Update progress
                progress = (i / total_frames) * 100
                progress_var.set(progress)
                progress_label.config(text=f"Processing frame {i+1}/{total_frames}")
                progress_window.update()
                
                # Generate image frame
                image_frame = self._generate_display_frame(frame_idx)
                if image_frame is None:
                    print(f"Warning: Failed to generate frame {frame_idx}")
                    continue
                
                # Generate signal plot frame
                signal_frame = self._generate_signal_plot_frame(
                    difference_signal, time_axis, frame_idx, 
                    plot_width, combined_height
                )
                
                # Combine frames horizontally
                combined_frame = np.hstack([image_frame, signal_frame])
                
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
            
            # Cleanup
            video_writer.release()
            progress_window.destroy()
            
            # Calculate video info
            duration = total_frames / fps
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            
            messagebox.showinfo("Success", 
                            f"MP4 with signal saved successfully!\n\n"
                            f"File: {os.path.basename(save_path)}\n"
                            f"Frames: {total_frames}\n"
                            f"Duration: {duration:.2f} seconds\n"
                            f"FPS: {fps:.1f}\n"
                            f"Size: {file_size:.1f} MB\n"
                            f"Layout: Image + Signal Plot")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Failed to save MP4 with signal: {str(e)}")
            import traceback
            traceback.print_exc()

    def _generate_signal_plot_frame(self, signal, time_axis, current_frame_idx, width, height):
        """Generate a signal plot frame with progress line at current time"""
        try:
            # Create matplotlib figure
            fig = Figure(figsize=(width/100, height/100), dpi=150, facecolor='white')
            ax = fig.add_subplot(111)
            
            # Plot the signal
            ax.plot(time_axis, signal, 'b-', linewidth=2, label='Difference Signal')
            
            # Calculate current time and add progress line
            current_time = current_frame_idx / self.frame_rate
            ax.axvline(x=current_time, color='red', linestyle='--', linewidth=3, alpha=0.8)
            
            # Set labels and title
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Intensity Difference', fontsize=10)
            # ax.set_title('Foreground - Background Signal', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Set axis limits
            ax.set_xlim(time_axis[0], time_axis[-1])
            y_min, y_max = np.min(signal), np.max(signal)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            
            # Adjust layout
            fig.tight_layout()
            
            # Convert to image array using io buffer - works with all matplotlib versions
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            
            # Open as PIL image and convert to numpy array
            pil_img = Image.open(buf)
            pil_img = pil_img.convert('RGB')  # Ensure RGB format
            img_array = np.array(pil_img)
            
            # Close buffer and figure to free memory
            buf.close()
            plt.close(fig)
            
            # Resize if necessary
            if img_array.shape[:2] != (height, width):
                pil_img_resized = Image.fromarray(img_array)
                pil_img_resized = pil_img_resized.resize((width, height), Image.LANCZOS)
                img_array = np.array(pil_img_resized)
            
            return img_array
            
        except Exception as e:
            print(f"Error generating signal plot frame: {e}")
            import traceback
            traceback.print_exc()
            # Return a white frame with error text as fallback
            fallback_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # Convert to PIL to add text
            fallback_pil = Image.fromarray(fallback_frame)
            return np.array(fallback_pil)

    def _generate_display_frame(self, frame_idx):
        """Generate a display frame with all overlays for the given frame index"""
        try:
            if frame_idx >= len(self.image_stack):
                return None
                
            # Get frame
            frame = self.image_stack[frame_idx].copy()
            
            # Normalize for display
            if frame.dtype != np.uint8:
                frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
            
            # Convert to RGB for ROI overlay
            if len(frame.shape) == 2:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                display_frame = frame.copy()
            
            # Create separate overlays for background and foreground
            background_overlay = display_frame.copy()
            foreground_overlay = display_frame.copy()

            # Draw background ROIs (more transparent)
            if self.background_roi_1 is not None:
                # Draw line
                for i in range(len(self.background_roi_1) - 1):
                    cv2.line(background_overlay, tuple(self.background_roi_1[i]), 
                            tuple(self.background_roi_1[i+1]), (0, 255, 0), 2)
                
                # Draw expanded area
                expanded_roi = self.create_expanded_line_roi(self.background_roi_1, self.background_line_width)
                if expanded_roi is not None:
                    cv2.fillPoly(background_overlay, [expanded_roi], (0, 255, 0))

            if self.background_roi_2 is not None:
                # Draw line
                for i in range(len(self.background_roi_2) - 1):
                    cv2.line(background_overlay, tuple(self.background_roi_2[i]), 
                            tuple(self.background_roi_2[i+1]), (0, 200, 100), 2)
                
                # Draw expanded area
                expanded_roi = self.create_expanded_line_roi(self.background_roi_2, self.background_line_width)
                if expanded_roi is not None:
                    cv2.fillPoly(background_overlay, [expanded_roi], (0, 200, 100))

            # Draw foreground ROI
            if self.foreground_roi is not None:
                # Draw line
                for i in range(len(self.foreground_roi) - 1):
                    cv2.line(foreground_overlay, tuple(self.foreground_roi[i]), 
                            tuple(self.foreground_roi[i+1]), (255, 0, 0), 2)
                
                # Draw expanded area
                expanded_roi = self.create_expanded_line_roi(self.foreground_roi, self.foreground_line_width)
                if expanded_roi is not None:
                    cv2.fillPoly(foreground_overlay, [expanded_roi], (255, 0, 0))

            # Apply transparency
            display_frame = cv2.addWeighted(display_frame, 0.95, background_overlay, 0.05, 0)
            display_frame = cv2.addWeighted(display_frame, 0.85, foreground_overlay, 0.15, 0)
            
            # Add frame number and time info
            margin = 5
            text_y_start = margin + 15
            text_line_height = 13
            
            # Calculate current time
            current_time = frame_idx / self.frame_rate
            total_frames = len(self.image_stack)
            
            # Add frame number text (larger font for video)
            frame_text = f"Frame: {frame_idx}/{total_frames-1}"
            cv2.putText(display_frame, frame_text, 
                    (margin, text_y_start), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add time text
            time_text = f"Time: {current_time:.3f}s"
            cv2.putText(display_frame, time_text, 
                    (margin, text_y_start + text_line_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add scalebar (200 μm)
            if self.show_scalebar:
                scalebar_length_um = 200.0
                scalebar_length_pixels = int(scalebar_length_um / self.pixel_size)
                
                # Position scalebar in top-right corner
                margin_sb = 25
                scalebar_y = margin_sb + 15
                scalebar_x_end = display_frame.shape[1] - margin_sb
                scalebar_x_start = scalebar_x_end - scalebar_length_pixels
                
                # Draw scalebar (thicker for video)
                cv2.line(display_frame, (scalebar_x_start, scalebar_y), 
                        (scalebar_x_end, scalebar_y), (255, 255, 255), 4)
                
                # Add text (larger font for video)
                cv2.putText(display_frame, f"{scalebar_length_um:.0f}", 
                        (scalebar_x_start, scalebar_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return display_frame
            
        except Exception as e:
            print(f"Error generating display frame {frame_idx}: {e}")
            return None

    # ========================================================================
    # ============== MOUSE EAR VESSEL ANALYSIS METHODS =======================
    # ========================================================================
    #
    # DESIGN PHILOSOPHY & PHYSICAL ASSUMPTIONS (read before trusting numbers):
    #
    #   * Diameter is measured from the FWHM of intensity profiles taken
    #     PERPENDICULAR to the vessel centerline (the foreground ROI). FWHM is
    #     a reliable proxy for vessel diameter ONLY when the true diameter is
    #     comfortably larger than the optical PSF. Below ~2-3 pixels the profile
    #     is dominated by the PSF, not the vessel, so those cross-sections are
    #     flagged and excluded from quantitative claims.
    #
    #   * Cross-sectional area assumes a CIRCULAR lumen: A = pi * (d/2)^2. This
    #     is a standard, defensible assumption for arterioles/venules in the ear
    #     but should be stated explicitly in any figure caption.
    #
    #   * Velocity is measured from the SLOPE of moving features (RBC / contrast
    #     streaks) in a kymograph built along the vessel axis. Streak slope is a
    #     DIRECT kinematic measurement (distance/time) and is far more defensible
    #     than intensity-amplitude-based flow surrogates. Because the input flow
    #     is DIRECTION-resolved, the sign of the slope gives flow direction.
    #
    #   * Volumetric flow rate Q = v_mean * A. If the measured velocity is a
    #     PEAK (centerline) velocity, a Poiseuille correction v_mean = v_peak / 2
    #     is applied (valid in the low-Reynolds, quasi-steady regime confirmed by
    #     the Reynolds / Womersley panel). This correction is toggled explicitly.
    #
    #   * Reynolds (Re = rho*v*D/mu) and Womersley (alpha = R*sqrt(omega*rho/mu))
    #     numbers are computed to CONFIRM the regime that justifies the Poiseuille
    #     assumptions. For mouse ear microvessels both are expected to be << 1.
    #
    #   * IMPORTANT SAMPLING CAVEAT: at low frame rates (e.g. 20 Hz) the mouse
    #     CARDIAC cycle (~8-12 Hz) is ALIASED and cannot be recovered. Pulsatility
    #     / pulse-wave-velocity analyses are therefore intentionally NOT provided.
    #     Slow phenomena (vasomotion ~0.1 Hz) ARE well sampled and are analysed.
    #
    # ========================================================================

    def update_ear_params(self, event=None):
        """Pull mouse-ear analysis parameters from the GUI into instance vars."""
        try:
            self.n_cross_sections = int(self.n_cross_sections_var.get())
            self.profile_half_len = int(self.profile_half_len_var.get())
            self.diameter_method = self.diameter_method_var.get()
            # cP -> Pa*s  (1 cP = 1e-3 Pa*s)
            self.blood_viscosity = float(self.blood_viscosity_var.get()) * 1e-3
            self.blood_density = float(self.blood_density_var.get())
            # Δframe flow-speed seeded peak-tracking controls (Panel 6)
            if hasattr(self, 'flow_seed_end_var'):
                self.flow_seed_end = self.flow_seed_end_var.get()   # auto/proximal/distal
                self.flow_track_method = self.flow_method_var.get()
                _vx = str(self.flow_vmax_var.get()).strip().lower()
                # entries are mm/s; store µm/s. 'auto'/blank -> None (resolvable ceiling)
                self.flow_vmax_um_s = (None if _vx in ('', 'auto', 'none')
                                       else float(_vx) * 1e3)
                self.flow_vmin_um_s = max(0.0, float(self.flow_vmin_var.get())) * 1e3
                self.flow_prominence_mad = max(0.5, float(self.flow_prom_var.get()))
                self.flow_max_miss = max(0, int(self.flow_maxmiss_var.get()))
                self.flow_smooth_sigma = max(0.0, float(self.flow_smooth_var.get()))
                _sw = str(self.flow_seedwin_var.get()).strip().lower()
                self.flow_seed_window_um = (None if _sw in ('', 'auto', 'none')
                                            else float(_sw))
            # flow-speed method selection + arrival-time (transit) controls (Panel 6)
            if hasattr(self, 'flow_method_kind_var'):
                self.flow_method_kind = self.flow_method_kind_var.get()
                self.flow_timing_feature = self.flow_timing_feature_var.get()
                self.flow_arrival_dir = self.flow_arrival_dir_var.get()
                self.flow_arrival_baseline_pct = float(self.flow_arrival_bpct_var.get())
                self.flow_arrival_smooth = max(1, int(self.flow_arrival_smooth_var.get()))
            # arrival-time trend rejection + local-speed window (analysis params)
            if hasattr(self, 'flow_trend_reject_var'):
                self.flow_trend_reject = bool(self.flow_trend_reject_var.get())
                self.flow_trend_k = max(0.5, float(self.flow_trend_k_var.get()))
                _mr = str(self.flow_max_resid_var.get()).strip()
                self.flow_max_resid_s = float(_mr) if _mr else None
                _lw = str(self.flow_local_win_var.get()).strip().lower()
                self.flow_local_win_um = (None if _lw in ('', 'auto', 'none')
                                          else float(_lw))
            # adaptive local-speed estimator params (analysis)
            if hasattr(self, 'flow_min_span_var'):
                _ms = str(self.flow_min_span_var.get()).strip().lower()
                self.flow_min_span_um = None if _ms in ('', 'auto', 'none') else float(_ms)
                _wm = str(self.flow_w_max_var.get()).strip().lower()
                self.flow_w_max_um = None if _wm in ('', 'auto', 'none') else float(_wm)
                self.flow_local_min_pts = max(3, int(self.flow_local_min_pts_var.get()))
                self.flow_edge_frames = max(1, int(self.flow_edge_frames_var.get()))
                self.flow_shrink = bool(self.flow_shrink_var.get())
                self.flow_k_profile = float(self.flow_k_profile_var.get())
            # waypoint method params (analysis)
            if hasattr(self, 'flow_wp_radius_var'):
                _wr = str(self.flow_wp_radius_var.get()).strip().lower()
                self.flow_wp_radius = None if _wr in ('', 'auto', 'none') else float(_wr)
                self.flow_wp_stat = self.flow_wp_stat_var.get()
                self.flow_wp_ncand = max(1, int(self.flow_wp_ncand_var.get()))
                self.flow_wp_assign = self.flow_wp_assign_var.get()
        except Exception as e:
            print(f"update_ear_params error: {e}")

    # ==================== BREATHING / MOTION FRAME EXCLUSION ====================
    def _compute_breathing_scores(self, start_frame=None, end_frame=None):
        """Per-frame respiratory-motion score over the analysis range.

        The reference is the temporal MEDIAN frame — a sharp static structure that
        rejects the (minority) breathing frames. Each frame's score is its mean
        absolute deviation from that reference, normalized by the reference energy
        (so it is scale-free). A rolling-median baseline is subtracted to remove
        slow photobleaching drift, leaving transient breathing spikes. Motion moves
        tissue off the static structure, so breathing frames score high.

        Returns (idx_array, score_array) or (None, None).
        """
        if self.image_stack is None:
            return None, None
        if start_frame is None or end_frame is None:
            start_frame, end_frame = self.get_analysis_frame_range()
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, len(self.image_stack) - 1)
        idx = np.arange(start_frame, end_frame + 1)
        if idx.size < 3:
            return idx, np.zeros(idx.size)

        block = self.image_stack[start_frame:end_frame + 1].astype(np.float32)
        ref = np.median(block, axis=0)                    # sharp, breathing-robust
        denom = float(np.mean(np.abs(ref))) + 1e-9
        raw = np.mean(np.abs(block - ref[None, :, :]), axis=(1, 2)) / denom

        # remove slow drift (bleaching) so only transient motion spikes remain
        k = int(max(3, min(idx.size // 2 * 2 - 1, round(self.frame_rate) | 1)))
        if k >= 3 and k < idx.size:
            try:
                baseline = medfilt(raw, kernel_size=k)
                scores = np.clip(raw - baseline, 0, None)
            except Exception:
                scores = raw - float(np.median(raw))
        else:
            scores = raw - float(np.median(raw))
        scores = np.clip(scores, 0, None)
        return idx, scores

    def detect_breathing_frames(self):
        """Auto-flag breathing-corrupted frames and exclude them from analysis.

        Uses a robust (median + sigma*MAD) threshold on the per-frame motion score.
        Opens a diagnostic window (motion trace + histogram + reference-vs-excluded
        preview) so the choice can be verified and the sigma re-tuned.
        """
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Load a TIF stack first.")
            return
        try:
            sigma = float(self.breathing_sigma_var.get())
        except Exception:
            sigma = 3.0
        self.breathing_sigma = sigma

        idx, scores = self._compute_breathing_scores()
        if idx is None or idx.size == 0:
            messagebox.showerror("Error", "No frames to analyze.")
            return

        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)))
        robust_sd = 1.4826 * mad if mad > 0 else (float(np.std(scores)) or 1e-9)
        thr = med + sigma * robust_sd
        flagged = idx[scores > thr]

        # merge into the running exclusion set (auto-detect adds, never removes
        # manually-excluded frames)
        self.excluded_frames.update(int(f) for f in flagged)
        self.breathing_scores = (idx, scores, thr)
        self._update_breathing_status()
        self.status_var.set(
            f"Breathing detection: {len(flagged)} frame(s) flagged "
            f"(sigma={sigma:g}, thr={thr:.4f}); {len(self.excluded_frames)} excluded total")
        self._show_breathing_window(idx, scores, thr, flagged)

    def _show_breathing_window(self, idx, scores, thr, flagged):
        """Diagnostic plot for breathing detection."""
        win = tk.Toplevel(self.root)
        win.title("Breathing / Motion Frame Exclusion")
        win.geometry("1100x750")
        fig = Figure(figsize=(11, 7))

        # (a) motion trace with threshold + flagged frames
        ax1 = fig.add_subplot(211)
        t = idx / self.frame_rate
        ax1.plot(t, scores, '-', color='steelblue', linewidth=1, label='motion score')
        ax1.axhline(thr, color='red', linestyle='--', linewidth=1,
                    label=f'threshold ({thr:.4f})')
        excl_sorted = sorted(self.excluded_frames)
        excl_in = [f for f in excl_sorted if idx[0] <= f <= idx[-1]]
        if excl_in:
            es = np.array(excl_in)
            sc = np.interp(es, idx, scores)
            ax1.plot(es / self.frame_rate, sc, 'rx', markersize=7,
                     label=f'excluded ({len(excl_in)})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Motion score (a.u., detrended)')
        ax1.set_title('Respiratory-motion score per frame '
                      '(deviation from the static median reference)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # (b) score histogram with threshold
        ax2 = fig.add_subplot(223)
        ax2.hist(scores, bins=max(10, len(scores) // 8), color='slategray',
                 alpha=0.8, edgecolor='black')
        ax2.axvline(thr, color='red', linestyle='--', linewidth=1.5, label='threshold')
        ax2.set_xlabel('Motion score (a.u.)')
        ax2.set_ylabel('Frame count')
        ax2.set_title('Score distribution')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # (c) summary text
        ax3 = fig.add_subplot(224)
        ax3.axis('off')
        pct = 100.0 * len(flagged) / max(1, idx.size)
        txt = "BREATHING EXCLUSION\n" + "=" * 26 + "\n\n"
        txt += f"Frames analyzed:   {idx.size}\n"
        txt += f"Flagged this run:  {len(flagged)} ({pct:.1f}%)\n"
        txt += f"Excluded total:    {len(self.excluded_frames)}\n\n"
        txt += f"sigma (MAD):       {self.breathing_sigma:g}\n"
        txt += f"threshold:         {thr:.4f}\n"
        txt += f"median score:      {np.median(scores):.4f}\n"
        txt += f"max score:         {np.max(scores):.4f}\n\n"
        if len(flagged):
            preview = ", ".join(str(int(f)) for f in flagged[:12])
            if len(flagged) > 12:
                preview += ", ..."
            txt += f"Flagged frames:\n{preview}\n\n"
        txt += ("Excluded frames are dropped from the\n"
                "temporal-mean image (morphometry) and\n"
                "the kymograph (velocity). Lower sigma =\n"
                "stricter (removes more).")
        ax3.text(0.02, 0.98, txt, transform=ax3.transAxes, fontsize=9,
                 va='top', fontfamily='monospace')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def exclude_current_frame(self):
        """Manually add the currently-displayed frame to the exclusion set."""
        if self.image_stack is None:
            return
        self.excluded_frames.add(int(self.current_frame))
        self._update_breathing_status()
        self.status_var.set(f"Frame {self.current_frame} excluded "
                            f"({len(self.excluded_frames)} total)")

    def include_current_frame(self):
        """Manually remove the currently-displayed frame from the exclusion set."""
        self.excluded_frames.discard(int(self.current_frame))
        self._update_breathing_status()
        self.status_var.set(f"Frame {self.current_frame} re-included "
                            f"({len(self.excluded_frames)} excluded)")

    def clear_excluded_frames(self):
        """Reset the exclusion set (keep all frames)."""
        self.excluded_frames = set()
        self.breathing_scores = None
        self._update_breathing_status()
        self.status_var.set("Cleared all excluded frames")

    def _update_breathing_status(self):
        """Refresh the on-screen excluded-frame count label if present."""
        if hasattr(self, 'breathing_status_var'):
            n = len(self.excluded_frames)
            self.breathing_status_var.set(
                f"{n} frame(s) excluded" if n else "no frames excluded")
    # ============================================================================

    def _get_mean_image(self, start_frame=None, end_frame=None):
        """Temporal average over the analysis range -> a clean structural image.

        Averaging suppresses shot noise and gives a stable vessel structure for
        morphometry (diameter, length). Uses the marker range if not overridden.
        """
        if self.image_stack is None:
            return None
        if start_frame is None or end_frame is None:
            start_frame, end_frame = self.get_analysis_frame_range()
        end_frame = min(end_frame, len(self.image_stack) - 1)
        start_frame = max(0, start_frame)
        keep = self._included_frames(start_frame, end_frame)
        if len(keep) == 0:                       # all excluded -> fall back to full range
            keep = list(range(start_frame, end_frame + 1))
        block = self.image_stack[keep].astype(np.float64)
        return np.mean(block, axis=0)

    def _included_frames(self, start_frame, end_frame):
        """Indices in [start_frame, end_frame] that are NOT breathing-excluded."""
        return [i for i in range(start_frame, end_frame + 1)
                if i not in self.excluded_frames]

    def _resample_centerline(self, line_points, n_points):
        """Resample a polyline into n_points equally spaced points + unit normals.

        Returns
        -------
        centers : (n_points, 2) float array of (x, y) sample coordinates
        normals : (n_points, 2) float array of unit normal vectors (perp to axis)
        s_um    : (n_points,)  arc-length position of each sample in micrometers
        """
        pts = np.asarray(line_points, dtype=np.float64)
        # cumulative arc length
        seg = np.diff(pts, axis=0)
        seg_len = np.sqrt((seg ** 2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1]
        if total <= 0:
            return None, None, None
        targets = np.linspace(0, total, n_points)
        centers = np.zeros((n_points, 2))
        tangents = np.zeros((n_points, 2))
        for k, t in enumerate(targets):
            # locate segment containing arc-length t
            idx = np.searchsorted(cum, t, side='right') - 1
            idx = np.clip(idx, 0, len(seg) - 1)
            seg_start = cum[idx]
            local = (t - seg_start) / seg_len[idx] if seg_len[idx] > 0 else 0.0
            centers[k] = pts[idx] + local * seg[idx]
            tang = seg[idx] / (seg_len[idx] if seg_len[idx] > 0 else 1.0)
            tangents[k] = tang
        # unit normals: rotate tangent by 90 deg
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        s_um = targets * self.pixel_size
        return centers, normals, s_um

    def _sample_profile(self, image, center, normal, half_len):
        """Bilinearly sample an intensity profile along `normal` through `center`.

        Returns offsets (in pixels, signed) and intensities. Sub-pixel sampling
        via map_coordinates keeps the FWHM estimate smooth and unbiased.
        """
        offsets = np.arange(-half_len, half_len + 1, dtype=np.float64)
        xs = center[0] + offsets * normal[0]
        ys = center[1] + offsets * normal[1]
        # map_coordinates expects (row, col) = (y, x)
        vals = ndimage.map_coordinates(image, [ys, xs], order=1, mode='nearest')
        return offsets, vals

    def _fwhm_from_profile(self, offsets, vals):
        """Estimate FWHM (in pixels) and a Gaussian-sigma from a 1D profile.

        Strategy: subtract a local baseline (profile edges), find half-maximum
        crossings by linear interpolation on each side of the peak. Also fit a
        Gaussian for a smooth sigma estimate and goodness-of-fit. Returns a dict.
        """
        vals = np.asarray(vals, dtype=np.float64)
        if len(vals) < 5:
            return None
        # baseline = mean of the outer 20% on each edge (assumed background)
        edge = max(1, int(0.2 * len(vals)))
        baseline = np.mean(np.concatenate([vals[:edge], vals[-edge:]]))
        prof = vals - baseline
        peak_idx = int(np.argmax(prof))
        peak_val = prof[peak_idx]
        if peak_val <= 0:
            return None
        half = peak_val / 2.0

        # --- half-max crossings (left) ---
        left = None
        for i in range(peak_idx, 0, -1):
            if prof[i] >= half >= prof[i - 1]:
                # linear interpolation between i-1 and i
                denom = (prof[i] - prof[i - 1])
                frac = (half - prof[i - 1]) / denom if denom != 0 else 0.0
                left = offsets[i - 1] + frac * (offsets[i] - offsets[i - 1])
                break
        # --- half-max crossings (right) ---
        right = None
        for i in range(peak_idx, len(prof) - 1):
            if prof[i] >= half >= prof[i + 1]:
                denom = (prof[i] - prof[i + 1])
                frac = (prof[i] - half) / denom if denom != 0 else 0.0
                right = offsets[i] + frac * (offsets[i + 1] - offsets[i])
                break

        if left is None or right is None:
            fwhm_px = np.nan
        else:
            fwhm_px = abs(right - left)

        # --- optional Gaussian fit for a robust sigma + R^2 ---
        sigma_px = np.nan
        r_squared = np.nan
        try:
            from scipy.optimize import curve_fit

            def gauss(x, a, mu, sig, off):
                return a * np.exp(-(x - mu) ** 2 / (2 * sig ** 2)) + off

            p0 = [peak_val, offsets[peak_idx], max(1.0, fwhm_px / 2.355 if np.isfinite(fwhm_px) else 2.0), baseline]
            popt, _ = curve_fit(gauss, offsets, vals, p0=p0, maxfev=5000)
            sigma_px = abs(popt[2])
            fit = gauss(offsets, *popt)
            ss_res = np.sum((vals - fit) ** 2)
            ss_tot = np.sum((vals - np.mean(vals)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            # prefer Gaussian FWHM if the direct estimate failed
            if not np.isfinite(fwhm_px):
                fwhm_px = 2.355 * sigma_px
        except Exception:
            pass

        return {
            'fwhm_px': fwhm_px,
            'sigma_px': sigma_px,
            'peak_val': peak_val,
            'baseline': baseline,
            'r_squared': r_squared,
            'center_offset': offsets[peak_idx],
        }

    def _threshold_width_from_profile(self, offsets, vals, frac=0.5):
        """Alternative diameter: width where intensity exceeds baseline+frac*amp."""
        vals = np.asarray(vals, dtype=np.float64)
        edge = max(1, int(0.2 * len(vals)))
        baseline = np.mean(np.concatenate([vals[:edge], vals[-edge:]]))
        amp = np.max(vals) - baseline
        if amp <= 0:
            return np.nan
        thresh = baseline + frac * amp
        above = np.where(vals >= thresh)[0]
        if len(above) < 2:
            return np.nan
        return abs(offsets[above[-1]] - offsets[above[0]])

    def _extract_vessel_morphometry(self):
        """Core routine: sample perpendicular profiles along the vessel and
        return per-cross-section diameter, area, and quality flags.

        Returns a results dict (also cached in self.vessel_results) or None.
        """
        if self.foreground_roi is None or len(self.foreground_roi) < 2:
            messagebox.showwarning("Warning",
                                   "Draw the vessel centerline as the Foreground ROI first.")
            return None
        self.update_ear_params()

        mean_img = self._get_mean_image()
        if mean_img is None:
            messagebox.showerror("Error", "No image data.")
            return None

        centers, normals, s_um = self._resample_centerline(
            self.foreground_roi, self.n_cross_sections)
        if centers is None:
            messagebox.showerror("Error", "Centerline has zero length.")
            return None

        px = self.pixel_size
        psf_limit_px = 2.5  # below this the FWHM is PSF-limited, not vessel-limited

        diam_um = np.full(self.n_cross_sections, np.nan)
        area_um2 = np.full(self.n_cross_sections, np.nan)
        rsq = np.full(self.n_cross_sections, np.nan)
        flags = []          # 'ok' | 'psf_limited' | 'fit_poor' | 'no_edge'
        all_profiles = []   # keep for optional display

        for k in range(self.n_cross_sections):
            offsets, vals = self._sample_profile(
                mean_img, centers[k], normals[k], self.profile_half_len)
            all_profiles.append((offsets * px, vals))

            if self.diameter_method == 'threshold':
                w_px = self._threshold_width_from_profile(offsets, vals, frac=0.5)
                info = {'fwhm_px': w_px, 'r_squared': np.nan}
            else:
                info = self._fwhm_from_profile(offsets, vals)
                if info is None:
                    flags.append('no_edge')
                    continue
                w_px = info['fwhm_px']

            if info is not None and np.isfinite(info.get('r_squared', np.nan)):
                rsq[k] = info['r_squared']

            if not np.isfinite(w_px):
                flags.append('no_edge')
                continue

            d_um = w_px * px
            diam_um[k] = d_um
            area_um2[k] = np.pi * (d_um / 2.0) ** 2

            # quality flags
            if w_px < psf_limit_px:
                flags.append('psf_limited')
            elif np.isfinite(rsq[k]) and rsq[k] < 0.8:
                flags.append('fit_poor')
            else:
                flags.append('ok')

        results = {
            'centers': centers,
            'normals': normals,
            's_um': s_um,
            'diam_um': diam_um,
            'area_um2': area_um2,
            'r_squared': rsq,
            'flags': np.array(flags),
            'mean_img': mean_img,
            'profiles': all_profiles,
            'psf_limit_um': psf_limit_px * px,
        }
        self.vessel_results = results
        return results

    def analyze_vessel_morphometry(self):
        """PANEL 1: vessel diameter (FWHM) along the centerline + statistics.

        Produces: overlay of measured cross-sections on the structural image,
        diameter-vs-position profile, diameter histogram, and a quality/summary
        box. This is the foundation every downstream quantity depends on.
        """
        res = self._extract_vessel_morphometry()
        if res is None:
            return

        diam = res['diam_um']
        area = res['area_um2']
        s_um = res['s_um']
        flags = res['flags']
        ok_mask = np.isfinite(diam) & (flags == 'ok')

        win = tk.Toplevel(self.root)
        win.title("Mouse Ear - Vessel Morphometry (FWHM)")
        win.geometry("1300x850")
        fig = Figure(figsize=(14, 9))

        # --- (a) structural image with cross-section markers ---
        ax1 = fig.add_subplot(221)
        img = res['mean_img']
        vmin, vmax = np.percentile(img, [1, 99])
        ax1.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        centers = res['centers']
        normals = res['normals']
        color_map = {'ok': 'lime', 'psf_limited': 'orange',
                     'fit_poor': 'yellow', 'no_edge': 'red'}
        for k in range(len(centers)):
            if not np.isfinite(diam[k]):
                continue
            half_px = (diam[k] / self.pixel_size) / 2.0
            c = centers[k]
            n = normals[k]
            p0 = c - half_px * n
            p1 = c + half_px * n
            fl = flags[k] if k < len(flags) else 'ok'
            ax1.plot([p0[0], p1[0]], [p0[1], p1[1]],
                     color=color_map.get(fl, 'cyan'), linewidth=1.5)
        ax1.plot(centers[:, 0], centers[:, 1], 'c.', markersize=2)
        ax1.set_title('Measured cross-sections\n(green=ok, orange=PSF-limited, red=no edge)')
        ax1.axis('off')

        # --- (b) diameter vs position ---
        ax2 = fig.add_subplot(222)
        ax2.plot(s_um, diam, 'o-', color='steelblue', markersize=4, alpha=0.5,
                 label='all')
        if ok_mask.any():
            ax2.plot(s_um[ok_mask], diam[ok_mask], 'o', color='green',
                     markersize=5, label='ok')
        ax2.axhline(res['psf_limit_um'], color='orange', linestyle='--',
                    linewidth=1, label=f"PSF limit ({res['psf_limit_um']:.1f} µm)")
        ax2.set_xlabel('Position along vessel (µm)')
        ax2.set_ylabel('Diameter FWHM (µm)')
        ax2.set_title('Diameter profile')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # --- (c) diameter histogram ---
        ax3 = fig.add_subplot(223)
        valid = diam[ok_mask]
        if len(valid) > 0:
            ax3.hist(valid, bins=max(5, len(valid) // 2), color='seagreen',
                     alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(valid), color='red', linewidth=2,
                        label=f'mean {np.mean(valid):.1f} µm')
            ax3.legend(fontsize=9)
        ax3.set_xlabel('Diameter (µm)')
        ax3.set_ylabel('Count')
        ax3.set_title('Diameter distribution (ok cross-sections)')
        ax3.grid(True, alpha=0.3)

        # --- (d) summary text ---
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        n_ok = int(ok_mask.sum())
        n_total = len(diam)
        txt = "VESSEL MORPHOMETRY SUMMARY\n" + "=" * 34 + "\n\n"
        if n_ok > 0:
            txt += f"Diameter (FWHM):  {np.mean(valid):.2f} ± {np.std(valid):.2f} µm\n"
            txt += f"  range:          {np.min(valid):.2f} – {np.max(valid):.2f} µm\n"
            txt += f"Cross-sec. area:  {np.nanmean(area[ok_mask]):.1f} ± {np.nanstd(area[ok_mask]):.1f} µm²\n"
            # approximate vessel-segment volume = integral of A ds over ok region
            if n_ok > 1:
                s_ok = s_um[ok_mask]
                a_ok = area[ok_mask]
                order = np.argsort(s_ok)
                vol = _trapz(a_ok[order], s_ok[order])  # µm^3
                txt += f"Segment volume:   {vol:.3e} µm³ = {vol*1e-9:.3f} nL\n"
                length = s_ok.max() - s_ok.min()
                txt += f"Segment length:   {length:.1f} µm\n"
                # tapering: linear fit of diameter vs position
                if n_ok > 2:
                    slope = np.polyfit(s_ok, diam[ok_mask], 1)[0]
                    txt += f"Tapering:         {slope*1000:.2f} µm per mm\n"
        else:
            txt += "No reliable cross-sections found.\n"
        txt += f"\nQuality:  {n_ok}/{n_total} cross-sections OK\n"
        for fl in ['psf_limited', 'fit_poor', 'no_edge']:
            c = int(np.sum(flags == fl))
            if c > 0:
                txt += f"  {fl}: {c}\n"
        txt += f"\nSettings:\n"
        txt += f"  pixel size:  {self.pixel_size:.3f} µm/px\n"
        txt += f"  method:      {self.diameter_method}\n"
        txt += f"  N sections:  {self.n_cross_sections}\n"
        txt += "\nAssumptions: circular lumen; FWHM≈diameter\nvalid only above PSF limit."
        ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, fontsize=9,
                 va='top', fontfamily='monospace')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_axial_kymograph(self, n_axial=None, interpolate_excluded=False,
                               return_flags=False):
        """Build a kymograph SAMPLED ALONG the vessel axis (position x time).

        Unlike the existing videokymography (which averages circular sub-ROIs),
        this samples the sub-pixel centerline value at each frame so that moving
        contrast features form slanted streaks whose slope = velocity.

        Breathing frames are handled in one of two ways:
          * interpolate_excluded=False (default): excluded frames are DROPPED and
            the time axis carries their real acquisition time (gaps stay gaps).
          * interpolate_excluded=True: leading/trailing excluded frames that have
            no bracketing good neighbour on one side are TRIMMED from the analysis
            window (unobserved data is never fabricated); the remaining INTERIOR
            excluded frames sit between two good frames and are reconstructed by
            motion-compensated interpolation, giving a uniform dt.

        Returns (kymo, s_um, t_s) or, if return_flags, (kymo, s_um, t_s, flags)
        where flags[j] in {'observed','interpolated'} labels each kept column.
        """
        if self.foreground_roi is None or len(self.foreground_roi) < 2:
            return None, None, None
        if n_axial is None:
            # sample roughly one point per pixel of centerline length
            centers, normals, s_um = self._resample_centerline(
                self.foreground_roi, 2)
            total_len_px = s_um[-1] / self.pixel_size
            n_axial = max(20, int(total_len_px))
        centers, normals, s_um = self._resample_centerline(
            self.foreground_roi, n_axial)

        start_frame, end_frame = self.get_analysis_frame_range()
        end_frame = min(end_frame, len(self.image_stack) - 1)

        half = max(1, self.foreground_line_width // 2)
        xs_c = centers[:, 0]
        ys_c = centers[:, 1]
        nx = normals[:, 0]
        ny = normals[:, 1]
        cross_off = np.arange(-half, half + 1, dtype=np.float64)

        def _sample(frame_idx):
            frame = self.image_stack[frame_idx].astype(np.float64)
            acc = np.zeros(n_axial)
            for co in cross_off:
                xs = xs_c + co * nx
                ys = ys_c + co * ny
                acc += ndimage.map_coordinates(frame, [ys, xs], order=1, mode='nearest')
            return acc / len(cross_off)

        if interpolate_excluded:
            excl = set(self.excluded_frames)
            inc_frames = [f for f in range(start_frame, end_frame + 1) if f not in excl]
            if len(inc_frames) < 2:
                # nothing meaningful to bracket -> treat the whole range as observed
                win = list(range(start_frame, end_frame + 1))
                kymo = np.zeros((n_axial, len(win)))
                for i, fidx in enumerate(win):
                    kymo[:, i] = _sample(fidx)
                flags = np.array(['observed'] * len(win), dtype='U16')
                t_s = np.asarray(win, dtype=np.float64) / self.frame_rate
                return (kymo, s_um, t_s, flags) if return_flags else (kymo, s_um, t_s)

            # TRIM un-bracketed leading/trailing excluded frames: the analysis
            # window is [first good .. last good]. Everything outside has no
            # bracketing pair, so we never fabricate it (no hold-last-value).
            a0, a1 = inc_frames[0], inc_frames[-1]
            n_trim_lead = a0 - start_frame
            n_trim_trail = end_frame - a1
            win = list(range(a0, a1 + 1))           # contiguous -> uniform dt
            n_frames = len(win)
            inc_local = [i for i, f in enumerate(win) if f not in excl]  # all bracketed
            kymo = np.zeros((n_axial, n_frames))
            for li in inc_local:
                kymo[:, li] = _sample(win[li])
            flags = np.array(['observed'] * n_frames, dtype='U16')
            excl_local = [i for i in range(n_frames) if i not in inc_local]
            for i in excl_local:
                flags[i] = 'interpolated'
            if excl_local:
                kymo = self._motion_fill_columns(kymo, inc_local)
            self._log_console(
                f"[flow-speed] columns -> observed={len(inc_local)}, "
                f"interpolated(bracketed)={len(excl_local)}, "
                f"trimmed(un-bracketed ends)={n_trim_lead + n_trim_trail} "
                f"(lead={n_trim_lead}, trail={n_trim_trail})")
            t_s = np.asarray(win, dtype=np.float64) / self.frame_rate
            return (kymo, s_um, t_s, flags) if return_flags else (kymo, s_um, t_s)

        # default: drop breathing-excluded frames (real, possibly non-uniform time)
        frames = self._included_frames(start_frame, end_frame)
        if len(frames) < 2:
            frames = list(range(start_frame, end_frame + 1))
        n_frames = len(frames)
        kymo = np.zeros((n_axial, n_frames))
        for fi, frame_idx in enumerate(frames):
            kymo[:, fi] = _sample(frame_idx)
        t_s = np.asarray(frames, dtype=np.float64) / self.frame_rate
        if return_flags:
            return kymo, s_um, t_s, np.array(['observed'] * n_frames, dtype='U16')
        return kymo, s_um, t_s

    def _log_console(self, msg):
        """Lightweight console log (kept separate from the status bar)."""
        try:
            print(msg)
        except Exception:
            pass

    @staticmethod
    def _xcorr_shift_1d(a, b, max_lag=None):
        """Sub-pixel position shift that best maps profile `a` onto `b`
        (b[i] ~= a[i - shift]), via normalized cross-correlation + parabolic
        refinement. Positive = pattern moved toward +position."""
        a = np.asarray(a, float) - np.mean(a)
        b = np.asarray(b, float) - np.mean(b)
        n = a.size
        if n < 3 or np.allclose(a, 0) or np.allclose(b, 0):
            return 0.0
        if max_lag is None:
            max_lag = n // 2
        corr = np.correlate(b, a, mode='full')          # lag = k-(n-1)
        lags = np.arange(-(n - 1), n)
        keep = np.abs(lags) <= max_lag
        corr = corr[keep]; lags = lags[keep]
        k = int(np.argmax(corr))
        lag = float(lags[k])
        if 0 < k < corr.size - 1:
            ym1, y0, yp1 = corr[k - 1], corr[k], corr[k + 1]
            den = ym1 - 2 * y0 + yp1
            if abs(den) > 1e-9:
                lag += 0.5 * (ym1 - yp1) / den
        return lag

    def _motion_fill_columns(self, kymo, inc_local):
        """Fill excluded (breathing) kymograph columns by MOTION-COMPENSATED
        interpolation instead of a flat per-row line.

        For each gap between two good columns L and R, the along-vessel shift of
        the pattern is estimated by cross-correlation; every missing column is
        then rebuilt by warping L forward and R backward by the appropriate
        fraction of that shift and blending. This makes the moving bolus advance
        continuously through the gap (its peak keeps moving) rather than
        collapsing to near-constant values."""
        out = kymo.copy()
        inc = sorted(inc_local)
        n_axial = kymo.shape[0]
        for L, R in zip(inc[:-1], inc[1:]):
            if R - L <= 1:
                continue
            colL = kymo[:, L]; colR = kymo[:, R]
            # total shift over the whole gap, capped to a sane range
            d = self._xcorr_shift_1d(colL, colR, max_lag=max(2, n_axial // 3))
            for m in range(L + 1, R):
                f = (m - L) / (R - L)
                warpedL = ndimage.shift(colL, f * d, order=1, mode='nearest')
                warpedR = ndimage.shift(colR, -(1.0 - f) * d, order=1, mode='nearest')
                out[:, m] = (1.0 - f) * warpedL + f * warpedR
        # NOTE: callers TRIM un-bracketed leading/trailing excluded columns before
        # calling this, so every gap here is bracketed. We deliberately do NOT
        # hold-last-value at the ends (that fabricated flat plateaus).
        return out

    def _velocity_from_kymo(self, kymo, s_um, t_s):
        """Estimate axial velocity from streak orientation in the kymograph.

        The local direction of moving-blood streaks (iso-intensity lines) is the
        eigenvector of the 2D structure tensor with the SMALLEST eigenvalue —
        i.e. the direction along which intensity changes least. The slope
        (position-step per time-step) of that eigenvector IS the velocity in
        index space; multiplying by ds/dt converts to physical units and the
        SIGN encodes flow direction. This is a direct, amplitude-independent
        kinematic readout, far stronger than intensity-amplitude surrogates.

        VALIDATED against synthetic ground truth: accurate for |slope_index| ≲ 2.
        Steeper (near-vertical) streaks correspond to flow that is fast relative
        to the spatial/temporal sampling and are UNDER-SAMPLED — they read low.
        A resolvable-velocity ceiling is computed and returned so callers can
        warn when the measurement is sampling-limited rather than physical.
        """
        # physical step sizes per index
        ds = np.mean(np.diff(s_um)) if len(s_um) > 1 else self.pixel_size  # µm
        dt = np.mean(np.diff(t_s)) if len(t_s) > 1 else 1.0 / self.frame_rate  # s
        conv = (ds / dt) * 1e-3  # mm/s per unit index-slope

        # smooth to stabilize gradients
        k = ndimage.gaussian_filter(kymo.astype(np.float64), sigma=1.0)
        g_row, g_col = np.gradient(k)  # d/d(position), d/d(time)

        # windowed structure tensor J = [[Jrr, Jrc], [Jrc, Jcc]]
        w = 2.0
        Jrr = ndimage.gaussian_filter(g_row * g_row, w)
        Jcc = ndimage.gaussian_filter(g_col * g_col, w)
        Jrc = ndimage.gaussian_filter(g_row * g_col, w)

        # eigenvalues of the 2x2 symmetric tensor
        tr = Jrr + Jcc
        det = Jrr * Jcc - Jrc * Jrc
        disc = np.sqrt(np.maximum(0.0, (tr / 2.0) ** 2 - det))
        lam_min = tr / 2.0 - disc
        lam_max = tr / 2.0 + disc

        # eigenvector (v_row, v_col) for the smallest eigenvalue = streak direction
        vr = Jrc
        vc = lam_min - Jrr
        norm = np.sqrt(vr * vr + vc * vc) + 1e-12
        vr, vc = vr / norm, vc / norm

        # slope in index space (position-step per time-step) -> physical velocity
        slope_index = np.where(np.abs(vc) > 1e-9,
                               vr / np.where(np.abs(vc) > 1e-9, vc, 1.0), 0.0)
        v_mm_per_s = slope_index * conv

        # coherence in [0,1]: how strongly oriented the local patch is
        coherence = (lam_max - lam_min) / (lam_max + lam_min + 1e-12)

        # per-time-column robust velocity (median over position where coherent)
        v_col = np.full(kymo.shape[1], np.nan)
        for c in range(kymo.shape[1]):
            good = coherence[:, c] > 0.3
            if good.sum() >= 3:
                v_col[c] = np.median(v_mm_per_s[good, c])

        # global robust estimate over well-oriented pixels
        good_all = coherence > 0.4
        if good_all.sum() > 10:
            v_global = np.median(v_mm_per_s[good_all])
            v_global_mad = np.median(np.abs(v_mm_per_s[good_all] - v_global))
        else:
            v_global = np.nan
            v_global_mad = np.nan

        # sampling-limited ceiling: reliable up to |slope_index| ~ 2
        v_resolvable_ceiling = 2.0 * conv  # mm/s

        # Detect under-sampling from the SLOPE SATURATION signature: when true
        # flow exceeds the ceiling, streaks become near-vertical, so coherent
        # pixels pile up at large |slope| and the slope distribution develops a
        # heavy near-vertical tail. Two complementary signals are combined:
        #   (1) saturation fraction: share of coherent pixels with |slope| > 1.5
        #   (2) slope spread: 90th percentile of |slope|
        # (Validated on synthetic vessels: fires for |v| >= ceiling, stays quiet
        # for resolvable slow flow.)
        if good_all.sum() > 10:
            slope_good = np.abs(slope_index[good_all])
            saturation_fraction = float(np.mean(slope_good > 1.5))
            slope_p90 = float(np.percentile(slope_good, 90))
        else:
            saturation_fraction = np.nan
            slope_p90 = np.nan
        sampling_limited = (np.isfinite(saturation_fraction) and
                            (saturation_fraction > 0.2 or slope_p90 > 2.5))

        return {
            'v_field_mm_s': v_mm_per_s,
            'coherence': coherence,
            'v_col_mm_s': v_col,
            'v_global_mm_s': v_global,
            'v_global_mad': v_global_mad,
            'ds_um': ds,
            'dt_s': dt,
            'conv_mm_s_per_slope': conv,
            'v_resolvable_ceiling_mm_s': v_resolvable_ceiling,
            'saturation_fraction': saturation_fraction,
            'sampling_limited': sampling_limited,
        }

    def _dp_peak_track(self, dksd, min_adv, max_adv, seed_rate, col_weight,
                       n_cand=5, prominence_mad=3.0, snr_cap=10.0, max_gap=6,
                       rate_penalty=0.3, gap_penalty=0.5, force_seed_win=None):
        """Global (Viterbi/DP) peak track over a difference kymograph oriented so
        that the flow direction is +row.

        Per column we extract up to `n_cand` prominent local maxima (emission =
        SNR, capped, times the column's provenance weight). The optimal path is
        the maximum-score sequence of candidates under a MONOTONE, VELOCITY-
        BOUNDED transition: an advance of `d` rows across `gap` skipped steps is
        admissible only if `min_adv*gap <= d <= max_adv*gap` (never backward).
        Skipped columns are 'no-detection' gaps; the allowance widens with the
        gap. Because the path is optimised globally, a single spurious bright
        column cannot poison the remainder of the track (unlike a greedy scan).
        Returns (sel_row[int, -1 where off-path], total_score)."""
        from scipy.signal import find_peaks
        n_axial, n_steps = dksd.shape
        med = np.median(dksd, axis=0)
        mad = np.median(np.abs(dksd - med[None, :]), axis=0) * 1.4826 + 1e-9
        cand = []                                   # per column: [(row, emission), ...]
        for k in range(n_steps):
            col = dksd[:, k]
            thr = max(prominence_mad * mad[k], 1e-9)
            pk, _props = find_peaks(col, prominence=thr)
            if pk.size == 0:
                pk = np.array([int(np.argmax(col))])
            val = col[pk]
            keep = val > med[k]                     # bright (positive) fronts only
            pk = pk[keep]; val = val[keep]
            if pk.size == 0:
                cand.append([]); continue
            emis = np.clip((val - med[k]) / mad[k], 0.0, snr_cap) * float(col_weight[k])
            order = np.argsort(emis)[::-1][:int(n_cand)]
            cand.append(list(zip(pk[order].tolist(), emis[order].tolist())))

        # --- proximal-seed gating: force the path to ORIGINATE near row 0 (the
        # seed end, in oriented space) at the first frame the front arrives there.
        # `arrival` is that seed column; only near-seed candidates at/after it may
        # start a path, so tracking begins at the proximal end and reaches distal
        # as far as signal allows (max total emission = max coverage). ---
        arrival = None
        if force_seed_win is not None:
            for k in range(n_steps):
                if any(r <= force_seed_win for (r, _e) in cand[k]):
                    arrival = k
                    break
            if arrival is None:                     # no near-seed front: fall back
                force_seed_win = None

        NEG = -1e18
        best = []
        for k in range(n_steps):
            arr = np.full(len(cand[k]), NEG)
            for m, (r, e) in enumerate(cand[k]):
                is_start = (force_seed_win is None) or \
                           (k >= arrival and r <= force_seed_win)
                if is_start:
                    arr[m] = e
            best.append(arr)
        back = [[None] * len(cand[k]) for k in range(n_steps)]
        for k in range(n_steps):
            for m, (r_a, _e) in enumerate(cand[k]):
                base = best[k][m]
                if base <= NEG / 2:
                    continue
                for k2 in range(k + 1, min(n_steps, k + 1 + int(max_gap))):
                    gap = k2 - k
                    lo = min_adv * gap - 1e-9
                    hi = max_adv * gap + 1e-9
                    for m2, (r_b, e_b) in enumerate(cand[k2]):
                        d = r_b - r_a
                        if d < lo or d > hi:
                            continue
                        rate = d / gap
                        pen = rate_penalty * abs(rate - seed_rate) + gap_penalty * (gap - 1)
                        sc = base + e_b - pen
                        if sc > best[k2][m2]:
                            best[k2][m2] = sc
                            back[k2][m2] = (k, m)
        bk = bm = -1; bs = NEG
        for k in range(n_steps):
            for m in range(len(cand[k])):
                if best[k][m] > bs:
                    bs = best[k][m]; bk = k; bm = m
        sel = np.full(n_steps, -1, dtype=int)
        node = (bk, bm) if bk >= 0 else None
        while node is not None:
            k, m = node
            sel[k] = cand[k][m][0]
            node = back[k][m]
        return sel, float(bs), arrival

    def _greedy_peak_track(self, dksd, min_adv, max_adv, col_weight,
                           prominence_mad=3.0):
        """Causal greedy tracker (fallback/debug only).

        Seeds at the highest-SNR column, then at each forward step takes the
        argmax within the admissible forward window [r_prev+min_adv,
        r_prev+max_adv]. FAILURE MODE: a single early false detection locks in
        and poisons every later step with no chance of recovery — always prefer
        the DP method for quantitative results. Returns (sel_row, score)."""
        n_axial, n_steps = dksd.shape
        med = np.median(dksd, axis=0)
        mad = np.median(np.abs(dksd - med[None, :]), axis=0) * 1.4826 + 1e-9
        snr_col = (np.max(dksd, axis=0) - med) / mad
        seed_k = int(np.argmax(snr_col))
        sel = np.full(n_steps, -1, dtype=int)
        sel[seed_k] = int(np.argmax(dksd[:, seed_k]))
        r_prev = sel[seed_k]
        for k in range(seed_k + 1, n_steps):
            lo = max(0, int(np.floor(r_prev + min_adv)))
            hi = min(n_axial - 1, int(np.ceil(r_prev + max_adv)))
            if hi < lo:
                break
            win = dksd[lo:hi + 1, k]
            r = lo + int(np.argmax(win))
            if (win.max() - med[k]) / mad[k] < 1.0:      # lost the front -> stop
                break
            sel[k] = r; r_prev = r
        return sel, float(np.nansum(snr_col))

    def _track_peak_speed(self, kymo, s_um, t_s, smooth_sigma=1.0, snr_thr=3.0,
                          local_win=5, col_flags=None, peak_floor_frac=0.1,
                          smooth_track_pchip=True, direction='auto', method='dp',
                          n_cand=5, v_min=0.0, v_max=None, prominence_mad=3.0,
                          seed_end='auto', seed_window_um=None, max_miss=5):
        """Seeded, direction-constrained, velocity-bounded consecutive-frame peak track.

        Subtracting frame N from N+1 cancels the static vessel and leaves the
        MOVING contrast (a bright bolus front). Instead of an independent per-
        column argmax (which scatters onto noise and has no flow direction), the
        peak track is recovered by a boundary-constrained DP/Viterbi search
        (`method='dp'`, default) that enforces a MONOTONE, velocity-bounded
        advance; `method='greedy'` is a causal fallback.

        SEEDING (`seed_end`): the path is FORCED to originate at one vessel end
        and reach as far as the signal allows (maximising coverage), instead of
        starting wherever the signal is brightest:
          * 'proximal' — anchor at the minimum arc-length s (s_min); front
            advances toward increasing s (proximal->distal).
          * 'distal'   — anchor at s_max; advances toward decreasing s.
          * 'auto' (default) — run BOTH and keep the higher-scoring path.
        WHY 'auto' is the default and not a hard proximal seed: the ROI drawing
        order (hence the sign of s_um) is arbitrary, so hard-forcing the proximal
        end would return a WRONG-SIGNED speed for ~half of all ROIs. 'auto' picks
        the end that yields the physically consistent monotone path; a mismatch
        against the xcorr cross-check is surfaced as an explicit warning.

        `seed_window_um` = width of the near-seed search band (default 15% of the
        ROI length). `max_miss` = max consecutive lost frames tolerated before the
        track terminates. `v_min`/`v_max` (µm/s) bound the per-step advance; v_max
        defaults to the sampling-resolvable ceiling 2*ds/dt. `col_flags` (per
        kymograph column) down-weights synthesised columns and EXCLUDES non-
        observed steps from the reported speed/R²/CI. Sub-pixel parabolic
        refinement and the SNR validity gate are preserved.

        Returns a dict incl. direction_resolved, seed_frame, coverage fractions,
        per-step track_state, an xcorr speed cross-check, and step counts.
        """
        from scipy.stats import theilslopes
        from scipy.interpolate import PchipInterpolator

        kymo = np.asarray(kymo, dtype=np.float64)
        if kymo.shape[1] < 3:
            return None
        dk = np.diff(kymo, axis=1)                       # (n_axial, n_steps)
        t_mid = 0.5 * (t_s[:-1] + t_s[1:])
        dks = ndimage.gaussian_filter1d(dk, sigma=smooth_sigma, axis=0)
        n_axial, n_steps = dks.shape
        cols = np.arange(n_steps)

        ds_um = float(np.mean(np.diff(s_um))) if s_um.size > 1 else self.pixel_size
        dt_s = float(np.mean(np.diff(t_s))) if t_s.size > 1 else 1.0 / self.frame_rate

        # velocity bounds (µm/s) -> admissible index-advance per step
        if v_max is None or not np.isfinite(v_max) or v_max <= 0:
            v_max_um_s = 2.0 * ds_um / dt_s              # sampling-resolvable ceiling
        else:
            v_max_um_s = float(v_max)
        v_min_um_s = max(0.0, float(v_min))
        max_adv = v_max_um_s * dt_s / ds_um
        min_adv = v_min_um_s * dt_s / ds_um

        # per-STEP provenance + emission weight (a step spans columns j, j+1)
        if col_flags is not None and len(col_flags) == kymo.shape[1]:
            cf = np.asarray(col_flags)
            step_flag = np.empty(n_steps, dtype='U16'); step_flag[:] = 'observed'
            for j in range(n_steps):
                pair = {str(cf[j]), str(cf[j + 1])}
                if 'extrapolated' in pair:
                    step_flag[j] = 'extrapolated'
                elif 'interpolated' in pair:
                    step_flag[j] = 'interpolated'
        else:
            step_flag = np.array(['observed'] * n_steps, dtype='U16')
        wmap = {'observed': 1.0, 'interpolated': 0.5, 'extrapolated': 0.2}
        col_weight = np.array([wmap.get(f, 1.0) for f in step_flag], float)

        # seed advance rate (index/step, signed) from robust xcorr displacement
        raw_shift = np.array([
            self._xcorr_shift_1d(kymo[:, k], kymo[:, k + 1],
                                 max_lag=max(2, int(np.ceil(max_adv)) + 1))
            for k in range(n_steps)])
        seed_signed = float(np.median(raw_shift))

        # near-seed search band (index) and max gap from max_miss
        s_len = float(s_um[-1] - s_um[0]) if s_um.size > 1 else n_axial * ds_um
        seed_win_um = float(seed_window_um) if seed_window_um else 0.15 * s_len
        seed_win_idx = int(np.clip(round(seed_win_um / ds_um), 2, max(2, n_axial // 2)))
        _max_gap = int(max(1, (max_miss if max_miss is not None else 5))) + 1

        # `direction` (legacy) can still force a sign; otherwise seed_end drives it.
        if direction in ('+', '-'):
            seed_end = 'proximal' if direction == '+' else 'distal'

        def _run(orient):
            # orient '+' : seed at s_min (row 0), forward = +row = increasing s.
            # orient '-' : flip so s_max becomes row 0, forward = decreasing s.
            dksd = dks if orient == '+' else dks[::-1, :]
            if method == 'greedy':
                sel_o, sc = self._greedy_peak_track(dksd, min_adv, max_adv, col_weight,
                                                    prominence_mad=prominence_mad)
                return sel_o, sc, None
            return self._dp_peak_track(dksd, min_adv, max_adv, abs(seed_signed),
                                       col_weight, n_cand=n_cand,
                                       prominence_mad=prominence_mad, max_gap=_max_gap,
                                       force_seed_win=seed_win_idx)

        if seed_end == 'proximal':
            orient = '+'; sel_o, score, arrival = _run('+'); how = 'seed=proximal'
        elif seed_end == 'distal':
            orient = '-'; sel_o, score, arrival = _run('-'); how = 'seed=distal'
        else:  # auto: seed BOTH ends, keep higher-scoring physically-consistent path
            sp, scp, ap = _run('+'); sm, scm, am = _run('-')
            if scp >= scm:
                orient, sel_o, score, arrival = '+', sp, scp, ap
            else:
                orient, sel_o, score, arrival = '-', sm, scm, am
            how = f'auto (score prox={scp:.1f} vs distal={scm:.1f})'
        sign = +1 if orient == '+' else -1
        direction_resolved = '+' if sign >= 0 else '-'
        seed_frame = int(arrival) if arrival is not None else -1

        # map oriented selected rows back to original row index
        sel = np.full(n_steps, -1, dtype=int)
        on = sel_o >= 0
        sel[on] = sel_o[on] if sign >= 0 else (n_axial - 1) - sel_o[on]

        # --- SUB-PIXEL parabolic refinement on the SELECTED candidate (preserved) ---
        peak_i = np.where(sel >= 0, sel, 0)
        pk = peak_i.astype(np.float64)
        inr = on & (peak_i > 0) & (peak_i < n_axial - 1)
        if inr.any():
            c = cols[inr]; ic = peak_i[inr]
            ym1 = dks[ic - 1, c]; y0 = dks[ic, c]; yp1 = dks[ic + 1, c]
            denom = (ym1 - 2 * y0 + yp1)
            shift = np.where(np.abs(denom) > 1e-9, 0.5 * (ym1 - yp1) / denom, 0.0)
            pk[inr] = ic + np.clip(shift, -1.0, 1.0)
        peak_s = np.full(n_steps, np.nan)
        peak_s[on] = s_um[0] + pk[on] * ds_um
        peak_val = np.full(n_steps, np.nan)
        peak_val[on] = dks[peak_i[on], cols[on]]

        # --- reinforced validity gate (SNR + relative floor) on selected peaks ---
        med = np.median(dks, axis=0)
        mad = np.median(np.abs(dks - med[None, :]), axis=0) * 1.4826 + 1e-9
        snr = np.full(n_steps, np.nan)
        snr[on] = (peak_val[on] - med[on]) / mad[on]
        peak_ref = np.nanmax(peak_val) if np.isfinite(np.nanmax(peak_val)) else 0.0
        floor = peak_floor_frac * peak_ref
        valid = on & (peak_val > 0) & (snr >= snr_thr) & (peak_val >= floor)

        # --- per-step track_state: observed/interpolated where accepted, else
        # no-detection (gap / not-yet-entered) or terminated (past last accept) ---
        track_state = np.array(['no-detection'] * n_steps, dtype='U16')
        det = np.where(valid)[0]
        if det.size:
            last_k = det[-1]
            for k in range(n_steps):
                if valid[k]:
                    track_state[k] = step_flag[k]
                elif k > last_k:
                    track_state[k] = 'terminated'   # bolus exited -> not pinned
        # fit uses ONLY observed accepted steps (no synthesised data in the fit)
        fit_mask = valid & (step_flag == 'observed')

        # --- smooth the accepted peak track over time (display / local speed) ---
        peak_s_sm = peak_s.copy()
        if valid.sum() >= 5:
            vs = peak_s[valid]
            kmed = int(min(len(vs) - (len(vs) + 1) % 2, 5))
            if kmed >= 3:
                vs = medfilt(vs, kernel_size=kmed)
            vs = ndimage.gaussian_filter1d(vs, sigma=1.5)
            peak_s_sm[valid] = vs

        speed_mm_s = np.nan; speed_lo = speed_hi = np.nan
        r_squared = np.nan
        if fit_mask.sum() >= 3:
            tv = t_mid[fit_mask]; sv = peak_s_sm[fit_mask]
            slope, intercept, lo, hi = theilslopes(sv, tv)   # µm/s
            speed_mm_s = slope * 1e-3
            speed_lo, speed_hi = lo * 1e-3, hi * 1e-3
            fit = intercept + slope * tv
            ss_res = np.sum((sv - fit) ** 2); ss_tot = np.sum((sv - np.mean(sv)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        direction = ('prox->dist (+)' if direction_resolved == '+' else 'dist->prox (-)')

        # --- independent cross-check: forward-restricted per-step xcorr advance,
        # reported SIGNED along s_um in the same convention as the peak speed ---
        fwd = (1.0 if sign >= 0 else -1.0) * raw_shift        # forward advance (>=0)
        fwd = np.clip(fwd, min_adv, max_adv)
        obs = (step_flag == 'observed')
        fwd_used = fwd[obs] if obs.sum() >= 3 else fwd
        xcorr_speed_mm_s = float((1.0 if sign >= 0 else -1.0)
                                 * np.median(fwd_used) * ds_um / dt_s * 1e-3)

        # --- PCHIP reconnection of the accepted track (visual continuity) ---
        track_t = np.array([]); track_s = np.array([])
        if smooth_track_pchip and valid.sum() >= 3:
            tf = t_mid[valid]; sf = peak_s_sm[valid]
            o = np.argsort(tf); tf = tf[o]; sf = sf[o]
            tu, idx = np.unique(tf, return_index=True); su = sf[idx]
            if tu.size >= 3:
                pch = PchipInterpolator(tu, su)
                track_t = np.linspace(tu[0], tu[-1], max(120, tu.size * 6))
                track_s = pch(track_t)

        # --- per-position LOCAL speed for the spatial map (accepted steps) ---
        local_pos = np.array([]); local_speed = np.array([])
        prof_pos = np.array([]); prof_speed = np.array([])
        if valid.sum() >= 3:
            tv = t_mid[valid]; sv = peak_s_sm[valid]
            o = np.argsort(tv); tv = tv[o]; sv = sv[o]
            lv = np.full(tv.size, np.nan)
            w = max(5, int(local_win))
            for i in range(tv.size):
                a = max(0, i - w // 2); b = min(tv.size, a + w)
                if b - a >= 2 and np.ptp(tv[a:b]) > 0:
                    lv[i] = np.polyfit(tv[a:b], sv[a:b], 1)[0] * 1e-3
            good = np.isfinite(lv)
            local_pos = sv[good]; local_speed = np.abs(lv[good])
            if local_pos.size >= 3:
                po = np.argsort(local_pos)
                pp = local_pos[po]; ps = local_speed[po]
                upp, inv = np.unique(np.round(pp, 3), return_inverse=True)
                ups = np.array([np.median(ps[inv == k]) for k in range(upp.size)])
                prof_pos = s_um.copy()
                prof_speed = np.interp(prof_pos, upp, ups)
                sig = max(1.0, (0.05 * s_um.size))
                prof_speed = ndimage.gaussian_filter1d(prof_speed, sigma=sig)

        n_acc = int(valid.sum())
        n_obs_fit = int(fit_mask.sum())
        n_nodet = int(np.sum(track_state == 'no-detection'))
        n_term = int(np.sum(track_state == 'terminated'))
        # --- honest coverage reporting ---
        if n_acc:
            tracked_s = peak_s[valid]
            reach_frac = float((np.nanmax(tracked_s) - np.nanmin(tracked_s)) /
                               max(1e-9, s_len))
        else:
            reach_frac = 0.0
        obs_frac = float(n_obs_fit) / max(1, n_steps)
        # sign disagreement between peak-track and independent xcorr cross-check
        sign_conflict = bool(np.isfinite(speed_mm_s) and np.isfinite(xcorr_speed_mm_s)
                             and abs(xcorr_speed_mm_s) > 0.02
                             and (speed_mm_s * xcorr_speed_mm_s < 0))
        agree = (abs(speed_mm_s - xcorr_speed_mm_s)
                 if np.isfinite(speed_mm_s) else np.nan)
        self._log_console(
            f"[flow-speed] direction={direction_resolved} [{how}]; seed_frame={seed_frame}; "
            f"v_bounds=[{v_min_um_s:.0f},{v_max_um_s:.0f}] µm/s; "
            f"coverage: reach={100*reach_frac:.0f}% of ROI, observed={100*obs_frac:.0f}% of frames; "
            f"steps: accepted={n_acc}/{n_steps}, no-detection={n_nodet}, terminated={n_term}")
        self._log_console(
            f"[flow-speed] peak speed={speed_mm_s:+.3f} mm/s vs xcorr cross-check="
            f"{xcorr_speed_mm_s:+.3f} mm/s"
            + ("  [WARN: seed-end forced a direction that CONFLICTS with the xcorr "
               "cross-check sign -- try seed_end='auto' or the opposite end]"
               if sign_conflict else
               (f"  [WARN: magnitude disagreement {agree:.3f} mm/s]"
                if np.isfinite(agree) and agree > 0.3 * max(1e-6, abs(xcorr_speed_mm_s))
                else "")))

        return {
            'diff_kymo': dks, 't_mid_s': t_mid, 's_um': s_um,
            'peak_s_um': peak_s_sm, 'peak_val': peak_val, 'valid': valid,
            'step_flag': step_flag, 'fit_mask': fit_mask, 'track_state': track_state,
            'track_t_s': track_t, 'track_s_um': track_s,
            'speed_mm_s': speed_mm_s, 'speed_ci': (speed_lo, speed_hi),
            'r_squared': r_squared, 'direction': direction,
            'direction_resolved': direction_resolved, 'seed_end': seed_end,
            'seed_frame': seed_frame, 'reach_frac': reach_frac, 'obs_frac': obs_frac,
            'sign_conflict': sign_conflict,
            'xcorr_speed_mm_s': xcorr_speed_mm_s,
            'v_min_um_s': v_min_um_s, 'v_max_um_s': v_max_um_s,
            'local_pos_um': local_pos, 'local_speed_mm_s': local_speed,
            'prof_pos_um': prof_pos, 'prof_speed_mm_s': prof_speed,
            'n_valid': n_acc, 'n_steps': int(n_steps), 'n_fit': n_obs_fit,
            'n_nodet': n_nodet, 'n_terminated': n_term,
        }

    # ===================== ARRIVAL-TIME (TRANSIT-TIME) METHOD =====================
    # Transposed estimator: for each POSITION, ask WHEN the bolus passed (a clean
    # single-passage waveform per row), extract one arrival time per position, and
    # take speed = 1 / d(t_arrival)/ds. No frame-to-frame tracking (no lock-in),
    # monotonicity is isotonic regression (PAVA, globally optimal), and breathing-
    # excluded frames are simply MISSING samples (uniform dt not required, so this
    # method uses interpolate_excluded=False and never fabricates intensity data).
    @staticmethod
    def _isotonic_fit(y, w=None, increasing=True):
        """Weighted pool-adjacent-violators (PAVA) isotonic regression, O(n).
        Returns the monotone least-squares fit to y (non-decreasing if increasing,
        else non-increasing), honouring optional weights w."""
        y = np.asarray(y, float); n = y.size
        if n == 0:
            return y.copy()
        w = np.ones(n) if w is None else np.asarray(w, float)
        if not increasing:
            return TIFAnalyzer._isotonic_fit(y[::-1], w[::-1], True)[::-1]
        vals = []; wts = []; cnts = []
        for i in range(n):
            v = float(y[i]); ww = float(w[i]) if w[i] > 0 else 1e-9; c = 1
            while vals and vals[-1] > v:            # pool adjacent violators
                pv = vals.pop(); pw = wts.pop(); pc = cnts.pop()
                v = (pv * pw + v * ww) / (pw + ww); ww += pw; c += pc
            vals.append(v); wts.append(ww); cnts.append(c)
        out = np.empty(n); pos = 0
        for v, c in zip(vals, cnts):
            out[pos:pos + c] = v; pos += c
        return out

    def _track_arrival_time(self, kymo, s_um, t_s, timing_feature='t50',
                            direction='auto', snr_thr=3.0, smooth_win=5,
                            baseline_pct=10.0, baseline_win=None, edge_frames=2,
                            v_max=None, v_min=0.0, min_valid_frac=0.5,
                            trend_reject=True, trend_k=2.5, max_resid_s=None,
                            trend_max_iter=5, local_win_um=None, local_min_pts=4,
                            min_span_um=None, w_max_um=None, ci_rel_thresh=0.5,
                            shrink_to_global=True):
        """Arrival-time (indicator-dilution) speed estimator.

        Each kymograph ROW is a fixed-position time course (baseline -> rise ->
        peak -> decay). We extract one arrival time per position and fit
        t_arrival(s); speed = 1 / slope. `timing_feature`:
          't50' (default) — half-rise time on the leading edge (steep -> well
            conditioned). 'max_slope' — time of max dI/dt (parabolic refine).
            'xcorr' — cross-correlation lag vs an upstream reference row (whole-
            waveform, usually most robust). 'ttp' — time-to-peak (argmax); WORST
            conditioned because the bolus peak is flat-topped, provided only for
            comparison.

        Excluded frames were already dropped by the (non-interpolated) kymograph
        builder, so columns are real samples with possibly non-uniform t_s; a time
        gap straddling a row's leading edge invalidates that row (no guessing).

        Regression note: we regress t_arrival ON s (not s on t) because s is a
        noise-free ROI parameterisation while t_arrival carries the measurement
        noise — this puts the noise on the dependent variable, where least squares
        expects it.
        """
        from scipy.stats import theilslopes
        try:
            from scipy.signal import savgol_filter, find_peaks
        except Exception:
            savgol_filter = None; find_peaks = None

        kymo = np.asarray(kymo, float)
        n_ax, T = kymo.shape
        t = np.asarray(t_s, float)
        if T < 5 or n_ax < 3:
            return None
        dt_nom = float(np.median(np.diff(t))) if T > 1 else 1.0 / self.frame_rate

        def _smooth(x):
            w = int(smooth_win)
            if savgol_filter is not None and w >= 3 and w < x.size and w % 2 == 1:
                try:
                    return savgol_filter(x, w, min(3, w - 1))
                except Exception:
                    pass
            if w > 1:
                k = np.ones(w) / w
                return np.convolve(x, k, mode='same')
            return x

        t_arr = np.full(n_ax, np.nan)
        weights = np.zeros(n_ax)
        reasons = np.array(['ok'] * n_ax, dtype='U16')
        norm_rows = [None] * n_ax          # for the waterfall panel

        # ---- reference row for xcorr: highest-SNR row in the upstream 30% ----
        ref_wave = None
        if timing_feature == 'xcorr':
            best = -1; best_snr = 0.0
            up = max(3, int(0.3 * n_ax))
            for i in range(up):
                x = kymo[i]; I0 = np.percentile(x, baseline_pct)
                xs = _smooth(x - I0); A = float(np.nanmax(xs))
                noise = np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-9
                if A / noise > best_snr:
                    best_snr = A / noise; best = i; ref_wave = xs / (A + 1e-9)
            ref_idx = best

        for i in range(n_ax):
            x = kymo[i].astype(float)
            if baseline_win is not None:
                a, b = baseline_win
                I0 = float(np.mean(x[max(0, a):max(a + 1, b)]))
            else:
                I0 = float(np.percentile(x, baseline_pct))
            dI = x - I0
            xs = _smooth(dI)
            A = float(np.nanmax(xs))
            noise = float(np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-9)
            # gate 1: passage amplitude SNR
            if A < snr_thr * noise or A <= 0:
                reasons[i] = 'low_snr'; continue
            norm = xs / A
            norm_rows[i] = norm
            pk = int(np.argmax(xs))
            # gate 1b: PASSAGE PEAK pinned at the window edge -> the passage was not
            # fully captured here (rising into, or decaying out of, the window). Its
            # amplitude/normalisation are unreliable, so reject rather than clamp.
            # (This is the transposed form of the pinned-track artefact and is the
            # gate that catches clipped rows even for the leading-edge t50 feature.)
            if pk <= edge_frames:
                reasons[i] = 'edge_pinned_start'; continue    # passage rising INTO window
            if pk >= T - 1 - edge_frames:
                reasons[i] = 'edge_pinned_end'; continue      # passage decaying OUT at end
            # gate 2: multi-passage / non-unimodal (recirculation, >1 bolus)
            if find_peaks is not None:
                pks, _p = find_peaks(xs, prominence=0.4 * A)
                if pks.size > 1:
                    reasons[i] = 'multimodal'; continue

            feat_t = None
            if timing_feature == 'ttp':
                # parabolic refine around the (flat) peak
                feat_t = t[pk]
                if 0 < pk < T - 1:
                    ym1, y0, yp1 = xs[pk - 1], xs[pk], xs[pk + 1]
                    den = ym1 - 2 * y0 + yp1
                    if abs(den) > 1e-9:
                        feat_t = t[pk] + 0.5 * (ym1 - yp1) / den * dt_nom
            elif timing_feature == 'max_slope':
                d = np.gradient(xs, t)
                lead = d[:pk + 1]
                if lead.size >= 2:
                    j = int(np.argmax(lead))
                    feat_t = t[j]
                    if 0 < j < lead.size - 1:
                        ym1, y0, yp1 = lead[j - 1], lead[j], lead[j + 1]
                        den = ym1 - 2 * y0 + yp1
                        if abs(den) > 1e-9:
                            feat_t = t[j] + 0.5 * (ym1 - yp1) / den * dt_nom
            elif timing_feature == 'xcorr' and ref_wave is not None:
                lag = self._xcorr_shift_1d(ref_wave, norm,
                                           max_lag=min(T - 1, int(T // 2)))
                feat_t = t[0] + (ref_idx and 0) + lag * dt_nom  # relative to ref frame0
            else:  # 't50' default: half-rise on the leading edge
                half = 0.5
                lead = norm[:pk + 1]
                cross = np.where(lead >= half)[0]
                if cross.size:
                    j = int(cross[0])
                    if j == 0:
                        feat_t = t[0]
                    else:
                        y0, y1 = norm[j - 1], norm[j]
                        frac = (half - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.0
                        feat_t = t[j - 1] + frac * (t[j] - t[j - 1])
                        # gate 3: a time GAP straddling the crossing invalidates it
                        if (t[j] - t[j - 1]) > 1.8 * dt_nom:
                            reasons[i] = 'gap'; continue
            if feat_t is None or not np.isfinite(feat_t):
                reasons[i] = 'no_feature'; continue
            # gate 4: window-edge pinning (passage not fully captured here)
            if feat_t <= t[0] + edge_frames * dt_nom or \
               feat_t >= t[-1] - edge_frames * dt_nom:
                reasons[i] = 'edge_pinned'; continue
            t_arr[i] = feat_t
            weights[i] = A / noise

        gate_valid = np.isfinite(t_arr)          # rows that passed the per-row gates
        n_gate = int(gate_valid.sum())

        # ---- Part A: iterative, residual-based TREND-OUTLIER rejection ----
        # (isotonic is an L2 projection with zero outlier resistance, so a few
        # late points force PAVA to pool a long run into a plateau-then-jump
        # staircase. We remove trend outliers with a direction-agnostic robust
        # line BEFORE isotonic/Theil-Sen, so they never touch the fit.)
        gate_idx = np.where(gate_valid)[0]
        trend_guard_fired = False; trend_sigma = np.nan; trend_iters = 0
        accept = gate_valid.copy()
        floor_n = max(4, int(0.4 * n_gate))

        if trend_reject and n_gate >= max(5, floor_n):
            # Robust global LINE + residual rejection. sigma is ANCHORED at its
            # initial (all-gate-pass) estimate so iterations cannot spiral: because
            # a genuine speed change makes t_arrival(s) piecewise-linear, the line
            # residuals include that curvature and inflate the initial sigma, giving
            # a generous band that tolerates curvature while still removing GROSS
            # outliers (~1 s off). Refitting on survivors would otherwise shrink
            # sigma and reject real curvature, so we never let it drop below sigma0.
            si0 = s_um[gate_idx]; yi0 = t_arr[gate_idx]
            sl0, ic0, _l0, _h0 = theilslopes(yi0, si0)
            res0 = yi0 - (ic0 + sl0 * si0)
            sigma0 = max(1.4826 * float(np.median(np.abs(res0 - np.median(res0)))),
                         1.0 / self.frame_rate)
            cur = gate_idx.copy()
            for it in range(int(trend_max_iter)):
                trend_iters = it + 1
                si = s_um[cur]; yi = t_arr[cur]
                sl, ic, _l, _h = theilslopes(yi, si)
                resid = yi - (ic + sl * si)
                sigma = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))),
                            sigma0)                        # anchored -> no spiral
                trend_sigma = sigma
                k = float(trend_k)
                while True:
                    keep = np.abs(resid) <= k * sigma
                    if max_resid_s is not None:
                        keep = keep & (np.abs(resid) <= float(max_resid_s))
                    if keep.sum() >= floor_n or k > 8.0:
                        break
                    k += 0.5; trend_guard_fired = True     # relax to hold the floor
                if keep.sum() < floor_n:
                    order = np.argsort(np.abs(resid))
                    keep = np.zeros(cur.size, bool); keep[order[:floor_n]] = True
                    trend_guard_fired = True
                new = cur[keep]
                if new.size == cur.size:
                    break
                cur = new
                if cur.size <= floor_n:
                    break
            accept = np.zeros(n_ax, bool); accept[cur] = True
            for i in np.where(gate_valid & ~accept)[0]:
                reasons[i] = 'trend_outlier'
        valid = accept                          # rows entering the fits
        sv = s_um[valid]; yv = t_arr[valid]; wv = weights[valid]

        # ---- resolve flow direction (on the CLEANED set) ----
        if valid.sum() >= 3:
            iso_inc = self._isotonic_fit(yv, wv, increasing=True)
            iso_dec = self._isotonic_fit(yv, wv, increasing=False)
            r_inc = float(np.sum(wv * (yv - iso_inc) ** 2))
            r_dec = float(np.sum(wv * (yv - iso_dec) ** 2))
            if direction == '+':
                inc = True
            elif direction == '-':
                inc = False
            else:
                inc = (r_inc <= r_dec)
            iso = iso_inc if inc else iso_dec
            direction_resolved = '+' if inc else '-'
            margin = abs(r_inc - r_dec) / (min(r_inc, r_dec) + 1e-9)
        else:
            iso = yv.copy(); direction_resolved = '+'; margin = 0.0

        if valid.sum() >= 1:
            corr = np.abs(yv - iso)
            iso_stats = {'mean': float(np.mean(corr)), 'max': float(np.max(corr))}
        else:
            iso_stats = {'mean': np.nan, 'max': np.nan}
        t_iso_full = np.full(n_ax, np.nan)
        t_iso_full[valid] = iso

        # ---- speed via Theil-Sen of t_arrival ON s (cleaned set only) ----
        speed_mm_s = np.nan; speed_ci = (np.nan, np.nan); r_squared = np.nan
        if valid.sum() >= 3:
            slope, intercept, lo, hi = theilslopes(yv, sv)   # s per µm
            if abs(slope) < 1e-12 or (lo < 0 < hi):
                speed_mm_s = np.nan; speed_ci = (np.nan, np.nan)
                self._log_console("[arrival] near-zero / sign-ambiguous slope -> speed invalid")
            else:
                speed_mm_s = (1.0 / slope) * 1e-3
                def _rec(x):
                    return (1.0 / x) * 1e-3 if abs(x) > 1e-12 else np.nan
                ci = sorted([_rec(hi), _rec(lo)])
                speed_ci = (ci[0], ci[1])
                fit = intercept + slope * sv
                ss_res = np.sum((yv - fit) ** 2); ss_tot = np.sum((yv - np.mean(yv)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # ---- Part B.3: ADAPTIVE-BANDWIDTH local speed with a fallback hierarchy ----
        # For each position we EXPAND the window (kNN in s) until it holds >=min_pts
        # accepted rows AND spans >= min_span_um (a genuine s-baseline), capped at
        # w_max. A local Theil-Sen slope gives v=1/slope. Where even w_max cannot
        # satisfy this, we FALL BACK to the global speed (honest: "no locally
        # resolved variation; the global transit speed applies") rather than to a
        # grey gap. Grey ('none') is reserved for positions outside the fitted s
        # range or when the global fit itself is invalid. A short s-baseline blows
        # up 1/slope variance, so min_span_um stops estimator noise being painted
        # as physiology, and wide-CI positions optionally shrink toward the global.
        vmax_um = (v_max if v_max else 2.0 * (np.median(np.diff(s_um)) * self.frame_rate))
        vmin_um = max(0.0, float(v_min))
        s_len = float(s_um[-1] - s_um[0]) if s_um.size > 1 else n_ax
        min_span = float(min_span_um) if min_span_um else 0.12 * s_len
        w_max = float(w_max_um) if w_max_um else 0.5 * s_len
        prof_pos = s_um.copy()
        prof_speed = np.full(n_ax, np.nan)       # mm/s, signed
        prof_speed_lo = np.full(n_ax, np.nan)
        prof_speed_hi = np.full(n_ax, np.nan)
        prof_clipped = np.zeros(n_ax, bool)
        eff_bw = np.full(n_ax, np.nan)           # effective bandwidth used (µm)
        provenance = np.array(['none'] * n_ax, dtype='U8')
        rel_ci = np.full(n_ax, np.nan)
        n_signrev = 0
        g_speed = speed_mm_s                     # signed global speed (mm/s)
        g_ci = speed_ci
        g_relci = (abs(g_ci[1] - g_ci[0]) / (abs(g_speed) + 1e-9)
                   if np.isfinite(g_speed) and np.isfinite(g_ci[0]) else np.inf)
        s_lo = float(sv.min()) if sv.size else 0.0
        s_hi = float(sv.max()) if sv.size else 0.0

        def _rec(x):
            return (1.0 / x) * 1e-3 if abs(x) > 1e-12 else np.nan

        if sv.size >= max(local_min_pts, 3) and np.isfinite(g_speed):
            svo = np.argsort(sv); ss = sv[svo]; yy_ = yv[svo]
            for j, s0 in enumerate(prof_pos):
                if s0 < s_lo - 1e-6 or s0 > s_hi + 1e-6:
                    continue                     # outside fitted range -> 'none' (grey)
                d = np.abs(ss - s0); order = np.argsort(d)
                sel = None; used_bw = np.nan
                for kk in range(int(local_min_pts), ss.size + 1):
                    idx = order[:kk]; bw = float(d[idx].max())
                    if bw > w_max:
                        break
                    if np.ptp(ss[idx]) >= min_span:
                        sel = idx; used_bw = bw; break
                if sel is None:                  # try full w_max reach
                    idx = order[d[order] <= w_max]
                    if idx.size >= local_min_pts and np.ptp(ss[idx]) >= min_span:
                        sel = idx; used_bw = w_max
                if sel is None:
                    # global fallback: no local baseline -> global transit speed
                    prof_speed[j] = g_speed; provenance[j] = 'global'
                    prof_speed_lo[j], prof_speed_hi[j] = g_ci; continue
                sl, ic, l_lo, l_hi = theilslopes(yy_[sel], ss[sel])
                if abs(sl) < 1e-12 or \
                   (direction_resolved == '+' and sl < 0) or \
                   (direction_resolved == '-' and sl > 0):
                    n_signrev += 1
                    prof_speed[j] = g_speed; provenance[j] = 'global'
                    prof_speed_lo[j], prof_speed_hi[j] = g_ci; continue
                v = (1.0 / sl) * 1e-3
                cc = sorted([_rec(l_hi), _rec(l_lo)])
                rw = abs(cc[1] - cc[0]) / (abs(v) + 1e-9)
                rel_ci[j] = rw; eff_bw[j] = used_bw
                # low-confidence: shrink toward global (precision weighting) or
                # fall back entirely if the local CI is worse than global.
                if rw > float(ci_rel_thresh):
                    if shrink_to_global and np.isfinite(g_speed):
                        alpha = float(np.clip(g_relci / (g_relci + rw + 1e-9), 0.0, 1.0))
                        v = alpha * v + (1 - alpha) * g_speed
                        provenance[j] = 'global'
                    else:
                        prof_speed[j] = g_speed; provenance[j] = 'global'
                        prof_speed_lo[j], prof_speed_hi[j] = g_ci; continue
                else:
                    provenance[j] = 'local'
                vv = abs(v)
                if vv > vmax_um * 1e-3 or vv < vmin_um * 1e-3:
                    prof_clipped[j] = True
                    vv = float(np.clip(vv, vmin_um * 1e-3, vmax_um * 1e-3))
                prof_speed[j] = np.sign(v) * vv
                prof_speed_lo[j], prof_speed_hi[j] = cc[0], cc[1]
        # light masked smoothing of the LOCAL points only (keep global-fallback and
        # gaps intact); provenance stays authoritative for rendering.
        loc_m = provenance == 'local'
        if loc_m.sum() >= 3:
            tmp = np.where(loc_m, prof_speed, np.nan)
            sm = self._masked_smooth1d(tmp, max(1.0, 0.02 * n_ax))
            prof_speed = np.where(loc_m & np.isfinite(sm), sm, prof_speed)
        local_pos = prof_pos[loc_m]
        local_speed = np.abs(prof_speed[loc_m])

        # ---- rejection tally (Part C: plain-str keys) ----
        rej = {}
        for r in reasons[~valid]:
            rej[str(r)] = rej.get(str(r), 0) + 1
        n_valid = int(valid.sum()); n_rej = int((~valid).sum())
        n_trend = int(np.sum(reasons == 'trend_outlier'))
        n_clip = int(prof_clipped.sum())
        n_none = int(np.sum(provenance == 'none'))
        n_loc = int(np.sum(provenance == 'local'))
        n_glob = int(np.sum(provenance == 'global'))
        med_bw = float(np.nanmedian(eff_bw)) if np.isfinite(eff_bw).any() else np.nan
        med_relci = float(np.nanmedian(rel_ci)) if np.isfinite(rel_ci).any() else np.nan
        # Part C: which window END triggered each edge_pinned rejection
        n_edge_start = int(np.sum(reasons == 'edge_pinned_start'))
        n_edge_end = int(np.sum(reasons == 'edge_pinned_end'))
        self._log_console(
            f"[arrival] feature={timing_feature} dir={direction_resolved} "
            f"(iso margin {margin:.2f}); gate-pass={n_gate} valid={n_valid}/{n_ax} "
            f"rejected={n_rej} {rej}; trend-outliers={n_trend} sigma={trend_sigma:.3f}s "
            f"iters={trend_iters} guard={'FIRED' if trend_guard_fired else 'ok'}; "
            f"iso corr mean={iso_stats['mean']:.3f}s max={iso_stats['max']:.3f}s")
        if n_edge_start or n_edge_end:
            self._log_console(
                f"[arrival] edge_pinned by window end: start={n_edge_start} end={n_edge_end}"
                + ("  [extend the analysis time range EARLIER to recover start-pinned rows]"
                   if n_edge_start > 2 * max(1, n_edge_end) else "")
                + ("  [extend the analysis time range LATER to recover end-pinned rows]"
                   if n_edge_end > 2 * max(1, n_edge_start) else ""))
        self._log_console(
            f"[arrival] local speed provenance: local={n_loc} global-fallback={n_glob} "
            f"none(grey)={n_none}; median bandwidth={med_bw:.0f}µm median rel-CI={med_relci:.2f}; "
            f"clipped={n_clip} sign-reversed={n_signrev}")
        self._log_console(
            f"[arrival] speed={speed_mm_s:+.3f} mm/s "
            f"CI[{speed_ci[0]:.3f},{speed_ci[1]:.3f}] R²={r_squared:.2f}")

        return {
            'method_kind': 'arrival_time', 'kymo_raw': kymo, 's_um': s_um, 't_s': t,
            't_arrival_s': t_arr, 't_arrival_isotonic_s': t_iso_full,
            'row_valid_mask': valid, 'gate_valid_mask': gate_valid,
            'row_weights': weights, 'reject_reasons': reasons, 'reject_tally': rej,
            'timing_feature': timing_feature, 'direction_resolved': direction_resolved,
            'direction': ('prox->dist (+)' if direction_resolved == '+' else 'dist->prox (-)'),
            'isotonic_correction_stats': iso_stats, 'iso_margin': margin,
            'norm_rows': norm_rows,
            'trend_k': float(trend_k), 'trend_sigma': trend_sigma,
            'trend_guard_fired': trend_guard_fired, 'trend_iters': trend_iters,
            'n_trend_outliers': n_trend,
            'min_span_um': min_span, 'w_max_um': w_max, 'local_min_pts': int(local_min_pts),
            'speed_mm_s': speed_mm_s, 'speed_ci': speed_ci, 'r_squared': r_squared,
            'local_pos_um': local_pos, 'local_speed_mm_s': local_speed,
            'prof_pos_um': prof_pos, 'prof_speed_mm_s': prof_speed,
            'prof_speed_lo': prof_speed_lo, 'prof_speed_hi': prof_speed_hi,
            'prof_clipped': prof_clipped, 'speed_provenance': provenance,
            'effective_bandwidth_um': eff_bw, 'prof_rel_ci': rel_ci,
            's_fit_range': (s_lo, s_hi),
            'n_local': n_loc, 'n_global': n_glob, 'n_none': n_none,
            'n_clipped': n_clip, 'n_sign_reversed': n_signrev,
            'n_edge_start': n_edge_start, 'n_edge_end': n_edge_end,
            'n_valid': n_valid, 'n_rejected': n_rej,
        }

    @staticmethod
    def _masked_smooth1d(y, sigma):
        """NaN-preserving Gaussian smoothing (masked/normalised convolution):
        smooth the values and the validity mask separately and divide, so gaps
        (NaN) remain NaN instead of being filled with fabricated data."""
        y = np.asarray(y, float)
        m = np.isfinite(y).astype(float)
        v = np.where(np.isfinite(y), y, 0.0)
        vs = ndimage.gaussian_filter1d(v, sigma, mode='nearest')
        ms = ndimage.gaussian_filter1d(m, sigma, mode='nearest')
        out = np.full_like(y, np.nan)
        ok = ms > 1e-3
        out[ok] = vs[ok] / ms[ok]
        out[m < 0.5] = np.nan        # positions with no measurement stay gaps
        return out

    # ======================= WAYPOINT METHOD (sparse, user-guided) =======================
    def set_waypoints_from_points(self, pts):
        """Populate the float waypoint list from clicked points (sub-pixel kept)."""
        self.waypoints = []
        self._wp_next_id = 0
        for (x, y) in pts:
            self.waypoints.append(dict(id=self._wp_next_id, x=float(x), y=float(y),
                                       label=None, manual_t=None, enabled=True))
            self._wp_next_id += 1
        # the waypoint set changed -> refresh the flow-speed window's "show WP"
        # selector immediately (reset to 'all'; indices no longer map to old nodes)
        self._fsm_sync_wp_selector(len(self._wp_enabled()), reset=True)

    def _wp_enabled(self):
        return [w for w in (self.waypoints or []) if w.get('enabled', True)]

    def save_waypoints(self):
        """Save the waypoint set + acquisition/analysis context to JSON."""
        import json
        if not self.waypoints:
            messagebox.showinfo("Waypoints", "No waypoints to save."); return
        stem = os.path.splitext(os.path.basename(self.tif_file_path or 'stack'))[0] \
            if getattr(self, 'tif_file_path', None) else 'stack'
        p = filedialog.asksaveasfilename(title="Save waypoints", initialfile=f"{stem}_waypoints",
                                         defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        H, W = (self.image_stack.shape[1:3] if self.image_stack is not None else (0, 0))
        sf, ef = self.get_analysis_frame_range()
        doc = dict(schema='mouse_ear_waypoints_v1', source=os.path.basename(self.tif_file_path or ''),
                   image_shape=[int(H), int(W)], pixel_size=self.pixel_size, frame_rate=self.frame_rate,
                   frame_range=[int(sf), int(ef)], waypoints=self.waypoints,
                   params=dict(k_profile=getattr(self, 'flow_k_profile', 0.6),
                               timing_feature=getattr(self, 'flow_timing_feature', 't50')))
        with open(p, 'w') as fh:
            json.dump(doc, fh, indent=2)
        self.status_var.set(f"Saved {len(self.waypoints)} waypoints to {os.path.basename(p)}")

    def load_waypoints(self):
        """Load a waypoint set; warn (but allow) if geometry differs."""
        import json
        p = filedialog.askopenfilename(title="Load waypoints", filetypes=[("JSON", "*.json")])
        if not p:
            return
        doc = json.load(open(p))
        if doc.get('schema') != 'mouse_ear_waypoints_v1':
            messagebox.showerror("Waypoints", "Not a waypoint file."); return
        H, W = (self.image_stack.shape[1:3] if self.image_stack is not None else (0, 0))
        ih, iw = doc.get('image_shape', [H, W])
        if (ih, iw) != (H, W) or abs(float(doc.get('pixel_size', self.pixel_size)) - self.pixel_size) > 1e-6:
            if not messagebox.askokcancel(
                    "Waypoints", f"Geometry differs (saved {iw}x{ih} @ {doc.get('pixel_size')} µm/px, "
                    f"current {W}x{H} @ {self.pixel_size}). Load anyway onto this recording?"):
                return
        self.waypoints = [dict(id=int(w.get('id', i)), x=float(w['x']), y=float(w['y']),
                               label=w.get('label'), manual_t=w.get('manual_t'),
                               enabled=bool(w.get('enabled', True)))
                          for i, w in enumerate(doc.get('waypoints', []))]
        self._wp_next_id = (max((w['id'] for w in self.waypoints), default=-1) + 1)
        # keep foreground_roi in sync so existing callers see the loaded geometry
        if self.waypoints:
            self.foreground_roi = np.array([[int(round(w['x'])), int(round(w['y']))]
                                            for w in self.waypoints], dtype=np.int32)
        self.status_var.set(f"Loaded {len(self.waypoints)} waypoints")
        self._fsm_sync_wp_selector(len(self._wp_enabled()), reset=True)
        self.display_current_frame()

    def snap_waypoints_to_vessel(self, radius=4):
        """Nudge each waypoint to the local intensity centroid (mean image) within
        `radius` px, so a slight misclick does not sample background. Undoable."""
        if self.image_stack is None or not self.waypoints:
            return
        img = self._get_mean_image()
        if img is None:
            return
        self._wp_snap_backup = [(w['x'], w['y']) for w in self.waypoints]
        H, W = img.shape[:2]; rr = int(radius)
        for w in self.waypoints:
            x0, y0 = int(round(w['x'])), int(round(w['y']))
            xa, xb = max(0, x0 - rr), min(W, x0 + rr + 1)
            ya, yb = max(0, y0 - rr), min(H, y0 + rr + 1)
            patch = img[ya:yb, xa:xb].astype(float)
            patch = patch - patch.min()
            if patch.sum() <= 0:
                continue
            yy, xx = np.mgrid[ya:yb, xa:xb]
            w['x'] = float((xx * patch).sum() / patch.sum())
            w['y'] = float((yy * patch).sum() / patch.sum())
        self.status_var.set(f"Snapped {len(self.waypoints)} waypoints to local centroid")
        self.display_current_frame()

    def undo_snap_waypoints(self):
        if getattr(self, '_wp_snap_backup', None):
            for w, (x, y) in zip(self.waypoints, self._wp_snap_backup):
                w['x'], w['y'] = x, y
            self._wp_snap_backup = None
            self.display_current_frame()

    def choose_waypoints(self):
        """Pick which waypoints are INCLUDED in the flow-speed computation.

        Opens a scrollable checklist (one row per drawn waypoint); ticking a box
        sets that waypoint's `enabled` flag, which is exactly what the tracker
        reads via `_wp_enabled()`. Apply re-runs the waypoint tracker if the
        flow-speed window is open, and always refreshes the image markers."""
        if not getattr(self, 'waypoints', None):
            messagebox.showinfo("Choose waypoints",
                                "No waypoints yet. Draw a foreground ROI first — each "
                                "clicked point becomes a waypoint.")
            return
        win = tk.Toplevel(self.root)
        win.title("Choose waypoints for flow-speed analysis")
        win.geometry("340x460")

        hint = ttk.Label(win, wraplength=320, foreground="#444",
                         text="Tick the waypoints to include in the flow-speed computation "
                              "(use at least 5–10 for a reliable speed). Green = included, "
                              "grey = excluded on the image.")
        hint.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

        count_var = tk.StringVar(value="")
        ttk.Label(win, textvariable=count_var, font=("TkDefaultFont", 9, "bold")).pack(
            side=tk.TOP, anchor="w", padx=8)

        # scrollable body (there may be many waypoints)
        body = ttk.Frame(win); body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        cvs = tk.Canvas(body, highlightthickness=0)
        sb = ttk.Scrollbar(body, orient="vertical", command=cvs.yview)
        inner = ttk.Frame(cvs)
        inner.bind("<Configure>", lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.create_window((0, 0), window=inner, anchor="nw")
        cvs.configure(yscrollcommand=sb.set)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        # mouse-wheel scrolling
        cvs.bind_all("<MouseWheel>", lambda e: cvs.yview_scroll(int(-e.delta / 120), "units"))

        wp_vars = []
        def _update_count():
            n_on = sum(1 for v in wp_vars if v.get())
            warn = "" if n_on >= 5 else "   ⚠ pick at least 5"
            if n_on < 3:
                warn = "   ⚠ need ≥3 to compute a speed"
            count_var.set(f"{n_on} / {len(wp_vars)} included{warn}")

        for i, w in enumerate(self.waypoints):
            v = tk.BooleanVar(value=bool(w.get('enabled', True)))
            wp_vars.append(v)
            lbl = f"WP{i}   (x={w['x']:.1f}, y={w['y']:.1f})"
            if w.get('manual_t') is not None:
                lbl += f"   [manual t={w['manual_t']:.3f}s]"
            ttk.Checkbutton(inner, text=lbl, variable=v,
                            command=_update_count).pack(side=tk.TOP, anchor="w")
        _update_count()

        def _set_all(state):
            for v in wp_vars:
                v.set(state)
            _update_count()

        btns = ttk.Frame(win); btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        ttk.Button(btns, text="All", width=6,
                   command=lambda: _set_all(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="None", width=6,
                   command=lambda: _set_all(False)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Invert", width=7,
                   command=lambda: ([v.set(not v.get()) for v in wp_vars], _update_count())
                   ).pack(side=tk.LEFT, padx=2)

        def _apply(close=False):
            n_on = sum(1 for v in wp_vars if v.get())
            if n_on < 3:
                messagebox.showwarning("Choose waypoints",
                                       "Select at least 3 waypoints (5–10 recommended) so the "
                                       "tracker can order them and fit a speed.")
                return
            for w, v in zip(self.waypoints, wp_vars):
                w['enabled'] = bool(v.get())
            # keep the flow-speed 'show WP' selector and image markers in sync
            self._fsm_sync_wp_selector(len(self._wp_enabled()), reset=True)
            self.display_current_frame()
            self.status_var.set(f"{n_on} waypoints included in flow-speed analysis")
            # if the flow-speed window is open, re-run tracking with the new subset
            if isinstance(getattr(self, '_fsm', None), dict) and self._fsm.get('win') is not None:
                try:
                    if self._fsm['win'].winfo_exists() and \
                       getattr(self, 'flow_method_kind', '') in ('waypoint', 'both'):
                        if self._fsm_recompute():
                            self._fsm_redraw()
                except Exception as e:
                    self._log_console(f"[waypoint] re-track after selection failed: {e}")
            if close:
                win.destroy()

        act = ttk.Frame(win); act.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 8))
        ttk.Button(act, text="Apply", command=lambda: _apply(False)).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Apply & close", command=lambda: _apply(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(act, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=2)

    def _wp_timecourse(self, x, y, frames, r, stat):
        """Neighbourhood time course at a float waypoint: disk of radius r sampled
        by bilinear interpolation, aggregated by median (robust) or mean."""
        rr = max(1, int(np.ceil(r)))
        gy, gx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        disk = (gx ** 2 + gy ** 2) <= r * r
        ox = gx[disk].astype(float); oy = gy[disk].astype(float)
        px = x + ox; py = y + oy
        tc = np.empty(len(frames))
        for i, fi in enumerate(frames):
            vals = ndimage.map_coordinates(self.image_stack[fi].astype(np.float64),
                                           [py, px], order=1, mode='nearest')
            tc[i] = np.median(vals) if stat == 'median' else np.mean(vals)
        return tc

    def _track_waypoint(self, timing_feature='max_slope', direction='auto',
                        assign_method='dp', r_wp=None, wp_stat='median', snr_thr=3.0,
                        smooth_win=5, baseline_pct=10.0, n_cand=5, v_min=0.0, v_max=None,
                        min_span_um=None, edge_frames=2, trend_reject=True, trend_k=2.5,
                        max_resid_s=None, min_dt_frames=2):
        """Waypoint-based sequential bolus tracking (sparse, user-guided).

        Measures at a few user-chosen waypoints only, detects the RISING-edge
        passage (positive dI/dt) as a LIST of candidates per node, then assigns one
        candidate per node by dynamic programming under a velocity-bounded ordering
        constraint (Viterbi, not greedy — a greedy scan locks in on an early false
        detection). Returns the SAME dict contract as `_track_arrival_time` plus
        waypoint-specific extras, so the existing panels/caption render it.

        Excluded (breathing) frames are treated as missing samples on the non-
        interpolated timebase — never fabricated. `manual_t` on a waypoint pins its
        arrival time; the DP optimises the rest around it."""
        from scipy.stats import theilslopes
        try:
            from scipy.signal import savgol_filter, find_peaks
        except Exception:
            savgol_filter = None; find_peaks = None

        wps = self._wp_enabled()
        if len(wps) < 3:
            self._log_console("[waypoint] need >=3 enabled waypoints"); return None
        start, end = self.get_analysis_frame_range()
        frames = self._included_frames(start, end)      # non-interpolated -> gaps are gaps
        if len(frames) < 5:
            return None
        t = np.asarray(frames, float) / self.frame_rate
        dt_nom = float(np.median(np.diff(t)))
        # arc-length position of each waypoint along the polyline (NOT the chord)
        xy = np.array([[w['x'], w['y']] for w in wps], float)
        seg = np.diff(xy, axis=0)
        s_px = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        s_um = s_px * self.pixel_size
        n = len(wps)
        r_def = max(3.0, float(r_wp) if r_wp else 3.0)
        vmax_um = (v_max if v_max else 2.0 * (self.pixel_size * self.frame_rate))
        vmin_um = max(0.0, float(v_min))
        s_len = float(s_um[-1] - s_um[0]) if s_um.size > 1 else 1.0
        min_span = float(min_span_um) if min_span_um else 0.12 * s_len

        def _smooth(x):
            w = int(smooth_win)
            if savgol_filter is not None and 3 <= w < x.size and w % 2 == 1:
                try:
                    return savgol_filter(x, w, min(3, w - 1))
                except Exception:
                    pass
            return x

        # ---- per-waypoint candidate detection (rising edge) ----
        cand_t = [[] for _ in range(n)]; cand_s = [[] for _ in range(n)]
        wp_state = np.array(['no-detection'] * n, dtype='U16')
        norm_rows = [None] * n; deriv_rows = [None] * n
        for i, w in enumerate(wps):
            if w.get('manual_t') is not None:            # user-pinned -> single fixed candidate
                cand_t[i] = [float(w['manual_t'])]; cand_s[i] = [1e6]; wp_state[i] = 'manual'
                continue
            tc = self._wp_timecourse(w['x'], w['y'], frames, r_def, wp_stat)
            I0 = np.percentile(tc, baseline_pct); dI = tc - I0
            xs = _smooth(dI); A = float(np.nanmax(xs))
            noise = float(np.median(np.abs(tc - np.median(tc))) * 1.4826 + 1e-9)
            if A < snr_thr * noise or A <= 0:
                wp_state[i] = 'low_snr'; continue
            norm_rows[i] = xs / A
            # POSITIVE derivative via Savitzky-Golay (smooth + differentiate in one step)
            if savgol_filter is not None and 5 <= int(smooth_win) < xs.size and int(smooth_win) % 2 == 1 \
               and np.allclose(np.diff(t), dt_nom):
                deriv = savgol_filter(xs, int(smooth_win), min(3, int(smooth_win) - 1),
                                      deriv=1, delta=dt_nom)
            else:
                deriv = np.gradient(xs, t)
            deriv_rows[i] = deriv
            dmed = np.median(deriv); dmad = np.median(np.abs(deriv - dmed)) * 1.4826 + 1e-9
            if find_peaks is not None:
                pk, _pr = find_peaks(deriv, prominence=snr_thr * dmad)
            else:
                pk = np.array([int(np.argmax(deriv))])
            pk = pk[deriv[pk] > 0]                        # RISING edges only
            if pk.size == 0:
                wp_state[i] = 'low_snr'; continue
            for p in pk:
                sc = float((deriv[p] - dmed) / dmad)
                # sub-frame time of the derivative peak (parabolic)
                tp = t[p]
                if 0 < p < deriv.size - 1:
                    ym1, y0, yp1 = deriv[p - 1], deriv[p], deriv[p + 1]
                    den = ym1 - 2 * y0 + yp1
                    if abs(den) > 1e-9:
                        tp = t[p] + 0.5 * (ym1 - yp1) / den * dt_nom
                # feature time depends on timing_feature
                if timing_feature == 't50':
                    lead = (xs[:p + 1] / A); cr = np.where(lead >= 0.5)[0]
                    if cr.size and cr[0] > 0:
                        j = int(cr[0]); y0b, y1b = lead[j - 1], lead[j]
                        fr = (0.5 - y0b) / (y1b - y0b) if y1b != y0b else 0.0
                        ft = t[j - 1] + fr * (t[j] - t[j - 1])
                    else:
                        ft = tp
                elif timing_feature == 'ttp':
                    q = int(np.argmax(xs[p:min(xs.size, p + int(2 * (smooth_win or 5)) + 1)])) + p
                    ft = t[min(q, t.size - 1)]
                else:                                    # 'max_slope' (default) / 'xcorr'
                    ft = tp
                # gate: edge pinning (passage not fully captured at this node)
                if ft <= t[0] + edge_frames * dt_nom or ft >= t[-1] - edge_frames * dt_nom:
                    continue
                cand_t[i].append(float(ft)); cand_s[i].append(sc)
            if not cand_t[i]:
                wp_state[i] = 'edge_pinned'

        # order candidates by score (best first) and cap to n_cand
        for i in range(n):
            if cand_t[i] and wp_state[i] not in ('manual',):
                order = np.argsort(cand_s[i])[::-1][:int(n_cand)]
                cand_t[i] = [cand_t[i][j] for j in order]; cand_s[i] = [cand_s[i][j] for j in order]

        # ---- DP sequential assignment (both directions) ----
        seed_rate = None
        # crude seed: median of pairwise (Δs/Δt) using each node's best candidate
        bt = [ct[0] if ct else np.nan for ct in cand_t]
        havev = [i for i in range(n) if np.isfinite(bt[i])]
        if len(havev) >= 2:
            rr = [(s_um[j] - s_um[i]) / (bt[j] - bt[i]) for i, j in zip(havev[:-1], havev[1:])
                  if bt[j] != bt[i]]
            seed_rate = float(np.median(np.abs(rr))) if rr else vmax_um * 0.5

        def _run_dp(order_sign):
            # orient position so 'forward' = increasing index; flow forward in time
            idxo = list(range(n)) if order_sign > 0 else list(range(n - 1, -1, -1))
            sp = s_um[idxo]; sp = sp - sp[0]
            ct = [cand_t[k] for k in idxo]; cs = [cand_s[k] for k in idxo]
            NEG = -1e18
            best = [np.array(cs[k], float) if ct[k] else np.array([]) for k in range(n)]
            back = [[None] * len(ct[k]) for k in range(n)]
            for k in range(n):
                for m in range(len(ct[k])):
                    base = best[k][m]
                    if base <= NEG / 2:
                        continue
                    for k2 in range(k + 1, n):       # allow skipping (no-detection)
                        ds = sp[k2] - sp[k]
                        if ds < 0:
                            continue
                        for m2 in range(len(ct[k2])):
                            dtt = ct[k2][m2] - ct[k][m]
                            # velocity-bounded ordering: ds/vmax <= dt <= ds/vmin
                            lo = ds / vmax_um if vmax_um > 0 else 0.0
                            hi = ds / vmin_um if vmin_um > 1e-9 else np.inf
                            if dtt < lo - 1e-9 or dtt > hi + 1e-9 or dtt <= 0:
                                continue
                            rate = ds / dtt if dtt > 0 else 0.0
                            pen = 0.2 * abs(rate - (seed_rate or rate)) / (seed_rate or 1.0)
                            sc = base + cs[k2][m2] - pen - 0.3 * (k2 - k - 1)  # gap penalty
                            if sc > best[k2][m2]:
                                best[k2][m2] = sc; back[k2][m2] = (k, m)
            bk = bm = -1; bs = NEG
            for k in range(n):
                for m in range(len(ct[k])):
                    if best[k][m] > bs:
                        bs = best[k][m]; bk = k; bm = m
            sel = [-1] * n; node = (bk, bm) if bk >= 0 else None
            while node is not None:
                k, m = node; sel[k] = m; node = back[k][m]
            # map back to original index order
            sel_orig = [-1] * n
            for kk, k in enumerate(idxo):
                sel_orig[k] = (ct[kk][sel[kk]] if sel[kk] >= 0 else None)
            return sel_orig, float(bs)

        def _run_greedy(order_sign):
            # CAUSAL forward scan: take the highest-score candidate after the
            # previous accepted time. Documented failure mode: a single early false
            # detection locks in and poisons all downstream nodes (why DP is default).
            idxo = list(range(n)) if order_sign > 0 else list(range(n - 1, -1, -1))
            sp = s_um[idxo]; ct = [cand_t[k] for k in idxo]; cs = [cand_s[k] for k in idxo]
            t_prev = -np.inf; s_prev = None; sel = [None] * n; tot = 0.0
            for k in range(n):
                best_m = -1; best_sc = -1e18
                for m in range(len(ct[k])):
                    tt = ct[k][m]
                    if tt <= t_prev:
                        continue
                    if s_prev is not None:
                        ds = sp[k] - s_prev; dtt = tt - t_prev
                        if dtt < (ds / vmax_um if vmax_um > 0 else 0) - 1e-9:
                            continue
                    if cs[k][m] > best_sc:
                        best_sc = cs[k][m]; best_m = m
                if best_m >= 0:
                    sel[k] = ct[k][best_m]; tot += best_sc; t_prev = ct[k][best_m]; s_prev = sp[k]
            out = [None] * n
            for kk, k in enumerate(idxo):
                out[k] = sel[kk]
            return out, tot

        _run = _run_greedy if assign_method == 'greedy' else _run_dp
        if direction == '+':
            assigned, score = _run(+1); dirres = '+'; margin = 0.0
        elif direction == '-':
            assigned, score = _run(-1); dirres = '-'; margin = 0.0
        else:
            ap, sp_ = _run(+1); am, sm_ = _run(-1)
            if sp_ >= sm_:
                assigned, score, dirres = ap, sp_, '+'
            else:
                assigned, score, dirres = am, sm_, '-'
            margin = abs(sp_ - sm_) / (abs(min(sp_, sm_)) + 1e-9)

        t_arr = np.full(n, np.nan)
        for i in range(n):
            if assigned[i] is not None:
                t_arr[i] = assigned[i]
                if wp_state[i] not in ('manual',):
                    wp_state[i] = 'detected'
            elif wp_state[i] == 'no-detection':
                pass

        valid = np.isfinite(t_arr)
        # ---- trend-outlier rejection on the sparse nodes ----
        if trend_reject and valid.sum() >= 5:
            sv = s_um[valid]; yv = t_arr[valid]; idxv = np.where(valid)[0]
            sl, ic, _l, _h = theilslopes(yv, sv); resid = yv - (ic + sl * sv)
            sig = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 1.0 / self.frame_rate)
            keep = np.abs(resid) <= trend_k * sig
            if max_resid_s is not None:
                keep = keep & (np.abs(resid) <= float(max_resid_s))
            if keep.sum() >= max(4, int(0.5 * valid.sum())):
                for j, iv in enumerate(idxv):
                    # a manually-pinned arrival time is fixed and immune to rejection
                    if not keep[j] and wp_state[iv] != 'manual':
                        valid[iv] = False; wp_state[iv] = 'trend_outlier'

        sv = s_um[valid]; yv = t_arr[valid]
        # ---- global speed (Theil-Sen) + isotonic (should be ~no-op after DP) ----
        speed_mm_s = np.nan; speed_ci = (np.nan, np.nan); r_squared = np.nan
        if valid.sum() >= 3:
            slope, intercept, lo, hi = theilslopes(yv, sv)
            if abs(slope) > 1e-12 and not (lo < 0 < hi):
                speed_mm_s = (1.0 / slope) * 1e-3
                def _rec(x):
                    return (1.0 / x) * 1e-3 if abs(x) > 1e-12 else np.nan
                ci = sorted([_rec(hi), _rec(lo)]); speed_ci = (ci[0], ci[1])
                fit = intercept + slope * sv
                ss_res = np.sum((yv - fit) ** 2); ss_tot = np.sum((yv - np.mean(yv)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        iso_full = np.full(n, np.nan)
        if valid.sum() >= 2:
            o = np.argsort(sv)
            inc = self._isotonic_fit(yv[o], increasing=(dirres == '+'))
            iso_sorted = np.empty_like(inc); iso_sorted[o] = inc
            iso_full[valid] = iso_sorted
            iso_corr = float(np.mean(np.abs(yv - iso_sorted)))
        else:
            iso_corr = np.nan

        # ---- segment speeds with short-baseline guard + CI ----
        seg_speeds = []; seg_ci = []
        vi = np.where(valid)[0]
        for a, b in zip(vi[:-1], vi[1:]):
            ds = s_um[b] - s_um[a]; dtt = t_arr[b] - t_arr[a]
            if abs(ds) < min_span or abs(dtt) < min_dt_frames * dt_nom or dtt == 0:
                seg_speeds.append(np.nan); seg_ci.append((np.nan, np.nan)); continue
            v = (ds / dtt) * 1e-3                    # mm/s
            rel = (dt_nom / abs(dtt))                # ~timing uncertainty
            seg_speeds.append(v); seg_ci.append((v * (1 - rel), v * (1 + rel)))

        # ---- profile arrays for the shared painter (same contract) ----
        prof_pos = s_um.copy()
        prof_speed = np.full(n, np.nan); provenance = np.array(['none'] * n, dtype='U8')
        if valid.sum() >= 2 and np.isfinite(speed_mm_s):
            # piecewise-constant segment speed assigned to node positions
            for k, (a, b) in enumerate(zip(vi[:-1], vi[1:])):
                v = seg_speeds[k]
                if np.isfinite(v):
                    prof_speed[a] = abs(v); prof_speed[b] = abs(v)
                    provenance[a] = 'local'; provenance[b] = 'local'
            # fill remaining valid nodes with the global speed
            for iv in vi:
                if not np.isfinite(prof_speed[iv]):
                    prof_speed[iv] = abs(speed_mm_s); provenance[iv] = 'global'
        loc = np.isfinite(prof_speed)
        local_pos = prof_pos[loc]; local_speed = np.abs(prof_speed[loc])

        rej = {}
        for st in wp_state:
            if st not in ('detected', 'manual'):
                rej[str(st)] = rej.get(str(st), 0) + 1
        n_manual = int(np.sum(wp_state == 'manual'))
        self._log_console(
            f"[waypoint] {assign_method} dir={dirres} (DP margin {margin:.2f}); nodes={n} "
            f"detected={int(np.sum(wp_state=='detected'))} manual={n_manual} "
            f"no-detection={int(np.sum(wp_state=='no-detection'))} rej={rej}")
        self._log_console(
            f"[waypoint] speed={speed_mm_s:+.3f} mm/s CI[{speed_ci[0]:.3f},{speed_ci[1]:.3f}] "
            f"R²={r_squared:.2f}; isotonic corr={iso_corr:.3f}s "
            + ("[WARN: large isotonic correction -> ordering/assignment suspect]"
               if np.isfinite(iso_corr) and iso_corr > 0.5 else ""))
        for k, (a, b) in enumerate(zip(vi[:-1], vi[1:])):
            v = seg_speeds[k]
            if not np.isfinite(v):
                self._log_console(f"[waypoint] segment {a}->{b}: short baseline / Δt -> merged/flagged")

        return {
            'method_kind': 'arrival_time', 'kymo_raw': None, 's_um': s_um, 't_s': t,
            't_arrival_s': t_arr, 't_arrival_isotonic_s': iso_full,
            'row_valid_mask': valid, 'gate_valid_mask': (wp_state != 'low_snr'),
            'row_weights': np.ones(n), 'reject_reasons': wp_state, 'reject_tally': rej,
            'timing_feature': timing_feature, 'direction_resolved': dirres,
            'direction': ('prox->dist (+)' if dirres == '+' else 'dist->prox (-)'),
            'isotonic_correction_stats': {'mean': iso_corr, 'max': iso_corr},
            'iso_margin': margin, 'norm_rows': norm_rows,
            'trend_k': float(trend_k), 'trend_sigma': np.nan,
            'trend_guard_fired': False, 'trend_iters': 1,
            'n_trend_outliers': int(np.sum(wp_state == 'trend_outlier')),
            'min_span_um': min_span, 'w_max_um': np.nan, 'local_min_pts': 2,
            'speed_mm_s': speed_mm_s, 'speed_ci': speed_ci, 'r_squared': r_squared,
            'local_pos_um': local_pos, 'local_speed_mm_s': local_speed,
            'prof_pos_um': prof_pos, 'prof_speed_mm_s': prof_speed,
            'prof_speed_lo': np.full(n, np.nan), 'prof_speed_hi': np.full(n, np.nan),
            'prof_clipped': np.zeros(n, bool), 'speed_provenance': provenance,
            'effective_bandwidth_um': np.full(n, np.nan), 'prof_rel_ci': np.full(n, np.nan),
            's_fit_range': (float(s_um.min()), float(s_um.max())),
            'n_local': int(np.sum(provenance == 'local')),
            'n_global': int(np.sum(provenance == 'global')),
            'n_none': int(np.sum(provenance == 'none')),
            'n_clipped': 0, 'n_sign_reversed': 0, 'n_edge_start': 0, 'n_edge_end': 0,
            'n_valid': int(valid.sum()), 'n_rejected': int((~valid).sum()),
            # waypoint-specific extras
            'is_waypoint': True, 'wp_xy': xy, 'wp_scores': cand_s, 'wp_candidates': cand_t,
            'wp_state': wp_state, 'wp_deriv_rows': deriv_rows,
            'segment_speeds': seg_speeds, 'segment_ci': seg_ci, 'dp_margin': margin,
            'n_manual': n_manual,
        }

    # ======================= FLOW SPEED MAP (Panel 6) =======================
    # Architecture: analyze_flow_speed_map() computes the tracking ONCE and caches
    # it in self._fsm; every display control (colormap, dynamic range, background
    # image, publication style) only calls _fsm_redraw(), which re-renders the
    # cached result WITHOUT re-running the tracker. "Re-run tracking" recomputes.
    _FSM_SEQ_CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    _FSM_DIV_CMAPS = ['RdBu_r', 'coolwarm', 'PuOr_r']

    def analyze_flow_speed_map(self):
        """PANEL 6: flow speed from consecutive-frame subtraction + seeded peak tracking.

        Builds the along-vessel kymograph (breathing frames motion-interpolated,
        uniform dt), subtracts consecutive frames to isolate the moving bolus,
        and recovers the front with a proximal/distal-seeded, velocity-bounded DP
        tracker. Opens an interactive window with display controls and a
        publication-mode toggle; all controls redraw from cache without recompute.
        """
        if self.foreground_roi is None:
            messagebox.showwarning("Warning", "Draw the vessel centerline (Foreground ROI) first.")
            return
        if not self._fsm_recompute():
            return
        self._fsm_build_window()

    def _run_peak_method(self):
        """Δframe seeded peak-tracking on the motion-interpolated kymograph."""
        kymo, s_um, t_s, col_flags = self._build_axial_kymograph(
            interpolate_excluded=True, return_flags=True)
        if kymo is None or kymo.shape[1] < 3:
            return None, None
        res = self._track_peak_speed(
            kymo, s_um, t_s, col_flags=col_flags,
            method=getattr(self, 'flow_track_method', 'dp'),
            v_min=getattr(self, 'flow_vmin_um_s', 0.0),
            v_max=getattr(self, 'flow_vmax_um_s', None),
            seed_end=getattr(self, 'flow_seed_end', 'auto'),
            seed_window_um=getattr(self, 'flow_seed_window_um', None),
            max_miss=getattr(self, 'flow_max_miss', 5),
            prominence_mad=getattr(self, 'flow_prominence_mad', 3.0),
            smooth_sigma=getattr(self, 'flow_smooth_sigma', 1.0))
        if res is not None:
            res['method_kind'] = 'dframe_peak'
        return res, kymo

    def _run_arrival_method(self):
        """Arrival-time transit method on the NON-interpolated kymograph (excluded
        frames are handled as missing samples — never fabricated)."""
        kymo, s_um, t_s = self._build_axial_kymograph(interpolate_excluded=False)
        if kymo is None or kymo.shape[1] < 5:
            return None, None
        res = self._track_arrival_time(
            kymo, s_um, t_s,
            timing_feature=getattr(self, 'flow_timing_feature', 't50'),
            direction=getattr(self, 'flow_arrival_dir', 'auto'),
            snr_thr=getattr(self, 'flow_prominence_mad', 3.0),
            smooth_win=getattr(self, 'flow_arrival_smooth', 5),
            baseline_pct=getattr(self, 'flow_arrival_baseline_pct', 10.0),
            v_max=getattr(self, 'flow_vmax_um_s', None),
            v_min=getattr(self, 'flow_vmin_um_s', 0.0),
            trend_reject=getattr(self, 'flow_trend_reject', True),
            trend_k=getattr(self, 'flow_trend_k', 2.5),
            max_resid_s=getattr(self, 'flow_max_resid_s', None),
            local_win_um=getattr(self, 'flow_local_win_um', None),
            min_span_um=getattr(self, 'flow_min_span_um', None),
            w_max_um=getattr(self, 'flow_w_max_um', None),
            local_min_pts=getattr(self, 'flow_local_min_pts', 4),
            shrink_to_global=getattr(self, 'flow_shrink', True),
            edge_frames=getattr(self, 'flow_edge_frames', 2))
        return res, kymo

    def _run_waypoint_method(self):
        """Sparse, user-guided waypoint method (Viterbi ordering over user nodes)."""
        res = self._track_waypoint(
            timing_feature=getattr(self, 'flow_timing_feature', 'max_slope'),
            direction=getattr(self, 'flow_arrival_dir', 'auto'),
            assign_method=getattr(self, 'flow_wp_assign', 'dp'),
            r_wp=getattr(self, 'flow_wp_radius', None),
            wp_stat=getattr(self, 'flow_wp_stat', 'median'),
            snr_thr=getattr(self, 'flow_prominence_mad', 3.0),
            smooth_win=getattr(self, 'flow_arrival_smooth', 5),
            baseline_pct=getattr(self, 'flow_arrival_baseline_pct', 10.0),
            n_cand=getattr(self, 'flow_wp_ncand', 5),
            v_min=getattr(self, 'flow_vmin_um_s', 0.0),
            v_max=getattr(self, 'flow_vmax_um_s', None),
            min_span_um=getattr(self, 'flow_min_span_um', None),
            edge_frames=getattr(self, 'flow_edge_frames', 2),
            trend_reject=getattr(self, 'flow_trend_reject', True),
            trend_k=getattr(self, 'flow_trend_k', 2.5),
            max_resid_s=getattr(self, 'flow_max_resid_s', None))
        return res, None

    # ======================= VESSEL MASK (Part 0) =========================
    # A binary vessel segmentation is the prerequisite for the isochrone map
    # (Part 1), the structural/functional distance maps (Part 2) and the
    # scaling analysis (Part 3). scikit-image is unavailable, so segmentation
    # is built from scipy.ndimage second derivatives (Frangi-style multi-scale
    # Hessian vesselness) + cv2 thresholding/morphology. The illuminated ear
    # aperture is stored as a separate FOV mask so the dark surround can never
    # contaminate the distance statistics.

    def _vessel_source_image(self, source='max'):
        """Structural source image for segmentation, over INCLUDED frames only.
        'max' (default) gives the most complete vessel tree; 'mean' is cleaner
        but drops faint vessels; 'current' uses the displayed frame."""
        if self.image_stack is None:
            return None
        start, end = self.get_analysis_frame_range()
        keep = self._included_frames(start, end)
        if not keep:
            keep = list(range(start, end + 1))
        if source == 'mean':
            img = np.mean(self.image_stack[keep].astype(np.float64), axis=0)
        elif source == 'current':
            ci = int(np.clip(self.current_frame, 0, len(self.image_stack) - 1))
            img = self.image_stack[ci].astype(np.float64)
        else:  # 'max' projection
            img = np.max(self.image_stack[keep].astype(np.float64), axis=0)
        return img

    def _detect_fov_mask(self, img, rel_thr=0.08):
        """Detect the circular illuminated aperture (field of view). Everything
        outside must be excluded from all statistics. Same idea as the flow-map
        `bg_crop` heuristic: threshold the normalised image low, keep the largest
        connected component, and fill holes so vessels inside are not punched out.
        """
        img = np.asarray(img, np.float64)
        lo, hi = np.nanmin(img), np.nanmax(img)
        base = (img - lo) / (hi - lo + 1e-12)
        binm = (base > float(rel_thr)).astype(np.uint8)
        # keep the single largest bright blob = the aperture
        n_lab, lab, stats, _c = cv2.connectedComponentsWithStats(binm, connectivity=8)
        if n_lab <= 1:
            return np.ones(img.shape, bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = 1 + int(np.argmax(areas))
        fov = (lab == biggest)
        fov = ndimage.binary_fill_holes(fov)
        return fov.astype(bool)

    def _frangi_vesselness(self, img, scales, beta=0.5, c_frac=0.5, bright=True):
        """Multi-scale 2D Frangi vesselness for bright tubular structures.

        At each scale sigma we form the (gamma=2 normalised) Hessian from
        Gaussian second derivatives, take its eigenvalues (|l1|<=|l2|), and score
        tubularity with V = exp(-Rb^2/2beta^2) * (1 - exp(-S^2/2c^2)), keeping only
        pixels whose l2 has the sign of a bright ridge. The response is the max
        over scales. Pure numpy/scipy — no scikit-image.
        """
        img = np.asarray(img, np.float64)
        # normalise so the structureness constant c is scale-comparable
        rng = np.nanmax(img) - np.nanmin(img)
        if rng > 0:
            img = (img - np.nanmin(img)) / rng
        vess = np.zeros(img.shape, np.float64)
        for sigma in scales:
            s = float(sigma)
            # scipy order is per-axis on [y, x]: Hyy=order(2,0), Hxx=order(0,2)
            Hyy = ndimage.gaussian_filter(img, s, order=[2, 0])
            Hxx = ndimage.gaussian_filter(img, s, order=[0, 2])
            Hxy = ndimage.gaussian_filter(img, s, order=[1, 1])
            g2 = s * s                                   # gamma=2 normalisation
            Hxx *= g2; Hyy *= g2; Hxy *= g2
            # eigenvalues of the symmetric 2x2 Hessian
            tmp = np.sqrt(((Hxx - Hyy) * 0.5) ** 2 + Hxy ** 2)
            mu = (Hxx + Hyy) * 0.5
            lam_a = mu + tmp; lam_b = mu - tmp
            # order by magnitude: |l1| <= |l2|
            swap = np.abs(lam_a) > np.abs(lam_b)
            l1 = np.where(swap, lam_b, lam_a)
            l2 = np.where(swap, lam_a, lam_b)
            Rb = np.abs(l1) / (np.abs(l2) + 1e-12)       # blobness
            S = np.sqrt(l1 ** 2 + l2 ** 2)               # structureness
            c = c_frac * float(np.nanmax(S) + 1e-12)
            V = np.exp(-(Rb ** 2) / (2 * beta ** 2)) * \
                (1.0 - np.exp(-(S ** 2) / (2 * c ** 2)))
            # bright vessel -> l2 < 0; dark vessel -> l2 > 0
            V = np.where((l2 < 0) if bright else (l2 > 0), V, 0.0)
            vess = np.maximum(vess, V)
        vess = np.nan_to_num(vess, nan=0.0)
        mx = float(vess.max())
        return vess / mx if mx > 0 else vess

    def build_vessel_mask(self, source='max', sigma_min=1.0, sigma_max=6.0,
                          n_scales=5, threshold=None, min_area_px=25,
                          close_radius=1, open_radius=1, fov_rel_thr=0.08):
        """Compute (and cache) the vessel mask from the structural source image.

        threshold is on the [0,1] vesselness map; None -> Otsu. Returns a dict of
        stats. This is ANALYSIS work — call it explicitly, not on redraw."""
        img = self._vessel_source_image(source)
        if img is None:
            return None
        fov = self._detect_fov_mask(img, rel_thr=fov_rel_thr)
        scales = np.linspace(float(sigma_min), float(sigma_max), max(1, int(n_scales)))
        vness = self._frangi_vesselness(img, scales)
        vness = np.where(fov, vness, 0.0)             # never respond outside the aperture
        self.vesselness = vness.astype(np.float32)
        # threshold: Otsu on the 8-bit vesselness, or a manual [0,1] level
        if threshold is None:
            u8 = np.clip(vness * 255.0, 0, 255).astype(np.uint8)
            thr_u8, _bw = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = float(thr_u8) / 255.0
        else:
            thr = float(threshold)
        mask = (vness >= thr) & fov
        mask = self._cleanup_mask(mask, min_area_px, close_radius, open_radius)
        self.vessel_mask = mask.astype(bool)
        self.fov_mask = fov.astype(bool)
        # stats + provenance
        fov_area = int(fov.sum())
        n_lab, _lab, stats, _c = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        n_comp = max(0, n_lab - 1)
        area_frac = float(mask.sum()) / (fov_area + 1e-9)
        meta = dict(source=source, sigma_min=float(sigma_min), sigma_max=float(sigma_max),
                    n_scales=int(n_scales), threshold=thr, min_area_px=int(min_area_px),
                    close_radius=int(close_radius), open_radius=int(open_radius),
                    fov_rel_thr=float(fov_rel_thr),
                    source_file=os.path.basename(getattr(self, 'tif_file_path', '') or ''),
                    image_shape=list(img.shape), pixel_size=float(self.pixel_size),
                    vessel_px=int(mask.sum()), fov_px=fov_area,
                    area_fraction=area_frac, n_components=n_comp)
        self._vessel_meta = meta
        self._log_console(
            f"[vessel-mask] source={source} sigma[{sigma_min:g}..{sigma_max:g}]x{n_scales} "
            f"thr={thr:.3f} -> {int(mask.sum())} px ({area_frac*100:.1f}% of FOV), "
            f"{n_comp} components; FOV={fov_area} px")
        return meta

    def _cleanup_mask(self, mask, min_area_px, close_radius, open_radius):
        """Morphological open/close + small-component removal (cv2)."""
        m = mask.astype(np.uint8)
        if int(open_radius) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * int(open_radius) + 1,) * 2)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if int(close_radius) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * int(close_radius) + 1,) * 2)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        if int(min_area_px) > 0:
            n_lab, lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = np.zeros(m.shape, np.uint8)
            for lb in range(1, n_lab):
                if stats[lb, cv2.CC_STAT_AREA] >= int(min_area_px):
                    keep[lab == lb] = 1
            m = keep
        return m.astype(bool)

    def _vessel_mask_hash(self):
        """Stable hash of the current mask + FOV, for cache keys (Parts 1-2)."""
        if self.vessel_mask is None:
            return None
        import hashlib
        h = hashlib.md5()
        h.update(np.ascontiguousarray(self.vessel_mask).tobytes())
        if self.fov_mask is not None:
            h.update(np.ascontiguousarray(self.fov_mask).tobytes())
        return h.hexdigest()

    def save_vessel_mask(self):
        """Persist the vessel + FOV masks (.npz) and the parameters (.json)."""
        if self.vessel_mask is None:
            messagebox.showinfo("Vessel mask", "No mask to save. Build one first.")
            return
        stem = os.path.splitext(os.path.basename(self.tif_file_path or 'stack'))[0] \
            if getattr(self, 'tif_file_path', None) else 'stack'
        p = filedialog.asksaveasfilename(
            title="Save vessel mask", initialfile=f"{stem}_vesselmask",
            defaultextension=".npz", filetypes=[("NumPy npz", "*.npz")])
        if not p:
            return
        import json
        np.savez_compressed(p, vessel_mask=self.vessel_mask, fov_mask=self.fov_mask,
                            vesselness=(self.vesselness if self.vesselness is not None
                                        else np.zeros(self.vessel_mask.shape, np.float32)))
        meta = dict(self._vessel_meta or {}); meta['schema'] = 'mouse_ear_vesselmask_v1'
        with open(os.path.splitext(p)[0] + '.json', 'w') as fh:
            json.dump(meta, fh, indent=2)
        self.status_var.set(f"Saved vessel mask to {os.path.basename(p)} (+ .json)")

    def load_vessel_mask(self):
        """Load a vessel + FOV mask saved by save_vessel_mask()."""
        p = filedialog.askopenfilename(title="Load vessel mask",
                                       filetypes=[("NumPy npz", "*.npz")])
        if not p:
            return
        import json
        d = np.load(p)
        self.vessel_mask = d['vessel_mask'].astype(bool)
        self.fov_mask = d['fov_mask'].astype(bool) if 'fov_mask' in d else None
        self.vesselness = (d['vesselness'].astype(np.float32)
                           if 'vesselness' in d else None)
        jp = os.path.splitext(p)[0] + '.json'
        if os.path.exists(jp):
            self._vessel_meta = json.load(open(jp))
        H, W = (self.image_stack.shape[1:3] if self.image_stack is not None else (0, 0))
        if self.vessel_mask.shape != (H, W) and self.image_stack is not None:
            messagebox.showwarning(
                "Vessel mask", f"Mask shape {self.vessel_mask.shape} != image "
                f"{(H, W)}. Loaded anyway; geometry may not align.")
        self._arrival_pixmap = None      # invalidate any cached per-pixel map
        self.status_var.set(f"Loaded vessel mask ({int(self.vessel_mask.sum())} px)")

    def _rethreshold_vessel_mask(self, thr, min_area_px=25, close_radius=1, open_radius=1):
        """Re-threshold the CACHED vesselness map without re-running Frangi — this
        is what the threshold slider uses, so it re-renders instantly."""
        if self.vesselness is None:
            return None
        fov = self.fov_mask if self.fov_mask is not None else np.ones(self.vesselness.shape, bool)
        mask = (self.vesselness >= float(thr)) & fov
        mask = self._cleanup_mask(mask, min_area_px, close_radius, open_radius)
        self.vessel_mask = mask.astype(bool)
        if isinstance(self._vessel_meta, dict):
            self._vessel_meta['threshold'] = float(thr)
            self._vessel_meta['vessel_px'] = int(mask.sum())
            self._vessel_meta['area_fraction'] = float(mask.sum()) / (fov.sum() + 1e-9)
        self._arrival_pixmap = None
        return mask

    # ---------------------- Part 0 mask editor window ----------------------
    def open_vessel_mask_editor(self):
        """Interactive vessel-mask builder/editor: live overlay, threshold slider
        (re-thresholds the cached vesselness instantly), paint/erase brush with
        undo, source/scale controls (explicit Recompute = analysis), save/load."""
        if self.image_stack is None:
            messagebox.showinfo("Vessel mask", "Load an image stack first.")
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        win = tk.Toplevel(self.root); win.title("Vessel mask editor")
        self._vm = dict(win=win, brush=6, mode='paint', undo=[], painting=False)

        ctl = ttk.Frame(win); ctl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        # row 1: source + scales + recompute
        r1 = ttk.Frame(ctl); r1.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(r1, text="source:").pack(side=tk.LEFT)
        self._vm_src = tk.StringVar(value='max')
        ttk.Combobox(r1, textvariable=self._vm_src, width=8, state='readonly',
                     values=("max", "mean", "current")).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r1, text="σ min/max:").pack(side=tk.LEFT)
        self._vm_smin = tk.StringVar(value='1.0'); self._vm_smax = tk.StringVar(value='6.0')
        ttk.Entry(r1, textvariable=self._vm_smin, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Entry(r1, textvariable=self._vm_smax, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Label(r1, text="n:").pack(side=tk.LEFT)
        self._vm_nsc = tk.StringVar(value='5')
        ttk.Entry(r1, textvariable=self._vm_nsc, width=3).pack(side=tk.LEFT, padx=(1, 8))
        ttk.Button(r1, text="Recompute (Frangi)", command=self._vm_recompute).pack(side=tk.LEFT, padx=2)

        # row 2: threshold + cleanup + opacity
        r2 = ttk.Frame(ctl); r2.pack(side=tk.TOP, fill=tk.X, pady=(3, 0))
        self._vm_otsu = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="Otsu", variable=self._vm_otsu,
                        command=self._vm_apply_threshold).pack(side=tk.LEFT)
        ttk.Label(r2, text="thr:").pack(side=tk.LEFT)
        self._vm_thr = tk.DoubleVar(value=0.30)
        ttk.Scale(r2, from_=0.0, to=1.0, variable=self._vm_thr, length=140,
                  command=lambda e: self._vm_apply_threshold()).pack(side=tk.LEFT, padx=2)
        ttk.Label(r2, text="min area:").pack(side=tk.LEFT)
        self._vm_minarea = tk.StringVar(value='25')
        ttk.Entry(r2, textvariable=self._vm_minarea, width=4).pack(side=tk.LEFT, padx=(1, 8))
        ttk.Label(r2, text="opacity:").pack(side=tk.LEFT)
        self._vm_opacity = tk.DoubleVar(value=0.45)
        ttk.Scale(r2, from_=0.0, to=1.0, variable=self._vm_opacity, length=90,
                  command=lambda e: self._vm_redraw()).pack(side=tk.LEFT, padx=2)

        # row 3: brush + persistence
        r3 = ttk.Frame(ctl); r3.pack(side=tk.TOP, fill=tk.X, pady=(3, 0))
        ttk.Label(r3, text="brush r:").pack(side=tk.LEFT)
        self._vm_brush = tk.StringVar(value='6')
        ttk.Entry(r3, textvariable=self._vm_brush, width=4).pack(side=tk.LEFT, padx=(1, 6))
        self._vm_mode = tk.StringVar(value='paint')
        ttk.Radiobutton(r3, text="paint", variable=self._vm_mode, value='paint').pack(side=tk.LEFT)
        ttk.Radiobutton(r3, text="erase", variable=self._vm_mode, value='erase').pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(r3, text="Undo", command=self._vm_undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(r3, text="Save…", command=self.save_vessel_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(r3, text="Load…", command=lambda: (self.load_vessel_mask(), self._vm_redraw())).pack(side=tk.LEFT, padx=2)
        self._vm_stat = tk.StringVar(value='')
        ttk.Label(r3, textvariable=self._vm_stat, foreground='#036').pack(side=tk.LEFT, padx=8)

        fig = Figure(figsize=(7.5, 7.0)); ax = fig.add_subplot(111)
        ax.set_xticks([]); ax.set_yticks([])
        self._vm['fig'] = fig; self._vm['ax'] = ax
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._vm['canvas'] = canvas
        canvas.mpl_connect('button_press_event', self._vm_on_press)
        canvas.mpl_connect('motion_notify_event', self._vm_on_motion)
        canvas.mpl_connect('button_release_event', self._vm_on_release)

        if self.vessel_mask is None:
            self._vm_recompute()
        else:
            self._vm_redraw()

    def _vm_bg(self):
        """Grayscale anatomical background for the editor (source image, [0,1])."""
        img = self._vessel_source_image(self._vm_src.get() if hasattr(self, '_vm_src') else 'max')
        lo, hi = np.percentile(img, [1, 99.5])
        return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)

    def _vm_recompute(self):
        """Full Frangi rebuild from the current controls (ANALYSIS)."""
        try:
            thr = None if self._vm_otsu.get() else float(self._vm_thr.get())
            meta = self.build_vessel_mask(
                source=self._vm_src.get(),
                sigma_min=float(self._vm_smin.get()), sigma_max=float(self._vm_smax.get()),
                n_scales=int(self._vm_nsc.get()), threshold=thr,
                min_area_px=int(self._vm_minarea.get()))
            if meta and self._vm_otsu.get():
                self._vm_thr.set(round(meta['threshold'], 3))
        except Exception as e:
            messagebox.showerror("Vessel mask", f"Build failed: {e}"); return
        self._vm_redraw()

    def _vm_apply_threshold(self):
        """Threshold slider / Otsu toggle -> re-threshold CACHED vesselness (fast)."""
        if self.vesselness is None:
            return
        if self._vm_otsu.get():
            u8 = np.clip(self.vesselness * 255.0, 0, 255).astype(np.uint8)
            thr_u8, _bw = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = float(thr_u8) / 255.0; self._vm_thr.set(round(thr, 3))
        else:
            thr = float(self._vm_thr.get())
        self._rethreshold_vessel_mask(thr, min_area_px=int(self._vm_minarea.get()))
        self._vm_redraw()

    def _vm_redraw(self):
        ax = self._vm['ax']; ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(self._vm_bg(), cmap='gray', vmin=0, vmax=1)
        if self.vessel_mask is not None:
            op = float(self._vm_opacity.get())
            over = np.zeros((*self.vessel_mask.shape, 4), np.float32)
            over[self.vessel_mask] = (1.0, 0.15, 0.15, op)      # red vessels
            ax.imshow(over)
            if self.fov_mask is not None:
                ax.contour(self.fov_mask.astype(float), levels=[0.5],
                           colors=['#00BFFF'], linewidths=0.8, alpha=0.7)
        m = self._vessel_meta or {}
        self._vm_stat.set(f"{m.get('vessel_px', 0)} px  "
                          f"{m.get('area_fraction', 0)*100:.1f}% FOV  "
                          f"{m.get('n_components', 0)} comp")
        self._vm['canvas'].draw_idle()

    # ---- brush painting ----
    def _vm_stamp(self, event):
        if event.xdata is None or event.ydata is None or self.vessel_mask is None:
            return
        cx, cy = int(round(event.xdata)), int(round(event.ydata))
        r = max(1, int(float(self._vm_brush.get() or 6)))
        H, W = self.vessel_mask.shape
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        if self._vm_mode.get() == 'erase':
            self.vessel_mask[y0:y1, x0:x1][disk] = False
        else:                                        # paint inside the FOV only
            fov = self.fov_mask if self.fov_mask is not None else np.ones((H, W), bool)
            self.vessel_mask[y0:y1, x0:x1][disk & fov[y0:y1, x0:x1]] = True

    def _vm_on_press(self, event):
        if event.inaxes is not self._vm['ax'] or self.vessel_mask is None:
            return
        self._vm['undo'].append(self.vessel_mask.copy())      # snapshot for undo
        if len(self._vm['undo']) > 20:
            self._vm['undo'].pop(0)
        self._vm['painting'] = True
        self._vm_stamp(event); self._vm_redraw()

    def _vm_on_motion(self, event):
        if self._vm.get('painting') and event.inaxes is self._vm['ax']:
            self._vm_stamp(event); self._vm_redraw()

    def _vm_on_release(self, event):
        self._vm['painting'] = False
        # refresh stats/hash after an edit stroke
        if self.vessel_mask is not None and isinstance(self._vessel_meta, dict):
            self._vessel_meta['vessel_px'] = int(self.vessel_mask.sum())
            fovsum = (self.fov_mask.sum() if self.fov_mask is not None
                      else self.vessel_mask.size)
            self._vessel_meta['area_fraction'] = float(self.vessel_mask.sum()) / (fovsum + 1e-9)
            self._vessel_meta['edited'] = True
        self._arrival_pixmap = None
        self._vm_redraw()

    def _vm_undo(self):
        if self._vm.get('undo'):
            self.vessel_mask = self._vm['undo'].pop()
            self._arrival_pixmap = None
            self._vm_redraw()

    def _fsm_recompute(self):
        """Run the selected flow-speed estimator(s) and cache everything to redraw.
        Method kind: 'dframe_peak' / 'arrival_time' / 'waypoint' / 'both'."""
        self.update_ear_params()
        kind = getattr(self, 'flow_method_kind', 'dframe_peak')
        res_peak = res_arr = res_wp = None; kymo = None
        if kind == 'waypoint':
            res_wp, _ = self._run_waypoint_method(); res_arr = res_wp
        if kind in ('dframe_peak', 'both'):
            res_peak, kymo = self._run_peak_method()
        if kind in ('arrival_time', 'both'):
            res_arr, ka = self._run_arrival_method()
            if kymo is None:
                kymo = ka
        # 'both' can also pair the WAYPOINT method against the arrival method
        if kind == 'both' and res_arr is not None and self._wp_enabled() and len(self._wp_enabled()) >= 3:
            res_wp, _ = self._run_waypoint_method()
            if res_wp and np.isfinite(res_wp.get('speed_mm_s', np.nan)):
                va, vw = res_arr.get('speed_mm_s'), res_wp.get('speed_mm_s')
                self._log_console(
                    f"[flow-compare] arrival {va:+.3f} vs waypoint {vw:+.3f} mm/s"
                    + ("  [agree]" if (np.isfinite(va) and np.isfinite(vw)
                       and abs(va - vw) <= 0.25 * max(abs(va), abs(vw), 1e-6)) else "  [WARN: disagree]"))
        # cross-validation log (§9)
        if kind == 'both' and res_peak and res_arr:
            vp, va = res_peak.get('speed_mm_s'), res_arr.get('speed_mm_s')
            cp, ca = res_peak.get('speed_ci'), res_arr.get('speed_ci')
            both_ok = np.isfinite(vp) and np.isfinite(va)
            disagree = both_ok and abs(vp - va) > 0.25 * max(abs(vp), abs(va), 1e-6)
            if not both_ok:
                tail = ("  [one method returned no finite speed — likely v_max under-"
                        "sampling (peak) or too few valid rows (arrival)]")
            elif disagree:
                tail = ("  [WARN: methods DISAGREE >25% — a recording assumption is likely "
                        "violated; inspect the kymograph]")
            else:
                tail = "  [agree — independent methods concur, speed is trustworthy]"
            self._log_console(
                f"[flow-compare] Δframe-peak {vp:+.3f} mm/s CI[{cp[0]:.3f},{cp[1]:.3f}] "
                f"vs arrival-time {va:+.3f} mm/s CI[{ca[0]:.3f},{ca[1]:.3f}]" + tail)
        # choose which result drives the figure. NOTE: the waypoint method stores
        # its result in res_arr (res_arr = res_wp above), so 'waypoint' must select
        # res_arr too — otherwise it falls through to res_peak (None) and every run
        # reports "Estimation failed" even when the tracker succeeded.
        if kind in ('arrival_time', 'waypoint'):
            res = res_arr
        elif kind == 'both':
            res = res_arr if res_arr is not None else res_peak
        else:
            res = res_peak
        if res is None:
            method_hint = ("Need at least 3 enabled waypoints and 5 unexcluded frames "
                           "in the range — use 'Choose WPs…' to include more, or widen "
                           "the frame range." if kind == 'waypoint' else
                           "Try a longer frame range, a cleaner ROI, raise v_max, adjust "
                           "the timing feature, or exclude more breathing frames.")
            messagebox.showwarning(
                "Flow speed map",
                "Estimation failed (too few valid rows/frames).\n" + method_hint)
            return False
        if self.vessel_results is None:
            self.vessel_results = {}
        if res_peak is not None:
            self.vessel_results['peak_speed'] = res_peak
        if res_arr is not None:
            self.vessel_results['arrival_time'] = res_arr
        s_um = res['s_um']
        # dense centreline resample for smooth painting (waypoint s_um is sparse)
        n_center = max(len(s_um), 200)
        centers, _n, s_c = self._resample_centerline(self.foreground_roi, n_center)
        start_f, end_f = self.get_analysis_frame_range()
        stem = os.path.splitext(os.path.basename(self.tif_file_path or 'stack'))[0] \
            if getattr(self, 'tif_file_path', None) else 'stack'
        fsm = getattr(self, '_fsm', {}) or {}
        fsm.update(dict(res=res, res_peak=res_peak, res_arr=res_arr, kymo=kymo,
                        s_um=s_um, centers=centers, s_c=s_c,
                        frame_range=(start_f, end_f), stem=stem))
        self._fsm = fsm
        return True

    # ---- background image for the spatial speed-map panel ----
    def _fsm_background_image(self):
        """Build the grayscale anatomical background (source / gamma / dynamic
        range / invert / auto-crop) as a DISPLAY transform only — the underlying
        data used for quantification is never modified. Returns (img01, bbox,
        label) where img01 is in [0,1] and bbox=(x0,x1,y0,y1) or None."""
        src = self._fsm.get('bg_source', 'center')
        start_f, end_f = self._fsm['frame_range']
        keep = self._included_frames(start_f, end_f)
        if src == 'mean':
            img = self._get_mean_image(); label = f"mean of frames {start_f}-{end_f}"
        elif src == 'max':
            img = np.max(self.image_stack[keep].astype(np.float64), axis=0)
            label = f"max-proj of frames {start_f}-{end_f}"
        elif src == 'current':
            ci = int(self.current_frame); img = self.image_stack[ci].astype(np.float64)
            label = f"frame {ci}"
        else:  # 'center' (default): temporal centre of the marker range
            ci = int(round((start_f + end_f) / 2.0))
            ci = int(np.clip(ci, 0, len(self.image_stack) - 1))
            img = self.image_stack[ci].astype(np.float64)
            label = f"centre frame {ci}"
        img = np.asarray(img, np.float64)
        # dynamic range: percentile or absolute
        if self._fsm.get('bg_mode', 'percentile') == 'percentile':
            lo = np.percentile(img, float(self._fsm.get('bg_lo', 1.0)))
            hi = np.percentile(img, float(self._fsm.get('bg_hi', 99.5)))
        else:
            lo = float(self._fsm.get('bg_lo', float(img.min())))
            hi = float(self._fsm.get('bg_hi', float(img.max())))
        if hi <= lo:
            hi = lo + 1e-6
        img01 = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        gamma = float(self._fsm.get('bg_gamma', 1.0))
        if gamma > 0 and abs(gamma - 1.0) > 1e-3:
            img01 = img01 ** (1.0 / gamma)      # >1 brightens mid-tones
        if self._fsm.get('bg_invert', False):
            img01 = 1.0 - img01
        bbox = None
        if self._fsm.get('bg_crop', True):
            base = (1.0 - img01) if self._fsm.get('bg_invert', False) else img01
            m = base > 0.08 * float(np.nanmax(base) or 1.0)
            ys, xs = np.where(m)
            if xs.size > 20:
                mx = int(0.04 * img01.shape[1]); my = int(0.04 * img01.shape[0])
                bbox = (max(0, xs.min() - mx), min(img01.shape[1] - 1, xs.max() + mx),
                        max(0, ys.min() - my), min(img01.shape[0] - 1, ys.max() + my))
        return img01, bbox, label

    # ---- speed colour normalisation from the display controls ----
    def _fsm_speed_norm(self):
        """Backward-compatible wrapper: speed-mode normalisation (used by the
        Δframe path and existing callers)."""
        cmap, lo, hi, _lab = self._fsm_paint_norm('speed')
        return cmap, lo, hi

    def _fsm_paint_norm(self, mode):
        """Paint-mode-aware colour normaliser. Returns (cmap, lo, hi, cbar_label).
        Both 'speed' and 'arrival' honour cmap, reverse and auto-range percentiles;
        manual vmin/vmax are stored PER MODE (mm/s vs ms) so switching modes never
        carries numbers across scales. symmetric@0 is ignored for 'arrival' (a
        one-sided elapsed quantity)."""
        res = self._fsm['res']
        if mode == 'arrival':
            ta = np.asarray(res.get('t_arrival_isotonic_s', np.array([])), float)
            vm = res.get('row_valid_mask')
            if vm is not None and vm.any():
                base = np.nanmin(ta[vm]); vals = (ta[vm] - base) * 1e3    # ms
            else:
                vals = np.array([])
            label = 'Arrival time (ms)'
            vmin_key, vmax_key = 'vmin_arrival', 'vmax_arrival'
            allow_sym = False
        else:
            v = res.get('prof_speed_mm_s', np.array([]))
            v = v if np.asarray(v).size else res.get('local_speed_mm_s', np.array([]))
            vals = np.abs(np.asarray(v, float)); label = 'Local speed (mm/s)'
            vmin_key, vmax_key = 'vmin_speed', 'vmax_speed'
            allow_sym = True
        vals = vals[np.isfinite(vals)]
        cmap = self._fsm.get('cmap', 'viridis')
        if cmap == 'turbo (discouraged)':
            cmap = 'turbo'
        if self._fsm.get('cmap_reverse', False):
            cmap = cmap[:-2] if cmap.endswith('_r') else cmap + '_r'
        if self._fsm.get('auto_range', True) and vals.size:
            lo = float(np.percentile(vals, float(self._fsm.get('pmin', 5))))
            hi = float(np.percentile(vals, float(self._fsm.get('pmax', 95))))
        else:
            lo = float(self._fsm.get(vmin_key, self._fsm.get('vmin', 0.0)))
            hi = float(self._fsm.get(vmax_key, self._fsm.get('vmax', 1.0)))
        if allow_sym and self._fsm.get('symmetric', False):
            a = max(abs(lo), abs(hi)); lo, hi = -a, a
        if hi <= lo:
            hi = lo + 1e-6
        return cmap, float(lo), float(hi), label

    def _fsm_caption(self):
        """Auto-generate a suggested Nature-style figure caption with methods +
        quantitative results, for copy-paste into a manuscript."""
        r = self._fsm['res']; start_f, end_f = self._fsm['frame_range']
        ci = r['speed_ci']
        ci_s = (f" (95% CI {ci[0]:.3f} to {ci[1]:.3f})"
                if np.isfinite(ci[0]) and np.isfinite(ci[1]) else "")
        px = self.pixel_size; fps = self.frame_rate
        if r.get('method_kind') == 'arrival_time':
            return (
                f"Flow speed in a mouse-ear vessel by arrival-time (indicator-dilution) "
                f"mapping. The vessel centreline (foreground ROI) was resampled and sampled "
                f"per frame (pixel size {px:.1f} um, {fps:.0f} fps, frames {start_f}-{end_f}); "
                f"each position's time course yielded one bolus arrival time via the "
                f"'{r['timing_feature']}' feature. Trend outliers >{r.get('trend_k',2.5):g}sigma "
                f"from a robust line were removed ({r.get('n_trend_outliers',0)} rows, "
                f"final sigma {r.get('trend_sigma',float('nan')):.2f} s); arrival times were "
                f"constrained monotone by isotonic (PAVA) regression and speed obtained as the "
                f"reciprocal Theil-Sen slope of t_arrival vs position: "
                f"{r['speed_mm_s']:+.3f} mm/s{ci_s}, R2 {r['r_squared']:.2f} "
                f"(direction {r['direction_resolved']}). Local speed was estimated by a "
                f"adaptive-bandwidth robust fit (min_pts {r.get('local_min_pts',4)}, min_span "
                f"{r.get('min_span_um',0):.0f} um, w_max {r.get('w_max_um',0):.0f} um) to the "
                f"cleaned raw arrival times, with a global-speed fallback where no local baseline "
                f"exists (paint sub-mode {self._fsm.get('arrival_submode','local')}). Provenance: "
                f"{r.get('n_local',0)} local, {r.get('n_global',0)} global-fallback, "
                f"{r.get('n_none',0)} unmeasured (grey); {r.get('n_clipped',0)} segments speed-"
                f"clipped. Panels: (a) kymograph with arrival times and isotonic "
                f"fit; (b) t_arrival vs position with fit and acceptance band; (c) time courses "
                f"showing progressive delay; (d) vessel painted by local speed. Assumes a single "
                f"dominant passage per position; recirculation/multi-bolus rows are rejected.")
        return (
            f"Flow speed in a mouse-ear vessel by consecutive-frame differencing and "
            f"seeded peak tracking. The vessel centreline (foreground ROI) was resampled "
            f"and sampled per frame to form a position-time kymograph (pixel size "
            f"{px:.1f} um, {fps:.0f} fps, frames {start_f}-{end_f}). Consecutive frames "
            f"were subtracted to isolate the moving bolus front, which was tracked with a "
            f"boundary-constrained dynamic program enforcing monotone, velocity-bounded "
            f"advance (v_max {r['v_max_um_s']:.0f} um/s) seeded at the "
            f"{'proximal' if r['direction_resolved']=='+' else 'distal'} end "
            f"(resolved automatically). Global front speed "
            f"{r['speed_mm_s']:+.3f} mm/s{ci_s}, R2 {r['r_squared']:.2f}; independent "
            f"cross-correlation cross-check {r['xcorr_speed_mm_s']:+.3f} mm/s. Tracking "
            f"covered {100*r['reach_frac']:.0f}% of the ROI length over "
            f"{100*r['obs_frac']:.0f}% of frames (observed, non-interpolated). Panels: "
            f"(a) difference kymograph with recovered front; (b) front position vs time "
            f"with linear fit; (c) local speed along the vessel; (d) speed map overlaid "
            f"on the anatomical image. Assumes a single dominant bolus front advancing "
            f"monotonically; speeds above v_max are sampling-limited and under-read.")

    def _fsm_build_window(self):
        """Create the Flow Speed Map window: control strip + embedded canvas."""
        win = tk.Toplevel(self.root)
        win.title("Mouse Ear - Flow Speed Map")
        win.geometry("1400x950")
        self._fsm['win'] = win

        ctl = ttk.Frame(win); ctl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=3)

        # --- row 1: speed colormap + dynamic range ---
        r1 = ttk.Frame(ctl); r1.pack(fill=tk.X, pady=1)
        ttk.Label(r1, text="Speed cmap:").pack(side=tk.LEFT)
        self._fsm_cmap_var = tk.StringVar(value="viridis")
        ttk.Combobox(r1, textvariable=self._fsm_cmap_var, width=12, state="readonly",
                     values=self._FSM_SEQ_CMAPS + self._FSM_DIV_CMAPS
                     + ["turbo (discouraged)"]).pack(side=tk.LEFT, padx=(3, 8))
        self._fsm_rev_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text="reverse", variable=self._fsm_rev_var).pack(side=tk.LEFT)
        self._fsm_sym_var = tk.BooleanVar(value=False)
        self._fsm_sym_cb = ttk.Checkbutton(r1, text="symmetric@0", variable=self._fsm_sym_var)
        self._fsm_sym_cb.pack(side=tk.LEFT, padx=(6, 8))
        self._fsm_auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r1, text="auto range (pct)", variable=self._fsm_auto_var).pack(side=tk.LEFT)
        ttk.Label(r1, text="p:").pack(side=tk.LEFT)
        self._fsm_pmin_var = tk.DoubleVar(value=5.0); self._fsm_pmax_var = tk.DoubleVar(value=95.0)
        ttk.Entry(r1, textvariable=self._fsm_pmin_var, width=4).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self._fsm_pmax_var, width=4).pack(side=tk.LEFT, padx=(1, 8))
        self._fsm_vlim_unit_var = tk.StringVar(value="vmin/vmax(mm/s):")
        ttk.Label(r1, textvariable=self._fsm_vlim_unit_var).pack(side=tk.LEFT)
        self._fsm_vmin_var = tk.DoubleVar(value=0.0); self._fsm_vmax_disp_var = tk.DoubleVar(value=1.0)
        ttk.Entry(r1, textvariable=self._fsm_vmin_var, width=5).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self._fsm_vmax_disp_var, width=5).pack(side=tk.LEFT, padx=(1, 0))

        # --- row 2: background image controls ---
        r2 = ttk.Frame(ctl); r2.pack(fill=tk.X, pady=1)
        ttk.Label(r2, text="BG source:").pack(side=tk.LEFT)
        self._fsm_bgsrc_var = tk.StringVar(value="center frame")
        ttk.Combobox(r2, textvariable=self._fsm_bgsrc_var, width=13, state="readonly",
                     values=("center frame", "mean", "max-proj", "current frame")
                     ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(r2, text="gamma:").pack(side=tk.LEFT)
        self._fsm_gamma_var = tk.DoubleVar(value=1.0)
        ttk.Entry(r2, textvariable=self._fsm_gamma_var, width=5).pack(side=tk.LEFT, padx=(1, 8))
        self._fsm_bgmode_var = tk.StringVar(value="percentile")
        ttk.Combobox(r2, textvariable=self._fsm_bgmode_var, width=10, state="readonly",
                     values=("percentile", "absolute")).pack(side=tk.LEFT)
        ttk.Label(r2, text="lo/hi:").pack(side=tk.LEFT)
        self._fsm_bglo_var = tk.DoubleVar(value=1.0); self._fsm_bghi_var = tk.DoubleVar(value=99.5)
        ttk.Entry(r2, textvariable=self._fsm_bglo_var, width=5).pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self._fsm_bghi_var, width=6).pack(side=tk.LEFT, padx=(1, 8))
        self._fsm_bginv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text="invert", variable=self._fsm_bginv_var).pack(side=tk.LEFT)
        self._fsm_bgcrop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="auto-crop", variable=self._fsm_bgcrop_var).pack(side=tk.LEFT, padx=(6, 0))

        # --- row 3: mode + actions ---
        r3 = ttk.Frame(ctl); r3.pack(fill=tk.X, pady=1)
        self._fsm_pub_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r3, text="Publication mode", variable=self._fsm_pub_var).pack(side=tk.LEFT)
        ttk.Label(r3, text="width:").pack(side=tk.LEFT, padx=(8, 0))
        self._fsm_width_var = tk.StringVar(value="double (183mm)")
        ttk.Combobox(r3, textvariable=self._fsm_width_var, width=13, state="readonly",
                     values=("single (89mm)", "double (183mm)")).pack(side=tk.LEFT, padx=(3, 12))
        ttk.Label(r3, text="Paint:").pack(side=tk.LEFT)
        self._fsm_paint_var = tk.StringVar(value="speed (mm/s)")
        _paint_cb = ttk.Combobox(r3, textvariable=self._fsm_paint_var, width=13, state="readonly",
                                 values=("speed (mm/s)", "arrival time"))
        _paint_cb.pack(side=tk.LEFT, padx=(3, 4))
        self._fsm_submode_var = tk.StringVar(value="local")
        ttk.Combobox(r3, textvariable=self._fsm_submode_var, width=8, state="readonly",
                     values=("local", "uniform")).pack(side=tk.LEFT, padx=(0, 12))
        # per-mode manual-limit memory + unit relabel + grey symmetric in arrival mode
        self._fsm_mode_limits = {'speed': [0.0, 1.0], 'arrival': [0.0, 1000.0]}

        def _on_paint_change(_e=None):
            paint = 'arrival' if self._fsm_paint_var.get().startswith('arrival') else 'speed'
            cur = [self._fsm_vmin_var.get(), self._fsm_vmax_disp_var.get()]
            # save the CURRENTLY displayed numbers to the mode they belong to,
            # then repopulate with the newly-selected mode's last-used values
            other = 'arrival' if paint == 'speed' else 'speed'
            self._fsm_mode_limits[other] = cur
            lo, hi = self._fsm_mode_limits[paint]
            self._fsm_vmin_var.set(lo); self._fsm_vmax_disp_var.set(hi)
            self._fsm_vlim_unit_var.set('vmin/vmax(ms):' if paint == 'arrival'
                                        else 'vmin/vmax(mm/s):')
            try:
                self._fsm_sym_cb.configure(state=('disabled' if paint == 'arrival' else 'normal'))
            except Exception:
                pass
            self._fsm_redraw()
        _paint_cb.bind('<<ComboboxSelected>>', _on_paint_change)
        ttk.Button(r3, text="Apply / Redraw", command=self._fsm_redraw).pack(side=tk.LEFT, padx=3)
        ttk.Button(r3, text="Re-run tracking",
                   command=lambda: (self._fsm_recompute() and self._fsm_redraw())).pack(side=tk.LEFT, padx=3)
        ttk.Button(r3, text="Copy caption", command=self._fsm_copy_caption).pack(side=tk.LEFT, padx=3)
        ttk.Button(r3, text="Export figure…", command=self._fsm_export).pack(side=tk.LEFT, padx=3)
        ttk.Button(r3, text="Derived metrics…", command=self._fsm_show_metrics).pack(side=tk.LEFT, padx=3)
        # waypoint inspector: pick one waypoint to zoom panel (c) into (display-only)
        ttk.Label(r3, text="show WP:").pack(side=tk.LEFT, padx=(10, 0))
        self._fsm_wp_sel_var = tk.StringVar(value="all")
        _wpsel = ttk.Combobox(r3, textvariable=self._fsm_wp_sel_var, width=6, state="readonly",
                              values=("all",))
        _wpsel.pack(side=tk.LEFT, padx=(2, 0))
        self._fsm_wp_selcb = _wpsel
        _wpsel.bind('<<ComboboxSelected>>', lambda e: self._fsm_redraw())

        fig = Figure(figsize=(13, 8.2))
        self._fsm['fig'] = fig
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._fsm['canvas'] = canvas
        self._fsm_redraw()

    def _fsm_pull_opts(self):
        """Copy the control-widget values into self._fsm (read by the drawers)."""
        f = self._fsm
        f['cmap'] = self._fsm_cmap_var.get(); f['cmap_reverse'] = self._fsm_rev_var.get()
        f['symmetric'] = self._fsm_sym_var.get(); f['auto_range'] = self._fsm_auto_var.get()
        f['pmin'] = self._fsm_pmin_var.get(); f['pmax'] = self._fsm_pmax_var.get()
        f['vmin'] = self._fsm_vmin_var.get(); f['vmax'] = self._fsm_vmax_disp_var.get()
        _bmap = {"center frame": "center", "mean": "mean", "max-proj": "max",
                 "current frame": "current"}
        f['bg_source'] = _bmap.get(self._fsm_bgsrc_var.get(), "center")
        f['bg_gamma'] = self._fsm_gamma_var.get()
        f['bg_mode'] = self._fsm_bgmode_var.get()
        f['bg_lo'] = self._fsm_bglo_var.get(); f['bg_hi'] = self._fsm_bghi_var.get()
        f['bg_invert'] = self._fsm_bginv_var.get(); f['bg_crop'] = self._fsm_bgcrop_var.get()
        f['pub'] = self._fsm_pub_var.get()
        f['width_mm'] = 89.0 if self._fsm_width_var.get().startswith('single') else 183.0
        # waypoint inspector selection (display-only): "all" or an integer index
        if hasattr(self, '_fsm_wp_sel_var'):
            _sel = self._fsm_wp_sel_var.get()
            if _sel is None or str(_sel).strip().lower() in ('', 'all'):
                f['wp_sel'] = None
            else:
                try:
                    f['wp_sel'] = int(_sel)
                except (TypeError, ValueError):
                    f['wp_sel'] = None
        # arrival-time paint mode + sub-mode are DISPLAY controls (redraw only)
        if hasattr(self, '_fsm_paint_var'):
            paint = ('arrival' if self._fsm_paint_var.get().startswith('arrival') else 'speed')
            f['arrival_paint'] = paint
            f['arrival_submode'] = (self._fsm_submode_var.get()
                                    if hasattr(self, '_fsm_submode_var') else 'local')
            # manual vmin/vmax are stored PER MODE (mm/s vs ms)
            vk = 'arrival' if paint == 'arrival' else 'speed'
            f['vmin_' + vk] = self._fsm_vmin_var.get()
            f['vmax_' + vk] = self._fsm_vmax_disp_var.get()

    def _fsm_redraw(self):
        """Re-render the cached tracking result with the current display options.
        Never recomputes the tracker."""
        import matplotlib as mpl
        self._fsm_pull_opts()
        fig = self._fsm['fig']
        # Part A: remove EVERY axes explicitly (incl. orphaned colorbar axes) and
        # drop cached colorbar refs so nothing survives into the next draw.
        for ax in list(fig.axes):
            fig.delaxes(ax)
        fig.clf()
        self._fsm.pop('cbar', None)
        pub = self._fsm.get('pub', False)
        width_in = self._fsm.get('width_mm', 183.0) / 25.4
        if pub:
            new_size = (width_in, min(width_in * 0.72, 247.0 / 25.4))
            rc = {'font.family': 'sans-serif',
                  'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                  'font.size': 6, 'axes.labelsize': 6, 'axes.titlesize': 6.5,
                  'xtick.labelsize': 5, 'ytick.labelsize': 5, 'legend.fontsize': 5,
                  'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                  'xtick.direction': 'out', 'ytick.direction': 'out',
                  'axes.spines.top': False, 'axes.spines.right': False,
                  'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}
        else:
            new_size = (13, 8.2)
            rc = {}
        # Part A: only resize the figure when the size ACTUALLY changes, and keep
        # the Tk widget in sync so a narrower bitmap cannot leave a stale strip.
        cur = tuple(round(v, 4) for v in fig.get_size_inches())
        if cur != tuple(round(v, 4) for v in new_size):
            fig.set_size_inches(*new_size)
            try:
                dpi = fig.get_dpi()
                w = fig.get_window_extent().width if False else int(round(new_size[0] * dpi))
                h = int(round(new_size[1] * dpi))
                self._fsm['canvas'].get_tk_widget().config(width=w, height=h)
            except Exception:
                pass
        with mpl.rc_context(rc):
            self._fsm_draw_panels(fig, pub)
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.94 if not pub else 0.98])
        except Exception:
            pass
        # full repaint (not draw_idle) so no widget region keeps an old bitmap
        self._fsm['canvas'].draw()
        if self._fsm['res'].get('sign_conflict'):
            self._log_console("[flow-speed] WARNING shown in figure: seed direction "
                              "conflicts with xcorr cross-check sign.")

    def _fsm_draw_panels(self, fig, pub):
        """Draw the four panels (shared by diagnostic and publication modes)."""
        from matplotlib.collections import LineCollection
        res = self._fsm['res']; s_um = self._fsm['s_um']
        if res.get('method_kind') == 'arrival_time':
            return self._fsm_draw_arrival_panels(fig, pub)
        dks = res['diff_kymo']; t_mid = res['t_mid_s']; peak_s = res['peak_s_um']
        valid = res['valid']; step_flag = res['step_flag']
        obs_m = valid & (step_flag == 'observed')
        int_m = valid & (step_flag == 'interpolated')
        track_t = res['track_t_s']; track_s = res['track_s_um']
        pp = res['prof_pos_um']; pv = res['prof_speed_mm_s']
        lp = res['local_pos_um']; ls = res['local_speed_mm_s']
        cmap, cv_lo, cv_hi = self._fsm_speed_norm()

        def _panel_label(ax, letter):
            ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=9,
                    fontweight='bold', va='top', ha='left')

        ax1 = fig.add_subplot(221)
        vlim = np.percentile(np.abs(dks), 99) + 1e-9
        im1 = ax1.imshow(dks, aspect='auto', cmap='RdBu_r', origin='lower',
                         vmin=-vlim, vmax=vlim,
                         extent=[t_mid[0], t_mid[-1], s_um[0], s_um[-1]])
        if track_t.size:
            ax1.plot(track_t, track_s, '-', color='0.15', linewidth=1.0, alpha=0.9)
        ax1.plot(t_mid[obs_m], peak_s[obs_m], 'k.', markersize=3 if pub else 5)
        if int_m.any():
            ax1.plot(t_mid[int_m], peak_s[int_m], 'o', mfc='none', mec='0.25',
                     markersize=3 if pub else 5)
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Position (µm)')
        cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
        cb1.set_label('Δ intensity (a.u.)')
        if not pub:
            ax1.set_title('Consecutive-frame difference (moving bolus front)')
            ax1.legend(['front', 'observed', 'interpolated'], fontsize=7, loc='upper right')
        _panel_label(ax1, 'a')

        ax2 = fig.add_subplot(222)
        ax2.plot(t_mid[obs_m], peak_s[obs_m], 'o', color='#0072B2',
                 markersize=3 if pub else 4, label='observed')
        if int_m.any():
            ax2.plot(t_mid[int_m], peak_s[int_m], 'o', mfc='none', mec='#0072B2',
                     markersize=4, label='interpolated')
        if track_t.size:
            ax2.plot(track_t, track_s, '-', color='0.3', linewidth=1.0)
        if np.isfinite(res['speed_mm_s']):
            fm = res['fit_mask']; tv = t_mid[fm]
            slope_um = res['speed_mm_s'] * 1e3
            b = np.median(peak_s[fm] - slope_um * tv)
            xs = np.array([tv.min(), tv.max()])
            ax2.plot(xs, b + slope_um * xs, '-', color='#D55E00', linewidth=1.2,
                     label=f"{res['speed_mm_s']:.3f} mm/s")
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Front position (µm)')
        if not pub:
            ax2.set_title('Peak advance (slope = speed)'); ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=5 if pub else 7)
        _panel_label(ax2, 'b')

        ax3 = fig.add_subplot(223)
        if lp.size and not pub:
            oo = np.argsort(lp)
            ax3.plot(lp[oo], ls[oo], '.', color='0.6', markersize=4, alpha=0.5, label='raw')
        if pp.size:
            ax3.plot(pp, pv, '-', color='#009E73', linewidth=1.6, label='local speed')
        if np.isfinite(res['speed_mm_s']):
            ax3.axhline(abs(res['speed_mm_s']), color='#D55E00', linestyle='--',
                        linewidth=0.9, label=f"global {abs(res['speed_mm_s']):.3f}")
        ax3.set_xlabel('Position (µm)'); ax3.set_ylabel('Local speed (mm/s)')
        ax3.legend(fontsize=5 if pub else 8)
        if not pub:
            ax3.set_title('Speed profile along vessel'); ax3.grid(True, alpha=0.3)
        _panel_label(ax3, 'c')

        ax4 = fig.add_subplot(224)
        img01, bbox, bglabel = self._fsm_background_image()
        ax4.imshow(img01, cmap='gray', vmin=0.0, vmax=1.0)
        centers = self._fsm['centers']; s_c = self._fsm['s_c']
        if centers is not None and pp.size:
            spd_on_s = np.interp(s_c, pp, pv)
            pts = centers.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_spd = 0.5 * (spd_on_s[:-1] + spd_on_s[1:])
            lc = LineCollection(segs, cmap=cmap, linewidth=4 if pub else 5,
                                norm=plt.Normalize(cv_lo, cv_hi))
            lc.set_array(seg_spd); ax4.add_collection(lc)
            cb4 = fig.colorbar(lc, ax=ax4, fraction=0.046, pad=0.02)
            cb4.set_label('Speed (mm/s)')
            try: cb4.set_ticks(np.linspace(cv_lo, cv_hi, 4))
            except Exception: pass
        if bbox is not None:
            ax4.set_xlim(bbox[0], bbox[1]); ax4.set_ylim(bbox[3], bbox[2])
        ax4.set_xticks([]); ax4.set_yticks([])
        for sp in ax4.spines.values():
            sp.set_visible(False)
        # µm scale bar
        bar_um = 200.0; bar_px = bar_um / self.pixel_size
        x0l, x1l = ax4.get_xlim(); y1l, y0l = ax4.get_ylim()  # note inverted y
        bx = x1l - 0.06 * (x1l - x0l) - bar_px
        by = y0l - 0.08 * (y0l - y1l)
        ax4.plot([bx, bx + bar_px], [by, by], '-', color='w', linewidth=2.5,
                 solid_capstyle='butt')
        ax4.text(bx + bar_px / 2, by - 0.02 * (y0l - y1l), f"{bar_um:.0f} µm",
                 color='w', ha='center', va='bottom', fontsize=6)
        if not pub:
            ax4.set_title(f'Flow speed map ({bglabel})')
        else:
            ax4.set_xlabel(bglabel, fontsize=5)
        _panel_label(ax4, 'd')

        # ---- header / status ----
        r = res; ci = r['speed_ci']
        if pub:
            fig.suptitle(f"Vessel front speed {r['speed_mm_s']:+.3f} mm/s  "
                         f"(dir {r['direction_resolved']})", fontsize=7)
        else:
            ci_txt = (f" 95%CI[{ci[0]:.3f},{ci[1]:.3f}]"
                      if np.isfinite(ci[0]) and np.isfinite(ci[1]) else "")
            warn = "  ⚠ SIGN CONFLICT vs xcorr" if r.get('sign_conflict') else ""
            fig.suptitle(
                f"Flow speed (seeded DP): {r['speed_mm_s']:+.3f} mm/s{ci_txt} | "
                f"R²={r['r_squared']:.2f} | dir {r['direction_resolved']} seed@f{r['seed_frame']} | "
                f"xcorr {r['xcorr_speed_mm_s']:+.3f} | reach {100*r['reach_frac']:.0f}% "
                f"obs {100*r['obs_frac']:.0f}% | acc {r['n_valid']}/{r['n_steps']} "
                f"term {r['n_terminated']} | vmax {r['v_max_um_s']:.0f}µm/s{warn}",
                fontsize=8, color=('#b00' if r.get('sign_conflict') else 'black'))

    def _fsm_derived_metrics(self, k_profile=None, mu=None):
        """Part D: derived haemodynamics from arrival-time speed + FWHM diameter.

        CRITICAL (D.1): the arrival-time method tracks the bolus LEADING EDGE,
        which propagates near the CENTRELINE (max) velocity, not the cross-
        sectional mean. v_mean = k_profile * v_front (Poiseuille k=0.5; blunted
        microvessel profile ~0.6-0.7). We report BOTH and never apply k silently.

        All quantities assume a circular lumen; Q ∝ d² so the diameter error
        dominates — the diameter uncertainty and the speed CI are propagated into
        CIs for Q, shear-rate and shear-stress. Returns a dict or None."""
        res = (self.vessel_results or {}).get('arrival_time')
        if res is None or not np.isfinite(res.get('speed_mm_s', np.nan)):
            return None
        kp = float(k_profile if k_profile is not None else getattr(self, 'flow_k_profile', 0.6))
        mu = float(mu if mu is not None else self.blood_viscosity)   # Pa·s
        rho = float(self.blood_density)
        # reuse the EXISTING FWHM morphometry (no reimplementation)
        vr = self.vessel_results.get('vessel_results_cache')
        if vr is None:
            vr = self._extract_vessel_morphometry()
        if vr is None:
            return None
        diam = np.asarray(vr['diam_um'], float); flags = vr['flags']; s_d = np.asarray(vr['s_um'], float)
        ok = np.isfinite(diam) & (flags == 'ok')
        if ok.sum() < 2:
            ok = np.isfinite(diam)
        if ok.sum() < 2:
            return None
        d_um = float(np.mean(diam[ok])); d_sem = float(np.std(diam[ok], ddof=1) / np.sqrt(ok.sum()))
        rel_d = d_sem / max(d_um, 1e-9)
        v_front = float(res['speed_mm_s'])                 # mm/s (signed)
        ci = res['speed_ci']
        rel_v = (abs(ci[1] - ci[0]) / (2 * abs(v_front))
                 if np.isfinite(ci[0]) and np.isfinite(ci[1]) and v_front != 0 else np.nan)
        v_mean = kp * v_front                               # mm/s
        # SI-ish working units: mm/s, µm
        d_m = d_um * 1e-6; v_m = abs(v_mean) * 1e-3         # m
        A = np.pi * (d_m / 2) ** 2                          # m²
        Q = v_m * A                                        # m³/s
        Q_pl_s = Q * 1e15                                  # 1 m³ = 1e15 pL
        Q_nl_min = Q * 1e12 * 60.0                         # 1 m³ = 1e12 nL, per minute
        shear = 8.0 * v_m / d_m                             # 1/s (Poiseuille wall shear rate)
        tau = mu * shear                                    # Pa
        L_um = float(s_d.max() - s_d.min()); transit = (L_um * 1e-6) / max(v_m, 1e-12)  # s
        Re = rho * v_m * d_m / max(mu, 1e-12)
        # uncertainty propagation (relative)
        rv = rel_v if np.isfinite(rel_v) else 0.0
        relQ = float(np.sqrt(rv ** 2 + (2 * rel_d) ** 2))
        relSh = float(np.sqrt(rv ** 2 + rel_d ** 2))
        def _ci(x, rel):
            return (x * (1 - rel), x * (1 + rel))
        # continuity: v(s)·d(s)² should be flat along an unbranched segment
        pp = res['prof_pos_um']; pv = np.abs(res['prof_speed_mm_s'])
        vhat = np.interp(s_d[ok], pp, pv, left=np.nan, right=np.nan)
        vhat = np.where(np.isfinite(vhat), vhat, abs(v_front))
        cont = vhat * diam[ok] ** 2
        cont_cv = float(np.std(cont) / (np.mean(cont) + 1e-12)) * 100.0
        out = dict(k_profile=kp, mu_Pa_s=mu, rho=rho, d_um=d_um, d_sem_um=d_sem,
                   rel_d=rel_d, rel_v=rv, v_front_mm_s=v_front, v_mean_mm_s=v_mean,
                   v_mean_ci=(_ci(v_mean, rv) if np.isfinite(rv) else (np.nan, np.nan)),
                   area_um2=A * 1e12, Q_nl_min=Q_nl_min, Q_ci_nl_min=_ci(Q_nl_min, relQ),
                   Q_pl_s=Q_pl_s, shear_1_s=shear, shear_ci=_ci(shear, relSh),
                   tau_Pa=tau, tau_ci=_ci(tau, relSh), transit_s=transit,
                   Re=Re, L_um=L_um, cont_pos_um=s_d[ok], cont_vd2=cont, cont_cv_pct=cont_cv,
                   n_diam=int(ok.sum()))
        self._log_console(
            f"[metrics] v_front={v_front:+.3f} mm/s, k_profile={kp:.2f} -> v_mean={v_mean:+.3f} "
            f"mm/s (CI±{100*rv:.0f}%); d={d_um:.1f}±{d_sem:.1f} µm; "
            f"Q={Q_nl_min:.2f} nL/min CI[{out['Q_ci_nl_min'][0]:.2f},{out['Q_ci_nl_min'][1]:.2f}] "
            f"({Q_pl_s:.1f} pL/s); shear={shear:.0f}/s CI[{out['shear_ci'][0]:.0f},{out['shear_ci'][1]:.0f}]; "
            f"tau={tau:.3f} Pa; transit={transit:.2f} s; Re={Re:.2e}")
        if Re > 1.0:
            self._log_console(f"[metrics] WARNING: Re={Re:.2f} > 1 — laminar/Poiseuille assumptions may not hold")
        self._log_console(f"[metrics] continuity v·d² CV={cont_cv:.0f}% "
                          f"(should be low for an unbranched vessel; high => missed branch / diameter bias)")
        return out

    def _fsm_show_metrics(self):
        """Derived-metrics window: a table of haemodynamic quantities with CIs and
        assumptions, plus the v·d² continuity plot (Part D presentation)."""
        self.update_ear_params()
        m = self._fsm_derived_metrics()
        if m is None:
            messagebox.showinfo("Derived metrics",
                                "Run the arrival-time method and draw a valid vessel ROI "
                                "with a diameter (FWHM) measurement first.")
            return
        win = tk.Toplevel(self._fsm.get('win', self.root)); win.title("Derived haemodynamic metrics")
        win.geometry("760x620")
        txt = tk.Text(win, wrap='word', height=17, font=('Consolas', 9))
        txt.pack(fill=tk.X, padx=6, pady=6)
        def _row(name, val, unit, ci=None):
            s = f"{name:<26s} {val:>12}  {unit}"
            if ci is not None and np.isfinite(ci[0]):
                s += f"   95% CI [{ci[0]:.3g}, {ci[1]:.3g}]"
            return s + "\n"
        body = "DERIVED HAEMODYNAMIC METRICS\n" + "=" * 60 + "\n\n"
        body += _row("v_front (leading edge)", f"{m['v_front_mm_s']:+.3f}", "mm/s")
        body += _row("v_mean (× k_profile)", f"{m['v_mean_mm_s']:+.3f}", "mm/s", m['v_mean_ci'])
        body += _row("  k_profile", f"{m['k_profile']:.2f}", "(v_mean/v_front)")
        body += _row("diameter d (FWHM)", f"{m['d_um']:.1f}±{m['d_sem_um']:.1f}", "µm")
        body += _row("cross-section area", f"{m['area_um2']:.0f}", "µm²")
        body += _row("flow rate Q", f"{m['Q_nl_min']:.3f}", "nL/min", m['Q_ci_nl_min'])
        body += _row("flow rate Q", f"{m['Q_pl_s']:.2f}", "pL/s")
        body += _row("wall shear rate", f"{m['shear_1_s']:.0f}", "1/s", m['shear_ci'])
        body += _row("wall shear stress τ", f"{m['tau_Pa']:.3f}", "Pa", m['tau_ci'])
        body += _row("transit time (L/v_mean)", f"{m['transit_s']:.2f}", "s")
        body += _row("Reynolds number", f"{m['Re']:.2e}", "" + ("  ⚠ >1!" if m['Re'] > 1 else "(≪1 OK)"))
        body += _row("v·d² continuity CV", f"{m['cont_cv_pct']:.0f}", "%")
        body += ("\nASSUMPTIONS (model-dependent, not direct measurements):\n"
                 f"  • v_mean = k_profile·v_front (leading edge ≈ centreline velocity); "
                 f"k={m['k_profile']:.2f}.\n"
                 f"  • circular lumen; Q=v_mean·π(d/2)². Q∝d² so diameter error dominates.\n"
                 f"  • wall shear from Poiseuille (γ̇=8v/d); viscosity µ={m['mu_Pa_s']*1e3:.1f} cP "
                 f"(fixed; blood is non-Newtonian / Fåhræus–Lindqvist in vivo).\n"
                 f"  • CIs propagate the speed CI (±{100*m['rel_v']:.0f}%) and diameter SEM "
                 f"(±{100*m['rel_d']:.0f}%).\n")
        txt.insert('1.0', body); txt.configure(state='disabled')
        # v·d² continuity plot
        from matplotlib.figure import Figure as _F
        f = _F(figsize=(7, 3))
        ax = f.add_subplot(111)
        ax.plot(m['cont_pos_um'], m['cont_vd2'], 'o-', color='#0072B2', ms=4)
        ax.axhline(np.mean(m['cont_vd2']), color='#D55E00', ls='--',
                   label=f"mean (CV={m['cont_cv_pct']:.0f}%)")
        ax.set_xlabel('Position (µm)'); ax.set_ylabel('v·d²  (∝ Q)')
        ax.set_title('Continuity check: v·d² should be flat (unbranched vessel)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        f.tight_layout()
        cv = FigureCanvasTkAgg(f, win); cv.draw()
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

    def _fsm_paint_vessel_smooth(self, fig, ax4, r, mode, submode, pub, tiso, vm):
        """Smooth-gradient vessel painter (Parts B & C).

        B: the DATA colormap stays pure (plt.get_cmap(name).copy() + set_bad);
        provenance/confidence is expressed as PER-SEGMENT RGBA alpha (never a
        collection-wide set_alpha, which would wash out the colorbar); the colorbar
        is drawn from an INDEPENDENT ScalarMappable so it always shows the true,
        fully-saturated scale.
        C: the centreline is resampled by arc length to a fine uniform grid and the
        colour value is PCHIP-interpolated onto it (shape-preserving, no ringing,
        no extrapolation beyond the fitted s range), then drawn with round caps/
        joins and antialiasing so no discrete segment boundaries remain."""
        from matplotlib.collections import LineCollection
        from scipy.interpolate import PchipInterpolator
        img01, bbox, _bgl = self._fsm_background_image()
        ax4.imshow(img01, cmap='gray', vmin=0, vmax=1)
        centers = self._fsm['centers']; s_c = np.asarray(self._fsm['s_c'], float)
        s_um = r['s_um']
        cmapn, cv_lo, cv_hi, cblab = self._fsm_paint_norm(mode)
        cmap = plt.get_cmap(cmapn).copy(); cmap.set_bad((0, 0, 0, 0))   # pure ramp
        norm = plt.Normalize(cv_lo, cv_hi)
        if centers is not None and len(centers) >= 2:
            centers = np.asarray(centers, float)
            # arc-length parameterisation of the drawn centreline
            seg = np.diff(centers, axis=0)
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
            total = float(arc[-1])
            n_fine = int(np.clip(total / 0.4, 400, 1200))   # ~0.4 px spacing
            au = np.linspace(0, total, n_fine)
            fx = np.interp(au, arc, centers[:, 0]); fy = np.interp(au, arc, centers[:, 1])
            s_fine = np.interp(au, arc, s_c)                 # position (µm) at each node
            sfit = r.get('s_fit_range', (s_um.min(), s_um.max()))

            # value(s) + provenance(s) on the fine grid via PCHIP within fitted range
            def _pchip(x, y):
                m = np.isfinite(y)
                if m.sum() < 2:
                    return np.full_like(s_fine, np.nan)
                xu, iu = np.unique(x[m], return_index=True); yu = y[m][iu]
                if xu.size < 2:
                    return np.full_like(s_fine, np.nan)
                out = PchipInterpolator(xu, yu, extrapolate=False)(s_fine)
                return out

            prov_code = np.zeros(n_fine)      # 2=local 1=global 0=none
            if mode == 'arrival' and vm.any():
                t0 = np.nanmin(tiso[vm])
                vfine = _pchip(s_um[vm], (tiso[vm] - t0) * 1e3)
                prov_code = np.where(np.isfinite(vfine), 2.0, 0.0)
            elif submode == 'uniform':
                gv = abs(r['speed_mm_s'])
                inf = (s_fine >= sfit[0]) & (s_fine <= sfit[1])
                vfine = np.where(inf, gv, np.nan); prov_code = np.where(inf, 1.0, 0.0)
            else:
                pp = r['prof_pos_um']; pv = np.abs(r['prof_speed_mm_s'])
                vfine = _pchip(pp, pv)
                pprov = r.get('speed_provenance')
                if pprov is not None:
                    code = np.where(pprov == 'local', 2.0,
                                    np.where(pprov == 'global', 1.0, 0.0))
                    prov_code = np.interp(s_fine, pp, code, left=0, right=0)
                else:
                    prov_code = np.where(np.isfinite(vfine), 2.0, 0.0)
            outside = (s_fine < sfit[0]) | (s_fine > sfit[1])
            vfine = np.where(outside, np.nan, vfine)

            # fine segments
            pts = np.column_stack([fx, fy]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            vseg = 0.5 * (vfine[:-1] + vfine[1:])
            pseg = 0.5 * (prov_code[:-1] + prov_code[1:])
            good = np.isfinite(vseg)
            # per-segment RGBA from the PURE colormap; alpha ramps smoothly with
            # provenance (local=1.0 -> global≈0.55) so colour stays continuous.
            rgba = cmap(norm(np.where(good, vseg, cv_lo)))
            alpha = np.clip(0.55 + 0.45 * np.clip((pseg - 1.0), 0, 1), 0.0, 1.0)
            rgba[:, 3] = np.where(good, alpha, 0.0)          # no-data -> transparent
            lw = (5.5 if not pub else 4.0)
            lc = LineCollection(segs, colors=rgba, linewidths=lw, antialiased=True,
                                capstyle='round', joinstyle='round')
            ax4.add_collection(lc)
            # grey 'no data' line UNDER the coloured one (not in the ramp)
            nod = ~good
            if nod.any():
                ax4.add_collection(LineCollection(
                    segs[nod], colors=[(0.6, 0.6, 0.6, 0.35)], linewidths=lw * 0.7,
                    antialiased=True, capstyle='round', zorder=1))
            # INDEPENDENT colorbar -> always the true, fully-saturated scale
            import matplotlib as mpl
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
            cb = fig.colorbar(sm, ax=ax4, fraction=0.046, pad=0.02); cb.set_label(cblab)
            # no-data / low-confidence go to a small legend, not the colorbar
            if not pub:
                from matplotlib.lines import Line2D
                leg = [Line2D([0], [0], color='0.6', lw=3, alpha=0.5, label='no data'),
                       Line2D([0], [0], color=cmap(0.7), lw=3, alpha=0.55, label='global-fallback')]
                ax4.legend(handles=leg, fontsize=5, loc='lower left', framealpha=0.5)
        if bbox is not None:
            ax4.set_xlim(bbox[0], bbox[1]); ax4.set_ylim(bbox[3], bbox[2])
        ax4.set_xticks([]); ax4.set_yticks([])
        for sp in ax4.spines.values():
            sp.set_visible(False)
        if not pub:
            tt = {'arrival': 'Arrival-time map', 'speed': 'Local-speed map'}[mode]
            if mode == 'speed' and submode == 'uniform':
                tt = 'Uniform speed map'
            ax4.set_title(tt, fontsize=8)

    def _fsm_sync_wp_selector(self, n_wp, reset=False):
        """Keep the 'show WP' combobox populated with ('all', 0, 1, ... n_wp-1).
        Display-only: never triggers a redraw itself. Preserves the current
        selection when it is still valid, unless reset=True (used when the
        waypoint SET itself changed — e.g. the ROI was redrawn — so the old
        index no longer refers to the same node)."""
        cb = getattr(self, '_fsm_wp_selcb', None)
        var = getattr(self, '_fsm_wp_sel_var', None)
        if cb is None or var is None:
            return
        try:
            if not cb.winfo_exists():         # FSM window was closed
                return
        except Exception:
            return
        want = ('all',) + tuple(str(i) for i in range(int(max(0, n_wp))))
        try:
            cb.configure(values=want)
            cur = var.get()
            if reset or cur not in want:
                var.set('all')
                if isinstance(getattr(self, '_fsm', None), dict):
                    self._fsm['wp_sel'] = None
        except Exception:
            pass

    def _fsm_draw_wp_inspector(self, ax, r, sel, t_s, ta, tiso, pub):
        """Detailed single-waypoint view for panel (c): the selected node's
        normalised time course, its positive derivative (feature signal), every
        ranked candidate passage time, and the chosen time — annotated with the
        node's assignment state."""
        s_um = r['s_um']
        nr = r['norm_rows'][sel] if r.get('norm_rows') is not None else None
        dr = r['wp_deriv_rows'][sel] if r.get('wp_deriv_rows') is not None else None
        cands = r['wp_candidates'][sel] if r.get('wp_candidates') is not None else []
        scores = r['wp_scores'][sel] if r.get('wp_scores') is not None else []
        state = str(r['wp_state'][sel]) if r.get('wp_state') is not None else '?'
        t_arr = ta[sel] if sel < len(ta) else np.nan

        drew = False
        if nr is not None:
            nr = np.asarray(nr, float)
            ax.plot(t_s, nr, '-', color='#333333', lw=1.1, label='norm. ΔI', zorder=3)
            drew = True
        # positive derivative (feature signal), scaled into the same axis for context
        if dr is not None:
            dr = np.asarray(dr, float)
            dm = np.nanmax(np.abs(dr))
            if np.isfinite(dm) and dm > 0:
                ax.plot(t_s, dr / dm, '-', color='#0072B2', lw=0.9, alpha=0.6,
                        label='deriv (scaled)', zorder=2)
                drew = True
        # every ranked candidate: open circle at (t_cand, score/normalised)
        cands = list(cands) if cands is not None else []
        scores = list(scores) if scores is not None else []
        if cands:
            sc = np.asarray(scores, float) if scores else np.ones(len(cands))
            smax = np.nanmax(sc) if sc.size and np.isfinite(np.nanmax(sc)) else 1.0
            smax = smax if smax > 0 else 1.0
            for k, tc in enumerate(cands):
                if not np.isfinite(tc):
                    continue
                yk = (sc[k] / smax) if k < len(sc) else 0.5
                ax.plot([tc], [yk], 'o', mfc='none', mec='#FF9500', mew=1.2,
                        markersize=7, zorder=4,
                        label='candidate' if k == 0 else None)
            drew = True
        # chosen passage time: vertical line + filled marker
        if np.isfinite(t_arr):
            ax.axvline(t_arr, color='#FF3B30', lw=1.2, alpha=0.8, zorder=1)
            ax.plot([t_arr], [0.0], 'v', color='#FF3B30', markersize=9, zorder=5,
                    label='chosen')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('norm. ΔI / score')
        ax.set_xlim(t_s[0], t_s[-1])
        manual = ' [MANUAL]' if state == 'manual' else ''
        title = f"WP{sel}  {s_um[sel]:.0f}µm  ·  {state}{manual}"
        if np.isfinite(t_arr):
            title += f"  ·  t={t_arr:.3f}s"
        if not pub:
            ax.set_title(title, fontsize=7)
            ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=6, loc='upper right')
        if not drew:
            ax.text(0.5, 0.5, f"WP{sel}: no time-course data\n(state: {state})",
                    ha='center', va='center', transform=ax.transAxes, fontsize=8)

    def _fsm_draw_arrival_panels(self, fig, pub):
        """Arrival-time method figure: (a) kymograph + per-row arrival times +
        isotonic fit; (b) t_arrival vs position + Theil-Sen fit; (c) waterfall of
        normalised time courses with the feature marked; (d) vessel painted by
        arrival time."""
        from matplotlib.collections import LineCollection
        r = self._fsm['res']; s_um = r['s_um']; t_s = r['t_s']
        kymo = r['kymo_raw']; ta = r['t_arrival_s']; tiso = r['t_arrival_isotonic_s']
        vm = r['row_valid_mask']
        is_wp = bool(r.get('is_waypoint', False))
        # resolve the "show WP" selection (display-only): sync the combobox to the
        # current waypoint count and clamp the selected index into range.
        sel = None
        if is_wp:
            n_wp = len(s_um)
            self._fsm_sync_wp_selector(n_wp)
            sel = self._fsm.get('wp_sel', None)
            if sel is not None and not (0 <= int(sel) < n_wp):
                sel = None

        def _pl(ax, letter):
            ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=9,
                    fontweight='bold', va='top', ha='left')

        def _sel_ring(ax, x, y):
            """Highlight the selected waypoint node with a magenta ring."""
            if sel is not None and np.isfinite(x) and np.isfinite(y):
                ax.plot([x], [y], 'o', mfc='none', mec='#FF00FF', mew=1.6,
                        markersize=11, zorder=6, label='selected WP')

        # three provenance classes: accepted / gate-rejected / trend-rejected
        reasons = r['reject_reasons']; gate_ok = r.get('gate_valid_mask', vm)
        trend_m = np.array([str(x) == 'trend_outlier' for x in reasons])
        gate_rej = (~gate_ok)                      # failed a per-row quality gate

        # (a) kymograph (arrival method) or node scatter (waypoint) + arrival points
        ax1 = fig.add_subplot(221)
        if kymo is not None:
            klo, khi = np.percentile(kymo, [2, 99])
            ax1.imshow(kymo, aspect='auto', cmap='gray', origin='lower', vmin=klo, vmax=khi,
                       extent=[t_s[0], t_s[-1], s_um[0], s_um[-1]])
        ax1.plot(ta[vm], s_um[vm], '.', color='#00E5FF', markersize=6 if is_wp else 4,
                 label='accepted')
        if trend_m.any():
            ax1.plot(ta[trend_m], s_um[trend_m], 'o', mfc='none', mec='#FF9500',
                     markersize=5, label='trend outlier')
        if gate_rej.any():
            ax1.plot(ta[gate_rej], s_um[gate_rej], 'x', color='0.55', markersize=3,
                     alpha=0.5, label='gate rejected')
        ax1.plot(tiso[vm], s_um[vm], '-', color='#FF3B30', lw=1.3, label='isotonic')
        if sel is not None:
            _sel_ring(ax1, ta[sel], s_um[sel])
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Position (µm)')
        if not pub:
            ax1.set_title('Node arrival times (isotonic)' if is_wp
                          else 'Kymograph + arrival times (isotonic)')
            ax1.legend(fontsize=6)
        _pl(ax1, 'a')

        # (b) t_arrival vs position + Theil-Sen fit + ±k·sigma acceptance band
        ax2 = fig.add_subplot(222)
        if np.isfinite(r['speed_mm_s']):
            sv = s_um[vm]; slope = 1.0 / (r['speed_mm_s'] * 1e3)
            b = np.median(ta[vm] - slope * sv)
            xs = np.array([s_um.min(), s_um.max()])
            line = b + slope * xs
            band = float(r.get('trend_k', 2.5)) * float(r.get('trend_sigma', 0.0) or 0.0)
            if band > 0:
                ax2.fill_between(xs, line - band, line + band, color='#D55E00', alpha=0.12,
                                 label=f"±{r.get('trend_k',2.5):g}σ ({band:.2f}s)")
            ax2.plot(xs, line, '-', color='#D55E00', lw=1.2, label=f"{r['speed_mm_s']:.3f} mm/s")
        ax2.plot(s_um[vm], ta[vm], 'o', color='#0072B2', markersize=3, label='accepted')
        if trend_m.any():
            ax2.plot(s_um[trend_m], ta[trend_m], 'o', mfc='none', mec='#FF9500',
                     markersize=5, label='trend outlier')
        if gate_rej.any():
            ax2.plot(s_um[gate_rej], ta[gate_rej], 'x', color='0.6', markersize=3,
                     alpha=0.5, label='gate rejected')
        if sel is not None:
            _sel_ring(ax2, s_um[sel], ta[sel])
        ax2.set_xlabel('Position (µm)'); ax2.set_ylabel('Arrival time (s)')
        ax2.legend(fontsize=6 if pub else 7)
        if not pub:
            ax2.set_title('t_arrival vs position (slope⁻¹ = speed)'); ax2.grid(True, alpha=0.3)
        _pl(ax2, 'b')

        # (c) EITHER a waterfall of node/row time courses (sel is None) OR — when a
        # single waypoint is selected — a detailed inspector for that one node:
        # its normalised time course, positive derivative, every ranked candidate,
        # and the chosen passage time, annotated with the assignment state.
        ax3 = fig.add_subplot(223)
        if is_wp and sel is not None:
            self._fsm_draw_wp_inspector(ax3, r, sel, t_s, ta, tiso, pub)
        else:
            vi = np.where(vm)[0]
            if vi.size:
                picks = vi[np.linspace(0, vi.size - 1, min(6, vi.size)).astype(int)]
                for r_i, ii in enumerate(picks):
                    nr = r['norm_rows'][ii]
                    if nr is None:
                        continue
                    off = r_i * 1.1
                    ax3.plot(t_s, nr + off, '-', color='#333333', lw=0.8)
                    # mark the extracted feature time
                    if np.isfinite(ta[ii]):
                        ax3.plot([ta[ii]], [0.5 + off], 'v', color='#FF3B30', markersize=6)
                    lab = (f"WP{ii} {s_um[ii]:.0f}µm" if is_wp else f"{s_um[ii]:.0f}µm")
                    ax3.text(t_s[0], off + 0.15, lab, fontsize=6, color='0.3')
            ax3.set_xlabel('Time (s)'); ax3.set_ylabel('norm. ΔI + offset')
            if not pub:
                ax3.set_title('Time courses at increasing position (progressive delay)')
        _pl(ax3, 'c')

        # (d) vessel painted by LOCAL SPEED / UNIFORM global speed / ARRIVAL TIME.
        # Smooth-gradient renderer (Parts B & C): pure colormap, per-segment RGBA
        # alpha for provenance, independent colorbar, arc-length + PCHIP resampling.
        ax4 = fig.add_subplot(224)
        paint = self._fsm.get('arrival_paint', 'speed')
        submode = self._fsm.get('arrival_submode', 'local')
        mode = 'arrival' if paint == 'arrival' else 'speed'
        self._fsm_paint_vessel_smooth(fig, ax4, r, mode, submode, pub, tiso, vm)
        # mark the selected waypoint on the vessel (image/pixel coordinates)
        if sel is not None and is_wp and r.get('wp_xy') is not None:
            xy = np.asarray(r['wp_xy'], float)
            if 0 <= sel < len(xy):
                ax4.plot([xy[sel, 0]], [xy[sel, 1]], 'o', mfc='none', mec='#FF00FF',
                         mew=1.8, markersize=13, zorder=7)
        _pl(ax4, 'd')

        ci = r['speed_ci']
        ci_txt = (f" CI[{ci[0]:.3f},{ci[1]:.3f}]"
                  if np.isfinite(ci[0]) and np.isfinite(ci[1]) else " CI[invalid]")
        st = r['isotonic_correction_stats']
        if pub:
            fig.suptitle(f"Arrival-time speed {r['speed_mm_s']:+.3f} mm/s "
                         f"(dir {r['direction_resolved']}, {r['timing_feature']})", fontsize=7)
        else:
            fig.suptitle(
                f"Arrival-time ({r['timing_feature']}): {r['speed_mm_s']:+.3f} mm/s{ci_txt} | "
                f"R²={r['r_squared']:.2f} | dir {r['direction_resolved']} | "
                f"valid {r['n_valid']} rej {r['n_rejected']} {dict(r['reject_tally'])} | "
                f"iso corr mean {st['mean']:.2f}s max {st['max']:.2f}s", fontsize=8)

    def _fsm_copy_caption(self):
        cap = self._fsm_caption()
        self._log_console("[flow-speed] suggested caption:\n" + cap)
        try:
            self.root.clipboard_clear(); self.root.clipboard_append(cap)
            self.status_var.set("Flow-speed caption copied to clipboard")
        except Exception:
            pass
        top = tk.Toplevel(self._fsm['win']); top.title("Suggested figure caption")
        txt = tk.Text(top, wrap='word', width=90, height=12); txt.pack(fill=tk.BOTH, expand=True)
        txt.insert('1.0', cap)

    def _fsm_export(self):
        """Export the current figure as vector PDF (+ optional SVG) and 600-dpi TIFF."""
        stem = self._fsm.get('stem', 'stack'); fr = self._fsm['frame_range']
        default = f"{stem}_flowspeed_f{fr[0]}-{fr[1]}"
        path = filedialog.asksaveasfilename(
            title="Export flow-speed figure", initialfile=default,
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("SVG", "*.svg"), ("TIFF 600dpi", "*.tif")])
        if not path:
            return
        base = os.path.splitext(path)[0]
        fig = self._fsm['fig']
        try:
            fig.savefig(base + ".pdf", bbox_inches='tight', facecolor='white')
            fig.savefig(base + ".svg", bbox_inches='tight', facecolor='white')
            fig.savefig(base + ".tif", dpi=600, bbox_inches='tight', facecolor='white',
                        pil_kwargs={'compression': 'tiff_lzw'})
            self._log_console(f"[flow-speed] exported {base}.pdf/.svg/.tif (600 dpi TIFF)")
            self.status_var.set(f"Exported figure to {os.path.basename(base)}.pdf/.svg/.tif")
            messagebox.showinfo("Export", f"Saved:\n{base}.pdf\n{base}.svg\n{base}.tif (600 dpi)")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ===================== Δframe INSPECTOR (QC viewer) =====================
    # Diagnostic tool: SEE the frame-to-frame difference the tracker sees, with
    # display controls, ROI + tracked-front overlay, synced kymograph and dye-
    # arrival curve, a sampling-adequacy readout, and MP4 export.
    #
    # GUIDING INVARIANT: what is shown == what the tracker consumes. The cached Δ
    # stack (self._dfi['D']) is the SAME consecutive-frame difference used by the
    # flow-speed tracker (gaussian pre-smoothing over position aside). Display
    # transforms (gamma, limit, colormap, sign filter) only remap the CACHED Δ for
    # rendering; they never alter the numbers. Computation is restricted to the
    # marker range and the Δ stack is stored ONCE as float32 (no float64 copies).
    _DFI_DIV_CMAPS = ['RdBu_r', 'coolwarm', 'seismic', 'bwr', 'PuOr_r', 'gray']

    def open_dframe_inspector(self):
        """Open the Δframe inspection window."""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Load a TIF stack first."); return
        self._dfi = getattr(self, '_dfi', {}) or {}
        # defaults (also used by headless tests that set self._dfi directly)
        self._dfi.setdefault('lag', 1)
        self._dfi.setdefault('mode', 'raw')
        self._dfi.setdefault('gauss_sigma', 0.0)
        self._dfi.setdefault('temporal_w', 1)
        self._dfi.setdefault('dc_remove', False)
        self._dfi.setdefault('motion_correct', False)
        self._dfi.setdefault('baseline_t0', None)
        self._dfi.setdefault('baseline_m', 5)
        self._dfi_build_window()

    # ---- compute engine (cached) --------------------------------------------
    def _dfi_compute(self):
        """Build the Δ stack (float32, marker range) + kymograph + arrival curve +
        per-frame global-shift estimate, from the current difference settings.
        Cached in self._dfi; display never recomputes this."""
        o = self._dfi
        start, end = self.get_analysis_frame_range()
        idx = np.arange(start, end + 1)
        T = int(idx.size)
        H, W = self.image_stack.shape[1:3]
        eps = 1e-6
        k = int(o['lag']); mode = o['mode']

        # working float32 stack over the range (motion-corr + DC + smoothing).
        work = np.empty((T, H, W), np.float32)
        ref = self.image_stack[start].astype(np.float32)
        shifts = np.zeros((T, 2), np.float32)
        win = None
        if o.get('motion_correct') or o.get('estimate_shift', True):
            try: win = cv2.createHanningWindow((W, H), cv2.CV_32F)
            except Exception: win = None
        prev = ref
        for i, fi in enumerate(idx):
            f = self.image_stack[fi].astype(np.float32)
            if i > 0:                                  # per-frame global shift vs previous
                try:
                    (dx, dy), _r = (cv2.phaseCorrelate(prev, f, win) if win is not None
                                    else cv2.phaseCorrelate(prev, f))
                    shifts[i] = (dx, dy)
                except Exception:
                    pass
            prev = f
            if o.get('motion_correct') and i > 0:      # rigidly register to the ref frame
                try:
                    (ax_, ay_), _r2 = (cv2.phaseCorrelate(ref, f, win) if win is not None
                                       else cv2.phaseCorrelate(ref, f))
                    M = np.float32([[1, 0, -ax_], [0, 1, -ay_]])
                    f = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
                except Exception:
                    pass
            if o.get('dc_remove'):
                f = f - float(f.mean())                # suppress global illumination drift
            work[i] = f
        if o.get('gauss_sigma', 0) and o['gauss_sigma'] > 0:
            for i in range(T):
                work[i] = ndimage.gaussian_filter(work[i], float(o['gauss_sigma']))
        if o.get('temporal_w', 1) and int(o['temporal_w']) > 1:
            wlen = int(o['temporal_w'])
            work = ndimage.uniform_filter1d(work, wlen, axis=0, mode='nearest')

        # difference (SAME definition the tracker uses for 'raw' consecutive diff)
        D = np.zeros((T, H, W), np.float32)
        valid = np.zeros(T, bool)
        if mode == 'baseline':
            t0 = o.get('baseline_t0'); t0 = start if t0 is None else int(t0)
            m = max(1, int(o.get('baseline_m', 5)))
            b0 = int(np.clip(t0 - start, 0, T - 1))
            base = work[b0:min(b0 + m, T)].mean(0)
            for i in range(T):
                D[i] = work[i] - base; valid[i] = True
        elif mode == 'rolling':
            m = max(1, int(o.get('baseline_m', 5)))
            for i in range(T):
                a, b = i - k - m, i - k
                if a >= 0 and b > a:
                    D[i] = work[i] - work[a:b].mean(0); valid[i] = True
        elif mode == 'normalized':
            for i in range(T):
                if i - k >= 0:
                    D[i] = (work[i] - work[i - k]) / (np.abs(work[i - k]) + eps); valid[i] = True
        else:  # raw
            for i in range(T):
                if i - k >= 0:
                    D[i] = work[i] - work[i - k]; valid[i] = True
        del work                                        # free the working copy

        # robust symmetric display limit from |Δ| (99th pct by default)
        av = np.abs(D[valid]) if valid.any() else np.abs(D)
        lim_auto = float(np.percentile(av, 99.0)) if av.size else 1.0
        if lim_auto <= 0:
            lim_auto = 1.0

        # kymograph + arrival curve from the SAME Δ, along the vessel ROI
        kymo = None; s_um = None; centers = None; arrival = np.zeros(T)
        if self.foreground_roi is not None and len(self.foreground_roi) >= 2:
            centers, normals, s_um = self._resample_centerline(
                self.foreground_roi, max(20, int(self._roi_len_px())))
            n_ax = centers.shape[0]
            xs = centers[:, 0]; ys = centers[:, 1]
            kymo = np.zeros((n_ax, T), np.float32)
            for i in range(T):
                kymo[:, i] = ndimage.map_coordinates(D[i], [ys, xs], order=1, mode='nearest')
            mask = np.zeros((H, W), np.uint8)
            pts = np.asarray(self.foreground_roi, np.int32)
            for j in range(len(pts) - 1):
                cv2.line(mask, tuple(pts[j]), tuple(pts[j + 1]), 255,
                         max(1, int(self.foreground_line_width)))
            mb = mask > 0
            for i in range(T):
                di = D[i]
                arrival[i] = float(np.sum(di[mb & (di > 0)]))   # summed positive Δ in ROI

        o.update(dict(D=D, idx=idx, start=start, valid=valid, lim_auto=lim_auto,
                      kymo=kymo, s_um=s_um, centers=centers, arrival=arrival,
                      shifts=shifts, T=T, cur=0))
        return True

    def _roi_len_px(self):
        p = np.asarray(self.foreground_roi, float)
        return float(np.sum(np.hypot(*(np.diff(p, axis=0).T)))) if len(p) >= 2 else 20.0

    # ---- display remap (cache -> signed [-1,1], gamma on magnitude) ----------
    def _dfi_signed(self, i):
        """Signed, gamma-corrected, sign-filtered display array in [-1,1] for
        frame i. sign(Δ)·(|Δ|/limit)^γ — gamma on MAGNITUDE, sign preserved."""
        o = self._dfi
        D = o['D'][i]
        lim = o.get('disp_limit') or o['lim_auto']
        g = float(o.get('gamma', 1.0))
        x = np.clip(D / max(lim, 1e-9), -1.0, 1.0)
        disp = np.sign(x) * (np.abs(x) ** g)
        sf = o.get('sign_filter', 'both')
        if sf == 'pos':
            disp = np.clip(disp, 0.0, 1.0)
        elif sf == 'neg':
            disp = np.clip(disp, -1.0, 0.0)
        return disp

    def _dfi_readout_text(self):
        """Sampling-adequacy readout: expected front displacement per lag and a
        sub-pixel warning + suggested minimum k."""
        o = self._dfi
        px = float(self.pixel_size); fps = float(self.frame_rate); k = int(o['lag'])
        v = o.get('expected_speed_mm_s')
        if v is None:
            pk = (self.vessel_results or {}).get('peak_speed') if self.vessel_results else None
            v = abs(pk['speed_mm_s']) if (pk and np.isfinite(pk.get('speed_mm_s', np.nan))) else None
        if not v or v <= 0:
            return "Δs readout: enter an expected speed (mm/s) or run tracking first."
        v_um_s = v * 1e3
        ds_um = v_um_s * k / fps
        ds_px = ds_um / px
        min_k = int(np.ceil(2.0 * px * fps / v_um_s))
        msg = (f"Δs = v·k/fps = {ds_um:.1f} µm = {ds_px:.2f} px/lag "
               f"(v={v:.3f} mm/s, k={k}, {fps:.0f} fps, {px:.1f} µm/px). ")
        if ds_px < 2.0:
            msg += (f"⚠ SUB-PIXEL (<2 px): dominated by shot noise. Use k≥{min_k} "
                    f"or 'baseline' mode. ")
        else:
            msg += f"OK (≥2 px). Min useful k≈{min_k}. "
        msg += "Larger k raises contrast but blurs temporal localization / smears once Δs exceeds front width."
        sh = o.get('shifts')
        if sh is not None and len(sh) > 1:
            med = float(np.median(np.hypot(sh[1:, 0], sh[1:, 1])))
            msg += (f"  | Global per-frame shift ≈ {med:.2f} px (motion artifact confound: "
                    f"breathing/heartbeat/drift can masquerade as a moving front).")
        return msg

    # ---- window + controls ---------------------------------------------------
    def _dfi_build_window(self):
        win = tk.Toplevel(self.root)
        win.title("Δframe Inspector (dye-transit QC)")
        win.geometry("1250x900")
        self._dfi['win'] = win
        self._dfi['playing'] = False
        self._dfi['job'] = None

        top = ttk.Frame(win); top.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        # row 1: difference computation
        r1 = ttk.Frame(top); r1.pack(fill=tk.X, pady=1)
        ttk.Label(r1, text="lag k:").pack(side=tk.LEFT)
        self._dfi_lag_var = tk.IntVar(value=int(self._dfi['lag']))
        ttk.Spinbox(r1, from_=1, to=50, width=4, textvariable=self._dfi_lag_var).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r1, text="mode:").pack(side=tk.LEFT)
        self._dfi_mode_var = tk.StringVar(value=self._dfi['mode'])
        ttk.Combobox(r1, textvariable=self._dfi_mode_var, width=13, state="readonly",
                     values=("raw", "normalized", "baseline", "rolling")).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r1, text="baseline t0/m:").pack(side=tk.LEFT)
        self._dfi_bt0_var = tk.StringVar(value="auto"); self._dfi_bm_var = tk.IntVar(value=5)
        ttk.Entry(r1, textvariable=self._dfi_bt0_var, width=5).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self._dfi_bm_var, width=4).pack(side=tk.LEFT, padx=(1, 8))
        ttk.Label(r1, text="gaussσ:").pack(side=tk.LEFT)
        self._dfi_gauss_var = tk.DoubleVar(value=0.0)
        ttk.Entry(r1, textvariable=self._dfi_gauss_var, width=4).pack(side=tk.LEFT, padx=(1, 6))
        ttk.Label(r1, text="temporal w:").pack(side=tk.LEFT)
        self._dfi_tw_var = tk.IntVar(value=1)
        ttk.Entry(r1, textvariable=self._dfi_tw_var, width=4).pack(side=tk.LEFT, padx=(1, 6))
        self._dfi_dc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text="DC remove", variable=self._dfi_dc_var).pack(side=tk.LEFT)
        self._dfi_mc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text="motion correct", variable=self._dfi_mc_var).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(r1, text="Recompute Δ", command=self._dfi_recompute).pack(side=tk.LEFT, padx=6)

        # row 2: display
        r2 = ttk.Frame(top); r2.pack(fill=tk.X, pady=1)
        ttk.Label(r2, text="cmap:").pack(side=tk.LEFT)
        self._dfi_cmap_var = tk.StringVar(value="RdBu_r")
        ttk.Combobox(r2, textvariable=self._dfi_cmap_var, width=9, state="readonly",
                     values=self._DFI_DIV_CMAPS).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r2, text="gamma:").pack(side=tk.LEFT)
        self._dfi_gamma_var = tk.DoubleVar(value=1.0)
        ttk.Entry(r2, textvariable=self._dfi_gamma_var, width=4).pack(side=tk.LEFT, padx=(1, 8))
        self._dfi_autolim_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="auto limit", variable=self._dfi_autolim_var).pack(side=tk.LEFT)
        ttk.Label(r2, text="pct:").pack(side=tk.LEFT)
        self._dfi_pct_var = tk.DoubleVar(value=99.0)
        ttk.Entry(r2, textvariable=self._dfi_pct_var, width=4).pack(side=tk.LEFT)
        ttk.Label(r2, text="manual lim:").pack(side=tk.LEFT)
        self._dfi_limit_var = tk.StringVar(value="")
        ttk.Entry(r2, textvariable=self._dfi_limit_var, width=7).pack(side=tk.LEFT, padx=(1, 8))
        ttk.Label(r2, text="sign:").pack(side=tk.LEFT)
        self._dfi_sign_var = tk.StringVar(value="both")
        ttk.Combobox(r2, textvariable=self._dfi_sign_var, width=6, state="readonly",
                     values=("both", "pos", "neg")).pack(side=tk.LEFT, padx=(2, 8))
        self._dfi_center_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="ROI", variable=self._dfi_center_var).pack(side=tk.LEFT)
        self._dfi_track_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="track", variable=self._dfi_track_var).pack(side=tk.LEFT)
        self._dfi_side_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text="side-by-side raw", variable=self._dfi_side_var).pack(side=tk.LEFT, padx=(4, 0))

        # row 3: playback + export
        r3 = ttk.Frame(top); r3.pack(fill=tk.X, pady=1)
        self._dfi_play_btn = ttk.Button(r3, text="▶ Play", command=self._dfi_toggle_play)
        self._dfi_play_btn.pack(side=tk.LEFT)
        ttk.Button(r3, text="◀", width=3, command=lambda: self._dfi_step(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(r3, text="▶|", width=3, command=lambda: self._dfi_step(1)).pack(side=tk.LEFT, padx=1)
        self._dfi_frame_var = tk.IntVar(value=0)
        self._dfi_slider = ttk.Scale(r3, from_=0, to=1, orient='horizontal',
                                     command=self._dfi_on_slider)
        self._dfi_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(r3, text="ms/frame:").pack(side=tk.LEFT)
        self._dfi_speed_var = tk.IntVar(value=80)
        ttk.Entry(r3, textvariable=self._dfi_speed_var, width=5).pack(side=tk.LEFT, padx=(1, 6))
        self._dfi_loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3, text="loop", variable=self._dfi_loop_var).pack(side=tk.LEFT)
        ttk.Label(r3, text="exp v(mm/s):").pack(side=tk.LEFT, padx=(8, 0))
        self._dfi_expv_var = tk.StringVar(value="")
        ttk.Entry(r3, textvariable=self._dfi_expv_var, width=6).pack(side=tk.LEFT, padx=(1, 6))
        ttk.Button(r3, text="Apply display", command=self._dfi_redraw).pack(side=tk.LEFT, padx=3)
        ttk.Button(r3, text="Export MP4…", command=self._dfi_export_mp4).pack(side=tk.LEFT, padx=3)

        # readout
        self._dfi_readout_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self._dfi_readout_var, foreground="#0055aa",
                  wraplength=1200, font=('TkDefaultFont', 8)).pack(fill=tk.X, pady=(2, 0))

        fig = Figure(figsize=(12, 7))
        self._dfi['fig'] = fig
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._dfi['canvas'] = canvas

        def _on_close():
            self._dfi['playing'] = False
            if self._dfi.get('job'):
                try: self.root.after_cancel(self._dfi['job'])
                except Exception: pass
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

        self._dfi_recompute()

    def _dfi_pull_opts(self):
        o = self._dfi
        o['lag'] = max(1, int(self._dfi_lag_var.get()))
        o['mode'] = self._dfi_mode_var.get()
        _bt = str(self._dfi_bt0_var.get()).strip().lower()
        o['baseline_t0'] = None if _bt in ('', 'auto') else int(float(_bt))
        o['baseline_m'] = max(1, int(self._dfi_bm_var.get()))
        o['gauss_sigma'] = max(0.0, float(self._dfi_gauss_var.get()))
        o['temporal_w'] = max(1, int(self._dfi_tw_var.get()))
        o['dc_remove'] = bool(self._dfi_dc_var.get())
        o['motion_correct'] = bool(self._dfi_mc_var.get())
        o['cmap'] = self._dfi_cmap_var.get()
        o['gamma'] = max(0.05, float(self._dfi_gamma_var.get()))
        o['auto_limit'] = bool(self._dfi_autolim_var.get())
        o['limit_pct'] = float(self._dfi_pct_var.get())
        _ml = str(self._dfi_limit_var.get()).strip()
        o['manual_limit'] = float(_ml) if _ml else None
        o['sign_filter'] = self._dfi_sign_var.get()
        o['show_center'] = bool(self._dfi_center_var.get())
        o['show_track'] = bool(self._dfi_track_var.get())
        o['side_raw'] = bool(self._dfi_side_var.get())
        _ev = str(self._dfi_expv_var.get()).strip()
        o['expected_speed_mm_s'] = float(_ev) if _ev else None

    def _dfi_recompute(self):
        self._dfi_pull_opts()
        self._dfi_compute()
        # resolve display limit (auto percentile of |Δ| or manual)
        o = self._dfi
        if o.get('manual_limit'):
            o['disp_limit'] = o['manual_limit']
        elif o.get('auto_limit', True):
            av = np.abs(o['D'][o['valid']]) if o['valid'].any() else np.abs(o['D'])
            o['disp_limit'] = float(np.percentile(av, o.get('limit_pct', 99.0))) if av.size else o['lim_auto']
        else:
            o['disp_limit'] = o['lim_auto']
        try:
            self._dfi_slider.configure(to=max(0, o['T'] - 1))
        except Exception:
            pass
        o['cur'] = int(np.clip(o.get('cur', 0), 0, o['T'] - 1))
        self._dfi_redraw()

    def _dfi_redraw(self):
        """Draw the current frame + overlays + kymograph + arrival curve. Only
        remaps the cached Δ (no recompute) unless called via _dfi_recompute."""
        import matplotlib.cm as _cm
        # display-only opts can change without recompute
        try:
            self._dfi_pull_opts()
            o = self._dfi
            if o.get('manual_limit'):
                o['disp_limit'] = o['manual_limit']
            elif o.get('auto_limit', True):
                av = np.abs(o['D'][o['valid']]) if o['valid'].any() else np.abs(o['D'])
                o['disp_limit'] = float(np.percentile(av, o.get('limit_pct', 99.0))) if av.size else o['lim_auto']
            else:
                o['disp_limit'] = o['lim_auto']
        except Exception:
            o = self._dfi
        fig = o['fig']; fig.clear()
        i = int(np.clip(o.get('cur', 0), 0, o['T'] - 1))
        cmap = o.get('cmap', 'RdBu_r')
        side = o.get('side_raw', False)
        gs = fig.add_gridspec(3, 2 if side else 1, height_ratios=[3, 1.1, 0.9],
                              hspace=0.35, wspace=0.15)
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(self._dfi_signed(i), cmap=cmap, vmin=-1, vmax=1)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        if side:
            ax_raw = fig.add_subplot(gs[0, 1])
            raw = self.image_stack[o['idx'][i]].astype(np.float32)
            lo, hi = np.percentile(raw, [1, 99.5])
            ax_raw.imshow(np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1), cmap='gray')
            ax_raw.set_xticks([]); ax_raw.set_yticks([]); ax_raw.set_title('raw', fontsize=8)
        # overlays
        if o.get('show_center', True) and o.get('centers') is not None:
            c = o['centers']
            ax_img.plot(c[:, 0], c[:, 1], '-', color='#00cc00', lw=1.0, alpha=0.7)
        self._dfi_overlay_track(ax_img, i)
        # burn-in text
        fr = int(o['idx'][i]); tsec = fr / self.frame_rate
        ax_img.text(0.02, 0.98, f"f{fr}  t={tsec:.2f}s  k={o['lag']}  {o['mode']}  γ={o['gamma']:.1f}",
                    transform=ax_img.transAxes, color='k', fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
        # scale bar
        self._dfi_scalebar(ax_img)

        # kymograph
        ax_k = fig.add_subplot(gs[1, :] if side else gs[1, 0])
        if o.get('kymo') is not None:
            kk = o['kymo']; s_um = o['s_um']; tt = o['idx'] / self.frame_rate
            klim = o['disp_limit']
            ax_k.imshow(kk, aspect='auto', cmap=cmap, origin='lower', vmin=-klim, vmax=klim,
                        extent=[tt[0], tt[-1], s_um[0], s_um[-1]])
            ax_k.axvline(tsec, color='k', ls='--', lw=1.2)     # time cursor (playback-synced)
            ax_k.set_ylabel('pos (µm)', fontsize=8)
        ax_k.set_xlabel('time (s)', fontsize=8); ax_k.set_title('Δ kymograph', fontsize=8)

        # arrival curve
        ax_a = fig.add_subplot(gs[2, :] if side else gs[2, 0])
        arr = o.get('arrival'); tt = o['idx'] / self.frame_rate
        if arr is not None:
            ax_a.plot(tt, arr, '-', color='#0072B2', lw=1.2)
            ax_a.axvline(tsec, color='k', ls='--', lw=1.2)
            ax_a.fill_between(tt, arr, alpha=0.15, color='#0072B2')
        ax_a.set_xlabel('time (s)', fontsize=8); ax_a.set_ylabel('Σ +Δ in ROI', fontsize=8)
        ax_a.set_title('dye-arrival curve (summed positive Δ in vessel)', fontsize=8)

        try: fig.tight_layout()
        except Exception: pass
        o['canvas'].draw()
        try:
            self._dfi_readout_var.set(self._dfi_readout_text())
            self._dfi_frame_var.set(i)
            self._dfi_slider.set(i)
        except Exception:
            pass

    def _dfi_overlay_track(self, ax, i):
        """Overlay the tracked front position (from a previous flow-speed run) on
        the vessel at this frame — the direct visual check of the tracker."""
        o = self._dfi
        if not o.get('show_track', True) or o.get('centers') is None:
            return
        pk = (self.vessel_results or {}).get('peak_speed') if self.vessel_results else None
        if not pk:
            return
        # tracker steps are diff steps over its own kymograph; map by frame index.
        # its t_mid_s are in ABSOLUTE seconds -> find the step nearest this frame.
        try:
            tsec = o['idx'][i] / self.frame_rate
            tm = pk['t_mid_s']
            j = int(np.argmin(np.abs(tm - tsec)))
            if abs(tm[j] - tsec) > 1.5 / self.frame_rate or not pk['valid'][j]:
                return
            s_hit = pk['peak_s_um'][j]
            c = o['centers']; s_um = o['s_um']
            xi = np.interp(s_hit, s_um, c[:, 0]); yi = np.interp(s_hit, s_um, c[:, 1])
            observed = (pk['step_flag'][j] == 'observed')
            ax.plot([xi], [yi], 'o', mfc=('yellow' if observed else 'none'),
                    mec='red', mew=1.5, markersize=11, zorder=6)
        except Exception:
            pass

    def _dfi_scalebar(self, ax):
        bar_um = 200.0; bar_px = bar_um / self.pixel_size
        x0, x1 = ax.get_xlim(); y1, y0 = ax.get_ylim()
        bx = x1 - 0.06 * (x1 - x0) - bar_px; by = y0 - 0.08 * (y0 - y1)
        ax.plot([bx, bx + bar_px], [by, by], '-', color='k', lw=3)
        ax.text(bx + bar_px / 2, by - 0.02 * (y0 - y1), f"{bar_um:.0f} µm",
                ha='center', va='bottom', fontsize=7)

    # ---- playback (mirrors the main-window after-loop pattern) ---------------
    def _dfi_toggle_play(self):
        if self._dfi.get('playing'):
            self._dfi['playing'] = False
            self._dfi_play_btn.config(text="▶ Play")
            if self._dfi.get('job'):
                try: self.root.after_cancel(self._dfi['job'])
                except Exception: pass
                self._dfi['job'] = None
        else:
            self._dfi['playing'] = True
            self._dfi_play_btn.config(text="⏸ Pause")
            self._dfi_play_next()

    def _dfi_play_next(self):
        if not self._dfi.get('playing'):
            return
        o = self._dfi
        nxt = o['cur'] + 1
        if nxt >= o['T']:
            if not self._dfi_loop_var.get():
                self._dfi_toggle_play(); return
            nxt = 0
        o['cur'] = nxt
        self._dfi_redraw()
        o['job'] = self.root.after(max(10, int(self._dfi_speed_var.get())), self._dfi_play_next)

    def _dfi_step(self, d):
        o = self._dfi
        o['cur'] = int(np.clip(o['cur'] + d, 0, o['T'] - 1))
        self._dfi_redraw()

    def _dfi_on_slider(self, val):
        try:
            i = int(float(val))
        except Exception:
            return
        if i != self._dfi.get('cur'):
            self._dfi['cur'] = i
            self._dfi_redraw()

    # ---- MP4 export ----------------------------------------------------------
    def _dfi_composite_rgb(self, i, side, arrstrip):
        """Compose an RGB uint8 frame for MP4: Δframe (optionally beside raw and
        above an arrival strip) with burned-in settings + scale bar."""
        import matplotlib.cm as _cm
        o = self._dfi
        cmap = _cm.get_cmap(o.get('cmap', 'RdBu_r'))
        disp = self._dfi_signed(i)
        rgb = (cmap((disp + 1.0) / 2.0)[:, :, :3] * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb[:, :, ::-1])          # to BGR for cv2 drawing
        H, W = rgb.shape[:2]
        if side:
            raw = self.image_stack[o['idx'][i]].astype(np.float32)
            lo, hi = np.percentile(raw, [1, 99.5])
            r8 = (np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            rraw = cv2.cvtColor(r8, cv2.COLOR_GRAY2BGR)
            rgb = np.hstack([rraw, rgb])
        # burn-in
        fr = int(o['idx'][i]); tsec = fr / self.frame_rate
        lim = o.get('disp_limit') or o['lim_auto']
        txt = f"f{fr} t={tsec:.2f}s k={o['lag']} {o['mode']} g={o['gamma']:.1f} lim={lim:.3g}"
        cv2.putText(rgb, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        bar_px = int(round(200.0 / self.pixel_size))
        x1 = rgb.shape[1] - 12; x0 = x1 - bar_px; yb = rgb.shape[0] - 12
        cv2.line(rgb, (x0, yb), (x1, yb), (255, 255, 255), 3)
        cv2.putText(rgb, "200um", (x0, yb - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if arrstrip and o.get('arrival') is not None:
            arr = o['arrival']; strip_h = 90; sw = rgb.shape[1]
            strip = np.zeros((strip_h, sw, 3), np.uint8)
            a = arr - np.nanmin(arr); a = a / (np.nanmax(a) + 1e-9)
            xs = (np.arange(o['T']) / max(1, o['T'] - 1) * (sw - 1)).astype(int)
            ys = (strip_h - 6 - a * (strip_h - 12)).astype(int)
            for j in range(1, o['T']):
                cv2.line(strip, (xs[j - 1], ys[j - 1]), (xs[j], ys[j]), (255, 150, 0), 1, cv2.LINE_AA)
            cx = int(i / max(1, o['T'] - 1) * (sw - 1))
            cv2.line(strip, (cx, 0), (cx, strip_h), (0, 0, 255), 1)
            cv2.putText(strip, "arrival", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            rgb = np.vstack([rgb, strip])
        # cv2 VideoWriter needs even dims
        H2, W2 = rgb.shape[:2]
        if H2 % 2 or W2 % 2:
            rgb = rgb[:H2 - (H2 % 2), :W2 - (W2 % 2)]
        return rgb

    def _dfi_export_mp4(self):
        o = self._dfi
        if 'D' not in o:
            return
        stem = os.path.splitext(os.path.basename(self.tif_file_path or 'stack'))[0] \
            if getattr(self, 'tif_file_path', None) else 'stack'
        default = f"{stem}_dframe_k{o['lag']}_{o['mode']}_f{o['idx'][0]}-{o['idx'][-1]}"
        path = filedialog.asksaveasfilename(title="Export Δframe MP4", initialfile=default,
                                            defaultextension=".mp4",
                                            filetypes=[("MP4", "*.mp4")])
        if not path:
            return
        side = bool(self._dfi_side_var.get()); arrstrip = True
        try:
            f0 = self._dfi_composite_rgb(0, side, arrstrip)
            H, W = f0.shape[:2]
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 float(self.frame_rate), (W, H))
            if not vw.isOpened():
                messagebox.showerror("Export", "Failed to open video writer."); return
            pw = tk.Toplevel(self.root); pw.title("Exporting Δframe MP4…")
            pv = tk.DoubleVar(); ttk.Progressbar(pw, variable=pv, maximum=100).pack(fill=tk.X, padx=20, pady=15)
            lab = ttk.Label(pw, text=""); lab.pack(); pw.update()
            for i in range(o['T']):
                fr = self._dfi_composite_rgb(i, side, arrstrip)
                if fr.shape[:2] != (H, W):
                    fr = cv2.resize(fr, (W, H))
                vw.write(fr)
                if i % 5 == 0:
                    pv.set(100 * i / o['T']); lab.config(text=f"frame {i+1}/{o['T']}"); pw.update()
            vw.release(); pw.destroy()
            self._log_console(f"[dframe] exported {path} ({o['T']} frames)")
            messagebox.showinfo("Export", f"Saved Δframe MP4:\n{path}\n{o['T']} frames @ {self.frame_rate:.0f} fps")
        except Exception as e:
            try: pw.destroy()
            except Exception: pass
            messagebox.showerror("Export failed", str(e))

    def analyze_flow_velocity(self):
        """PANEL 2: axial flow velocity from kymograph streak slope.

        Builds an axis-aligned kymograph, extracts streak orientation via the
        structure tensor, and reports a signed velocity (direction-resolved).
        Shows the kymograph, the orientation-derived velocity field, and the
        velocity-over-time trace.
        """
        if self.foreground_roi is None:
            messagebox.showwarning("Warning", "Draw the vessel centerline (Foreground ROI) first.")
            return
        self.update_ear_params()

        kymo, s_um, t_s = self._build_axial_kymograph()
        if kymo is None:
            messagebox.showerror("Error", "Could not build kymograph.")
            return
        vel = self._velocity_from_kymo(kymo, s_um, t_s)

        # cache velocity for the flow-rate panel
        if self.vessel_results is None:
            self.vessel_results = {}
        self.vessel_results['velocity'] = vel
        self.vessel_results['kymo'] = kymo
        self.vessel_results['kymo_s_um'] = s_um
        self.vessel_results['kymo_t_s'] = t_s

        win = tk.Toplevel(self.root)
        win.title("Mouse Ear - Flow Velocity (kymograph slope)")
        win.geometry("1300x850")
        fig = Figure(figsize=(14, 9))

        # (a) kymograph
        ax1 = fig.add_subplot(221)
        vmin, vmax = np.percentile(kymo, [2, 98])
        im1 = ax1.imshow(kymo, aspect='auto', cmap='gray', origin='lower',
                         vmin=vmin, vmax=vmax,
                         extent=[t_s[0], t_s[-1], s_um[0], s_um[-1]])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position along vessel (µm)')
        ax1.set_title('Axial kymograph\n(slanted streaks = moving blood)')
        fig.colorbar(im1, ax=ax1, label='Intensity')

        # (b) velocity field from orientation
        ax2 = fig.add_subplot(222)
        vfield = vel['v_field_mm_s'].copy()
        # mask low-coherence pixels for clarity
        vfield[vel['coherence'] < 0.3] = np.nan
        vabs = np.nanpercentile(np.abs(vfield), 95) if np.isfinite(vfield).any() else 1.0
        im2 = ax2.imshow(vfield, aspect='auto', cmap='RdBu_r', origin='lower',
                         vmin=-vabs, vmax=vabs,
                         extent=[t_s[0], t_s[-1], s_um[0], s_um[-1]])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position along vessel (µm)')
        ax2.set_title('Local velocity (signed)\nblue/red = opposite directions')
        fig.colorbar(im2, ax=ax2, label='Velocity (mm/s)')

        # (c) velocity over time
        ax3 = fig.add_subplot(223)
        ax3.plot(t_s, vel['v_col_mm_s'], 'o-', color='darkred', markersize=3)
        if np.isfinite(vel['v_global_mm_s']):
            ax3.axhline(vel['v_global_mm_s'], color='blue', linestyle='--',
                        label=f"median {vel['v_global_mm_s']:.3f} mm/s")
        ax3.axhline(0, color='gray', linewidth=0.8)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Axial velocity (mm/s)')
        ax3.set_title('Velocity over time (sign = direction)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # (d) summary
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        vg = vel['v_global_mm_s']
        limited = vel.get('sampling_limited', False)
        txt = "FLOW VELOCITY SUMMARY\n" + "=" * 30 + "\n\n"
        if np.isfinite(vg):
            direction = "forward (+s)" if vg > 0 else "reverse (−s)"
            qualifier = " (LOWER BOUND)" if limited else ""
            txt += f"Median speed:   {abs(vg):.3f} mm/s{qualifier}\n"
            txt += f"  = {abs(vg)*1000:.1f} µm/s\n"
            txt += f"Direction:      {direction}\n"
            txt += f"Robust scatter: ±{vel['v_global_mad']:.3f} mm/s (MAD)\n"
            vt = vel['v_col_mm_s']
            vt = vt[np.isfinite(vt)]
            if len(vt) > 1:
                txt += f"Temporal range: {np.min(vt):.3f} … {np.max(vt):.3f} mm/s\n"
        else:
            txt += "Streaks too weak/short to orient reliably.\n"
            txt += "Try: longer vessel segment, brighter\ncontrast, or more frames.\n"
        txt += "\nMethod: structure-tensor eigenvector\n"
        txt += "orientation of kymograph streaks.\n"
        txt += "Slope = position/time = velocity.\n"
        txt += "Amplitude-independent, direction-resolved.\n\n"
        # Sampling ceiling (validated): reliable up to |slope|~2
        ceil = vel.get('v_resolvable_ceiling_mm_s', np.nan)
        txt += "SAMPLING LIMITS\n"
        txt += f"  frame rate: {self.frame_rate:.1f} Hz\n"
        txt += f"  streak-resolvable speed: |v| < {ceil:.2f} mm/s\n"
        txt += "  (faster flow makes near-vertical\n   streaks -> under-sampled, reads LOW).\n"
        txt += "  Cardiac pulsatility (8-12 Hz) is at/\n  above Nyquist -> aliased, not recovered.\n"
        if vel.get('sampling_limited', False):
            txt += "\n*** WARNING: estimate is near the\n"
            txt += "    resolvable ceiling. True velocity may\n"
            txt += "    be HIGHER. Use finer pixels or a\n"
            txt += "    faster frame rate to confirm. ***"
        ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, fontsize=9,
                 va='top', fontfamily='monospace',
                 color=('darkred' if vel.get('sampling_limited', False) else 'black'))

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def analyze_flow_rate_and_regime(self):
        """PANEL 3: volumetric flow rate Q = v_mean * A, plus flow regime.

        Combines morphometry (area) with velocity to compute Q, then computes
        Reynolds and Womersley numbers to CONFIRM the low-Re, quasi-steady
        regime that justifies the Poiseuille correction and circular-lumen area.
        """
        self.update_ear_params()
        # ensure both morphometry and velocity are available
        if self.vessel_results is None or 'diam_um' not in self.vessel_results:
            res = self._extract_vessel_morphometry()
            if res is None:
                return
        if 'velocity' not in self.vessel_results:
            messagebox.showinfo("Info",
                                "Run '2. Flow Velocity' first so a velocity estimate exists.")
            return

        res = self.vessel_results
        vel = res['velocity']
        diam = res['diam_um']
        area = res['area_um2']
        flags = res['flags']
        ok = np.isfinite(diam) & (flags == 'ok')
        if ok.sum() == 0:
            messagebox.showwarning("Warning", "No reliable diameter cross-sections.")
            return

        mean_d_um = float(np.mean(diam[ok]))
        mean_a_um2 = float(np.nanmean(area[ok]))
        v_peak_mm_s = vel['v_global_mm_s']
        if not np.isfinite(v_peak_mm_s):
            messagebox.showwarning("Warning", "Velocity estimate is not finite.")
            return

        # --- ask whether the measured velocity is peak (centerline) or mean ---
        use_poiseuille = messagebox.askyesno(
            "Velocity interpretation",
            "Treat the measured velocity as PEAK (centerline) velocity and apply "
            "the Poiseuille correction v_mean = v_peak / 2?\n\n"
            "Yes = peak (recommended for a centerline kymograph)\n"
            "No  = already a cross-section mean")
        v_peak = abs(v_peak_mm_s)
        v_mean = v_peak / 2.0 if use_poiseuille else v_peak  # mm/s

        # --- volumetric flow rate  Q = v_mean * A ---
        # Unit conversion done transparently and verified by two routes:
        #   A: µm² -> m²  (×1e-12)
        #   v: mm/s -> m/s (×1e-3)
        #   1 m³ = 1e3 L = 1e3 · 1e9 nL = 1e12 nL
        A_m2 = mean_a_um2 * 1e-12
        v_mean_m_s = v_mean * 1e-3
        Q_m3_s = v_mean_m_s * A_m2
        NL_PER_M3 = 1e12
        Q_nl_s = Q_m3_s * NL_PER_M3          # nL/s
        Q_nl_min = Q_nl_s * 60.0             # nL/min

        # --- dimensionless regime numbers ---
        rho = self.blood_density
        mu = self.blood_viscosity
        D_m = mean_d_um * 1e-6
        R_m = D_m / 2.0
        Re = rho * v_mean_m_s * D_m / mu
        # Womersley uses an angular frequency; at 20 Hz cardiac is aliased, but
        # we report alpha for a nominal mouse HR to characterize the regime the
        # vessel would sit in (documented as nominal, not measured here).
        nominal_hr_hz = 10.0
        omega = 2 * np.pi * nominal_hr_hz
        alpha = R_m * np.sqrt(omega * rho / mu)

        win = tk.Toplevel(self.root)
        win.title("Mouse Ear - Flow Rate & Regime")
        win.geometry("1200x800")
        fig = Figure(figsize=(13, 8))

        # (a) area profile
        ax1 = fig.add_subplot(221)
        s_um = res['s_um']
        ax1.plot(s_um[ok], area[ok], 'o-', color='teal')
        ax1.axhline(mean_a_um2, color='red', linestyle='--',
                    label=f'mean {mean_a_um2:.1f} µm²')
        ax1.set_xlabel('Position (µm)')
        ax1.set_ylabel('Cross-sectional area (µm²)')
        ax1.set_title('Area along vessel')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # (b) velocity profile assumption illustration (Poiseuille)
        ax2 = fig.add_subplot(222)
        r = np.linspace(-1, 1, 100)
        prof = v_peak * (1 - r ** 2) if use_poiseuille else np.full_like(r, v_mean)
        ax2.plot(prof, r * mean_d_um / 2, color='darkorange', linewidth=2)
        ax2.fill_betweenx(r * mean_d_um / 2, 0, prof, color='orange', alpha=0.2)
        ax2.axvline(v_mean, color='blue', linestyle='--',
                    label=f'v_mean {v_mean:.3f} mm/s')
        ax2.set_xlabel('Velocity (mm/s)')
        ax2.set_ylabel('Radial position (µm)')
        ax2.set_title('Assumed velocity profile' +
                      (' (Poiseuille)' if use_poiseuille else ' (plug)'))
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # (c) regime bar
        ax3 = fig.add_subplot(223)
        labels = ['Reynolds', 'Womersley']
        vals = [Re, alpha]
        bars = ax3.bar(labels, vals, color=['slateblue', 'indianred'])
        ax3.axhline(1.0, color='k', linestyle='--', linewidth=1, label='unity')
        ax3.set_yscale('log')
        ax3.set_ylabel('Dimensionless value (log)')
        ax3.set_title('Flow regime (both ≪ 1 → viscous, quasi-steady)')
        for b, v in zip(bars, vals):
            ax3.text(b.get_x() + b.get_width() / 2, v,
                     f'{v:.2e}', ha='center', va='bottom', fontsize=9)
        ax3.legend(fontsize=8)

        # (d) summary
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        vel_limited = vel.get('sampling_limited', False)
        lb = " (LOWER BOUND)" if vel_limited else ""
        txt = "VOLUMETRIC FLOW & REGIME\n" + "=" * 32 + "\n\n"
        txt += f"Mean diameter:   {mean_d_um:.2f} µm\n"
        txt += f"Mean area:       {mean_a_um2:.1f} µm²\n"
        txt += f"Peak velocity:   {v_peak:.3f} mm/s{lb}\n"
        txt += f"Mean velocity:   {v_mean:.3f} mm/s"
        txt += "  (Poiseuille /2)\n" if use_poiseuille else "  (as measured)\n"
        txt += "\n"
        txt += f"Flow rate Q:     {Q_nl_s:.4f} nL/s{lb}\n"
        txt += f"                 {Q_nl_min:.3f} nL/min\n"
        if vel_limited:
            txt += "  (velocity is sampling-limited, so Q\n   is a LOWER BOUND — see velocity panel)\n"
        txt += "\n"
        txt += f"Reynolds Re:     {Re:.2e}\n"
        txt += f"Womersley α:     {alpha:.3f}  (@ {nominal_hr_hz:.0f} Hz nominal)\n"
        txt += "\n"
        if Re < 1 and alpha < 1:
            txt += "Re≪1 and α≪1: creeping, quasi-steady\n"
            txt += "flow. Poiseuille & circular-lumen\n"
            txt += "assumptions are self-consistent. ✓\n"
        else:
            txt += "Regime not clearly viscous/quasi-steady;\n"
            txt += "re-examine Poiseuille assumption.\n"
        txt += "\nInputs: µ={:.2f} cP, ρ={:.0f} kg/m³".format(
            mu * 1e3, rho)
        ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, fontsize=9,
                 va='top', fontfamily='monospace')

        # stash scalars for CSV export
        res['flow_summary'] = {
            'mean_diameter_um': mean_d_um,
            'mean_area_um2': mean_a_um2,
            'v_peak_mm_s': v_peak,
            'v_mean_mm_s': v_mean,
            'poiseuille_correction': use_poiseuille,
            'Q_nL_per_s': Q_nl_s,
            'Q_nL_per_min': Q_nl_min,
            'Reynolds': Re,
            'Womersley_nominal': alpha,
        }

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def analyze_vasomotion(self):
        """PANEL 4: low-frequency diameter / intensity oscillation (vasomotion).

        At 20 Hz the cardiac band is aliased, but vasomotion (~0.05–0.3 Hz) is
        richly sampled. We track a width-averaged intensity (proxy for local
        blood content / diameter) over time and analyse its low-frequency
        spectrum. This is a genuinely well-sampled physiological readout here.
        """
        if self.foreground_roi is None:
            messagebox.showwarning("Warning", "Draw the vessel centerline (Foreground ROI) first.")
            return
        self.update_ear_params()

        kymo, s_um, t_s = self._build_axial_kymograph()
        if kymo is None:
            messagebox.showerror("Error", "Could not build kymograph.")
            return

        # spatially averaged signal over the vessel (mean blood content proxy)
        signal = np.mean(kymo, axis=0)
        signal = signal - np.mean(signal)
        n = len(signal)
        if n < 16:
            messagebox.showwarning("Warning", "Not enough frames for spectral analysis.")
            return

        # detrend (remove slow bleaching drift) with a low-order polynomial
        tt = np.arange(n)
        try:
            coeffs = np.polyfit(tt, signal, 2)
            trend = np.polyval(coeffs, tt)
            detr = signal - trend
        except Exception:
            detr = signal

        # spectrum
        fftv = np.abs(fft(detr))
        freqs = fftfreq(n, 1.0 / self.frame_rate)
        pos = freqs > 0
        f = freqs[pos]
        p = fftv[pos]

        # vasomotion band 0.03–0.5 Hz
        band = (f >= 0.03) & (f <= 0.5)
        dom_f = np.nan
        if band.any() and p[band].max() > 0:
            dom_f = f[band][np.argmax(p[band])]

        win = tk.Toplevel(self.root)
        win.title("Mouse Ear - Vasomotion (low-frequency dynamics)")
        win.geometry("1200x750")
        fig = Figure(figsize=(13, 7))

        # (a) time series
        ax1 = fig.add_subplot(221)
        ax1.plot(t_s, signal, color='gray', alpha=0.6, label='raw (mean-sub)')
        ax1.plot(t_s, detr, color='crimson', linewidth=1.2, label='detrended')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Mean blood-content proxy (a.u.)')
        ax1.set_title('Vessel intensity over time')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # (b) spectrum (low-freq zoom)
        ax2 = fig.add_subplot(222)
        ax2.plot(f, p, color='navy')
        ax2.axvspan(0.03, 0.5, color='orange', alpha=0.15, label='vasomotion band')
        if np.isfinite(dom_f):
            ax2.axvline(dom_f, color='red', linestyle='--',
                        label=f'peak {dom_f:.3f} Hz')
        ax2.set_xlim(0, min(2.0, self.frame_rate / 2))
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power')
        ax2.set_title('Low-frequency spectrum')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # (c) spatiotemporal map (kymo of intensity fluctuation)
        ax3 = fig.add_subplot(223)
        fluct = kymo - np.mean(kymo, axis=1, keepdims=True)
        vlim = np.percentile(np.abs(fluct), 95)
        im3 = ax3.imshow(fluct, aspect='auto', cmap='seismic', origin='lower',
                         vmin=-vlim, vmax=vlim,
                         extent=[t_s[0], t_s[-1], s_um[0], s_um[-1]])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position (µm)')
        ax3.set_title('Intensity fluctuation (spatial coordination)')
        fig.colorbar(im3, ax=ax3, label='ΔIntensity')

        # (d) summary
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        rec_dur = t_s[-1] - t_s[0]
        txt = "VASOMOTION / LOW-FREQ SUMMARY\n" + "=" * 33 + "\n\n"
        txt += f"Recording length: {rec_dur:.1f} s ({n} frames)\n"
        txt += f"Frame rate:       {self.frame_rate:.1f} Hz\n"
        txt += f"Freq resolution:  {1.0/rec_dur:.4f} Hz\n\n"
        if np.isfinite(dom_f):
            txt += f"Dominant vasomotion: {dom_f:.3f} Hz\n"
            txt += f"  period ≈ {1.0/dom_f:.1f} s\n"
        else:
            txt += "No clear vasomotion peak in 0.03–0.5 Hz.\n"
        txt += "\nWHY THIS BAND IS VALID AT 20 Hz:\n"
        txt += "  vasomotion (~0.1 Hz) is far below\n"
        txt += "  Nyquist (10 Hz) → well sampled.\n"
        txt += "  Cardiac pulsatility (8–12 Hz) is at/\n"
        txt += "  above Nyquist → aliased, NOT analysed.\n"
        txt += "\nThe spatiotemporal map reveals whether\n"
        txt += "oscillations propagate along the vessel —\n"
        txt += "only visible with volumetric simultaneous\n"
        txt += "capture."
        ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, fontsize=9,
                 va='top', fontfamily='monospace')

        if self.vessel_results is None:
            self.vessel_results = {}
        self.vessel_results['vasomotion'] = {
            'dominant_freq_hz': dom_f,
            'signal': signal,
            't_s': t_s,
        }

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_summary_figure(self):
        """PANEL 5: publication-style multi-panel summary combining structure,
        diameter, velocity, and flow into a single figure suitable as a main
        figure draft. Runs morphometry (and reuses velocity if available)."""
        res = self._extract_vessel_morphometry()
        if res is None:
            return
        has_vel = self.vessel_results is not None and 'velocity' in self.vessel_results

        diam = res['diam_um']
        area = res['area_um2']
        s_um = res['s_um']
        flags = res['flags']
        ok = np.isfinite(diam) & (flags == 'ok')

        win = tk.Toplevel(self.root)
        win.title("Mouse Ear - Summary Figure (draft main figure)")
        win.geometry("1400x900")
        fig = Figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # (a) structure with diameter color
        axa = fig.add_subplot(gs[0, 0])
        img = res['mean_img']
        vmin, vmax = np.percentile(img, [1, 99])
        axa.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        centers = res['centers']
        normals = res['normals']
        dmax = np.nanmax(diam[ok]) if ok.any() else 1.0
        dmin = np.nanmin(diam[ok]) if ok.any() else 0.0
        for k in range(len(centers)):
            if not np.isfinite(diam[k]) or not ok[k]:
                continue
            frac = (diam[k] - dmin) / (dmax - dmin + 1e-9)
            col = plt.cm.plasma(frac)
            half_px = (diam[k] / self.pixel_size) / 2.0
            c, nrm = centers[k], normals[k]
            p0 = c - half_px * nrm
            p1 = c + half_px * nrm
            axa.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, linewidth=2)
        axa.set_title('(a) Vessel structure\ncolor = diameter')
        axa.axis('off')

        # (b) diameter profile
        axb = fig.add_subplot(gs[0, 1])
        if ok.any():
            axb.plot(s_um[ok], diam[ok], 'o-', color='seagreen', markersize=4)
        axb.set_xlabel('Position (µm)')
        axb.set_ylabel('Diameter (µm)')
        axb.set_title('(b) Diameter profile')
        axb.grid(True, alpha=0.3)

        # (c) diameter histogram
        axc = fig.add_subplot(gs[0, 2])
        if ok.any():
            axc.hist(diam[ok], bins=max(5, ok.sum() // 2), color='mediumseagreen',
                     alpha=0.8, edgecolor='black')
            axc.axvline(np.mean(diam[ok]), color='red',
                        label=f'{np.mean(diam[ok]):.1f} µm')
            axc.legend(fontsize=8)
        axc.set_xlabel('Diameter (µm)')
        axc.set_ylabel('Count')
        axc.set_title('(c) Diameter distribution')
        axc.grid(True, alpha=0.3)

        # (d) kymograph or placeholder
        axd = fig.add_subplot(gs[1, 0])
        if has_vel and 'kymo' in self.vessel_results:
            kymo = self.vessel_results['kymo']
            ks = self.vessel_results['kymo_s_um']
            kt = self.vessel_results['kymo_t_s']
            vmn, vmx = np.percentile(kymo, [2, 98])
            axd.imshow(kymo, aspect='auto', cmap='gray', origin='lower',
                       vmin=vmn, vmax=vmx, extent=[kt[0], kt[-1], ks[0], ks[-1]])
            axd.set_xlabel('Time (s)')
            axd.set_ylabel('Position (µm)')
            axd.set_title('(d) Axial kymograph')
        else:
            axd.text(0.5, 0.5, "Run 'Flow Velocity'\nfor kymograph",
                     ha='center', va='center', transform=axd.transAxes)
            axd.set_title('(d) Kymograph')
            axd.axis('off')

        # (e) velocity over time
        axe = fig.add_subplot(gs[1, 1])
        if has_vel:
            vel = self.vessel_results['velocity']
            kt = self.vessel_results['kymo_t_s']
            axe.plot(kt, vel['v_col_mm_s'], 'o-', color='darkred', markersize=3)
            axe.axhline(0, color='gray', linewidth=0.8)
            if np.isfinite(vel['v_global_mm_s']):
                axe.axhline(vel['v_global_mm_s'], color='blue', linestyle='--',
                            label=f"{vel['v_global_mm_s']:.3f} mm/s")
                axe.legend(fontsize=8)
            axe.set_xlabel('Time (s)')
            axe.set_ylabel('Velocity (mm/s)')
            axe.set_title('(e) Axial velocity (signed)')
            axe.grid(True, alpha=0.3)
        else:
            axe.text(0.5, 0.5, "Run 'Flow Velocity'", ha='center', va='center',
                     transform=axe.transAxes)
            axe.set_title('(e) Velocity')
            axe.axis('off')

        # (f) key numbers
        axf = fig.add_subplot(gs[1, 2])
        axf.axis('off')
        txt = "KEY QUANTITIES\n" + "=" * 22 + "\n\n"
        if ok.any():
            txt += f"Diameter:  {np.mean(diam[ok]):.1f} ± {np.std(diam[ok]):.1f} µm\n"
            txt += f"Area:      {np.nanmean(area[ok]):.1f} µm²\n"
            if ok.sum() > 1:
                s_ok = s_um[ok]; a_ok = area[ok]
                o = np.argsort(s_ok)
                vol = _trapz(a_ok[o], s_ok[o])
                txt += f"Volume:    {vol*1e-9:.3f} nL\n"
        if has_vel:
            vel = self.vessel_results['velocity']
            if np.isfinite(vel['v_global_mm_s']):
                txt += f"Velocity:  {vel['v_global_mm_s']:.3f} mm/s\n"
        if self.vessel_results and 'flow_summary' in self.vessel_results:
            fs = self.vessel_results['flow_summary']
            txt += f"Flow Q:    {fs['Q_nL_per_min']:.3f} nL/min\n"
            txt += f"Reynolds:  {fs['Reynolds']:.2e}\n"
        txt += f"\npixel: {self.pixel_size:.2f} µm/px\n"
        txt += f"fps:   {self.frame_rate:.1f} Hz\n"
        axf.text(0.02, 0.98, txt, transform=axf.transAxes, fontsize=10,
                 va='top', fontfamily='monospace')

        fig.suptitle('Mouse Ear Vessel Analysis — Summary', fontsize=14, fontweight='bold')
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_vessel_results(self):
        """Export per-cross-section morphometry and scalar summaries to CSV."""
        if self.vessel_results is None or 'diam_um' not in self.vessel_results:
            messagebox.showinfo("Info", "Run 'Vessel Morphometry' first.")
            return
        res = self.vessel_results
        path = filedialog.asksaveasfilename(
            title="Save vessel results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            import csv
            with open(path, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(["# Mouse Ear Vessel Analysis export"])
                w.writerow(["# pixel_size_um_per_px", self.pixel_size])
                w.writerow(["# frame_rate_hz", self.frame_rate])
                w.writerow(["# diameter_method", self.diameter_method])
                w.writerow([])
                w.writerow(["position_um", "diameter_um", "area_um2",
                            "r_squared", "flag"])
                s = res['s_um']; d = res['diam_um']
                a = res['area_um2']; rq = res['r_squared']; fl = res['flags']
                for i in range(len(d)):
                    w.writerow([f"{s[i]:.3f}",
                                f"{d[i]:.4f}" if np.isfinite(d[i]) else "",
                                f"{a[i]:.4f}" if np.isfinite(a[i]) else "",
                                f"{rq[i]:.4f}" if np.isfinite(rq[i]) else "",
                                fl[i] if i < len(fl) else ""])
                # scalar summaries
                if 'flow_summary' in res:
                    w.writerow([])
                    w.writerow(["# flow_summary"])
                    for kk, vv in res['flow_summary'].items():
                        w.writerow([kk, vv])
                if 'vasomotion' in res and np.isfinite(res['vasomotion']['dominant_freq_hz']):
                    w.writerow([])
                    w.writerow(["# vasomotion_dominant_freq_hz",
                                res['vasomotion']['dominant_freq_hz']])
            messagebox.showinfo("Saved", f"Results written to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")
    # ========================================================================
    # ============ END MOUSE EAR VESSEL ANALYSIS METHODS =====================
    # ========================================================================


root = tk.Tk()
app = TIFAnalyzer(root)
root.mainloop()
# %%