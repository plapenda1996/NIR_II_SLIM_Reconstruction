# NIR-II SLIM - mouse ear ROI kymograph (cross-check) - produces Fig. 3e-g (arrival-time cross-check)
# environment: heart_valve_py314
# copied from D:\NIR2SLIM\demotion\ROI_analysis_2.py on 2026-09-17
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

class TIFAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("3D TIF Stack ROI Analyzer with Blood Flow Analysis")
        self.root.geometry("1400x900")
        
        # Data variables
        self.image_stack = None
        self.current_frame = 0
        self.background_roi_1 = None  # First background line
        self.background_roi_2 = None  # Second background line
        self.foreground_roi = None
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
        self.pixel_size = 17.0  # µm per pixel: low-magnification 256-px grid used for Fig. 3e-g (was 11.0)
        self.frame_rate = 20.0  # volumes/s of the mouse-ear recordings (was 30.0)
        
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
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
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
    
    def extract_segmented_signals(self):
        """Extract segmented foreground and background signals within marker range"""
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
            
            # Extract segmented background signals for the analysis range
            fg_signals, bg_signals = self.extract_segmented_signals()
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
            
            # Safe frame range that matches the extracted signals
            frame_range = range(start_frame, min(end_frame + 1, total_frames))
            num_analysis_frames = len(list(frame_range))
            
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
            
            # Time axis in ms
            time_axis = (np.arange(kymo_matrix.shape[1]) + start_frame) / self.frame_rate
            
            # Main kymography plot
            ax1 = fig.add_subplot(131)
            im = ax1.imshow(kymo_matrix, aspect='auto', cmap='viridis', origin='lower', 
                        extent=[time_axis[0], time_axis[-1], positions_um[0], positions_um[-1]])
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position (μm)')
            
            # 动态标题根据设置
            bg_text = "FG-BG" if SUBTRACT_BACKGROUND else "FG only"
            ax1.set_title(f'Videokymography\n({bg_text}, {norm_label})')
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


root = tk.Tk()
app = TIFAnalyzer(root)
root.mainloop()
# %%