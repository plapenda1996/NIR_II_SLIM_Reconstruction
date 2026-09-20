classdef main_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                     matlab.ui.Figure
        GridLayout                   matlab.ui.container.GridLayout
        SavewarpedPSFButton          matlab.ui.control.Button
        DButton                      matlab.ui.control.Button
        ShallowLabel_2               matlab.ui.control.Label
        DeepLabel_3                  matlab.ui.control.Label
        TmpFolderButton              matlab.ui.control.Button
        ApplyButton                  matlab.ui.control.Button
        LoadButton                   matlab.ui.control.Button
        TrainButton                  matlab.ui.control.Button
        saveDepthButton              matlab.ui.control.Button
        DDarkButton                  matlab.ui.control.Button
        SetprocessparamButton        matlab.ui.control.Button
        ViewingRawButton             matlab.ui.control.Button
        RButton                      matlab.ui.control.Button
        Button_5                     matlab.ui.control.Button
        MIP3DclampSlider             matlab.ui.control.RangeSlider
        MIP3DclampSliderLabel        matlab.ui.control.Label
        MIPcolorrangeSlider          matlab.ui.control.RangeSlider
        MIPcolorrangeSliderLabel     matlab.ui.control.Label
        MIP3DgammaEditField          matlab.ui.control.NumericEditField
        MIP3DgammaEditFieldLabel     matlab.ui.control.Label
        SaveSlicedtMovieButton       matlab.ui.control.Button
        MIPProjButton                matlab.ui.control.StateButton
        TakeavgperEditField          matlab.ui.control.NumericEditField
        TakeavgperEditFieldLabel     matlab.ui.control.Label
        Button_plus_3                matlab.ui.control.Button
        Button_minus_3               matlab.ui.control.Button
        Button_play_frame            matlab.ui.control.StateButton
        Label_frame                  matlab.ui.control.Label
        FrameSlider                  matlab.ui.control.Slider
        FrameSliderLabel             matlab.ui.control.Label
        AutoButton_3                 matlab.ui.control.StateButton
        UpdateloadButton             matlab.ui.control.Button
        xy_jog_range_reset           matlab.ui.control.Button
        z_jog_range_reset            matlab.ui.control.Button
        xy_jog_range_minus           matlab.ui.control.Button
        xy_jog_range_plus            matlab.ui.control.Button
        z_jog_range_minus            matlab.ui.control.Button
        z_jog_range_plus             matlab.ui.control.Button
        y_jog_minus                  matlab.ui.control.Button
        y_jog_plus                   matlab.ui.control.Button
        x_jog_minus                  matlab.ui.control.Button
        x_jog_plus                   matlab.ui.control.Button
        nDepthEditField              matlab.ui.control.NumericEditField
        nDepthEditFieldLabel         matlab.ui.control.Label
        StepYmmEditField             matlab.ui.control.NumericEditField
        StepYmmEditFieldLabel        matlab.ui.control.Label
        StepXmmEditField             matlab.ui.control.NumericEditField
        StepXmmEditFieldLabel        matlab.ui.control.Label
        Button_4                     matlab.ui.control.Button
        Button_3                     matlab.ui.control.Button
        Button_2                     matlab.ui.control.Button
        PSFsloadedLamp               matlab.ui.control.Lamp
        PSFsloadedLampLabel          matlab.ui.control.Label
        CaptureNStitchButton         matlab.ui.control.Button
        StitchYSpinner               matlab.ui.control.Spinner
        StitchYSpinnerLabel          matlab.ui.control.Label
        StitchXSpinner               matlab.ui.control.Spinner
        StitchXSpinnerLabel          matlab.ui.control.Label
        FullrangeumEditField         matlab.ui.control.NumericEditField
        FullrangeumEditFieldLabel_2  matlab.ui.control.Label
        ymmEditField                 matlab.ui.control.NumericEditField
        ymmEditFieldLabel            matlab.ui.control.Label
        xmmEditField                 matlab.ui.control.NumericEditField
        xmmEditFieldLabel            matlab.ui.control.Label
        ConnectxystageButton         matlab.ui.control.StateButton
        Lamp_2                       matlab.ui.control.Lamp
        ReadouteEditField            matlab.ui.control.NumericEditField
        ReadouteEditFieldLabel       matlab.ui.control.Label
        DamparEditField              matlab.ui.control.NumericEditField
        DamparEditFieldLabel         matlab.ui.control.Label
        PrepareReconButton           matlab.ui.control.Button
        Lamp                         matlab.ui.control.Lamp
        AvgforsyncCaptureNEditField  matlab.ui.control.NumericEditField
        AvgforsyncCaptureNEditFieldLabel  matlab.ui.control.Label
        PosdegEditField              matlab.ui.control.NumericEditField
        PosdegEditFieldLabel         matlab.ui.control.Label
        StepdegEditField             matlab.ui.control.NumericEditField
        stepdegLabel                 matlab.ui.control.Label
        JogEditField                 matlab.ui.control.NumericEditField
        JogEditFieldLabel            matlab.ui.control.Label
        zmmEditField                 matlab.ui.control.NumericEditField
        zmmEditFieldLabel            matlab.ui.control.Label
        OutlierEditField             matlab.ui.control.NumericEditField
        OutlierEditFieldLabel        matlab.ui.control.Label
        TempSpinner                  matlab.ui.control.Spinner
        TempSpinnerLabel             matlab.ui.control.Label
        NEditField                   matlab.ui.control.NumericEditField
        NEditFieldLabel              matlab.ui.control.Label
        AvgEditField                 matlab.ui.control.NumericEditField
        AvgEditFieldLabel            matlab.ui.control.Label
        FPSEditField                 matlab.ui.control.NumericEditField
        FPSEditFieldLabel            matlab.ui.control.Label
        EditField                    matlab.ui.control.NumericEditField
        EditFieldLabel               matlab.ui.control.Label
        DepthrangeEditField          matlab.ui.control.NumericEditField
        DepthrangeEditFieldLabel     matlab.ui.control.Label
        PSFnEditField                matlab.ui.control.NumericEditField
        PSFnEditFieldLabel           matlab.ui.control.Label
        nItersSpinner                matlab.ui.control.Spinner
        nItersSpinnerLabel           matlab.ui.control.Label
        RealtimecaptureSwitch        matlab.ui.control.Switch
        RealtimecaptureSwitchLabel   matlab.ui.control.Label
        GeoloadedLamp                matlab.ui.control.Lamp
        GeoloadedLampLabel           matlab.ui.control.Label
        VarButton                    matlab.ui.control.StateButton
        SyncwhencaptureNButton       matlab.ui.control.StateButton
        ConnectrotstageButton        matlab.ui.control.StateButton
        CamerastatusLabel            matlab.ui.control.Label
        ButtonDownRange              matlab.ui.control.Button
        ButtonUpRange                matlab.ui.control.Button
        ButtonDownJog                matlab.ui.control.Button
        ButtonUpJog                  matlab.ui.control.Button
        Button                       matlab.ui.control.Button
        ConnectzstageButton          matlab.ui.control.StateButton
        CaptureCalibButton           matlab.ui.control.Button
        BuildbiasButton              matlab.ui.control.Button
        CaptureNButton               matlab.ui.control.Button
        ConnectCamButton             matlab.ui.control.StateButton
        ResetButton_2                matlab.ui.control.Button
        DiracPSFCheckBox             matlab.ui.control.CheckBox
        TabGroup                     matlab.ui.container.TabGroup
        RawTab                       matlab.ui.container.Tab
        GridLayout5                  matlab.ui.container.GridLayout
        AmpLabel                     matlab.ui.control.Label
        GammaSlider                  matlab.ui.control.Slider
        GammaSliderLabel             matlab.ui.control.Label
        DclipSlider                  matlab.ui.control.RangeSlider
        DclipSliderLabel             matlab.ui.control.Label
        Button_minus                 matlab.ui.control.Button
        Button_plus                  matlab.ui.control.Button
        RealtimeFPSLabel             matlab.ui.control.Label
        ResetButton                  matlab.ui.control.Button
        AutoButton                   matlab.ui.control.Button
        UIAxes                       matlab.ui.control.UIAxes
        ROITab                       matlab.ui.container.Tab
        avgnumberEditField           matlab.ui.control.NumericEditField
        avgnumberEditFieldLabel      matlab.ui.control.Label
        ROIdepthfullbandwidthSlider  matlab.ui.control.Slider
        ROIdepthfullbandwidthSliderLabel  matlab.ui.control.Label
        ROIdepthcenterSlider         matlab.ui.control.Slider
        ROIdepthcenterSliderLabel    matlab.ui.control.Label
        ROIEditField                 matlab.ui.control.NumericEditField
        ROIEditFieldLabel            matlab.ui.control.Label
        RealtimeFPSLabel_3           matlab.ui.control.Label
        Simple3DTab                  matlab.ui.container.Tab
        GridLayout3                  matlab.ui.container.GridLayout
        DeepLabel_4                  matlab.ui.control.Label
        ShallowLabel_3               matlab.ui.control.Label
        BlurradiusSpinner            matlab.ui.control.Spinner
        BlurradiusSpinnerLabel       matlab.ui.control.Label
        MaskROISpinner               matlab.ui.control.Spinner
        MaskROISpinnerLabel          matlab.ui.control.Label
        MaskCenterYSpinner           matlab.ui.control.Spinner
        MaskCenterYSpinnerLabel      matlab.ui.control.Label
        MaskCenterXSpinner           matlab.ui.control.Spinner
        MaskCenterXSpinnerLabel      matlab.ui.control.Label
        DclipSlider_2                matlab.ui.control.RangeSlider
        DclipSlider_2Label           matlab.ui.control.Label
        DSliceSlider_2               matlab.ui.control.Slider
        DSliceSlider_2Label          matlab.ui.control.Label
        Button_minus_2               matlab.ui.control.Button
        Button_plus_2                matlab.ui.control.Button
        ButtonDownRange_2            matlab.ui.control.Button
        ButtonUpRange_2              matlab.ui.control.Button
        CheckBox5                    matlab.ui.control.CheckBox
        CheckBox4                    matlab.ui.control.CheckBox
        CheckBox2                    matlab.ui.control.CheckBox
        CheckBox1                    matlab.ui.control.CheckBox
        CheckBox3                    matlab.ui.control.CheckBox
        ScanButton                   matlab.ui.control.StateButton
        RealtimeFPSLabel_2           matlab.ui.control.Label
        AutoButton_2                 matlab.ui.control.Button
        UIAxes3                      matlab.ui.control.UIAxes
        DTab_2                       matlab.ui.container.Tab
        GridLayout4                  matlab.ui.container.GridLayout
        ScalebarYSpinner             matlab.ui.control.Spinner
        ScalebarYSpinnerLabel        matlab.ui.control.Label
        ScalebarXSpinner             matlab.ui.control.Spinner
        ScalebarXSpinnerLabel        matlab.ui.control.Label
        BlurradiusSpinner_2          matlab.ui.control.Spinner
        BlurradiusSpinner_2Label     matlab.ui.control.Label
        MaskROISpinner_2             matlab.ui.control.Spinner
        MaskROISpinner_2Label        matlab.ui.control.Label
        MaskCenterYSpinner_2         matlab.ui.control.Spinner
        MaskCenterYSpinner_2Label    matlab.ui.control.Label
        MaskCenterXSpinner_2         matlab.ui.control.Spinner
        MaskCenterXSpinner_2Label    matlab.ui.control.Label
        PlayfpsEditField             matlab.ui.control.NumericEditField
        PlayfpsEditFieldLabel        matlab.ui.control.Label
        ColorDropDown                matlab.ui.control.DropDown
        ColorDropDownLabel           matlab.ui.control.Label
        DSliceSlider                 matlab.ui.control.Slider
        DSliceSliderLabel            matlab.ui.control.Label
        PlayButton                   matlab.ui.control.StateButton
        UIAxes2                      matlab.ui.control.UIAxes
        LoadGeoFileButton            matlab.ui.control.Button
        SaveMovieButton              matlab.ui.control.Button
        ReconButton                  matlab.ui.control.Button
        ListBox                      matlab.ui.control.ListBox
        OpenDataFolderButton         matlab.ui.control.Button
        LoadPSFsButton               matlab.ui.control.Button
        StartPSFestimationButton     matlab.ui.control.Button
        StartgeometricregistrationButton  matlab.ui.control.Button
    end

    
    properties (Access = private)
        CalApp1
        CalApp2
        CalApp3
        CalApp4
        cal1_flag
        cal2_flag
        lastCallTime
        scanCount
        scanSpeed
    end
    
    properties (Access = public)
        cal_path
        data_path
        n2n_path
        cal_file1
        cal_file2

        im_data
        im_adj
        im_amp
        h_2d
        h_roi
        roi_bound
        h_3d
        h_3d_title
        h_3d_simple

        dark_param
        dark_param_3d

        ellipse_mask
        resample_size
        stretched_size
        tform_list
        tform_init
        center_frame_idx
        patch_size
        PSFs
        PSFs_warped
        optimized_tform_list
        tform_bkwd_ref
        tform_bkwd_rel
        pixel_size_x
        pixel_size_y
        
        im_data_cell
        measurements
        measurements_trans
        PSF_interp
        V_interp
        roi_mask
        roi_mask_gpu
        obj_mask
        obj_mask_gpu
        output_mask
        otf_gpu
        forward_tform_list
        backward_tform_list
        recon_ready_flag
        scale_bar_pos

        reconResult
        ssResult
        M

        context
        width
        height
        bufferSize

        device
        device_rot
        device_xy
        genCLI
        sync
        Timer_buffer
        Timer_plot
        Timer_3D
        Timer_stage
        Timer_stage_xy
        Timer_rot
        Timer_status
        Timer_snapshot
        Timer_frame
    end
    
    methods (Access = public)
        
        function updateParams1(app,cal_path,cal_file1)
            app.cal_path = cal_path;
            app.cal_file1 = cal_file1;
            % load
            load_cal1(app);
            if app.cal1_flag
                app.GeoloadedLamp.Color=[0,1,0];
            end
        end
        function updateParams2(app,cal_path,cal_file2)
            app.cal_path = cal_path;
            app.cal_file2 = cal_file2;
            % load 
            load_cal2(app);
            if app.cal2_flag
                app.PSFsloadedLamp.Color=[0,1,0];
            end
        end
        function updateParams3(app,cal_path)
            app.cal_path = cal_path;
        end
        
        function load_cal1(app)
            S = load(app.cal_file1);
            app.ellipse_mask = S.ellipse_mask;
            app.resample_size = S.resample_size;
            app.stretched_size = S.stretched_size;
            app.tform_list = S.tform_list;
            app.tform_init = S.tform_init;
            app.center_frame_idx = S.center_frame_idx;

            try
                % get resize ratio
                sy = app.resample_size(1)/640;
                sx = app.resample_size(2)/512;
                [rows, cols] = find(app.ellipse_mask{1});
                row_min = min(rows); row_max = max(rows);
                col_min = min(cols); col_max = max(cols);
                sy = sy * app.stretched_size(1)/(row_max-row_min+1);
                sx = sx * app.stretched_size(2)/(col_max-col_min+1);
                app.dark_param.PixelSizeX = sx*app.pixel_size_x * 1000;
                app.dark_param.PixelSizeY  = sy*app.pixel_size_y * 1000;
                app.dark_param_3d.PixelSizeX = app.pixel_size_x* 1000;
                app.dark_param_3d.PixelSizeY = app.pixel_size_y* 1000;
            end

            app.cal1_flag = true;
        end
        function load_cal2(app)
            S = load(app.cal_file2);
            app.patch_size = S.patch_size;
            app.PSFs = S.PSFs;
            app.optimized_tform_list = S.optimized_tform_list;
            
            app.pixel_size_x = S.pixel_size_x;
            app.pixel_size_y = S.pixel_size_y;

            try
                % get resize ratio
                sy = app.resample_size(1)/640;
                sx = app.resample_size(2)/512;
                [rows, cols] = find(app.ellipse_mask{1});
                row_min = min(rows); row_max = max(rows);
                col_min = min(cols); col_max = max(cols);
                sy = sy * app.stretched_size(1)/(row_max-row_min+1);
                sx = sx * app.stretched_size(2)/(col_max-col_min+1);
                app.dark_param.PixelSizeX = sx*app.pixel_size_x * 1000;
                app.dark_param.PixelSizeY  = sy*app.pixel_size_y * 1000;
                app.dark_param_3d.PixelSizeX = app.pixel_size_x* 1000;
                app.dark_param_3d.PixelSizeY = app.pixel_size_y* 1000;
            end

            app.cal2_flag = true;
        end

        function background_remove(app)
            
            if app.dark_param.if_dark
                app.im_adj = darkChannelRemoveBackground(app.im_adj, 'dark', app.dark_param);
            end
            if app.dark_param.if_tophat
                if app.dark_param.tophat_roll
                    app.im_adj = darkChannelRemoveBackground(app.im_adj, 'rolling ball', app.dark_param);
                else
                    app.im_adj = darkChannelRemoveBackground(app.im_adj, 'tophat', app.dark_param);
                end
            end
            if app.dark_param.if_clip
                app.im_adj = imadjust(app.im_adj, app.dark_param.clip);
            end
        end

        %% 时域动画相关函数
        function gen_recon_frame_init(app)
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                init_data(app);
                init_interp(app);
                app.MaskCenterXSpinner.Value = floor(app.stretched_size(2)/2);
                app.MaskCenterYSpinner.Value = floor(app.stretched_size(1)/2);
                init_roi_mask(app);
                init_obj_mask(app);
            elseif strcmp(app.TabGroup.SelectedTab.Title, '3D')
                PrepareReconButtonPushed(app);
            end
        end
        function [reconVol,amp] = gen_recon_frame(app, frame_idx, depth_ratio)
            app.FrameSlider.Value = frame_idx;
            if nargin>2
                lam = depth_ratio*(app.EditField.Value-app.DepthrangeEditField.Value) + app.DepthrangeEditField.Value;
                lam = max(min(lam,app.DSliceSlider_2.Limits(2)),app.DSliceSlider_2.Limits(1));
                app.DSliceSlider_2.Value = lam;
                app.DSliceSlider.Value = lam;
                depth_idx = lam2idx(app,lam);
            end
            % update_2D(app);
            init_2D(app);
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                init_data(app);
                reconVol = init_3D_simple(app); 
                amp = app.im_amp;
            elseif strcmp(app.TabGroup.SelectedTab.Title, '3D')
                % PrepareReconButtonPushed(app);
                app.Lamp.Color = [1,0,0];
                init_data(app);
                init_gpu(app);
                app.Lamp.Color = [0,1,0];
                app.recon_ready_flag = true;

                ReconButtonPushed(app);
                reconVol = init_3D(app); 
                amp = app.im_amp;
            end
            if nargin>2
                reconVol = reconVol(depth_idx);
                amp = app.im_amp;
            end
        end
        function fileList = get_items_with_cfg(app)
            % 已经列出的所有数据文件
            allItems = app.ListBox.Items;
            if isempty(allItems)
                fileList = {};
                return;                 % 列表本来就空，直接退出
            end
            keepMask = false(1, numel(allItems));  % 标记哪些需要保留

            for k = 1:numel(allItems)
                [~, baseName, ~] = fileparts(allItems{k});
                cfgPath = fullfile(app.data_path, [baseName '.txt']);
                % exist(..., 'file') == 2 说明确实存在一个文件
                keepMask(k) = (exist(cfgPath, 'file') == 2);
            end
            % 重新过滤列表
            fileList = allItems(keepMask);
        end
        function [filename,frame_range] = set_item_with_cfg(app, filename)
            app.ListBox.Value = filename;
            [~,filename,~] = fileparts(filename);
            frame_range = app.FrameSlider.Limits;
            ListBoxValueChanged(app);
        end
        function [filename,frame_range,depth,n_depth] = get_current_item(app)
            [~,filename,~] = fileparts(app.ListBox.Value);
            frame_range = app.FrameSlider.Limits;
            depth = app.FullrangeumEditField.Value;
            n_depth = app.nDepthEditField.Value;
        end

        %% 其他函数
        function depth_idx = lam2idx(app,lambda)
            ratio = (lambda-app.DepthrangeEditField.Value) / (app.EditField.Value-app.DepthrangeEditField.Value);
            depth_idx = round(ratio*(app.nDepthEditField.Value-1)) +1;
            % depth_idx = round(lambda*(app.nDepthEditField.Value-1)) +1;
        end
        function lambda = idx2lam(app,idx)
            % lambda = (idx-1)/(app.nDepthEditField.Value-1);
            ratio = (idx-1)/(app.nDepthEditField.Value-1);
            lambda = app.DepthrangeEditField.Value + ratio * (app.EditField.Value-app.DepthrangeEditField.Value);
        end
        function im = bad_pixel_correction(app,im)
            % bad pixel correction
            if app.dark_param.if_bad
                im = correctHotPixels(im, 'zScore', app.dark_param.zScore,...
                    'WinSize', app.dark_param.WinSize,...
                    'MaxSize', app.dark_param.MaxSize);
            end
        end
        function view_idx_list = get_view_idx_list(app)
            view_idx_list = [];
            if app.CheckBox1.Value
                view_idx_list = [view_idx_list, 1];
            end
            if app.CheckBox2.Value
                view_idx_list = [view_idx_list, 2];
            end
            if app.CheckBox3.Value
                view_idx_list = [view_idx_list, 3];
            end
            if app.CheckBox4.Value
                view_idx_list = [view_idx_list, 4];
            end
            if app.CheckBox5.Value
                view_idx_list = [view_idx_list, 5];
            end
        end

        function update_snap(app)
            app.data_path = 'D:\C_RED2_recordings\';
            init_data_folder(app);
            ListBoxValueChanged(app);
        end

        function init_2D(app)
            num_images = size(app.im_data, 3);
            num_avg = app.TakeavgperEditField.Value;
            num_frame = ceil(num_images / num_avg);
            % set frames slider
            if num_images>1
                app.TakeavgperEditField.Enable = true;
            else
                app.TakeavgperEditField.Enable = false;
            end
            if num_frame>1
                app.FrameSlider.Enable = true;
                app.FrameSlider.Limits = [1, num_frame];
                app.Button_plus_3.Enable = true;
                app.Button_minus_3.Enable = true;
                app.Button_play_frame.Enable = true;
            else
                app.FrameSlider.Value=1;
                app.FrameSlider.Enable = false;
                app.Button_plus_3.Enable = false;
                app.Button_minus_3.Enable = false;
                app.Button_play_frame.Enable = false;
            end
            % get indices
            idx_img = (round(app.FrameSlider.Value)-1) * num_avg + (1:num_avg);
            app.Label_frame.Text = sprintf('%d/%d', round(app.FrameSlider.Value), app.FrameSlider.Limits(2));
            % update show
            if app.VarButton.Value
                app.im_adj = rescale(std(app.im_data(:,:,idx_img),0,3));
            else
                app.im_adj = mean(app.im_data(:,:,idx_img),3);
            end
            
            %% adjust
            % bad pixel correction
            app.im_adj = bad_pixel_correction(app,app.im_adj);

            %% get global amp
            app.im_amp = max(app.im_adj(:));
            app.AmpLabel.Text =  sprintf('Amp: %.2f',app.im_amp);

            %% adj/rescale
            value = app.DclipSlider.Value;
            gamma_val = app.GammaSlider.Value;
            app.im_adj =  imadjust(rescale(app.im_adj), value, [0,1], gamma_val);

            %% background remove
            background_remove(app);
            
            %% update
            app.h_2d = imagesc(app.UIAxes,app.im_adj);
            axis(app.UIAxes,'off')
            colormap(app.UIAxes,'gray')
            colorbar(app.UIAxes)
        end

        function update_2D(app)
            % get indices
            num_avg = app.TakeavgperEditField.Value;
            idx_img = (round(app.FrameSlider.Value)-1) * num_avg + (1:num_avg);
            app.Label_frame.Text = sprintf('%d/%d', round(app.FrameSlider.Value), app.FrameSlider.Limits(2));
            % update show
            if app.VarButton.Value
                app.im_adj = rescale(std(app.im_data(:,:,idx_img),0,3));
            else
                app.im_adj = mean(app.im_data(:,:,idx_img),3);
            end
            % bad pixel correction
            app.im_adj = bad_pixel_correction(app,app.im_adj);
            % get global amp
            app.im_amp = max(app.im_adj(:));
            app.AmpLabel.Text =  sprintf('Amp: %.2f',app.im_amp);
            % adj/rescale
            value = app.DclipSlider.Value;
            gamma_val = app.GammaSlider.Value;
            app.im_adj = imadjust(rescale(app.im_adj), value, [0,1],gamma_val);
            % background remove
            background_remove(app);
            set(app.h_2d, 'CData', app.im_adj);
            drawnow
        end

        function reconVol = init_3D(app)
            % update show
            lam = app.DSliceSlider.Value;
            cla(app.UIAxes2); colorbar(app.UIAxes2,'off'); 
            if app.MIPProjButton.Value
                % get MIP
                [reconVol,depthFrame] = max(double(app.reconResult),[],3);
                reconVol = rescale(reconVol.* app.output_mask);
                
                depthFrame = (depthFrame - 1) / (size(app.reconResult,3) - 1) * (app.MIPcolorrangeSlider.Value(2)-app.MIPcolorrangeSlider.Value(1)) + app.MIPcolorrangeSlider.Value(1);  
                [RGB_mip,cmap] = depthIntensityMap(depthFrame,rescale(imadjust(reconVol, app.MIP3DclampSlider.Value, [0,1], app.MIP3DgammaEditField.Value)));

                app.h_3d = imshow(RGB_mip,'Parent',app.UIAxes2);
                axis(app.UIAxes2,'off')
                % colormap(app.UIAxes2,'gray')
                colormap(app.UIAxes2, cmap);
                caxis(app.UIAxes2, [app.MIPcolorrangeSlider.Value(1) app.MIPcolorrangeSlider.Value(2)]);
                cb = colorbar(app.UIAxes2);
                cb.Ticks     = app.MIPcolorrangeSlider.Value;
                cb.Position = [0.65 0.3 0.025 0.6];
                cb.Label.String = 'Depth (\mum)';
                cb.TickLabels = {num2str(app.FullrangeumEditField.Value*app.DepthrangeEditField.Value),...
                    num2str(app.FullrangeumEditField.Value*app.EditField.Value)};

                app.h_3d_title = title(app.UIAxes2,sprintf('MIP'));
                app.scale_bar_pos = [app.ScalebarXSpinner.Value,app.ScalebarYSpinner.Value];
                addScaleBar(app.UIAxes2, 100/app.pixel_size_x, '100um', 14,0.01,app.scale_bar_pos);
            else
                depth_idx = lam2idx(app,lam);
                reconVol = rescale(app.reconResult .* app.output_mask);
                reconVol = imadjustn(reconVol, app.MIP3DclampSlider.Value, [0,1], app.MIP3DgammaEditField.Value);
                app.h_3d = imagesc(app.UIAxes2,reconVol(:,:,depth_idx));
                axis(app.UIAxes2,'off')
                colormap(app.UIAxes2,'gray')
                colorbar(app.UIAxes2,'off');
                depth = lam*app.FullrangeumEditField.Value - app.FullrangeumEditField.Value/2;
                app.h_3d_title = title(app.UIAxes2,sprintf('Depth #%d/%d, %.0f um', depth_idx, size(app.reconResult,3), depth));
                app.scale_bar_pos = [app.ScalebarXSpinner.Value,app.ScalebarYSpinner.Value];
                addScaleBar(app.UIAxes2, 100/app.pixel_size_x, '100um', 14,0.01,app.scale_bar_pos);
            end

        end
        function update_3D(app,lam)
            % update show
            if app.MIPProjButton.Value
                 % get MIP
                [reconFrame,depthFrame] = max(double(app.reconResult),[],3);
                reconFrame = (reconFrame.* app.output_mask);
                depthFrame = (depthFrame - 1) / (size(app.reconResult,3) - 1) * (app.MIPcolorrangeSlider.Value(2)-app.MIPcolorrangeSlider.Value(1)) + app.MIPcolorrangeSlider.Value(1);  
                RGB_mip = depthIntensityMap(depthFrame,rescale(imadjust(reconFrame, app.MIP3DclampSlider.Value, [0,1], app.MIP3DgammaEditField.Value)));
                set(app.h_3d, 'CData', RGB_mip);
                s = sprintf('MIP');
                set(app.h_3d_title, 'String', s)
                drawnow
            else
                depth_idx = lam2idx(app,lam);
                reconFrame = rescale(app.reconResult(:,:,depth_idx) .* app.output_mask);
                reconFrame = imadjust(reconFrame, app.MIP3DclampSlider.Value, [0,1], app.MIP3DgammaEditField.Value);
                set(app.h_3d, 'CData', reconFrame);
                depth = lam*app.FullrangeumEditField.Value - app.FullrangeumEditField.Value/2;
                s = sprintf('Depth #%d/%d, %.0f um', depth_idx, size(app.reconResult,3), depth);
                set(app.h_3d_title, 'String', s)
                drawnow
            end
            
        end
        function play_step_3D(app,~,~)
            % n = size(app.reconResult,3);
            new_idx = lam2idx(app,app.DSliceSlider.Value) +1;
            new_lam = idx2lam(app,new_idx);
            if new_lam>app.DSliceSlider.Limits(2)
                new_lam = app.DSliceSlider.Limits(1);
            end
            app.DSliceSlider.Value = new_lam;
            update_3D(app,new_lam)
        end
        function update_status(app,~,~)
            [~, t1] = FliSdk.sendCommandToCamera(app.context, 'temperatures frontend');
            [~, t2] = FliSdk.sendCommandToCamera(app.context, 'temperatures sensor');
            app.CamerastatusLabel.Text = [string(t1);string(t2)];
        end
        function shift_and_sum(app,view_idx_list,depth_idx)
            % % shift and sum
            app.ssResult = zeros(size(app.measurements(:,:,1)));
            if nargin>2
                im_sum = zeros([size(app.measurements(:,:,1)),length(view_idx_list)]);
                for i=1:length(view_idx_list)
                    view_idx = view_idx_list(i);
                    % apply ROI mask
                    im_mov = app.measurements(:,:,view_idx) .* app.roi_mask(:,:,view_idx);
                    im_reg = imwarp(im_mov, app.backward_tform_list(view_idx,depth_idx), 'OutputView', imref2d(size(im_mov)));
                    % im_reg = real(ifft2(fft2(im_reg) .* H(:,:,i)));
                    im_sum(:,:,i) = im_reg;
                end
                app.ssResult = rescale(mean(im_sum,3));
                % apply obj mask
                app.ssResult = app.ssResult.*app.obj_mask(:,:,1);
                % apply dark
                if app.dark_param_3d.if_dark
                    app.ssResult = darkChannelRemoveBackground(app.ssResult, 'dark', app.dark_param_3d);
                end
            else
                app.ssResult = zeros(size(app.measurements(:,:,1),1),size(app.measurements(:,:,1),2),size(app.backward_tform_list,2));
                for j=1:size(app.backward_tform_list,2)
                    im_sum = zeros([size(app.measurements(:,:,1)),length(view_idx_list)]);
                    for i=1:length(view_idx_list)
                        view_idx = view_idx_list(i);
                        % apply ROI mask
                        im_mov = app.measurements(:,:,view_idx) .* app.roi_mask(:,:,view_idx);
                        im_reg = imwarp(im_mov, app.backward_tform_list(view_idx,j), 'OutputView', imref2d(size(im_mov)));
                        % im_reg = real(ifft2(fft2(im_reg) .* H(:,:,i)));
                        im_sum(:,:,i) = im_reg;
                    end
                    app.ssResult(:,:,j) = mean(im_sum,3);
                    % apply obj mask
                    app.ssResult(:,:,j) = app.ssResult(:,:,j).*app.obj_mask(:,:,1);
                end
            end
        end

        function reconVol = init_3D_simple(app)
            colorbar(app.UIAxes3,'off');
            if app.MIPProjButton.Value
                % recon
                view_idx_list = get_view_idx_list(app);
                shift_and_sum(app,view_idx_list);
                % get MIP
                [reconFrame,depthFrame] = max(app.ssResult,[],3);
                depthFrame = (depthFrame - 1) / (size(app.ssResult,3) - 1) * (app.MIPcolorrangeSlider.Value(2)-app.MIPcolorrangeSlider.Value(1)) + app.MIPcolorrangeSlider.Value(1);  
                app.ssResult = depthIntensityMap(depthFrame,rescale(imadjust(reconFrame, app.MIP3DclampSlider.Value, [0,1], app.MIP3DgammaEditField.Value)));
                % update show
                app.h_3d_simple = imshow(app.ssResult,'Parent',app.UIAxes3);
                axis(app.UIAxes3,'off')
                axis(app.UIAxes3,'equal')
            else
                % recon
                depth_idx = lam2idx(app,app.DSliceSlider_2.Value);
                view_idx_list = get_view_idx_list(app);
                if nargout>0
                    shift_and_sum(app,view_idx_list);
                    reconVol = imadjustn(app.ssResult, app.DclipSlider_2.Value, [0,1]);
                    app.ssResult = reconVol(depth_idx);
                else
                    shift_and_sum(app,view_idx_list,depth_idx);
                    app.ssResult = imadjust(rescale(app.ssResult), app.DclipSlider_2.Value, [0,1]);
                end
                
                app.h_3d_simple = imagesc(app.UIAxes3,app.ssResult);
                axis(app.UIAxes3,'off')
                axis(app.UIAxes3,'equal')
                colormap(app.UIAxes3,'gray')
            end
            
            % addScaleBar(app.UIAxes3, 100/app.pixel_size_x, '100um', 14,0.01);
        end
        function update_3D_simple(app,depth_idx,view_idx_list)
            if app.MIPProjButton.Value
                % recon
                view_idx_list = get_view_idx_list(app);
                shift_and_sum(app,view_idx_list);
                % get MIP
                [reconFrame,depthFrame] = max(app.ssResult,[],3);
                depthFrame = (depthFrame - 1) / (size(app.ssResult,3) - 1) * (app.MIPcolorrangeSlider.Value(2)-app.MIPcolorrangeSlider.Value(1)) + app.MIPcolorrangeSlider.Value(1);  
                app.ssResult = depthIntensityMap(depthFrame,rescale(imadjust(reconFrame, app.MIP3DclampSlider.Value, [0,1], app.MIP3DgammaEditField.Value)));
                % update plot
                set(app.h_3d_simple, 'CData', app.ssResult);
                axis(app.UIAxes3,'off')
                axis(app.UIAxes3,'equal')
            else
                % recon
                shift_and_sum(app,view_idx_list,depth_idx);
                % update plot
                ss_adj = imadjust(rescale(app.ssResult), app.DclipSlider_2.Value, [0,1]);
                set(app.h_3d_simple, 'CData', ss_adj);
            end
            
        end

        function init_interp(app)
            %% Prepare transformation
            num_depth = app.nDepthEditField.Value;
            lambda = linspace(app.DepthrangeEditField.Value, app.EditField.Value, num_depth);
            % % prepare forward transformation
            % invert
            app.forward_tform_list = app.optimized_tform_list;
            num_frame = size(app.optimized_tform_list,2);
            num_view = size(app.optimized_tform_list,1);
            for i=1:num_view
                for j=1:num_frame
                    app.forward_tform_list(i,j) = app.forward_tform_list(i,j).invert;
                end
            end
            %% interp transformation
            % reconstruction number of depth setting            
            app.forward_tform_list = interp_transformations(app.forward_tform_list,lambda);
            app.backward_tform_list = interp_transformations(app.optimized_tform_list,lambda);
            % % interp rel transformation
            app.tform_bkwd_ref = repmat(affinetform2d(),[1,num_view]);
            app.tform_bkwd_rel = repmat(affinetform2d(),[num_view,num_depth]);
            mid_z = floor(num_depth/2)+1;
            for i=1:num_view
                tform_tmp = invert(app.backward_tform_list(i,mid_z));
                A_ref_fwd = tform_tmp.A;
                % bkwd_ref: LF space -> 3D obj space (depth invariant)
                app.tform_bkwd_ref(i) = app.backward_tform_list(i,mid_z);
                for j=1:num_depth
                    A = app.backward_tform_list(i,j).A * A_ref_fwd;
                    app.tform_bkwd_rel(i,j) = affinetform2d(A);
                end
            end
        end
        
        function init_obj_mask(app)
            centerX = app.MaskCenterXSpinner.Value;
            centerY = app.MaskCenterYSpinner.Value;
            roiRatio = app.MaskROISpinner.Value/100;
            imgSize = app.stretched_size;
            blurRadius = app.BlurradiusSpinner.Value;
            num_depth = app.nDepthEditField.Value;
            app.obj_mask = repmat(createCircularMask(centerX,centerY,roiRatio,imgSize,blurRadius),1,1,num_depth);
        end

        function init_roi_mask(app)
            % % Prepare ROI mask
            roi_ratio = app.ROIEditField.Value/100; % ROI ratio (both height and width)
            depth_center= app.ROIdepthcenterSlider.Value; % choose depth for ROI mask generation (geo. transform)
            depth_bw = app.ROIdepthfullbandwidthSlider.Value;
            depth_avg_num = app.avgnumberEditField.Value;
            blur_range = 20; % gaussian filter param. for soft mask
            lambda = linspace(depth_center-depth_bw/2,depth_center+depth_bw/2,depth_avg_num);
            tform_list_roi = interp_transformations(app.forward_tform_list,lambda);
            app.roi_mask = generate_roi_masks(tform_list_roi,...
                size(app.measurements(:,:,1)), roi_ratio, blur_range);
            app.roi_bound = cell(1,size(app.measurements,3));
            for c = 1:length(app.roi_bound)
                mask = imbinarize(app.roi_mask(:,:,c)); 
                app.roi_bound{c} = bwboundaries(mask);
            end
        end
        function init_output_mask(app)
            % get output mask
            centerX = app.MaskCenterXSpinner_2.Value;
            centerY = app.MaskCenterYSpinner_2.Value;
            roiRatio = app.MaskROISpinner_2.Value/100;
            imgSize = app.stretched_size;
            blurRadius = app.BlurradiusSpinner_2.Value;
            app.output_mask = createCircularMask(centerX,centerY,roiRatio,imgSize,blurRadius);
        end
        function init_roi_plot(app)
            % update data
            % init_data(app);
            t=tiledlayout(app.ROITab,2,3,'TileSpacing','none'); drawnow
            app.h_roi = [];
            for i=1:size(app.measurements,3)
                ax = nexttile(t,i);
                % depth_center = ceil(size(app.backward_tform_list,2)/2);
                % im_reg = imwarp(app.measurements(:,:,i), app.backward_tform_list(i,depth_center), 'OutputView', imref2d(size(app.measurements(:,:,i))));
                app.h_roi(i) = imagesc(app.measurements(:,:,i),'Parent', ax);
                % app.h_roi(i) = imagesc(im_reg,'Parent', ax);
                colormap(ax,"gray")
                axis(ax,'off')
                hold(ax,"on")
                % 在每个子图中绘制对应的边界
                boundaries = app.roi_bound{i};
                for k = 1:length(boundaries)
                    boundary = boundaries{k};
                    plot(ax, boundary(:,2), boundary(:,1), 'r--', 'LineWidth', 1); % 红色虚线
                end
                hold(ax,"off")
                clim(ax,"auto") 
            end
            drawnow
        end
        function init_psf(app)
            % % prepare measurement/PSFs on GPU and model functions
            h = msgbox('Calc warped 3D OTF for each view...');
            %% interp PSFs
            [num_view, num_depth] = size(app.tform_bkwd_rel);
            lambda = linspace(app.DepthrangeEditField.Value, app.EditField.Value, num_depth);
            if app.DiracPSFCheckBox.Value
                % delta function-like PSF
                app.PSF_interp = zeros(size(app.PSFs,1),size(app.PSFs,2),num_depth,num_view);
                app.PSF_interp(ceil((size(app.PSFs,1)+1)/2),ceil((size(app.PSFs,1)+1)/2),:,:) = 1;
            else
                pca_k = 7;
                [app.PSF_interp,app.V_interp] = interpolate_PSFs(app.PSFs, lambda, pca_k);
            end
            %% warp PSFs
            [~, PSFs_warped_all_view] = warpPSFs(app.PSF_interp, app.stretched_size, app.tform_bkwd_rel);
            % PSF 取n次方
            n = app.PSFnEditField.Value;
            app.PSFs_warped = PSFs_warped_all_view.^n;
            % 归一化
            app.PSFs_warped = app.PSFs_warped ./ sum(app.PSFs_warped, [1,2]);
            % 计算OTF
            num_view = size(app.PSFs_warped,4);
            psfsize = size(app.PSFs_warped(:,:,:,1));
            app.otf_gpu = zeros([psfsize,num_view],'single','gpuArray');
            for i=1:num_view
                app.otf_gpu(:,:,:,i) = gpuArray(single(psf2otf_gpu(app.PSFs_warped(:,:,:,i), psfsize)));
                app.otf_gpu(:,:,:,i) = app.otf_gpu(:,:,:,i) ./ abs(app.otf_gpu(1,1,1,i));
            end
            if isvalid(h)
                close(h)
            end
        end

        function init_gpu(app)
            % h=msgbox('Apply geoTrans. on mea...', 'modal');
            %% apply [bkwd ref. tform] on measurement for each view
            [num_view, ~] = size(app.tform_bkwd_rel);
            app.measurements_trans = zeros(app.stretched_size(1),app.stretched_size(2),num_view);
            for i=1:num_view
                app.measurements_trans(:,:,i) = imwarp(app.measurements(:,:,i),...
                    app.tform_bkwd_ref(i),...
                    'OutputView', imref2d(size(app.measurements(:,:,i))));
            end
            % measurement moved to GPU
            app.measurements_trans = gpuArray(single(app.measurements_trans)); 

            % h=msgbox('Move masks to GPU...', 'modal');
            %% move masks to GPU
            app.obj_mask_gpu = gpuArray(single(app.obj_mask));
            app.roi_mask_gpu = gpuArray(single(app.roi_mask));
            % if isvalid(h)
            %     close(h)
            % end
        end

        function init_data(app)
            % h = msgbox('Init data...');
            %% Crop out sub-apertures and stretch data images
            % % prepare test data as measurements
            app.im_data_cell = cell(1,length(app.ellipse_mask));
            % loop through apertures
            im_data2 = imresize(double(app.im_adj'),app.resample_size);
            for i=1:length(app.ellipse_mask)
                % Find the bounding box of each ellipse region
                [rows, cols] = find(app.ellipse_mask{i});
                row_min = min(rows);
                row_max = max(rows);
                col_min = min(cols);
                col_max = max(cols);
                % Extract the subimages
                im_data_sub = im_data2;
                im_data_sub(~app.ellipse_mask{i}) = 0;
                im_data_sub = im_data_sub(row_min:row_max, col_min:col_max, :);
                % Stretch the subimage to make the ellipse a circle
                im_data_sub = imresize(im_data_sub,app.stretched_size);
                % save sub-images
                app.im_data_cell{i} = rescale(im_data_sub);
            end
            % prepare data
            app.measurements = cat(3, app.im_data_cell{:});
            % if isvalid(h)
            %     close(h)
            % end
        end
        function init_data_folder(app)
            % 获取所选文件夹内所有文件的列表
            files = dir(fullfile(app.data_path, '*')); % 获取所有文件和文件夹
            files = files(~[files.isdir]); % 移除文件夹，只保留文件

            % 初始化一个空cell数组来存储符合条件的文件名
            validFiles = {};
            validFilesDates = [];  % 存储文件日期

            % 遍历文件列表，筛选出后缀名为.mat或.hdf5的文件
            for i = 1:length(files)
                [~, ~, ext] = fileparts(files(i).name); % 获取文件的扩展名
                % if strcmp(ext, '.mat') || strcmp(ext, '.hdf5') || strcmp(ext, '.raw')
                %     % 如果文件扩展名为.mat或.hdf5，加入到validFiles中
                %     validFiles{end+1} = files(i).name; % 添加文件名到列表中
                %     validFilesDates(end+1) = files(i).datenum; % 添加文件日期
                % end
                if strcmp(ext, '.hdf5') || strcmp(ext, '.raw')
                    % 如果文件扩展名为.raw或.hdf5，加入到validFiles中
                    validFiles{end+1} = files(i).name; % 添加文件名到列表中
                    validFilesDates(end+1) = files(i).datenum; % 添加文件日期
                end
            end

            % 如果有有效文件，则按日期排序（最新的在前）
            if ~isempty(validFiles)
                % 获取排序索引，按日期降序排列
                [~, sortIdx] = sort(validFilesDates, 'descend');

                % 重新排列文件列表
                validFiles = validFiles(sortIdx);

                % 设置列表框的值
                app.ListBox.Items = validFiles;
                app.ListBox.Value = validFiles{1};
            else
                % 如果没有有效文件，清空列表
                app.ListBox.Items = {};
                app.ListBox.Value = {};
            end
        end

        function imgcube = bufferN(app,N,out_ratio)
            if nargin<3
                out_ratio = 100;
            end
            offset = 400;
            try
                % capture
                imgcube=zeros(app.width,app.height,N);
                indexEnd = FliSdk.getBufferFilling(app.context);
                for d=1:N
                    index = mod(indexEnd - (N-d), app.bufferSize+1);
                    buffer = double(typecast(FliSdk.getProcessedImage(app.context, index), 'int16'));
                    buffer = buffer + offset;
                    buffer(buffer<0) = 0;
                    % imgcube(:,:,d)=reshape(buffer,app.width,app.height,[])/(16384+offset);
                    imgcube(:,:,d)=reshape(buffer,app.width,app.height,[]);
                end
                % crop edges
                imgcube(1,:,:) = imgcube(2,:,:);
                imgcube(end,:,:) = imgcube(end-1,:,:);
                imgcube(:,1,:) = imgcube(:,2,:);
                imgcube(:,end,:) = imgcube(:,end-1,:);
                % remove outliers
                if out_ratio<100
                    threshold_x = prctile(imgcube(:), out_ratio);
                    imgcube(imgcube>threshold_x) = threshold_x;
                end
            catch
                warning('Error in bufferN()');
            end
        end

        function imgcube = captureN(app, N, out_ratio)
            if nargin<3
                out_ratio = 100;
            end

            if app.sync
                % init
                imgcube=zeros(app.width,app.height,N);
                % start capturing
                curr_pos = System.Decimal.ToDouble(app.device_rot.Position);
                pos_list = mod(curr_pos + ((1:N)*app.StepdegEditField.Value), 360);
                n = app.AvgforsyncCaptureNEditField.Value;
                h = waitbar(0,'GrabN...');
                for i=1:N
                    % rotation
                    app.device_rot.MoveTo(pos_list(i),60000);

                    % grabN
                    FliSdk.enableGrabN(app.context,n);
                    isGrabNFinished = FliSdk.isGrabNFinished(app.context);
                    while ~isGrabNFinished
                        isGrabNFinished = FliSdk.isGrabNFinished(app.context);
                    end
                    
                    % capture buffer
                    imgcube(:,:,i) = mean(bufferN(app,n,out_ratio),3);
                    waitbar(i/N,h,sprintf('Frame %d/%d grabbing...',i,N))
                end
                % turn off grabN
                FliSdk.disableGrabN(app.context);
                if isvalid(h)
                    close(h)
                end
            else
                % normal grabN
                % wait
                if N>5
                    startIdx = FliSdk.getBufferFilling(app.context);
                    h = waitbar(0,'GrabN...');
                    % turn on grabN
                    FliSdk.enableGrabN(app.context,N);
                    while ~FliSdk.isGrabNFinished(app.context)
                        currIdx = FliSdk.getBufferFilling(app.context);
                        if currIdx<startIdx
                            currIdx = app.bufferSize -startIdx +1 + currIdx;
                        end
                        waitbar((currIdx-startIdx+1)/(N),h,sprintf('Frame %d/%d grabbing...',currIdx-startIdx+1,N))
                    end
                    if isvalid(h)
                        close(h)
                    end
                else
                    % turn on grabN
                    FliSdk.enableGrabN(app.context,N);
                    isGrabNFinished = FliSdk.isGrabNFinished(app.context);
                    while ~isGrabNFinished
                        isGrabNFinished = FliSdk.isGrabNFinished(app.context);
                    end
                end
                % capture buffer
                imgcube = bufferN(app,N,out_ratio);
                % turn off grabN
                FliSdk.disableGrabN(app.context);
            end
            % % start camera (if not)
            % FliSdk.start(app.context);

            % imgcube = permute(imgcube,[2 1 3]);
        end

        function update_buffer(app,~,~)
            try
                N = app.AvgEditField.Value;
                out_ratio = app.OutlierEditField.Value;
                pause(0.025)
                app.im_data = mean(permute(bufferN(app, N, out_ratio),[2,1,3]), 3);
                pause(0.025)
            catch
                warning('Error in update_buffer()');
            end
        end

        function refresh(app,~,~)
            switch app.TabGroup.SelectedTab.Title
                case 'Raw'
                    % update plot
                    update_2D(app);
                    % update label   
                    currentTime = now;
                    if ~isempty(app.lastCallTime)
                        interval = (currentTime - app.lastCallTime) * 24 * 3600;
                        app.RealtimeFPSLabel.Text = sprintf('Real-time FPS: %.0f', 1/interval);
                    end
                    app.lastCallTime = currentTime;
                case 'Simple 3D'
                    % update data
                    value = app.DclipSlider.Value;
                    gamma_val = app.GammaSlider.Value;
                    app.im_adj = imadjust(rescale(app.im_data), value, [0,1], gamma_val);
                    init_data(app);
                    % auto focal-scan
                    if app.ScanButton.Value
                        app.scanCount = mod(app.scanCount+1, app.scanSpeed);
                        if app.scanCount==0
                            lam = app.DSliceSlider_2.Value;
                            num_depth = app.nDepthEditField.Value;
                            lam_new = lam + (app.DSliceSlider_2.Limits(2)-app.DSliceSlider_2.Limits(1))/(num_depth-1);
                            if lam_new > app.DSliceSlider_2.Limits(2)
                                lam_new = app.DSliceSlider_2.Limits(1);
                            end
                            app.DSliceSlider_2.Value = lam_new;
                        end
                    end
                    % get index
                    depth_idx = lam2idx(app,app.DSliceSlider_2.Value);
                    view_idx_list = get_view_idx_list(app);
                    % update plot
                    update_3D_simple(app,depth_idx,view_idx_list);
                    % update label   
                    currentTime = now;
                    if ~isempty(app.lastCallTime)
                        interval = (currentTime - app.lastCallTime) * 24 * 3600;
                        app.RealtimeFPSLabel_2.Text = sprintf('Real-time FPS: %.0f', 1/interval);
                    end
                    app.lastCallTime = currentTime;
                case 'ROI'
                    % update data
                    value = app.DclipSlider.Value;
                    gamma_val = app.GammaSlider.Value;
                    app.im_adj = imadjust(rescale(app.im_data), value, [0,1],gamma_val);
                    init_data(app);
                    % update plot
                    for i=1:length(app.h_roi)
                        % depth_center = ceil(size(app.backward_tform_list,2)/2);
                        % im_reg = imwarp(app.measurements(:,:,i), app.backward_tform_list(i,depth_center), 'OutputView', imref2d(size(app.measurements(:,:,i))));
                        set(app.h_roi(i), 'CData', rescale(app.measurements(:,:,i)))
                        % set(app.h_roi(i), 'CData', rescale(im_reg))
                    end
                    drawnow
                    % update label                    
                    currentTime = now;
                    if ~isempty(app.lastCallTime)
                        interval = (currentTime - app.lastCallTime) * 24 * 3600;
                        app.RealtimeFPSLabel_3.Text = sprintf('Real-time FPS: %.0f', 1/interval);
                    end
                    app.lastCallTime = currentTime;
            end
            
        end
        function update_z_pos(app,~,~)
            app.zmmEditField.Value = System.Decimal.ToDouble(app.device.Position);
        end
        function update_xy_pos(app,~,~)
            try
                app.xmmEditField.Value = -queryPosition(app, 2);
                app.ymmEditField.Value = queryPosition(app, 1);
            end
        end
        function mm = queryPosition(app, chan)
            flush(app.device_xy);
            cmd = uint8([0x0A,0x04, chan, 0x00, 0x00, 0x00]);
            write(app.device_xy, cmd, "uint8");
            tStart = tic;
            while app.device_xy.NumBytesAvailable <12 && toc(tStart) < 2
                pause(0.001);
            end
            resp = read(app.device_xy, 12, "uint8");
            if ~isempty(resp)
                cnt  = typecast(uint8(resp(9:12)), "int32");
                % PLS-XY 分辨率 0.2116667 μm/count
                conv_um_per_count = 0.2116667;
                mm = double(cnt) * conv_um_per_count / 1000;
            else
                mm = 0;
            end
        end
        function goToPosition(app, chan, mm, tol)
            if nargin<4
                tol = 15e-3;
            end
            stop(app.Timer_stage_xy);pause(0.2);
            % 1) Header (bytes 0–5)
            h = msgbox('Moving...');
            header = uint8([ ...
                0x53, ...  % cmd低字节
                0x04, ...  % cmd高字节
                0x06, 0x00, 0x00, 0x00]);
            % 以 PLS-XY 为例，分辨率 0.2116667 μm/count
            conv_um_per_count = 0.2116667;
            count = round(mm * 1000 / conv_um_per_count);
            chanBytes = typecast(uint16(chan), 'uint8');
            countBytes = typecast(int32(count), 'uint8');
            msg = [header, chanBytes, countBytes];
            flush(app.device_xy);
            write(app.device_xy, msg, "uint8");
            while 1
                pos = queryPosition(app, chan);
                err = pos-mm;
                msgbox(sprintf('Err: %.0f um',err*1000),'modal');
                if abs(err) < tol
                    break;
                end
                pause(0.3);
            end
            pause(0.4);
            try
            close(h)
            end
            start(app.Timer_stage_xy);
        end
        function setZero(app)
            % stop(app.Timer_stage_xy);pause(0.2);
            header = uint8([ ...
                0x09, 0x04, ...  
                0x06, 0x00, 0x00, 0x00]);
            for chan =1:2
                chanBytes = typecast(uint16(chan), 'uint8');
                countBytes = typecast(int32(0), 'uint8');
                msg = [header, chanBytes, countBytes];
                flush(app.device_xy);
                write(app.device_xy, msg, "uint8");
                pause(0.2);
            end
            % start(app.Timer_stage_xy);
        end
        function update_rot(app,~,~)
            app.PosdegEditField.Value = System.Decimal.ToDouble(app.device_rot.Position);
        end
        function pause_update(app)
            app.RealtimecaptureSwitch.Value = 'Off';
            RealtimecaptureSwitchValueChanged(app)
        end
        function resume_update(app)
            app.RealtimecaptureSwitch.Value = 'On';
            RealtimecaptureSwitchValueChanged(app)
        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            app.cal_path = 'E:\';
            app.n2n_path = [];
            addpath('utils\')
            app.dark_param = struct;
            app.dark_param.if_dark = 0;
            app.dark_param.if_tophat = 0;
            app.dark_param.tophat_roll = 1;
            app.dark_param.if_bad = 1;
            app.dark_param.if_clip =0;

            app.dark_param.strel_size = 30;
            app.dark_param.Wavelength = 1100;
            app.dark_param.PixelSizeX = 17000;
            app.dark_param.PixelSizeY = 17000;
            app.dark_param.NA = 0.16;
            app.dark_param.DivideCoeff = 0.5;
            app.dark_param.Threshold = 70;
            app.dark_param.Denoise = 0;
            app.dark_param.BackgroundMode = 1;

            app.dark_param.zScore = 2;
            app.dark_param.WinSize = 3;
            app.dark_param.MaxSize = 3;
            app.dark_param.clip = [0,1.0];

            app.dark_param_3d = struct;
            app.dark_param_3d.if_dark = false;
            app.dark_param_3d.Wavelength = 1100;
            app.dark_param_3d.PixelSizeX = 17000;
            app.dark_param_3d.PixelSizeY = 17000;
            app.dark_param_3d.NA = 0.16;
            app.dark_param_3d.DivideCoeff = 0.5;
            app.dark_param_3d.Threshold = 70;
            app.dark_param_3d.Denoise = 0;
            app.dark_param_3d.BackgroundMode = 1;

            app.cal1_flag = false;
            app.cal2_flag = false;
            app.DSliceSlider.Enable = false;
            app.scanSpeed = 3;
            app.scanCount = 0;
            app.recon_ready_flag = false;
            app.scale_bar_pos = [0.8,0.1];
        end

        % Button pushed function: StartgeometricregistrationButton
        function StartgeometricregistrationButtonPushed(app, event)
            % Disable Plot Options button while dialog is open
            app.StartgeometricregistrationButton.Enable = "off";
            % Call dialog box with input values
            app.CalApp1 = cal_registration(app,app.cal_path);
            
        end

        % Button pushed function: StartPSFestimationButton
        function StartPSFestimationButtonPushed(app, event)
            % Disable Plot Options button while dialog is open
            app.StartPSFestimationButton.Enable = "off";
            % Call dialog box with input values
            app.CalApp2 = cal_psf_estimation(app,app.cal_file1);
            
        end

        % Button pushed function: LoadGeoFileButton
        function LoadGeoFileButtonPushed(app, event)
            if ~isequal(app.cal_path,0)
                [file,location] = uigetfile(fullfile(app.cal_path,'*.mat'));
            else
                [file,location] = uigetfile('*.mat');
            end
            figure(app.UIFigure); drawnow
            if file~=0
                app.cal_path = location;
                app.cal_file1 = fullfile(location,file);
                load_cal1(app);
            end
            if app.cal1_flag
                app.GeoloadedLamp.Color=[0,1,0];
            end
        end

        % Button pushed function: LoadPSFsButton
        function LoadPSFsButtonPushed(app, event)
            if ~isequal(app.cal_path,0)
                [file,location] = uigetfile(fullfile(app.cal_path,'*.mat'));
            else
                [file,location] = uigetfile('*.mat');
            end
            figure(app.UIFigure); drawnow
            if file~=0
                app.cal_path = location;
                app.cal_file2 = fullfile(location,file);
                load_cal2(app);
            end
            if app.cal2_flag
                app.PSFsloadedLamp.Color=[0,1,0];
            end
        end

        % Button pushed function: OpenDataFolderButton
        function OpenDataFolderButtonPushed(app, event)
            warning('off')
            % 让用户选择一个文件夹
            app.data_path = uigetdir('E:\');
            if app.data_path == 0
                % 如果用户取消了选择，就退出函数
                return;
            end
            figure(app.UIFigure);drawnow
            % update folder list
            init_data_folder(app);
        end

        % Value changed function: ListBox
        function ListBoxValueChanged(app, event)
            % tic;
            filename = app.ListBox.Value;
            [~,name,ext] = fileparts(filename);
            
            if strcmp(ext,'.hdf5')
                app.im_data = load_hdf5_cam(app.data_path,filename);
                app.im_data = (double(app.im_data));
            % elseif strcmp(ext,'.mat')
            %     S = load(fullfile(app.data_path,filename));
            %     app.im_data = permute(S.imgcube,[2,1,3]);
            elseif strcmp(ext,'.raw')
                app.im_data = load_raw_block(app.data_path,filename);
                app.im_data = (double(app.im_data));
            end
            app.recon_ready_flag = false;

            init_2D(app);
            update_2D(app);

            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                Simple3DTabButtonDown(app);
            end
            if strcmp(app.TabGroup.SelectedTab.Title, 'ROI')
                ROITabButtonDown(app);
            end

            fname_txt = fullfile(app.data_path,[name,'.txt']);
            if isfile(fname_txt)
                fid = fopen(fname_txt,'r');

                % 读取depth_range行
                line1 = fgetl(fid);
                depth_values = sscanf(line1, 'depth_range = [%f, %f]');

                % 读取PSF_n行
                line2 = fgetl(fid);
                PSF_n_value = sscanf(line2, 'PSF_n = %f');

                % 读取nIters行
                line3 = fgetl(fid);
                nIters_value = sscanf(line3, 'nIters = %d');

                fclose(fid);

                % 更新控件值
                if ~isempty(nIters_value)
                    app.nItersSpinner.Value = nIters_value;
                end
                if length(depth_values) >= 2
                    app.DepthrangeEditField.Value = depth_values(1);
                    app.EditField.Value = depth_values(2);
                    EditFieldValueChanged(app);
                end
                if ~isempty(PSF_n_value)
                    app.PSFnEditField.Value = PSF_n_value;
                    PSFnEditFieldValueChanged(app);
                end

            end

        end

        % Value changed function: DclipSlider
        function DclipSliderValueChanged(app, event)
            init_2D(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Button pushed function: AutoButton
        function AutoButtonPushed(app, event)
            num_avg = app.TakeavgperEditField.Value;
            idx_img = (round(app.FrameSlider.Value)-1) * num_avg + (1:num_avg);
            min_val_auto = double(prctile(reshape(rescale(app.im_data(:,:,idx_img)),1,[]), 2)); % 2nd percentile
            max_val_auto = double(prctile(reshape(rescale(app.im_data(:,:,idx_img)),1,[]), 99)); % 98th percentile
            app.DclipSlider.Limits = [0,1];
            app.DclipSlider.Value = [min_val_auto,max_val_auto];
            init_2D(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Button pushed function: ReconButton
        function ReconButtonPushed(app, event)
            %% get iter parameters
            app.Lamp_2.Color = [1,0,0];
            drawnow
            if ~app.recon_ready_flag
                PrepareReconButtonPushed(app);
            end

            nIter = app.nItersSpinner.Value;
            readout = app.ReadouteEditField.Value/2.36/16384;
            view_idx_list = get_view_idx_list(app);

            %% fwd/bkwd function
            forward_fun = forward_model(app.otf_gpu(:,:,:,view_idx_list));
            backward_fun = backward_model(app.otf_gpu(:,:,:,view_idx_list));

            app.reconResult = deconvlucy_gpu(app.measurements_trans(:,:,view_idx_list),...
                           forward_fun, backward_fun, ...
                           nIter, app.DamparEditField.Value, app.roi_mask_gpu(:,:,view_idx_list), ...
                           readout);

            % app.reconResult = RL_deconv_solver(app.measurements_trans,...
            %     app.roi_mask_gpu,...
            %     app.obj_mask_gpu,...
            %     nIter,...
            %     forward_fun, backward_fun,...
            %     0.5);
            app.reconResult = rescale(app.reconResult);
            init_output_mask(app);
            init_3D(app);
            app.DSliceSlider.Enable = true;
            app.Lamp_2.Color = [0,1,0];

        end

        % Value changed function: GammaSlider
        function GammaSliderValueChanged(app, event)
            init_2D(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Button pushed function: ResetButton
        function ResetButtonPushed(app, event)
            % update show
            app.DclipSlider.Limits = [0,1];
            app.DclipSlider.Value = [0,1];
            app.DclipSlider.Step = 0.01;
            app.GammaSlider.Value = 1;
            init_2D(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: ROIEditField
        function ROIEditFieldValueChanged(app, event)
            init_interp(app);
            init_roi_mask(app);
            init_roi_plot(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: DSliceSlider
        function DSliceSliderValueChanged(app, event)
            update_3D(app,app.DSliceSlider.Value);
        end

        % Value changed function: PlayButton
        function PlayButtonValueChanged(app, event)
            if app.PlayButton.Value
                period = 1/app.PlayfpsEditField.Value;
                app.Timer_3D = timer('ExecutionMode','fixedRate','Period',period,'TimerFcn',@app.play_step_3D);
                start(app.Timer_3D)
            else
                if isa(app.Timer_3D,'timer')
                    if isvalid(app.Timer_3D)
                        stop(app.Timer_3D)
                        delete(app.Timer_3D)
                    end
                end
            end

        end

        % Value changing function: DSliceSlider
        function DSliceSliderValueChanging(app, event)
            update_3D(app,event.Value);
        end

        % Value changed function: RealtimecaptureSwitch
        function RealtimecaptureSwitchValueChanged(app, event)
            N = app.AvgEditField.Value;
            
            app.im_data = mean(permute(bufferN(app, N),[2,1,3]), 3);
            switch app.TabGroup.SelectedTab.Title
                case 'Raw'
                    init_2D(app);
                case 'Simple 3D'
                    init_data(app);
                    init_interp(app);
                    init_roi_mask(app);
                    init_obj_mask(app);
                    init_3D_simple(app);
                case 'ROI'
                    init_data(app);
                    init_interp(app);
                    init_roi_mask(app);
                    init_roi_plot(app);
            end
            value = app.RealtimecaptureSwitch.Value;
            if strcmp(value,'Off')
                if isa(app.Timer_plot, 'timer') && isvalid(app.Timer_plot)
                    stop(app.Timer_plot);
                    delete(app.Timer_plot);
                end
                if isa(app.Timer_buffer, 'timer') && isvalid(app.Timer_buffer)
                    stop(app.Timer_buffer);
                    delete(app.Timer_buffer);
                end
            else
                % refresh rate for buffer loading
                t_grab = 1/app.FPSEditField.Value*app.AvgEditField.Value;
                period = max(t_grab, 0.1);
                % start timer for buffer
                app.Timer_buffer = timer('ExecutionMode','fixedRate','Period',period,'TimerFcn',@app.update_buffer);
                start(app.Timer_buffer);
                % start timer for plot updater
                period = max(t_grab, 0.25);
                app.Timer_plot = timer('ExecutionMode','fixedRate','Period',period,'TimerFcn',@app.refresh);
                start(app.Timer_plot);
            end
                        
        end

        % Value changed function: avgnumberEditField
        function avgnumberEditFieldValueChanged(app, event)
            init_roi_mask(app);
            init_roi_plot(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: ROIdepthcenterSlider
        function ROIdepthcenterSliderValueChanged(app, event)
            init_roi_mask(app);
            init_roi_plot(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: ROIdepthfullbandwidthSlider
        function ROIdepthfullbandwidthSliderValueChanged(app, event)
            init_roi_mask(app);
            init_roi_plot(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Button down function: ROITab
        function ROITabButtonDown(app, event)
            if isa(app.Timer_buffer, 'timer')
                if isvalid(app.Timer_buffer)
                    stop(app.Timer_buffer);
                end
            end
            
            init_data(app);
            init_interp(app);
            init_roi_mask(app);
            init_roi_plot(app);
            drawnow

            if isa(app.Timer_buffer, 'timer')
                if isvalid(app.Timer_buffer)
                    start(app.Timer_buffer);
                end
            end
        end

        % Button down function: Simple3DTab
        function Simple3DTabButtonDown(app, event)
            if isa(app.Timer_plot, 'timer')
                if isvalid(app.Timer_plot)
                    stop(app.Timer_plot);
                end
            end

            init_data(app);
            init_interp(app);

            app.MaskCenterXSpinner.Value = floor(app.stretched_size(2)/2);
            app.MaskCenterYSpinner.Value = floor(app.stretched_size(1)/2);

            init_roi_mask(app);
            init_obj_mask(app);
            init_3D_simple(app);   
            
            if isa(app.Timer_plot, 'timer')
                if isvalid(app.Timer_plot)
                    start(app.Timer_plot);
                end
            end
        end

        % Button pushed function: AutoButton_2
        function AutoButton_2Pushed(app, event)
            min_val_auto = double(prctile(app.ssResult(:), 2)); % 2nd percentile
            max_val_auto = double(prctile(app.ssResult(:), 99)); % 98th percentile
            app.DclipSlider_2.Limits = [0,1];
            app.DclipSlider_2.Value = [min_val_auto,max_val_auto];
            % init_3D_simple(app);
        end

        % Value changed function: DepthrangeEditField
        function DepthrangeEditFieldValueChanged(app, event)
            app.DSliceSlider.Limits = [app.DepthrangeEditField.Value,app.EditField.Value];
            app.DSliceSlider_2.Limits = [app.DepthrangeEditField.Value,app.EditField.Value];
            app.DSliceSlider.Value = mean(app.DSliceSlider.Limits);
            app.DSliceSlider_2.Value = mean(app.DSliceSlider_2.Limits);
            init_data(app);
            init_interp(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: EditField
        function EditFieldValueChanged(app, event)
            app.DSliceSlider.Limits = [app.DepthrangeEditField.Value,app.EditField.Value];
            app.DSliceSlider_2.Limits = [app.DepthrangeEditField.Value,app.EditField.Value];
            app.DSliceSlider.Value = mean(app.DSliceSlider.Limits);
            app.DSliceSlider_2.Value = mean(app.DSliceSlider_2.Limits);
            init_data(app);
            init_interp(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Button pushed function: ResetButton_2
        function ResetButton_2Pushed(app, event)
            try
                if isa(app.Timer_plot, 'timer')
                stop(app.Timer_plot);
                end
            end
            app.DepthrangeEditField.Value = 0;
            app.EditField.Value=1;
            DepthrangeEditFieldValueChanged(app, event);
            try
                if isa(app.Timer_plot, 'timer')
                    start(app.Timer_plot);
                end
            end
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: ConnectCamButton
        function ConnectCamButtonValueChanged(app, event)
            if app.ConnectCamButton.Value
                %% init camera
                fps = 20;
                tint = 1/fps;
                build_bias = false;
                burst = false;
                [app.context,ok] = init_camera(fps,tint,build_bias,burst);
                if ~ok
                    app.ConnectCamButton.Value=false;
                    return
                end
                [app.width, app.height] = FliSdk.getCurrentImageDimension(app.context);
                app.bufferSize = FliSdk.getbufferImagesCapacity(app.context);
                app.RealtimecaptureSwitch.Enable = true;
                % app.CaptureCalibButton.Enable = true;
                % auto update status
                app.Timer_status = timer('ExecutionMode','fixedRate','Period',1,'TimerFcn',@app.update_status);
                start(app.Timer_status);
            else
                %% disconnect
                stop(app.Timer_status);
                delete(app.Timer_status);
                FliSdk.stop(app.context);
                FliSdk.exit(app.context);
                FliSdk.closeLib();
                app.RealtimecaptureSwitch.Enable = false;
                % app.CaptureCalibButton.Enable = false;
                msgbox('Camera Disconnected!')
            end
        end

        % Value changed function: FPSEditField
        function FPSEditFieldValueChanged(app, event)
            value = app.FPSEditField.Value;
            fps = value; tint = 1/fps;
            [ok] = FliSdk.setCameraFps(app.context, fps); [ok, fps] = FliSdk.getCameraFps(app.context);
            [ok] = FliSdk.setTint(app.context, tint); [ok, tint] = FliSdk.getTint(app.context);
            msgbox([sprintf("Current Tint: %.4f ms\n",tint*1e3);sprintf("Current FPS: %.4f",fps)]);
        end

        % Close request function: UIFigure
        function UIFigureCloseRequest(app, event)
            % disconnect
            if app.ConnectCamButton.Value
                FliSdk.stop(app.context);
                FliSdk.exit(app.context);
                FliSdk.closeLib();
                % msgbox('Camera Disconnected!')
            end
            if app.ConnectzstageButton.Value
                stop(app.Timer_stage);
                delete(app.Timer_stage);
                app.device.StopPolling();
                app.device.Disconnect();
            end
            if app.ConnectrotstageButton.Value
                stop(app.Timer_rot);
                delete(app.Timer_rot);
                app.device_rot.StopPolling();
                app.device_rot.Disconnect();
            end
            delete(app)
            
        end

        % Value changed function: TempSpinner
        function TempSpinnerValueChanged(app, event)
            value = app.TempSpinner.Value;
            [ok, response] = FliSdk.sendCommandToCamera(app.context, sprintf('set temperatures sensor %.0f',value));
            msgbox(["Set temperature:";response])
        end

        % Button pushed function: BuildbiasButton
        function BuildbiasButtonPushed(app, event)
            nb = 256;
            FliSdk.sendCommandToCamera(app.context, 'set bias off'); 
            pause(1);
            
            FliSdk.sendCommandToCamera(app.context, sprintf('buildnuc bias %d',nb));
            [~, response] = FliSdk.sendCommandToCamera(app.context, 'buildnuc progress');
            
            total_time = sscanf(response,'Result: %d');
            f = waitbar(0,'Build Bias...');
            remain_time = total_time;
            while(remain_time>0)
                pause(0.5);
                [~, response] = FliSdk.sendCommandToCamera(app.context, 'buildnuc progress');
                remain_time = sscanf(response,'Result: %d');
                waitbar((total_time-remain_time)/total_time,f,sprintf('%ds remaining...',remain_time));
            end
            close(f);
            FliSdk.sendCommandToCamera(app.context, 'set bias on'); pause(1);
            msgbox('Built bias successfully!')
        end

        % Value changed function: ConnectzstageButton
        function ConnectzstageButtonValueChanged(app, event)
            if app.ConnectzstageButton.Value
                [app.device,app.genCLI] = init_stage(false,2);
                pos = System.Decimal.ToDouble(app.device.Position);
                app.zmmEditField.Value = pos;
                app.JogEditField.Value = System.Decimal.ToDouble(app.device.GetJogStepSize());

                % auto update
                app.Timer_stage = timer('ExecutionMode','fixedRate','Period',0.25,'TimerFcn',@app.update_z_pos);
                start(app.Timer_stage);
                
            else
                stop(app.Timer_stage);
                delete(app.Timer_stage);
                app.device.StopPolling();
                app.device.Disconnect();
                msgbox('Stage Disconnected')
            end
        end

        % Value changed function: zmmEditField
        function zmmEditFieldValueChanged(app, event)
            prev = event.PreviousValue;
            value = app.zmmEditField.Value;
            if abs(prev-value)>1
                msg = [sprintf("Move from %.5g to %.5g", prev, value);"which is > 1mm. Confirm?"];
                selection = uiconfirm(app.UIFigure,msg,'Warning');
                switch selection
                    case 'OK'
                        app.device.MoveTo(value, 60000);
                        % msgbox(sprintf('Motor pos %.4f',System.Decimal.ToDouble(app.device.Position)));
                        return
                    case 'Cancel'
                        return
                end
            else
                app.device.MoveTo(value, 60000);
            end
        end

        % Value changed function: JogEditField
        function JogEditFieldValueChanged(app, event)
            value = app.JogEditField.Value;
            app.device.SetJogStepSize(value);
            % msgbox(sprintf('Jog step: %.4f',System.Decimal.ToDouble(app.device.GetJogStepSize())));
        end

        % Button pushed function: Button
        function ButtonPushed(app, event)
            msgbox("Stage Homing...");
            app.device.Home(60000);
            msgbox("Stage Homed!",'modal')
        end

        % Button pushed function: ButtonDownJog
        function ButtonDownJogPushed(app, event)
            optionTypeHandle = app.genCLI.AssemblyHandle.GetType('Thorlabs.MotionControl.GenericMotorCLI.MotorDirection');
            enum = optionTypeHandle.GetEnumValues();
            direction = enum.Get(1); % Backward
            app.device.MoveJog(direction, 60000);
        end

        % Button pushed function: ButtonUpJog
        function ButtonUpJogPushed(app, event)
            optionTypeHandle = app.genCLI.AssemblyHandle.GetType('Thorlabs.MotionControl.GenericMotorCLI.MotorDirection');
            enum = optionTypeHandle.GetEnumValues();
            direction = enum.Get(0); % Forward
            app.device.MoveJog(direction, 60000);
        end

        % Button pushed function: CaptureCalibButton
        function CaptureCalibButtonPushed(app, event)
            % Disable Plot Options button while dialog is open
            % app.CaptureCalibButton.Enable = "off";
            % Call dialog box with input values
            app.CalApp3 = cal_capture(app,app.cal_path);
        end

        % Button pushed function: CaptureNButton
        function CaptureNButtonPushed(app, event)
            %% stop grabbing
            % if strcmp(app.RealtimecaptureSwitch.Value,'On')
            %     app.RealtimecaptureSwitch.Value = 'Off';
            %     RealtimecaptureSwitchValueChanged(app);
            % end
            %% capture
            N = app.NEditField.Value;
            out_ratio = app.OutlierEditField.Value;
            imgcube = captureN(app, N, out_ratio);
            %% save
            filename = '.mat';
            [filename, pathname] = uiputfile(fullfile(app.data_path,filename),...
                'Save data:');
            if ~isequal(filename,0) && ~isequal(pathname,0)
                app.data_path = pathname;
                FileName = fullfile(pathname, filename);
                save(FileName,'imgcube');
            end
            % %% show
            % figure
            % subplot(121)
            % imagesc(mean(rescale(permute(imgcube,[2 1 3])),3))
            % colormap('gray');title('Mean.')
            % subplot(122)
            % imagesc(std(rescale(permute(imgcube,[2 1 3])),0,3))
            % colormap('gray');title('Var.')
            %% stop grabbing
            if strcmp(app.RealtimecaptureSwitch.Value,'On')
                app.RealtimecaptureSwitch.Value = 'Off';
                RealtimecaptureSwitchValueChanged(app);
            end
            %% update
            app.im_data = permute(imgcube,[2,1,3]);
            % app.im_data = mean(imgcube,3)';
            init_2D(app);
            % try
            %     init_data(app);
            %     init_3D_simple(app);
            %     init_roi_mask(app);
            %     init_roi_plot(app);
            % end
            
            %% update folder list
            init_data_folder(app);

            msgbox('GrabN finished!')
        end

        % Button down function: RawTab
        function RawTabButtonDown(app, event)
            if isa(app.Timer_plot, 'timer')
                if isvalid(app.Timer_plot)
                    stop(app.Timer_plot);
                end
            end
            init_2D(app);
            if isa(app.Timer_plot, 'timer')
                if isvalid(app.Timer_plot)
                    start(app.Timer_plot);
                end
            end
        end

        % Value changed function: DSliceSlider_2
        function DSliceSlider_2ValueChanged(app, event)
            % get index
            depth_idx = lam2idx(app,app.DSliceSlider_2.Value);
            view_idx_list = get_view_idx_list(app);
            % update plot
            update_3D_simple(app,depth_idx,view_idx_list);
        end

        % Value changed function: ScanButton
        function ScanButtonValueChanged(app, event)
            app.scanSpeed = 3;
            app.scanCount = 0;
        end

        % Value changed function: DclipSlider_2
        function DclipSlider_2ValueChanged(app, event)
            depth_idx = lam2idx(app, app.DSliceSlider_2.Value);
            view_idx_list = get_view_idx_list(app);
            update_3D_simple(app,depth_idx,view_idx_list);
        end

        % Value changing function: DSliceSlider_2
        function DSliceSlider_2ValueChanging(app, event)
            depth_idx = lam2idx(app, event.Value);
            view_idx_list = get_view_idx_list(app);
            % update plot
            update_3D_simple(app,depth_idx,view_idx_list);
        end

        % Button pushed function: ButtonUpRange
        function ButtonUpRangePushed(app, event)
            stop(app.Timer_plot);
            v1 = app.DepthrangeEditField.Value;
            v2 = app.EditField.Value;
            app.DepthrangeEditField.Value = round(v1 - (v2-v1)/4, 2);
            app.EditField.Value = round(v2 + (v2-v1)/4, 2);
            DepthrangeEditFieldValueChanged(app);
            start(app.Timer_plot);
        end

        % Button pushed function: ButtonDownRange
        function ButtonDownRangePushed(app, event)
            stop(app.Timer_plot);
            v1 = app.DepthrangeEditField.Value;
            v2 = app.EditField.Value;
            app.DepthrangeEditField.Value = round(v1 + (v2-v1)/4, 2);
            app.EditField.Value = round(v2 - (v2-v1)/4, 2);
            DepthrangeEditFieldValueChanged(app);
            start(app.Timer_plot);
        end

        % Button pushed function: ButtonDownRange_2
        function ButtonDownRange_2Pushed(app, event)
            lam = app.DSliceSlider_2.Value;
            num_depth = app.nDepthEditField.Value;
            lam_new = lam - (app.DSliceSlider_2.Limits(2)-app.DSliceSlider_2.Limits(1))/(num_depth-1);
            if lam_new < 0
                lam_new = 0;
            end
            app.DSliceSlider_2.Value = lam_new;
            app.DSliceSlider_2ValueChanged();
        end

        % Button pushed function: ButtonUpRange_2
        function ButtonUpRange_2Pushed(app, event)
            % app.scanSpeed = max(app.scanSpeed-1, 1);
            lam = app.DSliceSlider_2.Value;
            num_depth = app.nDepthEditField.Value;
            lam_new = lam + (app.DSliceSlider_2.Limits(2)-app.DSliceSlider_2.Limits(1))/(num_depth-1);
            if lam_new > app.DSliceSlider_2.Limits(2)
                lam_new = app.DSliceSlider_2.Limits(1);
            end
            app.DSliceSlider_2.Value = lam_new;
            app.DSliceSlider_2ValueChanged();
        end

        % Value changed function: CheckBox1
        function CheckBox1ValueChanged(app, event)
            depth_idx = lam2idx(app, app.DSliceSlider_2.Value);
            view_idx_list = get_view_idx_list(app);
            update_3D_simple(app,depth_idx,view_idx_list);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: CheckBox2
        function CheckBox2ValueChanged(app, event)
            depth_idx = lam2idx(app, app.DSliceSlider_2.Value);
            view_idx_list = get_view_idx_list(app);
            update_3D_simple(app,depth_idx,view_idx_list);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: CheckBox3
        function CheckBox3ValueChanged(app, event)
            depth_idx = lam2idx(app, app.DSliceSlider_2.Value);
            view_idx_list = get_view_idx_list(app);
            update_3D_simple(app,depth_idx,view_idx_list);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: CheckBox4
        function CheckBox4ValueChanged(app, event)
            depth_idx = lam2idx(app, app.DSliceSlider_2.Value);
            view_idx_list = get_view_idx_list(app);
            update_3D_simple(app,depth_idx,view_idx_list);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: CheckBox5
        function CheckBox5ValueChanged(app, event)
            depth_idx = lam2idx(app, app.DSliceSlider_2.Value);
            view_idx_list = get_view_idx_list(app);
            update_3D_simple(app,depth_idx,view_idx_list);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Button down function: DTab_2
        function DTab_2ButtonDown(app, event)
            app.MaskCenterXSpinner_2.Value = floor(app.stretched_size(2)/2);
            app.MaskCenterYSpinner_2.Value = floor(app.stretched_size(1)/2);
            try
                init_output_mask(app);
                init_3D(app);
            catch
                msgbox('Please click reconstruct first')
            end
        end

        % Value changed function: ColorDropDown
        function ColorDropDownValueChanged(app, event)
            value = app.ColorDropDown.Value;
            colormap(app.UIAxes2,value)
            drawnow
        end

        % Button pushed function: SaveMovieButton
        function SaveMovieButtonPushed(app, event)
            app.PlayButton.Value = false;
            PlayButtonValueChanged(app);
            % 初始化变量
            frameRate = app.PlayfpsEditField.Value; % 设置帧率（帧每秒）
            h = msgbox('Generating...');
            % 创建一个动画并捕获帧
            frames = [];
            app.DSliceSlider.Value = app.DSliceSlider.Limits(1);
            f = figure;
            while(1)
                new_idx = lam2idx(app,app.DSliceSlider.Value);
                new_lam = app.DSliceSlider.Value;
                depth = new_lam*app.FullrangeumEditField.Value - app.FullrangeumEditField.Value/2;
                reconFrame = rescale(app.reconResult(:,:,new_idx).*app.output_mask);
                reconFrame = imadjust(reconFrame, app.MIP3DclampSlider.Value, [0,1], app.MIP3DgammaEditField.Value);
                imagesc(reconFrame);
                colormap(app.ColorDropDown.Value)
                title( sprintf('Depth #%d/%d, %.0f um', new_idx, size(app.reconResult,3), depth))
                addScaleBar(gca,100/app.pixel_size_x, '100um', 14,0.01,app.scale_bar_pos);
                drawnow
                frames = [frames, getframe(gcf)];
                % update
                new_idx = new_idx +1;
                new_lam = idx2lam(app,new_idx);
                if new_lam>app.DSliceSlider.Limits(2)
                    break;
                end
                app.DSliceSlider.Value = new_lam;
            end 
            close(f)

            % 保存为GIF
            [~,filename,~] = fileparts(app.ListBox.Value);
            filename = [filename '.gif'];
            [filename, pathname] = uiputfile(fullfile(app.data_path,filename),...
                'Save movie:');
            if isequal(filename,0) || isequal(pathname,0)
                return;
            end
            FileName = fullfile(pathname, filename);
            for i = 1:length(frames)
                % 将帧转换为图像数据
                [imind, cm] = rgb2ind(frame2im(frames(i)), 256);
                % 写入GIF文件
                if i == 1
                    imwrite(imind, cm, FileName, 'gif', 'LoopCount', Inf, 'DelayTime', 1/frameRate);
                else
                    imwrite(imind, cm, FileName, 'gif', 'WriteMode', 'append', 'DelayTime', 1/frameRate);
                end
            end
            if isvalid(h)
                close(h);
            end
        end

        % Value changed function: ConnectrotstageButton
        function ConnectrotstageButtonValueChanged(app, event)
            value = app.ConnectrotstageButton.Value;
            if value
                home = false;
                [app.device_rot,~] = init_stage(home,1);
                pos = System.Decimal.ToDouble(app.device_rot.Position);
                app.PosdegEditField.Value = pos;
                % auto update
                app.Timer_rot = timer('ExecutionMode','fixedRate','Period',0.25,'TimerFcn',@app.update_rot);
                start(app.Timer_rot);
                app.device_rot.SetVelocityParams(25,25);
                msgbox('rot Stage Connected!')
            else
                stop(app.Timer_rot);
                delete(app.Timer_rot);
                app.device_rot.StopPolling();
                app.device_rot.Disconnect();
                msgbox('rot Stage Disconnected!')
            end
        end

        % Value changed function: PosdegEditField
        function PosdegEditFieldValueChanged(app, event)
            value = app.PosdegEditField.Value;
            app.device_rot.MoveTo(value, 60000);
        end

        % Value changed function: SyncwhencaptureNButton
        function SyncwhencaptureNButtonValueChanged(app, event)
            value = app.SyncwhencaptureNButton.Value;
            if value
                app.sync = true;
            else
                app.sync = false;
            end
        end

        % Value changed function: VarButton
        function VarButtonValueChanged(app, event)
            init_2D(app);
            % try
            %     init_data(app);
            %     init_3D_simple(app);
            %     init_roi_mask(app);
            %     init_roi_plot(app);
            % end
        end

        % Button pushed function: Button_plus
        function Button_plusPushed(app, event)
            value = app.DclipSlider.Value;
            range = app.DclipSlider.Limits;
            app.DclipSlider.Limits = [0,round((value(2)+range(2))/2,2,"significant")];
            app.DclipSlider.Step = min(diff(app.DclipSlider.Limits)/100,0.001);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Button pushed function: Button_minus
        function Button_minusPushed(app, event)
            value = app.DclipSlider.Value;
            range = app.DclipSlider.Limits;
            app.DclipSlider.Limits = [0, min(round(range(2)+(range(2)-value(2)),2,"significant"), 1)];
            app.DclipSlider.Step = min(diff(app.DclipSlider.Limits)/100,0.001);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Button pushed function: Button_plus_2
        function Button_plus_2Pushed(app, event)
            value = app.DclipSlider_2.Value;
            range = app.DclipSlider_2.Limits;
            app.DclipSlider_2.Limits = [0,round((value(2)+range(2))/2,2,"significant")];
            app.DclipSlider_2.Step = diff(app.DclipSlider_2.Limits)/100;
        end

        % Button pushed function: Button_minus_2
        function Button_minus_2Pushed(app, event)
            value = app.DclipSlider_2.Value;
            range = app.DclipSlider_2.Limits;
            app.DclipSlider_2.Limits = [0,min(round(range(2)+(range(2)-value(2)),2,"significant"),1)];
            app.DclipSlider_2.Step = diff(app.DclipSlider_2.Limits)/100;
        end

        % Double-clicked callback: ListBox
        function ListBoxDoubleClicked(app, event)
            ListBoxValueChanged(app);
        end

        % Value changed function: MaskCenterXSpinner
        function MaskCenterXSpinnerValueChanged(app, event)
            %% masks
            init_obj_mask(app);
            init_3D_simple(app);   
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: MaskCenterYSpinner
        function MaskCenterYSpinnerValueChanged(app, event)
            init_obj_mask(app);
            init_3D_simple(app);   
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: MaskROISpinner
        function MaskROISpinnerValueChanged(app, event)
            init_obj_mask(app);
            init_3D_simple(app);   
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: BlurradiusSpinner
        function BlurradiusSpinnerValueChanged(app, event)
            init_obj_mask(app);
            init_3D_simple(app);   
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: MaskCenterXSpinner_2
        function MaskCenterXSpinner_2ValueChanged(app, event)
            init_output_mask(app);
            init_3D(app);
        end

        % Value changed function: MaskCenterYSpinner_2
        function MaskCenterYSpinner_2ValueChanged(app, event)
            init_output_mask(app);
            init_3D(app);
        end

        % Value changed function: MaskROISpinner_2
        function MaskROISpinner_2ValueChanged(app, event)
            init_output_mask(app)
            init_3D(app);
        end

        % Value changed function: BlurradiusSpinner_2
        function BlurradiusSpinner_2ValueChanged(app, event)
            init_output_mask(app);
            init_3D(app);
        end

        % Button down function: UIAxes2
        function UIAxes2ButtonDown(app, event)
            % % 获取点击位置在坐标系中的位置
            % clickedPoint = app.UIAxes2.CurrentPoint;
            % % CurrentPoint返回一个2x3矩阵，第一行包含近平面点，第二行包含远平面点
            % % 通常我们使用第一行的x和y值
            % x = clickedPoint(1,1);
            % y = clickedPoint(1,2);
            % xl = app.UIAxes2.XLim;
            % yl = app.UIAxes2.YLim;
            % app.scale_bar_pos=[(x-xl(1))/diff(xl), (y-yl(1))/diff(yl)];
            % init_3D(app);
        end

        % Button pushed function: PrepareReconButton
        function PrepareReconButtonPushed(app, event)
            app.Lamp.Color = [1,0,0]; drawnow
            init_data(app);
            init_interp(app);
            init_roi_mask(app);
            init_obj_mask(app);
            init_psf(app);
            init_gpu(app);
            app.Lamp.Color = [0,1,0];
            app.recon_ready_flag = true; drawnow
        end

        % Value changed function: PSFnEditField
        function PSFnEditFieldValueChanged(app, event)
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: DiracPSFCheckBox
        function DiracPSFCheckBoxValueChanged(app, event)
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Value changed function: ConnectxystageButton
        function ConnectxystageButtonValueChanged(app, event)
            if app.ConnectxystageButton.Value
                port = "COM3";
                baud = 460800;
                app.device_xy = serialport(port, baud, "Timeout", 5,...
                    "DataBits", 8, "Parity", "none",...
                    "StopBits", 1, "FlowControl", "none");
                % auto update
                app.Timer_stage_xy = timer('ExecutionMode','fixedRate','Period',0.25,'TimerFcn',@app.update_xy_pos);
                start(app.Timer_stage_xy); 
            else
                try
                    stop(app.Timer_stage_xy);
                    delete(app.Timer_stage_xy);
                    clear app.device_xy;
                end
                msgbox('Stage XY Disconnected')
            end
        end

        % Value changed function: xmmEditField
        function xmmEditFieldValueChanged(app, event)
            value = app.xmmEditField.Value;
            goToPosition(app, 2, -value);
        end

        % Value changed function: ymmEditField
        function ymmEditFieldValueChanged(app, event)
            value = app.ymmEditField.Value;
            goToPosition(app, 1, value);
        end

        % Button pushed function: Button_2
        function Button_2Pushed(app, event)
            setZero(app);
        end

        % Button pushed function: Button_3
        function Button_3Pushed(app, event)
            goToPosition(app, 2, 0);
        end

        % Button pushed function: Button_4
        function Button_4Pushed(app, event)
            goToPosition(app, 1, 0);
        end

        % Button pushed function: CaptureNStitchButton
        function CaptureNStitchButtonPushed(app, event)
            %% stitch parameters
            x_step = app.StepXmmEditField.Value;
            y_step = app.StepYmmEditField.Value;
            num_x_stitch = app.StitchXSpinner.Value;
            num_y_stitch = app.StitchYSpinner.Value;

            %% capture stitch
            imgcell = cell(num_y_stitch,num_x_stitch);
            N = app.NEditField.Value;
            out_ratio = app.OutlierEditField.Value;
            x_pos0 = queryPosition(app,2);
            y_pos0 = queryPosition(app,1);
            for i=1:num_x_stitch
                % move x
                x_pos = x_pos0 + x_step * (i-1 - floor(num_x_stitch/2));
                goToPosition(app, 2, x_pos);
                pause(0.5);
                for j=1:num_y_stitch
                    % move y
                    y_pos = y_pos0 + y_step * (j-1 - floor(num_y_stitch/2));
                    goToPosition(app, 1, y_pos);
                    pause(0.5);
                    imgcell{j,i} = captureN(app, N, out_ratio);
                    pause(0.5);
                end
            end
            goToPosition(app, 1, y_pos0);
            goToPosition(app, 2, x_pos0);
            %% save
            defaultName = 'data.mat';
            [filename, pathname] = uiputfile('*.mat', 'Save data:', fullfile(app.data_path, defaultName));
            if isequal(filename,0) || isequal(pathname,0)
                % 用户取消
                return;
            end
            % 更新 app.data_path
            app.data_path = pathname;
            % 3) 拆分出 basename 和 ext
            [~, baseName, ext] = fileparts(filename);
            for i = 1:num_x_stitch
                for j = 1:num_y_stitch
                    % 取出要保存的变量
                    imgcube = imgcell{j,i};
                    % 拼接新文件名：basename_i_j.mat
                    newFileName = sprintf('%s_%d_%d%s', baseName, i, j, ext);
                    % 完整路径 + 保存
                    save(fullfile(pathname, newFileName), 'imgcube');
                end
            end

            %% stop grabbing
            if strcmp(app.RealtimecaptureSwitch.Value,'On')
                app.RealtimecaptureSwitch.Value = 'Off';
                RealtimecaptureSwitchValueChanged(app);
            end

            %% update with last imgcube
            app.im_data = permute(imgcube,[2,1,3]);
            % app.im_data = mean(imgcube,3)';
            init_2D(app);
            %% update folder list
            init_data_folder(app);
            msgbox('GrabN finished!')
        end

        % Value changed function: nDepthEditField
        function nDepthEditFieldValueChanged(app, event)
            init_data(app);
            init_interp(app);
            app.recon_ready_flag = false;
            app.Lamp.Color = [0.5,0.5,0.5];
        end

        % Button pushed function: x_jog_plus
        function x_jog_plusPushed(app, event)
            value = queryPosition(app,2) + app.StepXmmEditField.Value;
            goToPosition(app, 2, value);
        end

        % Button pushed function: x_jog_minus
        function x_jog_minusPushed(app, event)
            value = queryPosition(app,2) - app.StepXmmEditField.Value;
            goToPosition(app, 2, value);
        end

        % Button pushed function: y_jog_plus
        function y_jog_plusPushed(app, event)
            value = queryPosition(app,1)  + app.StepYmmEditField.Value;
            goToPosition(app, 1, value);
        end

        % Button pushed function: y_jog_minus
        function y_jog_minusPushed(app, event)
            value = queryPosition(app,1) - app.StepYmmEditField.Value;
            goToPosition(app, 1, value);
        end

        % Button pushed function: z_jog_range_plus
        function z_jog_range_plusButtonPushed(app, event)
            app.JogEditField.Value = app.JogEditField.Value*2;
            JogEditFieldValueChanged(app);
        end

        % Button pushed function: z_jog_range_minus
        function z_jog_range_minusButtonPushed(app, event)
            app.JogEditField.Value = round(app.JogEditField.Value/2, 2);
            JogEditFieldValueChanged(app);
        end

        % Button pushed function: xy_jog_range_plus
        function xy_jog_range_plusButtonPushed(app, event)
            app.StepXmmEditField.Value = app.StepXmmEditField.Value*2;
            app.StepYmmEditField.Value = app.StepYmmEditField.Value*2;
        end

        % Button pushed function: xy_jog_range_minus
        function xy_jog_range_minusButtonPushed(app, event)
            app.StepXmmEditField.Value = round(app.StepXmmEditField.Value/2, 1);
            app.StepYmmEditField.Value = round(app.StepYmmEditField.Value/2, 1);
        end

        % Button pushed function: z_jog_range_reset
        function z_jog_range_resetButtonPushed(app, event)
            app.JogEditField.Value = 0.1;
            JogEditFieldValueChanged(app);
        end

        % Button pushed function: xy_jog_range_reset
        function xy_jog_range_resetButtonPushed(app, event)
            app.StepXmmEditField.Value = 1.2;
            app.StepYmmEditField.Value = 1.2;
        end

        % Callback function
        function StitchButtonPushed(app, event)
            % 获取当前选中的文件名
            filename = app.ListBox.Value;
            % 分离文件名和扩展名
            [~, name, ext] = fileparts(filename);
            % 提取前缀(双序号前的部分)
            parts = split(name, '_');
            if length(parts) >= 3 && all(isstrprop(parts{end-1}, 'digit')) && all(isstrprop(parts{end}, 'digit'))
                % 如果当前文件有双序号后缀
                prefix = strjoin(parts(1:end-2), '_');
            else
                % 如果当前文件没有双序号后缀
                prefix = name;
            end
            % 初始化变量以跟踪最大行列索引
            max_row = 0;
            max_col = 0;

            % 第一次遍历：找出所有匹配文件并确定数组维度
            matching_files = {};
            indices = {};

            for i = 1:length(app.ListBox.Items)
                current_file = app.ListBox.Items{i};
                % 分离文件名和扩展名
                [~, current_name, current_ext] = fileparts(current_file);
                % 只处理与当前文件具有相同扩展名的文件
                if strcmp(current_ext, ext)
                    % 检查该文件是否具有相同前缀和双序号模式
                    parts = split(current_name, '_');
                    if length(parts) >= 3 && all(isstrprop(parts{end-1}, 'digit')) && all(isstrprop(parts{end}, 'digit'))
                        file_prefix = strjoin(parts(1:end-2), '_');

                        % 如果该文件与目标前缀相同
                        if strcmp(file_prefix, prefix)
                            % 提取行列索引
                            row = str2double(parts{end-1});
                            col = str2double(parts{end});

                            % 更新最大维度
                            max_row = max(max_row, row);
                            max_col = max(max_col, col);

                            % 存储文件名及其索引
                            matching_files{end+1} = current_file;
                            indices{end+1} = [row, col];
                        end
                    end
                end
            end

            % 初始化具有确定维度的cell数组
            recon_array = cell(max_row, max_col);

            % 第二次遍历：加载每个文件并存储数据
            % 3D重建参数
            app.Lamp_2.Color = [1,0,0]; drawnow
            nIter = app.nItersSpinner.Value;
            readout = app.ReadouteEditField.Value/2.36/16384;
            forward_fun = forward_model(app.otf_gpu);
            backward_fun = backward_model(app.otf_gpu);

            for i = 1:length(matching_files)
                % 获取文件名及其索引
                filename_with_suffix = matching_files{i};
                row_idx = indices{i}(1);
                col_idx = indices{i}(2);
                msgbox(['loading ' filename_with_suffix], 'modal');
                % 加载文件
                S = load(fullfile(app.data_path, filename_with_suffix));
                % 置换维度
                app.im_data = permute(S.imgcube, [2, 1, 3]);
                % 图像调整
                init_2D(app);
                % 重采样
                init_data(app);
                % 变换，上传gpu
                init_gpu(app);
                % 3D 重建
                h = msgbox(['Recon. ' filename_with_suffix], 'modal');
                recon_tmp = deconvlucy_gpu(app.measurements_trans,...
                    forward_fun, backward_fun, ...
                    nIter, app.DamparEditField.Value, app.roi_mask_gpu, ...
                    readout);
                recon_array{row_idx, col_idx} = rescale(recon_tmp.* app.output_mask);
                close(h);
            end
            app.Lamp_2.Color = [0,1,0]; drawnow

            interactive_stitch_viewer(recon_array);
        end

        % Value changed function: ScalebarXSpinner
        function ScalebarXSpinnerValueChanged2(app, event)
            init_3D(app);
        end

        % Value changed function: ScalebarYSpinner
        function ScalebarYSpinnerValueChanged(app, event)
            init_3D(app);
        end

        % Button pushed function: UpdateloadButton
        function UpdateloadButtonPushed(app, event)
            app.data_path = 'E:\C_RED2_snapshot\';
            % system('.\c_red_2_snap.ahk'); 
            % pause(0.2);
            figure(app.UIFigure); drawnow
            init_data_folder(app);
            ListBoxValueChanged(app);
        end

        % Value changed function: AutoButton_3
        function AutoButton_3ValueChanged(app, event)
            if app.AutoButton_3.Value
                % app.data_path = 'D:\C_RED2_recordings\';
                init_data_folder(app);
                ListBoxValueChanged(app);
                period = 1;
                app.Timer_snapshot = timer('ExecutionMode','fixedRate','Period',period,'TimerFcn',@(~,~)update_snap(app));
                start(app.Timer_snapshot)
            else
                if isa(app.Timer_snapshot,'timer')
                    if isvalid(app.Timer_snapshot)
                        stop(app.Timer_snapshot)
                        delete(app.Timer_snapshot)
                    end
                end
            end
        end

        % Value changed function: FrameSlider
        function FrameSliderValueChanged(app, event)
            update_2D(app);
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                Simple3DTabButtonDown(app);
            end
            if strcmp(app.TabGroup.SelectedTab.Title, 'ROI')
                ROITabButtonDown(app);
            end
            app.recon_ready_flag = false;
        end

        % Value changed function: TakeavgperEditField
        function TakeavgperEditFieldValueChanged(app, event)
            init_2D(app);
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                Simple3DTabButtonDown(app);
            end
            if strcmp(app.TabGroup.SelectedTab.Title, 'ROI')
                ROITabButtonDown(app);
            end
        end

        % Button pushed function: Button_plus_3
        function Button_plus_3Pushed(app, event)
            frame_range = app.FrameSlider.Limits;
            app.FrameSlider.Value = mod(app.FrameSlider.Value, frame_range(2))+1;
            update_2D(app);
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                Simple3DTabButtonDown(app);
            end
            if strcmp(app.TabGroup.SelectedTab.Title, 'ROI')
                ROITabButtonDown(app);
            end
        end

        % Button pushed function: Button_minus_3
        function Button_minus_3Pushed(app, event)
            frame_range = app.FrameSlider.Limits;
            app.FrameSlider.Value = mod(app.FrameSlider.Value-2, frame_range(2))+1;
            update_2D(app);
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                Simple3DTabButtonDown(app);
            end
            if strcmp(app.TabGroup.SelectedTab.Title, 'ROI')
                ROITabButtonDown(app);
            end
        end

        % Value changed function: Button_play_frame
        function Button_play_frameValueChanged(app, event)
            if app.Button_play_frame.Value
                period = 0.05;
                app.Timer_frame = timer('ExecutionMode','fixedRate','Period',period,'TimerFcn',@(~,~)Button_plus_3Pushed(app));
                start(app.Timer_frame)
            else
                if isa(app.Timer_frame,'timer')
                    if isvalid(app.Timer_frame)
                        stop(app.Timer_frame)
                        delete(app.Timer_frame)
                    end
                end
            end
        end

        % Value changed function: MIPProjButton
        function MIPProjButtonValueChanged(app, event)
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                Simple3DTabButtonDown(app);
            end
            if strcmp(app.TabGroup.SelectedTab.Title, '3D')
                DTab_2ButtonDown(app);
            end
        end

        % Value changed function: MIP3DgammaEditField
        function MIP3DgammaEditFieldValueChanged(app, event)
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                Simple3DTabButtonDown(app);
            end
            if strcmp(app.TabGroup.SelectedTab.Title, '3D')
                DTab_2ButtonDown(app);
            end
        end

        % Value changed function: MIPcolorrangeSlider
        function MIPcolorrangeSliderValueChanged(app, event)
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                Simple3DTabButtonDown(app);
            end
            if strcmp(app.TabGroup.SelectedTab.Title, '3D')
                DTab_2ButtonDown(app);
            end
        end

        % Value changed function: MIP3DclampSlider
        function MIP3DclampSliderValueChanged(app, event)
            if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                Simple3DTabButtonDown(app);
            end
            if strcmp(app.TabGroup.SelectedTab.Title, '3D')
                DTab_2ButtonDown(app);
            end
        end

        % Button pushed function: Button_5
        function Button_5Pushed(app, event)
            curr = app.MIP3DclampSlider.Limits(2);
            app.MIP3DclampSlider.Value = [0,curr/2];
            app.MIP3DclampSlider.Limits = [0,curr/2];
            drawnow
            MIP3DclampSliderValueChanged(app);
        end

        % Button pushed function: RButton
        function RButtonPushed(app, event)
            app.MIP3DclampSlider.Limits = [0,1];
            app.MIP3DclampSlider.Value = [0,1];
            drawnow
            MIP3DclampSliderValueChanged(app);
        end

        % Button pushed function: ViewingRawButton
        function ViewingRawButtonPushed(app, event)
            [~,filename,~] = fileparts(app.ListBox.Value);
            % 使用正则表达式匹配 _数字hz 的模式
            pattern = '_(\d+)hz';
            tokens = regexp(filename, pattern, 'tokens', 'ignorecase');
            if ~isempty(tokens)
                framerate = str2double(tokens{1}{1});
            else
                framerate = 20;
            end

            windowTitle = sprintf('Viewing: %s', app.ListBox.Value);

            % 调用独立的图像查看器函数
            try
                ImageStackViewer(app.im_data, 'Title', windowTitle, 'Framerate',framerate);
            catch ME
                errordlg(sprintf('Error opening viewer: %s', ME.message), 'Error');
            end
        end

        % Button pushed function: SetprocessparamButton
        function SetprocessparamButtonPushed(app, event)
            % Call dialog box with input values
            app.CalApp4 = cal_dark(app);
        end

        % Button pushed function: SaveSlicedtMovieButton
        function SaveSlicedtMovieButtonPushed(app, event)
            [~,filename,~] = fileparts(app.ListBox.Value);
            frame_range = app.FrameSlider.Limits;
            depth = app.FullrangeumEditField.Value;
            save_animation(app,filename,frame_range,depth,app.nDepthEditField.Value);
        end

        % Button pushed function: DDarkButton
        function DDarkButtonPushed(app, event)
            cal_dark_3d(app);
        end

        % Button pushed function: saveDepthButton
        function saveDepthButtonPushed(app, event)
            % get filename
            filename = app.ListBox.Value;
            [~,filename,~] = fileparts(filename);
            % write config file
            fid = fopen([fullfile(app.data_path,[filename,'.txt'])],'w');
            % 需要写入下面的变量：
            depth_range = [app.DepthrangeEditField.Value,app.EditField.Value];
            PSF_n = app.PSFnEditField.Value;
            nIters = app.nItersSpinner.Value;
            % 写入配置参数
            fprintf(fid, 'depth_range = [%.4f, %.4f]\n', depth_range(1), depth_range(2));
            fprintf(fid, 'PSF_n = %.4f\n', PSF_n);
            fprintf(fid, 'nIters = %d\n', nIters);
            fclose(fid);
        end

        % Button pushed function: TrainButton
        function TrainButtonPushed(app, event)
            addpath('DeepCAD_RT_pytorch\')
            try
                terminate(pyenv)
            end
            pyenv("Version",'C:\Research\deep-learning\.venv\Scripts\python.exe')
            % app.TrainButton.Enable = false;drawnow
            try
                deepcad_path = 'D:\NIR_SLIM2\DeepCAD_RT_pytorch';
                [~,model_name,~] = fileparts(app.ListBox.Value);
                %% adjust
                train_data = rescale(double(app.im_data));
                for i=1:size(app.im_data,3)
                    % im = bad_pixel_correction(app,train_data(:,:,i));
                    im = train_data(:,:,i);

                    value = app.DclipSlider.Value;
                    gamma_val = app.GammaSlider.Value;
                    train_data(:,:,i) = imadjust(im, value, [0,1], gamma_val);
                end

                app.im_data = deepcad_train(train_data,deepcad_path, ...
                                'model_name', model_name,...
                                'n_epochs', 5, ...
                                'train_datasets_size', min(size(app.im_data,3),4000));
                init_2D(app);
                update_2D(app);
    
                if strcmp(app.TabGroup.SelectedTab.Title, 'Simple 3D')
                    Simple3DTabButtonDown(app);
                end
                if strcmp(app.TabGroup.SelectedTab.Title, 'ROI')
                    ROITabButtonDown(app);
                end
            catch
                app.TrainButton.Enable = true;
            end
            app.TrainButton.Enable = true;
        end

        % Button pushed function: LoadButton
        function LoadButtonPushed(app, event)
            warning('off')
            % 让用户选择一个文件夹
            app.n2n_path = uigetdir('D:\NIR_SLIM2\DeepCAD_RT_pytorch');
            if app.n2n_path == 0
                % 如果用户取消了选择，就退出函数
                return;
            end
            figure(app.UIFigure);drawnow
        end

        % Button pushed function: ApplyButton
        function ApplyButtonPushed(app, event)
            deepcad_path = 'D:\NIR_SLIM2\DeepCAD_RT_pytorch';
            addpath('DeepCAD_RT_pytorch\')
            pyenv("Version",'C:\Research\deep-learning\.venv\Scripts\python.exe')
            [~, model_name, ~] = fileparts(app.n2n_path);
            % app.ApplyButton.Enable = false;drawnow
            try 
                %% adjust
                train_data = rescale(app.im_data);
                for i=1:size(app.im_data,3)
                    % im = bad_pixel_correction(app,train_data(:,:,i));
                    im = train_data(:,:,i);
                    value = app.DclipSlider.Value;
                    gamma_val = app.GammaSlider.Value;
                    train_data(:,:,i) = imadjust(im, value, [0,1], gamma_val);
                end
                app.im_data = deepcad_denoise(train_data, model_name, deepcad_path);
            catch ME
                app.ApplyButton.Enable = true;
                errordlg(ME.message, 'Denoising Error', 'modal');
            end
            app.ApplyButton.Enable = true;
        end

        % Button pushed function: TmpFolderButton
        function TmpFolderButtonPushed(app, event)
            app.data_path = 'E:\C_RED2_recordings\';
            init_data_folder(app);
            ListBoxValueChanged(app);
        end

        % Button pushed function: DButton
        function DButtonPushed(app, event)
            [file,location] = uigetfile(fullfile(app.data_path,'*.mat'));
            [~,basename,~] = fileparts(file);
            h = msgbox('Loading data...');
            S = load(fullfile(location,file));
            if isvalid(h), close(h); end
            Image4DViewer(S.reconVol_all,...
                'AmpAll',S.amp_all,...
                'VolumeRate',S.frame_rate,...
                'ZResolution',S.depth_size,'XYResolution',S.pixel_size,...
                'DefaultSavePath',app.data_path, ...
                'FilePrefix',basename)
        end

        % Button pushed function: SavewarpedPSFButton
        function SavewarpedPSFButtonPushed(app, event)
            init_interp(app);
            init_psf(app);
            % dim: [H,W,D,Views]

            % 确定默认路径
            if ~isequal(app.cal_path, 0)
                default_path = app.cal_path;
            else
                default_path = pwd;  % 当前路径
            end
            
            % 从app.cal_file2提取文件名并添加warped后缀
            [~, base_name, ~] = fileparts(app.cal_file2);
            default_filename = [base_name, '_warped.mat'];
            
            % 弹出保存文件对话框
            [filename, pathname] = uiputfile('*.mat', '保存PSF数据', ...
                fullfile(default_path, default_filename));
            
            % 检查用户是否取消了保存操作
            if ~isequal(filename, 0)
                % 将app.PSFs_warped赋值给变量psf
                psf = app.PSFs_warped;
                % [H,W,D,V] -> [H,W,V,D]
                psf = permute(psf, [1, 2, 4, 3]);  
                % [H,W,V,D] -> [H,W,1,V,D]
                psf = reshape(psf, [size(psf,1), size(psf,2), 1, size(psf,3), size(psf,4)]);  
                
                % 保存到mat文件
                save(fullfile(pathname, filename), 'psf', "-v7.3");

                % 生成对应的txt文件名（相同文件名，不同扩展名）
                [~, name, ~] = fileparts(filename);
                txt_filename = fullfile(pathname, [name, '.txt']);

                % 将像素大小写入txt文件
                fid = fopen(txt_filename, 'w');
                fprintf(fid, 'pixel_size_x: %.10f\n', app.pixel_size_x);
                fprintf(fid, 'pixel_size_y: %.10f\n', app.pixel_size_y);
                fclose(fid);
            end

            figure(app.UIFigure); drawnow
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.UIFigure.Position = [100 100 1085 854];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {40, 40, 20, 15, 20, 24, 15, 36, 16, '1x', 15, 9, 27, 10, 25, '1.04x', '1x', 25, 24, 10, 21, 20, '1x', '1x', 23, 20, '1.26x', 20, 35, 38, 20, 43, 20, 40, 9, 29, 11, 14, 14, 29, 22, 22};
            app.GridLayout.RowHeight = {23, 21, 12, 12, 12, 13, 25, 30, 30, 19, 20, 20, 29, 25, 22, 0, 22, '1x', 22, 25, 22, 28, 21, 21, 15, 15, 15, 15, 15, 15, 0, 15, 15, 15, 15, 15, 15, 15, 15, 22, 31, '2x', 15, 12, 19};
            app.GridLayout.ColumnSpacing = 1.70999298095703;
            app.GridLayout.RowSpacing = 1.57499525282118;
            app.GridLayout.Padding = [1.70999298095703 1.57499525282118 1.70999298095703 1.57499525282118];
            app.GridLayout.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];

            % Create StartgeometricregistrationButton
            app.StartgeometricregistrationButton = uibutton(app.GridLayout, 'push');
            app.StartgeometricregistrationButton.ButtonPushedFcn = createCallbackFcn(app, @StartgeometricregistrationButtonPushed, true);
            app.StartgeometricregistrationButton.WordWrap = 'on';
            app.StartgeometricregistrationButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.StartgeometricregistrationButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StartgeometricregistrationButton.Layout.Row = [1 2];
            app.StartgeometricregistrationButton.Layout.Column = [1 4];
            app.StartgeometricregistrationButton.Text = 'Start geometric registration';

            % Create StartPSFestimationButton
            app.StartPSFestimationButton = uibutton(app.GridLayout, 'push');
            app.StartPSFestimationButton.ButtonPushedFcn = createCallbackFcn(app, @StartPSFestimationButtonPushed, true);
            app.StartPSFestimationButton.WordWrap = 'on';
            app.StartPSFestimationButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.StartPSFestimationButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StartPSFestimationButton.Layout.Row = [5 7];
            app.StartPSFestimationButton.Layout.Column = [1 4];
            app.StartPSFestimationButton.Text = 'Start PSF estimation';

            % Create LoadPSFsButton
            app.LoadPSFsButton = uibutton(app.GridLayout, 'push');
            app.LoadPSFsButton.ButtonPushedFcn = createCallbackFcn(app, @LoadPSFsButtonPushed, true);
            app.LoadPSFsButton.VerticalAlignment = 'top';
            app.LoadPSFsButton.WordWrap = 'on';
            app.LoadPSFsButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.LoadPSFsButton.FontSize = 10;
            app.LoadPSFsButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.LoadPSFsButton.Layout.Row = 8;
            app.LoadPSFsButton.Layout.Column = [1 4];
            app.LoadPSFsButton.Text = 'Load PSFs';

            % Create OpenDataFolderButton
            app.OpenDataFolderButton = uibutton(app.GridLayout, 'push');
            app.OpenDataFolderButton.ButtonPushedFcn = createCallbackFcn(app, @OpenDataFolderButtonPushed, true);
            app.OpenDataFolderButton.WordWrap = 'on';
            app.OpenDataFolderButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.OpenDataFolderButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.OpenDataFolderButton.Layout.Row = 13;
            app.OpenDataFolderButton.Layout.Column = [1 4];
            app.OpenDataFolderButton.Text = 'Open Data Folder';

            % Create ListBox
            app.ListBox = uilistbox(app.GridLayout);
            app.ListBox.Items = {'Empty'};
            app.ListBox.ValueChangedFcn = createCallbackFcn(app, @ListBoxValueChanged, true);
            app.ListBox.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ListBox.Layout.Row = [15 37];
            app.ListBox.Layout.Column = [1 4];
            app.ListBox.DoubleClickedFcn = createCallbackFcn(app, @ListBoxDoubleClicked, true);
            app.ListBox.Value = 'Empty';

            % Create ReconButton
            app.ReconButton = uibutton(app.GridLayout, 'push');
            app.ReconButton.ButtonPushedFcn = createCallbackFcn(app, @ReconButtonPushed, true);
            app.ReconButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ReconButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ReconButton.Layout.Row = [28 31];
            app.ReconButton.Layout.Column = [37 42];
            app.ReconButton.Text = 'Recon.';

            % Create SaveMovieButton
            app.SaveMovieButton = uibutton(app.GridLayout, 'push');
            app.SaveMovieButton.ButtonPushedFcn = createCallbackFcn(app, @SaveMovieButtonPushed, true);
            app.SaveMovieButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.SaveMovieButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.SaveMovieButton.Layout.Row = [32 33];
            app.SaveMovieButton.Layout.Column = [37 42];
            app.SaveMovieButton.Text = 'Save Movie';

            % Create LoadGeoFileButton
            app.LoadGeoFileButton = uibutton(app.GridLayout, 'push');
            app.LoadGeoFileButton.ButtonPushedFcn = createCallbackFcn(app, @LoadGeoFileButtonPushed, true);
            app.LoadGeoFileButton.VerticalAlignment = 'top';
            app.LoadGeoFileButton.WordWrap = 'on';
            app.LoadGeoFileButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.LoadGeoFileButton.FontSize = 10;
            app.LoadGeoFileButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.LoadGeoFileButton.Layout.Row = [3 4];
            app.LoadGeoFileButton.Layout.Column = [1 4];
            app.LoadGeoFileButton.Text = 'Load Geo. File';

            % Create TabGroup
            app.TabGroup = uitabgroup(app.GridLayout);
            app.TabGroup.Layout.Row = [12 45];
            app.TabGroup.Layout.Column = [5 35];

            % Create RawTab
            app.RawTab = uitab(app.TabGroup);
            app.RawTab.Title = 'Raw';
            app.RawTab.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.RawTab.ForegroundColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.RawTab.ButtonDownFcn = createCallbackFcn(app, @RawTabButtonDown, true);

            % Create GridLayout5
            app.GridLayout5 = uigridlayout(app.RawTab);
            app.GridLayout5.ColumnWidth = {42, 24, '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', 20, '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1.12x'};
            app.GridLayout5.RowHeight = {22, '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', 15, 35, '1x', 22, 23};
            app.GridLayout5.RowSpacing = 3.4;
            app.GridLayout5.Padding = [10 3.4 10 3.4];
            app.GridLayout5.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];

            % Create UIAxes
            app.UIAxes = uiaxes(app.GridLayout5);
            zlabel(app.UIAxes, 'Z')
            app.UIAxes.XTick = [];
            app.UIAxes.YTick = [];
            app.UIAxes.Layout.Row = [2 24];
            app.UIAxes.Layout.Column = [1 21];

            % Create AutoButton
            app.AutoButton = uibutton(app.GridLayout5, 'push');
            app.AutoButton.ButtonPushedFcn = createCallbackFcn(app, @AutoButtonPushed, true);
            app.AutoButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.AutoButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.AutoButton.Layout.Row = 26;
            app.AutoButton.Layout.Column = [1 2];
            app.AutoButton.Text = 'Auto';

            % Create ResetButton
            app.ResetButton = uibutton(app.GridLayout5, 'push');
            app.ResetButton.ButtonPushedFcn = createCallbackFcn(app, @ResetButtonPushed, true);
            app.ResetButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ResetButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ResetButton.Layout.Row = 26;
            app.ResetButton.Layout.Column = [13 14];
            app.ResetButton.Text = 'Reset';

            % Create RealtimeFPSLabel
            app.RealtimeFPSLabel = uilabel(app.GridLayout5);
            app.RealtimeFPSLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.RealtimeFPSLabel.Layout.Row = 1;
            app.RealtimeFPSLabel.Layout.Column = [1 3];
            app.RealtimeFPSLabel.Text = 'Real-time FPS: ';

            % Create Button_plus
            app.Button_plus = uibutton(app.GridLayout5, 'push');
            app.Button_plus.ButtonPushedFcn = createCallbackFcn(app, @Button_plusPushed, true);
            app.Button_plus.VerticalAlignment = 'top';
            app.Button_plus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_plus.FontSize = 14;
            app.Button_plus.FontWeight = 'bold';
            app.Button_plus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_plus.Layout.Row = 25;
            app.Button_plus.Layout.Column = 11;
            app.Button_plus.Text = '+';

            % Create Button_minus
            app.Button_minus = uibutton(app.GridLayout5, 'push');
            app.Button_minus.ButtonPushedFcn = createCallbackFcn(app, @Button_minusPushed, true);
            app.Button_minus.VerticalAlignment = 'top';
            app.Button_minus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_minus.FontSize = 14;
            app.Button_minus.FontWeight = 'bold';
            app.Button_minus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_minus.Layout.Row = 26;
            app.Button_minus.Layout.Column = 11;
            app.Button_minus.Text = '-';

            % Create DclipSliderLabel
            app.DclipSliderLabel = uilabel(app.GridLayout5);
            app.DclipSliderLabel.HorizontalAlignment = 'right';
            app.DclipSliderLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DclipSliderLabel.Layout.Row = 25;
            app.DclipSliderLabel.Layout.Column = [1 2];
            app.DclipSliderLabel.Text = '2D clip';

            % Create DclipSlider
            app.DclipSlider = uislider(app.GridLayout5, 'range');
            app.DclipSlider.Limits = [0 1];
            app.DclipSlider.ValueChangedFcn = createCallbackFcn(app, @DclipSliderValueChanged, true);
            app.DclipSlider.FontSize = 10;
            app.DclipSlider.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DclipSlider.Layout.Row = [25 26];
            app.DclipSlider.Layout.Column = [3 10];
            app.DclipSlider.Value = [0 1];

            % Create GammaSliderLabel
            app.GammaSliderLabel = uilabel(app.GridLayout5);
            app.GammaSliderLabel.HorizontalAlignment = 'right';
            app.GammaSliderLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.GammaSliderLabel.Layout.Row = 25;
            app.GammaSliderLabel.Layout.Column = [13 14];
            app.GammaSliderLabel.Text = 'Gamma';

            % Create GammaSlider
            app.GammaSlider = uislider(app.GridLayout5);
            app.GammaSlider.Limits = [0.5 5];
            app.GammaSlider.ValueChangedFcn = createCallbackFcn(app, @GammaSliderValueChanged, true);
            app.GammaSlider.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.GammaSlider.Layout.Row = [25 26];
            app.GammaSlider.Layout.Column = [15 21];
            app.GammaSlider.Value = 1;

            % Create AmpLabel
            app.AmpLabel = uilabel(app.GridLayout5);
            app.AmpLabel.Layout.Row = 1;
            app.AmpLabel.Layout.Column = [4 7];
            app.AmpLabel.Text = 'Amp: ';

            % Create ROITab
            app.ROITab = uitab(app.TabGroup);
            app.ROITab.Title = 'ROI';
            app.ROITab.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ROITab.ForegroundColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ROITab.ButtonDownFcn = createCallbackFcn(app, @ROITabButtonDown, true);

            % Create RealtimeFPSLabel_3
            app.RealtimeFPSLabel_3 = uilabel(app.ROITab);
            app.RealtimeFPSLabel_3.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.RealtimeFPSLabel_3.Position = [11 561 186 22];
            app.RealtimeFPSLabel_3.Text = 'Real-time FPS: ';

            % Create ROIEditFieldLabel
            app.ROIEditFieldLabel = uilabel(app.ROITab);
            app.ROIEditFieldLabel.HorizontalAlignment = 'right';
            app.ROIEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ROIEditFieldLabel.Position = [54 33 37 17];
            app.ROIEditFieldLabel.Text = 'ROI%';

            % Create ROIEditField
            app.ROIEditField = uieditfield(app.ROITab, 'numeric');
            app.ROIEditField.Limits = [0 100];
            app.ROIEditField.ValueChangedFcn = createCallbackFcn(app, @ROIEditFieldValueChanged, true);
            app.ROIEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ROIEditField.Position = [101 33 53 17];
            app.ROIEditField.Value = 75;

            % Create ROIdepthcenterSliderLabel
            app.ROIdepthcenterSliderLabel = uilabel(app.ROITab);
            app.ROIdepthcenterSliderLabel.HorizontalAlignment = 'right';
            app.ROIdepthcenterSliderLabel.WordWrap = 'on';
            app.ROIdepthcenterSliderLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ROIdepthcenterSliderLabel.Position = [161 2 73 44];
            app.ROIdepthcenterSliderLabel.Text = 'ROI depth center';

            % Create ROIdepthcenterSlider
            app.ROIdepthcenterSlider = uislider(app.ROITab);
            app.ROIdepthcenterSlider.Limits = [0 1];
            app.ROIdepthcenterSlider.ValueChangedFcn = createCallbackFcn(app, @ROIdepthcenterSliderValueChanged, true);
            app.ROIdepthcenterSlider.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ROIdepthcenterSlider.Position = [250 49 175 3];
            app.ROIdepthcenterSlider.Value = 0.5;

            % Create ROIdepthfullbandwidthSliderLabel
            app.ROIdepthfullbandwidthSliderLabel = uilabel(app.ROITab);
            app.ROIdepthfullbandwidthSliderLabel.HorizontalAlignment = 'right';
            app.ROIdepthfullbandwidthSliderLabel.WordWrap = 'on';
            app.ROIdepthfullbandwidthSliderLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ROIdepthfullbandwidthSliderLabel.Position = [441 2 85 44];
            app.ROIdepthfullbandwidthSliderLabel.Text = 'ROI depth full bandwidth';

            % Create ROIdepthfullbandwidthSlider
            app.ROIdepthfullbandwidthSlider = uislider(app.ROITab);
            app.ROIdepthfullbandwidthSlider.Limits = [0 1];
            app.ROIdepthfullbandwidthSlider.ValueChangedFcn = createCallbackFcn(app, @ROIdepthfullbandwidthSliderValueChanged, true);
            app.ROIdepthfullbandwidthSlider.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ROIdepthfullbandwidthSlider.Position = [542 49 165 3];
            app.ROIdepthfullbandwidthSlider.Value = 0.1;

            % Create avgnumberEditFieldLabel
            app.avgnumberEditFieldLabel = uilabel(app.ROITab);
            app.avgnumberEditFieldLabel.HorizontalAlignment = 'right';
            app.avgnumberEditFieldLabel.WordWrap = 'on';
            app.avgnumberEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.avgnumberEditFieldLabel.Position = [16 12 77 16];
            app.avgnumberEditFieldLabel.Text = 'avg. number';

            % Create avgnumberEditField
            app.avgnumberEditField = uieditfield(app.ROITab, 'numeric');
            app.avgnumberEditField.Limits = [0 500];
            app.avgnumberEditField.RoundFractionalValues = 'on';
            app.avgnumberEditField.ValueChangedFcn = createCallbackFcn(app, @avgnumberEditFieldValueChanged, true);
            app.avgnumberEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.avgnumberEditField.Position = [103 11 53 17];
            app.avgnumberEditField.Value = 3;

            % Create Simple3DTab
            app.Simple3DTab = uitab(app.TabGroup);
            app.Simple3DTab.Title = 'Simple 3D';
            app.Simple3DTab.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Simple3DTab.ForegroundColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Simple3DTab.ButtonDownFcn = createCallbackFcn(app, @Simple3DTabButtonDown, true);

            % Create GridLayout3
            app.GridLayout3 = uigridlayout(app.Simple3DTab);
            app.GridLayout3.ColumnWidth = {17, 31, 17, 93, '1x', 35, 35, 45, 25, 17, 35, 35, '2.76x', 20, 18, 63};
            app.GridLayout3.RowHeight = {22, '1x', 22, 22, '1.12x', 22, 22, '1x', 22, 22, 22, 22, '1x', '1x', '1x', '1x', '1x', 35, 22};
            app.GridLayout3.ColumnSpacing = 3.41176470588235;
            app.GridLayout3.RowSpacing = 4.5625;
            app.GridLayout3.Padding = [3.41176470588235 4.5625 3.41176470588235 4.5625];
            app.GridLayout3.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];

            % Create UIAxes3
            app.UIAxes3 = uiaxes(app.GridLayout3);
            zlabel(app.UIAxes3, 'Z')
            app.UIAxes3.XTick = [];
            app.UIAxes3.YTick = [];
            app.UIAxes3.Layout.Row = [2 17];
            app.UIAxes3.Layout.Column = [1 15];

            % Create AutoButton_2
            app.AutoButton_2 = uibutton(app.GridLayout3, 'push');
            app.AutoButton_2.ButtonPushedFcn = createCallbackFcn(app, @AutoButton_2Pushed, true);
            app.AutoButton_2.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.AutoButton_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.AutoButton_2.Layout.Row = 19;
            app.AutoButton_2.Layout.Column = 13;
            app.AutoButton_2.Text = 'Auto';

            % Create RealtimeFPSLabel_2
            app.RealtimeFPSLabel_2 = uilabel(app.GridLayout3);
            app.RealtimeFPSLabel_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.RealtimeFPSLabel_2.Layout.Row = 1;
            app.RealtimeFPSLabel_2.Layout.Column = [1 4];
            app.RealtimeFPSLabel_2.Text = 'Real-time FPS: ';

            % Create ScanButton
            app.ScanButton = uibutton(app.GridLayout3, 'state');
            app.ScanButton.ValueChangedFcn = createCallbackFcn(app, @ScanButtonValueChanged, true);
            app.ScanButton.Text = 'Scan';
            app.ScanButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ScanButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ScanButton.Layout.Row = 19;
            app.ScanButton.Layout.Column = [6 7];

            % Create CheckBox3
            app.CheckBox3 = uicheckbox(app.GridLayout3);
            app.CheckBox3.ValueChangedFcn = createCallbackFcn(app, @CheckBox3ValueChanged, true);
            app.CheckBox3.Text = 'Center View';
            app.CheckBox3.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.CheckBox3.Layout.Row = 1;
            app.CheckBox3.Layout.Column = [8 10];
            app.CheckBox3.Value = true;

            % Create CheckBox1
            app.CheckBox1 = uicheckbox(app.GridLayout3);
            app.CheckBox1.ValueChangedFcn = createCallbackFcn(app, @CheckBox1ValueChanged, true);
            app.CheckBox1.Text = '#1';
            app.CheckBox1.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.CheckBox1.Layout.Row = 1;
            app.CheckBox1.Layout.Column = 6;
            app.CheckBox1.Value = true;

            % Create CheckBox2
            app.CheckBox2 = uicheckbox(app.GridLayout3);
            app.CheckBox2.ValueChangedFcn = createCallbackFcn(app, @CheckBox2ValueChanged, true);
            app.CheckBox2.Text = '#2';
            app.CheckBox2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.CheckBox2.Layout.Row = 1;
            app.CheckBox2.Layout.Column = 7;
            app.CheckBox2.Value = true;

            % Create CheckBox4
            app.CheckBox4 = uicheckbox(app.GridLayout3);
            app.CheckBox4.ValueChangedFcn = createCallbackFcn(app, @CheckBox4ValueChanged, true);
            app.CheckBox4.Text = '#4';
            app.CheckBox4.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.CheckBox4.Layout.Row = 1;
            app.CheckBox4.Layout.Column = 11;
            app.CheckBox4.Value = true;

            % Create CheckBox5
            app.CheckBox5 = uicheckbox(app.GridLayout3);
            app.CheckBox5.ValueChangedFcn = createCallbackFcn(app, @CheckBox5ValueChanged, true);
            app.CheckBox5.Text = '#5';
            app.CheckBox5.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.CheckBox5.Layout.Row = 1;
            app.CheckBox5.Layout.Column = 12;
            app.CheckBox5.Value = true;

            % Create ButtonUpRange_2
            app.ButtonUpRange_2 = uibutton(app.GridLayout3, 'push');
            app.ButtonUpRange_2.ButtonPushedFcn = createCallbackFcn(app, @ButtonUpRange_2Pushed, true);
            app.ButtonUpRange_2.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ButtonUpRange_2.FontSize = 8;
            app.ButtonUpRange_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ButtonUpRange_2.Layout.Row = 19;
            app.ButtonUpRange_2.Layout.Column = 5;
            app.ButtonUpRange_2.Text = '🔺';

            % Create ButtonDownRange_2
            app.ButtonDownRange_2 = uibutton(app.GridLayout3, 'push');
            app.ButtonDownRange_2.ButtonPushedFcn = createCallbackFcn(app, @ButtonDownRange_2Pushed, true);
            app.ButtonDownRange_2.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ButtonDownRange_2.FontSize = 8;
            app.ButtonDownRange_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ButtonDownRange_2.Layout.Row = 19;
            app.ButtonDownRange_2.Layout.Column = 8;
            app.ButtonDownRange_2.Text = '🔻';

            % Create Button_plus_2
            app.Button_plus_2 = uibutton(app.GridLayout3, 'push');
            app.Button_plus_2.ButtonPushedFcn = createCallbackFcn(app, @Button_plus_2Pushed, true);
            app.Button_plus_2.VerticalAlignment = 'top';
            app.Button_plus_2.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_plus_2.FontSize = 14;
            app.Button_plus_2.FontWeight = 'bold';
            app.Button_plus_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_plus_2.Layout.Row = 19;
            app.Button_plus_2.Layout.Column = 16;
            app.Button_plus_2.Text = '+';

            % Create Button_minus_2
            app.Button_minus_2 = uibutton(app.GridLayout3, 'push');
            app.Button_minus_2.ButtonPushedFcn = createCallbackFcn(app, @Button_minus_2Pushed, true);
            app.Button_minus_2.VerticalAlignment = 'top';
            app.Button_minus_2.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_minus_2.FontSize = 14;
            app.Button_minus_2.FontWeight = 'bold';
            app.Button_minus_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_minus_2.Layout.Row = 19;
            app.Button_minus_2.Layout.Column = 12;
            app.Button_minus_2.Text = '-';

            % Create DSliceSlider_2Label
            app.DSliceSlider_2Label = uilabel(app.GridLayout3);
            app.DSliceSlider_2Label.HorizontalAlignment = 'right';
            app.DSliceSlider_2Label.WordWrap = 'on';
            app.DSliceSlider_2Label.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DSliceSlider_2Label.Layout.Row = 18;
            app.DSliceSlider_2Label.Layout.Column = [2 3];
            app.DSliceSlider_2Label.Text = '3D Slice';

            % Create DSliceSlider_2
            app.DSliceSlider_2 = uislider(app.GridLayout3);
            app.DSliceSlider_2.Limits = [0 1];
            app.DSliceSlider_2.ValueChangedFcn = createCallbackFcn(app, @DSliceSlider_2ValueChanged, true);
            app.DSliceSlider_2.ValueChangingFcn = createCallbackFcn(app, @DSliceSlider_2ValueChanging, true);
            app.DSliceSlider_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DSliceSlider_2.Layout.Row = 18;
            app.DSliceSlider_2.Layout.Column = [4 10];
            app.DSliceSlider_2.Value = 0.5;

            % Create DclipSlider_2Label
            app.DclipSlider_2Label = uilabel(app.GridLayout3);
            app.DclipSlider_2Label.HorizontalAlignment = 'right';
            app.DclipSlider_2Label.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DclipSlider_2Label.Layout.Row = 18;
            app.DclipSlider_2Label.Layout.Column = [10 11];
            app.DclipSlider_2Label.Text = '2D clip';

            % Create DclipSlider_2
            app.DclipSlider_2 = uislider(app.GridLayout3, 'range');
            app.DclipSlider_2.Limits = [0 1];
            app.DclipSlider_2.ValueChangedFcn = createCallbackFcn(app, @DclipSlider_2ValueChanged, true);
            app.DclipSlider_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DclipSlider_2.Layout.Row = 18;
            app.DclipSlider_2.Layout.Column = [12 16];
            app.DclipSlider_2.Value = [0 1];

            % Create MaskCenterXSpinnerLabel
            app.MaskCenterXSpinnerLabel = uilabel(app.GridLayout3);
            app.MaskCenterXSpinnerLabel.HorizontalAlignment = 'right';
            app.MaskCenterXSpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskCenterXSpinnerLabel.Layout.Row = 3;
            app.MaskCenterXSpinnerLabel.Layout.Column = [15 16];
            app.MaskCenterXSpinnerLabel.Text = 'Mask CenterX';

            % Create MaskCenterXSpinner
            app.MaskCenterXSpinner = uispinner(app.GridLayout3);
            app.MaskCenterXSpinner.ValueChangedFcn = createCallbackFcn(app, @MaskCenterXSpinnerValueChanged, true);
            app.MaskCenterXSpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskCenterXSpinner.Layout.Row = 4;
            app.MaskCenterXSpinner.Layout.Column = 16;
            app.MaskCenterXSpinner.Value = 128;

            % Create MaskCenterYSpinnerLabel
            app.MaskCenterYSpinnerLabel = uilabel(app.GridLayout3);
            app.MaskCenterYSpinnerLabel.HorizontalAlignment = 'right';
            app.MaskCenterYSpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskCenterYSpinnerLabel.Layout.Row = 6;
            app.MaskCenterYSpinnerLabel.Layout.Column = [15 16];
            app.MaskCenterYSpinnerLabel.Text = 'Mask CenterY';

            % Create MaskCenterYSpinner
            app.MaskCenterYSpinner = uispinner(app.GridLayout3);
            app.MaskCenterYSpinner.ValueChangedFcn = createCallbackFcn(app, @MaskCenterYSpinnerValueChanged, true);
            app.MaskCenterYSpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskCenterYSpinner.Layout.Row = 7;
            app.MaskCenterYSpinner.Layout.Column = 16;
            app.MaskCenterYSpinner.Value = 128;

            % Create MaskROISpinnerLabel
            app.MaskROISpinnerLabel = uilabel(app.GridLayout3);
            app.MaskROISpinnerLabel.HorizontalAlignment = 'right';
            app.MaskROISpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskROISpinnerLabel.Layout.Row = 9;
            app.MaskROISpinnerLabel.Layout.Column = [15 16];
            app.MaskROISpinnerLabel.Text = 'Mask ROI%';

            % Create MaskROISpinner
            app.MaskROISpinner = uispinner(app.GridLayout3);
            app.MaskROISpinner.ValueChangedFcn = createCallbackFcn(app, @MaskROISpinnerValueChanged, true);
            app.MaskROISpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskROISpinner.Layout.Row = 10;
            app.MaskROISpinner.Layout.Column = 16;
            app.MaskROISpinner.Value = 100;

            % Create BlurradiusSpinnerLabel
            app.BlurradiusSpinnerLabel = uilabel(app.GridLayout3);
            app.BlurradiusSpinnerLabel.HorizontalAlignment = 'right';
            app.BlurradiusSpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.BlurradiusSpinnerLabel.Layout.Row = 11;
            app.BlurradiusSpinnerLabel.Layout.Column = 16;
            app.BlurradiusSpinnerLabel.Text = 'Blur radius';

            % Create BlurradiusSpinner
            app.BlurradiusSpinner = uispinner(app.GridLayout3);
            app.BlurradiusSpinner.ValueChangedFcn = createCallbackFcn(app, @BlurradiusSpinnerValueChanged, true);
            app.BlurradiusSpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.BlurradiusSpinner.Layout.Row = 12;
            app.BlurradiusSpinner.Layout.Column = 16;

            % Create ShallowLabel_3
            app.ShallowLabel_3 = uilabel(app.GridLayout3);
            app.ShallowLabel_3.Layout.Row = 19;
            app.ShallowLabel_3.Layout.Column = 4;
            app.ShallowLabel_3.Text = 'Shallow';

            % Create DeepLabel_4
            app.DeepLabel_4 = uilabel(app.GridLayout3);
            app.DeepLabel_4.Layout.Row = 19;
            app.DeepLabel_4.Layout.Column = [9 10];
            app.DeepLabel_4.Text = 'Deep';

            % Create DTab_2
            app.DTab_2 = uitab(app.TabGroup);
            app.DTab_2.Title = '3D';
            app.DTab_2.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.DTab_2.ForegroundColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DTab_2.ButtonDownFcn = createCallbackFcn(app, @DTab_2ButtonDown, true);

            % Create GridLayout4
            app.GridLayout4 = uigridlayout(app.DTab_2);
            app.GridLayout4.ColumnWidth = {48, '1.32x', 31, '10.78x', 34, '1x', '1.32x', 81};
            app.GridLayout4.RowHeight = {'2x', 22, 22, '1.12x', 22, 22, '1x', 22, 22, 22, 22, '1x', '1x', '1x', '1x', '1x', 23, 22};
            app.GridLayout4.ColumnSpacing = 4.22222222222222;
            app.GridLayout4.RowSpacing = 6.53333333333333;
            app.GridLayout4.Padding = [4.22222222222222 6.53333333333333 4.22222222222222 6.53333333333333];
            app.GridLayout4.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];

            % Create UIAxes2
            app.UIAxes2 = uiaxes(app.GridLayout4);
            zlabel(app.UIAxes2, 'Z')
            app.UIAxes2.Layout.Row = [1 16];
            app.UIAxes2.Layout.Column = [1 7];
            app.UIAxes2.ButtonDownFcn = createCallbackFcn(app, @UIAxes2ButtonDown, true);

            % Create PlayButton
            app.PlayButton = uibutton(app.GridLayout4, 'state');
            app.PlayButton.ValueChangedFcn = createCallbackFcn(app, @PlayButtonValueChanged, true);
            app.PlayButton.Text = 'Play';
            app.PlayButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.PlayButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.PlayButton.Layout.Row = 17;
            app.PlayButton.Layout.Column = 2;

            % Create DSliceSliderLabel
            app.DSliceSliderLabel = uilabel(app.GridLayout4);
            app.DSliceSliderLabel.HorizontalAlignment = 'right';
            app.DSliceSliderLabel.WordWrap = 'on';
            app.DSliceSliderLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DSliceSliderLabel.Layout.Row = [17 18];
            app.DSliceSliderLabel.Layout.Column = 3;
            app.DSliceSliderLabel.Text = '3D Slice';

            % Create DSliceSlider
            app.DSliceSlider = uislider(app.GridLayout4);
            app.DSliceSlider.Limits = [0 1];
            app.DSliceSlider.ValueChangedFcn = createCallbackFcn(app, @DSliceSliderValueChanged, true);
            app.DSliceSlider.ValueChangingFcn = createCallbackFcn(app, @DSliceSliderValueChanging, true);
            app.DSliceSlider.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DSliceSlider.Layout.Row = [17 18];
            app.DSliceSlider.Layout.Column = 4;
            app.DSliceSlider.Value = 0.5;

            % Create ColorDropDownLabel
            app.ColorDropDownLabel = uilabel(app.GridLayout4);
            app.ColorDropDownLabel.HorizontalAlignment = 'right';
            app.ColorDropDownLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ColorDropDownLabel.Layout.Row = 17;
            app.ColorDropDownLabel.Layout.Column = 5;
            app.ColorDropDownLabel.Text = 'Color';

            % Create ColorDropDown
            app.ColorDropDown = uidropdown(app.GridLayout4);
            app.ColorDropDown.Items = {'gray', 'parula', 'bone', 'turbo', 'abyss', 'sky'};
            app.ColorDropDown.ValueChangedFcn = createCallbackFcn(app, @ColorDropDownValueChanged, true);
            app.ColorDropDown.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ColorDropDown.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ColorDropDown.Layout.Row = 17;
            app.ColorDropDown.Layout.Column = [6 7];
            app.ColorDropDown.Value = 'gray';

            % Create PlayfpsEditFieldLabel
            app.PlayfpsEditFieldLabel = uilabel(app.GridLayout4);
            app.PlayfpsEditFieldLabel.HorizontalAlignment = 'right';
            app.PlayfpsEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.PlayfpsEditFieldLabel.Layout.Row = 18;
            app.PlayfpsEditFieldLabel.Layout.Column = 1;
            app.PlayfpsEditFieldLabel.Text = 'Play fps';

            % Create PlayfpsEditField
            app.PlayfpsEditField = uieditfield(app.GridLayout4, 'numeric');
            app.PlayfpsEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.PlayfpsEditField.Layout.Row = 18;
            app.PlayfpsEditField.Layout.Column = 2;
            app.PlayfpsEditField.Value = 15;

            % Create MaskCenterXSpinner_2Label
            app.MaskCenterXSpinner_2Label = uilabel(app.GridLayout4);
            app.MaskCenterXSpinner_2Label.HorizontalAlignment = 'right';
            app.MaskCenterXSpinner_2Label.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskCenterXSpinner_2Label.Layout.Row = 2;
            app.MaskCenterXSpinner_2Label.Layout.Column = 8;
            app.MaskCenterXSpinner_2Label.Text = 'Mask CenterX';

            % Create MaskCenterXSpinner_2
            app.MaskCenterXSpinner_2 = uispinner(app.GridLayout4);
            app.MaskCenterXSpinner_2.ValueChangedFcn = createCallbackFcn(app, @MaskCenterXSpinner_2ValueChanged, true);
            app.MaskCenterXSpinner_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskCenterXSpinner_2.Layout.Row = 3;
            app.MaskCenterXSpinner_2.Layout.Column = 8;
            app.MaskCenterXSpinner_2.Value = 128;

            % Create MaskCenterYSpinner_2Label
            app.MaskCenterYSpinner_2Label = uilabel(app.GridLayout4);
            app.MaskCenterYSpinner_2Label.HorizontalAlignment = 'right';
            app.MaskCenterYSpinner_2Label.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskCenterYSpinner_2Label.Layout.Row = 4;
            app.MaskCenterYSpinner_2Label.Layout.Column = 8;
            app.MaskCenterYSpinner_2Label.Text = 'Mask CenterY';

            % Create MaskCenterYSpinner_2
            app.MaskCenterYSpinner_2 = uispinner(app.GridLayout4);
            app.MaskCenterYSpinner_2.ValueChangedFcn = createCallbackFcn(app, @MaskCenterYSpinner_2ValueChanged, true);
            app.MaskCenterYSpinner_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskCenterYSpinner_2.Layout.Row = 5;
            app.MaskCenterYSpinner_2.Layout.Column = 8;
            app.MaskCenterYSpinner_2.Value = 128;

            % Create MaskROISpinner_2Label
            app.MaskROISpinner_2Label = uilabel(app.GridLayout4);
            app.MaskROISpinner_2Label.HorizontalAlignment = 'right';
            app.MaskROISpinner_2Label.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskROISpinner_2Label.Layout.Row = 6;
            app.MaskROISpinner_2Label.Layout.Column = 8;
            app.MaskROISpinner_2Label.Text = 'Mask ROI%';

            % Create MaskROISpinner_2
            app.MaskROISpinner_2 = uispinner(app.GridLayout4);
            app.MaskROISpinner_2.ValueChangedFcn = createCallbackFcn(app, @MaskROISpinner_2ValueChanged, true);
            app.MaskROISpinner_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MaskROISpinner_2.Layout.Row = 7;
            app.MaskROISpinner_2.Layout.Column = 8;
            app.MaskROISpinner_2.Value = 85;

            % Create BlurradiusSpinner_2Label
            app.BlurradiusSpinner_2Label = uilabel(app.GridLayout4);
            app.BlurradiusSpinner_2Label.HorizontalAlignment = 'right';
            app.BlurradiusSpinner_2Label.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.BlurradiusSpinner_2Label.Layout.Row = 8;
            app.BlurradiusSpinner_2Label.Layout.Column = 8;
            app.BlurradiusSpinner_2Label.Text = 'Blur radius';

            % Create BlurradiusSpinner_2
            app.BlurradiusSpinner_2 = uispinner(app.GridLayout4);
            app.BlurradiusSpinner_2.ValueChangedFcn = createCallbackFcn(app, @BlurradiusSpinner_2ValueChanged, true);
            app.BlurradiusSpinner_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.BlurradiusSpinner_2.Layout.Row = 9;
            app.BlurradiusSpinner_2.Layout.Column = 8;

            % Create ScalebarXSpinnerLabel
            app.ScalebarXSpinnerLabel = uilabel(app.GridLayout4);
            app.ScalebarXSpinnerLabel.HorizontalAlignment = 'right';
            app.ScalebarXSpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ScalebarXSpinnerLabel.Layout.Row = 10;
            app.ScalebarXSpinnerLabel.Layout.Column = 8;
            app.ScalebarXSpinnerLabel.Text = 'Scale bar X';

            % Create ScalebarXSpinner
            app.ScalebarXSpinner = uispinner(app.GridLayout4);
            app.ScalebarXSpinner.Step = 0.01;
            app.ScalebarXSpinner.Limits = [-1 2];
            app.ScalebarXSpinner.ValueChangedFcn = createCallbackFcn(app, @ScalebarXSpinnerValueChanged2, true);
            app.ScalebarXSpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ScalebarXSpinner.Layout.Row = 11;
            app.ScalebarXSpinner.Layout.Column = 8;
            app.ScalebarXSpinner.Value = 0.7;

            % Create ScalebarYSpinnerLabel
            app.ScalebarYSpinnerLabel = uilabel(app.GridLayout4);
            app.ScalebarYSpinnerLabel.HorizontalAlignment = 'right';
            app.ScalebarYSpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ScalebarYSpinnerLabel.Layout.Row = 12;
            app.ScalebarYSpinnerLabel.Layout.Column = 8;
            app.ScalebarYSpinnerLabel.Text = 'Scale bar Y';

            % Create ScalebarYSpinner
            app.ScalebarYSpinner = uispinner(app.GridLayout4);
            app.ScalebarYSpinner.Step = 0.01;
            app.ScalebarYSpinner.ValueChangedFcn = createCallbackFcn(app, @ScalebarYSpinnerValueChanged, true);
            app.ScalebarYSpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ScalebarYSpinner.Layout.Row = 13;
            app.ScalebarYSpinner.Layout.Column = 8;
            app.ScalebarYSpinner.Value = 0.1;

            % Create DiracPSFCheckBox
            app.DiracPSFCheckBox = uicheckbox(app.GridLayout);
            app.DiracPSFCheckBox.ValueChangedFcn = createCallbackFcn(app, @DiracPSFCheckBoxValueChanged, true);
            app.DiracPSFCheckBox.Text = 'Dirac PSF';
            app.DiracPSFCheckBox.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DiracPSFCheckBox.Layout.Row = 17;
            app.DiracPSFCheckBox.Layout.Column = [38 41];

            % Create ResetButton_2
            app.ResetButton_2 = uibutton(app.GridLayout, 'push');
            app.ResetButton_2.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_2Pushed, true);
            app.ResetButton_2.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ResetButton_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ResetButton_2.Layout.Row = 11;
            app.ResetButton_2.Layout.Column = 32;
            app.ResetButton_2.Text = 'Reset';

            % Create ConnectCamButton
            app.ConnectCamButton = uibutton(app.GridLayout, 'state');
            app.ConnectCamButton.ValueChangedFcn = createCallbackFcn(app, @ConnectCamButtonValueChanged, true);
            app.ConnectCamButton.Text = 'Connect Cam.';
            app.ConnectCamButton.WordWrap = 'on';
            app.ConnectCamButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ConnectCamButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ConnectCamButton.Layout.Row = 2;
            app.ConnectCamButton.Layout.Column = [6 10];

            % Create CaptureNButton
            app.CaptureNButton = uibutton(app.GridLayout, 'push');
            app.CaptureNButton.ButtonPushedFcn = createCallbackFcn(app, @CaptureNButtonPushed, true);
            app.CaptureNButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.CaptureNButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.CaptureNButton.Layout.Row = 13;
            app.CaptureNButton.Layout.Column = [37 41];
            app.CaptureNButton.Text = 'CaptureN';

            % Create BuildbiasButton
            app.BuildbiasButton = uibutton(app.GridLayout, 'push');
            app.BuildbiasButton.ButtonPushedFcn = createCallbackFcn(app, @BuildbiasButtonPushed, true);
            app.BuildbiasButton.WordWrap = 'on';
            app.BuildbiasButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.BuildbiasButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.BuildbiasButton.Layout.Row = [7 8];
            app.BuildbiasButton.Layout.Column = [37 41];
            app.BuildbiasButton.Text = 'Build bias';

            % Create CaptureCalibButton
            app.CaptureCalibButton = uibutton(app.GridLayout, 'push');
            app.CaptureCalibButton.ButtonPushedFcn = createCallbackFcn(app, @CaptureCalibButtonPushed, true);
            app.CaptureCalibButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.CaptureCalibButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.CaptureCalibButton.Layout.Row = [10 11];
            app.CaptureCalibButton.Layout.Column = [37 41];
            app.CaptureCalibButton.Text = 'Capture Calib.';

            % Create ConnectzstageButton
            app.ConnectzstageButton = uibutton(app.GridLayout, 'state');
            app.ConnectzstageButton.ValueChangedFcn = createCallbackFcn(app, @ConnectzstageButtonValueChanged, true);
            app.ConnectzstageButton.Text = 'Connect z-stage';
            app.ConnectzstageButton.WordWrap = 'on';
            app.ConnectzstageButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ConnectzstageButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ConnectzstageButton.Layout.Row = 2;
            app.ConnectzstageButton.Layout.Column = [17 21];

            % Create Button
            app.Button = uibutton(app.GridLayout, 'push');
            app.Button.ButtonPushedFcn = createCallbackFcn(app, @ButtonPushed, true);
            app.Button.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button.Layout.Row = 2;
            app.Button.Layout.Column = 22;
            app.Button.Text = '💒';

            % Create ButtonUpJog
            app.ButtonUpJog = uibutton(app.GridLayout, 'push');
            app.ButtonUpJog.ButtonPushedFcn = createCallbackFcn(app, @ButtonUpJogPushed, true);
            app.ButtonUpJog.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ButtonUpJog.FontSize = 10;
            app.ButtonUpJog.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ButtonUpJog.Layout.Row = [5 6];
            app.ButtonUpJog.Layout.Column = 18;
            app.ButtonUpJog.Text = '🔺';

            % Create ButtonDownJog
            app.ButtonDownJog = uibutton(app.GridLayout, 'push');
            app.ButtonDownJog.ButtonPushedFcn = createCallbackFcn(app, @ButtonDownJogPushed, true);
            app.ButtonDownJog.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ButtonDownJog.FontSize = 10;
            app.ButtonDownJog.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ButtonDownJog.Layout.Row = [5 6];
            app.ButtonDownJog.Layout.Column = 22;
            app.ButtonDownJog.Text = '🔻';

            % Create ButtonUpRange
            app.ButtonUpRange = uibutton(app.GridLayout, 'push');
            app.ButtonUpRange.ButtonPushedFcn = createCallbackFcn(app, @ButtonUpRangePushed, true);
            app.ButtonUpRange.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ButtonUpRange.FontSize = 10;
            app.ButtonUpRange.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ButtonUpRange.Layout.Row = 11;
            app.ButtonUpRange.Layout.Column = 31;
            app.ButtonUpRange.Text = '🔺';

            % Create ButtonDownRange
            app.ButtonDownRange = uibutton(app.GridLayout, 'push');
            app.ButtonDownRange.ButtonPushedFcn = createCallbackFcn(app, @ButtonDownRangePushed, true);
            app.ButtonDownRange.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ButtonDownRange.FontSize = 10;
            app.ButtonDownRange.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ButtonDownRange.Layout.Row = 11;
            app.ButtonDownRange.Layout.Column = 33;
            app.ButtonDownRange.Text = '🔻';

            % Create CamerastatusLabel
            app.CamerastatusLabel = uilabel(app.GridLayout);
            app.CamerastatusLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.CamerastatusLabel.Layout.Row = [1 3];
            app.CamerastatusLabel.Layout.Column = [36 42];
            app.CamerastatusLabel.Text = 'Camera status:';

            % Create ConnectrotstageButton
            app.ConnectrotstageButton = uibutton(app.GridLayout, 'state');
            app.ConnectrotstageButton.ValueChangedFcn = createCallbackFcn(app, @ConnectrotstageButtonValueChanged, true);
            app.ConnectrotstageButton.Text = 'Connect rot-stage';
            app.ConnectrotstageButton.WordWrap = 'on';
            app.ConnectrotstageButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ConnectrotstageButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ConnectrotstageButton.Layout.Row = 2;
            app.ConnectrotstageButton.Layout.Column = [31 34];

            % Create SyncwhencaptureNButton
            app.SyncwhencaptureNButton = uibutton(app.GridLayout, 'state');
            app.SyncwhencaptureNButton.ValueChangedFcn = createCallbackFcn(app, @SyncwhencaptureNButtonValueChanged, true);
            app.SyncwhencaptureNButton.Text = '🔒Sync when captureN';
            app.SyncwhencaptureNButton.WordWrap = 'on';
            app.SyncwhencaptureNButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.SyncwhencaptureNButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.SyncwhencaptureNButton.Layout.Row = [40 41];
            app.SyncwhencaptureNButton.Layout.Column = [37 42];

            % Create VarButton
            app.VarButton = uibutton(app.GridLayout, 'state');
            app.VarButton.ValueChangedFcn = createCallbackFcn(app, @VarButtonValueChanged, true);
            app.VarButton.Text = 'Var';
            app.VarButton.WordWrap = 'on';
            app.VarButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.VarButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.VarButton.Layout.Row = [42 44];
            app.VarButton.Layout.Column = [37 42];

            % Create GeoloadedLampLabel
            app.GeoloadedLampLabel = uilabel(app.GridLayout);
            app.GeoloadedLampLabel.HorizontalAlignment = 'right';
            app.GeoloadedLampLabel.WordWrap = 'on';
            app.GeoloadedLampLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.GeoloadedLampLabel.Layout.Row = 11;
            app.GeoloadedLampLabel.Layout.Column = [1 2];
            app.GeoloadedLampLabel.Text = 'Geo. loaded';

            % Create GeoloadedLamp
            app.GeoloadedLamp = uilamp(app.GridLayout);
            app.GeoloadedLamp.Layout.Row = 11;
            app.GeoloadedLamp.Layout.Column = [3 4];
            app.GeoloadedLamp.Color = [0.651 0.651 0.651];

            % Create RealtimecaptureSwitchLabel
            app.RealtimecaptureSwitchLabel = uilabel(app.GridLayout);
            app.RealtimecaptureSwitchLabel.HorizontalAlignment = 'center';
            app.RealtimecaptureSwitchLabel.WordWrap = 'on';
            app.RealtimecaptureSwitchLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.RealtimecaptureSwitchLabel.Layout.Row = [6 7];
            app.RealtimecaptureSwitchLabel.Layout.Column = [6 10];
            app.RealtimecaptureSwitchLabel.Text = 'Real-time capture';

            % Create RealtimecaptureSwitch
            app.RealtimecaptureSwitch = uiswitch(app.GridLayout, 'slider');
            app.RealtimecaptureSwitch.ValueChangedFcn = createCallbackFcn(app, @RealtimecaptureSwitchValueChanged, true);
            app.RealtimecaptureSwitch.Enable = 'off';
            app.RealtimecaptureSwitch.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.RealtimecaptureSwitch.Layout.Row = [4 5];
            app.RealtimecaptureSwitch.Layout.Column = [7 9];

            % Create nItersSpinnerLabel
            app.nItersSpinnerLabel = uilabel(app.GridLayout);
            app.nItersSpinnerLabel.HorizontalAlignment = 'right';
            app.nItersSpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.nItersSpinnerLabel.Layout.Row = 19;
            app.nItersSpinnerLabel.Layout.Column = [37 39];
            app.nItersSpinnerLabel.Text = 'nIters';

            % Create nItersSpinner
            app.nItersSpinner = uispinner(app.GridLayout);
            app.nItersSpinner.Limits = [1 100];
            app.nItersSpinner.RoundFractionalValues = 'on';
            app.nItersSpinner.ValueDisplayFormat = '%d';
            app.nItersSpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.nItersSpinner.Layout.Row = 19;
            app.nItersSpinner.Layout.Column = [40 42];
            app.nItersSpinner.Value = 6;

            % Create PSFnEditFieldLabel
            app.PSFnEditFieldLabel = uilabel(app.GridLayout);
            app.PSFnEditFieldLabel.HorizontalAlignment = 'right';
            app.PSFnEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.PSFnEditFieldLabel.Layout.Row = 15;
            app.PSFnEditFieldLabel.Layout.Column = [37 39];
            app.PSFnEditFieldLabel.Text = 'PSF^n';

            % Create PSFnEditField
            app.PSFnEditField = uieditfield(app.GridLayout, 'numeric');
            app.PSFnEditField.Limits = [0 100];
            app.PSFnEditField.ValueChangedFcn = createCallbackFcn(app, @PSFnEditFieldValueChanged, true);
            app.PSFnEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.PSFnEditField.Layout.Row = 15;
            app.PSFnEditField.Layout.Column = [40 41];
            app.PSFnEditField.Value = 2;

            % Create DepthrangeEditFieldLabel
            app.DepthrangeEditFieldLabel = uilabel(app.GridLayout);
            app.DepthrangeEditFieldLabel.HorizontalAlignment = 'right';
            app.DepthrangeEditFieldLabel.WordWrap = 'on';
            app.DepthrangeEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DepthrangeEditFieldLabel.Layout.Row = 11;
            app.DepthrangeEditFieldLabel.Layout.Column = [23 26];
            app.DepthrangeEditFieldLabel.Text = 'Depth range ';

            % Create DepthrangeEditField
            app.DepthrangeEditField = uieditfield(app.GridLayout, 'numeric');
            app.DepthrangeEditField.ValueChangedFcn = createCallbackFcn(app, @DepthrangeEditFieldValueChanged, true);
            app.DepthrangeEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DepthrangeEditField.Layout.Row = 11;
            app.DepthrangeEditField.Layout.Column = [27 28];

            % Create EditFieldLabel
            app.EditFieldLabel = uilabel(app.GridLayout);
            app.EditFieldLabel.HorizontalAlignment = 'center';
            app.EditFieldLabel.WordWrap = 'on';
            app.EditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.EditFieldLabel.Layout.Row = 11;
            app.EditFieldLabel.Layout.Column = 29;
            app.EditFieldLabel.Text = '-';

            % Create EditField
            app.EditField = uieditfield(app.GridLayout, 'numeric');
            app.EditField.ValueChangedFcn = createCallbackFcn(app, @EditFieldValueChanged, true);
            app.EditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.EditField.Layout.Row = 11;
            app.EditField.Layout.Column = 30;
            app.EditField.Value = 1;

            % Create FPSEditFieldLabel
            app.FPSEditFieldLabel = uilabel(app.GridLayout);
            app.FPSEditFieldLabel.HorizontalAlignment = 'right';
            app.FPSEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.FPSEditFieldLabel.Layout.Row = 2;
            app.FPSEditFieldLabel.Layout.Column = [12 13];
            app.FPSEditFieldLabel.Text = 'FPS';

            % Create FPSEditField
            app.FPSEditField = uieditfield(app.GridLayout, 'numeric');
            app.FPSEditField.ValueChangedFcn = createCallbackFcn(app, @FPSEditFieldValueChanged, true);
            app.FPSEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.FPSEditField.Layout.Row = 2;
            app.FPSEditField.Layout.Column = [14 15];
            app.FPSEditField.Value = 15;

            % Create AvgEditFieldLabel
            app.AvgEditFieldLabel = uilabel(app.GridLayout);
            app.AvgEditFieldLabel.HorizontalAlignment = 'right';
            app.AvgEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.AvgEditFieldLabel.Layout.Row = [3 4];
            app.AvgEditFieldLabel.Layout.Column = [11 13];
            app.AvgEditFieldLabel.Text = 'Avg#';

            % Create AvgEditField
            app.AvgEditField = uieditfield(app.GridLayout, 'numeric');
            app.AvgEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.AvgEditField.Layout.Row = [3 4];
            app.AvgEditField.Layout.Column = [14 15];
            app.AvgEditField.Value = 1;

            % Create NEditFieldLabel
            app.NEditFieldLabel = uilabel(app.GridLayout);
            app.NEditFieldLabel.HorizontalAlignment = 'right';
            app.NEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.NEditFieldLabel.Layout.Row = 12;
            app.NEditFieldLabel.Layout.Column = [35 36];
            app.NEditFieldLabel.Text = 'N';

            % Create NEditField
            app.NEditField = uieditfield(app.GridLayout, 'numeric');
            app.NEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.NEditField.Layout.Row = 12;
            app.NEditField.Layout.Column = [38 41];
            app.NEditField.Value = 10;

            % Create TempSpinnerLabel
            app.TempSpinnerLabel = uilabel(app.GridLayout);
            app.TempSpinnerLabel.HorizontalAlignment = 'right';
            app.TempSpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.TempSpinnerLabel.Layout.Row = [5 6];
            app.TempSpinnerLabel.Layout.Column = [36 37];
            app.TempSpinnerLabel.Text = 'Temp.';

            % Create TempSpinner
            app.TempSpinner = uispinner(app.GridLayout);
            app.TempSpinner.ValueChangedFcn = createCallbackFcn(app, @TempSpinnerValueChanged, true);
            app.TempSpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.TempSpinner.Layout.Row = [5 6];
            app.TempSpinner.Layout.Column = [39 42];
            app.TempSpinner.Value = -40;

            % Create OutlierEditFieldLabel
            app.OutlierEditFieldLabel = uilabel(app.GridLayout);
            app.OutlierEditFieldLabel.HorizontalAlignment = 'right';
            app.OutlierEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.OutlierEditFieldLabel.Layout.Row = [5 6];
            app.OutlierEditFieldLabel.Layout.Column = [11 13];
            app.OutlierEditFieldLabel.Text = 'Outlier%';

            % Create OutlierEditField
            app.OutlierEditField = uieditfield(app.GridLayout, 'numeric');
            app.OutlierEditField.Limits = [0 100];
            app.OutlierEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.OutlierEditField.Layout.Row = [5 6];
            app.OutlierEditField.Layout.Column = [14 15];
            app.OutlierEditField.Value = 100;

            % Create zmmEditFieldLabel
            app.zmmEditFieldLabel = uilabel(app.GridLayout);
            app.zmmEditFieldLabel.HorizontalAlignment = 'right';
            app.zmmEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.zmmEditFieldLabel.Layout.Row = [3 4];
            app.zmmEditFieldLabel.Layout.Column = [16 17];
            app.zmmEditFieldLabel.Text = 'z (mm)';

            % Create zmmEditField
            app.zmmEditField = uieditfield(app.GridLayout, 'numeric');
            app.zmmEditField.ValueDisplayFormat = '%7.6g';
            app.zmmEditField.ValueChangedFcn = createCallbackFcn(app, @zmmEditFieldValueChanged, true);
            app.zmmEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.zmmEditField.Layout.Row = [3 4];
            app.zmmEditField.Layout.Column = [18 21];

            % Create JogEditFieldLabel
            app.JogEditFieldLabel = uilabel(app.GridLayout);
            app.JogEditFieldLabel.HorizontalAlignment = 'right';
            app.JogEditFieldLabel.WordWrap = 'on';
            app.JogEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.JogEditFieldLabel.Layout.Row = [5 6];
            app.JogEditFieldLabel.Layout.Column = 17;
            app.JogEditFieldLabel.Text = 'Jog';

            % Create JogEditField
            app.JogEditField = uieditfield(app.GridLayout, 'numeric');
            app.JogEditField.ValueDisplayFormat = '%11.5g';
            app.JogEditField.ValueChangedFcn = createCallbackFcn(app, @JogEditFieldValueChanged, true);
            app.JogEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.JogEditField.Layout.Row = [5 6];
            app.JogEditField.Layout.Column = [19 21];
            app.JogEditField.Value = 0.1;

            % Create stepdegLabel
            app.stepdegLabel = uilabel(app.GridLayout);
            app.stepdegLabel.HorizontalAlignment = 'right';
            app.stepdegLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.stepdegLabel.Layout.Row = [3 4];
            app.stepdegLabel.Layout.Column = [30 32];
            app.stepdegLabel.Text = 'Step(deg)';

            % Create StepdegEditField
            app.StepdegEditField = uieditfield(app.GridLayout, 'numeric');
            app.StepdegEditField.ValueDisplayFormat = '%7.6g';
            app.StepdegEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StepdegEditField.Layout.Row = [3 4];
            app.StepdegEditField.Layout.Column = [33 34];
            app.StepdegEditField.Value = 1.5;

            % Create PosdegEditFieldLabel
            app.PosdegEditFieldLabel = uilabel(app.GridLayout);
            app.PosdegEditFieldLabel.HorizontalAlignment = 'right';
            app.PosdegEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.PosdegEditFieldLabel.Layout.Row = [5 6];
            app.PosdegEditFieldLabel.Layout.Column = [30 32];
            app.PosdegEditFieldLabel.Text = 'Pos (deg)';

            % Create PosdegEditField
            app.PosdegEditField = uieditfield(app.GridLayout, 'numeric');
            app.PosdegEditField.ValueDisplayFormat = '%7.6g';
            app.PosdegEditField.ValueChangedFcn = createCallbackFcn(app, @PosdegEditFieldValueChanged, true);
            app.PosdegEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.PosdegEditField.Layout.Row = [5 6];
            app.PosdegEditField.Layout.Column = [33 34];

            % Create AvgforsyncCaptureNEditFieldLabel
            app.AvgforsyncCaptureNEditFieldLabel = uilabel(app.GridLayout);
            app.AvgforsyncCaptureNEditFieldLabel.HorizontalAlignment = 'right';
            app.AvgforsyncCaptureNEditFieldLabel.WordWrap = 'on';
            app.AvgforsyncCaptureNEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.AvgforsyncCaptureNEditFieldLabel.Layout.Row = 39;
            app.AvgforsyncCaptureNEditFieldLabel.Layout.Column = [36 40];
            app.AvgforsyncCaptureNEditFieldLabel.Text = 'Avg# for sync CaptureN';

            % Create AvgforsyncCaptureNEditField
            app.AvgforsyncCaptureNEditField = uieditfield(app.GridLayout, 'numeric');
            app.AvgforsyncCaptureNEditField.RoundFractionalValues = 'on';
            app.AvgforsyncCaptureNEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.AvgforsyncCaptureNEditField.Layout.Row = 39;
            app.AvgforsyncCaptureNEditField.Layout.Column = [41 42];
            app.AvgforsyncCaptureNEditField.Value = 1;

            % Create Lamp
            app.Lamp = uilamp(app.GridLayout);
            app.Lamp.Layout.Row = 24;
            app.Lamp.Layout.Column = 36;
            app.Lamp.Color = [0.502 0.502 0.502];

            % Create PrepareReconButton
            app.PrepareReconButton = uibutton(app.GridLayout, 'push');
            app.PrepareReconButton.ButtonPushedFcn = createCallbackFcn(app, @PrepareReconButtonPushed, true);
            app.PrepareReconButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.PrepareReconButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.PrepareReconButton.Layout.Row = [23 25];
            app.PrepareReconButton.Layout.Column = [37 42];
            app.PrepareReconButton.Text = 'Prepare Recon.';

            % Create DamparEditFieldLabel
            app.DamparEditFieldLabel = uilabel(app.GridLayout);
            app.DamparEditFieldLabel.HorizontalAlignment = 'right';
            app.DamparEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DamparEditFieldLabel.Layout.Row = 20;
            app.DamparEditFieldLabel.Layout.Column = [36 38];
            app.DamparEditFieldLabel.Text = 'Dampar';

            % Create DamparEditField
            app.DamparEditField = uieditfield(app.GridLayout, 'numeric');
            app.DamparEditField.ValueDisplayFormat = '%.3e';
            app.DamparEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DamparEditField.Layout.Row = 20;
            app.DamparEditField.Layout.Column = [39 42];

            % Create ReadouteEditFieldLabel
            app.ReadouteEditFieldLabel = uilabel(app.GridLayout);
            app.ReadouteEditFieldLabel.HorizontalAlignment = 'right';
            app.ReadouteEditFieldLabel.WordWrap = 'on';
            app.ReadouteEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ReadouteEditFieldLabel.Layout.Row = [21 22];
            app.ReadouteEditFieldLabel.Layout.Column = [36 38];
            app.ReadouteEditFieldLabel.Text = 'Readout (e-)';

            % Create ReadouteEditField
            app.ReadouteEditField = uieditfield(app.GridLayout, 'numeric');
            app.ReadouteEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ReadouteEditField.Layout.Row = 21;
            app.ReadouteEditField.Layout.Column = [39 42];

            % Create Lamp_2
            app.Lamp_2 = uilamp(app.GridLayout);
            app.Lamp_2.Layout.Row = 29;
            app.Lamp_2.Layout.Column = 36;
            app.Lamp_2.Color = [0.502 0.502 0.502];

            % Create ConnectxystageButton
            app.ConnectxystageButton = uibutton(app.GridLayout, 'state');
            app.ConnectxystageButton.ValueChangedFcn = createCallbackFcn(app, @ConnectxystageButtonValueChanged, true);
            app.ConnectxystageButton.Text = 'Connect xy-stage';
            app.ConnectxystageButton.WordWrap = 'on';
            app.ConnectxystageButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ConnectxystageButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ConnectxystageButton.Layout.Row = 2;
            app.ConnectxystageButton.Layout.Column = [24 27];

            % Create xmmEditFieldLabel
            app.xmmEditFieldLabel = uilabel(app.GridLayout);
            app.xmmEditFieldLabel.HorizontalAlignment = 'right';
            app.xmmEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.xmmEditFieldLabel.Layout.Row = [3 4];
            app.xmmEditFieldLabel.Layout.Column = [23 24];
            app.xmmEditFieldLabel.Text = 'x (mm)';

            % Create xmmEditField
            app.xmmEditField = uieditfield(app.GridLayout, 'numeric');
            app.xmmEditField.ValueDisplayFormat = '%.3f';
            app.xmmEditField.ValueChangedFcn = createCallbackFcn(app, @xmmEditFieldValueChanged, true);
            app.xmmEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.xmmEditField.Layout.Row = [3 4];
            app.xmmEditField.Layout.Column = [25 27];

            % Create ymmEditFieldLabel
            app.ymmEditFieldLabel = uilabel(app.GridLayout);
            app.ymmEditFieldLabel.HorizontalAlignment = 'right';
            app.ymmEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ymmEditFieldLabel.Layout.Row = [5 6];
            app.ymmEditFieldLabel.Layout.Column = [23 24];
            app.ymmEditFieldLabel.Text = 'y (mm)';

            % Create ymmEditField
            app.ymmEditField = uieditfield(app.GridLayout, 'numeric');
            app.ymmEditField.ValueDisplayFormat = '%.3f';
            app.ymmEditField.ValueChangedFcn = createCallbackFcn(app, @ymmEditFieldValueChanged, true);
            app.ymmEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ymmEditField.Layout.Row = [5 6];
            app.ymmEditField.Layout.Column = [25 27];

            % Create FullrangeumEditFieldLabel_2
            app.FullrangeumEditFieldLabel_2 = uilabel(app.GridLayout);
            app.FullrangeumEditFieldLabel_2.HorizontalAlignment = 'right';
            app.FullrangeumEditFieldLabel_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.FullrangeumEditFieldLabel_2.Layout.Row = 10;
            app.FullrangeumEditFieldLabel_2.Layout.Column = [22 26];
            app.FullrangeumEditFieldLabel_2.Text = 'Full range (um)';

            % Create FullrangeumEditField
            app.FullrangeumEditField = uieditfield(app.GridLayout, 'numeric');
            app.FullrangeumEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.FullrangeumEditField.Layout.Row = 10;
            app.FullrangeumEditField.Layout.Column = [27 28];
            app.FullrangeumEditField.Value = 600;

            % Create StitchXSpinnerLabel
            app.StitchXSpinnerLabel = uilabel(app.GridLayout);
            app.StitchXSpinnerLabel.HorizontalAlignment = 'right';
            app.StitchXSpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StitchXSpinnerLabel.Layout.Row = 7;
            app.StitchXSpinnerLabel.Layout.Column = [23 24];
            app.StitchXSpinnerLabel.Text = 'Stitch X';

            % Create StitchXSpinner
            app.StitchXSpinner = uispinner(app.GridLayout);
            app.StitchXSpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StitchXSpinner.Layout.Row = 7;
            app.StitchXSpinner.Layout.Column = [25 26];
            app.StitchXSpinner.Value = 3;

            % Create StitchYSpinnerLabel
            app.StitchYSpinnerLabel = uilabel(app.GridLayout);
            app.StitchYSpinnerLabel.HorizontalAlignment = 'right';
            app.StitchYSpinnerLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StitchYSpinnerLabel.Layout.Row = 8;
            app.StitchYSpinnerLabel.Layout.Column = [23 24];
            app.StitchYSpinnerLabel.Text = 'Stitch Y';

            % Create StitchYSpinner
            app.StitchYSpinner = uispinner(app.GridLayout);
            app.StitchYSpinner.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StitchYSpinner.Layout.Row = 8;
            app.StitchYSpinner.Layout.Column = [25 26];
            app.StitchYSpinner.Value = 3;

            % Create CaptureNStitchButton
            app.CaptureNStitchButton = uibutton(app.GridLayout, 'push');
            app.CaptureNStitchButton.ButtonPushedFcn = createCallbackFcn(app, @CaptureNStitchButtonPushed, true);
            app.CaptureNStitchButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.CaptureNStitchButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.CaptureNStitchButton.Layout.Row = 14;
            app.CaptureNStitchButton.Layout.Column = [37 41];
            app.CaptureNStitchButton.Text = 'CaptureN+Stitch';

            % Create PSFsloadedLampLabel
            app.PSFsloadedLampLabel = uilabel(app.GridLayout);
            app.PSFsloadedLampLabel.HorizontalAlignment = 'right';
            app.PSFsloadedLampLabel.WordWrap = 'on';
            app.PSFsloadedLampLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.PSFsloadedLampLabel.Layout.Row = 12;
            app.PSFsloadedLampLabel.Layout.Column = [1 2];
            app.PSFsloadedLampLabel.Text = 'PSFs loaded';

            % Create PSFsloadedLamp
            app.PSFsloadedLamp = uilamp(app.GridLayout);
            app.PSFsloadedLamp.Layout.Row = 12;
            app.PSFsloadedLamp.Layout.Column = [3 4];
            app.PSFsloadedLamp.Color = [0.651 0.651 0.651];

            % Create Button_2
            app.Button_2 = uibutton(app.GridLayout, 'push');
            app.Button_2.ButtonPushedFcn = createCallbackFcn(app, @Button_2Pushed, true);
            app.Button_2.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_2.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_2.Layout.Row = 2;
            app.Button_2.Layout.Column = [28 29];
            app.Button_2.Text = '📍';

            % Create Button_3
            app.Button_3 = uibutton(app.GridLayout, 'push');
            app.Button_3.ButtonPushedFcn = createCallbackFcn(app, @Button_3Pushed, true);
            app.Button_3.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_3.FontSize = 10;
            app.Button_3.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_3.Layout.Row = [3 4];
            app.Button_3.Layout.Column = 28;
            app.Button_3.Text = '🔽';

            % Create Button_4
            app.Button_4 = uibutton(app.GridLayout, 'push');
            app.Button_4.ButtonPushedFcn = createCallbackFcn(app, @Button_4Pushed, true);
            app.Button_4.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_4.FontSize = 10;
            app.Button_4.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_4.Layout.Row = [5 6];
            app.Button_4.Layout.Column = 28;
            app.Button_4.Text = '🔽';

            % Create StepXmmEditFieldLabel
            app.StepXmmEditFieldLabel = uilabel(app.GridLayout);
            app.StepXmmEditFieldLabel.HorizontalAlignment = 'right';
            app.StepXmmEditFieldLabel.FontSize = 10;
            app.StepXmmEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StepXmmEditFieldLabel.Layout.Row = 7;
            app.StepXmmEditFieldLabel.Layout.Column = [27 28];
            app.StepXmmEditFieldLabel.Text = 'Step X (mm)';

            % Create StepXmmEditField
            app.StepXmmEditField = uieditfield(app.GridLayout, 'numeric');
            app.StepXmmEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StepXmmEditField.Layout.Row = 7;
            app.StepXmmEditField.Layout.Column = 29;
            app.StepXmmEditField.Value = 1.2;

            % Create StepYmmEditFieldLabel
            app.StepYmmEditFieldLabel = uilabel(app.GridLayout);
            app.StepYmmEditFieldLabel.HorizontalAlignment = 'right';
            app.StepYmmEditFieldLabel.FontSize = 10;
            app.StepYmmEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StepYmmEditFieldLabel.Layout.Row = 8;
            app.StepYmmEditFieldLabel.Layout.Column = [27 28];
            app.StepYmmEditFieldLabel.Text = 'Step Y (mm)';

            % Create StepYmmEditField
            app.StepYmmEditField = uieditfield(app.GridLayout, 'numeric');
            app.StepYmmEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.StepYmmEditField.Layout.Row = 8;
            app.StepYmmEditField.Layout.Column = 29;
            app.StepYmmEditField.Value = 1.2;

            % Create nDepthEditFieldLabel
            app.nDepthEditFieldLabel = uilabel(app.GridLayout);
            app.nDepthEditFieldLabel.HorizontalAlignment = 'right';
            app.nDepthEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.nDepthEditFieldLabel.Layout.Row = 10;
            app.nDepthEditFieldLabel.Layout.Column = [29 30];
            app.nDepthEditFieldLabel.Text = 'nDepth';

            % Create nDepthEditField
            app.nDepthEditField = uieditfield(app.GridLayout, 'numeric');
            app.nDepthEditField.Limits = [1 2048];
            app.nDepthEditField.RoundFractionalValues = 'on';
            app.nDepthEditField.ValueChangedFcn = createCallbackFcn(app, @nDepthEditFieldValueChanged, true);
            app.nDepthEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.nDepthEditField.Layout.Row = 10;
            app.nDepthEditField.Layout.Column = [31 32];
            app.nDepthEditField.Value = 51;

            % Create x_jog_plus
            app.x_jog_plus = uibutton(app.GridLayout, 'push');
            app.x_jog_plus.ButtonPushedFcn = createCallbackFcn(app, @x_jog_plusPushed, true);
            app.x_jog_plus.VerticalAlignment = 'top';
            app.x_jog_plus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.x_jog_plus.FontSize = 8;
            app.x_jog_plus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.x_jog_plus.Layout.Row = 3;
            app.x_jog_plus.Layout.Column = 29;
            app.x_jog_plus.Text = '➕';

            % Create x_jog_minus
            app.x_jog_minus = uibutton(app.GridLayout, 'push');
            app.x_jog_minus.ButtonPushedFcn = createCallbackFcn(app, @x_jog_minusPushed, true);
            app.x_jog_minus.VerticalAlignment = 'top';
            app.x_jog_minus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.x_jog_minus.FontSize = 8;
            app.x_jog_minus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.x_jog_minus.Layout.Row = 4;
            app.x_jog_minus.Layout.Column = 29;
            app.x_jog_minus.Text = '➖';

            % Create y_jog_plus
            app.y_jog_plus = uibutton(app.GridLayout, 'push');
            app.y_jog_plus.ButtonPushedFcn = createCallbackFcn(app, @y_jog_plusPushed, true);
            app.y_jog_plus.VerticalAlignment = 'top';
            app.y_jog_plus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.y_jog_plus.FontSize = 8;
            app.y_jog_plus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.y_jog_plus.Layout.Row = 5;
            app.y_jog_plus.Layout.Column = 29;
            app.y_jog_plus.Text = '➕';

            % Create y_jog_minus
            app.y_jog_minus = uibutton(app.GridLayout, 'push');
            app.y_jog_minus.ButtonPushedFcn = createCallbackFcn(app, @y_jog_minusPushed, true);
            app.y_jog_minus.VerticalAlignment = 'top';
            app.y_jog_minus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.y_jog_minus.FontSize = 8;
            app.y_jog_minus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.y_jog_minus.Layout.Row = 6;
            app.y_jog_minus.Layout.Column = 29;
            app.y_jog_minus.Text = '➖';

            % Create z_jog_range_plus
            app.z_jog_range_plus = uibutton(app.GridLayout, 'push');
            app.z_jog_range_plus.ButtonPushedFcn = createCallbackFcn(app, @z_jog_range_plusButtonPushed, true);
            app.z_jog_range_plus.VerticalAlignment = 'top';
            app.z_jog_range_plus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.z_jog_range_plus.FontSize = 10;
            app.z_jog_range_plus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.z_jog_range_plus.Layout.Row = 7;
            app.z_jog_range_plus.Layout.Column = 22;
            app.z_jog_range_plus.Text = '2x';

            % Create z_jog_range_minus
            app.z_jog_range_minus = uibutton(app.GridLayout, 'push');
            app.z_jog_range_minus.ButtonPushedFcn = createCallbackFcn(app, @z_jog_range_minusButtonPushed, true);
            app.z_jog_range_minus.VerticalAlignment = 'top';
            app.z_jog_range_minus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.z_jog_range_minus.FontSize = 10;
            app.z_jog_range_minus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.z_jog_range_minus.Layout.Row = 7;
            app.z_jog_range_minus.Layout.Column = [19 20];
            app.z_jog_range_minus.Text = '0.5x';

            % Create xy_jog_range_plus
            app.xy_jog_range_plus = uibutton(app.GridLayout, 'push');
            app.xy_jog_range_plus.ButtonPushedFcn = createCallbackFcn(app, @xy_jog_range_plusButtonPushed, true);
            app.xy_jog_range_plus.VerticalAlignment = 'top';
            app.xy_jog_range_plus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.xy_jog_range_plus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.xy_jog_range_plus.Layout.Row = 8;
            app.xy_jog_range_plus.Layout.Column = 30;
            app.xy_jog_range_plus.Text = '2x';

            % Create xy_jog_range_minus
            app.xy_jog_range_minus = uibutton(app.GridLayout, 'push');
            app.xy_jog_range_minus.ButtonPushedFcn = createCallbackFcn(app, @xy_jog_range_minusButtonPushed, true);
            app.xy_jog_range_minus.VerticalAlignment = 'top';
            app.xy_jog_range_minus.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.xy_jog_range_minus.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.xy_jog_range_minus.Layout.Row = 7;
            app.xy_jog_range_minus.Layout.Column = 30;
            app.xy_jog_range_minus.Text = '0.5x';

            % Create z_jog_range_reset
            app.z_jog_range_reset = uibutton(app.GridLayout, 'push');
            app.z_jog_range_reset.ButtonPushedFcn = createCallbackFcn(app, @z_jog_range_resetButtonPushed, true);
            app.z_jog_range_reset.VerticalAlignment = 'top';
            app.z_jog_range_reset.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.z_jog_range_reset.FontSize = 10;
            app.z_jog_range_reset.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.z_jog_range_reset.Layout.Row = 7;
            app.z_jog_range_reset.Layout.Column = 21;
            app.z_jog_range_reset.Text = 'R';

            % Create xy_jog_range_reset
            app.xy_jog_range_reset = uibutton(app.GridLayout, 'push');
            app.xy_jog_range_reset.ButtonPushedFcn = createCallbackFcn(app, @xy_jog_range_resetButtonPushed, true);
            app.xy_jog_range_reset.VerticalAlignment = 'top';
            app.xy_jog_range_reset.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.xy_jog_range_reset.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.xy_jog_range_reset.Layout.Row = 8;
            app.xy_jog_range_reset.Layout.Column = 31;
            app.xy_jog_range_reset.Text = 'R';

            % Create UpdateloadButton
            app.UpdateloadButton = uibutton(app.GridLayout, 'push');
            app.UpdateloadButton.ButtonPushedFcn = createCallbackFcn(app, @UpdateloadButtonPushed, true);
            app.UpdateloadButton.WordWrap = 'on';
            app.UpdateloadButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.UpdateloadButton.FontSize = 9;
            app.UpdateloadButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.UpdateloadButton.Layout.Row = 14;
            app.UpdateloadButton.Layout.Column = 1;
            app.UpdateloadButton.Text = 'Update&load';

            % Create AutoButton_3
            app.AutoButton_3 = uibutton(app.GridLayout, 'state');
            app.AutoButton_3.ValueChangedFcn = createCallbackFcn(app, @AutoButton_3ValueChanged, true);
            app.AutoButton_3.Text = 'Auto';
            app.AutoButton_3.WordWrap = 'on';
            app.AutoButton_3.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.AutoButton_3.FontSize = 10;
            app.AutoButton_3.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.AutoButton_3.Layout.Row = 14;
            app.AutoButton_3.Layout.Column = [3 4];

            % Create FrameSliderLabel
            app.FrameSliderLabel = uilabel(app.GridLayout);
            app.FrameSliderLabel.HorizontalAlignment = 'right';
            app.FrameSliderLabel.WordWrap = 'on';
            app.FrameSliderLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.FrameSliderLabel.Enable = 'off';
            app.FrameSliderLabel.Layout.Row = 38;
            app.FrameSliderLabel.Layout.Column = 1;
            app.FrameSliderLabel.Text = 'Frame#';

            % Create FrameSlider
            app.FrameSlider = uislider(app.GridLayout);
            app.FrameSlider.Limits = [1 100];
            app.FrameSlider.ValueChangedFcn = createCallbackFcn(app, @FrameSliderValueChanged, true);
            app.FrameSlider.FontSize = 10;
            app.FrameSlider.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.FrameSlider.Enable = 'off';
            app.FrameSlider.Layout.Row = 41;
            app.FrameSlider.Layout.Column = [1 4];
            app.FrameSlider.Value = 1;

            % Create Label_frame
            app.Label_frame = uilabel(app.GridLayout);
            app.Label_frame.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Label_frame.Layout.Row = 38;
            app.Label_frame.Layout.Column = [2 3];
            app.Label_frame.Text = '1/1';

            % Create Button_play_frame
            app.Button_play_frame = uibutton(app.GridLayout, 'state');
            app.Button_play_frame.ValueChangedFcn = createCallbackFcn(app, @Button_play_frameValueChanged, true);
            app.Button_play_frame.Text = 'Play';
            app.Button_play_frame.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_play_frame.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_play_frame.Layout.Row = [43 44];
            app.Button_play_frame.Layout.Column = 1;

            % Create Button_minus_3
            app.Button_minus_3 = uibutton(app.GridLayout, 'push');
            app.Button_minus_3.ButtonPushedFcn = createCallbackFcn(app, @Button_minus_3Pushed, true);
            app.Button_minus_3.VerticalAlignment = 'top';
            app.Button_minus_3.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_minus_3.FontSize = 18;
            app.Button_minus_3.FontWeight = 'bold';
            app.Button_minus_3.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_minus_3.Enable = 'off';
            app.Button_minus_3.Layout.Row = [43 44];
            app.Button_minus_3.Layout.Column = 3;
            app.Button_minus_3.Text = '-';

            % Create Button_plus_3
            app.Button_plus_3 = uibutton(app.GridLayout, 'push');
            app.Button_plus_3.ButtonPushedFcn = createCallbackFcn(app, @Button_plus_3Pushed, true);
            app.Button_plus_3.VerticalAlignment = 'top';
            app.Button_plus_3.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_plus_3.FontSize = 18;
            app.Button_plus_3.FontWeight = 'bold';
            app.Button_plus_3.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_plus_3.Enable = 'off';
            app.Button_plus_3.Layout.Row = [43 44];
            app.Button_plus_3.Layout.Column = 4;
            app.Button_plus_3.Text = '+';

            % Create TakeavgperEditFieldLabel
            app.TakeavgperEditFieldLabel = uilabel(app.GridLayout);
            app.TakeavgperEditFieldLabel.HorizontalAlignment = 'right';
            app.TakeavgperEditFieldLabel.WordWrap = 'on';
            app.TakeavgperEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.TakeavgperEditFieldLabel.Enable = 'off';
            app.TakeavgperEditFieldLabel.Layout.Row = 45;
            app.TakeavgperEditFieldLabel.Layout.Column = 1;
            app.TakeavgperEditFieldLabel.Text = 'Take avg. per';

            % Create TakeavgperEditField
            app.TakeavgperEditField = uieditfield(app.GridLayout, 'numeric');
            app.TakeavgperEditField.RoundFractionalValues = 'on';
            app.TakeavgperEditField.ValueChangedFcn = createCallbackFcn(app, @TakeavgperEditFieldValueChanged, true);
            app.TakeavgperEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.TakeavgperEditField.Enable = 'off';
            app.TakeavgperEditField.Layout.Row = 45;
            app.TakeavgperEditField.Layout.Column = [2 4];
            app.TakeavgperEditField.Value = 1;

            % Create MIPProjButton
            app.MIPProjButton = uibutton(app.GridLayout, 'state');
            app.MIPProjButton.ValueChangedFcn = createCallbackFcn(app, @MIPProjButtonValueChanged, true);
            app.MIPProjButton.Text = 'MIP Proj.';
            app.MIPProjButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.MIPProjButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MIPProjButton.Layout.Row = 11;
            app.MIPProjButton.Layout.Column = [6 8];

            % Create SaveSlicedtMovieButton
            app.SaveSlicedtMovieButton = uibutton(app.GridLayout, 'push');
            app.SaveSlicedtMovieButton.ButtonPushedFcn = createCallbackFcn(app, @SaveSlicedtMovieButtonPushed, true);
            app.SaveSlicedtMovieButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.SaveSlicedtMovieButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.SaveSlicedtMovieButton.Layout.Row = [34 35];
            app.SaveSlicedtMovieButton.Layout.Column = [37 42];
            app.SaveSlicedtMovieButton.Text = 'Save Sliced t-Movie';

            % Create MIP3DgammaEditFieldLabel
            app.MIP3DgammaEditFieldLabel = uilabel(app.GridLayout);
            app.MIP3DgammaEditFieldLabel.HorizontalAlignment = 'right';
            app.MIP3DgammaEditFieldLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MIP3DgammaEditFieldLabel.Layout.Row = 10;
            app.MIP3DgammaEditFieldLabel.Layout.Column = [5 8];
            app.MIP3DgammaEditFieldLabel.Text = 'MIP/3D gamma';

            % Create MIP3DgammaEditField
            app.MIP3DgammaEditField = uieditfield(app.GridLayout, 'numeric');
            app.MIP3DgammaEditField.Limits = [0.01 100];
            app.MIP3DgammaEditField.ValueChangedFcn = createCallbackFcn(app, @MIP3DgammaEditFieldValueChanged, true);
            app.MIP3DgammaEditField.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MIP3DgammaEditField.Layout.Row = 10;
            app.MIP3DgammaEditField.Layout.Column = [9 11];
            app.MIP3DgammaEditField.Value = 1;

            % Create MIPcolorrangeSliderLabel
            app.MIPcolorrangeSliderLabel = uilabel(app.GridLayout);
            app.MIPcolorrangeSliderLabel.HorizontalAlignment = 'right';
            app.MIPcolorrangeSliderLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MIPcolorrangeSliderLabel.Layout.Row = 8;
            app.MIPcolorrangeSliderLabel.Layout.Column = [5 8];
            app.MIPcolorrangeSliderLabel.Text = 'MIP color range';

            % Create MIPcolorrangeSlider
            app.MIPcolorrangeSlider = uislider(app.GridLayout, 'range');
            app.MIPcolorrangeSlider.Limits = [0 1];
            app.MIPcolorrangeSlider.ValueChangedFcn = createCallbackFcn(app, @MIPcolorrangeSliderValueChanged, true);
            app.MIPcolorrangeSlider.FontSize = 10;
            app.MIPcolorrangeSlider.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MIPcolorrangeSlider.Layout.Row = 8;
            app.MIPcolorrangeSlider.Layout.Column = [9 16];
            app.MIPcolorrangeSlider.Value = [0 1];

            % Create MIP3DclampSliderLabel
            app.MIP3DclampSliderLabel = uilabel(app.GridLayout);
            app.MIP3DclampSliderLabel.HorizontalAlignment = 'right';
            app.MIP3DclampSliderLabel.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MIP3DclampSliderLabel.Layout.Row = 9;
            app.MIP3DclampSliderLabel.Layout.Column = [5 8];
            app.MIP3DclampSliderLabel.Text = 'MIP/3D clamp';

            % Create MIP3DclampSlider
            app.MIP3DclampSlider = uislider(app.GridLayout, 'range');
            app.MIP3DclampSlider.Limits = [0 1];
            app.MIP3DclampSlider.ValueChangedFcn = createCallbackFcn(app, @MIP3DclampSliderValueChanged, true);
            app.MIP3DclampSlider.FontSize = 10;
            app.MIP3DclampSlider.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.MIP3DclampSlider.Layout.Row = 9;
            app.MIP3DclampSlider.Layout.Column = [9 16];
            app.MIP3DclampSlider.Value = [0 1];

            % Create Button_5
            app.Button_5 = uibutton(app.GridLayout, 'push');
            app.Button_5.ButtonPushedFcn = createCallbackFcn(app, @Button_5Pushed, true);
            app.Button_5.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.Button_5.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.Button_5.Layout.Row = 9;
            app.Button_5.Layout.Column = 17;
            app.Button_5.Text = '-';

            % Create RButton
            app.RButton = uibutton(app.GridLayout, 'push');
            app.RButton.ButtonPushedFcn = createCallbackFcn(app, @RButtonPushed, true);
            app.RButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.RButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.RButton.Layout.Row = 8;
            app.RButton.Layout.Column = 17;
            app.RButton.Text = 'R';

            % Create ViewingRawButton
            app.ViewingRawButton = uibutton(app.GridLayout, 'push');
            app.ViewingRawButton.ButtonPushedFcn = createCallbackFcn(app, @ViewingRawButtonPushed, true);
            app.ViewingRawButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ViewingRawButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ViewingRawButton.Layout.Row = 40;
            app.ViewingRawButton.Layout.Column = [1 2];
            app.ViewingRawButton.Text = 'Viewing Raw';

            % Create SetprocessparamButton
            app.SetprocessparamButton = uibutton(app.GridLayout, 'push');
            app.SetprocessparamButton.ButtonPushedFcn = createCallbackFcn(app, @SetprocessparamButtonPushed, true);
            app.SetprocessparamButton.WordWrap = 'on';
            app.SetprocessparamButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.SetprocessparamButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.SetprocessparamButton.Layout.Row = [10 11];
            app.SetprocessparamButton.Layout.Column = [18 22];
            app.SetprocessparamButton.Text = 'Set process param';

            % Create DDarkButton
            app.DDarkButton = uibutton(app.GridLayout, 'push');
            app.DDarkButton.ButtonPushedFcn = createCallbackFcn(app, @DDarkButtonPushed, true);
            app.DDarkButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.DDarkButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.DDarkButton.Layout.Row = 11;
            app.DDarkButton.Layout.Column = [9 11];
            app.DDarkButton.Text = '3D Dark';

            % Create saveDepthButton
            app.saveDepthButton = uibutton(app.GridLayout, 'push');
            app.saveDepthButton.ButtonPushedFcn = createCallbackFcn(app, @saveDepthButtonPushed, true);
            app.saveDepthButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.saveDepthButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.saveDepthButton.Layout.Row = 11;
            app.saveDepthButton.Layout.Column = 34;
            app.saveDepthButton.Text = 'save';

            % Create TrainButton
            app.TrainButton = uibutton(app.GridLayout, 'push');
            app.TrainButton.ButtonPushedFcn = createCallbackFcn(app, @TrainButtonPushed, true);
            app.TrainButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.TrainButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.TrainButton.Layout.Row = 9;
            app.TrainButton.Layout.Column = 1;
            app.TrainButton.Text = 'Train';

            % Create LoadButton
            app.LoadButton = uibutton(app.GridLayout, 'push');
            app.LoadButton.ButtonPushedFcn = createCallbackFcn(app, @LoadButtonPushed, true);
            app.LoadButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.LoadButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.LoadButton.Layout.Row = 9;
            app.LoadButton.Layout.Column = 2;
            app.LoadButton.Text = 'Load';

            % Create ApplyButton
            app.ApplyButton = uibutton(app.GridLayout, 'push');
            app.ApplyButton.ButtonPushedFcn = createCallbackFcn(app, @ApplyButtonPushed, true);
            app.ApplyButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.ApplyButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.ApplyButton.Layout.Row = 9;
            app.ApplyButton.Layout.Column = [3 4];
            app.ApplyButton.Text = 'Apply';

            % Create TmpFolderButton
            app.TmpFolderButton = uibutton(app.GridLayout, 'push');
            app.TmpFolderButton.ButtonPushedFcn = createCallbackFcn(app, @TmpFolderButtonPushed, true);
            app.TmpFolderButton.WordWrap = 'on';
            app.TmpFolderButton.BackgroundColor = [0.96078431372549 0.96078431372549 0.96078431372549];
            app.TmpFolderButton.FontSize = 9;
            app.TmpFolderButton.FontColor = [0.129411764705882 0.129411764705882 0.129411764705882];
            app.TmpFolderButton.Layout.Row = 14;
            app.TmpFolderButton.Layout.Column = 2;
            app.TmpFolderButton.Text = 'Tmp Folder';

            % Create DeepLabel_3
            app.DeepLabel_3 = uilabel(app.GridLayout);
            app.DeepLabel_3.FontSize = 8;
            app.DeepLabel_3.Layout.Row = 1;
            app.DeepLabel_3.Layout.Column = [17 19];
            app.DeepLabel_3.Text = '↑ Deep';

            % Create ShallowLabel_2
            app.ShallowLabel_2 = uilabel(app.GridLayout);
            app.ShallowLabel_2.FontSize = 8;
            app.ShallowLabel_2.Layout.Row = 1;
            app.ShallowLabel_2.Layout.Column = [20 23];
            app.ShallowLabel_2.Text = '     ↓Shallow';

            % Create DButton
            app.DButton = uibutton(app.GridLayout, 'push');
            app.DButton.ButtonPushedFcn = createCallbackFcn(app, @DButtonPushed, true);
            app.DButton.Layout.Row = 40;
            app.DButton.Layout.Column = [3 4];
            app.DButton.Text = '3D';

            % Create SavewarpedPSFButton
            app.SavewarpedPSFButton = uibutton(app.GridLayout, 'push');
            app.SavewarpedPSFButton.ButtonPushedFcn = createCallbackFcn(app, @SavewarpedPSFButtonPushed, true);
            app.SavewarpedPSFButton.Layout.Row = [26 27];
            app.SavewarpedPSFButton.Layout.Column = [37 42];
            app.SavewarpedPSFButton.Text = 'Save warped PSF';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = main_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end