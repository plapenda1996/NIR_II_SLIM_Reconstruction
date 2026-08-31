function ImageStackViewer(im_data, varargin)
    % ImageStackViewer - An interactive viewer for 3D image stacks.
    %
    % Syntax:
    %   ImageStackViewer(im_data)
    %   ImageStackViewer(im_data, 'Title', 'My Custom Title')
    %
    % Inputs:
    %   im_data - 3D image data (height x width x frames).
    %   'Title' - Optional title for the viewer window.
    % --- Input Parser ---
    p = inputParser;
    addRequired(p, 'im_data', @(x) ndims(x) == 3 && ~isempty(x));
    addParameter(p, 'Title', 'Image Stack Viewer', @ischar);
    addParameter(p, 'Framerate', 20, @isscalar);
    parse(p, im_data, varargin{:});
    % --- Image and Figure Setup ---
    [height, width, num_frames] = size(im_data);
    fig = figure('Name', p.Results.Title, ...
                 'NumberTitle', 'off', ...
                 'Position', [100, 100, 900, 700], ...
                 'ResizeFcn', @resizeCallback, ...
                 'Visible', 'off'); % Hide until fully drawn
    % --- Initial Parameters ---
    current_frame = 1;
    avg_window = 1;
    frame_rate = p.Results.Framerate; % Original frame rate
    wheel_step = 5;
    arrow_step = 1;
    normalize_flag = false;
    low_percentile = 0.1;
    high_percentile = 99.9;
    aspect_ratio = width / height; % This is the visual width/height ratio
    play_skip = 1;
    is_playing = false;
    play_timer = [];
    % --- Processed Data ---
    im_data = rescale(im_data);
    processed_data = im_data;
    effective_frames = num_frames;
    % --- UI Layout ---
    main_panel = uipanel('Parent', fig, 'Position', [0 0 1 1], 'BorderType', 'none');
    ax = axes('Parent', main_panel, ...
              'Position', [0.05 0.25 0.9 0.7], ...
              'XTick', [], 'YTick', [], ...
              'Box', 'on');
    img_handle = imshow(processed_data(:,:,1), [0,1], 'Parent', ax);
    control_y = 0.02;
    % --- Row 1: Averaging and Frame Rate ---
    uicontrol('Parent', main_panel, 'Style', 'text', 'String', 'Avg Window:', ...
              'Units', 'normalized', 'Position', [0.02 control_y+0.15 0.08 0.03], 'HorizontalAlignment', 'right');
    avg_edit = uicontrol('Parent', main_panel, 'Style', 'edit', 'String', '1', ...
                         'Units', 'normalized', 'Position', [0.105 control_y+0.15 0.04 0.03], 'Callback', @avgWindowCallback);
    uicontrol('Parent', main_panel, 'Style', 'text', 'String', 'Frame Rate (Hz):', ...
              'Units', 'normalized', 'Position', [0.16 control_y+0.15 0.09 0.03], 'HorizontalAlignment', 'right');
    fps_edit = uicontrol('Parent', main_panel, 'Style', 'edit', 'String', num2str(frame_rate), ...
                         'Units', 'normalized', 'Position', [0.255 control_y+0.15 0.04 0.03], 'Callback', @fpsCallback);
    frame_info_text = uicontrol('Parent', main_panel, 'Style', 'text', ...
                                'String', sprintf('Orig. Frames: %d | Eff. Frames: %d', num_frames, effective_frames), ...
                                'Units', 'normalized', 'Position', [0.31 control_y+0.15 0.25 0.03], 'HorizontalAlignment', 'left');
    % --- Row 2: Normalization and Aspect Ratio ---
    norm_checkbox = uicontrol('Parent', main_panel, 'Style', 'checkbox', 'String', 'Normalize', ...
                              'Units', 'normalized', 'Position', [0.02 control_y+0.11 0.07 0.03], 'Callback', @normalizeCallback);
    uicontrol('Parent', main_panel, 'Style', 'text', 'String', 'Low %:', ...
              'Units', 'normalized', 'Position', [0.09 control_y+0.11 0.05 0.03], 'HorizontalAlignment', 'right');
    low_perc_edit = uicontrol('Parent', main_panel, 'Style', 'edit', 'String', '0.01', ...
                              'Units', 'normalized', 'Position', [0.145 control_y+0.11 0.04 0.03], 'Callback', @percCallback);
    uicontrol('Parent', main_panel, 'Style', 'text', 'String', 'High %:', ...
              'Units', 'normalized', 'Position', [0.19 control_y+0.11 0.05 0.03], 'HorizontalAlignment', 'right');
    high_perc_edit = uicontrol('Parent', main_panel, 'Style', 'edit', 'String', '99.99', ...
                               'Units', 'normalized', 'Position', [0.245 control_y+0.11 0.04 0.03], 'Callback', @percCallback);
    
    % *** NEW: Add compact percentile preset buttons ***
    uicontrol('Parent', main_panel, 'Style', 'pushbutton', 'String', '0.1%', 'Units', 'normalized', ...
              'Position', [0.29, control_y+0.11, 0.04, 0.03], 'Callback', @(~,~) setPercentiles(0.1));
    uicontrol('Parent', main_panel, 'Style', 'pushbutton', 'String', '0.01%', 'Units', 'normalized', ...
              'Position', [0.335, control_y+0.11, 0.045, 0.03], 'Callback', @(~,~) setPercentiles(0.01));
    uicontrol('Parent', main_panel, 'Style', 'pushbutton', 'String', '0.001%', 'Units', 'normalized', ...
              'Position', [0.385, control_y+0.11, 0.05, 0.03], 'Callback', @(~,~) setPercentiles(0.001));

    % *** MODIFIED: Adjust positions of subsequent controls ***
    uicontrol('Parent', main_panel, 'Style', 'text', 'String', 'Aspect Ratio:', ...
              'Units', 'normalized', 'Position', [0.45 control_y+0.11 0.07 0.03], 'HorizontalAlignment', 'right');
    aspect_edit = uicontrol('Parent', main_panel, 'Style', 'edit', 'String', sprintf('%.2f', aspect_ratio), ...
                            'Units', 'normalized', 'Position', [0.525 control_y+0.11 0.05 0.03], 'Callback', @aspectCallback);
    uicontrol('Parent', main_panel, 'Style', 'pushbutton', 'String', 'Orig', 'Units', 'normalized', ...
              'Position', [0.58 control_y+0.11 0.04 0.03], 'Callback', @(~,~) setAspectRatio(1.7*width/height));
    uicontrol('Parent', main_panel, 'Style', 'pushbutton', 'String', 'x2', 'Units', 'normalized', ...
              'Position', [0.625 control_y+0.11 0.03 0.03], 'Callback', @(~,~) setAspectRatio(2*width/height));
    uicontrol('Parent', main_panel, 'Style', 'pushbutton', 'String', 'x4', 'Units', 'normalized', ...
              'Position', [0.66 control_y+0.11 0.03 0.03], 'Callback', @(~,~) setAspectRatio(4*width/height));
    
    % --- Row 3: Slider ---
    slider = uicontrol('Parent', main_panel, 'Style', 'slider', 'Units', 'normalized', ...
                       'Position', [0.05 control_y+0.07 0.90 0.03], ...
                       'Min', 1, 'Max', max(effective_frames, 1), 'Value', 1);
                       % 'Callback' has been removed from here

    % Add a listener for continuous value changes for real-time response
    addlistener(slider, 'ContinuousValueChange', @sliderCallback);

    updateSliderSteps();
    % --- Row 4: Navigation and Playback ---
    uicontrol('Parent', main_panel, 'Style', 'text', 'String', 'Wheel Step:', ...
              'Units', 'normalized', 'Position', [0.02 control_y+0.03 0.07 0.03], 'HorizontalAlignment', 'right');
    wheel_edit = uicontrol('Parent', main_panel, 'Style', 'edit', 'String', '5', ...
                           'Units', 'normalized', 'Position', [0.095 control_y+0.03 0.04 0.03], 'Callback', @wheelStepCallback);
    uicontrol('Parent', main_panel, 'Style', 'text', 'String', 'Arrow Step:', ...
              'Units', 'normalized', 'Position', [0.15 control_y+0.03 0.07 0.03], 'HorizontalAlignment', 'right');
    arrow_edit = uicontrol('Parent', main_panel, 'Style', 'edit', 'String', '1', ...
                           'Units', 'normalized', 'Position', [0.225 control_y+0.03 0.04 0.03], 'Callback', @arrowStepCallback);
    play_button = uicontrol('Parent', main_panel, 'Style', 'togglebutton', 'String', '▶ Play', ...
                            'Units', 'normalized', 'Position', [0.28 control_y+0.03 0.06 0.03], 'Callback', @playCallback);
    uicontrol('Parent', main_panel, 'Style', 'text', 'String', 'Skip:', ...
              'Units', 'normalized', 'Position', [0.35 control_y+0.03 0.03 0.03], 'HorizontalAlignment', 'right');
    skip_edit = uicontrol('Parent', main_panel, 'Style', 'edit', 'String', '1', ...
                          'Units', 'normalized', 'Position', [0.385 control_y+0.03 0.04 0.03], 'Callback', @skipCallback);
    % --- Callbacks and Final Setup ---
    set(fig, 'WindowScrollWheelFcn', @scrollWheelCallback);
    set(fig, 'KeyPressFcn', @keyPressCallback);
    set(fig, 'CloseRequestFcn', @closeCallback);
    setAspectRatio(aspect_ratio); 
    
    updateDisplay();
    set(fig, 'Visible', 'on'); 
    % --- Callback Function Definitions ---
    function avgWindowCallback(~, ~)
        new_avg = round(str2double(get(avg_edit, 'String')));
        if isscalar(new_avg) && ~isnan(new_avg) && new_avg >= 1 && new_avg <= num_frames
            avg_window = new_avg;
            processData();
            updateDisplay();
        else
            set(avg_edit, 'String', num2str(avg_window));
        end
    end
    function fpsCallback(~, ~)
        new_fps = str2double(get(fps_edit, 'String'));
        if isscalar(new_fps) && ~isnan(new_fps) && new_fps > 0
            frame_rate = new_fps;
            updateFrameTimeInfo();
            if is_playing
                stopPlayback();
                startPlayback();
            end
        else
            set(fps_edit, 'String', num2str(frame_rate));
        end
    end
    function normalizeCallback(~, ~)
        normalize_flag = get(norm_checkbox, 'Value');
        updateDisplay();
    end
    
    % *** NEW: Callback for preset percentile buttons ***
    function setPercentiles(p)
        low_percentile = p;
        high_percentile = 100 - p;
        set(low_perc_edit, 'String', num2str(low_percentile));
        set(high_perc_edit, 'String', num2str(high_percentile));
        if get(norm_checkbox, 'Value')
            updateDisplay();
        end
    end

    function percCallback(~, ~)
        low_val = str2double(get(low_perc_edit, 'String'));
        high_val = str2double(get(high_perc_edit, 'String'));
        valid = true;
        if isscalar(low_val) && ~isnan(low_val) && low_val >= 0 && low_val < high_val
            low_percentile = low_val;
        else
            set(low_perc_edit, 'String', num2str(low_percentile));
            valid = false;
        end
        if isscalar(high_val) && ~isnan(high_val) && high_val > low_percentile && high_val <= 100
            high_percentile = high_val;
        else
            set(high_perc_edit, 'String', num2str(high_percentile));
            valid = false;
        end
        if valid && get(norm_checkbox, 'Value')
            updateDisplay();
        end
    end
    function aspectCallback(~, ~)
        new_aspect = str2double(get(aspect_edit, 'String'));
        if isscalar(new_aspect) && ~isnan(new_aspect) && new_aspect > 0
            setAspectRatio(new_aspect);
        else
            set(aspect_edit, 'String', sprintf('%.2f', aspect_ratio));
        end
    end
    function setAspectRatio(ratio)
        aspect_ratio = ratio;
        set(aspect_edit, 'String', sprintf('%.2f', aspect_ratio));
        daspect(ax, [1/aspect_ratio 1 1]);
    end
    function sliderCallback(~, ~)
        current_frame = round(get(slider, 'Value'));
        updateDisplay();
    end
    function wheelStepCallback(~, ~)
        new_step = round(str2double(get(wheel_edit, 'String')));
        if isscalar(new_step) && ~isnan(new_step) && new_step >= 1
            wheel_step = new_step;
        else
            set(wheel_edit, 'String', num2str(wheel_step));
        end
    end
    function arrowStepCallback(~, ~)
        new_step = round(str2double(get(arrow_edit, 'String')));
        if isscalar(new_step) && ~isnan(new_step) && new_step >= 1
            arrow_step = new_step;
        else
            set(arrow_edit, 'String', num2str(arrow_step));
        end
    end
    function skipCallback(~, ~)
        new_skip = round(str2double(get(skip_edit, 'String')));
        if isscalar(new_skip) && ~isnan(new_skip) && new_skip >= 1
            play_skip = new_skip;
        else
            set(skip_edit, 'String', num2str(play_skip));
        end
    end
    function scrollWheelCallback(~, event)
        if event.VerticalScrollCount > 0
            current_frame = min(current_frame + wheel_step, effective_frames);
        else
            current_frame = max(current_frame - wheel_step, 1);
        end
        set(slider, 'Value', current_frame);
        updateDisplay();
    end
    function keyPressCallback(~, event)
        switch event.Key
            case 'rightarrow'
                current_frame = min(current_frame + arrow_step, effective_frames);
                set(slider, 'Value', current_frame);
                updateDisplay();
            case 'leftarrow'
                current_frame = max(current_frame - arrow_step, 1);
                set(slider, 'Value', current_frame);
                updateDisplay();
            case 'space'
                set(play_button, 'Value', ~get(play_button, 'Value'));
                playCallback();
        end
    end
    function playCallback(~, ~)
        if get(play_button, 'Value')
            set(play_button, 'String', '■ Stop');
            startPlayback();
        else
            set(play_button, 'String', '▶ Play');
            stopPlayback();
        end
    end
    function startPlayback()
        is_playing = true;
        effective_fps = frame_rate / avg_window;
        play_timer = timer('ExecutionMode', 'fixedRate', ...
                           'Period', 1/effective_fps, ...
                           'TimerFcn', @playTimerCallback);
        start(play_timer);
    end
    function stopPlayback()
        is_playing = false;
        if ~isempty(play_timer) && isvalid(play_timer)
            stop(play_timer);
            delete(play_timer);
            play_timer = [];
        end
    end
    function playTimerCallback(~, ~)
        current_frame = current_frame + play_skip;
        if current_frame > effective_frames
            current_frame = 1; % Loop playback
        end
        set(slider, 'Value', current_frame);
        updateDisplay();
    end
    function processData()
        if avg_window > 1
            effective_frames = floor(num_frames / avg_window);
            temp_data = cumsum(single(im_data), 3);
            processed_data_temp = (temp_data(:,:,avg_window:avg_window:end) - ...
                                   [zeros(height,width,1,'single'), temp_data(:,:,avg_window:avg_window:end-avg_window)]) / avg_window;
            processed_data = cast(processed_data_temp, class(im_data));
        else
            processed_data = im_data;
            effective_frames = num_frames;
        end
        current_frame = min(max(1, current_frame), effective_frames);
        set(slider, 'Max', max(effective_frames, 1), 'Value', current_frame);
        updateSliderSteps();
        
        set(frame_info_text, 'String', sprintf('Orig. Frames: %d | Eff. Frames: %d', num_frames, effective_frames));
    end
    
    function updateSliderSteps()
        if effective_frames > 1
             set(slider, 'SliderStep', [1/(effective_frames-1), 10/(effective_frames-1)]);
        else
             set(slider, 'SliderStep', [0.1, 0.1]);
        end
    end
    function updateFrameTimeInfo()
        if effective_frames > 0
            effective_fps = frame_rate / avg_window;
            current_time = (current_frame - 1) / effective_fps;
            total_time = (effective_frames - 1) / effective_fps;
            
            info_str = sprintf('Frame: %d/%d | Time: %.2fs / %.2fs', ...
                               current_frame, effective_frames, current_time, total_time);
        else
            info_str = 'No data to display';
        end
        title(ax, info_str, 'FontSize', 10);
    end
    function updateDisplay()
        if isempty(processed_data) || effective_frames < 1
            set(img_handle, 'CData', zeros(height, width, class(im_data)));
            updateFrameTimeInfo();
            return;
        end
        
        current_frame = min(max(1, current_frame), effective_frames);
        frame = processed_data(:,:,current_frame);
        
        if get(norm_checkbox, 'Value')
            frame = rescale(frame);
            low_in = prctile(frame(:), low_percentile);
            high_in = prctile(frame(:), high_percentile);
            if high_in > low_in
                frame = imadjust(frame, [low_in; high_in], []);
            end
        else
            frame = imadjust(frame, [low_percentile/100; high_percentile/100], []);
        end
        
        set(img_handle, 'CData', frame);
        updateFrameTimeInfo();
        drawnow('limitrate');
    end
    function resizeCallback(~, ~)
        % Future implementation for responsive layout
    end
    function closeCallback(~, ~)
        stopPlayback();
        delete(fig);
    end
end